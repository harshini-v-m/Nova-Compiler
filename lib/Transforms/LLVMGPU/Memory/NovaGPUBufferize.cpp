#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::nova {

// ---------------------------------------------------------------------------
// Helpers shared by allocation fn and barrier pass
// ---------------------------------------------------------------------------

static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

static bool isBlockLevelForall(scf::ForallOp op) {
  auto mapping = op.getMapping();
  if (!mapping) return false;
  for (Attribute attr : mapping->getValue())
    if (isa<gpu::GPUBlockMappingAttr>(attr)) return true;
  return false;
}

// ---------------------------------------------------------------------------
// GPU allocation function
// ---------------------------------------------------------------------------
static FailureOr<Value> gpuRequireMemSpaceAllocationFn(OpBuilder &builder,
                                                       Location loc,
                                                       MemRefType memRefType,
                                                       ValueRange dynamicSizes,
                                                       unsigned alignment) {
  Attribute memSpace = memRefType.getMemorySpace();

  // NovaVectorizeVectorExtOpsPass emits alloc_tensor with memory_space as
  // IntegerAttr (e.g. 1 : i64) rather than gpu::AddressSpaceAttr. These are
  // thread-local temporaries — treat as untagged → private alloca.
  if (memSpace && isa<IntegerAttr>(memSpace))
    memSpace = nullptr;

  auto privateSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  auto wkgpSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  if (memSpace && cast<gpu::AddressSpaceAttr>(memSpace).getValue() ==
                      gpu::GPUDialect::getWorkgroupAddressSpace()) {
    // Workgroup (shared) memory must be a module-level global in NVPTX —
    // use AllocOp (not AllocaOp) so ConvertSharedMemAllocs can hoist it to
    // a global symbol later. Hoist above any enclosing scf.for to avoid
    // per-iteration re-allocation.
    auto allocType = MemRefType::get(memRefType.getShape(),
                                     memRefType.getElementType(),
                        AffineMap(), wkgpSpace);
    OpBuilder::InsertionGuard guard(builder);
    Operation *hoistTarget = nullptr;
    Operation *cur = builder.getInsertionBlock()->getParentOp();
    while (cur) {
      if (isa<gpu::LaunchOp>(cur)) break;
      if (auto fa = dyn_cast<scf::ForallOp>(cur))
        if (isBlockLevelForall(fa)) break;
      if (isa<scf::ForOp>(cur)) hoistTarget = cur;
      cur = cur->getParentOp();
    }
    if (hoistTarget)
      builder.setInsertionPoint(hoistTarget);
    return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // Private or untagged: only use alloca (private address space) when we are
  // actually inside a GPU kernel (scf.forall). At function scope these are
  // inter-kernel staging buffers — emit plain heap alloc with no address space
  // so they become CPU-visible managed allocations, not host stack frames.
  bool insideKernel = false;
  for (Operation *p = builder.getInsertionBlock()->getParentOp(); p;
       p = p->getParentOp()) {
    if (isa<scf::ForallOp>(p)) { insideKernel = true; break; }
  }

  if (insideKernel) {
    auto allocType = MemRefType::get(memRefType.getShape(),
                                     memRefType.getElementType(),
                        AffineMap(), privateSpace);
    return memref::AllocaOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // Function scope → plain heap alloc, no address space tag.
  return memref::AllocOp::create(builder, loc, memRefType, dynamicSizes)
      .getResult();
}

// ---------------------------------------------------------------------------
// GPU copy function helpers
// ---------------------------------------------------------------------------

// Peel SubViewOp/CastOp chains to find the root alloc or block argument.
static Value getRootBuffer(Value v) {
  while (true) {
    if (auto sv = v.getDefiningOp<memref::SubViewOp>()) { v = sv.getSource(); continue; }
    if (auto cv = v.getDefiningOp<memref::CastOp>())    { v = cv.getSource(); continue; }
    break;
  }
  return v;
}

// Returns true when `from` and `to` refer to the same memory location.
// Handles: identical SSA values, scf.for iter_args, and subviews of the same
// root buffer at identical offsets (both constant and dynamic SSA values).
static bool isSelfCopy(Value from, Value to) {
  if (from == to) return true;

  // An scf.for iter_arg (block arg) and the corresponding loop result both
  // alias the same initial buffer — peel both back to the init value.
  auto peelForAlias = [](Value v) -> Value {
    // Block argument of scf.for body → initial value
    if (auto bbArg = dyn_cast<BlockArgument>(v))
      if (auto forOp = dyn_cast<scf::ForOp>(bbArg.getOwner()->getParentOp()))
        return forOp.getInitArgs()[bbArg.getArgNumber() -
                                   forOp.getNumInductionVars()];
    // scf.for result → initial value of the corresponding iter_arg
    if (auto forResult = dyn_cast<OpResult>(v))
      if (auto forOp = dyn_cast<scf::ForOp>(forResult.getOwner()))
        return forOp.getInitArgs()[forResult.getResultNumber()];
    return v;
  };
  from = peelForAlias(from);
  to   = peelForAlias(to);
  if (from == to) return true;

  if (getRootBuffer(from) != getRootBuffer(to)) return false;

  // Same root — offsets must also match element-by-element.
  auto getOffsets = [](Value v) -> SmallVector<OpFoldResult> {
    if (auto sv = v.getDefiningOp<memref::SubViewOp>())
      return SmallVector<OpFoldResult>(sv.getMixedOffsets());
    return {};
  };
  SmallVector<OpFoldResult> offsFrom = getOffsets(from);
  SmallVector<OpFoldResult> offsTo   = getOffsets(to);
  if (offsFrom.empty() && offsTo.empty()) return true;
  if (offsFrom.size() != offsTo.size()) return false;
  for (auto [a, b] : llvm::zip(offsFrom, offsTo)) {
    auto ca = getConstantIntValue(a), cb = getConstantIntValue(b);
    if (ca && cb) { if (*ca != *cb) return false; continue; }
    auto va = dyn_cast<Value>(a), vb = dyn_cast<Value>(b);
    if (!va || !vb || va != vb) return false;
  }
  return true;
}

// Emit nvgpu.device_async_copy for a global→workgroup tile copy.
//
// Each cp.async instruction copies one contiguous row (innerVec elements).
// All row copies are issued first (fully unrolled — no scf.for), then a single
// commit.group + wait.group 0 completes the entire tile.  This matches the
// hardware model: cp.async is non-blocking; the commit/wait is the barrier.
//
// The subview passed in is already the per-thread tile (e.g. 8×4 or 4×4), so
// the outer dimensions are statically known small integers that we unroll.
static void emitAsyncTileCopy(OpBuilder &builder, Location loc,
                              Value srcBuf, Value dstBuf) {
  MLIRContext *ctx = builder.getContext();
  auto dstType = cast<MemRefType>(dstBuf.getType());
  ArrayRef<int64_t> shape = dstType.getShape();
  int64_t rank = shape.size();

  int64_t innerVec  = rank > 0 ? shape[rank - 1] : 1;
  int64_t elemBytes = dstType.getElementTypeBitWidth() / 8;
  bool useBypassL1  = (innerVec * elemBytes >= 16);
  auto srcType = cast<MemRefType>(srcBuf.getType());

  // Compute the total number of rows = product of all dims except the last.
  int64_t numRows = 1;
  for (int64_t d = 0; d < rank - 1; ++d)
    numRows *= shape[d];

  // Emit one cp.async per row, all back-to-back with no loop.
  // Row index is a static constant — fully unrolled.
  for (int64_t row = 0; row < numRows; ++row) {
    // Build multi-dim index for this row: decompose flat row index into
    // per-dimension indices for all dims except the last.
    SmallVector<Value> dstIdx, srcIdx;
    int64_t rem = row;
    for (int64_t d = rank - 2; d >= 0; --d) {
      int64_t dimSize = shape[d];
      int64_t idx = rem % dimSize;
      rem /= dimSize;
      dstIdx.insert(dstIdx.begin(),
                    builder.create<arith::ConstantIndexOp>(loc, idx));
    }
    // column offset is always 0 — cp.async reads innerVec contiguous elements
    Value c0col = builder.create<arith::ConstantIndexOp>(loc, 0);
    dstIdx.push_back(c0col);
    srcIdx = dstIdx;
    // Pad src indices to its rank in case source has higher rank.
    while ((int64_t)srcIdx.size() < srcType.getRank())
      srcIdx.insert(srcIdx.begin(),
                    builder.create<arith::ConstantIndexOp>(loc, 0));

    builder.create<nvgpu::DeviceAsyncCopyOp>(
        loc,
        nvgpu::DeviceAsyncTokenType::get(ctx),
        dstBuf, dstIdx,
        srcBuf, srcIdx,
        builder.getIndexAttr(innerVec),
        /*srcElements=*/Value{},
        /*bypassL1=*/useBypassL1 ? builder.getUnitAttr() : UnitAttr());
  }

  // One commit + wait after all row copies — the whole tile lands atomically.
  builder.create<NVVM::CpAsyncCommitGroupOp>(loc);
  builder.create<NVVM::CpAsyncWaitGroupOp>(loc, builder.getI32IntegerAttr(0));
}

// ---------------------------------------------------------------------------
// GPU copy function
// ---------------------------------------------------------------------------
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc,
                               Value from, Value to) {
  // Elide copies where source and destination are the same buffer location.
  // This eliminates:
  //   Bug 2 — warp forall linalg.copy after vector.transfer_write
  //   Bug 3 — outer block-level copies produced by materialize_in_destination
  if (isSelfCopy(from, to))
    return success();

  auto fromType = cast<MemRefType>(from.getType());
  auto toType   = cast<MemRefType>(to.getType());

  // Global → workgroup: emit cp.async directly instead of going through
  // memref.copy → ConvertMemRefToGpu. This is Bug 1 / Fix A.
  // "Global" here means either: untagged, gpu::AddressSpace::Global, or the
  // integer address space 1 that InferMemorySpacePass uses for device tensors.
  auto isGlobalMemref = [](MemRefType t) -> bool {
    Attribute space = t.getMemorySpace();
    if (!space) return true;
    if (auto intSpace = dyn_cast<IntegerAttr>(space))
      return intSpace.getInt() == 1;
    if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(space))
      return gpuSpace.getValue() == gpu::AddressSpace::Global;
    return false;
  };
  bool toWorkgroup = isWorkgroupMemref(toType);
  if (isGlobalMemref(fromType) && toWorkgroup && fromType.hasStaticShape()) {
    emitAsyncTileCopy(builder, loc, from, to);
    return success();
  }

  Operation *parent = builder.getInsertionBlock()->getParentOp();
  bool insideForall = false;
  while (parent) {
    if (isa<scf::ForallOp>(parent)) { insideForall = true; break; }
    parent = parent->getParentOp();
  }

  if (insideForall) {
    if (fromType.getRank() == 0) {
      bool definedOutside = true;
      if (Operation *def = from.getDefiningOp())
        if (parent->isAncestor(def)) definedOutside = false;
      if (auto arg = dyn_cast<BlockArgument>(from))
        if (parent->isAncestor(arg.getOwner()->getParentOp()))
          definedOutside = false;
      bool toDefinedOutside = true;
      if (Operation *toDef = to.getDefiningOp())
        if (parent->isAncestor(toDef)) toDefinedOutside = false;
      if (auto toArg = dyn_cast<BlockArgument>(to))
        if (parent->isAncestor(toArg.getOwner()->getParentOp()))
          toDefinedOutside = false;

      if (definedOutside && toDefinedOutside) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(parent);
        Value scalar = builder.create<memref::LoadOp>(loc, from);
        builder.setInsertionPointAfter(parent);
        builder.create<memref::StoreOp>(loc, scalar, to);
        return success();
      }
    }

    // Fix shape mismatch: static source vs dynamic subview dest.
    if (fromType.getRank() == toType.getRank() && fromType.getRank() > 0) {
      int rank = fromType.getRank();
      bool needsSubview = false;
      SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes;
      for (int i = 0; i < rank; ++i) {
        if (!fromType.isDynamicDim(i) && toType.isDynamicDim(i)) {
          sizes.push_back(builder.create<memref::DimOp>(loc, to, i).getResult());
          needsSubview = true;
        } else if (!fromType.isDynamicDim(i)) {
          sizes.push_back(builder.getIndexAttr(fromType.getDimSize(i)));
        } else {
          sizes.push_back(builder.create<memref::DimOp>(loc, from, i).getResult());
        }
      }
      if (needsSubview)
        from = builder.create<memref::SubViewOp>(loc, from, offsets, sizes, strides);
    }

    linalg::CopyOp::create(builder, loc, from, to);
  } else {
    memref::CopyOp::create(builder, loc, from, to);
  }
  return success();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------
struct NovaGPUComprehensiveBufferizePass
    : public PassWrapper<NovaGPUComprehensiveBufferizePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUComprehensiveBufferizePass)

  NovaGPUComprehensiveBufferizePass() = default;
  NovaGPUComprehensiveBufferizePass(const NovaGPUComprehensiveBufferizePass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect,
                    gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, nova::NovaDialect,
                    nvgpu::NVGPUDialect, NVVM::NVVMDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Erase nova.fusion_barrier — tiling identity with no bufferized form.
    IRRewriter rewriter(moduleOp.getContext());
    SmallVector<FusionBarrierOp> fusionBarriers;
    moduleOp.walk([&](FusionBarrierOp b) { fusionBarriers.push_back(b); });
    for (FusionBarrierOp b : fusionBarriers)
      rewriter.replaceOp(b, b.getSource());

    // nova.value_barrier bufferizes to gpu::BarrierOp via
    // ValueBarrierOpBufferizationInterface (NovaGPUBufferizationInterfaces.cpp).

    bufferization::OneShotBufferizationOptions opts;
    opts.allocationFn = gpuRequireMemSpaceAllocationFn;
    opts.memCpyFn = gpuCopyFn;
    opts.bufferizeFunctionBoundaries = true;
    opts.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    opts.checkParallelRegions = false;
    opts.allowReturnAllocsFromLoops = true;
    opts.allowUnknownOps = true;

    bufferization::BufferizationState bufState;
    if (failed(bufferization::runOneShotBufferize(moduleOp, opts, bufState))) {
      moduleOp.emitOpError("GPU-aware bufferization failed");
      return signalPassFailure();
    }

    // Post-bufferize fixups on the memref-level IR.
    IRRewriter rw(moduleOp.getContext());

    // Fix A: linalg.copy {nova.promote_to_workgroup} global→workgroup
    //        → nvgpu.device_async_copy + commit + wait
    // Fix B/C: any linalg.copy or memref.copy where src aliases dst → erase
    {
      SmallVector<linalg::CopyOp>  linalgCopies;
      SmallVector<memref::CopyOp>  memrefSelfCopies;
      moduleOp.walk([&](linalg::CopyOp cp) { linalgCopies.push_back(cp); });
      moduleOp.walk([&](memref::CopyOp  cp) { memrefSelfCopies.push_back(cp); });

      for (linalg::CopyOp cp : linalgCopies) {
        Value src = cp.getInputs()[0];
        Value dst = cp.getOutputs()[0];
        auto srcType = cast<MemRefType>(src.getType());
        auto dstType = cast<MemRefType>(dst.getType());

        // Self-copy (Bug 1b, Bug 2): same buffer, same offsets → erase.
        if (isSelfCopy(src, dst)) { cp.erase(); continue; }

        // Fix A: global→workgroup with promote_to_workgroup attr → async copy.
        auto isGlobalSpace = [](MemRefType t) {
          Attribute s = t.getMemorySpace();
          if (!s) return true;
          if (auto ia = dyn_cast<IntegerAttr>(s)) return ia.getInt() == 1;
          if (auto ga = dyn_cast<gpu::AddressSpaceAttr>(s))
            return false; // workgroup/private are never global
          return false;
        };
        if (cp->hasAttr("nova.promote_to_workgroup") &&
            isGlobalSpace(srcType) && isWorkgroupMemref(dstType) &&
            srcType.hasStaticShape()) {
          rw.setInsertionPoint(cp);
          emitAsyncTileCopy(rw, cp.getLoc(), src, dst);
          cp.erase();
          continue;
        }
      }

      // Fix C: memref.copy %x, %x → erase (Bug 3, module-level self-copies).
      for (memref::CopyOp cp : memrefSelfCopies)
        if (isSelfCopy(cp.getSource(), cp.getTarget()))
          cp.erase();
    }

  }

  StringRef getArgument() const override {
    return "nova-gpu-comprehensive-bufferize";
  }
  StringRef getDescription() const override {
    return "Erases nova.fusion_barrier then runs OneShotBufferize with "
           "GPU-aware allocation and copy functions.";
  }
};

std::unique_ptr<Pass> createNovaGPUComprehensiveBufferizePass() {
  return std::make_unique<NovaGPUComprehensiveBufferizePass>();
}
void registerNovaGPUComprehensiveBufferizePass() {
  PassRegistration<NovaGPUComprehensiveBufferizePass>();
}

// ---------------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------------
void addNovaPostBufferizationPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(memref::createResolveShapedTypeResultDimsPass());
  // Fold subview chains first so that identical subviews become the same SSA
  // value. dropEquivalentBufferResults can only eliminate a copy when src and
  // dst are provably the same Value — running CSE before it ensures subviews
  // computed with the same base+offsets collapse to one SSA value first.
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCSEPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  // Skip OwnershipBasedBufferDeallocation: GPU kernels contain nvvm/nvgpu ops
  // that don't implement MemoryEffectOpInterface, causing the pass to fail.
  // Workgroup alloca and private alloca are stack-scoped (no dealloc needed);
  // function-scope heap allocs are managed by the JIT runtime externally.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void addNovaComprehensiveBufferizePasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createNovaEliminateEmptyTensorsPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addNestedPass<func::FuncOp>(createNovaGPUInferMemorySpacePass());
  pm.addPass(createNovaGPUComprehensiveBufferizePass());
  addNovaPostBufferizationPasses(pm);
}

//===----------------------------------------------------------------------===//
// Workgroup barrier insertion
//===----------------------------------------------------------------------===//

static bool isWorkgroupValue(Value v) {
  auto mt = dyn_cast<MemRefType>(v.getType());
  return mt && isWorkgroupMemref(mt);
}

static bool isWorkgroupOrGlobalValue(Value v) {
  auto mt = dyn_cast<MemRefType>(v.getType());
  if (!mt)
    return false;
  if (isWorkgroupMemref(mt))
    return true;
  // Also treat global memory (address space 1) as workgroup-like for barrier
  // insertion purposes if it's used for intra-kernel accumulation (e.g. matmul
  // output initialization followed by loads).
  auto space = dyn_cast_or_null<IntegerAttr>(mt.getMemorySpace());
  return space && space.getInt() == 1;
}

static bool isKernelLocalStaging(Value v) {
  auto mt = dyn_cast<MemRefType>(v.getType());
  if (!mt || mt.getMemorySpace()) return false;
  Value base = v;
  while (auto sv = base.getDefiningOp<memref::SubViewOp>())
    base = sv.getSource();
  if (!base.getDefiningOp<memref::AllocOp>()) return false;
  for (Operation *p = base.getDefiningOp()->getParentOp(); p;
       p = p->getParentOp()) {
    if (isa<gpu::LaunchOp>(p)) return true;
    if (isa<func::FuncOp>(p)) return false;
  }
  return false;
}

static bool hasWorkgroupStores(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) -> WalkResult {
    if (auto w = dyn_cast<vector::TransferWriteOp>(inner))
      if (isWorkgroupOrGlobalValue(w.getBase()) ||
          isKernelLocalStaging(w.getBase())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto w = dyn_cast<vector::StoreOp>(inner))
      if (isWorkgroupOrGlobalValue(w.getBase()) ||
          isKernelLocalStaging(w.getBase())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto w = dyn_cast<memref::StoreOp>(inner))
      if (isWorkgroupOrGlobalValue(w.getMemref()) ||
          isKernelLocalStaging(w.getMemref())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto w = dyn_cast<affine::AffineStoreOp>(inner))
      if (isWorkgroupOrGlobalValue(w.getMemref()) ||
          isKernelLocalStaging(w.getMemref())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto c = dyn_cast<linalg::CopyOp>(inner))
      for (Value out : c.getDpsInits())
        if (isWorkgroupOrGlobalValue(out) || isKernelLocalStaging(out)) {
          found = true;
          return WalkResult::interrupt();
        }
    if (auto c = dyn_cast<memref::CopyOp>(inner))
      if (isWorkgroupOrGlobalValue(c.getTarget()) ||
          isKernelLocalStaging(c.getTarget())) {
        found = true;
        return WalkResult::interrupt();
      }
    // nvgpu.device_async_copy writes to workgroup (shared) memory.
    if (auto acp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(inner))
      if (isWorkgroupOrGlobalValue(acp.getDst()) ||
          isKernelLocalStaging(acp.getDst())) {
        found = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return found;
}

static bool hasWorkgroupLoads(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) -> WalkResult {
    if (auto r = dyn_cast<vector::TransferReadOp>(inner))
      if (isWorkgroupOrGlobalValue(r.getBase()) ||
          isKernelLocalStaging(r.getBase())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto r = dyn_cast<vector::LoadOp>(inner))
      if (isWorkgroupOrGlobalValue(r.getBase()) ||
          isKernelLocalStaging(r.getBase())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto r = dyn_cast<memref::LoadOp>(inner))
      if (isWorkgroupOrGlobalValue(r.getMemref()) ||
          isKernelLocalStaging(r.getMemref())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto r = dyn_cast<affine::AffineLoadOp>(inner))
      if (isWorkgroupOrGlobalValue(r.getMemref()) ||
          isKernelLocalStaging(r.getMemref())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto c = dyn_cast<linalg::CopyOp>(inner))
      for (Value in : c.getDpsInputs())
        if (isWorkgroupOrGlobalValue(in) || isKernelLocalStaging(in)) {
          found = true;
          return WalkResult::interrupt();
        }
    if (auto c = dyn_cast<memref::CopyOp>(inner))
      if (isWorkgroupOrGlobalValue(c.getSource()) ||
          isKernelLocalStaging(c.getSource())) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto ldm = dyn_cast<nvgpu::LdMatrixOp>(inner))
      if (isWorkgroupOrGlobalValue(ldm.getSrcMemref()) ||
          isKernelLocalStaging(ldm.getSrcMemref())) {
        found = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return found;
}

// True when `forOp`'s body is one of the "trivially per-thread, no cross-warp
// data dependence" copy patterns:
//
//   (A) 1 workgroup load + 1 workgroup store + 0 other ops
//       (smem→smem swap path)
//
//   (B) 1 non-workgroup (gmem) load + 1 workgroup store + 0 other ops
//       (gmem→smem prologue copy emitted by promotion when cp.async hasn't
//       lowered the linalg.copy yet — each thread writes its own SMEM slot
//       and reads its own gmem location, so no cross-thread aliasing inside
//       the loop. Barriers inside this loop are zero-utility; the producer-
//       consumer fence happens at the K-loop boundary, not per element.)
//
//   (C) Body is a single nested `scf.for` (with literal index bounds) whose
//       body matches (A) or (B) recursively, plus 0 other ops.
//       This catches the unrolled prologue shape that promotion+linalg-to-
//       loops emits for matmul A-tile loads:
//
//           scf.for outer = 0 to 4 step 1 {
//             scf.for inner = 0 to 8 step 1 {
//               %v = memref.load gmem[outer, inner]
//               memref.store %v, smem[inner, outer]
//             }
//           }
//
//       Without classifying the outer as a simple copy too, the trailing-
//       store-before-yield logic in insertBarriersInBlock inserts a barrier
//       at the outer loop's yield (4× per K iteration), which Nsight reports
//       as a ~3× spike in Stall Barrier.
//
// Both inner and outer per-thread copies are race-free across warps: each
// thread reads/writes its own gmem/smem locations. The producer→consumer
// fence is enforced at the K-loop boundary by the existing top-of-iter
// barrier and by the post-wait_group barrier added in
// NovaGPUPipelining::mergeAsyncCommitsInKLoop.
static bool isSimplePerThreadCopy(scf::ForOp forOp) {
  unsigned wgLoads = 0, nonWgLoads = 0;
  unsigned wgStores = 0, nonWgStores = 0;
  unsigned other = 0;
  scf::ForOp nestedFor;
  for (Operation &op : *forOp.getBody()) {
    if (isa<scf::YieldOp>(op)) continue;
    if (auto r = dyn_cast<memref::LoadOp>(op)) {
      (isWorkgroupValue(r.getMemref()) ? wgLoads : nonWgLoads)++;
      continue;
    }
    if (auto w = dyn_cast<memref::StoreOp>(op)) {
      (isWorkgroupValue(w.getMemref()) ? wgStores : nonWgStores)++;
      continue;
    }
    if (auto inner = dyn_cast<scf::ForOp>(op)) {
      if (nestedFor) {
        // More than one nested loop — not a simple copy.
        return false;
      }
      nestedFor = inner;
      continue;
    }
    other++;
  }
  if (other != 0)
    return false;
  // Pattern (C): single nested simple-copy loop, no scalar ops at this level.
  if (nestedFor) {
    if (wgLoads != 0 || nonWgLoads != 0 || wgStores != 0 || nonWgStores != 0)
      return false;
    return isSimplePerThreadCopy(nestedFor);
  }
  // Pattern (A): smem load + smem store
  if (wgLoads == 1 && wgStores == 1 && nonWgLoads == 0 && nonWgStores == 0)
    return true;
  // Pattern (B): gmem load + smem store (per-thread prologue)
  if (nonWgLoads == 1 && wgStores == 1 && wgLoads == 0 && nonWgStores == 0)
    return true;
  return false;
}

static bool isUnconditionalForBody(Block *block) {
  auto parentFor = dyn_cast_or_null<scf::ForOp>(block->getParentOp());
  if (!parentFor) return false;
  for (Operation *cur = parentFor->getParentOp(); cur; cur = cur->getParentOp()) {
    if (isa<scf::IfOp>(cur)) return false;
    if (isa<gpu::LaunchOp, func::FuncOp>(cur)) break;
  }
  return true;
}

static void insertBarriersInBlock(OpBuilder &builder, Block *block) {
  auto isConstIdx = [](Value v) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>())
      return isa<IntegerAttr>(cst.getValue());
    return v.getDefiningOp<arith::ConstantIndexOp>() != nullptr;
  };

  SmallVector<Operation *> barrierPoints;

  // If this for-body opens with stores before any loads, barrier at top.
  if (isUnconditionalForBody(block)) {
    for (Operation &op : *block) {
      if (isa<scf::YieldOp, gpu::TerminatorOp, NVVM::Barrier0Op,
              gpu::BarrierOp>(op))
        break;
      if (hasWorkgroupLoads(&op))
        break;
      if (hasWorkgroupStores(&op)) {
        barrierPoints.push_back(&block->front());
        break;
      }
    }
  }

  bool seenStore = false, seenNonAtomicStore = false;
  for (Operation &op : *block) {
    if (isa<scf::YieldOp, gpu::TerminatorOp>(op)) continue;
    if (isa<NVVM::Barrier0Op, gpu::BarrierOp>(op)) {
      seenStore = seenNonAtomicStore = false;
      continue;
    }
    if (seenStore && hasWorkgroupLoads(&op)) {
      // Hoist barrier past any enclosing scf.if.
      Operation *target = &op;
      for (Operation *cur = op.getParentOp(); cur; cur = cur->getParentOp()) {
        if (isa<gpu::LaunchOp, func::FuncOp>(cur)) break;
        if (isa<scf::IfOp>(cur)) { target = cur; break; }
      }
      if (barrierPoints.empty() || barrierPoints.back() != target)
        barrierPoints.push_back(target);
      seenStore = seenNonAtomicStore = false;
    }
    if (hasWorkgroupStores(&op))
      seenStore = seenNonAtomicStore = true;

    // Recurse into uniform for/forall.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      bool uniform = isConstIdx(forOp.getLowerBound()) &&
                     isConstIdx(forOp.getUpperBound()) &&
                     isConstIdx(forOp.getStep());
      if (uniform && !isSimplePerThreadCopy(forOp))
        insertBarriersInBlock(builder, forOp.getBody());
    }
    if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
      bool uniform = true;
      for (Value ub : forallOp.getUpperBound(builder))
        if (!isConstIdx(ub)) { uniform = false; break; }
      if (uniform)
        for (Block &inner : forallOp.getRegion())
          insertBarriersInBlock(builder, &inner);
    }
  }

  // Trailing stores in a uniform for-loop need a barrier before yield.
  if (seenNonAtomicStore && isa<scf::YieldOp>(block->getTerminator())) {
    if (auto parentFor = dyn_cast_or_null<scf::ForOp>(block->getParentOp())) {
      if (!isSimplePerThreadCopy(parentFor)) {
        bool uniform = isConstIdx(parentFor.getLowerBound()) &&
                       isConstIdx(parentFor.getUpperBound()) &&
                       isConstIdx(parentFor.getStep());
        bool insideIf = false;
        for (Operation *cur = parentFor->getParentOp(); cur;
             cur = cur->getParentOp()) {
          if (isa<scf::IfOp>(cur)) { insideIf = true; break; }
          if (isa<gpu::LaunchOp, func::FuncOp>(cur)) break;
        }
        bool alreadyHasTopBarrier =
            !barrierPoints.empty() && barrierPoints.front() == &block->front();
        if (uniform && !insideIf && !alreadyHasTopBarrier)
          barrierPoints.push_back(block->getTerminator());
      }
    }
  }

  for (Operation *pt : llvm::reverse(barrierPoints)) {
    builder.setInsertionPoint(pt);
    builder.create<NVVM::Barrier0Op>(pt->getLoc());
  }
}

struct NovaGPUInsertWorkgroupBarriersPass
    : public PassWrapper<NovaGPUInsertWorkgroupBarriersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUInsertWorkgroupBarriersPass)

  NovaGPUInsertWorkgroupBarriersPass() = default;
  NovaGPUInsertWorkgroupBarriersPass(const NovaGPUInsertWorkgroupBarriersPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect,
                NVVM::NVVMDialect, nvgpu::NVGPUDialect, vector::VectorDialect,
                linalg::LinalgDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    funcOp.walk([&](gpu::LaunchOp launchOp) {
      for (Block &block : launchOp.getBody())
        insertBarriersInBlock(builder, &block);
    });
  }

  StringRef getArgument() const override {
    return "nova-gpu-insert-workgroup-barriers";
  }
  StringRef getDescription() const override {
    return "Inserts nvvm.barrier0 at workgroup memory write→read transitions "
           "inside gpu.launch bodies.";
  }
};

std::unique_ptr<Pass> createNovaGPUInsertWorkgroupBarriersPass() {
  return std::make_unique<NovaGPUInsertWorkgroupBarriersPass>();
}
void registerNovaGPUInsertWorkgroupBarriersPass() {
  PassRegistration<NovaGPUInsertWorkgroupBarriersPass>();
}

} // namespace mlir::nova