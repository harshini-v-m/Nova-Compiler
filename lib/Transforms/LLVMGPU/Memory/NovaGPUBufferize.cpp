#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
    // Hoist above any enclosing scf.for to avoid per-iteration reallocation.
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
// GPU copy function
// ---------------------------------------------------------------------------
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc,
                               Value from, Value to) {
  Operation *parent = builder.getInsertionBlock()->getParentOp();
  bool insideForall = false;
  while (parent) {
    if (isa<scf::ForallOp>(parent)) { insideForall = true; break; }
    parent = parent->getParentOp();
  }

  if (insideForall) {
    auto fromType = cast<MemRefType>(from.getType());
    if (fromType.getRank() == 0) {
      // Rank-0: load from source (hoisted outside forall if defined there).
      bool definedOutside = true;
      if (Operation *def = from.getDefiningOp())
        if (parent->isAncestor(def)) definedOutside = false;
      if (auto arg = dyn_cast<BlockArgument>(from))
        if (parent->isAncestor(arg.getOwner()->getParentOp()))
          definedOutside = false;
      // Also check that `to` is not defined inside the forall — if it is,
      // placing the store after the forall would violate dominance.
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

    // Fix shape mismatch: static alloca source vs dynamic subview dest.
    auto toType = cast<MemRefType>(to.getType());
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
    registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                    gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, nova::NovaDialect, scf::SCFDialect>();
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

    // Convert linalg.copy {nova.promote_to_workgroup} on memrefs → memref.copy
    // so that ConvertMemRefToGpu can match the global→workgroup pattern and emit
    // nvgpu.device_async_copy. ConvertLinalgToLoops (which runs later) would
    // otherwise dissolve these into load/store loops that ConvertMemRefToGpu
    // cannot recognize.
    SmallVector<linalg::CopyOp> promoteCopies;
    moduleOp.walk([&](linalg::CopyOp copyOp) {
      if (copyOp->hasAttr("nova.promote_to_workgroup"))
        promoteCopies.push_back(copyOp);
    });
    IRRewriter rewriter2(moduleOp.getContext());
    for (linalg::CopyOp copyOp : promoteCopies) {
      rewriter2.setInsertionPoint(copyOp);
      rewriter2.replaceOpWithNewOp<memref::CopyOp>(
          copyOp, copyOp.getInputs()[0], copyOp.getOutputs()[0]);
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
  bufferization::BufferDeallocationPipelineOptions deallocOpts;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);
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
      if (isWorkgroupValue(w.getBase()) || isKernelLocalStaging(w.getBase()))
        { found = true; return WalkResult::interrupt(); }
    if (auto w = dyn_cast<vector::StoreOp>(inner))
      if (isWorkgroupValue(w.getBase()) || isKernelLocalStaging(w.getBase()))
        { found = true; return WalkResult::interrupt(); }
    if (auto w = dyn_cast<memref::StoreOp>(inner))
      if (isWorkgroupValue(w.getMemref()) || isKernelLocalStaging(w.getMemref()))
        { found = true; return WalkResult::interrupt(); }
    if (auto w = dyn_cast<affine::AffineStoreOp>(inner))
      if (isWorkgroupValue(w.getMemref()) || isKernelLocalStaging(w.getMemref()))
        { found = true; return WalkResult::interrupt(); }
    if (auto c = dyn_cast<linalg::CopyOp>(inner))
      for (Value out : c.getDpsInits())
        if (isWorkgroupValue(out) || isKernelLocalStaging(out))
          { found = true; return WalkResult::interrupt(); }
    if (auto c = dyn_cast<memref::CopyOp>(inner))
      if (isWorkgroupValue(c.getTarget()) || isKernelLocalStaging(c.getTarget()))
        { found = true; return WalkResult::interrupt(); }
    return WalkResult::advance();
  });
  return found;
}

static bool hasWorkgroupLoads(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) -> WalkResult {
    if (auto r = dyn_cast<vector::TransferReadOp>(inner))
      if (isWorkgroupValue(r.getBase()) || isKernelLocalStaging(r.getBase()))
        { found = true; return WalkResult::interrupt(); }
    if (auto r = dyn_cast<vector::LoadOp>(inner))
      if (isWorkgroupValue(r.getBase()) || isKernelLocalStaging(r.getBase()))
        { found = true; return WalkResult::interrupt(); }
    if (auto r = dyn_cast<memref::LoadOp>(inner))
      if (isWorkgroupValue(r.getMemref()) || isKernelLocalStaging(r.getMemref()))
        { found = true; return WalkResult::interrupt(); }
    if (auto r = dyn_cast<affine::AffineLoadOp>(inner))
      if (isWorkgroupValue(r.getMemref()) || isKernelLocalStaging(r.getMemref()))
        { found = true; return WalkResult::interrupt(); }
    if (auto c = dyn_cast<linalg::CopyOp>(inner))
      for (Value in : c.getDpsInputs())
        if (isWorkgroupValue(in) || isKernelLocalStaging(in))
          { found = true; return WalkResult::interrupt(); }
    if (auto c = dyn_cast<memref::CopyOp>(inner))
      if (isWorkgroupValue(c.getSource()) || isKernelLocalStaging(c.getSource()))
        { found = true; return WalkResult::interrupt(); }
    return WalkResult::advance();
  });
  return found;
}

static bool isSimplePerThreadCopy(scf::ForOp forOp) {
  unsigned loads = 0, stores = 0, other = 0;
  for (Operation &op : *forOp.getBody()) {
    if (isa<scf::YieldOp>(op)) continue;
    if (auto r = dyn_cast<memref::LoadOp>(op))
      { (isWorkgroupValue(r.getMemref()) ? loads : other)++; continue; }
    if (auto w = dyn_cast<memref::StoreOp>(op))
      { (isWorkgroupValue(w.getMemref()) ? stores : other)++; continue; }
    other++;
  }
  return loads == 1 && stores == 1 && other == 0;
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
      if (isa<scf::YieldOp, gpu::TerminatorOp, NVVM::Barrier0Op, gpu::BarrierOp>(op))
        break;
      if (hasWorkgroupLoads(&op)) break;
      if (hasWorkgroupStores(&op)) { barrierPoints.push_back(&block->front()); break; }
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
                    NVVM::NVVMDialect, vector::VectorDialect,
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
