#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-bufferization"

using namespace mlir;

namespace mlir::nova {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

static bool isBlockLevelForall(scf::ForallOp forallOp) {
  auto mappingAttr = forallOp.getMapping();
  if (!mappingAttr)
    return false;
  for (Attribute attr : mappingAttr->getValue())
    if (auto gpuAttr = dyn_cast<gpu::GPUBlockMappingAttr>(attr))
      return true;
  return false;
}

// ---------------------------------------------------------------------------
// GPU allocation function (unchanged from original)
// ---------------------------------------------------------------------------
static FailureOr<Value> gpuRequireMemSpaceAllocationFn(OpBuilder &builder,
                                                        Location loc,
                                                        MemRefType memRefType,
                                                        ValueRange dynamicSizes,
                                                        unsigned alignment) {
  Attribute memSpace = memRefType.getMemorySpace();
  if (memSpace && !isa<gpu::AddressSpaceAttr>(memSpace))
    return failure();

  auto privateSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  auto wkgpSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  if (memSpace && cast<gpu::AddressSpaceAttr>(memSpace).getValue() ==
                      gpu::GPUDialect::getWorkgroupAddressSpace()) {
    bool insideBlockKernel = false;
    Operation *check = builder.getInsertionBlock()->getParentOp();
    while (check) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(check))
        if (isBlockLevelForall(forallOp)) { insideBlockKernel = true; break; }
      if (isa<gpu::LaunchOp>(check)) { insideBlockKernel = true; break; }
      check = check->getParentOp();
    }
    if (!insideBlockKernel) {
      auto globalType = MemRefType::get(memRefType.getShape(),
                                        memRefType.getElementType());
      return memref::AllocOp::create(builder, loc, globalType, dynamicSizes)
          .getResult();
    }
    auto allocType = MemRefType::get(memRefType.getShape(),
                                     memRefType.getElementType(),
                                     AffineMap(), wkgpSpace);
    OpBuilder::InsertionGuard guard(builder);
    Operation *hoistTarget = nullptr;
    Operation *cur = builder.getInsertionBlock()->getParentOp();
    while (cur) {
      if (isa<gpu::LaunchOp>(cur)) break;
      if (auto forallOp = dyn_cast<scf::ForallOp>(cur))
        if (isBlockLevelForall(forallOp)) break;
      if (isa<scf::ForOp>(cur)) hoistTarget = cur;
      cur = cur->getParentOp();
    }
    if (hoistTarget) builder.setInsertionPoint(hoistTarget);
    return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  if (memSpace) {
    bool insideKernel = false;
    Operation *insertionParent = builder.getInsertionBlock()->getParentOp();
    while (insertionParent) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(insertionParent))
        if (isBlockLevelForall(forallOp)) { insideKernel = true; break; }
      if (isa<gpu::LaunchOp>(insertionParent)) { insideKernel = true; break; }
      insertionParent = insertionParent->getParentOp();
    }
    if (!insideKernel) {
      auto globalType = MemRefType::get(memRefType.getShape(),
                                        memRefType.getElementType());
      return memref::AllocOp::create(builder, loc, globalType, dynamicSizes)
          .getResult();
    }
    auto allocType = MemRefType::get(memRefType.getShape(),
                                     memRefType.getElementType(),
                                     AffineMap(), privateSpace);
    if (!dynamicSizes.empty()) {
      SmallVector<int64_t> staticShape;
      for (int d = 0; d < memRefType.getRank(); ++d)
        staticShape.push_back(memRefType.isDynamicDim(d) ? 4
                                                         : memRefType.getDimSize(d));
      auto staticAllocType = MemRefType::get(staticShape,
                                             memRefType.getElementType(),
                                             AffineMap(), privateSpace);
      SmallVector<Value> emptyDynamicSizes;
      return memref::AllocaOp::create(builder, loc, staticAllocType,
                                      emptyDynamicSizes).getResult();
    }
    return memref::AllocaOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }
  return memref::AllocOp::create(builder, loc, memRefType, dynamicSizes)
      .getResult();
}

static Value maybeSubviewToMatchDest(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  auto fromType = cast<MemRefType>(from.getType());
  auto toType   = cast<MemRefType>(to.getType());
  if (fromType.getRank() != toType.getRank() || fromType.getRank() == 0)
    return from;
  int rank = fromType.getRank();
  bool needsSubview = false;
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int i = 0; i < rank; ++i) {
    if (!fromType.isDynamicDim(i) && toType.isDynamicDim(i)) {
      sizes.push_back(builder.getIndexAttr(fromType.getDimSize(i)));
      needsSubview = true;
    } else if (!fromType.isDynamicDim(i)) {
      sizes.push_back(builder.getIndexAttr(fromType.getDimSize(i)));
    } else {
      sizes.push_back(builder.create<memref::DimOp>(loc, from, i).getResult());
    }
  }
  if (needsSubview)
    return builder.create<memref::SubViewOp>(loc, from, offsets, sizes, strides);
  return from;
}

static bool isSameMemoryRegion(Value a, Value b) {
  if (a == b) return true;
  auto aSV = a.getDefiningOp<memref::SubViewOp>();
  auto bSV = b.getDefiningOp<memref::SubViewOp>();
  if (!aSV || !bSV) return false;
  if (!isSameMemoryRegion(aSV.getSource(), bSV.getSource())) return false;
  if (aSV.getStaticOffsets() != bSV.getStaticOffsets()) return false;
  if (aSV.getStaticSizes()   != bSV.getStaticSizes())   return false;
  if (aSV.getStaticStrides() != bSV.getStaticStrides()) return false;
  if (aSV.getOffsets() != bSV.getOffsets()) return false;
  if (aSV.getSizes()   != bSV.getSizes())   return false;
  if (aSV.getStrides() != bSV.getStrides()) return false;
  return true;
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  if (isSameMemoryRegion(from, to)) return success();
  auto fromType = cast<MemRefType>(from.getType());
  auto toType   = cast<MemRefType>(to.getType());
  bool fromIsGlobal    = !fromType.getMemorySpace();
  bool toIsWorkgroup   = isWorkgroupMemref(toType);
  bool fromIsWorkgroup = isWorkgroupMemref(fromType);
  Operation *parent = builder.getInsertionBlock()->getParentOp();
  bool insideForall = false;
  while (parent) {
    if (isa<scf::ForallOp>(parent)) { insideForall = true; break; }
    parent = parent->getParentOp();
  }
  if (fromIsGlobal && toIsWorkgroup) {
    from = maybeSubviewToMatchDest(builder, loc, from, to);
    memref::CopyOp::create(builder, loc, from, to);
    return success();
  }
  if (fromIsWorkgroup && toIsWorkgroup) {
    if (insideForall) {
      from = maybeSubviewToMatchDest(builder, loc, from, to);
      linalg::CopyOp::create(builder, loc, from, to);
    } else {
      memref::CopyOp::create(builder, loc, from, to);
    }
    return success();
  }
  if (insideForall) {
    if (fromType.getRank() == 0) {
      bool isDefinedOutside = true;
      if (Operation *defOp = from.getDefiningOp())
        if (parent->isAncestor(defOp)) isDefinedOutside = false;
      if (auto arg = dyn_cast<BlockArgument>(from))
        if (parent->isAncestor(arg.getOwner()->getParentOp()))
          isDefinedOutside = false;
      if (isDefinedOutside) {
        Value scalar = builder.create<memref::LoadOp>(loc, from);
        builder.create<memref::StoreOp>(loc, scalar, to);
        return success();
      }
    }
    from = maybeSubviewToMatchDest(builder, loc, from, to);
    linalg::CopyOp::create(builder, loc, from, to);
  } else {
    memref::CopyOp::create(builder, loc, from, to);
  }
  return success();
}

// ---------------------------------------------------------------------------
// Bufferize pass (unchanged)
// ---------------------------------------------------------------------------
struct NovaGPUComprehensiveBufferizePass
    : public PassWrapper<NovaGPUComprehensiveBufferizePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUComprehensiveBufferizePass)
  NovaGPUComprehensiveBufferizePass() = default;
  NovaGPUComprehensiveBufferizePass(
      const NovaGPUComprehensiveBufferizePass &) = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                    gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, nova::NovaDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(moduleOp.getContext());
    SmallVector<FusionBarrierOp> fusionBarriers;
    moduleOp.walk([&](FusionBarrierOp b) { fusionBarriers.push_back(b); });
    for (FusionBarrierOp b : fusionBarriers) rewriter.replaceOp(b, b.getSource());
    SmallVector<ValueBarrierOp> valueBarriers;
    moduleOp.walk([&](ValueBarrierOp b) { valueBarriers.push_back(b); });
    for (ValueBarrierOp b : valueBarriers) rewriter.replaceOp(b, b.getInputs());
    bufferization::OneShotBufferizationOptions opts;
    opts.allocationFn = gpuRequireMemSpaceAllocationFn;
    opts.memCpyFn     = gpuCopyFn;
    opts.bufferizeFunctionBoundaries = true;
    opts.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    opts.checkParallelRegions = false;
    bufferization::BufferizationState bufState;
    if (failed(bufferization::runOneShotBufferize(moduleOp, opts, bufState))) {
      moduleOp.emitOpError("GPU-aware bufferization failed");
      return signalPassFailure();
    }
    SmallVector<linalg::CopyOp> deadCopies, liveCopies;
    moduleOp.walk([&](linalg::CopyOp copyOp) {
      if (copyOp.getInputs().size() != 1 || copyOp.getOutputs().size() != 1)
        return;
      if (isSameMemoryRegion(copyOp.getInputs()[0], copyOp.getOutputs()[0]))
        deadCopies.push_back(copyOp);
      else
        liveCopies.push_back(copyOp);
    });
    for (linalg::CopyOp dead : deadCopies) dead.erase();
    for (linalg::CopyOp copyOp : liveCopies) {
      if (!copyOp->hasAttr("nova.promote_to_workgroup")) continue;
      OpBuilder b(copyOp);
      Value src = copyOp.getInputs()[0];
      Value dst = copyOp.getOutputs()[0];
      auto srcType = cast<MemRefType>(src.getType());
      auto dstType = cast<MemRefType>(dst.getType());
      if (srcType.getNumDynamicDims() > dstType.getNumDynamicDims())
        src = maybeSubviewToMatchDest(b, copyOp.getLoc(), src, dst);
      memref::CopyOp::create(b, copyOp.getLoc(), src, dst);
      copyOp.erase();
    }
  }
  StringRef getArgument() const override { return "nova-gpu-comprehensive-bufferize"; }
  StringRef getDescription() const override {
    return "GPU-aware OneShotBufferize with allocation routing and copy lowering.";
  }
};

std::unique_ptr<Pass> createNovaGPUComprehensiveBufferizePass() {
  return std::make_unique<NovaGPUComprehensiveBufferizePass>();
}
void registerNovaGPUComprehensiveBufferizePass() {
  PassRegistration<NovaGPUComprehensiveBufferizePass>();
}

//===----------------------------------------------------------------------===//
// §1  Memory-space helpers
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
  Operation *defOp = base.getDefiningOp();
  Operation *parent = defOp ? defOp->getParentOp() : nullptr;
  while (parent) {
    if (isa<gpu::LaunchOp>(parent)) return true;
    if (isa<func::FuncOp>(parent)) return false;
    parent = parent->getParentOp();
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

static bool hasNonAtomicWorkgroupStores(Operation *op) {
  return hasWorkgroupStores(op);
}

static bool isSimplePerThreadCopy(scf::ForOp forOp) {
  Block *body = forOp.getBody();
  unsigned loadCount = 0, storeCount = 0, otherCount = 0;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op)) continue;
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (isWorkgroupValue(load.getMemref())) ++loadCount; else ++otherCount;
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (isWorkgroupValue(store.getMemref())) ++storeCount; else ++otherCount;
      continue;
    }
    ++otherCount;
  }
  return loadCount == 1 && storeCount == 1 && otherCount == 0;
}

static Operation *getOutermostIfAncestor(Operation *op) {
  Operation *outermostIf = nullptr;
  Operation *cur = op->getParentOp();
  while (cur) {
    if (isa<gpu::LaunchOp, func::FuncOp>(cur)) break;
    if (isa<scf::IfOp>(cur)) outermostIf = cur;
    cur = cur->getParentOp();
  }
  return outermostIf;
}

static Operation *hoistPastIfOps(Operation *op) {
  if (Operation *outerIf = getOutermostIfAncestor(op))
    return outerIf;
  return op;
}

/// Returns true if the block is the body of a scf.for that is NOT inside
/// any scf.if (i.e. the loop itself is at unconditional scope).
static bool isUnconditionalForBody(Block *block) {
  auto parentFor = dyn_cast_or_null<scf::ForOp>(block->getParentOp());
  if (!parentFor) return false;
  // Check the for-loop is not inside any scf.if.
  Operation *ancestor = parentFor->getParentOp();
  while (ancestor) {
    if (isa<scf::IfOp>(ancestor)) return false;
    if (isa<gpu::LaunchOp, func::FuncOp>(ancestor)) break;
    ancestor = ancestor->getParentOp();
  }
  return true;
}


static void insertBarriersInBlock(OpBuilder &builder, Block *block) {
  SmallVector<Operation *> barrierPoints;


  if (isUnconditionalForBody(block)) {
    // Scan forward to find if the block has workgroup stores before any
    // workgroup loads (ignoring terminators and existing barriers).
    bool hasStoresBeforeLoads = false;
    for (Operation &op : block->getOperations()) {
      if (isa<scf::YieldOp, gpu::TerminatorOp>(op)) break;
      if (isa<NVVM::Barrier0Op, gpu::BarrierOp>(op)) break; // already fenced
      if (hasWorkgroupLoads(&op)) break;  // loads come first, no need for loop-top barrier
      if (hasWorkgroupStores(&op)) { hasStoresBeforeLoads = true; break; }
    }
    if (hasStoresBeforeLoads) {
      // Insert before the first non-terminator op in the block.
      Operation *firstOp = &block->front();

      barrierPoints.push_back(firstOp);
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] loop-top read→write barrier at: "
                 << firstOp->getName() << "\n");
    }
  }

  // ── Barrier #2: write→read transition analysis ───────────────────────────
  bool seenWorkgroupStore = false;
  bool seenNonAtomicStore = false;

  for (Operation &op : block->getOperations()) {
    if (isa<scf::YieldOp, gpu::TerminatorOp>(op)) continue;

    // An existing barrier resets the seen-store flag.
    if (isa<NVVM::Barrier0Op, gpu::BarrierOp>(op)) {
      seenWorkgroupStore = false;
      seenNonAtomicStore = false;
      continue;
    }

    bool hasLoads  = hasWorkgroupLoads(&op);
    bool hasStores = hasWorkgroupStores(&op);

    // Write→read transition: insert barrier before the reader.
    // Hoist past any enclosing scf.if so the barrier is at unconditional scope.
    if (seenWorkgroupStore && hasLoads) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] write→read transition before: "
                 << op.getName() << "\n");
      Operation *insertionTarget = hoistPastIfOps(&op);
      // Avoid duplicating a point already recorded (e.g. multiple transitions
      // that collapse to the same outer if-block after hoisting).
      if (barrierPoints.empty() || barrierPoints.back() != insertionTarget)
        barrierPoints.push_back(insertionTarget);
      seenWorkgroupStore = false;
      seenNonAtomicStore = false;
    }

    if (hasStores) {
      seenWorkgroupStore = true;
      if (hasNonAtomicWorkgroupStores(&op))
        seenNonAtomicStore = true;
    }

    // Recurse into scf.for bodies (uniform — all threads execute every iter).
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto isConstIdx = [](Value v) -> bool {
        if (auto cst = v.getDefiningOp<arith::ConstantOp>())
          return isa<IntegerAttr>(cst.getValue());
        return v.getDefiningOp<arith::ConstantIndexOp>() != nullptr;
      };
      bool uniformBounds = isConstIdx(forOp.getLowerBound()) &&
                           isConstIdx(forOp.getUpperBound()) &&
                           isConstIdx(forOp.getStep());
      if (uniformBounds && !isSimplePerThreadCopy(forOp))
        insertBarriersInBlock(builder, forOp.getBody());
    }

    // Recurse into scf.forall bodies (uniform — all threads execute).
    if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
      auto isConstIdx = [](Value v) -> bool {
        if (auto cst = v.getDefiningOp<arith::ConstantOp>())
          return isa<IntegerAttr>(cst.getValue());
        return v.getDefiningOp<arith::ConstantIndexOp>() != nullptr;
      };
      bool uniformBounds = true;
      for (Value ub : forallOp.getUpperBound(builder))
        if (!isConstIdx(ub)) { uniformBounds = false; break; }
      if (uniformBounds)
        for (Block &innerBlock : forallOp.getRegion())
          insertBarriersInBlock(builder, &innerBlock);
    }

  }

  if (seenNonAtomicStore && isa<scf::YieldOp>(block->getTerminator())) {
    bool safeToInsert = false;
    if (auto parentFor = dyn_cast_or_null<scf::ForOp>(block->getParentOp())) {
      if (!isSimplePerThreadCopy(parentFor)) {
        auto isConst = [](Value v) {
          return v.getDefiningOp<arith::ConstantIndexOp>() != nullptr ||
                 (v.getDefiningOp<arith::ConstantOp>() &&
                  isa<IntegerAttr>(v.getDefiningOp<arith::ConstantOp>()
                                   ->getAttrOfType<Attribute>("value")));
        };
        bool uniformBounds = isConst(parentFor.getLowerBound()) &&
                             isConst(parentFor.getUpperBound()) &&
                             isConst(parentFor.getStep());
        // Only safe if the for-loop is NOT inside a scf.if.
        bool insideIf = false;
        Operation *ancestor = parentFor->getParentOp();
        while (ancestor) {
          if (isa<scf::IfOp>(ancestor)) { insideIf = true; break; }
          if (isa<gpu::LaunchOp, func::FuncOp>(ancestor)) break;
          ancestor = ancestor->getParentOp();
        }
        safeToInsert = uniformBounds && !insideIf;
      }
    }
    if (safeToInsert) {

      bool alreadyHasLoopTopBarrier =
          !barrierPoints.empty() && barrierPoints.front() == &block->front();
      if (!alreadyHasLoopTopBarrier)
        barrierPoints.push_back(block->getTerminator());
    }
  }

  // Insert barriers in reverse order to preserve iterator validity.
  for (Operation *pt : llvm::reverse(barrierPoints)) {
    builder.setInsertionPoint(pt);
    builder.create<NVVM::Barrier0Op>(pt->getLoc());
  }
}

//===----------------------------------------------------------------------===//
// §4  Pass
//===----------------------------------------------------------------------===//

struct NovaGPUInsertWorkgroupBarriersPass
    : public PassWrapper<NovaGPUInsertWorkgroupBarriersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUInsertWorkgroupBarriersPass)
  NovaGPUInsertWorkgroupBarriersPass() = default;
  NovaGPUInsertWorkgroupBarriersPass(
      const NovaGPUInsertWorkgroupBarriersPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect,
                    NVVM::NVVMDialect, vector::VectorDialect,
                    linalg::LinalgDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    funcOp.walk([&](gpu::LaunchOp launchOp) {
      Region &launchRegion = launchOp.getBody();
      for (Block &block : launchRegion)
        insertBarriersInBlock(builder, &block);
    });
  }

  StringRef getArgument() const override {
    return "nova-gpu-insert-workgroup-barriers";
  }
  StringRef getDescription() const override {
    return "Inserts nvvm.barrier0 at workgroup memory transition points inside "
           "gpu.launch bodies. Two barriers per K-loop iteration: barrier #1 "
           "at loop-top (read→write, protects previous iter's smem reads) and "
           "barrier #2 at write→read transitions (protects smem writes before "
           "compute reads). All barriers placed at unconditional scope — never "
           "inside scf.if — preventing bar.sync deadlocks.";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUInsertWorkgroupBarriersPass() {
  return std::make_unique<NovaGPUInsertWorkgroupBarriersPass>();
}

void registerNovaGPUInsertWorkgroupBarriersPass() {
  PassRegistration<NovaGPUInsertWorkgroupBarriersPass>();
}

} // namespace mlir::nova