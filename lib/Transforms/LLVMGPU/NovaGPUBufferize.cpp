// Nova GPU Comprehensive Bufferize Pass
//
// This is Nova's equivalent of IREE's IREEComprehensiveBufferizePass.
// It performs two jobs:
//
//  1. Erase nova.fusion_barrier ops (they are pure identities that
//     only block fusion analysis during tiling; they have no
//     bufferized form and must be removed before bufferization).
//
//  2. Run OneShotBufferize on the whole module with:
//     - bufferizeFunctionBoundaries = true
//       → converts function signature from tensors to memrefs
//     - function-boundary-type-conversion = identity-layout-map
//       → uses contiguous memref types (no strided layouts)
//       → produces clean `memref<NxMxf32>` function args for GPU kernels
//     - GPU-aware allocation function:
//       * #gpu.address_space<workgroup>  → memref.alloc   (shared SRAM)
//       * #gpu.address_space<private>   → memref.alloca  (per-thread register)
//         BUT only when the insertion point is inside an scf.forall kernel;
//         at function scope, falls back to memref.alloc (global memory).
//       * no memory space specified     → memref.alloc   (default/global)
//     - GPU-aware copy function:
//       * emits memref.copy
//       * wraps the copy with gpu.barrier when either operand is in
//         workgroup (shared) memory.
//
// IREE reference:
//   IREEComprehensiveBufferizePass::runOnOperation()

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::nova {

// ---------------------------------------------------------------------------
// GPU allocation function — mirrors IREE's gpuRequireMemSpaceAllocationFn
// ---------------------------------------------------------------------------
// Routes the allocation to the right MLIR op based on memory space.
// With bufferizeFunctionBoundaries=true, this is called for ALL allocations
// including function result buffers.
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

  // Workgroup (shared) memory → heap allocated for the whole workgroup.
  if (memSpace && cast<gpu::AddressSpaceAttr>(memSpace).getValue() ==
                      gpu::GPUDialect::getWorkgroupAddressSpace()) {
    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), wkgpSpace);
    return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // Private → memref.alloca (per-thread register/stack storage)
  //
  // IMPORTANT: memref.alloca with a private address space is only valid *inside*
  // a GPU kernel (an scf.forall that maps to GPU blocks/threads).  If the
  // builder insertion point is at function scope (outside every scf.forall),
  // the alloca would appear between kernels — which is illegal because there is
  // no active GPU thread to own the private storage at host scope.
  //
  // Defence-in-depth guard: if we are outside every scf.forall, demote this
  // private allocation to a plain global-memory memref.alloc.  The primary
  // prevention is in NovaGPUInferMemorySpacePass (which now only tags private
  // for alloc_tensors nested inside kernels), but this guard catches any edge
  // cases that slip through.
  if (memSpace) {
    // Check whether the current builder insertion point is inside a kernel.
    bool insideKernel = false;
    Operation *insertionParent =
        builder.getInsertionBlock()->getParentOp();
    while (insertionParent) {
      if (isa<scf::ForallOp>(insertionParent)) {
        insideKernel = true;
        break;
      }
      insertionParent = insertionParent->getParentOp();
    }

    if (!insideKernel) {
      // We are at function scope — emit a plain global-memory alloc instead.
      // Strip the private address space so the resulting memref is compatible
      // with host-scope ops and survives across kernel launches.
      auto globalType = MemRefType::get(memRefType.getShape(),
                                        memRefType.getElementType());
      return memref::AllocOp::create(builder, loc, globalType, dynamicSizes)
                 .getResult();
    }

    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), privateSpace);

    // GPU cannot allocate dynamic private memory (illegal on NVIDIA with PTX < 7.3)
    // If we have dynamic dimensions, allocate a conservative static size instead
    if (!dynamicSizes.empty()) {
      // Convert dynamic dimensions to static using maximum thread tile size
      SmallVector<int64_t> staticShape;
      for (int d = 0; d < memRefType.getRank(); ++d) {
        if (memRefType.isDynamicDim(d)) {
          // Use conservative maximum: 4 (standard thread tile size)
          // This matches the thread-level tiling configuration used in NovaGPUApplyTilingLevelThreadPass
          staticShape.push_back(4);
        } else {
          staticShape.push_back(memRefType.getDimSize(d));
        }
      }

      auto staticAllocType =
          MemRefType::get(staticShape, memRefType.getElementType(),
                          AffineMap(), privateSpace);
      // Create allocation without dynamic sizes - bufferization will handle subview
      SmallVector<Value> emptyDynamicSizes;
      return memref::AllocaOp::create(builder, loc, staticAllocType, emptyDynamicSizes)
          .getResult();
    }

    return memref::AllocaOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // No memory space → default (global memory for cross-kernel buffers).
  // These are managed by buffer-deallocation-pipeline.
  return memref::AllocOp::create(builder, loc, memRefType, dynamicSizes)
      .getResult();
}

// Helper: true when the memref is in GPU workgroup (shared) address space.
static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

// GPU copy function — mirrors IREE's defaultMemCpyFn.
//
// Emits a linalg.copy op instead of memref.copy. This is critical for GPU
// kernels:
//
//   memref.copy → FinalizeMemRefToLLVM emits llvm.call @memrefCopy (a HOST
//                 runtime symbol) which does not exist inside a gpu.func.
//                 This causes a missing-symbol error during NVVM lowering.
//
//   linalg.copy → createConvertLinalgToLoopsPass (Step 11 in Passes.cpp)
//                 expands it into scf.for + memref.load/store loops that
//                 compile cleanly to PTX load/store instructions inside the
//                 kernel. This is exactly how IREE handles padding copies.
//
// NOTE: gpu.barrier is NOT inserted here for the same reason as before
// ("unknown memory side effects" break OneShotBufferize analysis).
// Barriers around workgroup memory copies are inserted post-bufferization
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  // If we are inside an scf.forall (which will be lowered to a GPU kernel),
  // we use linalg.copy. This is expanded into scf.for + memref.load/store
  // loops that compile cleanly to PTX load/store instructions.
  //
  // Outside scf.forall (host side), we use memref.copy, which can be
  // efficiently handled by the host runtime or lowered to specialized
  // host-side copy routines.
  Operation *parent = builder.getInsertionBlock()->getParentOp();
  bool insideForall = false;
  while (parent) {
    if (isa<scf::ForallOp>(parent)) {
      insideForall = true;
      break;
    }
    parent = parent->getParentOp();
  }

  if (insideForall) {
    if (auto memRefType = llvm::dyn_cast<MemRefType>(from.getType())) {
      if (memRefType.getRank() == 0) {
        // Only hoist if `from` is defined OUTSIDE the scf.forall loop.
        bool isDefinedOutside = true;
        if (Operation *defOp = from.getDefiningOp()) {
          if (parent->isAncestor(defOp)) isDefinedOutside = false;
        } else if (auto arg = llvm::dyn_cast<BlockArgument>(from)) {
          if (parent->isAncestor(arg.getOwner()->getParentOp())) isDefinedOutside = false;
        }

        if (isDefinedOutside) {
          Value scalarInit;
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(parent); // parent is the scf::ForallOp
            scalarInit = builder.create<memref::LoadOp>(loc, from);
          }
          builder.create<memref::StoreOp>(loc, scalarInit, to);
          return success();
        }
      }
    }
    linalg::CopyOp::create(builder, loc, from, to);
  } else {
    memref::CopyOp::create(builder, loc, from, to);
  }
  return success();
}

// ---------------------------------------------------------------------------
// Pass definition — operates on ModuleOp for function boundary bufferization
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
                    gpu::GPUDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    nova::NovaDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Erase nova.fusion_barrier ops across all functions.
    IRRewriter rewriter(moduleOp.getContext());
    SmallVector<FusionBarrierOp> barriers;
    moduleOp.walk([&](FusionBarrierOp b) { barriers.push_back(b); });
    for (FusionBarrierOp b : barriers)
      rewriter.replaceOp(b, b.getSource());

    // GPU-aware OneShotBufferize on the whole module.
    bufferization::OneShotBufferizationOptions opts;
    opts.allocationFn = gpuRequireMemSpaceAllocationFn;
    opts.memCpyFn = gpuCopyFn;
    // Bufferize function boundaries: convert function signatures from
    // tensors to memrefs, eliminating bufferization.to_buffer/to_tensor.
    opts.bufferizeFunctionBoundaries = true;
    // Use identity layout: produces clean contiguous memref types
    // (memref<NxMxf32>) instead of strided types for function args/results.
    opts.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    opts.checkParallelRegions = false;

    bufferization::BufferizationState bufState;
    if (failed(
            bufferization::runOneShotBufferize(moduleOp, opts, bufState))) {
      moduleOp.emitOpError("GPU-aware bufferization failed");
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-comprehensive-bufferize";
  }
  StringRef getDescription() const override {
    return "Erases nova.fusion_barrier ops then runs OneShotBufferize with "
           "GPU-aware allocation, identity layout map for function boundaries, "
           "and barrier-fenced copies for workgroup memory";
  }
};

std::unique_ptr<Pass> createNovaGPUComprehensiveBufferizePass() {
  return std::make_unique<NovaGPUComprehensiveBufferizePass>();
}

void registerNovaGPUComprehensiveBufferizePass() {
  PassRegistration<NovaGPUComprehensiveBufferizePass>();
}

// ---------------------------------------------------------------------------
// Post-bufferization barrier insertion pass
// Walks memref.copy ops and inserts gpu.barrier before/after copies
// involving workgroup (shared) memory. This must run AFTER bufferization
// because gpu.barrier has "unknown side effects" that break
// OneShotBufferize analysis.
// ---------------------------------------------------------------------------

/// Returns true if `op` (or any op nested inside it) stores to workgroup memory.
static bool hasWorkgroupStores(Operation *op) {
  bool found = false;
  op->walk([&](memref::StoreOp storeOp) {
    if (isWorkgroupMemref(cast<MemRefType>(storeOp.getMemRef().getType())))
      found = true;
  });
  if (!found) {
    op->walk([&](memref::CopyOp copyOp) {
      if (isWorkgroupMemref(cast<MemRefType>(copyOp.getTarget().getType())))
        found = true;
    });
  }
  return found;
}

/// Returns true if `op` (or any op nested inside it) loads from workgroup memory.
static bool hasWorkgroupLoads(Operation *op) {
  bool found = false;
  op->walk([&](memref::LoadOp loadOp) {
    if (isWorkgroupMemref(cast<MemRefType>(loadOp.getMemRef().getType())))
      found = true;
  });
  if (!found) {
    op->walk([&](memref::CopyOp copyOp) {
      if (isWorkgroupMemref(cast<MemRefType>(copyOp.getSource().getType())))
        found = true;
    });
  }
  return found;
}

struct NovaGPUInsertWorkgroupBarriersPass
    : public PassWrapper<NovaGPUInsertWorkgroupBarriersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUInsertWorkgroupBarriersPass)

  NovaGPUInsertWorkgroupBarriersPass() = default;
  NovaGPUInsertWorkgroupBarriersPass(
      const NovaGPUInsertWorkgroupBarriersPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Strategy: walk inside gpu.launch bodies and find transition points
    // where workgroup stores are followed by workgroup loads. Insert
    // gpu.barrier at each such transition.
    //
    // The pattern in the K-loop body is:
    //   scf.for { store to workgroup }    // global → shared copy
    //   scf.for { load from workgroup }   // shared → private copy
    //   scf.for { store to workgroup }    // global → shared copy (op B)
    //   scf.for { load from workgroup }   // shared → private copy (op B)
    //   scf.for { compute }               // matmul from private
    //
    // We need barriers between write-then-read transitions.

    funcOp.walk([&](gpu::LaunchOp launchOp) {
      // Walk all scf.for ops to find K-loops or any loop with workgroup
      // write→read transitions in its body.
      launchOp.walk([&](scf::ForOp forOp) {
        insertBarriersAtTransitions(builder, forOp);
      });
    });

    // Also handle memref.copy ops (in case any remain un-lowered).
    funcOp.walk([&](memref::CopyOp copyOp) {
      bool needsBarrier = false;
      if (isWorkgroupMemref(cast<MemRefType>(copyOp.getSource().getType())))
        needsBarrier = true;
      if (isWorkgroupMemref(cast<MemRefType>(copyOp.getTarget().getType())))
        needsBarrier = true;

      if (!needsBarrier)
        return;

      builder.setInsertionPoint(copyOp);
      gpu::BarrierOp::create(builder, copyOp.getLoc());
      builder.setInsertionPointAfter(copyOp);
      gpu::BarrierOp::create(builder, copyOp.getLoc());
    });
  }

  /// Walk the body of `forOp` and insert gpu.barrier between consecutive
  /// top-level ops where the first writes to workgroup memory and the
  /// next reads from workgroup memory.
  void insertBarriersAtTransitions(OpBuilder &builder, scf::ForOp forOp) {
    Block *body = forOp.getBody();
    SmallVector<std::pair<Operation *, Operation *>> transitions;

    Operation *prevOp = nullptr;
    for (Operation &op : body->getOperations()) {
      // Skip the yield terminator.
      if (isa<scf::YieldOp>(op))
        continue;

      if (prevOp && hasWorkgroupStores(prevOp) && hasWorkgroupLoads(&op)) {
        transitions.push_back({prevOp, &op});
      }
      prevOp = &op;
    }

    // Insert barriers (in reverse to avoid invalidating iterators).
    for (auto [writerOp, readerOp] : llvm::reverse(transitions)) {
      builder.setInsertionPoint(readerOp);
      gpu::BarrierOp::create(builder, readerOp->getLoc());
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-insert-workgroup-barriers";
  }
  StringRef getDescription() const override {
    return "Inserts gpu.barrier at workgroup memory write→read transitions "
           "inside gpu.launch bodies. Runs post-bufferization.";
  }
};

std::unique_ptr<Pass> createNovaGPUInsertWorkgroupBarriersPass() {
  return std::make_unique<NovaGPUInsertWorkgroupBarriersPass>();
}

void registerNovaGPUInsertWorkgroupBarriersPass() {
  PassRegistration<NovaGPUInsertWorkgroupBarriersPass>();
}

} // namespace mlir::nova