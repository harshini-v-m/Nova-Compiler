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
//       * #gpu.address_space<workgroup>  → memref.alloc   (shared SRAM),
//         hoisted above any enclosing scf.for to avoid per-iteration
//         reallocation of statically-partitioned shared memory.
//       * #gpu.address_space<workgroup>  → memref.alloc   (shared SRAM),
//         hoisted above any enclosing scf.for to avoid per-iteration
//         reallocation of statically-partitioned shared memory.
//       * #gpu.address_space<private>   → memref.alloca  (per-thread register)
//         BUT only when the insertion point is inside an scf.forall kernel;
//         at function scope, falls back to memref.alloc (global memory).
//       * no memory space specified     → memref.alloc   (default/global)
//     - GPU-aware copy function:
//       * inside scf.forall → linalg.copy (expands to load/store loops,
//         avoids host-only @memrefCopy symbol inside gpu.func)
//       * outside scf.forall → memref.copy (host runtime path)
//     - Barriers are NOT inserted during bufferization (gpu.barrier has
//       "unknown side effects" that break OneShotBufferize analysis).
//       They are inserted post-bufferization by
//       NovaGPUInsertWorkgroupBarriersPass.
//       * inside scf.forall → linalg.copy (expands to load/store loops,
//         avoids host-only @memrefCopy symbol inside gpu.func)
//       * outside scf.forall → memref.copy (host runtime path)
//     - Barriers are NOT inserted during bufferization (gpu.barrier has
//       "unknown side effects" that break OneShotBufferize analysis).
//       They are inserted post-bufferization by
//       NovaGPUInsertWorkgroupBarriersPass.
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
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
  //
  // IMPORTANT: GPU workgroup (shared) memory is statically partitioned at
  // kernel launch. Allocating inside an scf.for loop body produces a new
  // alloc/dealloc pair every iteration, which is semantically wrong — the
  // shared memory slot is fixed for the lifetime of the kernel. We hoist
  // the alloc to just before the outermost enclosing scf.for that is still
  // inside the gpu.launch / scf.forall kernel boundary.
  if (memSpace && cast<gpu::AddressSpaceAttr>(memSpace).getValue() ==
                      gpu::GPUDialect::getWorkgroupAddressSpace()) {
    // Guard: workgroup (shared) memory is only valid inside a GPU kernel.
    // If the bufferizer is inserting at function scope (outside every
    // scf.forall), the alloc_tensor was defined at function level due to CSE
    // hoisting. Allocating memref<..., 3> at host scope and passing it as a
    // kernel argument is wrong — __shared__ has no host-visible address.
    // Fall back to plain global memory so the pipeline doesn't produce
    // illegal PTX. The real fix is to sink the alloc_tensor into the
    // block-forall scope before bufferization (see NovaGPUSinkAllocTensors).
    bool insideKernel = false;
    Operation *check = builder.getInsertionBlock()->getParentOp();
    while (check) {
      if (isa<scf::ForallOp>(check)) {
        insideKernel = true;
        break;
      }
      check = check->getParentOp();
    }
    if (!insideKernel) {
      auto globalType = MemRefType::get(memRefType.getShape(),
                                        memRefType.getElementType());
      return memref::AllocOp::create(builder, loc, globalType, dynamicSizes)
          .getResult();
    }

    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), wkgpSpace);

    // Walk up the op-parent chain and record the outermost scf.for that is
    // still inside the kernel. Stop at gpu.launch / scf.forall boundaries.
    OpBuilder::InsertionGuard guard(builder);
    Operation *hoistTarget = nullptr;
    Operation *cur = builder.getInsertionBlock()->getParentOp();
    while (cur) {
      if (isa<gpu::LaunchOp, scf::ForallOp>(cur))
        break;
      if (isa<scf::ForOp>(cur))
        hoistTarget = cur;
      cur = cur->getParentOp();
    }
    if (hoistTarget)
      builder.setInsertionPoint(hoistTarget);

    return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // Private → memref.alloca (per-thread register/stack storage)
  //
  // IMPORTANT: memref.alloca with a private address space is only valid *inside*
  // a GPU kernel (an scf.forall that maps to GPU blocks/threads). If the
  // builder insertion point is at function scope (outside every scf.forall),
  // the alloca would appear between kernels — which is illegal because there is
  // no active GPU thread to own the private storage at host scope.
  //
  // Defence-in-depth guard: if we are outside every scf.forall, demote this
  // private allocation to a plain global-memory memref.alloc. The primary
  // prevention is in NovaGPUInferMemorySpacePass (which now only tags private
  // for alloc_tensors nested inside kernels), but this guard catches any edge
  // cases that slip through.
  if (memSpace) {
    // Check whether the current builder insertion point is inside a kernel.
    bool insideKernel = false;
    Operation *insertionParent = builder.getInsertionBlock()->getParentOp();
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

    // GPU cannot allocate dynamic private memory (illegal on NVIDIA with PTX < 7.3).
    // If we have dynamic dimensions, allocate a conservative static size instead.
    if (!dynamicSizes.empty()) {
      SmallVector<int64_t> staticShape;
      for (int d = 0; d < memRefType.getRank(); ++d) {
        if (memRefType.isDynamicDim(d)) {
          // Use conservative maximum: 4 (standard thread tile size).
          // This matches the thread-level tiling configuration used in
          // NovaGPUApplyTilingLevelThreadPass.
          staticShape.push_back(4);
        } else {
          staticShape.push_back(memRefType.getDimSize(d));
        }
      }
      auto staticAllocType =
          MemRefType::get(staticShape, memRefType.getElementType(),
                          AffineMap(), privateSpace);
      SmallVector<Value> emptyDynamicSizes;
      return memref::AllocaOp::create(builder, loc, staticAllocType,
                                      emptyDynamicSizes)
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns true when the memref is in GPU workgroup (shared) address space.
static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

// ---------------------------------------------------------------------------
// GPU copy function — mirrors IREE's defaultMemCpyFn.
//
// Emits linalg.copy inside GPU kernels (scf.forall), memref.copy on the host.
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
// NOTE: gpu.barrier is NOT inserted here — "unknown memory side effects"
// break OneShotBufferize analysis. Barriers are inserted post-bufferization
// by NovaGPUInsertWorkgroupBarriersPass.
// ---------------------------------------------------------------------------
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
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
        // Scalar (rank-0) copy: hoist the load outside the forall if `from`
        // is defined outside the kernel, then store inside.
        // Scalar (rank-0) copy: hoist the load outside the forall if `from`
        // is defined outside the kernel, then store inside.
        bool isDefinedOutside = true;
        if (Operation *defOp = from.getDefiningOp()) {
          if (parent->isAncestor(defOp))
            isDefinedOutside = false;
          if (parent->isAncestor(defOp))
            isDefinedOutside = false;
        } else if (auto arg = llvm::dyn_cast<BlockArgument>(from)) {
          if (parent->isAncestor(arg.getOwner()->getParentOp()))
            isDefinedOutside = false;
          if (parent->isAncestor(arg.getOwner()->getParentOp()))
            isDefinedOutside = false;
        }

        if (isDefinedOutside) {
          // Load the current value of `from` at the current insertion point
          // (inside the forall body, AFTER any preceding accumulation). Do
          // NOT hoist the load before the forall: `from` is a GPU-side
          // accumulator that gets updated within this forall block (e.g. by
          // a preceding atomic-add reduction), so hoisting the load would
          // capture the pre-reduction zero instead of the computed sum.
          Value scalarInit = builder.create<memref::LoadOp>(loc, from);
          builder.create<memref::StoreOp>(loc, scalarInit, to);
          return success();
        }
      }
    }

    // Fix shape mismatches between source and destination.
    //
    // gpuRequireMemSpaceAllocationFn converts dynamic private alloc_tensors
    // (e.g. tensor<4x?xf32>) to static allocas (memref<4x4xf32>) because
    // NVPTX cannot do dynamic stack allocation.  When OneShotBufferize
    // lowers tensor.insert_slice, it calls this copy function with:
    //   from = static alloca (4x4)
    //   to   = dynamic subview of workgroup buffer (4x?)
    //
    // convert-linalg-to-loops resolves the dimension conflict by preferring
    // the static size (4), causing out-of-bounds writes that overwrite
    // zero-padding in the destination tile.
    //
    // Fix: subview the static source to match the destination's dynamic shape.
    auto fromType = cast<MemRefType>(from.getType());
    auto toType = cast<MemRefType>(to.getType());
    if (fromType.getRank() == toType.getRank() && fromType.getRank() > 0) {
      bool needsSubview = false;
      int rank = fromType.getRank();
      SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes;
      for (int i = 0; i < rank; ++i) {
        if (!fromType.isDynamicDim(i) && toType.isDynamicDim(i)) {
          // Source is static, dest is dynamic → use dest's dynamic size
          sizes.push_back(
              builder.create<memref::DimOp>(loc, to, i).getResult());
          needsSubview = true;
        } else if (!fromType.isDynamicDim(i)) {
          sizes.push_back(builder.getIndexAttr(fromType.getDimSize(i)));
        } else {
          sizes.push_back(
              builder.create<memref::DimOp>(loc, from, i).getResult());
        }
      }
      if (needsSubview)
        from = builder.create<memref::SubViewOp>(loc, from, offsets, sizes,
                                                 strides);
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
                    gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, nova::NovaDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Erase nova.fusion_barrier ops across all functions. These are pure
    // identity ops used only to block fusion analysis during tiling; they
    // have no bufferized form.
    IRRewriter rewriter(moduleOp.getContext());
    SmallVector<FusionBarrierOp> fusionBarriers;
    moduleOp.walk([&](FusionBarrierOp b) { fusionBarriers.push_back(b); });
    for (FusionBarrierOp b : fusionBarriers)
      rewriter.replaceOp(b, b.getSource());

    // Erase nova.value_barrier ops — tensor-level synchronization markers
    // generated by LowerBarrierRegion. At the memref level, actual
    // gpu.barrier synchronization is inserted post-bufferization around
    // workgroup memory accesses by NovaGPUInsertWorkgroupBarriersPass.
    SmallVector<ValueBarrierOp> valueBarriers;
    moduleOp.walk([&](ValueBarrierOp b) { valueBarriers.push_back(b); });
    for (ValueBarrierOp b : valueBarriers)
      rewriter.replaceOp(b, b.getInputs());

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
    if (failed(bufferization::runOneShotBufferize(moduleOp, opts, bufState))) {
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
//
// Walks gpu.launch bodies and inserts nvvm.barrier0 at workgroup memory
// write→read transitions. Must run AFTER bufferization because nvvm.barrier0
// has "unknown side effects" that break OneShotBufferize analysis.
//
// Strategy:
//   insertBarriersInBlock walks ANY block recursively (gpu.launch body or
//   scf.for body) and detects write→read transitions at every level.
//   This ensures transitions at the top-level launch block (outside any
//   scf.for) are not missed — a gap present in approaches that only walk
//   scf.for bodies.
//
//   scf.if branches are intentionally NOT recursed into: nvvm.barrier0
//   requires ALL threads in the workgroup to reach it. Inserting a barrier
//   inside a non-uniform conditional (e.g. `if thread_id < 64`) causes
//   deadlock. The parent-level scan treats the entire scf.if as one unit:
//   if it contains workgroup loads, the barrier is placed before the
//   scf.if op itself, where all threads execute.
// ---------------------------------------------------------------------------

/// Returns true if `op` (or any op nested inside it) performs a non-atomic
/// store to workgroup memory.
static bool hasNonAtomicWorkgroupStores(Operation *op) {
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
  // vector.transfer_write / vector.store to workgroup memory.
  if (!found) {
    op->walk([&](vector::TransferWriteOp writeOp) {
      if (auto mt = dyn_cast<MemRefType>(writeOp.getBase().getType()))
        if (isWorkgroupMemref(mt)) found = true;
    });
  }
  if (!found) {
    op->walk([&](vector::StoreOp storeOp) {
      if (isWorkgroupMemref(cast<MemRefType>(storeOp.getBase().getType())))
        found = true;
    });
  }
  return found;
}

/// Returns true if `op` (or any op nested inside it) performs ANY store
/// (atomic or non-atomic) to workgroup memory.
static bool hasWorkgroupStores(Operation *op) {
  if (hasNonAtomicWorkgroupStores(op))
    return true;
  bool found = false;
  op->walk([&](memref::AtomicRMWOp rmwOp) {
    if (isWorkgroupMemref(cast<MemRefType>(rmwOp.getMemref().getType())))
      found = true;
  });
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
    op->walk([&](memref::AtomicRMWOp rmwOp) {
      if (isWorkgroupMemref(cast<MemRefType>(rmwOp.getMemref().getType())))
        found = true;
    });
  }
  if (!found) {
    op->walk([&](memref::CopyOp copyOp) {
      if (isWorkgroupMemref(cast<MemRefType>(copyOp.getSource().getType())))
        found = true;
    });
  }
  // vector.transfer_read / vector.load from workgroup memory.
  if (!found) {
    op->walk([&](vector::TransferReadOp readOp) {
      if (auto mt = dyn_cast<MemRefType>(readOp.getBase().getType()))
        if (isWorkgroupMemref(mt)) found = true;
    });
  }
  if (!found) {
    op->walk([&](vector::LoadOp loadOp) {
      if (isWorkgroupMemref(cast<MemRefType>(loadOp.getBase().getType())))
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
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect,
                    NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Walk each gpu.launch body. insertBarriersInBlock recurses into
    // scf.for bodies itself, so a single top-level call per launch suffices
    // and correctly handles transitions both at the launch-body level and
    // inside any nested loops.
    funcOp.walk([&](gpu::LaunchOp launchOp) {
      Region &launchRegion = launchOp.getBody();
      for (Block &block : launchRegion)
        insertBarriersInBlock(builder, &block);
    });

    // Handle any memref.copy ops involving workgroup memory that remain
    // un-lowered after bufferization. Wrap each with a barrier on both sides.
    funcOp.walk([&](memref::CopyOp copyOp) {
      bool needsBarrier =
          isWorkgroupMemref(cast<MemRefType>(copyOp.getSource().getType())) ||
          isWorkgroupMemref(cast<MemRefType>(copyOp.getTarget().getType()));
      if (!needsBarrier)
        return;
      builder.setInsertionPoint(copyOp);
      builder.create<NVVM::Barrier0Op>(copyOp.getLoc());
      builder.setInsertionPointAfter(copyOp);
      builder.create<NVVM::Barrier0Op>(copyOp.getLoc());
    });
  }

  /// Walk the ops in `block` and insert nvvm.barrier0 at workgroup memory
  /// write→read transitions. Recurses into scf.for bodies only.
  ///
  /// scf.if branches are intentionally NOT recursed into — see class comment.
  void insertBarriersInBlock(OpBuilder &builder, Block *block) {
    SmallVector<Operation *> barrierPoints;
    bool seenWorkgroupStore = false;
    bool seenNonAtomicStore = false;

    for (Operation &op : block->getOperations()) {
      if (isa<scf::YieldOp, gpu::TerminatorOp>(op))
        continue;

      bool hasLoads  = hasWorkgroupLoads(&op);
      bool hasStores = hasWorkgroupStores(&op);

      // Write→read transition: place barrier before the reading op so all
      // threads see completed stores before any thread proceeds to load.
      if (seenWorkgroupStore && hasLoads) {
        barrierPoints.push_back(&op);
        seenWorkgroupStore = false;
      }

      if (hasStores) {
        seenWorkgroupStore = true;
        if (hasNonAtomicWorkgroupStores(&op))
          seenNonAtomicStore = true;
      }

      // Recurse into scf.for bodies ONLY when loop bounds are compile-time
      // constants (uniform across all threads).
      //
      // Dynamic bounds (e.g. affine.min of a thread-id-derived value) mean
      // different threads execute different iteration counts.  Placing a
      // barrier inside such a loop — even a tail barrier before scf.yield —
      // causes a deadlock because some threads exit earlier than others and
      // never reach the barrier.  Treat the entire scf.for as an opaque
      // unit: hasWorkgroupStores/Loads already recurse into it for
      // write→read transition detection at the enclosing scope.
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto isConstIdx = [](Value v) -> bool {
          if (auto cst = v.getDefiningOp<arith::ConstantOp>())
            return isa<IntegerAttr>(cst.getValue());
          if (v.getDefiningOp<arith::ConstantIndexOp>())
            return true;
          return false;
        };
        bool uniformBounds = isConstIdx(forOp.getLowerBound()) &&
                             isConstIdx(forOp.getUpperBound()) &&
                             isConstIdx(forOp.getStep());
        if (uniformBounds)
          insertBarriersInBlock(builder, forOp.getBody());
      }
    }

    // Tail barrier: protect the next loop iteration from reading shared memory
    // before this iteration's non-atomic stores are globally visible.
    // Only needed inside a loop (scf.yield terminator). At gpu.launch level
    // (gpu.terminator) there is no next iteration, so no tail barrier needed.
    if (seenNonAtomicStore && isa<scf::YieldOp>(block->getTerminator()))
      barrierPoints.push_back(block->getTerminator());

    // Insert in reverse order to preserve iterator validity.
    for (Operation *pt : llvm::reverse(barrierPoints)) {
      builder.setInsertionPoint(pt);
      builder.create<NVVM::Barrier0Op>(pt->getLoc());
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-insert-workgroup-barriers";
  }
  StringRef getDescription() const override {
    return "Inserts nvvm.barrier0 at workgroup memory write→read transitions "
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