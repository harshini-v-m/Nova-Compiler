// Nova GPU Comprehensive Bufferize Pass
//
// This is Nova's equivalent of IREE's IREEComprehensiveBufferizePass.
// It performs two jobs:
//
//  1. Erase nova.fusion_barrier / nova.value_barrier ops (pure identities
//     that only block fusion analysis during tiling; no bufferized form).
//
//  2. Run OneShotBufferize on the whole module with:
//     - bufferizeFunctionBoundaries = true
//       → converts function signature from tensors to memrefs
//     - function-boundary-type-conversion = identity-layout-map
//       → uses contiguous memref types (no strided layouts)
//       → produces clean `memref<NxMxf32>` function args for GPU kernels
//     - GPU-aware allocation function:
//       * #gpu.address_space<workgroup>  → memref.alloc (shared SRAM),
//         hoisted above any enclosing scf.for that is still inside the
//         BLOCK-LEVEL scf.forall (distinguished via gpu.block mapping).
//       * #gpu.address_space<private>    → memref.alloca (per-thread),
//         only inside a kernel; falls back to memref.alloc at host scope.
//       * no memory space               → memref.alloc (global memory).
//     - GPU-aware copy function:
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
// Helpers
// ---------------------------------------------------------------------------

/// Returns true when the memref is in GPU workgroup (shared) address space.
static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Returns true if `forallOp` is a BLOCK-level kernel, i.e. its mapping
/// attribute contains at least one gpu.block<x/y/z> dimension.
/// Thread-level foralls use gpu.thread<...> or gpu.warp<...> mappings.
///
/// This is used by the allocation hoisting logic to stop hoisting AT the
/// block-level forall rather than at the first (possibly thread-level) forall
/// it encounters while walking up the parent chain.
static bool isBlockLevelForall(scf::ForallOp forallOp) {
  auto mappingAttr = forallOp.getMapping();
  if (!mappingAttr)
    return false;
  for (Attribute attr : mappingAttr->getValue()) {
    if (auto gpuAttr = dyn_cast<gpu::GPUBlockMappingAttr>(attr))
      return true;
  }
  return false;
}

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

  // -------------------------------------------------------------------------
  // Workgroup (shared) memory → heap allocated for the whole workgroup.
  //
  // IMPORTANT: GPU workgroup (shared) memory is statically partitioned at
  // kernel launch. Allocating inside an scf.for loop body produces a new
  // alloc/dealloc pair every iteration, which is semantically wrong — the
  // shared memory slot is fixed for the lifetime of the kernel. We hoist
  // the alloc to just before the outermost enclosing scf.for that is still
  // inside the BLOCK-LEVEL gpu.launch / scf.forall kernel boundary.
  //
  // FIX vs original: the original code stopped at the FIRST scf::ForallOp
  // encountered while walking up the parent chain. When thread-level
  // scf.forall ops are nested inside a block-level scf.forall, this caused
  // workgroup allocs to be placed inside the thread forall rather than at
  // the block forall level, producing illegal PTX (shared memory address
  // taken per-thread instead of per-block).
  //
  // The fix uses isBlockLevelForall() to distinguish block-mapped foralls
  // (gpu.block<x/y/z>) from thread-mapped ones (gpu.thread<...>), and only
  // stops walking at the block-level boundary.
  // -------------------------------------------------------------------------
  if (memSpace && cast<gpu::AddressSpaceAttr>(memSpace).getValue() ==
                      gpu::GPUDialect::getWorkgroupAddressSpace()) {
    // Guard: workgroup (shared) memory is only valid inside a GPU kernel.
    // If the bufferizer is inserting at function scope (outside every
    // block-level scf.forall), the alloc_tensor was defined at function level
    // due to CSE hoisting. Allocating memref<..., 3> at host scope and
    // passing it as a kernel argument is wrong — __shared__ has no
    // host-visible address. Fall back to plain global memory so the pipeline
    // does not produce illegal PTX. The real fix is to sink the alloc_tensor
    // into the block-forall scope before bufferization
    // (see NovaGPUSinkAllocTensors).
    bool insideBlockKernel = false;
    Operation *check = builder.getInsertionBlock()->getParentOp();
    while (check) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(check)) {
        if (isBlockLevelForall(forallOp)) {
          insideBlockKernel = true;
          break;
        }
      }
      if (isa<gpu::LaunchOp>(check)) {
        insideBlockKernel = true;
        break;
      }
      check = check->getParentOp();
    }
    if (!insideBlockKernel) {
      auto globalType = MemRefType::get(memRefType.getShape(),
                                        memRefType.getElementType());
      return memref::AllocOp::create(builder, loc, globalType, dynamicSizes)
          .getResult();
    }

    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), wkgpSpace);

    // Walk up the op-parent chain and record the outermost scf.for that is
    // still inside the BLOCK-LEVEL kernel boundary. Stop at gpu.launch or
    // block-level scf.forall.
    //
    // FIX vs original: original stopped at ANY scf::ForallOp. Now we
    // continue walking through thread-level foralls and only stop at the
    // block-level forall (or gpu.launch), so the hoist target is always
    // inside the block-level kernel scope, not inside a thread sub-region.
    OpBuilder::InsertionGuard guard(builder);
    Operation *hoistTarget = nullptr;
    Operation *cur = builder.getInsertionBlock()->getParentOp();
    while (cur) {
      if (isa<gpu::LaunchOp>(cur))
        break;
      if (auto forallOp = dyn_cast<scf::ForallOp>(cur)) {
        if (isBlockLevelForall(forallOp))
          break;
        // Thread-level forall: keep walking up (do not stop here).
      }
      if (isa<scf::ForOp>(cur))
        hoistTarget = cur;
      cur = cur->getParentOp();
    }
    if (hoistTarget)
      builder.setInsertionPoint(hoistTarget);

    return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // -------------------------------------------------------------------------
  // Private → memref.alloca (per-thread register/stack storage)
  //
  // IMPORTANT: memref.alloca with a private address space is only valid
  // *inside* a GPU kernel (a block-level scf.forall). If the builder
  // insertion point is at function scope (outside every block-level
  // scf.forall), demote to a plain global-memory memref.alloc.
  // -------------------------------------------------------------------------
  if (memSpace) {
    bool insideKernel = false;
    Operation *insertionParent = builder.getInsertionBlock()->getParentOp();
    while (insertionParent) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(insertionParent)) {
        if (isBlockLevelForall(forallOp)) {
          insideKernel = true;
          break;
        }
      }
      if (isa<gpu::LaunchOp>(insertionParent)) {
        insideKernel = true;
        break;
      }
      insertionParent = insertionParent->getParentOp();
    }

    if (!insideKernel) {
      auto globalType = MemRefType::get(memRefType.getShape(),
                                        memRefType.getElementType());
      return memref::AllocOp::create(builder, loc, globalType, dynamicSizes)
          .getResult();
    }

    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), privateSpace);

    // GPU cannot allocate dynamic private memory (illegal on NVIDIA with
    // PTX < 7.3). If we have dynamic dimensions, use a conservative static
    // size matching the standard thread tile size from
    // NovaGPUApplyTilingLevelThreadPass.
    if (!dynamicSizes.empty()) {
      SmallVector<int64_t> staticShape;
      for (int d = 0; d < memRefType.getRank(); ++d) {
        staticShape.push_back(memRefType.isDynamicDim(d)
                                  ? 4
                                  : memRefType.getDimSize(d));
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
  // Guard 1: self-copy — src and dst are the same memref.
  // This happens when the bufferizer resolves a tensor.parallel_insert_slice
  // of a linalg.copy result into an in-place update: both the copy output
  // slice and the insert target map to the same underlying alloc (e.g. both
  // are subviews of alloc_0 at the same offset). Emitting a copy here would
  // produce "linalg.copy ins(%x) outs(%x)" — a no-op that adds noise and
  // confuses downstream analysis. Skip it.
  if (from == to)
    return success();

  // Also check underlying base: if both are subviews of the same alloc at
  // identical offsets (the A-tile smem→smem case), elide the copy.
  // We detect this conservatively: if both Values are block arguments or
  // define the same memref::SubViewOp source, they alias and the copy is
  // a no-op.
  if (auto fromSV = from.getDefiningOp<memref::SubViewOp>())
    if (auto toSV = to.getDefiningOp<memref::SubViewOp>())
      if (fromSV.getSource() == toSV.getSource() &&
          fromSV.getStaticOffsets() == toSV.getStaticOffsets() &&
          fromSV.getStaticSizes() == toSV.getStaticSizes())
        return success();

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
        // Scalar (rank-0) copy: load the current value of `from` at the
        // current insertion point (inside the forall body, AFTER any preceding
        // accumulation) then store into `to`.
        // Do NOT hoist the load before the forall: `from` may be a GPU-side
        // accumulator updated within this forall block.
        bool isDefinedOutside = true;
        if (Operation *defOp = from.getDefiningOp()) {
          if (parent->isAncestor(defOp))
            isDefinedOutside = false;
        } else if (auto arg = llvm::dyn_cast<BlockArgument>(from)) {
          if (parent->isAncestor(arg.getOwner()->getParentOp()))
            isDefinedOutside = false;
        }

        if (isDefinedOutside) {
          Value scalarInit = builder.create<memref::LoadOp>(loc, from);
          builder.create<memref::StoreOp>(loc, scalarInit, to);
          return success();
        }
      }
    }

    // Guard 2: shape mismatch subview.
    // When `from` is statically shaped but `to` has dynamic dims (or vice
    // versa), we need to subview `from` to match `to`'s live size.
    // Use the STATIC size from `fromType` when available — don't read
    // memref.dim(to, i) for a dimension that fromType already knows statically.
    auto fromType = cast<MemRefType>(from.getType());
    auto toType   = cast<MemRefType>(to.getType());
    if (fromType.getRank() == toType.getRank() && fromType.getRank() > 0) {
      bool needsSubview = false;
      int rank = fromType.getRank();
      SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes;
      for (int i = 0; i < rank; ++i) {
        if (!fromType.isDynamicDim(i) && toType.isDynamicDim(i)) {
          // Use the static size from fromType directly — no need for dim op.
          sizes.push_back(builder.getIndexAttr(fromType.getDimSize(i)));
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
    opts.memCpyFn     = gpuCopyFn;
    // Bufferize function boundaries: convert function signatures from
    // tensors to memrefs, eliminating bufferization.to_buffer/to_tensor.
    opts.bufferizeFunctionBoundaries = true;
    // Use identity layout: produces clean contiguous memref types
    // (memref<NxMxf32>) instead of strided types for function args/results.
    opts.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    // Parallel region checking is disabled: gpu.barrier (inserted
    // post-bufferization) has "unknown side effects" that break the
    // analysis. Race detection is handled by NovaGPUInsertWorkgroupBarriersPass.
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
//   insertBarriersInBlock walks ANY block recursively and detects write→read
//   transitions at every level. The recursion covers both scf.for bodies
//   (for loop-carried tile reuse) and scf.forall bodies (for the thread-level
//   promoted copies that are the primary source of workgroup traffic).
//
//   scf.if branches are intentionally NOT recursed into: nvvm.barrier0
//   requires ALL threads in the workgroup to reach it. Inserting a barrier
//   inside a non-uniform conditional (e.g. `if thread_id < 64`) causes
//   deadlock. The parent-level scan treats the entire scf.if as one unit:
//   if it contains workgroup loads, the barrier is placed before the
//   scf.if op itself, where all threads execute.
//
// FIX vs original:
//   1. linalg.copy to workgroup memory is now detected as a workgroup store.
//      The original only checked memref.copy, memref.store, vector ops —
//      but gpuCopyFn emits linalg.copy (not memref.copy) inside foralls.
//      Without this fix every nova.promote_to_workgroup tile load was
//      invisible to the barrier analysis, causing races on ALL matmul tiles.
//
//   2. scf.forall bodies are now recursed into (with the same uniform-bounds
//      guard used for scf.for). Thread-level foralls contain the promoted
//      copies; not recursing into them meant write→read transitions inside
//      thread tile loops were never seen.
// ---------------------------------------------------------------------------

/// Returns true if `op` (or any op nested inside it) performs a non-atomic
/// store to workgroup memory.
///
/// FIX: added linalg::CopyOp check. gpuCopyFn emits linalg.copy (not
/// memref.copy) for all copies inside scf.forall regions. Without this,
/// every nova.promote_to_workgroup tile load is invisible to the barrier
/// analysis, causing data races on all promoted shared memory tiles.
static bool hasNonAtomicWorkgroupStores(Operation *op) {
  bool found = false;

  // memref.store directly to workgroup memory.
  op->walk([&](memref::StoreOp storeOp) {
    if (isWorkgroupMemref(cast<MemRefType>(storeOp.getMemRef().getType())))
      found = true;
  });

  // memref.copy targeting workgroup memory (host-scope copies).
  if (!found) {
    op->walk([&](memref::CopyOp copyOp) {
      if (isWorkgroupMemref(cast<MemRefType>(copyOp.getTarget().getType())))
        found = true;
    });
  }

  // linalg.copy targeting workgroup memory (GPU kernel copies — the primary
  // form emitted by gpuCopyFn for nova.promote_to_workgroup tile loads).
  // THIS WAS MISSING IN THE ORIGINAL and caused all barrier analysis to
  // silently skip every promoted tile, leading to races on matmul tiles.
  if (!found) {
    op->walk([&](linalg::CopyOp copyOp) {
      if (copyOp.getOutputs().empty())
        return;
      if (auto mt = dyn_cast<MemRefType>(copyOp.getOutputs()[0].getType()))
        if (isWorkgroupMemref(mt))
          found = true;
    });
  }

  // vector.transfer_write to workgroup memory.
  if (!found) {
    op->walk([&](vector::TransferWriteOp writeOp) {
      if (auto mt = dyn_cast<MemRefType>(writeOp.getBase().getType()))
        if (isWorkgroupMemref(mt))
          found = true;
    });
  }

  // vector.store to workgroup memory.
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

/// Returns true if `op` (or any op nested inside it) loads from workgroup
/// memory.
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

  // linalg.copy reading FROM workgroup memory.
  if (!found) {
    op->walk([&](linalg::CopyOp copyOp) {
      if (copyOp.getInputs().empty())
        return;
      if (auto mt = dyn_cast<MemRefType>(copyOp.getInputs()[0].getType()))
        if (isWorkgroupMemref(mt))
          found = true;
    });
  }

  if (!found) {
    op->walk([&](vector::TransferReadOp readOp) {
      if (auto mt = dyn_cast<MemRefType>(readOp.getBase().getType()))
        if (isWorkgroupMemref(mt))
          found = true;
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
    // scf.for and scf.forall bodies itself, so a single top-level call per
    // launch suffices and correctly handles transitions both at the
    // launch-body level and inside any nested loops or tile foralls.
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
  /// write→read transitions. Recurses into scf.for and scf.forall bodies.
  ///
  /// scf.if branches are intentionally NOT recursed into — see class comment.
  ///
  /// FIX vs original: added scf.forall recursion. Thread-level foralls
  /// contain the promoted copies (nova.promote_to_workgroup → linalg.copy →
  /// workgroup memory). Without recursing into them, write→read transitions
  /// inside thread tile loops were never detected and barriers were not
  /// inserted between the tile copy and the subsequent vector reads.
  void insertBarriersInBlock(OpBuilder &builder, Block *block) {
    SmallVector<Operation *> barrierPoints;
    bool seenWorkgroupStore    = false;
    bool seenNonAtomicStore    = false;

    for (Operation &op : block->getOperations()) {
      if (isa<scf::YieldOp, gpu::TerminatorOp>(op))
        continue;

      bool hasLoads  = hasWorkgroupLoads(&op);
      bool hasStores = hasWorkgroupStores(&op);

      // Write→read transition: place barrier before the reading op so all
      // threads see completed stores before any thread proceeds to load.
      if (seenWorkgroupStore && hasLoads) {
        barrierPoints.push_back(&op);
        seenWorkgroupStore  = false;
        seenNonAtomicStore  = false;
      }

      if (hasStores) {
        seenWorkgroupStore = true;
        if (hasNonAtomicWorkgroupStores(&op))
          seenNonAtomicStore = true;
      }

      // -----------------------------------------------------------------------
      // Recurse into scf.for bodies when loop bounds are compile-time
      // constants (uniform across all threads).
      //
      // Dynamic bounds mean different threads may execute different iteration
      // counts. Placing a barrier inside such a loop causes deadlock because
      // some threads exit earlier and never reach the barrier.
      // -----------------------------------------------------------------------
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

      // -----------------------------------------------------------------------
      // Recurse into scf.forall bodies (thread-level tile loops).
      //
      // FIX vs original: the original did not recurse into scf.forall at all.
      // Thread-level foralls are the primary location of promoted tile copies:
      //
      //   scf.forall (%t) in (32, 32) {               ← thread forall
      //     linalg.copy ins(global) outs(workgroup)    ← tile promoted here
      //   }
      //   vector.transfer_read workgroup[...]          ← must be fenced
      //
      // By recursing here, write→read transitions INSIDE the thread forall
      // (e.g. a copy followed by an immediate vector read in the same forall)
      // are also correctly fenced.
      //
      // The same uniform-bounds guard is applied: if the forall trip counts
      // are not compile-time constants (thread-id-derived), we treat the
      // forall as opaque and rely on the outer write→read detection.
      // -----------------------------------------------------------------------
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        auto isConstIdx = [](Value v) -> bool {
          if (auto cst = v.getDefiningOp<arith::ConstantOp>())
            return isa<IntegerAttr>(cst.getValue());
          if (v.getDefiningOp<arith::ConstantIndexOp>())
            return true;
          return false;
        };
        // Check all upper bounds are constants (lower bounds are always 0,
        // steps are always 1 for forall in our lowering).
        bool uniformBounds = true;
        for (Value ub : forallOp.getUpperBound(builder))
          if (!isConstIdx(ub)) { uniformBounds = false; break; }
        if (uniformBounds) {
          for (Block &innerBlock : forallOp.getRegion())
            insertBarriersInBlock(builder, &innerBlock);
        }
      }
    }

    // Tail barrier: protect the next loop iteration from reading shared memory
    // before this iteration's non-atomic stores are globally visible.
    // Only needed inside a loop (scf.yield terminator). At gpu.launch level
    // (gpu.terminator) there is no next iteration, so no tail barrier needed.
    //
    // NOTE: seenNonAtomicStore is now also set by linalg.copy (fix #1), so
    // tail barriers are correctly emitted for tile-copy loops.
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