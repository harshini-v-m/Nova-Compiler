//===- NovaGPUMultiBuffering.cpp - Multi-buffer workgroup memory ---------===//
//
// Port of IREE's GPUMultiBufferingPass, adapted to Nova's promoted-operand
// IR shape where the workgroup alloc is only reached indirectly via an outer
// `memref.subview` that sits *between* the block-forall and the K-loop.
//
// Purpose
// ───────
// Enables software pipelining of shared-memory copies for matmul K-loops.
// For every workgroup-memory `memref.alloc` whose uses all land inside one
// `scf.for`, rewrite
//     %buf = memref.alloc() : memref<TxSxf32, #gpu.address_space<workgroup>>
// into
//     %buf = memref.alloc() : memref<N x TxSxf32, #gpu.address_space<workgroup>>
// and pick `%buf[iv mod N, …]` per iteration.
//
// Bug-fix vs IREE / earlier Nova port
// ───────────────────────────────────
// 1. Per-kernel decision: the previous port counted workgroup allocs across
//    the whole function. We now collect outer block-mapped scf.forall ops
//    (one per kernel) and process each kernel independently. The previous
//    gate that capped each kernel at ≤ 2 allocs has been removed — fused
//    matmul kernels (matmul + bias + residual) promote 3+ operands and now
//    get multi-buffered.
// 2. Pipelining gate: NovaGPUPipelining used to run on every K-loop with
//    nvgpu.device_async_copy ops, even on loops where multiBuffer() had
//    failed. That raced the SMEM (a non-widened buffer is reused immediately,
//    so pipelining's depth-1 time-shift makes producer i+2 stomp consumer i).
//    We tag every successfully multi-buffered scf.for with
//    `kNovaMultiBufferedLoopMarker`; pipelining gates on it. The marker is
//    ALL-OR-NOTHING: a loop is only marked if EVERY workgroup alloc inside
//    it was widened. Partial widening leaves the loop unmarked, so
//    pipelining (and the silent-WAR-race miscompile) is skipped.
// 3. Nova-specific widener: upstream `memref::multiBuffer()` rejects 1-D
//    allocs and allocs whose leading dim equals the K-loop step (very
//    common in our matmul/gather kernels). It also rejects allocs whose
//    only users are `vector.transfer_*` or `nvgpu.device_async_copy`. Our
//    `novaMultiBuffer` handles all of these by unconditionally prepending
//    a slot dim and rewriting every direct user (subview / load / store /
//    vector transfer / cp.async) to thread the slot index through.
// 4. Per-kernel SMEM-budget pre-flight: before widening any alloc, sum
//    the post-widening bytes for all workgroup allocs in the kernel. If
//    that total would exceed the device's per-block opt-in cap (99 KB,
//    portable across sm_86/sm_89/sm_90), we skip the entire kernel so
//    none of its allocs are widened and the loop is not marked. Without
//    this gate, kernels with large promoted operands (e.g. the GeLU-
//    backward fused matmul whose 3-buffered SMEM is 113 KB) would launch
//    with `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, …)` failing
//    with CUDA_ERROR_INVALID_VALUE, and every subsequent launch would
//    fail in the runtime.
//
// Run order
// ─────────
//   After  : bufferization, memory-space inference, shared-alloc materialisation,
//            coalesce-workgroup-buffers.
//   Before : NovaGPUMapForallToGPU (the marker survives forall-to-launch),
//            NovaConvertSharedMemAllocs, NovaGPUPipelining.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-multi-buffering"

using namespace mlir;

namespace mlir::nova {

namespace {

/// True if the memref type lives in GPU workgroup (shared) memory. Matches
/// `NovaConvertSharedMemAllocs::hasSharedMemoryAddressSpace`.
static bool hasSharedMemoryAddressSpace(MemRefType t) {
  auto attr = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return attr &&
         attr.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Try to sink a subview into the unique `scf.for` that contains all of its
/// transitive uses. Returns the scf.for if sinking happened (or if the subview
/// was already inside exactly one scf.for), nullptr otherwise.
///
/// Safe to move because `%alloc` is hoisted to function entry and dominates
/// any point inside any descendant scf.for.
static scf::ForOp sinkSubviewIntoCommonFor(memref::SubViewOp subview) {
  if (auto parentFor = subview->getParentOfType<scf::ForOp>())
    return parentFor;

  // Every non-dealloc user must share a single scf.for ancestor.
  scf::ForOp commonFor;
  for (Operation *user : subview->getUsers()) {
    if (isa<memref::DeallocOp>(user))
      continue;
    auto userFor = user->getParentOfType<scf::ForOp>();
    if (!userFor)
      return nullptr;
    if (!commonFor) {
      commonFor = userFor;
      continue;
    }
    if (commonFor == userFor)
      continue;
    if (commonFor->isAncestor(userFor))
      continue;
    if (userFor->isAncestor(commonFor)) {
      commonFor = userFor;
      continue;
    }
    return nullptr;
  }
  if (!commonFor)
    return nullptr;

  // Move the subview to the top of the for body. The alloc is at function
  // entry → trivially dominates. For static-offset subviews (the Nova
  // promoted-operand path always uses [0, 0]) there are no other operands.
  subview->moveBefore(&commonFor.getBody()->front());
  return commonFor;
}

/// Returns the scf.for that would become the multiBuffer candidate loop, or
/// nullptr if the alloc isn't eligible. `alloc` must already be hoisted to
/// function entry.
static scf::ForOp pickCandidateFor(memref::AllocOp alloc) {
  scf::ForOp candidate;
  bool hasNonDealloc = false;
  for (Operation *user : alloc->getUsers()) {
    if (isa<memref::DeallocOp>(user))
      continue;
    hasNonDealloc = true;
    auto forOp = user->getParentOfType<scf::ForOp>();
    if (!forOp)
      return nullptr;
    if (!candidate)
      candidate = forOp;
    else if (candidate != forOp) {
      if (candidate->isAncestor(forOp))
        continue;
      if (forOp->isAncestor(candidate)) {
        candidate = forOp;
        continue;
      }
      return nullptr;
    }
  }
  if (!hasNonDealloc)
    return nullptr;
  return candidate;
}

//===----------------------------------------------------------------------===//
// novaMultiBuffer – Nova-specific buffer widener
//===----------------------------------------------------------------------===//

/// Build `((iv - lb) floordiv step) mod numBuffers` as an affine.apply op.
///
/// We index by ITERATION NUMBER, not by induction-variable value. Otherwise,
/// when `step % numBuffers == 0` (e.g. step=4, numBuffers=2), `(iv - lb) mod N`
/// is a constant — every iteration maps to the same slot, the buffer is
/// effectively single-slot, and the pipeliner's time-shift creates a
/// producer/consumer WAR race on that single slot.
///
/// Iteration n has iv = lb + n * step, so (iv - lb) / step = n.
/// Then slot = n mod numBuffers selects a fresh slot every iteration as
/// long as iteration count > numBuffers.
///
/// Using `iv - lb` rather than bare `iv` keeps the modulo correct for K-loops
/// whose lower bound is non-zero (e.g. tiled loops after fusion where lb is a
/// block-id expression). If lb == 0 the subtraction folds away.
static Value computeSlotIndex(OpBuilder &b, Location loc, Value iv, Value lb,
                               Value step, unsigned numBuffers) {
  AffineExpr d0, d1, d2;
  bindDims(b.getContext(), d0, d1, d2);
  // ((d0 - d1) floordiv d2) mod N
  auto map = AffineMap::get(
      /*dimCount=*/3, /*symCount=*/0,
      ((d0 - d1).floorDiv(d2)) % static_cast<int64_t>(numBuffers),
      b.getContext());
  return b.create<affine::AffineApplyOp>(loc, map, ValueRange{iv, lb, step});
}

/// Nova-specific replacement for upstream `memref::multiBuffer`.
///
/// Motivation
/// ──────────
/// Upstream's multiBuffer() rejects two common Nova IR shapes:
///   1. 1-D allocs  (`memref<N x f32, #wg>`) — upstream requires rank ≥ 2.
///   2. Square tiles where K-step == innermost dim — upstream's dimension
///      analysis is ambiguous and bails out.
///
/// This function sidesteps both by never inspecting existing dimensions at all.
/// Instead it unconditionally prepends a new slot dimension at position 0 and
/// rewrites every user to index it with `(iv - lb) mod numBuffers`. SubViewOp
/// users get the slot dim rank-reduced away so downstream consumers see no
/// type change; direct LoadOp/StoreOp users get `slotIdx` prepended to their
/// index lists.
///
/// Pre-conditions (enforced by the caller)
/// ────────────────────────────────────────
///   • DeallocOps for `allocOp` have already been erased.
///   • All SubViewOp direct users have been sunk inside `kLoop` by
///     sinkSubviewIntoCommonFor.
///   • All remaining users of `allocOp` are SubViewOp, LoadOp, or StoreOp.
///
/// Post-conditions
/// ────────────────
///   • On success: `allocOp` is erased; `kLoop` may be tagged by the caller.
///   • On failure: IR is left unmodified (widened alloc is cleaned up before
///     returning).
static LogicalResult novaMultiBuffer(memref::AllocOp allocOp,
                                      unsigned numBuffers,
                                      scf::ForOp kLoop) {
  MLIRContext *ctx = allocOp.getContext();
  Location loc = allocOp.getLoc();
  auto origType = allocOp.getType();

  // ── Pre-check ─────────────────────────────────────────────────────────────
  // Bail early (before touching the IR) if we encounter a user kind we don't
  // know how to rewrite. The supported set covers every direct-alloc user we
  // observe in Nova post-bufferization IR:
  //
  //   • SubViewOp                       — indirection through tile/slot subviews
  //   • LoadOp / StoreOp                — scalar memref accesses
  //   • vector.transfer_read/write      — vectorized matmul/elementwise SMEM I/O
  //   • nvgpu.device_async_copy         — gmem→smem prologue/loop copies
  //
  // Any other op kind (e.g. memref.collapse_shape, memref.reinterpret_cast on
  // the alloc directly) means the alloc isn't in a shape we can cleanly widen,
  // so we leave the IR alone for that alloc.
  for (Operation *user : allocOp->getUsers()) {
    if (!isa<memref::SubViewOp, memref::ViewOp, memref::LoadOp, memref::StoreOp,
             vector::TransferReadOp, vector::TransferWriteOp,
             nvgpu::DeviceAsyncCopyOp, nvgpu::LdMatrixOp>(user)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-multi-buf] novaMultiBuffer: unhandled user '"
                 << user->getName() << "' on " << allocOp << "\n");
      return failure();
    }
  }

  // ── 1. Widened alloc type ─────────────────────────────────────────────────
  // Prepend the slot count as dimension 0.  The address space and element
  // type are preserved; the layout is left default (row-major contiguous)
  // which is correct for freshly-allocated workgroup memory.
  SmallVector<int64_t> widenedShape{static_cast<int64_t>(numBuffers)};
  widenedShape.append(origType.getShape().begin(), origType.getShape().end());
  auto widenedType =
      MemRefType::get(widenedShape, origType.getElementType(),
                      MemRefLayoutAttrInterface{}, origType.getMemorySpace());

  // ── 2. Create the widened alloc ───────────────────────────────────────────
  OpBuilder b(allocOp);
  auto widenedAlloc = b.create<memref::AllocOp>(loc, widenedType);

  // ── 3. Slot index: ((iv - lb) floordiv step) mod numBuffers ──────────────
  // Inserted at the very top of the K-loop body so it dominates every use
  // of the alloc inside the loop. Indexed by iteration number, NOT by raw
  // iv value — see computeSlotIndex docstring.
  b.setInsertionPointToStart(kLoop.getBody());
  Value slotIdx = computeSlotIndex(b, loc, kLoop.getInductionVar(),
                                    kLoop.getLowerBound(), kLoop.getStep(),
                                    numBuffers);

  // ── 4. Rewrite SubViewOp users ────────────────────────────────────────────
  // Collect into a local vector first to avoid iterator invalidation while
  // erasing subviews.
  SmallVector<memref::SubViewOp> subviews;
  SmallVector<memref::ViewOp> views;
  for (Operation *user : allocOp->getUsers()) {
    if (auto sv = dyn_cast<memref::SubViewOp>(user))
      subviews.push_back(sv);
    if (auto v = dyn_cast<memref::ViewOp>(user))
      views.push_back(v);
  }

  for (memref::SubViewOp sv : subviews) {
    b.setInsertionPoint(sv);

    // Build the new offset/size/stride lists by prepending the slot dimension:
    //   offset[0] = slotIdx   – dynamic, selects the active buffer slot
    //   size[0]   = 1         – we access exactly one slot per iteration
    //   stride[0] = 1         – slots are laid out contiguously
    SmallVector<OpFoldResult> offsets, sizes, strides;
    offsets.push_back(slotIdx); // Value is implicitly OpFoldResult
    sizes.push_back(b.getIndexAttr(1));
    strides.push_back(b.getIndexAttr(1));

    auto origOffsets = sv.getMixedOffsets();
    auto origSizes   = sv.getMixedSizes();
    auto origStrides = sv.getMixedStrides();
    offsets.append(origOffsets.begin(), origOffsets.end());
    sizes.append(origSizes.begin(),     origSizes.end());
    strides.append(origStrides.begin(), origStrides.end());

    // Infer the rank-reduced result type.  Because size[0] == 1 the slot
    // dimension is dropped, so the result shape is identical to the original
    // subview's result shape — all downstream users remain type-correct.
    auto resultType = cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType(
            sv.getType().getShape(), widenedType, offsets, sizes, strides));

    auto newSv = b.create<memref::SubViewOp>(sv.getLoc(), resultType,
                                              widenedAlloc, offsets, sizes,
                                              strides);
    sv.replaceAllUsesWith(newSv.getResult());
    sv.erase();
  }
  for (auto v : views) {
    b.setInsertionPoint(v);
    // Create a subview of the widened buffer to isolate the current slot.
    // offsets=[slotIdx, 0], sizes=[1, -1], strides=[1, 1]
    SmallVector<OpFoldResult> offsets{slotIdx, b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{b.getIndexAttr(1),
                                    b.getIndexAttr(ShapedType::kDynamic)};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1), b.getIndexAttr(1)};

    auto subviewType = memref::SubViewOp::inferRankReducedResultType(
        v.getSource().getType().getShape(), widenedAlloc.getType(), offsets,
        sizes, strides);

    auto sub = b.create<memref::SubViewOp>(v.getLoc(), cast<MemRefType>(subviewType),
                                           widenedAlloc, offsets, sizes,
                                           strides);
    auto newView = b.create<memref::ViewOp>(v.getLoc(), v.getType(), sub,
                                            v.getByteShift(), v.getSizes());
    v.replaceAllUsesWith(newView.getResult());
    v.erase();
  }

  // ── 5. Rewrite remaining direct users ────────────────────────────────────
  //
  // After step 4, only non-subview users of the original alloc remain. We
  // handle each kind by prepending `slotIdx` to its index list. The widened
  // memref's leading dim is the slot dim; the remainder of the indexing
  // arithmetic is unchanged because we widened by *prepending* the slot dim
  // (so original indices map to the same elements within a single slot).
  //
  // For ops that produce a result (load, transfer_read, async_copy) we have
  // to materialize a new op — we can't just patch operands of the old op
  // because the source memref type changed. For the void-result store/write
  // we still create a new op and erase the old one for symmetry.
  //
  // Iteration order: cp.async writes the buffer for iter (i+depth-1) and
  // vector reads the buffer for iter i; both happen in the same loop body.
  // Walking direct users is safe because we only replace direct alloc edges
  // (the alloc's SSA value is the source operand in every case below).
  SmallVector<Operation *> directUsers(allocOp->getUsers().begin(),
                                        allocOp->getUsers().end());
  for (Operation *user : directUsers) {
    b.setInsertionPoint(user);
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      SmallVector<Value> indices{slotIdx};
      indices.append(load.getIndices().begin(), load.getIndices().end());
      auto newLoad =
          b.create<memref::LoadOp>(load.getLoc(), widenedAlloc, indices);
      load.replaceAllUsesWith(newLoad.getResult());
      load.erase();
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      SmallVector<Value> indices{slotIdx};
      indices.append(store.getIndices().begin(), store.getIndices().end());
      b.create<memref::StoreOp>(store.getLoc(), store.getValue(),
                                 widenedAlloc, indices);
      store.erase();
      continue;
    }
    if (auto vread = dyn_cast<vector::TransferReadOp>(user)) {
      // vector.transfer_read on workgroup memory in matmul kernels — clone
      // with slot prepended to indices and the source replaced.
      //
      // The permutation map's input rank must equal the source rank. Since
      // we widened the source by prepending one dim (the slot dim, accessed
      // at exactly slotIdx with size 1), the new map is the original map
      // composed with a "shift-all-inputs-by-1" function — i.e. drop the
      // new leading input dim and use the rest unchanged. The result dims
      // are unchanged because the slot dim is not read into the vector.
      //
      // Concretely: original map (d0, d1, …, dN-1) -> results becomes
      // (d0_new, d1_new, …, dN_new) -> results' where d_i_new corresponds
      // to d_(i-1) in the old map. We construct this by shifting each
      // existing result expression's dim positions up by 1.
      SmallVector<Value> indices{slotIdx};
      indices.append(vread.getIndices().begin(), vread.getIndices().end());

      AffineMap oldMap = vread.getPermutationMap();
      AffineMap newMap = oldMap.shiftDims(/*shift=*/1);

      // The "in_bounds" attribute is per-result-dim, not per-input-dim, so it
      // doesn't change when we add an input dim that maps to no result.
      auto newRead = b.create<vector::TransferReadOp>(
          vread.getLoc(), vread.getVectorType(), widenedAlloc, indices,
          AffineMapAttr::get(newMap), vread.getPadding(), vread.getMask(),
          vread.getInBoundsAttr());
      vread.replaceAllUsesWith(newRead.getResult());
      vread.erase();
      continue;
    }
    if (auto vwrite = dyn_cast<vector::TransferWriteOp>(user)) {
      SmallVector<Value> indices{slotIdx};
      indices.append(vwrite.getIndices().begin(), vwrite.getIndices().end());

      AffineMap oldMap = vwrite.getPermutationMap();
      AffineMap newMap = oldMap.shiftDims(/*shift=*/1);

      b.create<vector::TransferWriteOp>(
          vwrite.getLoc(), vwrite.getVector(), widenedAlloc, indices,
          AffineMapAttr::get(newMap), vwrite.getMask(),
          vwrite.getInBoundsAttr());
      vwrite.erase();
      continue;
    }
    if (auto cp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(user)) {
      // The alloc may appear as either the SRC or the DST of cp.async.
      // In Nova's gmem→smem promotion path it's always the DST, but rewrite
      // both sides defensively so this remains correct if the IR shape
      // changes upstream.
      bool isDst = (cp.getDst() == allocOp.getResult());
      bool isSrc = (cp.getSrc() == allocOp.getResult());
      SmallVector<Value> dstIdx;
      SmallVector<Value> srcIdx;
      if (isDst) {
        dstIdx.push_back(slotIdx);
        dstIdx.append(cp.getDstIndices().begin(), cp.getDstIndices().end());
      } else {
        dstIdx.append(cp.getDstIndices().begin(), cp.getDstIndices().end());
      }
      if (isSrc) {
        srcIdx.push_back(slotIdx);
        srcIdx.append(cp.getSrcIndices().begin(), cp.getSrcIndices().end());
      } else {
        srcIdx.append(cp.getSrcIndices().begin(), cp.getSrcIndices().end());
      }
      Value newDst = isDst ? widenedAlloc.getResult() : cp.getDst();
      Value newSrc = isSrc ? widenedAlloc.getResult() : cp.getSrc();
      auto newCp = b.create<nvgpu::DeviceAsyncCopyOp>(
          cp.getLoc(), cp.getResult().getType(), newDst, dstIdx, newSrc,
          srcIdx, cp.getDstElementsAttr(), cp.getSrcElements(),
          cp.getBypassL1Attr());
      cp.replaceAllUsesWith(newCp.getResult());
      cp.erase();
      continue;
    }
    if (auto ldm = dyn_cast<nvgpu::LdMatrixOp>(user)) {
      SmallVector<Value> indices{slotIdx};
      indices.append(ldm.getIndices().begin(), ldm.getIndices().end());
      auto newLdm = b.create<nvgpu::LdMatrixOp>(
          ldm.getLoc(), ldm.getRes().getType(), widenedAlloc, indices,
          ldm.getTranspose(), ldm.getNumTiles());
      ldm.replaceAllUsesWith(newLdm.getRes());
      ldm.erase();
      continue;
    }
    // No else-branch: the pre-check above guarantees no other kinds reach
    // this loop. If a new user kind appears post-pre-check (e.g. inserted by
    // a parallel pass) we'll silently skip it — but the alloc's SSA value
    // would still have a use of the old type, so verifier would catch it.
  }

  allocOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// NovaGPUMultiBufferingPass
//===----------------------------------------------------------------------===//
struct NovaGPUMultiBufferingPass
    : public PassWrapper<NovaGPUMultiBufferingPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUMultiBufferingPass)

  NovaGPUMultiBufferingPass() = default;
  NovaGPUMultiBufferingPass(const NovaGPUMultiBufferingPass &o)
      : PassWrapper(o), numBuffers(o.numBuffers) {}
  explicit NovaGPUMultiBufferingPass(unsigned n) : numBuffers(n) {}

  unsigned numBuffers = 3;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    scf::SCFDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.getFunctionBody().empty())
      return;

    // ── Step 1: collect outer block-mapped scf.forall ops (each is one ────
    // GPU kernel after MapForallToGPU). Multi-buffering decisions are made
    // per outer forall: a function may contain several matmul kernels (e.g.
    // the forward and backward), each with its own A/B SMEM allocs, and we
    // don't want one kernel's alloc count to disqualify another's.
    SmallVector<scf::ForallOp> outerForalls;
    funcOp.walk([&](scf::ForallOp forall) {
      if (forall->getParentOfType<scf::ForallOp>())
        return; // Only outermost foralls.
      auto mapping = forall.getMappingAttr();
      if (!mapping || mapping.empty())
        return;
      if (isa<gpu::GPUBlockMappingAttr>(mapping.getValue().front()))
        outerForalls.push_back(forall);
    });

    OpBuilder builder(funcOp.getContext());
    UnitAttr marker = builder.getUnitAttr();
    unsigned totalRewritten = 0;
    unsigned totalEligible = 0;

    // ── Step 2: per-kernel multi-buffer pass ──────────────────────────────
    for (scf::ForallOp kernel : outerForalls) {
      SmallVector<memref::AllocOp> workgroupAllocs;
      kernel.walk([&](memref::AllocOp allocOp) {
        if (hasSharedMemoryAddressSpace(allocOp.getType()))
          workgroupAllocs.push_back(allocOp);
      });

      // No alloc-count gate: fused matmul kernels (matmul + bias + residual,
      // attention QKV fused with output projection, etc.) promote 3+ operands
      // to SMEM and we want to multi-buffer all of them. The all-or-nothing
      // marker rule below protects correctness: if any one alloc fails to
      // widen, the loop is left un-marked and pipelining skips it.
      if (workgroupAllocs.empty())
        continue;

      // ── Per-kernel SMEM budget pre-flight ──────────────────────────────
      //
      // After widening, the kernel's total workgroup-memory bytes must fit
      // the device's per-block opt-in cap (`cudaDevAttrMaxSharedMemory-
      // PerBlockOptin`):  100 KB on sm_86, 99 KB on sm_89, 228 KB on sm_90.
      // We use 99 KB (101376 B) as a portable cap — it works on every Ampere
      // and Ada device.
      //
      // `maybeMaterializeDynamicSharedMemory` (in NovaGPUMapForallToGPU)
      // sums the alloc bytes and emits that as the launch op's
      // `dynamic_shared_memory_size`. If the total exceeds the cap, the
      // runtime call `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, …)`
      // fails with CUDA_ERROR_INVALID_VALUE and every launch fails.
      //
      // We pre-flight here: sum each alloc's widened byte count
      // (`numBuffers × elements × elemBytes`, rounded up to 16 B for
      // cp.async.cg alignment, matching the downstream pass's calculation).
      // If the kernel exceeds the cap, we skip multi-buffering for the
      // ENTIRE kernel — none of its allocs are widened, and no scf.for
      // gets the marker. Pipelining skips it; correctness preserved, just
      // no overlap.
      constexpr int64_t kSMEMCapBytes = 99 * 1024; // 101376 B
      int64_t widenedBytesTotal = 0;
      for (memref::AllocOp alloc : workgroupAllocs) {
        MemRefType ty = alloc.getType();
        int64_t elemBits = ty.getElementTypeBitWidth();
        int64_t elems = ty.getNumElements();
        int64_t bytes = (elems * elemBits + 7) / 8;
        bytes = llvm::alignTo(bytes, (int64_t)16);
        // After widening this alloc by numBuffers, it occupies numBuffers ×
        // bytes. Round-up alignment is preserved because numBuffers × 16 = 16k.
        widenedBytesTotal +=
            static_cast<int64_t>(numBuffers) * bytes;
      }
      if (widenedBytesTotal > kSMEMCapBytes) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-multi-buf] skip kernel: widened SMEM total "
                   << widenedBytesTotal << " B (> " << kSMEMCapBytes
                   << " B device cap)\n");
        continue;
      }

      totalEligible += workgroupAllocs.size();

      // ── Step 2a: sink direct SubViewOp users into their common scf.for ──
      for (memref::AllocOp allocOp : workgroupAllocs) {
        SmallVector<memref::SubViewOp> directSubviews;
        for (Operation *user : allocOp->getUsers())
          if (auto sv = dyn_cast<memref::SubViewOp>(user))
            directSubviews.push_back(sv);
        for (memref::SubViewOp sv : directSubviews)
          (void)sinkSubviewIntoCommonFor(sv);
      }

      // ── Step 2b: multi-buffer eligible allocs ───────────────────────────
      //
      // CRITICAL: a loop with N workgroup allocs is only safe to pipeline if
      // ALL N were successfully widened. If any alloc stays single-slot while
      // pipelining time-shifts the cp.async into it, the producer of iter i+2
      // and the consumer of iter i race on the same SMEM slot — the consumer
      // sees the wrong tile and gradients drift. We collect candidate loops
      // and only set the multi-buffer marker after EVERY alloc succeeds.
      //
      // Per-loop bookkeeping: count how many allocs target each scf.for, and
      // tally how many of those succeeded. The marker goes on a loop only
      // when (succeeded == total).
      llvm::SmallDenseMap<scf::ForOp, std::pair<unsigned, unsigned>>
          loopAllocCounts; // forOp → (total, succeeded)

      for (memref::AllocOp allocOp : workgroupAllocs) {
        scf::ForOp candidate = pickCandidateFor(allocOp);
        if (!candidate) {
          LLVM_DEBUG(llvm::dbgs() << "[nova-multi-buf] skip (no single for): "
                                  << allocOp << "\n");
          continue;
        }
        loopAllocCounts[candidate].first++;

        SmallVector<memref::DeallocOp> deallocs;
        for (Operation *user : allocOp->getUsers())
          if (auto d = dyn_cast<memref::DeallocOp>(user))
            deallocs.push_back(d);
        for (memref::DeallocOp d : deallocs)
          d.erase();

        if (failed(novaMultiBuffer(allocOp, numBuffers, candidate))) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[nova-multi-buf] novaMultiBuffer() failed on: "
                     << allocOp << "\n");
          continue; // total++ but succeeded stays — loop will not get the marker.
        }

        loopAllocCounts[candidate].second++;
        ++totalRewritten;
      }

      // Only mark loops where every alloc was widened. Unmarked loops stay
      // un-pipelined — the safe fallback that produces numerically-correct
      // (though un-overlapped) code.
      for (auto &[forOp, counts] : loopAllocCounts) {
        if (counts.first == counts.second && counts.first > 0)
          forOp->setAttr(kNovaMultiBufferedLoopMarker, marker);
        else
          LLVM_DEBUG(llvm::dbgs()
                     << "[nova-multi-buf] loop NOT marked: " << counts.second
                     << "/" << counts.first << " allocs widened\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[nova-multi-buf] rewrote " << totalRewritten
                            << " / " << totalEligible
                            << " workgroup allocs across " << outerForalls.size()
                            << " kernels\n");
  }

  StringRef getArgument() const override {
    return "nova-gpu-multi-buffering";
  }
  StringRef getDescription() const override {
    return "Multi-buffers workgroup-memory allocs used inside scf.for to "
           "enable cp.async software pipelining. Tags the rewritten scf.for "
           "with a marker so NovaGPUPipelining only time-shifts safely-widened "
           "loops.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUMultiBufferingPass(unsigned numBuffers) {
  return std::make_unique<NovaGPUMultiBufferingPass>(numBuffers);
}

void registerNovaGPUMultiBufferingPass() {
  PassRegistration<NovaGPUMultiBufferingPass>();
}

} // namespace mlir::nova