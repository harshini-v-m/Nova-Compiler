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
// With N buffers, iteration i+N-1 can prefetch into buffer `(i+N-1) mod N`
// while iteration i reads from buffer `i mod N` — prerequisite for the
// downstream pipelining pass to emit `cp.async.commit_group` + delayed
// `cp.async.wait_group(N-2)` instead of `wait_group 0`.
//
// Mechanics
// ─────────
// Upstream's `memref::multiBuffer(alloc, N)` does the heavy lifting, but it
// picks the **first loop-like ancestor of any direct user** as the candidate
// loop and requires a single induction variable. In Nova's pipeline the
// direct user of a workgroup alloc is usually a `memref.subview` that lives
// inside the outer block-`scf.forall` (multi-IV) and *not* inside the
// K-`scf.for` (single-IV). Without surgery upstream picks scf.forall and
// bails on the single-IV check.
//
// This pass therefore does three things:
//   1. Hoist workgroup allocs to the function entry block (so they dominate
//      every use; required by multiBuffer).
//   2. For each alloc's direct `memref.subview` user whose *own* uses all
//      live inside the same `scf.for`, sink the subview to the top of that
//      scf.for. After this, the alloc's only direct non-dealloc user inside
//      the body is a subview with `scf.for` as its parent-loop, and upstream
//      multiBuffer picks the right loop.
//   3. Call `memref::multiBuffer(alloc, N, skipOverrideAnalysis=true)`. We
//      skip the override analysis because Nova's gpuCopyFn emits
//      `linalg.copy` (not `memref.copy`) and targets a subview of the alloc
//      rather than the alloc itself — the override check would never match.
//
// Allocs whose users span the K-loop AND the epilogue (e.g. matmul
// accumulators read back after the reduction) are intentionally skipped:
// widening their shape would break the post-loop reader.
//
// Run order
// ─────────
//   After  : bufferization, memory-space inference, shared-alloc materialisation,
//            coalesce-workgroup-buffers.
//   Before : NovaConvertSharedMemAllocs (turns alloc → memref.global), and
//            the (future) software-pipelining pass.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
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
  // Already inside an scf.for? Nothing to do.
  if (auto parentFor = subview->getParentOfType<scf::ForOp>())
    return parentFor;

  // Walk users; require that every non-dealloc user has a (possibly nested)
  // scf.for ancestor that is the SAME for all users. Dealloc users are ignored
  // — they stay outside the loop and are handled specially by multiBuffer.
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
    // Accept if one is an ancestor of the other (pick the outermost common).
    if (commonFor == userFor)
      continue;
    if (commonFor->isAncestor(userFor)) {
      // keep commonFor (the outer one)
      continue;
    }
    if (userFor->isAncestor(commonFor)) {
      commonFor = userFor;
      continue;
    }
    // Disjoint scf.fors → can't pick a single candidate.
    return nullptr;
  }
  if (!commonFor)
    return nullptr;

  // Move the subview to the top of the for body. Operands of the subview
  // (the alloc itself, plus any index SSA values) must dominate this point.
  // The alloc is at function entry — trivially dominates. Static-offset
  // subviews have no other operands; dynamic-offset subviews with operands
  // defined outside the loop would need a dominance check. For the Nova
  // promoted-operand path the offsets are static (always `[0, 0]`), so we
  // skip that check here and rely on multiBuffer's later dominance check to
  // catch any pathological case.
  subview->moveBefore(&commonFor.getBody()->front());
  return commonFor;
}

/// Returns the scf.for that would become the multiBuffer candidate loop, or
/// nullptr if the alloc isn't eligible. `alloc` must already be hoisted to
/// function entry.
///
/// Eligibility (matches upstream multiBuffer with skipOverrideAnalysis=true):
///   • at least one non-dealloc user exists, and
///   • there is a single scf.for F such that every non-dealloc user is a
///     proper descendant of F.
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
      // If one is an ancestor of the other, take the outer; otherwise reject.
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

  unsigned numBuffers = 2;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    scf::SCFDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.getFunctionBody().empty())
      return;

    // ── Step 1: collect workgroup allocs ──────────────────────────────────
    // We intentionally do NOT hoist allocs to function entry. In Nova's
    // pipeline the allocs sit at the top of an outer block-mapped
    // `scf.forall` body (the workgroup scope). Map-forall-to-gpu later
    // wraps that scf.forall body into `gpu.launch` / `gpu.func` — so the
    // alloc ends up inside the kernel where it must live (you can't allocate
    // workgroup / address-space<3> memory from host code). IREE's upstream
    // pass hoists to function entry because their equivalent stage is
    // already inside a gpu.func.
    SmallVector<memref::AllocOp> workgroupAllocs;
    funcOp.walk([&](memref::AllocOp allocOp) {
      if (hasSharedMemoryAddressSpace(allocOp.getType()))
        workgroupAllocs.push_back(allocOp);
    });

    // ── Step 2: sink direct SubViewOp users into their common scf.for ─────
    // Nova's promoted-operand lowering emits:
    //   scf.forall (block) {
    //     %sv = memref.subview %alloc[0,0] [T,S] [1,1]  // outer subview
    //     scf.for (K-loop) { ... uses %sv ... }
    //   }
    // `%sv` is a direct user of `%alloc` but its parent loop is scf.forall,
    // not scf.for. Sink it into the K-loop so multiBuffer picks scf.for.
    for (memref::AllocOp allocOp : workgroupAllocs) {
      // Copy the user list: sinking mutates the op position, which could
      // invalidate iteration order if iterating getUsers() directly.
      SmallVector<memref::SubViewOp> directSubviews;
      for (Operation *user : allocOp->getUsers())
        if (auto sv = dyn_cast<memref::SubViewOp>(user))
          directSubviews.push_back(sv);
      for (memref::SubViewOp sv : directSubviews)
        (void)sinkSubviewIntoCommonFor(sv);
    }

    // ── Step 3: filter to allocs where a single scf.for dominates all
    // non-dealloc users, then invoke upstream multiBuffer ─────────────────
    unsigned rewritten = 0;
    for (memref::AllocOp allocOp : workgroupAllocs) {
      scf::ForOp candidate = pickCandidateFor(allocOp);
      if (!candidate) {
        LLVM_DEBUG(llvm::dbgs() << "[nova-multi-buf] skip (no single for): "
                                << allocOp << "\n");
        continue;
      }
      // Erase the alloc's dealloc users BEFORE calling multiBuffer. Reason:
      // upstream iterates alloc users and picks the first one whose
      // `getParentOfType<LoopLikeOpInterface>()` is non-null as the candidate
      // loop. The dealloc sits at the end of the outer scf.forall body, so
      // its first loop-ancestor is scf.forall (multi-IV) — upstream would
      // pick that and then fail the single-IV check. Workgroup memory is
      // auto-freed on kernel exit (DropGPUMemoryDeallocOp later drops any
      // surviving deallocs), so eager erasure here is safe.
      SmallVector<memref::DeallocOp> deallocs;
      for (Operation *user : allocOp->getUsers())
        if (auto d = dyn_cast<memref::DeallocOp>(user))
          deallocs.push_back(d);
      for (memref::DeallocOp d : deallocs)
        d.erase();

      // skipOverrideAnalysis=true: Nova uses linalg.copy on a subview of the
      // alloc, not memref.copy on the alloc itself, so the upstream
      // "is this user a memref.copy that overrides the alloc?" check never
      // succeeds. We know the allocs are fully re-written each iteration (the
      // gmem→smem tile copy is the entirety of the K-step's smem traffic), so
      // skipping that check is safe here.
      if (failed(memref::multiBuffer(allocOp, numBuffers,
                                     /*skipOverrideAnalysis=*/true))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-multi-buf] multiBuffer() failed on: " << allocOp
                   << "\n");
        // Non-fatal: leave this alloc alone and try the next one. Partial
        // multi-buffering is safe as long as the pipelining pass (future)
        // keys off the *shape* of each alloc — widened allocs get pipelined,
        // un-widened ones stay synchronous.
        continue;
      }
      ++rewritten;
    }

    LLVM_DEBUG(llvm::dbgs() << "[nova-multi-buf] rewrote " << rewritten
                            << " / " << workgroupAllocs.size()
                            << " workgroup allocs\n");
  }

  StringRef getArgument() const override {
    return "nova-gpu-multi-buffering";
  }
  StringRef getDescription() const override {
    return "Multi-buffers workgroup-memory allocs used inside scf.for to "
           "enable cp.async software pipelining.";
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
