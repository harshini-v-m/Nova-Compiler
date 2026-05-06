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
//    the whole function and bailed when the count exceeded 2. A function with
//    multiple matmul kernels (e.g. forward + backward) trivially has 4+ allocs
//    so that gate killed multi-buffering for ALL kernels. We now collect
//    outer block-mapped scf.forall ops (one per kernel) and apply the
//    > 2 alloc gate inside each kernel independently.
// 2. Pipelining gate: NovaGPUPipelining used to run on every K-loop with
//    nvgpu.device_async_copy ops, even on loops where multiBuffer() had
//    failed. That raced the SMEM (a non-widened buffer is reused immediately,
//    so pipelining's depth-1 time-shift makes producer i+2 stomp consumer i).
//    We tag every successfully multi-buffered scf.for with
//    `kNovaMultiBufferedLoopMarker`; pipelining gates on it.
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
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
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

      // Only multi-buffer if at most 2 promoted operands (matmul A/B). Fused
      // ops promote more inputs to SMEM and widening risks exceeding budget.
      // Note: budget enforcement is final-checked by
      // maybeMaterializeDynamicSharedMemory in NovaGPUMapForallToGPU, which
      // switches to gpu.dynamic_shared_memory when 3× widened buffers cross
      // the 48 KB static cap.
      if (workgroupAllocs.size() > 2) {
        LLVM_DEBUG(llvm::dbgs() << "[nova-multi-buf] skip kernel: "
                                << workgroupAllocs.size()
                                << " workgroup allocs > 2\n");
        continue;
      }
      if (workgroupAllocs.empty())
        continue;
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
      for (memref::AllocOp allocOp : workgroupAllocs) {
        scf::ForOp candidate = pickCandidateFor(allocOp);
        if (!candidate) {
          LLVM_DEBUG(llvm::dbgs() << "[nova-multi-buf] skip (no single for): "
                                  << allocOp << "\n");
          continue;
        }

        SmallVector<memref::DeallocOp> deallocs;
        for (Operation *user : allocOp->getUsers())
          if (auto d = dyn_cast<memref::DeallocOp>(user))
            deallocs.push_back(d);
        for (memref::DeallocOp d : deallocs)
          d.erase();

        if (failed(memref::multiBuffer(allocOp, numBuffers,
                                       /*skipOverrideAnalysis=*/true))) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[nova-multi-buf] multiBuffer() failed on: " << allocOp
                     << "\n");
          continue;
        }

        candidate->setAttr(kNovaMultiBufferedLoopMarker, marker);
        ++totalRewritten;
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
