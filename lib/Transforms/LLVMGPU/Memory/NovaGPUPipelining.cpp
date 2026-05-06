//===- NovaGPUPipelining.cpp - cp.async software pipelining --------------===//
//
// Port of IREE's GPUPipelining.cpp `loadGlobalStage0` strategy.
//
// Purpose
// ───────
// Time-shift `nvgpu.device_async_copy` ops (and their backward-slice deps)
// from "current iteration" to "iteration-N earlier", so that the gmem→smem
// copy for iteration i+N-1 is in flight while iteration i's compute runs on
// the previously-loaded buffer. The scheduling work is delegated to upstream
// `mlir::scf::pipelineForLoop`; this pass only contributes the schedule
// callbacks.
//
// Bug-fix vs IREE / earlier Nova port
// ───────────────────────────────────
// Pipelining ONLY operates on scf.for ops carrying
// `kNovaMultiBufferedLoopMarker`, which NovaGPUMultiBuffering attaches to
// every loop whose workgroup buffers it successfully widened. Without this
// gate, an scf.for with `nvgpu.device_async_copy` but un-widened buffers
// would be time-shifted, causing successive cp.async groups to alias the
// same SMEM region (data race).
//
// Bug-fix vs original port (NVVM dialect mismatch)
// ─────────────────────────────────────────────────
// The input IR mixes NVGPU dialect ops with raw NVVM intrinsics:
//
//   nvgpu.device_async_create_group  ↔  nvvm.cp.async.commit.group
//   nvgpu.device_async_wait          ↔  nvvm.cp.async.wait.group
//   gpu.barrier / nvgpu.barrier      ↔  nvvm.barrier0
//
// The original pass only handled the NVGPU dialect forms, so:
//   1. nvvm.cp.async.commit.group was never marked as a first-stage op,
//      causing it to land in Stage 1 and be separated from the cp.async
//      ops it commits — the group was never actually committed before the
//      Stage-1 wait, so the wait drained a different (or empty) group.
//   2. nvvm.cp.async.wait.group counts were never rewritten from 0 to
//      depth-1, so every wait in the prologue and kernel body was a
//      wait-all, making the pipeline functionally depth-1 despite the 3×
//      SMEM allocation.
//   3. scf.for bodies containing nvvm.barrier0 were accepted for
//      pipelining, producing per-element barriers inside the epilogue
//      predicated region that deadlock because not all threads reach them.
//
// The PTX confirms all three: every cp.async.wait_group in the output
// assembly is 0, and the two consecutive bar.sync instructions in the
// kernel body are the doubled-barrier signature.
//
// Pre-conditions
// ──────────────
//   * Workgroup-memory allocs are widened to <N x …, #ws> by
//     NovaGPUMultiBuffering, which sets the loop marker.
//   * The K-loop body contains nvgpu.device_async_copy +
//     commit (NVGPU or NVVM form) + wait (NVGPU or NVVM form) + barrier +
//     compute reading the smem subview.
//   * The K-loop body is a flat region (or contains scf.if, which we accept
//     because MapForallToGPU lowers thread-masked copies to scf.if).
//
// Effect
// ──────
//   * Stage 0 = cp.async ops + commit + their backward-slice deps.
//   * Stage `depth-1` = everything else (compute + wait + barrier).
//   * The wait's count is rewritten to `depth-1` in the kernel and
//     prologue, decreasing iteration-by-iteration in the epilogue.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-pipelining"

using namespace mlir;

namespace mlir::nova {

namespace {

// Markers on the loop body — adopted verbatim from IREE.
constexpr StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
constexpr StringLiteral kPipeliningFirstStage = "__pipelining_first_stage__";

static bool hasSharedMemoryAddressSpace(MemRefType t) {
  auto attr = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return attr &&
         attr.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

//===----------------------------------------------------------------------===//
// Dialect-agnostic predicate helpers.
// All barrier / commit / wait checks go through these so every code path
// handles both the NVGPU dialect form and the raw NVVM intrinsic form.
//===----------------------------------------------------------------------===//

static bool isBarrierOp(Operation *op) {
  return isa<gpu::BarrierOp, NVVM::Barrier0Op>(op);
}

static bool isCommitGroupOp(Operation *op) {
  // nvgpu.device_async_create_group with operands is a "real" commit.
  // The zero-operand form is a wait-separator; treat it the same as the
  // NVVM commit for stage-marking purposes — both must stay in Stage 0.
  if (isa<NVVM::CpAsyncCommitGroupOp>(op))
    return true;
  if (auto cg = dyn_cast<nvgpu::DeviceAsyncCreateGroupOp>(op))
    return cg.getNumOperands() > 0;
  return false;
}

static bool isWaitGroupOp(Operation *op) {
  return isa<nvgpu::DeviceAsyncWaitOp, NVVM::CpAsyncWaitGroupOp>(op);
}

//===----------------------------------------------------------------------===//
// Compute how many async groups should remain in flight after the wait.
//===----------------------------------------------------------------------===//

static int computeNumGroupsInFlight(scf::PipeliningOption::PipelinerPart part,
                                    unsigned iteration, unsigned depth) {
  if (part == scf::PipeliningOption::PipelinerPart::Kernel ||
      part == scf::PipeliningOption::PipelinerPart::Prologue)
    return static_cast<int>(depth) - 1;
  // Epilogue: one fewer group per iteration.
  return std::max(0, static_cast<int>(depth) - 2 - static_cast<int>(iteration));
}

//===----------------------------------------------------------------------===//
// Rewrite a single async-wait op (either dialect) to the correct count.
// Returns true if anything was changed.
//===----------------------------------------------------------------------===//

static bool rewriteWaitGroupCount(Operation *op, int count) {
  OpBuilder b(op);

  // NVGPU dialect: nvgpu.device_async_wait {numGroups = <n>}
  // Only rewrite if the attribute is absent (0 is the "unset" sentinel used
  // by the original pass) or if it encodes a larger count than needed.
  if (auto wait = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op)) {
    if (wait.getNumGroups())
      return false; // already set by a previous annotation pass
    wait->setAttr(wait.getNumGroupsAttrName(),
                  b.getI32IntegerAttr(count));
    return true;
  }

  // NVVM intrinsic: nvvm.cp.async.wait.group {n = <n>}
  // The attribute is named "n" and is always present (defaults to 0 which
  // means wait-all). We overwrite unconditionally because 0 is never the
  // right value in a depth>1 pipeline except in the last epilogue step,
  // and computeNumGroupsInFlight already returns 0 in that case.
  if (auto wait = dyn_cast<NVVM::CpAsyncWaitGroupOp>(op)) {
    wait->setAttr("n", b.getI32IntegerAttr(count));
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Predicated cp.async for un-peeled epilogue (port of IREE :44–108).
//===----------------------------------------------------------------------===//

static Operation *replaceOpWithPredicatedOp(RewriterBase &rewriter,
                                            Operation *op, Value pred) {
  if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
    forallOp.walk([&](nvgpu::DeviceAsyncCopyOp asyncOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(asyncOp);
      replaceOpWithPredicatedOp(rewriter, asyncOp, pred);
    });
    return forallOp;
  }

  // After MapForallToGPU, predicated copy blocks are scf.if ops. AND the
  // pipeline predicate with the existing condition so that (a) the thread mask
  // still applies and (b) out-of-bounds iterations don't issue copies into
  // stale SMEM slots.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp.getNumResults() != 0)
      return nullptr;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);
    Value combinedCond = rewriter.create<arith::AndIOp>(
        ifOp.getLoc(), pred, ifOp.getCondition());
    ifOp->setOperand(0, combinedCond);
    return ifOp;
  }

  // Plain scf.for inside the K-loop body: safe to pipeline only when it
  // contains neither cp.async (srcElements predication required) nor any
  // barrier (requires all threads to reach it — unsafe inside a loop that
  // only some threads enter).
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    bool hasAsync = false;
    bool hasBarrier = false;
    forOp.walk([&](Operation *nested) {
      if (isa<nvgpu::DeviceAsyncCopyOp>(nested))
        hasAsync = true;
      if (isBarrierOp(nested))
        hasBarrier = true;
    });
    if (hasAsync || hasBarrier)
      return nullptr;
    return op;
  }

  if (!isa<nvgpu::DeviceAsyncCopyOp>(op)) {
    if (isMemoryEffectFree(op))
      return op;
    // Barriers, commits, and waits are safe to leave unguarded in the
    // epilogue — they are either no-ops when no groups are in flight or
    // are needed to drain the last groups.
    if (isBarrierOp(op) || isCommitGroupOp(op) || isWaitGroupOp(op))
      return op;
    if (auto vload = dyn_cast<vector::LoadOp>(op)) {
      if (hasSharedMemoryAddressSpace(cast<MemRefType>(vload.getBase().getType())))
        return op;
    }
    if (auto mload = dyn_cast<memref::LoadOp>(op)) {
      if (hasSharedMemoryAddressSpace(mload.getMemRefType()))
        return op;
    }
    return nullptr;
  }

  // Predicate a nvgpu.device_async_copy by wrapping it in an scf.if.
  // Using scf.if instead of size-zeroing is safer on hardware that faults on
  // invalid pointers even with zero size (Ampere).
  auto async = cast<nvgpu::DeviceAsyncCopyOp>(op);
  Location loc = async.getLoc();

  auto ifOp = rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());

  auto newAsync = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(async.getContext()),
      async.getDst(), async.getDstIndices(),
      async.getSrc(), async.getSrcIndices(),
      async.getDstElementsAttr(), async.getSrcElements(),
      async.getBypassL1Attr());

  rewriter.setInsertionPointAfter(ifOp);
  rewriter.replaceOp(async, newAsync.getResult());
  return ifOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Backward-slice helper (port of IREE :112–123).
//===----------------------------------------------------------------------===//

static void addDepOps(llvm::SmallDenseSet<Operation *> &dep, Operation *op,
                      Block *block) {
  if (!dep.insert(op).second)
    return;
  for (Value operand : op->getOperands())
    if (Operation *def = operand.getDefiningOp())
      if (def->getBlock() == block)
        addDepOps(dep, def, block);
  // Trace through nested regions (e.g. scf.if).
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (def->getBlock() == block)
          addDepOps(dep, def, block);
  });
}

//===----------------------------------------------------------------------===//
// Schedule: stage 0 = cp.async + commit + deps; stage depth-1 = everything else.
//===----------------------------------------------------------------------===//

static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation *, unsigned>> &ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker))
    return;

  llvm::SmallDenseSet<Operation *> stage0Deps;
  for (Operation &op : forOp.getBody()->getOperations())
    if (op.hasAttr(kPipeliningFirstStage))
      addDepOps(stage0Deps, &op, forOp.getBody());

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    if (stage0Deps.contains(&op))
      ops.push_back({&op, 0});
  }
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    if (!stage0Deps.contains(&op))
      ops.push_back({&op, depth - 1});
  }
}

//===----------------------------------------------------------------------===//
// Annotate async-wait counts after each clone.
// Handles both nvgpu.device_async_wait and nvvm.cp.async.wait.group.
// Called by the pipeliner for every cloned op in prologue, kernel, epilogue.
//===----------------------------------------------------------------------===//

static void setAsyncAnnotations(Operation *op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth) {
  const int count = computeNumGroupsInFlight(part, iteration, depth);
  op->walk([&](Operation *inner) {
    rewriteWaitGroupCount(inner, count);
  });
}

//===----------------------------------------------------------------------===//
// Tag the loop & its first-stage ops.
//===----------------------------------------------------------------------===//

static bool setPipeliningMarkers(scf::ForOp forOp) {
  bool hasAsyncCopy = false;
  OpBuilder b(forOp.getContext());

  for (Operation &op : forOp.getBody()->getOperations()) {
    // Reject unknown region-bearing ops — we can't safely pipeline through
    // them. scf.if, scf.forall, and scf.for are explicitly allowed.
    if (op.getNumRegions() > 0 &&
        !isa<scf::ForallOp, scf::IfOp, scf::ForOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-pipelining] skip loop: child op "
                 << op.getName() << " carries a region\n");
      return false;
    }

    // Reject any scf.for whose body contains a barrier. A barrier inside a
    // non-uniformly-entered loop deadlocks: threads not in the copy partition
    // are already past the outer barrier while threads inside this loop are
    // still spinning on inner barriers.
    if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
      bool hasBarrier = false;
      innerFor.walk([&](Operation *nested) {
        if (isBarrierOp(nested))
          hasBarrier = true;
      });
      if (hasBarrier) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-pipelining] skip loop: child scf.for contains "
                      "barrier — non-uniform barrier would deadlock\n");
        return false;
      }
    }

    // Determine whether this op belongs in Stage 0.
    // An op is first-stage if it IS or CONTAINS:
    //   • nvgpu.device_async_copy        — the actual async transfer
    //   • nvgpu.device_async_create_group (non-empty) — NVGPU commit
    //   • nvvm.cp.async.commit.group     — NVVM commit
    //
    // Both commit forms must stay in Stage 0 with the copies they commit.
    // If a commit lands in Stage 1, the pipeliner time-shifts the copies to
    // the previous iteration but leaves the commit in place, so the group
    // that Stage 1's wait drains is the one committed in the *current*
    // iteration — not the shifted one — and the SMEM data for the current
    // compute is never guaranteed to be ready.
    bool isFirstStage = false;
    op.walk([&](Operation *nested) {
      if (isa<nvgpu::DeviceAsyncCopyOp>(nested))
        isFirstStage = true;
      if (isCommitGroupOp(nested))
        isFirstStage = true;
    });

    if (isFirstStage) {
      hasAsyncCopy = true;
      op.setAttr(kPipeliningFirstStage, b.getUnitAttr());
    }
  }

  if (hasAsyncCopy)
    forOp->setAttr(kPipeliningLoopMarker, b.getUnitAttr());
  return hasAsyncCopy;
}

//===----------------------------------------------------------------------===//
// Pre-pipelining: merge per-tile cp.async commit/wait ops into a single
// commit/wait pair at the end of the K-loop body.
//
// Bufferization emits one `linalg.copy` per promoted operand (A and B in
// matmul), and each lowering wraps its own commit_group + wait_group around
// its scf.if. So the K-loop body looks like:
//
//   scf.for K {
//     scf.if (predA) { cp.async A...; commit_group; wait_group 0 }
//     scf.if (predB) { cp.async B...; commit_group; wait_group 0 }
//     ... compute ...
//   }
//
// With this shape pipelining is fundamentally broken at depth=N>1: each
// iteration issues 2 commits, so after the prologue there are 2(N-1) groups
// in flight, not (N-1). wait_group(N-1) drains the wrong number of groups
// and the consumer reads SMEM that hasn't been filled. The PTX symptom is
// `wait_group 2` after a single commit (effectively wait-all=no-op),
// scattered between commits — which produces wrong values and, with the
// dynamic-SMEM addressing, illegal memory accesses on the final iterations.
//
// This pass rewrites the body to:
//
//   scf.for K {
//     scf.if (predA) { cp.async A... }      // commit/wait removed
//     scf.if (predB) { cp.async B... }      // commit/wait removed
//     commit_group                          // single commit
//     wait_group 0                          // single wait (rewritten to
//                                            // depth-1 by setAsyncAnnotations
//                                            // during pipelining)
//     ... compute ...
//   }
//
// Returns true if the body was modified, false if there was nothing to merge.
//===----------------------------------------------------------------------===//
static bool mergeAsyncCommitsInKLoop(scf::ForOp forOp) {
  Block *body = forOp.getBody();
  MLIRContext *ctx = forOp.getContext();

  // Collect every commit/wait op nested inside scf.if children of the K-loop
  // body. We only consider those whose direct ancestor (within the K-loop
  // body) is an scf.if — these are the per-tile commit/wait pairs that the
  // bufferize lowering emitted alongside cp.async.
  SmallVector<Operation *> commits;
  SmallVector<Operation *> waits;
  unsigned ifsWithCommit = 0;

  for (Operation &child : body->getOperations()) {
    auto ifOp = dyn_cast<scf::IfOp>(&child);
    if (!ifOp)
      continue;
    bool localCommit = false;
    bool localAsync = false;
    ifOp.walk([&](Operation *nested) {
      if (isa<nvgpu::DeviceAsyncCopyOp>(nested))
        localAsync = true;
      if (isCommitGroupOp(nested)) {
        commits.push_back(nested);
        localCommit = true;
      }
      if (isWaitGroupOp(nested))
        waits.push_back(nested);
    });
    if (localCommit && localAsync)
      ++ifsWithCommit;
  }

  // Only merge when the dual-commit-per-iter pattern is actually present:
  // at least two scf.if blocks each containing both cp.async and commit_group.
  // Single-commit kernels (LayerNorm, single-tile-promoted matmuls, etc.)
  // already have the canonical pipeline shape; touching them risks moving
  // commit/wait across barriers in ways that change semantics.
  if (ifsWithCommit < 2)
    return false;

  // Erase the per-tile commit/wait ops. They were inside scf.if blocks, so
  // we have to be careful: the scf.if may now have an empty body, but
  // canonicalize will clean that up later (the scf.if has no results).
  for (Operation *w : waits)
    w->erase();
  for (Operation *c : commits)
    c->erase();

  // Insert a single commit_group + wait_group at the end of the K-loop body,
  // just before the terminator and before the compute that consumes the
  // SMEM. We use the NVVM forms (commit.group, wait.group) because the
  // existing per-tile ops were lowered to NVVM by ConvertNVGPUToNVVM. Match
  // their dialect so the rest of the pipeline keeps working.
  OpBuilder b(ctx);
  Operation *terminator = body->getTerminator();
  // Place commit/wait at the very start of the body so the wait happens
  // before any compute reads SMEM. (We can't put them at the end and still
  // expect compute on iter k to read iter k's data — the wait must precede
  // the read.)
  //
  // BUT: putting commit at the start doesn't make sense because there is
  // nothing to commit yet. The right place is: AFTER the cp.async-emitting
  // scf.ifs, BEFORE any compute. Since the bufferize lowering puts the
  // cp.async scf.ifs first in the body and compute after, we insert the
  // commit/wait between them.
  //
  // Find the insertion point: just after the last scf.if that contains
  // a cp.async.
  Operation *insertAfter = nullptr;
  for (Operation &child : body->getOperations()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(&child)) {
      bool hasAsync = false;
      ifOp.walk([&](nvgpu::DeviceAsyncCopyOp) { hasAsync = true; });
      if (hasAsync)
        insertAfter = &child;
    }
  }
  if (!insertAfter)
    insertAfter = terminator->getPrevNode();

  b.setInsertionPointAfter(insertAfter);
  Location loc = insertAfter->getLoc();
  b.create<NVVM::CpAsyncCommitGroupOp>(loc);
  // numGroups = 0 here is the "wait-all" sentinel that setAsyncAnnotations
  // will rewrite to depth-1 (= 2) during pipelining. If pipelining bails
  // out for any reason, wait-all (0) is the safe fallback that preserves
  // correctness (just at the cost of zero overlap).
  b.create<NVVM::CpAsyncWaitGroupOp>(loc, b.getI32IntegerAttr(0));
  // Barrier after wait so all threads in the block see the just-arrived
  // SMEM data before the compute reads it. The reference PTX shows the
  // same pattern (`wait_group 2; bar.sync 0; ldmatrix...`).
  b.create<NVVM::Barrier0Op>(loc);
  return true;
}

//===----------------------------------------------------------------------===//
// Apply pipelining to one scf.for.
//===----------------------------------------------------------------------===//

static FailureOr<scf::ForOp> applyPipelining(scf::ForOp forOp, unsigned depth) {
  // Step 1: merge per-tile commit/wait pairs into a single commit/wait pair
  // before pipelining sees the body. Without this, depth-N pipelining races
  // because each iteration issues N copies but commits N times, and
  // wait_group(N-1) drains the wrong number of groups.
  (void)mergeAsyncCommitsInKLoop(forOp);

  if (!setPipeliningMarkers(forOp))
    return failure();

  scf::PipeliningOption options;
  options.getScheduleFn =
      [depth](scf::ForOp f,
              std::vector<std::pair<Operation *, unsigned>> &ops) {
        getPipelineStages(f, ops, depth);
      };
  options.annotateFn =
      [depth](Operation *op, scf::PipeliningOption::PipelinerPart part,
              unsigned iteration) {
        setAsyncAnnotations(op, part, iteration, depth);
      };
  options.peelEpilogue = false;
  // After kernel outlining + index sinking the K-loop bounds become
  // gpu.func block arguments, not arith.constant. Without this flag
  // upstream's initializeLoopInfo bails on getConstantIntValue().
  options.supportDynamicLoops = true;
  options.predicateFn =
      [](RewriterBase &rewriter, Operation *op, Value pred) {
        return replaceOpWithPredicatedOp(rewriter, op, pred);
      };

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return scf::pipelineForLoop(rewriter, forOp, options);
}

//===----------------------------------------------------------------------===//
// NovaGPUPipeliningPass
//===----------------------------------------------------------------------===//

struct NovaGPUPipeliningPass
    : public PassWrapper<NovaGPUPipeliningPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUPipeliningPass)

  NovaGPUPipeliningPass() = default;
  NovaGPUPipeliningPass(const NovaGPUPipeliningPass &o)
      : PassWrapper(o), depth(o.depth) {}
  explicit NovaGPUPipeliningPass(unsigned d) : depth(d) {}

  unsigned depth = 3;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect,
                    memref::MemRefDialect, nvgpu::NVGPUDialect,
                    NVVM::NVVMDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.getFunctionBody().empty() || depth < 2)
      return;

    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp f) {
      if (f->hasAttr(kNovaMultiBufferedLoopMarker))
        loops.push_back(f);
    });

    unsigned pipelined = 0;
    for (scf::ForOp loop : loops) {
      if (succeeded(applyPipelining(loop, depth)))
        ++pipelined;
      else
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-pipelining] failed to pipeline loop at "
                   << loop.getLoc() << "\n");
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[nova-pipelining] pipelined " << pipelined << " / "
               << loops.size() << " marked loops at depth=" << depth << "\n");
  }

  StringRef getArgument() const override { return "nova-gpu-pipelining"; }
  StringRef getDescription() const override {
    return "Software-pipelines K-loops containing nvgpu.device_async_copy: "
           "stage 0 = cp.async + commit + deps; stage depth-1 = compute. "
           "Handles NVGPU dialect and raw NVVM intrinsic forms of commit/wait. "
           "Only operates on loops carrying the multi-buffer marker; wait "
           "counts are rewritten from wait-all to wait(depth-1) per iteration.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUPipeliningPass(unsigned depth) {
  return std::make_unique<NovaGPUPipeliningPass>(depth);
}

void registerNovaGPUPipeliningPass() {
  PassRegistration<NovaGPUPipeliningPass>();
}

} // namespace mlir::nova