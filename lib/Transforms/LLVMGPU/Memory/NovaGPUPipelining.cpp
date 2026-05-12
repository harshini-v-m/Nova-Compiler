//===- NovaGPUPipelining.cpp - cp.async software pipelining --------------===//
//
// Port of IREE's GPUPipelining.cpp `loadGlobalStage0` strategy.
//
// Bug fixes in this refactor
// ────────────────────────────
//
// BUG 1 — splitConsumerIndexChainsFromProducers producer chain construction:
// the chain must contain (a) every body-level op the pipeliner will time-shift
// into stage 0, AND (b) every body-level op that lives in stage 0's transitive
// backward slice through nested regions. Earlier attempts either pulled
// scf::if-internal ops directly into the chain (wrong — those aren't body-
// level so the clone code couldn't insert clones in the right block), or
// stopped at body-level ops without walking into stage-0 regions (also wrong
// — body-level ops referenced from inside the cp.async's wrapping scf::if
// were missed, so the scf::if itself was treated as external and clones got
// rewired incorrectly). The fix: addProducerDeps inserts only body-level ops
// into the chain, but walks into nested regions of any body-level op in the
// chain to find more body-level operands. This mirrors the pipeliner's own
// addDepOps logic.
//
// BUG 2 — splitConsumerIndexChainsFromProducers iterated CONSUMERS and walked
// each consumer's operand chain backward, cloning producer-chain ops it found.
// That misses the canonical case: producer cp.async writes through subview_a
// and consumer reads through subview_b, where both subviews share an
// affine.apply (`(iv floordiv step) mod N`) by CSE. subview_a is in the
// producer chain; subview_b is NOT (no path from cp.async to subview_b), so
// consumer-walk reaches subview_b first, stops, and never clones the shared
// apply. Fix: iterate PRODUCERS instead. For each clonable producer-chain op
// with at least one user outside the producer chain, clone the op once after
// the original and rewire the external users to the clone. Now the consumer's
// path through the shared apply goes via the clone (which the pipeliner
// doesn't time-shift), while the original keeps feeding cp.async.
//
// BUG 3 — setAsyncAnnotations called op->walk() on every cloned op, causing
// DeviceAsyncWaitOps inside cloned scf::if blocks to be re-annotated with the
// outer op stage/iteration — not the wait's own context. Prologue waits were
// overwritten with epilogue counts, collapsing effective pipeline depth to 1.
// Fix: call rewriteWaitGroupCount directly on the op itself, no walk.
//
// BUG 4 — mergeAsyncCommitsInKLoop searched for insertAfter AFTER erasing
// commit/wait ops. If the last async-containing op was a commit (now erased),
// insertAfter was a dangling pointer. Fix: find insertAfter before any erases.
//
// BUG 5 — mergeAsyncCommitsInKLoop did not erase barriers adjacent to the
// pre-existing commit/wait ops (placed by NovaGPUInsertWorkgroupBarriers).
// The new canonical triple also inserts a barrier, so the body ended up with
// two barriers both landing in stage depth-1 — producing back-to-back bar.sync
// instructions in the epilogue that deadlock. Fix: collect and erase top-level
// body barriers whose immediate neighbour is a commit or wait op before erasing
// commits/waits.
//
// BUG 6 (primary numerical bug) — computeNumGroupsInFlight returned depth-1
// for ALL prologue iterations. On prologue iteration 0 only 1 group has been
// committed, so wait(depth-1=2) is a no-op — the first SMEM tile is never
// confirmed ready before compute reads it. The matmul on that iteration reads
// uninitialized or stale SMEM, which is the direct cause of the loss drift.
// Fix: prologue iteration i should wait until only i groups remain in flight
// (return i). On iteration 0 this is wait(0) = wait-all, confirming the first
// tile is fully loaded before any compute proceeds.
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

constexpr StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
constexpr StringLiteral kPipeliningFirstStage = "__pipelining_first_stage__";

static bool hasSharedMemoryAddressSpace(MemRefType t) {
  auto attr = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return attr && attr.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

//===----------------------------------------------------------------------===//
// Dialect-agnostic predicate helpers.
//===----------------------------------------------------------------------===//

static bool isBarrierOp(Operation *op) {
  return isa<gpu::BarrierOp, NVVM::Barrier0Op>(op);
}

static bool isCommitGroupOp(Operation *op) {
  return isa<NVVM::CpAsyncCommitGroupOp, nvgpu::DeviceAsyncCreateGroupOp>(op);
}

static bool isWaitGroupOp(Operation *op) {
  return isa<nvgpu::DeviceAsyncWaitOp, NVVM::CpAsyncWaitGroupOp>(op);
}

//===----------------------------------------------------------------------===//
// Compute how many async groups should remain in flight after the wait.
//===----------------------------------------------------------------------===//

// How many async groups should remain in-flight after the wait in each
// part of the pipelined loop:
//
//   Prologue iteration i (0-based):
//     After iteration i, exactly i+1 groups have been committed (one per
//     prologue step). We want compute in the NEXT iteration to drain the
//     oldest one, so we leave i groups in flight — i.e. wait until only
//     i remain. Returning (depth-1) for every prologue step was wrong: on
//     prologue iteration 0 only 1 group exists, so wait(depth-1=2) is a
//     no-op and the first SMEM tile is never confirmed ready before compute.
//
//   Kernel: always depth-1 groups remain (steady-state overlap).
//
//   Epilogue iteration i (0-based):
//     One fewer group per drain step. depth-2-i, floored at 0.
static int computeNumGroupsInFlight(scf::PipeliningOption::PipelinerPart part,
                                    unsigned iteration, unsigned depth) {
  if (part == scf::PipeliningOption::PipelinerPart::Prologue)
    return static_cast<int>(iteration);
  if (part == scf::PipeliningOption::PipelinerPart::Kernel)
    return static_cast<int>(depth) - 1;
  // Epilogue: one fewer group per step.
  return std::max(0, static_cast<int>(depth) - 2 - static_cast<int>(iteration));
}

//===----------------------------------------------------------------------===//
// Rewrite a single async-wait op (either dialect) to the correct count.
//
// FIX (BUG 3): Do NOT use op->walk() here. This function is called directly
// on the cloned op from setAsyncAnnotations. Walking into nested regions of
// unrelated ops caused double-rewrites with wrong stage/iteration values.
//===----------------------------------------------------------------------===//

static bool rewriteWaitGroupCount(Operation *op, int count) {
  OpBuilder b(op);

  if (auto wait = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op)) {
    wait->setAttr(wait.getNumGroupsAttrName(), b.getI32IntegerAttr(count));
    return true;
  }

  if (auto wait = dyn_cast<NVVM::CpAsyncWaitGroupOp>(op)) {
    wait->setAttr("n", b.getI32IntegerAttr(count));
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Predicated cp.async for un-peeled epilogue (port of IREE).
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

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp.getNumResults() != 0)
      return nullptr;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);
    Value combinedCond = rewriter.create<arith::AndIOp>(ifOp.getLoc(), pred,
                                                        ifOp.getCondition());
    ifOp->setOperand(0, combinedCond);
    return ifOp;
  }

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
    if (isBarrierOp(op) || isCommitGroupOp(op) || isWaitGroupOp(op))
      return op;
    if (auto vload = dyn_cast<vector::LoadOp>(op)) {
      if (hasSharedMemoryAddressSpace(
              cast<MemRefType>(vload.getBase().getType())))
        return op;
    }
    if (auto mload = dyn_cast<memref::LoadOp>(op)) {
      if (hasSharedMemoryAddressSpace(mload.getMemRefType()))
        return op;
    }
    return nullptr;
  }

  auto async = cast<nvgpu::DeviceAsyncCopyOp>(op);
  Location loc = async.getLoc();

  auto ifOp = rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());

  auto newAsync = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(async.getContext()), async.getDst(),
      async.getDstIndices(), async.getSrc(), async.getSrcIndices(),
      async.getDstElementsAttr(), async.getSrcElements(),
      async.getBypassL1Attr());

  rewriter.setInsertionPointAfter(ifOp);
  rewriter.replaceOp(async, newAsync.getResult());
  return ifOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Backward-slice helper (port of IREE).
//===----------------------------------------------------------------------===//

static void addDepOps(llvm::SmallDenseSet<Operation *> &dep, Operation *op,
                      Block *block) {
  if (!dep.insert(op).second)
    return;
  for (Value operand : op->getOperands())
    if (Operation *def = operand.getDefiningOp())
      if (def->getBlock() == block)
        addDepOps(dep, def, block);
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (def->getBlock() == block)
          addDepOps(dep, def, block);
  });
}

//===----------------------------------------------------------------------===//
// Stage assignment.
//===----------------------------------------------------------------------===//

static void
getPipelineStages(scf::ForOp forOp,
                  std::vector<std::pair<Operation *, unsigned>> &ops,
                  unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker))
    return;

  llvm::SmallDenseSet<Operation *> stage0Deps;
  for (Operation &op : forOp.getBody()->getOperations())
    if (op.hasAttr(kPipeliningFirstStage))
      addDepOps(stage0Deps, &op, forOp.getBody());

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (stage0Deps.contains(&op))
      ops.push_back({&op, 0});
  }
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (!stage0Deps.contains(&op))
      ops.push_back({&op, depth - 1});
  }
}

//===----------------------------------------------------------------------===//
// Annotate async-wait counts after each clone.
//
// FIX (BUG 3): Call rewriteWaitGroupCount directly on `op`, NOT via
// op->walk(). The pipeliner calls annotateFn for every cloned op individually.
// Walking into nested regions here caused wait ops nested inside cloned scf::if
// blocks to be re-annotated with the outer op's stage/iteration context,
// collapsing all prologue wait counts to epilogue values and making the
// pipeline behave as depth-1.
//===----------------------------------------------------------------------===//

static void setAsyncAnnotations(Operation *op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth) {
  const int count = computeNumGroupsInFlight(part, iteration, depth);
  // Only rewrite this exact op — do not walk into nested regions.
  rewriteWaitGroupCount(op, count);
}

//===----------------------------------------------------------------------===//
// Tag the loop & its first-stage ops.
//===----------------------------------------------------------------------===//

static bool setPipeliningMarkers(scf::ForOp forOp) {
  bool hasAsyncCopy = false;
  OpBuilder b(forOp.getContext());

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (op.getNumRegions() > 0 &&
        !isa<scf::ForallOp, scf::IfOp, scf::ForOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[nova-pipelining] skip loop: child op "
                              << op.getName() << " carries a region\n");
      return false;
    }

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
// Pre-pipelining: split CSE'd index chains so the pipeliner's stage-0
// time-shift moves only the producer's chain, not the consumer's.
//
// FIX (BUG 1): Build producerChain using only direct operand edges whose
// defining op lives in the loop body block — do NOT use op->walk() to descend
// into nested regions when collecting producer deps. ops inside scf::if
// bodies have a different parent Block; pulling them into producerChain caused
// getCloned to insert clones into the wrong block and left the original SSA
// edge intact, so no split occurred.
//
// FIX (BUG 2): Iterate PRODUCERS, not consumers. Walking forward from each
// producer-chain op to its users finds the shared affine.apply directly: the
// apply is in the producer chain, and its consumer-side user (subview_b) has a
// body-level ancestor that is NOT in the producer chain. We clone the producer
// op once after the original and rewire those external users to the clone.
// The previous consumer-walk direction missed this because the consumer's
// operand chain stops at subview_b (not in producer chain) before ever
// reaching the shared apply.
//===----------------------------------------------------------------------===//
static void splitConsumerIndexChainsFromProducers(scf::ForOp forOp) {
  Block *body = forOp.getBody();

  // Build the producer chain: body-level ops that the pipeliner will pull into
  // stage 0 via its backward-slice walk on stage-0-tagged ops. The body-level
  // op that gets stage-0-tagged is whichever op CONTAINS the cp.async — when
  // cp.async sits inside an scf.if, the scf.if is the body-level ancestor and
  // it is what the pipeliner marks as first-stage. So our producerChain must
  // include that body-level ancestor, and tracing must walk into its nested
  // regions to find every operand the pipeliner will sweep.
  //
  // (Without including the scf.if, the iteration below would mark the scf.if
  // as an "external" user of every shared op — and the clone we'd insert
  // would be used by BOTH the producer cp.async and the consumer reads, since
  // both sit inside an scf.if that's "external" by that mistaken
  // classification.)
  llvm::SmallDenseSet<Operation *> producerChain;
  std::function<void(Operation *)> addProducerDeps = [&](Operation *op) {
    if (!op || op->getBlock() != body || producerChain.count(op))
      return;
    producerChain.insert(op);
    // Direct operands of this body-level op.
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (def->getBlock() == body)
          addProducerDeps(def);
    // If this body-level op carries regions (e.g. an scf.if wrapping the
    // cp.async), trace operand chains of every nested op back to body level.
    // The pipeliner's addDepOps does the same (it walks into nested regions
    // of stage-0 ops), so we must mirror that here to know which body-level
    // ops will end up in stage 0.
    if (op->getNumRegions() > 0) {
      op->walk([&](Operation *nested) {
        if (nested == op)
          return;
        for (Value operand : nested->getOperands())
          if (Operation *def = operand.getDefiningOp())
            if (def->getBlock() == body)
              addProducerDeps(def);
      });
    }
  };

  // Seed from each body-level op that walks-contains a cp.async. That
  // body-level op is what the pipeliner stage-0-tags; we add it (and its
  // backward slice through nested regions) to producerChain.
  for (Operation &topOp : body->getOperations()) {
    bool containsAsync = false;
    topOp.walk([&](nvgpu::DeviceAsyncCopyOp) { containsAsync = true; });
    if (containsAsync)
      addProducerDeps(&topOp);
  }

  if (producerChain.empty())
    return;

  // FIX (BUG 2): iterate PRODUCERS, not consumers. The previous direction was
  // wrong: the canonical case is producer cp.async writes through subview_a
  // and consumer reads through subview_b, where both subviews share an
  // affine.apply (`(iv floordiv step) mod N`) by CSE. The producer chain
  // (backward slice from cp.async) contains subview_a and the apply, but NOT
  // subview_b — so a consumer-walk reaches subview_b first, sees it's not in
  // the producer chain, stops, and never touches the actually-shared apply.
  //
  // Iterating producers and looking forward at users finds the shared apply
  // directly: it's a producer-chain op with a user (subview_b) outside the
  // producer chain. We clone the producer-chain op once after the original
  // and rewire those external uses to the clone. The clone is left out of
  // producerChain, so the pipeliner won't time-shift it; the original keeps
  // feeding cp.async and gets shifted to iv+depth-1 as expected.
  auto isClonable = [](Operation *op) {
    return isMemoryEffectFree(op) && op->getNumRegions() == 0;
  };

  // Snapshot in REVERSE body order. We want to clone the deepest-downstream
  // producer-chain op (closest to the consumer) first. When we then iterate
  // up the chain to upstream ops, those see their downstream clones as
  // additional external users — and we clone them too, so the consumer's
  // entire chain becomes a fresh SSA copy. Forward order misses this: when
  // iterating an upstream op like `%subview_5`, its downstream `%subview_7`
  // hasn't been cloned yet, so the upstream op's only "external" user (the
  // future `%subview_7'`) doesn't exist and isn't visible.
  //
  // New clones are not added to producerChain, so by walking ops from
  // last-to-first within producerChain we naturally process downstream first.
  SmallVector<Operation *> producerOps;
  producerOps.reserve(producerChain.size());
  for (Operation &op : body->getOperations())
    if (producerChain.contains(&op))
      producerOps.push_back(&op);

  for (Operation *prod : llvm::reverse(producerOps)) {
    if (!isClonable(prod))
      continue;

    // External uses = uses whose owning op's body-level ancestor is not in
    // producerChain. Walk up to find the body-level ancestor of each user
    // (the user may be nested inside an scf.if or another region-bearing op).
    SmallVector<OpOperand *> externalUses;
    for (OpResult res : prod->getResults())
      for (OpOperand &use : res.getUses()) {
        Operation *userOp = use.getOwner();
        Operation *bodyAncestor = userOp;
        while (bodyAncestor && bodyAncestor->getBlock() != body)
          bodyAncestor = bodyAncestor->getParentOp();
        if (!bodyAncestor)
          continue; // user lives outside this loop body — leave it.
        if (producerChain.contains(bodyAncestor))
          continue; // producer-chain user — keep the original SSA edge.
        externalUses.push_back(&use);
      }
    if (externalUses.empty())
      continue;

    // Clone once, immediately after the original. The clone's results
    // dominate every consumer that follows in program order.
    OpBuilder b(prod);
    b.setInsertionPointAfter(prod);
    Operation *clone = prod->clone();
    b.insert(clone);
    for (OpOperand *use : externalUses) {
      unsigned resIdx = cast<OpResult>(use->get()).getResultNumber();
      use->set(clone->getResult(resIdx));
    }
  }
}

//===----------------------------------------------------------------------===//
// Pre-pipelining: ensure the K-loop body has exactly one
// commit + wait + barrier triple after all cp.async ops.
//
// FIX (BUG 4): Find `insertAfter` BEFORE erasing any pre-existing commit/wait
// ops. The old code erased first, then searched — if the last cp.async-
// containing op was itself a commit (now erased), insertAfter was a dangling
// pointer, causing a use-after-free when setting the insertion point.
//===----------------------------------------------------------------------===//
static bool mergeAsyncCommitsInKLoop(scf::ForOp forOp) {
  Block *body = forOp.getBody();
  MLIRContext *ctx = forOp.getContext();

  SmallVector<Operation *> commits;
  SmallVector<Operation *> waits;
  SmallVector<Operation *> barriers;
  bool hasAnyAsync = false;

  body->walk([&](Operation *op) {
    if (isa<nvgpu::DeviceAsyncCopyOp>(op))
      hasAnyAsync = true;
    if (isCommitGroupOp(op))
      commits.push_back(op);
    if (isWaitGroupOp(op))
      waits.push_back(op);
  });

  if (!hasAnyAsync)
    return false;

  // FIX (BUG 4): Find insertAfter BEFORE erasing ops.
  Operation *insertAfter = nullptr;
  for (Operation &child : body->getOperations()) {
    bool hasAsync = false;
    child.walk([&](nvgpu::DeviceAsyncCopyOp) { hasAsync = true; });
    if (hasAsync)
      insertAfter = &child;
  }

  // Collect barriers that are adjacent to commit/wait ops (i.e. part of an
  // existing async-fence triple). These were inserted by
  // NovaGPUInsertWorkgroupBarriers and are now redundant — the new canonical
  // triple we insert below provides the full fence. Leaving them in produces
  // a double bar.sync in the pipelined epilogue (the doubled-barrier deadlock
  // described in the file header).
  //
  // A barrier is "fence-adjacent" if its immediate predecessor or successor
  // is a commit, wait, or another barrier that is itself fence-adjacent.
  // In practice the pattern is always: [barrier?] commit wait [barrier?]
  // or: commit wait barrier, so we just collect top-level body barriers
  // whose prev or next neighbour is a commit or wait op.
  {
    // Build a set of commit+wait ops for fast lookup.
    llvm::SmallDenseSet<Operation *> asyncFenceOps;
    for (Operation *c : commits)
      asyncFenceOps.insert(c);
    for (Operation *w : waits)
      asyncFenceOps.insert(w);

    for (Operation &child : body->getOperations()) {
      if (!isBarrierOp(&child))
        continue;
      Operation *prev = child.getPrevNode();
      Operation *next = child.getNextNode();
      bool prevIsFence = prev && asyncFenceOps.contains(prev);
      bool nextIsFence = next && asyncFenceOps.contains(next);
      if (prevIsFence || nextIsFence)
        barriers.push_back(&child);
    }
  }

  // Erase in safe order: waits first (they use commit tokens), then commits,
  // then the now-redundant fence barriers.
  for (Operation *w : waits)
    w->erase();
  for (Operation *c : commits)
    c->erase();
  for (Operation *b : barriers)
    b->erase();

  if (!insertAfter)
    insertAfter = body->getTerminator()->getPrevNode();

  OpBuilder b(ctx);
  b.setInsertionPointAfter(insertAfter);
  Location loc = insertAfter->getLoc();

  auto tokenType = nvgpu::DeviceAsyncTokenType::get(ctx);
  Value token =
      b.create<nvgpu::DeviceAsyncCreateGroupOp>(loc, tokenType, ValueRange{})
          .getResult();
  // numGroups left null — setAsyncAnnotations rewrites it per-stage/iteration.
  // If pipelining bails, ConvertNVGPUToNVVMPass lowers null to wait_group 0
  // (wait-all), the safe fallback.
  b.create<nvgpu::DeviceAsyncWaitOp>(loc, token, /*numGroups=*/IntegerAttr());
  b.create<gpu::BarrierOp>(loc);
  return true;
}

//===----------------------------------------------------------------------===//
// Apply pipelining to one scf.for.
//===----------------------------------------------------------------------===//

static FailureOr<scf::ForOp> applyPipelining(scf::ForOp forOp, unsigned depth) {
  // Step 1: merge per-tile commit/wait pairs into one canonical triple.
  // mergeAsyncCommitsInKLoop now finds insertAfter before erasing (BUG 4 fix).
  (void)mergeAsyncCommitsInKLoop(forOp);

  // Step 2: split CSE'd index chains so stage-0 time-shift does not drag the
  // consumer's slot index along with the producer's (BUG 1 + BUG 2 fix).
  splitConsumerIndexChainsFromProducers(forOp);

  if (!setPipeliningMarkers(forOp))
    return failure();

  scf::PipeliningOption options;
  options.getScheduleFn =
      [depth](scf::ForOp f,
              std::vector<std::pair<Operation *, unsigned>> &ops) {
        getPipelineStages(f, ops, depth);
      };
  // FIX (BUG 3): annotateFn no longer calls op->walk — it calls
  // rewriteWaitGroupCount directly on the cloned op only. This prevents
  // re-annotation of nested wait ops with the wrong stage context.
  options.annotateFn = [depth](Operation *op,
                               scf::PipeliningOption::PipelinerPart part,
                               unsigned iteration) {
    setAsyncAnnotations(op, part, iteration, depth);
  };
  options.peelEpilogue = false;
  options.supportDynamicLoops = true;
  options.predicateFn = [](RewriterBase &rewriter, Operation *op, Value pred) {
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
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    nvgpu::NVGPUDialect, NVVM::NVVMDialect, scf::SCFDialect,
                    vector::VectorDialect>();
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
    return "Software-pipelines K-loops containing nvgpu.device_async_copy. "
           "Stage 0 = cp.async + commit + deps; stage depth-1 = compute. "
           "Handles NVGPU dialect and raw NVVM intrinsic forms of commit/wait. "
           "Only operates on loops carrying the multi-buffer marker.";
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