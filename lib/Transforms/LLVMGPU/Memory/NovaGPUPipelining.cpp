//===- NovaGPUPipelining.cpp - cp.async software pipelining --------------===//
//
// Port of IREE's GPUPipelining.cpp (compiler/src/iree/compiler/Codegen/
// Common/GPU/GPUPipelining.cpp) — `loadGlobalStage0` strategy only.
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
// Pre-conditions (after NovaGPUMultiBuffering + NovaGPUCreateAsyncCopies)
// ──────────────────────────────────────────────────────────────────────
//   * Workgroup-memory allocs are widened to <N x …, #ws>; subviews use
//     `(iv mod N)` indexing.
//   * The K-loop body contains:
//        nvgpu.device_async_copy …
//        nvgpu.device_async_create_group …
//        nvgpu.device_async_wait %g                    // numGroups unset
//        gpu.barrier
//        <compute reading the smem subview>
//   * The K-loop body is a flat region (no scf.if / scf.forall / scf.for
//     children). When it isn't, this pass leaves the loop unchanged — the
//     async copies still run synchronously via wait(0), so correctness is
//     preserved.
//
// Effect
// ──────
//   * Stage 0 = cp.async ops + create_group + their backward-slice deps.
//   * Stage `depth-1` = everything else (compute + wait + barrier).
//   * The `wait`'s `numGroups` attribute is rewritten to `depth-1` in the
//     kernel and prologue, and decreased iteration-by-iteration in the
//     epilogue.
//
// Notes
// ─────
//   * `peelEpilogue=false` + a `predicateFn` → un-peeled epilogue. The
//     `replaceOpWithPredicatedOp` callback rewrites the cp.async to its
//     zfill form (srcElements = pred ? N : 0) for the partial last
//     iterations.
//   * `bypassL1` is intentionally NOT toggled in the predicated form —
//     matches the safe choice in NovaGPUCreateAsyncCopies (matmul.cpp:450
//     warns cp.async.cg corrupted memory in this tree).
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

// Marker attrs on the loop body — adopted verbatim from IREE.
constexpr StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
constexpr StringLiteral kPipeliningFirstStage = "__pipelining_first_stage__";

// True if the memref lives in GPU workgroup (shared) memory.
static bool hasSharedMemoryAddressSpace(MemRefType t) {
  auto attr = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return attr &&
         attr.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

//===----------------------------------------------------------------------===//
// Predicated cp.async for un-peeled epilogue (port of IREE :44–108).
//===----------------------------------------------------------------------===//

static Operation *replaceOpWithPredicatedOp(RewriterBase &rewriter,
                                            Operation *op, Value pred) {
  if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
    // Recursively predicate any async copies found inside the forall.
    // We must ensure the rewriter insertion point is shifted into the forall
    // so the new predicated ops (and their arith.select deps) land inside.
    forallOp.walk([&](nvgpu::DeviceAsyncCopyOp asyncOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(asyncOp);
      replaceOpWithPredicatedOp(rewriter, asyncOp, pred);
    });
    return forallOp;
  }

  // After MapForallToGPU, predicated copy blocks become scf.if ops.
  // In the epilogue, AND the pipeline predicate with the existing condition
  // so that: (a) the thread mask still applies, and (b) out-of-bounds
  // iterations don't issue copies into stale shared-memory slots.
  // We update the condition in-place to avoid block-manipulation issues.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Only handle void-result if blocks (copy emission blocks have no yields).
    if (ifOp.getNumResults() != 0)
      return nullptr;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);
    Value combinedCond = rewriter.create<arith::AndIOp>(
        ifOp.getLoc(), pred, ifOp.getCondition());
    ifOp->setOperand(0, combinedCond);
    return ifOp;
  }

  if (!isa<nvgpu::DeviceAsyncCopyOp>(op)) {
    // Side-effect-free ops execute speculatively.
    if (isMemoryEffectFree(op))
      return op;
    // Group / wait / barrier execute speculatively too — they're idempotent
    // or harmless when run on already-drained pipelines.
    if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncCreateGroupOp,
            nvgpu::DeviceAsyncWaitOp>(op))
      return op;
    // Loads from shared memory are also OK speculatively — the data is
    // there from earlier iterations.
    if (auto vload = dyn_cast<vector::LoadOp>(op)) {
      auto ty = cast<MemRefType>(vload.getBase().getType());
      if (hasSharedMemoryAddressSpace(ty))
        return op;
    }
    if (auto mload = dyn_cast<memref::LoadOp>(op)) {
      if (hasSharedMemoryAddressSpace(mload.getMemRefType()))
        return op;
    }
    // Unknown op without predication support — give up; caller will leave
    // the loop unmodified.
    return nullptr;
  }

  auto async = cast<nvgpu::DeviceAsyncCopyOp>(op);
  Location loc = async.getLoc();

  // srcElements = pred ? (existing srcElements ?? dstElements) : 0
  Value dstElems = rewriter.create<arith::ConstantOp>(
      loc, async.getDstElementsAttr());
  Value origSrc =
      async.getSrcElements() ? async.getSrcElements() : dstElems;
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value newSrcElems =
      rewriter.create<arith::SelectOp>(loc, pred, origSrc, zero);

  auto newAsync = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(async.getContext()),
      async.getDst(), async.getDstIndices(), async.getSrc(),
      async.getSrcIndices(), async.getDstElementsAttr(), newSrcElems,
      /*bypassL1=*/UnitAttr{});
  rewriter.replaceOp(async, newAsync.getResult());
  return newAsync.getOperation();
}

//===----------------------------------------------------------------------===//
// Backward-slice helper (port of IREE :112–123).
//===----------------------------------------------------------------------===//

static void addDepOps(llvm::SmallDenseSet<Operation *> &dep, Operation *op,
                      Block *block) {
  if (!dep.insert(op).second)
    return;

  // Trace dependencies of the op itself.
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp())
      if (def->getBlock() == block)
        addDepOps(dep, def, block);
  }

  // Trace dependencies of all operations nested inside regions (e.g. scf.forall).
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      if (Operation *def = operand.getDefiningOp())
        if (def->getBlock() == block)
          addDepOps(dep, def, block);
    }
  });
}

//===----------------------------------------------------------------------===//
// Schedule: stage 0 = cp.async + deps; stage depth-1 = everything else
// (port of IREE :128–157, simplified — single strategy).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Utility to detect "real" first-stage operations.
//===----------------------------------------------------------------------===//

static bool isRealAsyncCopy(Operation *op) {
  bool isFirstStage = false;
  op->walk([&](Operation *nested) {
    if (isa<nvgpu::DeviceAsyncCopyOp>(nested))
      isFirstStage = true;
    if (auto cg = dyn_cast<nvgpu::DeviceAsyncCreateGroupOp>(nested)) {
      if (cg.getNumOperands() > 0)
        isFirstStage = true;
    }
  });
  return isFirstStage;
}

static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation *, unsigned>> &ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker))
    return;

  // Collect Stage 0 = first-stage ops + their transitive backward-slice deps.
  // Without this closure, address computations (affine.apply, remui, etc.)
  // that feed into Stage 0 scf.if blocks would be left in Stage 1, causing
  // "operation scheduled before its operands" failures in scf::pipelineForLoop.
  llvm::SmallDenseSet<Operation *> stage0Deps;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningFirstStage))
      addDepOps(stage0Deps, &op, forOp.getBody());
  }

  // Emit in stage order: Stage 0 (transfers) first, then Stage 1 (compute).
  // This allows Issue(N) to happen before Wait(N-1), enabling overlap.
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
// Annotate cp.async.wait_group counts after each clone (port of IREE :159–197,
// simplified — only the wait-count branch).
//===----------------------------------------------------------------------===//

static void setAsyncAnnotations(Operation *op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth) {
  op->walk([&](nvgpu::DeviceAsyncWaitOp wait) {
    if (wait.getNumGroups())
      return;

    int numGroupsInFlight = 0;
    if (part == scf::PipeliningOption::PipelinerPart::Kernel ||
        part == scf::PipeliningOption::PipelinerPart::Prologue) {
      // Kernel: depth-1 groups in flight at any time.
      numGroupsInFlight = (int)depth - 1;
    } else {
      // Epilogue: count down each peeled iteration.
      numGroupsInFlight = (int)depth - 1 - (int)iteration;
    }
    if (numGroupsInFlight < 0)
      numGroupsInFlight = 0;

    OpBuilder b(wait);
    wait->setAttr(wait.getNumGroupsAttrName(),
                  b.getI32IntegerAttr(numGroupsInFlight));
  });
}

//===----------------------------------------------------------------------===//
// Tag the loop & its first-stage ops (port of IREE :202–256, simplified).
//===----------------------------------------------------------------------===//

static bool setPipeliningMarkers(scf::ForOp forOp) {
  bool hasAsyncCopy = false;
  OpBuilder b(forOp.getContext());
  for (Operation &op : forOp.getBody()->getOperations()) {
    // Pipeline flat loops OR loops containing scf.if / scf.forall region ops
    // (thread-masked cp.async). After MapForallToGPU, scf.forall is lowered
    // to scf.if blocks, so we must allow scf.if here.
    if (op.getNumRegions() > 0 &&
        !isa<scf::ForallOp, scf::IfOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[nova-pipelining] skip loop: child op "
                              << op.getName() << " carries a region\n");
      return false;
    }

    // Mark the op as first-stage if it IS an async copy/group OR if it
    // CONTAINS one (handles the predicated scf.forall case).
    bool isFirstStage = false;
    op.walk([&](Operation *nested) {
      if (isa<nvgpu::DeviceAsyncCopyOp>(nested))
        isFirstStage = true;
      // Mark create_group as first stage ONLY if it has operands (the tokens
      // from real copies). Dummy groups used for wait separation stay in
      // the later stage.
      if (auto cg = dyn_cast<nvgpu::DeviceAsyncCreateGroupOp>(nested)) {
        if (cg.getNumOperands() > 0)
          isFirstStage = true;
      }
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
// Apply pipelining to one scf.for (port of IREE :634–679, simplified).
//===----------------------------------------------------------------------===//

static FailureOr<scf::ForOp> applyPipelining(scf::ForOp forOp, unsigned depth) {
  if (!setPipeliningMarkers(forOp))
    return failure();

  scf::PipeliningOption options;
  options.getScheduleFn = [depth](scf::ForOp f,
                                  std::vector<std::pair<Operation *, unsigned>>
                                      &ops) { getPipelineStages(f, ops, depth); };
  options.annotateFn = [depth](Operation *op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration) {
    setAsyncAnnotations(op, part, iteration, depth);
  };
  options.peelEpilogue = false;
  options.predicateFn = [](RewriterBase &rewriter, Operation *op,
                           Value pred) {
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

  unsigned depth = 2;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    nvgpu::NVGPUDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.getFunctionBody().empty() || depth < 2)
      return;

    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp f) { loops.push_back(f); });

    unsigned pipelined = 0;
    for (scf::ForOp loop : loops) {
      if (succeeded(applyPipelining(loop, depth)))
        ++pipelined;
      else
        LLVM_DEBUG(llvm::dbgs() << "[nova-pipelining] failed to pipeline loop at " 
                                << loop.getLoc() << "\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "[nova-pipelining] pipelined " << pipelined
                            << " / " << loops.size() << " loops at depth="
                            << depth << "\n");
  }

  StringRef getArgument() const override { return "nova-gpu-pipelining"; }
  StringRef getDescription() const override {
    return "Software-pipelines K-loops containing nvgpu.device_async_copy: "
           "stage 0 = cp.async + deps; stage depth-1 = compute. Wait counts "
           "are rewritten from wait-all to wait(depth-1) per iteration.";
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
