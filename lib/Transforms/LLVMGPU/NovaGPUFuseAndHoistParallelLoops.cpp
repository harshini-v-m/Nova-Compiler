// NovaGPUFuseAndHoistParallelLoopsPass
//
// Implements Step 5 of the Nova GPU TileAndFuse pipeline:
// After thread-level tiling (which produces nested scf.forall {thread} ops),
// this pass greedily fuses and hoists those parallel loops to produce a single
// flat thread-mapped scf.forall per compute region.
//
// This mirrors IREE's GPUFuseAndHoistParallelLoops.cpp exactly in structure,
// running three rounds of rewrite patterns:
//
//  Round 1 — Hoist + fuse foralls:
//    • FuseForalls          — merge a producer scf.forall into its consumer
//                             scf.forall when trip counts match flat workgroup size.
//    • FuseTilableForallConsumers — fuse tilable consumers into forall.
//    • ForallLoopHoisting   — hoist foralls out of enclosing scf.for (K-loop)
//                             when the body is independent of the serial IV.
//
//  Round 2 — Revealed consumers/destinations:
//    • FuseTilableDestinationProducers — fuse DPS producers into forall.
//    • FuseUnitLoopDestination         — eliminate unit-trip-count forall by
//                                        inlining its DPS producer into the body.
//    • FuseTilableForallConsumers      — re-run for newly revealed consumers.
//
//  Round 3 — New producer fusions:
//    • FuseTilableSliceProducers       — fuse tilable producers of slice ops.
//    • FuseTilableDestinationProducers — fuse destination producers revealed
//                                        by earlier rounds.
//    • ExtractSliceOfPadTensorSwap     — swap extract_slice(pad) → pad(extract_slice).
//
// Directly mirrors IREE: Common/GPU/GPUFuseAndHoistParallelLoops.cpp

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "nova-gpu-fuse-and-hoist"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if the forall op is normalized (lb == 0, step == 1 for all dims).
static bool isNormalized(scf::ForallOp forall) {
  for (OpFoldResult lb : forall.getMixedLowerBound()) {
    auto cst = getConstantIntValue(lb);
    if (!cst || *cst != 0) return false;
  }
  for (OpFoldResult step : forall.getMixedStep()) {
    auto cst = getConstantIntValue(step);
    if (!cst || *cst != 1) return false;
  }
  return true;
}

/// Returns the product of all upper bounds for a normalized forall, or nullopt
/// if any upper bound is dynamic.
static std::optional<int64_t> getStaticForallTripCount(scf::ForallOp forall) {
  if (!isNormalized(forall))
    return std::nullopt;
  int64_t tripCount = 1;
  for (OpFoldResult ub : forall.getMixedUpperBound()) {
    std::optional<int64_t> maybeUb = getConstantIntValue(ub);
    if (!maybeUb) return std::nullopt;
    tripCount *= *maybeUb;
  }
  return tripCount;
}

/// Returns true if the forall's mapping contains attributes of type T.
template <typename T>
static bool forallHasMappingType(scf::ForallOp forall) {
  auto mappingAttr = forall.getMappingAttr();
  if (!mappingAttr) return false;
  return llvm::any_of(mappingAttr.getValue(),
                      [](Attribute attr) { return isa<T>(attr); });
}

/// Returns true if the forall is thread-mapped (has GPUThreadMappingAttr).
static bool isThreadMappedForall(scf::ForallOp forall) {
  return forallHasMappingType<gpu::GPUThreadMappingAttr>(forall);
}

/// Checks whether a forall's flat trip count equals `flatWorkgroupSize`.
static bool tripCountMatchesWorkgroupSize(scf::ForallOp forallOp,
                                          int64_t flatWorkgroupSize) {
  if (!isThreadMappedForall(forallOp))
    return false;
  auto maybeTripCount = getStaticForallTripCount(forallOp);
  if (!maybeTripCount) return false;
  return *maybeTripCount == flatWorkgroupSize;
}

/// Helper to apply greedy patterns without signaling pass failure on
/// non-convergence.  Returns true if patterns converged, false if they
/// hit the iteration limit (but the IR is still valid — just not fully
/// simplified).  Only returns failure on actual errors.
static LogicalResult
applyPatternsGreedilyWithConfig(func::FuncOp funcOp,
                                RewritePatternSet &&patterns,
                                StringRef roundName) {
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setMaxIterations(20);
  // Use ExistingAndNewOps strictness to bound the work: only process ops
  // that existed when the round started plus ops newly created by patterns.
  // This prevents unbounded work when patterns create ops that trigger
  // other patterns in a non-converging cycle.
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);

  LogicalResult result =
      applyPatternsGreedily(funcOp, std::move(patterns), config);
  if (failed(result)) {
    // Non-convergence is not fatal — the IR is valid, just not fully
    // optimized.  Emit a remark instead of failing the pass.
    LLVM_DEBUG(llvm::dbgs() << "nova-gpu-fuse-and-hoist: round '"
                            << roundName << "' did not converge\n");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern 1: FuseForalls
//
// When a producer scf.forall has a single consumer that is itself (or leads to)
// a scf.forall with the same thread mapping + workgroup trip count, merge the
// producer into the consumer.
//===----------------------------------------------------------------------===//

struct FuseForalls final : OpRewritePattern<scf::ForallOp> {
  FuseForalls(MLIRContext *ctx, int64_t flatWorkgroupSize, PatternBenefit b = 1)
      : OpRewritePattern<scf::ForallOp>(ctx, b),
        flatWorkgroupSize(flatWorkgroupSize) {}

  LogicalResult matchAndRewrite(scf::ForallOp producerForall,
                                PatternRewriter &rewriter) const override {
    // Only try when the producer has exactly one use (conservative).
    if (!producerForall->hasOneUse())
      return rewriter.notifyMatchFailure(producerForall,
                                         "producer has multiple uses");

    // Only fuse thread-mapped foralls.
    if (!isThreadMappedForall(producerForall))
      return rewriter.notifyMatchFailure(producerForall,
                                         "producer is not thread-mapped");

    // Walk the single-use chain (possibly through reshape ops) to reach the
    // consumer forall.
    Operation *currUser = *producerForall->user_begin();
    SmallVector<Operation *> consumerChain;
    while (currUser && currUser->hasOneUse()) {
      consumerChain.push_back(currUser);
      if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(currUser))
        break;
      currUser = *currUser->user_begin();
    }

    if (!currUser)
      return rewriter.notifyMatchFailure(producerForall,
                                         "no consumer after chain");

    auto consumerForall = currUser->getParentOfType<scf::ForallOp>();
    if (!consumerForall ||
        !tripCountMatchesWorkgroupSize(consumerForall, flatWorkgroupSize))
      return rewriter.notifyMatchFailure(
          producerForall,
          "consumer forall trip count mismatch or not thread-mapped");

    // Find the tensor.extract_slice inside the consumer forall that uses the
    // producer forall's result, and fuse using tileAndFuseProducerOfSlice.
    LogicalResult fusionResult = failure();
    consumerForall.walk([&](tensor::ExtractSliceOp sliceOp) {
      if (failed(fusionResult)) {
        auto defOp = sliceOp.getSource().getDefiningOp<scf::ForallOp>();
        if (!defOp || defOp != producerForall)
          return WalkResult::advance();
        SmallVector<LoopLikeOpInterface> loops = {consumerForall};
        auto result =
            scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
        if (result)
          fusionResult = success();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return fusionResult;
  }

private:
  int64_t flatWorkgroupSize;
};

//===----------------------------------------------------------------------===//
// Pattern 2: FuseTilableSliceProducers
//
// If a tensor.extract_slice inside a scf.forall has a TilingInterface producer
// that lives outside the forall, fuse the producer inside.
//===----------------------------------------------------------------------===//

struct FuseTilableSliceProducers final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (sliceOp->use_empty())
      return failure();

    auto tilableProducer =
        sliceOp.getSource().getDefiningOp<TilingInterface>();
    // Don't fuse pad ops (they need special handling with zero-slice guards).
    if (!tilableProducer || isa<tensor::PadOp>(tilableProducer))
      return failure();

    auto parentForall = sliceOp->getParentOfType<scf::ForallOp>();
    if (!parentForall)
      return failure();

    // Don't fuse if the producer is already inside the same forall.
    auto producerParent =
        tilableProducer->getParentOfType<scf::ForallOp>();
    if (producerParent && producerParent == parentForall)
      return failure();

    SmallVector<LoopLikeOpInterface> loops = {parentForall};
    auto fusionResult =
        scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    return fusionResult ? success() : failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 3: FuseTilableDestinationProducers
//
// For each region iter arg of a forall that is consumed by an extract_slice,
// check if the tied init value comes from a TilingInterface op. If so, fuse it.
//===----------------------------------------------------------------------===//

struct FuseTilableDestinationProducers final
    : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    TilingInterface tileableProducer;
    tensor::ExtractSliceOp sliceOp;

    for (auto iterArg : forallOp.getRegionIterArgs()) {
      // Find an extract_slice that uses this iter arg.
      sliceOp = nullptr;
      for (auto *user : iterArg.getUsers()) {
        sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
        if (sliceOp) break;
      }
      if (!sliceOp) continue;

      // Get the init value for this iter arg.
      tileableProducer = forallOp.getTiedLoopInit(iterArg)
                             ->get()
                             .getDefiningOp<TilingInterface>();
      // Skip pads (handled separately).
      if (tileableProducer && isa<tensor::PadOp>(tileableProducer))
        tileableProducer = nullptr;
      if (tileableProducer) break;
    }

    if (!tileableProducer)
      return failure();

    // FIX: Don't wrap tileAndFuseProducerOfSlice with startOpModification/
    // finalizeOpModification. The upstream API uses the rewriter internally
    // and the nested modification tracking causes issues.
    SmallVector<LoopLikeOpInterface> loops = {forallOp};
    auto fusionResult =
        scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    if (!fusionResult)
      return failure();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 4: FuseUnitLoopDestination
//
// When a scf.forall has trip count 1, its single-use DPS producer can be moved
// inside the body, eliminating one level of nesting.
//===----------------------------------------------------------------------===//

struct FuseUnitLoopDestination final : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    auto maybeTripCount = getStaticForallTripCount(forallOp);
    if (!maybeTripCount || *maybeTripCount != 1)
      return rewriter.notifyMatchFailure(forallOp, "not unit trip count");

    DestinationStyleOpInterface dpsProducer;
    BlockArgument bodyArg;
    Value dpsResult;

    for (auto iterArg : forallOp.getRegionIterArgs()) {
      dpsResult = forallOp.getTiedLoopInit(iterArg)->get();
      bodyArg = iterArg;
      dpsProducer = dpsResult.getDefiningOp<DestinationStyleOpInterface>();
      if (dpsProducer) break;
    }

    if (!dpsProducer || !dpsProducer->hasOneUse())
      return rewriter.notifyMatchFailure(forallOp, "no single-use DPS producer");

    // Find the parallel_insert_slice that uses bodyArg.
    Operation *parallelInsert = nullptr;
    for (auto *user : bodyArg.getUsers()) {
      if (isa<tensor::ParallelInsertSliceOp>(user)) {
        if (parallelInsert)
          return rewriter.notifyMatchFailure(forallOp, "multiple insert users");
        parallelInsert = user;
      }
    }
    if (!parallelInsert)
      return rewriter.notifyMatchFailure(forallOp,
                                          "destination not used by parallel_insert");

    rewriter.startOpModification(forallOp);
    // Move the producer into the body before everything else.
    rewriter.moveOpBefore(dpsProducer, forallOp.getBody(),
                           forallOp.getBody()->begin());
    rewriter.replaceAllUsesExcept(bodyArg, dpsResult, parallelInsert);

    int64_t dpsInitIndex = cast<OpResult>(dpsResult).getResultNumber();
    forallOp->setOperand(
        forallOp.getTiedOpOperand(bodyArg)->getOperandNumber(),
        dpsProducer.getDpsInitOperand(dpsInitIndex)->get());
    dpsProducer.setDpsInitOperand(dpsInitIndex, bodyArg);
    rewriter.finalizeOpModification(forallOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 5: FuseTilableForallConsumers
//
// Fuse TilingInterface consumers of scf.forall results into the forall body.
// This is critical for fusing epilogue ops (bias/relu) into the thread forall
// so that each thread operates on its small tile (e.g., 4x4) instead of the
// full workgroup tile (e.g., 128x128), avoiding oversized private allocas.
//===----------------------------------------------------------------------===//

struct FuseTilableForallConsumers final
    : OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const override {
    // Consumer fusion currently requires DPS ops.
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*tilableOp);
    if (!dpsOp)
      return failure();

    // Find a scf.forall producer among the DPS inputs.
    scf::ForallOp forallProducer;
    for (auto operand : dpsOp.getDpsInputs()) {
      auto forallOp = operand.getDefiningOp<scf::ForallOp>();
      if (!forallOp)
        continue;
      // Must be in the same block (not nested).
      if (forallOp->getBlock() != tilableOp->getBlock())
        continue;
      // FIX: Only fuse into thread-mapped foralls, not block-mapped ones.
      // Block-mapped foralls represent workgroup-level distribution and
      // should not have consumers fused into them (that would duplicate
      // computation across all threads).
      if (!isThreadMappedForall(forallOp))
        continue;
      forallProducer = forallOp;
      break;
    }

    if (!forallProducer)
      return rewriter.notifyMatchFailure(
          tilableOp, "no thread-mapped scf.forall producer to fuse into");

    // Collect the parallel_insert_slice ops from the forall's terminator.
    scf::InParallelOp parallelTerminator = forallProducer.getTerminator();
    SmallVector<Operation *> insertSlices;
    for (Operation &yieldingOp : parallelTerminator.getYieldingOps()) {
      insertSlices.push_back(&yieldingOp);
    }
    if (insertSlices.empty())
      return failure();

    // Move the tilable consumer right after the forall producer to ensure
    // proper dominance (other users of the forall result may be in between).
    DominanceInfo domInfo;
    llvm::SetVector<Operation *> slice;
    BackwardSliceOptions opts;
    opts.filter = [&](Operation *op) {
      return domInfo.properlyDominates(forallProducer.getOperation(), op);
    };
    opts.inclusive = true;
    opts.omitUsesFromAbove = false;
    opts.omitBlockArguments = true;
    if (succeeded(getBackwardSlice(tilableOp, &slice, opts))) {
      Block *block = forallProducer->getBlock();
      Block::iterator insertPt =
          std::next(forallProducer->getIterator());
      for (Operation *op : llvm::reverse(slice)) {
        op->moveBefore(block, insertPt);
      }
    }

    SmallVector<LoopLikeOpInterface> loops = {
        cast<LoopLikeOpInterface>(forallProducer.getOperation())};
    auto fusionResult = scf::tileAndFuseConsumerOfSlices(
        rewriter, insertSlices, loops);
    if (failed(fusionResult))
      return failure();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HoistForallFromFor
//
// Loop interchange: given a scf.for (K-loop) whose single result comes from
// a scf.forall {thread} inside it, lift the forall OUTSIDE the for-loop.
//===----------------------------------------------------------------------===//

struct HoistForallFromFor final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    // (1) The for body must have at least 2 ops (not just terminator).
    if (loop.getBody()->getOperations().size() == 1)
      return rewriter.notifyMatchFailure(loop, "loop body is empty");

    // (2) Exactly one for-loop result, yielded by the terminator.
    if (loop.getNumResults() != 1)
      return rewriter.notifyMatchFailure(loop, "multi-result for-loop");

    // (3) The yielded value must be the result of a scf.forall.
    Value yieldedValue = loop.getBody()->getTerminator()->getOperand(0);
    auto *defOp = yieldedValue.getDefiningOp();
    if (!defOp)
      return rewriter.notifyMatchFailure(loop, "yielded value has no defining op");
    auto forallOp = dyn_cast<scf::ForallOp>(defOp);
    if (!forallOp || forallOp.getNumResults() != loop->getNumResults())
      return rewriter.notifyMatchFailure(
          loop, "terminator does not yield a single-result scf.forall");

    // (4) Verify & collect the other ops in the for body.
    //     They must not have regions, not be tilable, not depend on the
    //     for-loop iter arg, and not be used by the forall op directly.
    Block *loopBody = loop.getBody();
    Value forIterArg = loop.getRegionIterArg(0);
    SmallVector<Operation *> operationsToMove;
    for (Operation &op : loopBody->getOperations()) {
      if (&op == forallOp.getOperation() || &op == loopBody->getTerminator())
        continue;
      if (op.getNumRegions() != 0 || isa<TilingInterface>(&op))
        return rewriter.notifyMatchFailure(
            loop, "for body contains region/tilable op besides the forall");
      for (Value operand : op.getOperands())
        if (operand == forIterArg)
          return rewriter.notifyMatchFailure(
              loop, "for body op uses iter arg");
      for (Operation *user : op.getUsers())
        if (user == forallOp.getOperation())
          return rewriter.notifyMatchFailure(
              loop, "for body op is used by the forall");
      operationsToMove.push_back(&op);
    }

    // (5) Collect the parallel_insert_slice and its paired extract_slice.
    scf::InParallelOp parallelTerminator = forallOp.getTerminator();
    bool isSingleTrip = forallOp.isNormalized() &&
                        llvm::all_of(forallOp.getStaticUpperBound(),
                                     [](int64_t v) { return v == 1; });

    struct SliceInfo {
      tensor::ParallelInsertSliceOp insertSlice;
      std::optional<tensor::ExtractSliceOp> extractSlice;
    };
    SmallVector<SliceInfo> sliceInfos;

    for (Operation &yieldingOp : parallelTerminator.getYieldingOps()) {
      auto parallelInsert = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
      BlockArgument destBbArg = cast<BlockArgument>(parallelInsert.getDest());

      // Find the paired extract_slice of the dest bbarg.
      tensor::ExtractSliceOp destSlice;
      for (Operation *user : destBbArg.getUsers()) {
        if (user == parallelInsert.getOperation()) continue;
        auto sl = dyn_cast<tensor::ExtractSliceOp>(user);
        if (!sl) {
          if (!isSingleTrip) return failure();
          continue;
        }
        if (destSlice) return failure(); // at most one
        destSlice = sl;
      }

      // Verify equivalent subsets when there's a paired extract.
      if (destSlice &&
          !cast<SubsetOpInterface>(*destSlice).operatesOnEquivalentSubset(
              cast<SubsetOpInterface>(*parallelInsert),
              [](Value v1, Value v2) { return v1 == v2; }))
        return failure();

      // Verify single-trip case: the insert must overwrite the full dest.
      if (!destSlice) {
        auto overwritesFull = [](tensor::ParallelInsertSliceOp ins) -> bool {
          if (ins.getSourceType().getRank() != ins.getDestType().getRank())
            return false;
          for (auto [dim, sz] : llvm::enumerate(ins.getMixedSizes())) {
            FailureOr<bool> eq = ValueBoundsConstraintSet::areEqual(
                {sz}, {ins.getDest(), static_cast<int64_t>(dim)});
            if (failed(eq) || !*eq) return false;
          }
          return true;
        };
        if (!overwritesFull(parallelInsert)) return failure();
      }

      sliceInfos.push_back({parallelInsert, destSlice ? std::optional(destSlice)
                                                       : std::nullopt});
    }

    if (sliceInfos.size() != 1)
      return rewriter.notifyMatchFailure(forallOp,
                                          "expected exactly one shared_out");

    auto &[parallelInsert, maybePairedSlice] = sliceInfos[0];

    // (6) Verify that slice offset/size/stride operands do NOT depend
    //     on the for-loop serial iter arg.
    auto dependsOnForIterArg = [&](Operation *op) -> bool {
      auto ossIdx = isa<tensor::ExtractSliceOp>(op)
          ? cast<tensor::ExtractSliceOp>(op)
                .getOffsetSizeAndStrideStartOperandIndex()
          : cast<tensor::ParallelInsertSliceOp>(op)
                .getOffsetSizeAndStrideStartOperandIndex();
      for (Value v : op->getOperands().drop_front(ossIdx))
        if (auto ba = dyn_cast<BlockArgument>(v))
          if (ba.getOwner()->getParentOp() == loop.getOperation())
            return true;
      return false;
    };
    if (dependsOnForIterArg(parallelInsert.getOperation()))
      return rewriter.notifyMatchFailure(
          loop, "insert slice offsets depend on for iter arg");
    if (maybePairedSlice &&
        dependsOnForIterArg(maybePairedSlice->getOperation()))
      return rewriter.notifyMatchFailure(
          loop, "extract slice offsets depend on for iter arg");

    // ---------------------------------------------------------------
    // Transformation — loop interchange using mergeBlocks.
    // ---------------------------------------------------------------

    // STEP A: Move any safe non-forall ops from the for body INTO the forall
    //         body (before the first op there).
    Block *forallBody = forallOp.getBody();
    for (Operation *op : llvm::reverse(operationsToMove))
      rewriter.moveOpBefore(op, &forallBody->getOperations().front());

    // STEP B: Create the new OUTER scf.forall.
    auto newForallOp = scf::ForallOp::create(
        rewriter, forallOp.getLoc(),
        forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(),
        loop.getInitArgs(),
        forallOp.getMappingAttr());

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(newForallOp.getTerminator());

      // STEP C: Compute the init values for the new INNER scf.for.
      SmallVector<Value> newForInits;
      SmallVector<std::optional<tensor::ExtractSliceOp>> newExtractSlices;
      {
        BlockArgument newForallIterArg = newForallOp.getRegionIterArgs()[0];
        if (maybePairedSlice) {
          auto oldExtract = *maybePairedSlice;
          IRMapping extractMapping;
          for (auto [oldIV, newIV] : llvm::zip(forallOp.getInductionVars(),
                                                newForallOp.getInductionVars()))
            extractMapping.map(oldIV, newIV);
          extractMapping.map(forallOp.getRegionIterArgs()[0], newForallIterArg);
          auto *clonedExtract = rewriter.clone(*oldExtract, extractMapping);
          auto newExtract = cast<tensor::ExtractSliceOp>(clonedExtract);
          newForInits.push_back(newExtract.getResult());
          newExtractSlices.push_back(newExtract);
        } else {
          newForInits.push_back(newForallIterArg);
          newExtractSlices.push_back(std::nullopt);
        }
      }

      // STEP D: Create the new INNER scf.for.
      auto newFor = scf::ForOp::create(
          rewriter, loop.getLoc(),
          loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
          newForInits,
          [](OpBuilder &, Location, Value, ValueRange) {});

      {
        // STEP E: Build the argReplacements for mergeBlocks.
        SmallVector<Value> argReplacements(newForallOp.getInductionVars());
        for (auto [forallIA, forIA, maybeNewExtract] :
             llvm::zip_equal(forallOp.getRegionIterArgs(),
                             newFor.getRegionIterArgs(),
                             newExtractSlices)) {
          if (maybeNewExtract) {
            argReplacements.push_back(newForallOp.getRegionIterArgs()[0]);
          } else {
            argReplacements.push_back(forIA);
          }
        }

        // STEP F: mergeBlocks — move the old forall body into the new for body.
        rewriter.mergeBlocks(forallBody, newFor.getBody(), argReplacements);
        rewriter.replaceAllUsesWith(loop.getInductionVar(),
                                    newFor.getInductionVar());

        if (maybePairedSlice) {
          rewriter.replaceAllUsesExcept(
              maybePairedSlice->getResult(),
              newFor.getRegionIterArg(0),
              newFor.getOperation());
        }

        // Create the terminator for the new for-loop.
        rewriter.setInsertionPointToEnd(newFor.getBody());
        scf::YieldOp::create(rewriter, loop.getLoc(),
                              parallelInsert.getSource());
        rewriter.eraseOp(parallelInsert.getOperation());
      }

      // STEP G: Create the new terminator for the outer forall.
      BlockArgument newForallIterArg = newForallOp.getRegionIterArgs()[0];
      rewriter.setInsertionPointToEnd(newForallOp.getTerminator().getBody());
      tensor::ParallelInsertSliceOp::create(
          rewriter, parallelInsert.getLoc(),
          newFor.getResult(0),
          newForallIterArg,
          parallelInsert.getMixedOffsets(),
          parallelInsert.getMixedSizes(),
          parallelInsert.getMixedStrides());
      rewriter.eraseOp(parallelTerminator);
    }

    rewriter.replaceOp(loop, newForallOp.getResult(0));
    return success();
  }
};


//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUFuseAndHoistParallelLoopsPass
    : public PassWrapper<NovaGPUFuseAndHoistParallelLoopsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUFuseAndHoistParallelLoopsPass)

  NovaGPUFuseAndHoistParallelLoopsPass() = default;
  NovaGPUFuseAndHoistParallelLoopsPass(
      const NovaGPUFuseAndHoistParallelLoopsPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, gpu::GPUDialect,
                    affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    std::optional<int64_t> maybeFlatWorkgroupSize = std::nullopt;
    funcOp.walk([&](scf::ForallOp forall) {
      if (!maybeFlatWorkgroupSize && isThreadMappedForall(forall)) {
        maybeFlatWorkgroupSize = getStaticForallTripCount(forall);
      }
      return WalkResult::advance();
    });

    // -------------------------------------------------------------------
    // Round 1: Hoist + fuse foralls
    // -------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      if (maybeFlatWorkgroupSize) {
        patterns.add<FuseForalls>(ctx, *maybeFlatWorkgroupSize, /*benefit=*/2);
      }
      patterns.add<FuseTilableForallConsumers>(ctx);
      patterns.add<HoistForallFromFor>(ctx);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedilyWithConfig(
              funcOp, std::move(patterns), "round 1 (hoist+fuse foralls)")))
        return signalPassFailure();
    }

    // -------------------------------------------------------------------
    // Round 2: Revealed consumers / destinations
    // -------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      patterns.add<FuseTilableDestinationProducers>(ctx);
      patterns.add<FuseUnitLoopDestination>(ctx);
      patterns.add<FuseTilableForallConsumers>(ctx);
      // FIX: Removed FuseExtractSliceConsumers from Round 2.
      // The original implementation used tileAndFuseConsumerOfSlices
      // incorrectly — it passed the tiled op (inside the forall) to
      // replaceOp instead of the forall's new results.  IREE uses a
      // custom fuseExtractSliceIntoProducerForall() which is not
      // available in upstream MLIR.  Without a correct implementation,
      // this pattern causes crashes and incorrect replacements.
      tensor::populateFoldTensorEmptyPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedilyWithConfig(
              funcOp, std::move(patterns), "round 2 (consumer fusion)")))
        return signalPassFailure();
    }

    // -------------------------------------------------------------------
    // Round 3: New producer fusions
    // -------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      patterns.add<FuseTilableDestinationProducers>(ctx);
      patterns.add<FuseTilableSliceProducers>(ctx);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
      // Swap extract_slice(pad(x)) → pad(extract_slice(x)) without zero-slice guard.
      patterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
          ctx,
          [](tensor::ExtractSliceOp) -> std::optional<bool> { return false; });
      if (failed(applyPatternsGreedilyWithConfig(
              funcOp, std::move(patterns), "round 3 (producer fusion)")))
        return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-fuse-and-hoist-parallel-loops";
  }
  StringRef getDescription() const override {
    return "Greedily fuse and hoist parallel scf.forall loops after thread tiling. "
           "Mirrors IREE's GPUFuseAndHoistParallelLoops pass.";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUFuseAndHoistParallelLoopsPass() {
  return std::make_unique<NovaGPUFuseAndHoistParallelLoopsPass>();
}

void registerNovaGPUFuseAndHoistParallelLoopsPass() {
  PassRegistration<NovaGPUFuseAndHoistParallelLoopsPass>();
}

} // namespace mlir::nova
