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
//    • FuseExtractSliceConsumers       — fuse extract_slice consumers.
//    • FuseCollapseShapeConsumers      — fuse collapse_shape into producer forall.
//
//  Round 3 — New producer fusions:
//    • FuseTilableSliceProducers       — fuse tilable producers of slice ops.
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

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if the forall op is normalized (lb == 0, step == 1 for all dims).
static bool isNormalized(scf::ForallOp forall) {
  // A forall is normalized when every lower bound is 0 and every step is 1.
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

/// Checks whether a forall's flat trip count equals `flatWorkgroupSize`.
/// For thread-mapped foralls: trip count must equal flatWorkgroupSize directly.
static bool tripCountMatchesWorkgroupSize(scf::ForallOp forallOp,
                                          int64_t flatWorkgroupSize) {
  // Only thread-mapped foralls participate in this check.
  if (!forallHasMappingType<gpu::GPUThreadMappingAttr>(forallOp))
    return false;
  auto maybeTripCount = getStaticForallTripCount(forallOp);
  if (!maybeTripCount) return false;
  return *maybeTripCount == flatWorkgroupSize;
}

//===----------------------------------------------------------------------===//
// Pattern 1: FuseForalls
//
// When a producer scf.forall has a single consumer that is itself (or leads to)
// a scf.forall with the same thread mapping + workgroup trip count, merge the
// producer into the consumer.
//
// Mirrors IREE's FuseForalls pattern.
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

    // The final currUser must live inside (or be) the consumer forall.
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
    // producer forall's result, and the corresponding tensor.parallel_insert_slice
    // inside the producer forall.
    //
    // We use MLIR's scf::tileAndFuseProducerOfSlice to do the actual fusion.
    // Walk all extract_slice ops inside the consumer forall that source from
    // the producer forall's results.
    LogicalResult fusionResult = failure();
    consumerForall.walk([&](tensor::ExtractSliceOp sliceOp) {
      if (failed(fusionResult)) {
        // Check that this slice sources from the producer forall.
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
//
// Mirrors IREE's FuseTilableSliceProducers.
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
//
// Mirrors IREE's FuseTilableDestinationProducers.
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

    SmallVector<LoopLikeOpInterface> loops = {forallOp};
    rewriter.startOpModification(forallOp);
    auto fusionResult =
        scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    if (!fusionResult) {
      rewriter.cancelOpModification(forallOp);
      return failure();
    }
    rewriter.finalizeOpModification(forallOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 4: FuseUnitLoopDestination
//
// When a scf.forall has trip count 1, its single-use DPS producer can be moved
// inside the body, eliminating one level of nesting.
//
// Mirrors IREE's FuseUnitLoopDestination.
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
// Pattern 5: FuseExtractSliceConsumers
//
// If an extract_slice consumes a scf.forall result, fuse it into the forall.
// Mirrors IREE's FuseExtractSliceConsumers.
//===----------------------------------------------------------------------===//

struct FuseExtractSliceConsumers final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    // Source must be a scf.forall result.
    auto forallOp =
        extractSliceOp.getSource().getDefiningOp<scf::ForallOp>();
    if (!forallOp)
      return rewriter.notifyMatchFailure(extractSliceOp, "no forall producer");

    // Find the parallel_insert_slice inside the forall that corresponds to the
    // result we're slicing.
    OpResult forallResult =
        cast<OpResult>(extractSliceOp.getSource());
    BlockArgument tiedArg = forallOp.getTiedBlockArgument(
        forallOp.getTiedOpOperand(forallResult));

    // Walk the combining ops for this block arg to find a parallel_insert_slice
    // whose result covers the extracted region, then fuse by inlining.
    // We use scf::tileAndFuseConsumerOfSlices as the canonical fusion utility.
    SmallVector<Operation *> insertSlices(
        llvm::map_range(forallOp.getCombiningOps(tiedArg),
                        [](Operation *op) { return op; }));
    if (insertSlices.empty())
      return failure();

    SmallVector<LoopLikeOpInterface> loops = {forallOp};
    auto fusionResult = scf::tileAndFuseConsumerOfSlices(
        rewriter,
        llvm::to_vector(llvm::map_range(insertSlices, [](Operation *op) {
          return op;
        })),
        loops);
    if (failed(fusionResult))
      return failure();

    rewriter.replaceOp(
        fusionResult->origConsumerOperands.front()->getOwner(),
        fusionResult->tiledOps.front());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HoistForallFromFor
//
// Loop interchange: given a scf.for (K-loop) whose single result comes from
// a scf.forall {thread} inside it, lift the forall OUTSIDE the for-loop:
//
//  Before:
//    %result = scf.for %k = lb..ub step s iter_args(%acc = init) {
//      %tile = scf.forall (%i,%j) shared_outs(%out = %acc) {
//        %s = extract_slice %out[%i,%j][4,4]   // acc read
//        ... uses %k and %s ...
//        parallel_insert_slice %new -> %out[%i,%j]
//      }
//      scf.yield %tile
//    }
//
//  After (loop interchange — K-loop moves INSIDE forall):
//    %result = scf.forall (%i,%j) shared_outs(%out = init) {
//      %s = extract_slice %out[%i,%j][4,4]     // hoisted outside new K-loop
//      %tile2 = scf.for %k = lb..ub step s iter_args(%acc2 = %s) {
//        ... body (old forall body minus the extract/insert of the dest) ...
//        scf.yield <insert source>
//      }
//      parallel_insert_slice %tile2 -> %out[%i,%j]
//    }
//
// Faithfully mirrors IREE's HoistForallFromFor in Transforms.cpp line 1008,
// using rewriter.mergeBlocks() to physically move ops (not clone them) so that
// SSA values from the old forall region are properly replaced.
//===----------------------------------------------------------------------===//

struct HoistForallFromFor final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    // ---------------------------------------------------------------
    // (1) The for body must have at least 2 ops (not just terminator).
    // ---------------------------------------------------------------
    if (loop.getBody()->getOperations().size() == 1)
      return rewriter.notifyMatchFailure(loop, "loop body is empty");

    // ---------------------------------------------------------------
    // (2) Exactly one for-loop result, yielded by the terminator.
    // ---------------------------------------------------------------
    if (loop.getNumResults() != 1)
      return rewriter.notifyMatchFailure(loop, "multi-result for-loop");

    // ---------------------------------------------------------------
    // (3) The yielded value must be the result of a scf.forall.
    // ---------------------------------------------------------------
    auto forallOp = dyn_cast<scf::ForallOp>(
        loop.getBody()->getTerminator()->getOperand(0).getDefiningOp());
    if (!forallOp || forallOp.getNumResults() != loop->getNumResults())
      return rewriter.notifyMatchFailure(
          loop, "terminator does not yield a single-result scf.forall");

    // ---------------------------------------------------------------
    // (4) Verify & collect the other ops in the for body (not the forall,
    //     not the terminator).  Mirror IREE: they must not have regions,
    //     not be tilable, not depend on the for-loop iter arg, and not be
    //     used by the forall op directly.
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // (5) Collect the parallel_insert_slice and its paired extract_slice.
    //     Mirrors IREE Steps 2-3.
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // (6) Verify that slice offset/size/stride operands do NOT depend
    //     on the for-loop serial iter arg.
    // ---------------------------------------------------------------
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
    //
    // STEP A: Move any safe non-forall ops from the for body INTO the forall
    //         body (before the first op there).
    // ---------------------------------------------------------------
    Block *forallBody = forallOp.getBody();
    for (Operation *op : llvm::reverse(operationsToMove))
      rewriter.moveOpBefore(op, &forallBody->getOperations().front());

    // ---------------------------------------------------------------
    // STEP B: Create the new OUTER scf.forall with for.getInitArgs() as
    //         shared_outs (same loop bounds and mapping as old forall).
    // ---------------------------------------------------------------
    auto newForallOp = scf::ForallOp::create(
        rewriter, forallOp.getLoc(),
        forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(),
        loop.getInitArgs(),          // <-- old FOR's init, not the old forall's
        forallOp.getMappingAttr());

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(newForallOp.getTerminator());

      // ---------------------------------------------------------------
      // STEP C: Compute the init values for the new INNER scf.for.
      //   - If there's a paired extract_slice: the new for-loop's init is
      //     the result of an extract_slice from the new forall's iter arg.
      //   - Single-trip case: just use the new forall's iter arg directly.
      // ---------------------------------------------------------------
      SmallVector<Value> newForInits;
      SmallVector<std::optional<tensor::ExtractSliceOp>> newExtractSlices;
      {
        BlockArgument newForallIterArg = newForallOp.getRegionIterArgs()[0];
        if (maybePairedSlice) {
          auto oldExtract = *maybePairedSlice;
          // Rebuild extract_slice with new forall's iter arg as source.
          // The offsets/sizes/strides are the same (they don't depend on the
          // old for-loop's iter arg, verified above).  We clone the extract
          // into the new forall body, remapping its source.
          IRMapping extractMapping;
          // Map old forall IVs to new forall IVs so any IV-dependent offsets
          // are correctly updated.
          for (auto [oldIV, newIV] : llvm::zip(forallOp.getInductionVars(),
                                                newForallOp.getInductionVars()))
            extractMapping.map(oldIV, newIV);
          // Map old forall iter arg → new forall iter arg.
          extractMapping.map(forallOp.getRegionIterArgs()[0], newForallIterArg);
          auto *clonedExtract = rewriter.clone(*oldExtract, extractMapping);
          auto newExtract = cast<tensor::ExtractSliceOp>(clonedExtract);
          newForInits.push_back(newExtract.getResult());
          newExtractSlices.push_back(newExtract);
        } else {
          // Single-trip: use the iter arg directly.
          newForInits.push_back(newForallIterArg);
          newExtractSlices.push_back(std::nullopt);
        }
      }

      // ---------------------------------------------------------------
      // STEP D: Create the new INNER scf.for.
      // ---------------------------------------------------------------
      auto newFor = scf::ForOp::create(
          rewriter, loop.getLoc(),
          loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
          newForInits,
          [](OpBuilder &, Location, Value, ValueRange) {});

      {
        // ---------------------------------------------------------------
        // STEP E: Build the argReplacements for mergeBlocks:
        //   position 0..(nIVs-1) : old forall IVs → new forall IVs
        //   position nIVs..      : old forall iter arg → new for iter arg
        //                          (or new forall iter arg for single-trip)
        // ---------------------------------------------------------------
        SmallVector<Value> argReplacements(newForallOp.getInductionVars());
        for (auto [forallIA, forIA, maybeNewExtract] :
             llvm::zip_equal(forallOp.getRegionIterArgs(),
                             newFor.getRegionIterArgs(),
                             newExtractSlices)) {
          if (maybeNewExtract) {
            // Old forall iter arg → new FORALL iter arg (so uses of the
            // forall iter arg inside the body now read from the forall's slot,
            // the extract_slice provides the per-trip init).
            argReplacements.push_back(newForallOp.getRegionIterArgs()[0]);
          } else {
            argReplacements.push_back(forIA);
          }
        }

        // ---------------------------------------------------------------
        // STEP F: mergeBlocks — move the old forall body into the new for
        //         body with argument substitution.
        //         Also replace all uses of the old for-loop's IV with the
        //         new for-loop's IV.
        // ---------------------------------------------------------------
        rewriter.mergeBlocks(forallBody, newFor.getBody(), argReplacements);
        rewriter.replaceAllUsesWith(loop.getInductionVar(),
                                    newFor.getInductionVar());

        // After mergeBlocks, the paired extract_slice (now in new for body)
        // and the new forInits extract_slice both exist. Replace uses of the
        // old paired-extract result (now inside the for body) with the new
        // for iter arg, except in the new for body itself
        // (or we'd create a self-referential cycle).
        if (maybePairedSlice) {
          // The old extract_slice was moved into new for body by mergeBlocks.
          // Replace its uses (the linalg.matmul outs operand etc.) with the
          // new for's iter arg.
          rewriter.replaceAllUsesExcept(
              maybePairedSlice->getResult(),
              newFor.getRegionIterArg(0),
              newFor.getOperation());
        }

        // Create the terminator for the new for-loop: yield the source of
        // the parallel_insert_slice.
        rewriter.setInsertionPointToEnd(newFor.getBody());
        scf::YieldOp::create(rewriter, loop.getLoc(),
                              parallelInsert.getSource());
        // Erase the old parallel_insert_slice (it was moved into new for body
        // by mergeBlocks and is no longer needed).
        rewriter.eraseOp(parallelInsert.getOperation());
      }

      // ---------------------------------------------------------------
      // STEP G: Create the new terminator for the outer forall: insert
      //         the for-loop's result back into the forall's shared_out.
      // ---------------------------------------------------------------
      BlockArgument newForallIterArg = newForallOp.getRegionIterArgs()[0];
      rewriter.setInsertionPointToEnd(newForallOp.getTerminator().getBody());
      tensor::ParallelInsertSliceOp::create(
          rewriter, parallelInsert.getLoc(),
          newFor.getResult(0),
          newForallIterArg,
          parallelInsert.getMixedOffsets(),
          parallelInsert.getMixedSizes(),
          parallelInsert.getMixedStrides());
      // Erase the now-dead old forall terminator block (mergeBlocks already
      // moved its ops out; the InParallelOp itself is now empty).
      rewriter.eraseOp(parallelTerminator);
    }

    // Replace the original for-loop with the result of the new forall.
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

    // Determine the flat workgroup size for FuseForalls validation.
    // Look for the first thread-mapped scf.forall with a known trip count
    // and use it as the reference workgroup size.
    // (In a complete implementation, we would read from a gpu.func attribute.)
    std::optional<int64_t> maybeFlatWorkgroupSize = std::nullopt;
    funcOp.walk([&](scf::ForallOp forall) {
      if (!maybeFlatWorkgroupSize &&
          forallHasMappingType<gpu::GPUThreadMappingAttr>(forall)) {
        maybeFlatWorkgroupSize = getStaticForallTripCount(forall);
      }
      return WalkResult::advance();
    });

    // -----------------------------------------------------------------------
    // Round 1: Hoist + fuse foralls
    // -----------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      if (maybeFlatWorkgroupSize) {
        patterns.add<FuseForalls>(ctx, *maybeFlatWorkgroupSize, /*benefit=*/2);
      }
      patterns.add<HoistForallFromFor>(ctx);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp->emitOpError(
            "nova-gpu-fuse-and-hoist: failed in round 1 (hoist+fuse foralls)");
        return signalPassFailure();
      }
    }

    // -----------------------------------------------------------------------
    // Round 2: Revealed consumers / destinations
    // -----------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      patterns.add<FuseTilableDestinationProducers>(ctx);
      patterns.add<FuseUnitLoopDestination>(ctx);
      patterns.add<FuseExtractSliceConsumers>(ctx);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp->emitOpError(
            "nova-gpu-fuse-and-hoist: failed in round 2 (consumer fusion)");
        return signalPassFailure();
      }
    }

    // -----------------------------------------------------------------------
    // Round 3: New producer fusions
    // -----------------------------------------------------------------------
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
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp->emitOpError(
            "nova-gpu-fuse-and-hoist: failed in round 3 (producer fusion)");
        return signalPassFailure();
      }
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
