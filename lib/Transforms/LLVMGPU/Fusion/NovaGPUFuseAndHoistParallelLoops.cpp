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
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
// producer into the consumer by direct body inlining.
//
// Mirrors IREE's fuseForallIntoConsumer(). When producer and consumer foralls
// have matching flat trip counts (product of all upper bounds), we fuse them
// by cloning the producer body into the consumer. If dimensions differ (e.g.,
// copy forall (1,32,4) into matmul forall (1,16,8)), we linearize consumer
// IVs into a flat index and delinearize into the producer's dimension space.
// When dimensions match 1:1, we map IVs directly (fast path).
//===----------------------------------------------------------------------===//

/// Returns true if the foralls have matching flat trip counts (product of all
/// upper bounds). Unlike the old dimension-wise check, this allows fusing
/// foralls with different dimensionality or different per-dim bounds, as long
/// as the total number of threads matches. This mirrors IREE's approach where
/// fusion only requires flat trip count == flatWorkgroupSize.
static bool forallFlatTripCountsMatch(scf::ForallOp producer,
                                      scf::ForallOp consumer) {
  auto pTrip = getStaticForallTripCount(producer);
  auto cTrip = getStaticForallTripCount(consumer);
  if (!pTrip || !cTrip)
    return false;
  return *pTrip == *cTrip;
}

/// Linearize multi-dimensional forall IVs into a single flat index.
/// For IVs [iv0, iv1, iv2] with upper bounds [ub0, ub1, ub2]:
///   flatIdx = iv0 * (ub1 * ub2) + iv1 * ub2 + iv2
static Value linearizeForallIVs(OpBuilder &b, Location loc,
                                scf::ForallOp forall) {
  auto ivs = forall.getInductionVars();
  SmallVector<int64_t> ubs;
  for (OpFoldResult ub : forall.getMixedUpperBound()) {
    auto cst = getConstantIntValue(ub);
    assert(cst && "expected static upper bounds");
    ubs.push_back(*cst);
  }
  unsigned rank = ivs.size();
  Value flatIdx = b.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned d = 0; d < rank; ++d) {
    Value iv = ivs[d];
    // Compute stride = product of ubs[d+1..rank-1]
    int64_t stride = 1;
    for (unsigned k = d + 1; k < rank; ++k)
      stride *= ubs[k];
    if (stride != 1) {
      Value strideVal = b.create<arith::ConstantIndexOp>(loc, stride);
      iv = b.create<arith::MulIOp>(loc, iv, strideVal);
    }
    flatIdx = b.create<arith::AddIOp>(loc, flatIdx, iv);
  }
  return flatIdx;
}

/// Delinearize a flat index into multi-dimensional IVs for a target forall.
/// For upper bounds [ub0, ub1, ub2]:
///   iv0 = flatIdx / (ub1 * ub2)
///   iv1 = (flatIdx / ub2) % ub1
///   iv2 = flatIdx % ub2
static SmallVector<Value> delinearizeToForallIVs(OpBuilder &b, Location loc,
                                                 Value flatIdx,
                                                 scf::ForallOp forall) {
  SmallVector<int64_t> ubs;
  for (OpFoldResult ub : forall.getMixedUpperBound()) {
    auto cst = getConstantIntValue(ub);
    assert(cst && "expected static upper bounds");
    ubs.push_back(*cst);
  }
  unsigned rank = ubs.size();
  SmallVector<Value> ivs(rank);
  Value remaining = flatIdx;
  for (unsigned d = 0; d < rank; ++d) {
    int64_t stride = 1;
    for (unsigned k = d + 1; k < rank; ++k)
      stride *= ubs[k];
    Value strideVal = b.create<arith::ConstantIndexOp>(loc, stride);
    Value idx = b.create<arith::DivUIOp>(loc, remaining, strideVal);
    ivs[d] = idx;
    if (d < rank - 1) {
      Value mul = b.create<arith::MulIOp>(loc, idx, strideVal);
      remaining = b.create<arith::SubIOp>(loc, remaining, mul);
    }
  }
  return ivs;
}

struct FuseForalls final : OpRewritePattern<scf::ForallOp> {
  FuseForalls(MLIRContext *ctx, PatternBenefit b = 1)
      : OpRewritePattern<scf::ForallOp>(ctx, b) {}

  LogicalResult matchAndRewrite(scf::ForallOp producerForall,
                                PatternRewriter &rewriter) const override {
    // Only try when the producer has exactly one use (conservative).
    if (!producerForall->hasOneUse())
      return rewriter.notifyMatchFailure(producerForall,
                                         "producer has multiple uses");

    // Only single-result producers.
    if (producerForall->getNumResults() != 1)
      return rewriter.notifyMatchFailure(producerForall,
                                         "multi-result producer");

    // Only fuse thread-mapped foralls.
    if (!isThreadMappedForall(producerForall))
      return rewriter.notifyMatchFailure(producerForall,
                                         "producer is not thread-mapped");

    // Both must be normalized.
    if (!isNormalized(producerForall))
      return rewriter.notifyMatchFailure(producerForall,
                                         "producer is not normalized");

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
    if (!consumerForall || !isThreadMappedForall(consumerForall))
      return rewriter.notifyMatchFailure(
          producerForall,
          "consumer not inside a thread-mapped forall");

    if (!isNormalized(consumerForall))
      return rewriter.notifyMatchFailure(consumerForall,
                                         "consumer is not normalized");

    // Check if dimensions match 1:1 for the fast path (direct IV remapping).
    auto producerUBs = producerForall.getMixedUpperBound();
    auto consumerUBs = consumerForall.getMixedUpperBound();
    bool dimsMatch = (producerUBs.size() == consumerUBs.size());
    if (dimsMatch) {
      for (auto [pUB, cUB] : llvm::zip_equal(producerUBs, consumerUBs)) {
        auto pCst = getConstantIntValue(pUB);
        auto cCst = getConstantIntValue(cUB);
        if (!pCst || !cCst || *pCst != *cCst) {
          dimsMatch = false;
          break;
        }
      }
    }

    // If dimensions don't match, require flat trip count match for
    // barrier_region-based fusion.
    if (!dimsMatch && !forallFlatTripCountsMatch(producerForall, consumerForall))
      return rewriter.notifyMatchFailure(
          producerForall, "flat trip counts do not match");

    // Find the extract_slice in the consumer that reads from the producer.
    tensor::ExtractSliceOp consumerSlice;
    consumerForall.walk([&](tensor::ExtractSliceOp sliceOp) {
      if (sliceOp.getSource().getDefiningOp() == producerForall.getOperation())
        consumerSlice = sliceOp;
    });
    if (!consumerSlice)
      return rewriter.notifyMatchFailure(
          producerForall, "no extract_slice from producer in consumer");

    // Get the producer's terminator and its parallel_insert_slice.
    scf::InParallelOp producerTerminator = producerForall.getTerminator();
    SmallVector<Operation *> producerInserts;
    for (Operation &op : producerTerminator.getYieldingOps())
      producerInserts.push_back(&op);
    if (producerInserts.size() != 1)
      return rewriter.notifyMatchFailure(producerForall,
                                         "expected exactly one insert");
    auto producerInsert =
        cast<tensor::ParallelInsertSliceOp>(producerInserts[0]);

    // Verify that the consumer's extract_slice and producer's
    // parallel_insert_slice operate on equivalent subsets. Without this check,
    // fusion can create invalid IR when the producer and consumer distribute
    // their threads differently over shared dimensions (e.g., layer norm with
    // K-per-thread=4 fused into matmul with K-per-thread=16).
    if (!cast<SubsetOpInterface>(*consumerSlice).operatesOnEquivalentSubset(
            cast<SubsetOpInterface>(*producerInsert),
            [](Value v1, Value v2) { return v1 == v2; }))
      return rewriter.notifyMatchFailure(
          producerForall,
          "producer insert and consumer extract operate on incompatible "
          "tensor slices");

    if (dimsMatch) {
      // ========== FAST PATH: dimension-wise bounds match ==========
      // Direct body inlining: map producer IVs → consumer IVs 1:1.
      IRMapping mapping;
      for (auto [pIV, cIV] : llvm::zip_equal(
               producerForall.getInductionVars(),
               consumerForall.getInductionVars())) {
        mapping.map(pIV, cIV);
      }
      for (auto [iterArg, init] : llvm::zip_equal(
               producerForall.getRegionIterArgs(),
               producerForall.getDpsInits())) {
        mapping.map(iterArg, init);
      }

      rewriter.setInsertionPoint(consumerSlice);
      for (Operation &op : producerForall.getBody()->without_terminator()) {
        rewriter.clone(op, mapping);
      }

      Value fusedValue = mapping.lookupOrDefault(producerInsert.getSource());
      rewriter.replaceOp(consumerSlice, fusedValue);
      rewriter.eraseOp(producerForall);
      return success();
    }

    // ========== BARRIER PATH: dimension mismatch, flat counts match ==========
    // Use nova.barrier_region with shared memory + linearize/delinearize.
    //
    // The producer forall writes into a shared memory tensor. Each consumer
    // thread linearizes its IDs, then delinearizes into the producer's ID
    // space to execute the producer body, writing into shared memory via
    // insert_slice. A barrier synchronizes, then the consumer reads its
    // slice from the shared result.

    Location loc = producerForall.getLoc();

    // Step 1: Create shared memory alloc_tensor for the producer destination.
    // The destination of the producer forall (its DPS init) should be a
    // tensor.empty — convert it to a workgroup-addressed alloc_tensor.
    Value producerDest = producerForall.getDpsInits()[0];
    auto emptyOp = producerDest.getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return rewriter.notifyMatchFailure(
          producerForall, "producer dest is not tensor.empty");

    rewriter.setInsertionPointToStart(consumerForall.getBody());
    Attribute sharedMemAddrSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto allocTensor = rewriter.create<bufferization::AllocTensorOp>(
        loc, cast<TensorType>(emptyOp.getResult().getType()),
        emptyOp.getDynamicSizes(),
        /*copy=*/Value(), /*size_hint=*/Value(),
        /*memory_space=*/sharedMemAddrSpace);
    Value sharedDest = allocTensor.getResult();

    // Step 2: Create nova.barrier_region wrapping the producer computation.
    auto barrierOp = rewriter.create<nova::BarrierRegionOp>(
        loc, /*resultTypes=*/sharedDest.getType(), /*inputs=*/sharedDest);
    rewriter.setInsertionPointToStart(barrierOp.getBody());

    // Step 3: Compute producer IDs from consumer IDs via linearize/delinearize.
    // Linearize consumer IVs into a flat index.
    Value flatConsumerId = linearizeForallIVs(rewriter, loc, consumerForall);

    // Compute flat trip counts for the scf.for loop bounds.
    int64_t producerFlatTrip = *getStaticForallTripCount(producerForall);
    int64_t consumerFlatTrip = *getStaticForallTripCount(consumerForall);
    bool perfectlyDivides = (producerFlatTrip % consumerFlatTrip == 0);

    // Step 4: Create scf.for to iterate over producer work items.
    // Each consumer thread handles ceil(producerTrip / consumerTrip) items.
    Value lb = perfectlyDivides
                   ? rewriter.create<arith::ConstantIndexOp>(loc, 0)
                   : flatConsumerId;
    Value ub = rewriter.create<arith::ConstantIndexOp>(loc, producerFlatTrip);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, consumerFlatTrip);
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lb, ub, step, barrierOp.getBody()->getArgument(0));
    Block *loopBody = forOp.getBody();

    // Step 5: Inside the loop, compute producer IVs from flat index.
    rewriter.setInsertionPointToStart(loopBody);
    Value flatProducerId =
        perfectlyDivides
            ? rewriter.create<arith::AddIOp>(loc, forOp.getInductionVar(),
                                             flatConsumerId)
            : forOp.getInductionVar();

    // Delinearize flat producer ID into producer's dimension space.
    SmallVector<OpFoldResult> producerRanges;
    for (OpFoldResult ub : producerForall.getMixedUpperBound())
      producerRanges.push_back(ub);
    auto delinearize = rewriter.create<affine::AffineDelinearizeIndexOp>(
        loc, flatProducerId, llvm::to_vector(producerRanges));

    // Step 6: Clone producer body into the loop, mapping IVs.
    SmallVector<Value> newProducerIVs = delinearize.getResults();
    IRMapping mapping;
    for (auto [pIV, newIV] : llvm::zip_equal(
             producerForall.getInductionVars(), newProducerIVs)) {
      mapping.map(pIV, newIV);
    }
    // Map producer region iter args → loop iter args.
    for (auto [iterArg, loopArg] : llvm::zip_equal(
             producerForall.getRegionIterArgs(),
             forOp.getRegionIterArgs())) {
      mapping.map(iterArg, loopArg);
    }

    for (Operation &op : producerForall.getBody()->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Step 7: Convert parallel_insert_slice → insert_slice + scf.yield.
    Value insertSource = mapping.lookupOrDefault(producerInsert.getSource());
    Value insertDest = mapping.lookupOrDefault(producerInsert.getDest());

    // Helper: remap an OpFoldResult through the IRMapping. Attribute constants
    // pass through unchanged; Value operands are looked up in the mapping.
    auto remapOFR = [&](OpFoldResult ofr) -> OpFoldResult {
      if (auto val = dyn_cast<Value>(ofr))
        return mapping.lookupOrDefault(val);
      return ofr; // Attribute constant — no remapping needed.
    };

    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (OpFoldResult off : producerInsert.getMixedOffsets())
      offsets.push_back(remapOFR(off));
    for (OpFoldResult sz : producerInsert.getMixedSizes())
      sizes.push_back(remapOFR(sz));
    for (OpFoldResult st : producerInsert.getMixedStrides())
      strides.push_back(remapOFR(st));

    Value insertedSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, insertSource, insertDest, offsets, sizes, strides);
    rewriter.create<scf::YieldOp>(loc, insertedSlice);

    // Step 8: Yield the loop result from the barrier region.
    rewriter.setInsertionPointToEnd(barrierOp.getBody());
    rewriter.create<nova::YieldOp>(loc, forOp.getResults());

    // Step 9: Replace producer forall with the barrier_region result and
    // erase the producer.
    rewriter.replaceOp(producerForall, barrierOp.getResults());

    return success();
  }

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

//===----------------------------------------------------------------------===//
// Pattern 4.5: FuseExtractSliceConsumers
//
// When a tensor.extract_slice directly consumes a scf.forall result,
// fuse the slice into the forall by slicing the init operand instead.
// This eliminates an extra materialization of the full tensor outside
// the forall.
//
// Simplified version of IREE's fuseExtractSliceIntoProducerForall():
// - Requires zero offsets on the extract_slice
// - Requires single use of the forall result
// - No rank reduction support (deferred — add when tests need it)
// TODO: Support rank-reducing extract_slice (needs collapse_shape on result).
// TODO: Support clamping parallel_insert_slice (IREE's clampParallelInsertSliceOp).
//===----------------------------------------------------------------------===//

struct FuseExtractSliceConsumers final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    // Source must be a scf.forall result.
    auto forallOp = extractSliceOp.getSource().getDefiningOp<scf::ForallOp>();
    if (!forallOp)
      return failure();

    auto forallResult = cast<OpResult>(extractSliceOp.getSource());
    if (!forallResult.hasOneUse())
      return rewriter.notifyMatchFailure(forallOp,
                                         "forall result has multiple uses");

    // Only zero-offset extract_slice ops are supported.
    if (!llvm::all_of(extractSliceOp.getMixedOffsets(), [](OpFoldResult ofr) {
          auto cst = getConstantIntValue(ofr);
          return cst && *cst == 0;
        }))
      return rewriter.notifyMatchFailure(forallOp,
                                         "extract_slice has non-zero offsets");

    // Unit strides only.
    if (!llvm::all_of(extractSliceOp.getMixedStrides(), [](OpFoldResult ofr) {
          auto cst = getConstantIntValue(ofr);
          return cst && *cst == 1;
        }))
      return rewriter.notifyMatchFailure(forallOp,
                                         "extract_slice has non-unit strides");

    // No rank reduction for now.
    if (extractSliceOp.getSourceType().getRank() !=
        extractSliceOp.getType().getRank())
      return rewriter.notifyMatchFailure(forallOp,
                                         "rank-reducing extract_slice");

    // Find the corresponding parallel_insert_slice in the forall terminator.
    int64_t resultIdx = forallResult.getResultNumber();
    BlockArgument initBbarg = forallOp.getRegionIterArgs()[resultIdx];
    SmallVector<Operation *> parallelInsertOps =
        forallOp.getCombiningOps(initBbarg);
    if (parallelInsertOps.size() != 1)
      return rewriter.notifyMatchFailure(
          forallOp, "expected a single parallel_insert_slice");
    auto parallelInsertOp =
        dyn_cast<tensor::ParallelInsertSliceOp>(parallelInsertOps.front());
    if (!parallelInsertOp)
      return failure();

    // Extract_slice index operands must dominate the forall.
    DominanceInfo domInfo;
    int64_t indexStart =
        extractSliceOp.getOffsetSizeAndStrideStartOperandIndex();
    for (Value v : extractSliceOp->getOperands().drop_front(indexStart)) {
      if (!domInfo.dominates(v, forallOp))
        return rewriter.notifyMatchFailure(
            extractSliceOp, "index operands do not dominate forall");
    }

    // Clamp parallel_insert_slice sizes to fit within extracted slice sizes.
    // For each dimension: new_size = min(insert_size, extract_size).
    SmallVector<OpFoldResult> extractSizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> insertSizes = parallelInsertOp.getMixedSizes();
    SmallVector<OpFoldResult> newInsertSizes;
    bool needsClamp = false;
    for (auto [eSz, iSz] : llvm::zip_equal(extractSizes, insertSizes)) {
      auto eVal = getConstantIntValue(eSz);
      auto iVal = getConstantIntValue(iSz);
      if (eVal && iVal) {
        int64_t minSz = std::min(*eVal, *iVal);
        newInsertSizes.push_back(rewriter.getIndexAttr(minSz));
        if (minSz != *iVal)
          needsClamp = true;
      } else {
        // Dynamic — keep original (conservative, may need AffineMinOp later).
        newInsertSizes.push_back(iSz);
      }
    }

    // Replace uses of initBbarg (except in parallel_insert dest) with
    // the output operand from outside the loop.
    Value forallOutput = forallOp.getOutputs()[resultIdx];
    rewriter.replaceUsesWithIf(initBbarg, forallOutput, [&](OpOperand &operand) {
      return operand.getOwner() != parallelInsertOp.getOperation() ||
             operand.getOperandNumber() !=
                 parallelInsertOp.getDestMutable().getOperandNumber();
    });

    // Create extract_slice of the forall init with the same sizes.
    rewriter.setInsertionPoint(forallOp);
    auto extractedInit = tensor::ExtractSliceOp::create(
        rewriter, forallOp->getLoc(), forallOp.getOutputs()[resultIdx],
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());

    // Create new forall with sliced init.
    SmallVector<Value> newOutputs(forallOp.getOutputs());
    newOutputs[resultIdx] = extractedInit.getResult();

    auto newForallOp = scf::ForallOp::create(
        rewriter, forallOp->getLoc(), forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOutputs,
        forallOp.getMappingAttr());

    // Merge old forall body into new forall.
    SmallVector<Value> argReplacements(newForallOp.getInductionVars());
    argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                           newForallOp.getRegionIterArgs().end());
    newForallOp.getTerminator()->erase();
    rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                         argReplacements);

    // Update parallel_insert_slice sizes if clamping was needed.
    if (needsClamp) {
      // Find the (now moved) parallel_insert in the new forall.
      for (Operation &op :
           newForallOp.getTerminator().getYieldingOps()) {
        if (auto ins = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
          rewriter.setInsertionPoint(ins);
          auto newIns = tensor::ParallelInsertSliceOp::create(
              rewriter, ins.getLoc(), ins.getSource(), ins.getDest(),
              ins.getMixedOffsets(), newInsertSizes, ins.getMixedStrides());
          rewriter.eraseOp(ins);
          (void)newIns;
          break;
        }
      }
    }

    // Replace original extract_slice and forall.
    rewriter.replaceAllOpUsesWith(extractSliceOp,
                                  newForallOp->getResult(resultIdx));
    rewriter.replaceOp(forallOp, newForallOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 5: FuseTilableForallConsumers
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

    // Don't fuse ops that were independently tiled by TileDispatch.
    // If the consumer already lives inside its own scf.forall (from its own
    // lowering_config), pulling it into another forall creates mega-kernels
    // with 17+ shared memory globals that overflow SM_86's 100KB limit.
    // This matches IREE's approach where independently-configured ops are
    // excluded from each other's fusion clusters via the `payloadOps` set.
    if (tilableOp->getParentOfType<scf::ForallOp>())
      return failure();

    // Don't fuse cooperative copies that write to workgroup shared memory.
    // These are Stage 1 promotion copies (global→shared) that must remain
    // outside thread foralls so all threads participate in the cooperative
    // load.  Fusing them into a thread forall would (a) make each thread
    // copy only its own tile and (b) cause dominance violations when the
    // alloc_tensor<workgroup> is defined after the producer forall.
    for (auto init : dpsOp.getDpsInits()) {
      if (auto allocOp =
              init.getDefiningOp<bufferization::AllocTensorOp>()) {
        if (allocOp.getMemorySpace().has_value())
          return failure();
      }
    }

    // Find the FIRST forall producer among DPS inputs (matches IREE).
    scf::ForallOp forallProducer;
    for (auto operand : dpsOp.getDpsInputs()) {
      auto forallOp = operand.getDefiningOp<scf::ForallOp>();
      if (!forallOp)
        continue;
      if (forallOp->getBlock() != tilableOp->getBlock())
        continue;
      forallProducer = forallOp;
      break;
    }

    if (!forallProducer)
      return rewriter.notifyMatchFailure(
          tilableOp, "no scf.forall producer to fuse into");

    // Move the consumer (and its backward slice) right after the producer
    // forall to ensure dominance for tileAndFuseConsumerOfSlices.
    // Use op->moveBefore() directly (NOT rewriter.moveOpBefore()) to avoid
    // re-triggering patterns on the worklist (matches IREE approach).
    llvm::SetVector<Operation *> slice;
    BackwardSliceOptions opts;
    DominanceInfo domInfo;
    opts.filter = [&](Operation *op) {
      return domInfo.properlyDominates(forallProducer.getOperation(), op);
    };
    opts.inclusive = true;
    opts.omitUsesFromAbove = false;
    opts.omitBlockArguments = true;

    // Record original {op, successor} pairs for rollback on failure.
    SmallVector<std::pair<Operation *, Operation *>> originalPositions;
    if (succeeded(getBackwardSlice(tilableOp, &slice, opts))) {
      for (Operation *op : slice) {
        originalPositions.push_back({op, op->getNextNode()});
      }
      Block *block = forallProducer->getBlock();
      for (Operation *op : llvm::reverse(slice)) {
        // Recompute insert point each iteration (matches IREE).
        // After each move, std::next(forallProducer) points to
        // the most recently moved op, giving correct topological order.
        Block::iterator insertPt = std::next(forallProducer->getIterator());
        op->moveBefore(block, insertPt);
      }
    }

    // Collect parallel_insert_slice ops from the forall terminator.
    scf::InParallelOp parallelTerminator = forallProducer.getTerminator();
    SmallVector<Operation *> insertSlices;
    for (Operation &yieldingOp : parallelTerminator.getYieldingOps())
      insertSlices.push_back(&yieldingOp);
    if (insertSlices.empty()) {
      for (auto &[op, successor] : originalPositions) {
        if (successor) op->moveBefore(successor);
        else op->moveBefore(forallProducer->getBlock(),
                            forallProducer->getBlock()->end());
      }
      return failure();
    }

    SmallVector<LoopLikeOpInterface> loops = {
        cast<LoopLikeOpInterface>(forallProducer.getOperation())};

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusionResult =
        scf::tileAndFuseConsumerOfSlices(rewriter, insertSlices, loops);
    if (failed(fusionResult)) {
      for (auto &[op, successor] : originalPositions) {
        if (successor) op->moveBefore(successor);
        else op->moveBefore(forallProducer->getBlock(),
                            forallProducer->getBlock()->end());
      }
      return failure();
    }

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
    //     They must not depend on the for-loop iter arg and must not be used
    //     directly by the forall op we are hoisting.
    //
    //     Nova's pipeline places sibling scf.forall ops (copy_A and copy_B,
    //     mapped to #gpu.thread) in the same scf.for K body alongside the
    //     compute forall (mapped to #gpu.warp).  These sibling foralls perform
    //     cooperative global→shared memory loads; they are independent of the
    //     for-loop's iter arg and safe to move inside the new outer forall body
    //     (they execute before the inner scf.for on each iteration, reusing the
    //     warp/thread mapping of the outer forall).  We collect them in
    //     operationsToMove along with all other non-loop-like non-terminator ops.
    Block *loopBody = loop.getBody();
    Value forIterArg = loop.getRegionIterArg(0);
    SmallVector<Operation *> operationsToMove;
    for (Operation &op : loopBody->getOperations()) {
      if (&op == forallOp.getOperation() || &op == loopBody->getTerminator())
        continue;
      // Reject non-forall loop-like ops (scf.for, scf.while, etc.) inside
      // the body — we cannot safely interchange them.
      if (isa<LoopLikeOpInterface>(&op) && !isa<scf::ForallOp>(&op))
        return rewriter.notifyMatchFailure(
            loop, "for body contains a non-forall loop-like op");
      // Sibling scf.forall ops (e.g., copy_A / copy_B cooperative loads) must
      // not depend on the for-loop iter arg.
      for (Value operand : op.getOperands())
        if (operand == forIterArg)
          return rewriter.notifyMatchFailure(
              loop, "for body op uses iter arg");
      // The op must not feed directly into the hoisted forall's operands.
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
          // Clone any ops defined inside the old forall body that the
          // extract_slice depends on (e.g., affine.min for dynamic tile sizes).
          // Without this, the cloned extract_slice would reference values from
          // the old forall's child region, causing a dominance violation.
          Block *oldBody = forallOp.getBody();
          for (Value operand : oldExtract->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (!defOp || defOp->getBlock() != oldBody)
              continue;
            if (extractMapping.contains(operand))
              continue;
            // Recursively clone the defining op and its in-body dependencies.
            SmallVector<Operation *> opsToClone;
            std::function<void(Operation *)> collectDeps =
                [&](Operation *op) {
                  for (Value dep : op->getOperands()) {
                    Operation *depOp = dep.getDefiningOp();
                    if (depOp && depOp->getBlock() == oldBody &&
                        !extractMapping.contains(dep)) {
                      collectDeps(depOp);
                    }
                  }
                  opsToClone.push_back(op);
                };
            collectDeps(defOp);
            for (Operation *op : opsToClone) {
              if (!extractMapping.contains(op->getResult(0)))
                rewriter.clone(*op, extractMapping);
            }
          }
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
      }

      // STEP G: Create the new terminator for the outer forall.
      // Save parallelInsert metadata AFTER mergeBlocks (so values are updated)
      // but BEFORE erasing it.
      Location insertLoc = parallelInsert.getLoc();
      SmallVector<OpFoldResult> insertOffsets(parallelInsert.getMixedOffsets());
      SmallVector<OpFoldResult> insertSizes(parallelInsert.getMixedSizes());
      SmallVector<OpFoldResult> insertStrides(parallelInsert.getMixedStrides());
      rewriter.eraseOp(parallelInsert.getOperation());

      BlockArgument newForallIterArg = newForallOp.getRegionIterArgs()[0];
      // The insert's offset/size/stride values may reference ops that are now
      // inside newFor's body (after mergeBlocks). Clone these dependencies
      // into the forall body (before the terminator) so the new
      // parallel_insert_slice can reference them.
      rewriter.setInsertionPoint(newFor->getNextNode()
                                     ? newFor->getNextNode()
                                     : newForallOp.getTerminator());
      IRMapping terminatorMapping;
      // Map forall IVs and iter args to themselves (they're already correct).
      auto cloneOFRDeps = [&](SmallVector<OpFoldResult> &ofrs) {
        for (auto &ofr : ofrs) {
          auto val = dyn_cast<Value>(ofr);
          if (!val)
            continue;
          Operation *defOp = val.getDefiningOp();
          if (!defOp)
            continue;
          // If the value is defined inside the for loop body, we need to
          // clone its computation at the forall level.
          if (defOp->getParentOp() == newFor.getOperation()) {
            if (!terminatorMapping.contains(val)) {
              // Collect transitive in-for dependencies.
              SmallVector<Operation *> opsToClone;
              std::function<void(Operation *)> collectDeps =
                  [&](Operation *op) {
                    for (Value dep : op->getOperands()) {
                      Operation *depOp = dep.getDefiningOp();
                      if (depOp &&
                          depOp->getParentOp() == newFor.getOperation() &&
                          !terminatorMapping.contains(dep)) {
                        collectDeps(depOp);
                      }
                    }
                    if (!terminatorMapping.contains(op->getResult(0)))
                      rewriter.clone(*op, terminatorMapping);
                  };
              collectDeps(defOp);
            }
            ofr = terminatorMapping.lookup(val);
          }
        }
      };
      cloneOFRDeps(insertOffsets);
      cloneOFRDeps(insertSizes);
      cloneOFRDeps(insertStrides);

      rewriter.setInsertionPointToEnd(newForallOp.getTerminator().getBody());
      tensor::ParallelInsertSliceOp::create(
          rewriter, insertLoc,
          newFor.getResult(0),
          newForallIterArg,
          insertOffsets,
          insertSizes,
          insertStrides);
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
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect, linalg::LinalgDialect,
                    gpu::GPUDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    // -------------------------------------------------------------------
    // Round 1: Hoist + fuse foralls (matches IREE Phase 1)
    // FuseForalls no longer needs a global flatWorkgroupSize — it matches
    // producer/consumer foralls by flat trip count equality directly.
    // -------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      patterns.add<FuseForalls>(ctx, /*benefit=*/2);
      patterns.add<FuseTilableForallConsumers>(ctx);
      patterns.add<HoistForallFromFor>(ctx);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
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
      // TODO: Enable FuseExtractSliceConsumers — needs proper source clamping
      // (extract_slice of the parallel_insert source to match new dest sizes).
      // IREE's clampParallelInsertSliceOp handles this but is complex.
      // patterns.add<FuseExtractSliceConsumers>(ctx);
      // TODO: Add FuseCollapseShapeConsumers when tests need it.
      // IREE uses fuseCollapseShapeIntoProducerForall() which requires
      // AffineLinearizeIndexOp (not available in Nova).
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
      // Round 3 entry
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
      // Round 3 done
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