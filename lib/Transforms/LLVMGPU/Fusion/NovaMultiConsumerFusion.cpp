#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "nova-fuse-parallels-with-multiple-consumers"

using namespace mlir;
using namespace mlir::linalg;
using llvm::dbgs;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isAllParallel(linalg::GenericOp op) {
  return llvm::all_of(op.getIteratorTypesArray(), linalg::isParallelIterator);
}

//===----------------------------------------------------------------------===//
// Core fusion routine
//===----------------------------------------------------------------------===//

/// Fuse `producerOp` into `consumerOp`.
///
/// `producerResultIdx`  — which result of producerOp feeds the consumer.
/// `consumerInputIdx`   — which DPS input of consumerOp receives that result.
///
/// This guarantees that multi-result producers (e.g. a fused add+relu op
/// that yields both add and relu) are completely erased once every result
/// has been taken over by a successive fusion step.

static LogicalResult tryFuse(PatternRewriter &rewriter,
                             linalg::GenericOp producerOp,
                             unsigned producerResultIdx,
                             linalg::GenericOp consumerOp,
                             unsigned consumerInputIdx) {

  unsigned numProdInputs  = producerOp.getNumDpsInputs();
  unsigned numProdResults = producerOp->getNumResults();
  unsigned numConsInputs  = consumerOp.getNumDpsInputs();
  unsigned numConsInits   = consumerOp.getNumDpsInits();

  auto producerMaps = producerOp.getIndexingMapsArray();
  auto consumerMaps = consumerOp.getIndexingMapsArray();

  // 1. Structural compatibility
  AffineMap prodResultMap = producerMaps[numProdInputs + producerResultIdx];
  AffineMap consInputMap  = consumerMaps[consumerInputIdx];

  if (!prodResultMap.isProjectedPermutation() ||
      !consInputMap.isProjectedPermutation())
    return failure();
  if (producerOp.getNumLoops() != consumerOp.getNumLoops())
    return failure();
  if (prodResultMap != consInputMap)
    return failure();

  // 2. Decide which producer results the fused op must yield
  // needsYield[i]         — true  iff producer result i must be re-exposed.
  // prodResultToOut[i]    — index into newOutputs (valid iff needsYield[i]).
  SmallVector<bool>     needsYield(numProdResults, false);
  SmallVector<unsigned> prodResultToOut(numProdResults, ~0u);

  OpOperand *specificUse = consumerOp.getDpsInputOperands()[consumerInputIdx];
  Block &producerBlock   = producerOp.getRegion().front();
  Block &consumerBlock   = consumerOp.getRegion().front();

  for (unsigned i = 0; i < numProdResults; ++i) {
    Value result = producerOp->getResult(i);

    if (i == producerResultIdx) {
      // Case A: any use other than the slot we are fusing away.
      needsYield[i] = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return &use != specificUse;
      });
      // Case B: producer's output block-arg is read inside its own body
      // (in-place accumulation), so the init tensor must be threaded through.
      if (!needsYield[i]) {
        BlockArgument outArg = producerBlock.getArgument(numProdInputs + i);
        for (Operation &bodyOp : producerBlock.without_terminator()) {
          if (llvm::any_of(bodyOp.getOperands(),
                           [outArg](Value v) { return v == outArg; })) {
            needsYield[i] = true;
            break;
          }
        }
      }
    } else {
      // Every other producer result: yield whenever it has any use, since
      // all those uses are external to this fusion step.
      needsYield[i] = !result.use_empty();
    }
  }

  // 3. Dominance guard
  // The fused op is inserted just before consumerOp. Any op between
  // producerOp and consumerOp that uses a producer result we are rerouting
  // would then reference a value defined after it — a dominance violation.
  if (producerOp->getBlock() == consumerOp->getBlock()) {
    bool inWindow = false;
    for (Operation &op : producerOp->getBlock()->getOperations()) {
      if (&op == producerOp.getOperation()) { inWindow = true;  continue; }
      if (&op == consumerOp.getOperation()) { inWindow = false; break;    }
      if (!inWindow) continue;
      for (Value operand : op.getOperands()) {
        for (unsigned i = 0; i < numProdResults; ++i) {
          if (needsYield[i] && operand == producerOp->getResult(i))
            return failure();
        }
      }
    }
  }

  // 4. Build the fused operand lists and indexing maps
  SmallVector<Value>     newInputs, newOutputs;
  SmallVector<AffineMap> newMaps;
  SmallVector<Type>      newResultTypes;

  // Producer inputs (all).
  for (unsigned i = 0; i < numProdInputs; ++i) {
    newInputs.push_back(producerOp.getDpsInputs()[i]);
    newMaps.push_back(producerMaps[i]);
  }
  // Consumer inputs — skip the slot that receives the inlined result.
  for (unsigned i = 0; i < numConsInputs; ++i) {
    if (i == consumerInputIdx) continue;
    newInputs.push_back(consumerOp.getDpsInputs()[i]);
    newMaps.push_back(consumerMaps[i]);
  }

  // Producer outputs — one entry per result that needs yielding, in order.
  // Record each one's position in newOutputs for block-arg mapping below.
  for (unsigned i = 0; i < numProdResults; ++i) {
    if (!needsYield[i]) continue;
    prodResultToOut[i] = newOutputs.size();
    newOutputs.push_back(producerOp.getDpsInits()[i]);
    newMaps.push_back(producerMaps[numProdInputs + i]);
    newResultTypes.push_back(producerOp->getResult(i).getType());
  }

  // Consumer outputs (all), appended after producer outputs.
  unsigned consOutBase = newOutputs.size();
  for (unsigned i = 0; i < numConsInits; ++i) {
    newOutputs.push_back(consumerOp.getDpsInits()[i]);
    newMaps.push_back(consumerMaps[numConsInputs + i]);
    newResultTypes.push_back(consumerOp->getResultTypes()[i]);
  }

  // 5. Create the fused GenericOp with an inline body builder
  // Block-arg layout (mirrors newInputs / newOutputs order):
  //   args[0 .. numProdInputs-1]                         producer inputs
  //   args[numProdInputs .. newInputs.size()-1]          consumer non-fused inputs
  //   args[newInputs.size() + prodResultToOut[i]]        producer output (per yielded result)
  //   args[newInputs.size() + consOutBase + j]           consumer output j
  rewriter.setInsertionPoint(consumerOp);

  auto fusedOp = rewriter.create<linalg::GenericOp>(
      consumerOp.getLoc(),
      TypeRange(newResultTypes),
      newInputs, newOutputs,
      newMaps,
      consumerOp.getIteratorTypesArray(),
      /*doc=*/"", /*libraryCall=*/"",
      [&](OpBuilder &b, Location fusedLoc, ValueRange args) {
        IRMapping mapping;

        // Map producer input block args.
        for (unsigned i = 0; i < numProdInputs; ++i)
          mapping.map(producerBlock.getArgument(i), args[i]);

        // Map consumer non-fused input block args.
        {
          unsigned slot = numProdInputs;
          for (unsigned i = 0; i < numConsInputs; ++i) {
            if (i == consumerInputIdx) continue;
            mapping.map(consumerBlock.getArgument(i), args[slot++]);
          }
        }

        // Map producer output block args for results being yielded.
        unsigned outArgBase = newInputs.size();
        for (unsigned i = 0; i < numProdResults; ++i) {
          if (!needsYield[i]) continue;
          mapping.map(producerBlock.getArgument(numProdInputs + i),
                      args[outArgBase + prodResultToOut[i]]);
        }

        // Map consumer output block args.
        for (unsigned i = 0; i < numConsInits; ++i)
          mapping.map(consumerBlock.getArgument(numConsInputs + i),
                      args[outArgBase + consOutBase + i]);

        // Inline the producer body.
        for (Operation &op : producerBlock.without_terminator())
          b.clone(op, mapping);

        // Collect all producer computed values via the producer yield.
        auto prodYield = cast<linalg::YieldOp>(producerBlock.getTerminator());
        SmallVector<Value> prodComputed(numProdResults);
        for (unsigned i = 0; i < numProdResults; ++i)
          prodComputed[i] = mapping.lookupOrDefault(prodYield.getOperand(i));

        // Expose the inlined result as the consumer's block arg for that slot.
        mapping.map(consumerBlock.getArgument(consumerInputIdx),
                    prodComputed[producerResultIdx]);

        // Inline the consumer body.
        for (Operation &op : consumerBlock.without_terminator())
          b.clone(op, mapping);

        // Build the fused yield:
        //   [producer yielded values, in result-index order]
        //   [consumer yield values]
        SmallVector<Value> yieldVals;
        for (unsigned i = 0; i < numProdResults; ++i) {
          if (!needsYield[i]) continue;
          yieldVals.push_back(prodComputed[i]);
        }
        auto consYield = cast<linalg::YieldOp>(consumerBlock.getTerminator());
        for (Value v : consYield.getOperands())
          yieldVals.push_back(mapping.lookupOrDefault(v));
        b.create<linalg::YieldOp>(fusedLoc, yieldVals);
      });

  // 6. Replace uses of all original results
  // Producer results that were yielded → corresponding fused results.
  unsigned fusedResIdx = 0;
  for (unsigned i = 0; i < numProdResults; ++i) {
    if (!needsYield[i]) continue;
    rewriter.replaceAllUsesWith(producerOp->getResult(i),
                                fusedOp->getResult(fusedResIdx++));
  }
  // Consumer results → fused results (always).
  for (unsigned i = 0; i < numConsInits; ++i)
    rewriter.replaceAllUsesWith(consumerOp->getResult(i),
                                fusedOp->getResult(fusedResIdx++));

  // 7. Erase original ops
  rewriter.eraseOp(consumerOp);
  // Erase the producer only when every result is now dead.
  if (llvm::all_of(producerOp->getResults(),
                   [](Value r) { return r.use_empty(); }))
    rewriter.eraseOp(producerOp);

  return success();
}

// Rewrite pattern

/// Matches any linalg.generic consumer whose DPS inputs include a result from
/// an all-parallel linalg.generic producer, then calls tryFuse.
///
/// The greedy driver re-applies the pattern until a fixed point, naturally
/// chaining fusions:
///
///   P (parallel) → C1 (parallel) → C2 (parallel) → C3 (reduction)
///
///   Round 1: fuse P into C1 → fused1
///            fused1 yields: every P result that is still used, plus C1 result.
///   Round 2: fuse fused1 into C2 → fused2
///            fused2 yields: every fused1 result still in use, plus C2 result.
///            fused1 is erased because all its results are now covered.
///   Round 3: fuse fused2 into C3 → fused3 (reduction)
///            fused3 yields: every fused2 result still in use, plus C3 result.
///            fused2 is erased.
///   Round 4: fused3 has reductions → isAllParallel() fails → stop.
struct FuseParallelGenericsWithMultipleConsumers
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumerOp,
                                PatternRewriter &rewriter) const override {
    // Skip contraction
    if (mlir::linalg::isaContractionOpInterface(consumerOp))
      return failure();

    for (auto [inputIdx, input] :
         llvm::enumerate(consumerOp.getDpsInputs())) {

      auto producerOp = input.getDefiningOp<linalg::GenericOp>();
      if (!producerOp || !isAllParallel(producerOp) || producerOp == consumerOp)
        continue;

      // Identify which result of producerOp feeds this input slot.
      unsigned producerResultIdx = ~0u;
      for (auto [rIdx, result] : llvm::enumerate(producerOp->getResults())) {
        if (result == input) { producerResultIdx = rIdx; break; }
      }
      if (producerResultIdx == ~0u) continue;

      if (succeeded(tryFuse(rewriter, producerOp, producerResultIdx,
                            consumerOp, (unsigned)inputIdx)))
        return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass wrapper
//===----------------------------------------------------------------------===//

struct NovaMultiConsumerFusion
    : public PassWrapper<NovaMultiConsumerFusion, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaMultiConsumerFusion)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  StringRef getArgument()    const final { return "nova-multi-consumer-fusion"; }
  StringRef getDescription() const final {
    return "Fuse parallel linalg.generic producers into each of their consumers.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FuseParallelGenericsWithMultipleConsumers>(ctx);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal();

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaMultiConsumerFusion() {
  return std::make_unique<NovaMultiConsumerFusion>();
}

void registerNovaMultiConsumerFusion() {
  PassRegistration<NovaMultiConsumerFusion>();
}

} // namespace mlir::nova
