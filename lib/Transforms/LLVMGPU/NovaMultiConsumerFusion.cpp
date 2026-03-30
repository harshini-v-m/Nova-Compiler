//===- NovaMultiConsumerFusion.cpp ----------------------------------------===//
//
// Fuses a parallel linalg.generic producer into each of its consumers,
// even when the producer has multiple uses (multi-consumer rematerialization).
//
// Unlike the standard MLIR elementwise fusion pass which requires the producer
// to have a single use, this pass rematerializes a parallel producer into
// every consumer that uses it. This is safe because:
//   1. The producer is all-parallel (no reduction) → pure computation on
//      tensors with no side effects.
//   2. linalg.generic on tensors is referentially transparent — recomputing
//      the same value at multiple sites is always semantically correct.
//   3. No other operations are affected: the producer remains in the IR until
//      DCE removes it once all uses are fused, and interim uses of any other
//      producer result continue to see the original SSA value unchanged.
//
// Algorithm (pattern fires on each CONSUMER):
//   For each DPS input operand of the consumer that is the result of an
//   all-parallel linalg.generic PRODUCER:
//     - Guard: producer must be all-parallel (iterative stop condition —
//       once the chain reaches a reduction, the guard prevents further
//       producer-side fusion for that chain).
//     - Guard: producer must not be a contraction op (matmul).
//     - Guard: producer must not write directly to a dispatch output boundary.
//     - Structural check: linalg::areElementwiseOpsFusable must pass.
//     - Fuse via linalg::fuseElementwiseOps (inlines producer body into
//       consumer, composes indexing maps, and merges block arguments).
//     - Replace the consumer with the fused op's back-results.
//   The greedy driver repeats until fixpoint; the original producer is erased
//   by DCE once all its results have no remaining uses.
//
// tensor.empty / linalg.fill handling:
//   Neither is a linalg.generic and neither is matched as a producer.
//   After all fusion steps the unused producer output inits (typically
//   tensor.empty or linalg.fill results) are removed by
//   populateEraseUnusedOperandsAndResultsPatterns, then DCE erases the
//   now-dead empty/fill ops.
//
// Pipeline position: Step -1, alongside other elementwise fusion passes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nova-fuse-parallels-with-multiple-consumers"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Returns true if any result of `op` flows directly into a
// tensor.insert_slice / tensor.parallel_insert_slice, i.e. is written
// to a dispatch-level output boundary.  Fusing through such a boundary
// would reorder the write, potentially producing incorrect results.
static bool hasDirectWriteResult(linalg::GenericOp op) {
  return llvm::any_of(op->getResults(), [](Value result) {
    return llvm::any_of(result.getUsers(), [](Operation *user) {
      return isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(user);
    });
  });
}

//===----------------------------------------------------------------------===//
// FuseParallelGenericsWithMultipleConsumers
//
// Pattern fires on the CONSUMER.  For each input operand whose defining op
// is an all-parallel linalg.generic PRODUCER, the pattern:
//   1. Verifies all guards (parallel, non-contraction, no boundary write).
//   2. Calls linalg::areElementwiseOpsFusable to check structural validity.
//   3. Calls linalg::fuseElementwiseOps to inline the producer body.
//   4. Replaces the consumer with the fused op.
//
// The producer is NOT erased here — it may still have other uses that will
// be fused in later greedy iterations.  DCE removes it once fully dead.
//
// Iterative chain termination:
//   After fusing P (parallel) into C, the resulting fused_PC may itself
//   become a consumer of another parallel op.  The greedy driver keeps firing
//   the pattern.  If C was a reduction, fused_PC has reduction iterators and
//   the guard "producer must be all-parallel" stops the chain correctly.
//===----------------------------------------------------------------------===//
struct FuseParallelGenericsWithMultipleConsumers
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumerOp,
                                PatternRewriter &rewriter) const override {
    // Skip contraction ops (matmul, batch-matmul, etc.) as consumers — their
    // tiling strategies are handled separately and premature fusion can destroy
    // the contraction structure that later passes rely on.
    if (linalg::isaContractionOpInterface(consumerOp))
      return failure();

    for (OpOperand *opOperand : consumerOp.getDpsInputOperands()) {
      // The operand must come from a linalg.generic producer.
      auto producerOp =
          opOperand->get().getDefiningOp<linalg::GenericOp>();
      if (!producerOp)
        continue;

      // ── Iterative stop condition ───────────────────────────────────────────
      // Only rematerialize producers that are fully parallel.  Once a chain
      // reaches a reduction the guard fires and fusion stops at that boundary.
      if (!producerOp.isAllParallelLoops())
        continue;

      // Skip contraction producers — their operand layout must be preserved.
      if (linalg::isaContractionOpInterface(producerOp))
        continue;

      // Do not fuse producers whose result flows into an output boundary; that
      // would move a write and corrupt dispatch-level memory management.
      if (hasDirectWriteResult(producerOp))
        continue;

      // Structural pre-condition: the operand's indexing map must be a
      // permutation and the producer's loops must be covered by the consumer's
      // loops.  Skipping this check causes an assertion inside fuseElementwiseOps.
      if (!linalg::areElementwiseOpsFusable(opOperand))
        continue;

      // ── Fuse producer body into this consumer use ──────────────────────────
      // fuseElementwiseOps:
      //   - Composes the producer's and consumer's indexing maps.
      //   - Merges block arguments (producer ins, consumer ins, shared output).
      //   - Produces a new GenericOp at the consumer's position.
      //   - Does NOT erase the producer (other uses remain intact).
      FailureOr<linalg::ElementwiseOpFusionResult> result =
          linalg::fuseElementwiseOps(rewriter, opOperand);
      if (failed(result))
        continue;

      // The fused op's results are ordered [producer_extra_results..., consumer_results].
      // Replace only the consumer's results; extra producer results are handled
      // by the remaining uses of the original producer.
      auto replacements =
          result->fusedOp->getResults().take_back(consumerOp.getNumResults());
      rewriter.replaceOp(consumerOp, replacements);
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
    return "Fuse parallel linalg.generic producers into each of their consumers "
           "(multi-consumer rematerialization). Safe because parallel generics "
           "on tensors are side-effect-free and referentially transparent.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());

    // Core multi-consumer fusion pattern.
    patterns.add<FuseParallelGenericsWithMultipleConsumers>(
        funcOp.getContext());

    // After fusing a producer into all its consumers the original producer
    // may have unused output operands (e.g. the tensor.empty / linalg.fill
    // that seeded its output).  This pattern strips those dead operands and
    // results from the IR before DCE removes the now-empty ops.
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

    // Top-down traversal ensures producers are visited before consumers,
    // which gives the greedy driver the best chance of fusing the full chain
    // in a single pass rather than requiring multiple fixpoint iterations.
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal();

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      return signalPassFailure();
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
