//===- NovaFuseReductionIntoProducer.cpp ----------------------------------===//
//
// Fuses a linalg.generic reduction consumer into its elementwise producer
// when the intermediate tensor is large enough to overflow GPU shared memory.
//
// Mirrors IREE's RematerializeParallelOps pass
// (iree/compiler/Codegen/Common/RematerializeParallelOps.cpp) with one
// addition: the pattern only fires when the CONSUMER has a reduction iterator,
// so it is complementary to (not redundant with) LinalgElementwiseOpFusionPass
// which already handles elementwise→elementwise chains at Step -1.
//
// Pipeline position: Step -1, immediately after LinalgElementwiseOpFusionPass,
// BEFORE tiling. At this stage ops are bare linalg.generic on tensors. After
// tiling the producer becomes an scf.forall and linalg::fuseElementwiseOps
// can no longer reach through it.
//
// Effect on layernorm:
//   BEFORE:  PRODUCER: linalg.generic → tensor<8×1024×384>  (centered)
//            CONSUMER: linalg.generic ins(centered)          (variance sum)
//   AFTER:   fused op reads x and mean_sum directly; centered is only needed
//            by the output op → bufferized to [1×128×128] tile = 64 KB.
//   Shared memory: 260 KB → 68 KB (fits sm_86 100 KB limit).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nova-fuse-reduction-into-producer"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool hasReductionIterator(linalg::GenericOp op) {
  return llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::reduction;
  });
}

// Returns the size in bytes of a statically-shaped ranked tensor type.
// Returns 0 for dynamic shapes.
static int64_t tensorSizeBytes(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || !type.hasStaticShape())
    return 0;
  return type.getNumElements() * (type.getElementTypeBitWidth() / 8);
}

// Returns true if any result of `op` is consumed by an op that writes the
// value directly to a destination tensor (e.g. tensor.insert_slice used as
// a dispatch output). Fusing through such a boundary would move the write,
// potentially producing incorrect results.
// Mirrors IREE's hasDirectWriteResult guard.
static bool hasDirectWriteResult(linalg::GenericOp op) {
  return llvm::any_of(op->getResults(), [](Value result) {
    return llvm::any_of(result.getUsers(), [](Operation *user) {
      return isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(user);
    });
  });
}

//===----------------------------------------------------------------------===//
// FuseReductionIntoProducerPattern
//
// Fires on a linalg.generic CONSUMER that has at least one reduction iterator.
// For each input operand, if:
//   1. The operand is fusable (linalg::areElementwiseOpsFusable),
//   2. The defining op is a linalg.generic PRODUCER,
//   3. The PRODUCER does not write directly to an output boundary,
//   4. The intermediate tensor exceeds the shared memory threshold,
// then fuse using linalg::fuseElementwiseOps — the same MLIR built-in that
// IREE uses. It handles all indexing-map composition internally.
//===----------------------------------------------------------------------===//
struct FuseReductionIntoProducerPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  // Intermediate tensors larger than this (bytes) trigger fusion.
  // 96 KB leaves headroom below sm_86's 100 KB shared memory limit.
  static constexpr int64_t kThresholdBytes = 96LL * 1024;

  LogicalResult matchAndRewrite(linalg::GenericOp consumerOp,
                                PatternRewriter &rewriter) const override {
    // Only fuse when the consumer performs a reduction — elementwise→elementwise
    // is already handled by LinalgElementwiseOpFusionPass at the same step.
    if (!hasReductionIterator(consumerOp))
      return failure();
    
    // skip matmul generics 
    if (mlir::linalg::isaContractionOpInterface(consumerOp))  
      return failure();

    for (OpOperand *opOperand : consumerOp.getDpsInputOperands()) {
      auto producerOp = opOperand->get().getDefiningOp<linalg::GenericOp>();
      if (!producerOp)
        continue;
      // The producer must be all-parallel (elementwise). Note:
      // linalg::areElementwiseOpsFusable only requires the PRODUCER to be
      // all-parallel — the consumer is allowed to have reduction iterators
      // (the helper has explicit coverage-check logic for that case).
      if (!llvm::all_of(producerOp.getIteratorTypesArray(),
                        linalg::isParallelIterator))
        continue;

      // Do not fuse producers that write directly to dispatch outputs.
      if (hasDirectWriteResult(producerOp))
        continue;

      // Only fuse when the intermediate is large enough to matter.
      if (tensorSizeBytes(opOperand->get()) < kThresholdBytes)
        continue;

      // Guard: verify all structural pre-conditions required by
      // fuseElementwiseOps (permutation output map, loop-bound coverage for
      // reduction dims, etc.) before calling it. Skipping this check causes
      // an assertion crash inside fuseElementwiseOps for cases where the
      // indexing maps do not satisfy its internal requirements.
      if (!linalg::areElementwiseOpsFusable(opOperand))
        continue;

      // Delegate ALL fusion logic (body merge, indexing maps, block args)
      // to the built-in infrastructure — mirrors IREE exactly.
      FailureOr<linalg::ElementwiseOpFusionResult> result =
          linalg::fuseElementwiseOps(rewriter, opOperand);
      if (succeeded(result)) {
        // Replace consumer with the fused op's results.
        // take_back: the fused op has PRODUCER results prepended; consumer
        // results are at the back.
        auto replacements =
            result->fusedOp->getResults().take_back(consumerOp.getNumResults());
        rewriter.replaceOp(consumerOp, replacements);
        return success();
      }
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct NovaFuseReductionIntoProducerPass
    : public PassWrapper<NovaFuseReductionIntoProducerPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaFuseReductionIntoProducerPass)

  NovaFuseReductionIntoProducerPass() = default;
  NovaFuseReductionIntoProducerPass(const NovaFuseReductionIntoProducerPass &) =
      default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<FuseReductionIntoProducerPattern>(funcOp.getContext());
    // Clean up unused PRODUCER results/operands after fusion.
    // e.g. after fusing `centered` into the variance reduction, the PRODUCER's
    // `mean` yield may become unused; this removes it.
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }

  StringRef getArgument() const override {
    return "nova-fuse-reduction-into-producer";
  }
  StringRef getDescription() const override {
    return "Fuses a linalg.generic reduction consumer into its elementwise "
           "producer to eliminate large intermediate workgroup tensors "
           "(e.g. alloc_5 [1x128x384] = 192 KB in layernorm). "
           "Runs pre-tiling at Step -1 alongside LinalgElementwiseOpFusion.";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaFuseReductionIntoProducerPass() {
  return std::make_unique<NovaFuseReductionIntoProducerPass>();
}

void registerNovaFuseReductionIntoProducerPass() {
  PassRegistration<NovaFuseReductionIntoProducerPass>();
}

} // namespace mlir::nova
