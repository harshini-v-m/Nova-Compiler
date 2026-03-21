//===- NovaFuseReductionIntoProducer.cpp ----------------------------------===//
//
// Wrapper for nova elementiwise fusion to prevent matmul from fusing
// pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
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
// Elementwise fusion operation
//
// Fires on a linalg.generic CONSUMER that has at least one reduction iterator.
// For each input operand, if:
//   1. The operand is matmul, then skips (matmul shouldn't be fused)
// then perfor other fusions using linalg::fuseElementwiseOps — the same MLIR built-in that
// IREE uses. It handles all indexing-map composition internally.
//===----------------------------------------------------------------------===//
struct ElementwiseOpFusionPass : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {

    // Skip matmul operations entirely - don't let them be consumers  
    if (mlir::linalg::isaContractionOpInterface(genericOp))  
      return failure();  
  
    for (OpOperand &opOperand : genericOp->getOpOperands()) {  
  
      // Check elementwise fusion preconditions  
      if (!linalg::areElementwiseOpsFusable(&opOperand))  
        continue;  
          
      // Delegate fusion logic to built-in infrastructure  
      FailureOr<linalg::ElementwiseOpFusionResult> result =   
          linalg::fuseElementwiseOps(rewriter, &opOperand);  
      if (succeeded(result)) {  
        auto replacements = result->fusedOp->getResults().take_back(genericOp.getNumResults());  
        rewriter.replaceOp(genericOp, replacements);  
        return success();  
      }  
    }  
    return failure();  
  }  
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct NovaElementwiseOpFusionPass: public PassWrapper<NovaElementwiseOpFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaElementwiseOpFusionPass)

  NovaElementwiseOpFusionPass() = default;
  NovaElementwiseOpFusionPass(const NovaElementwiseOpFusionPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<ElementwiseOpFusionPass>(funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }

  StringRef getArgument() const override {
    return "nova-element-wise-fusion";
  }
  StringRef getDescription() const override {
    return "Ignores matmul and fuses the linalg parallel and reduction cases";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaElementwiseOpFusionPass() {
  return std::make_unique<NovaElementwiseOpFusionPass>();
}

void registerNovaElementwiseOpFusionPass() {
  PassRegistration<NovaElementwiseOpFusionPass>();
}

} // namespace mlir::nova
