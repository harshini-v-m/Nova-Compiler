//===- NovaEliminateEmptyTensors.cpp - Pre-bufferization empty elimination ===//
//
// Pre-bufferization pass that tries to eliminate `tensor.empty` ops by finding
// an existing destination tensor that can be reused, thus avoiding unnecessary
// buffer allocations during bufferization.
//
// Two-step approach:
//   Step 1 — Convert-to-DPS patterns:
//     Rewrites ops like linalg.generic with tensor.empty outputs into
//     destination-passing style (DPS) form where the output can be reused.
//
//   Step 2 — Bufferization analysis:
//     Runs bufferization.OneShotAnalysis and replaces tensor.empty ops with
//     existing destination tensors wherever the analysis proves the empty
//     would alias an existing buffer after bufferization anyway.
//
// IREE equivalent: EliminateEmptyTensorsPass
//   (iree/compiler/src/iree/compiler/Codegen/Common/
//    IREEComprehensiveBufferizePass.cpp)
//
// IREE also has duplicateTensorEmptyOps and moveUpMemrefReshapeOps which are
// IREE-HAL specific. We only port the core upstream MLIR parts.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

struct NovaEliminateEmptyTensorsPass
    : public PassWrapper<NovaEliminateEmptyTensorsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaEliminateEmptyTensorsPass)

  NovaEliminateEmptyTensorsPass() = default;
  NovaEliminateEmptyTensorsPass(const NovaEliminateEmptyTensorsPass &) =
      default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    // ALGORITHM STEP 1: Convert ops to destination-passing style.
    // Rewrites patterns like linalg ops with tensor.empty outputs into forms
    // where the output tensor can be reused as the destination — enabling the
    // analysis in Step 2 to eliminate the empty.
    {
      RewritePatternSet patterns(context);
      linalg::populateConvertToDestinationStylePatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp->emitOpError(
            "failed in conversion to destination style patterns");
        return signalPassFailure();
      }
    }

    // ALGORITHM STEP 2: Run bufferization analysis and eliminate empty tensors.
    // OneShotAnalysis determines which tensor.empty results would alias an
    // existing buffer after bufferization. Those empties are replaced with the
    // existing tensor, saving one allocation per eliminated empty.
    {
      bufferization::OneShotBufferizationOptions opts;
      opts.bufferizeFunctionBoundaries = false;
      bufferization::OneShotAnalysisState state(funcOp, opts);

      if (failed(bufferization::analyzeOp(funcOp, state)))
        return signalPassFailure();

      IRRewriter rewriter(context);
      if (failed(
              bufferization::eliminateEmptyTensors(rewriter, funcOp, state)))
        return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-eliminate-empty-tensors";
  }
  StringRef getDescription() const override {
    return "Eliminates tensor.empty ops by finding existing destination "
           "tensors, reducing allocations during bufferization";
  }
};

std::unique_ptr<Pass> createNovaEliminateEmptyTensorsPass() {
  return std::make_unique<NovaEliminateEmptyTensorsPass>();
}

void registerNovaEliminateEmptyTensorsPass() {
  PassRegistration<NovaEliminateEmptyTensorsPass>();
}

} // namespace mlir::nova
