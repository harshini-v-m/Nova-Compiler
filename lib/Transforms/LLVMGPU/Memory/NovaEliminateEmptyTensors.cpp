//===- NovaEliminateEmptyTensors.cpp - Pre-bufferization empty elimination ===//
//
// Pre-bufferization pass that tries to eliminate `tensor.empty` ops by finding
// an existing destination tensor that can be reused, thus avoiding unnecessary
// buffer allocations during bufferization.
//
// Two-step approach (mirrors IREE's EliminateEmptyTensorsPass):
//   Step 1 — Convert-to-DPS patterns:
//     Rewrites linalg ops with tensor.empty outputs into destination-passing
//     style so the output can be reused as the destination.
//
//   Step 2 — Bufferization analysis:
//     Runs bufferization.OneShotAnalysis with allowUnknownOps=true (so ops
//     without bufferization interfaces are skipped conservatively rather than
//     causing a hard failure) and replaces tensor.empty ops with existing
//     destination tensors wherever the analysis proves the empty would alias
//     an existing buffer after bufferization.
//
// Note: IREE's version also runs duplicateTensorEmptyOps and
// moveUpMemrefReshapeOps here. The duplicate step is intentionally omitted
// because Nova's K-loop writes into the outer tensor.empty (not the iter_arg
// itself) — duplicating would split the common ancestor and break the alias
// chain that OneShotBufferize relies on. The reshape hoist is HAL-specific.
//
// IREE reference: EliminateEmptyTensorsPass in
//   iree/compiler/src/iree/compiler/Codegen/Common/
//   IREEComprehensiveBufferizePass.cpp
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
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
    //
    // allowUnknownOps=true: ops without a BufferizableOpInterface (e.g.
    // nova_vector_ext.to_layout, nova.value_barrier) are treated conservatively
    // — their tensor operands/results are assumed to alias nothing — instead of
    // causing a hard analysis failure. This is essential because Nova has ops
    // that only get bufferization interfaces at OneShotBufferize time (inside
    // the comprehensive bufferize pass), not during this pre-pass analysis.
    {
      bufferization::OneShotBufferizationOptions opts;
      opts.bufferizeFunctionBoundaries = false;
      // allowUnknownOps: ops without BufferizableOpInterface (e.g.
      // nova_vector_ext.to_layout) are treated conservatively rather than
      // causing a hard failure.
      opts.allowUnknownOps = true;
      // allowReturnAllocsFromLoops: suppresses the scf.for verifyAnalysis
      // check that requires scf.yield operands to be equivalent to their
      // corresponding iter_args. Nova's K-loop accumulator writes into an
      // outer tensor.empty (not directly into the iter_arg), which legitimately
      // violates this strict equivalence requirement.
      opts.allowReturnAllocsFromLoops = true;
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
