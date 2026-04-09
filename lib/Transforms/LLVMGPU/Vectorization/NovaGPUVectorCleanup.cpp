//===- NovaGPUVectorCleanup.cpp - Vectorization Cleanup Passes -------------===//
//
// Defines cleanup and optimization passes for the Nova GPU vector pipeline,
// porting core patterns from IREE's OptimizeVectorTransfer, DropVectorUnitDims,
// and Hoisting passes.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::nova;

namespace {

//===----------------------------------------------------------------------===//
// NovaGPUOptimizeVectorTransferPass
//===----------------------------------------------------------------------===//

struct NovaGPUOptimizeVectorTransferPass
    : public PassWrapper<NovaGPUOptimizeVectorTransferPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUOptimizeVectorTransferPass)

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Basic transfer read/write optimization patterns.
    vector::populateSinkVectorOpsPatterns(patterns);
    vector::populateSinkVectorMemOpsPatterns(patterns);
    
    // Additional aggressive hoisting/folding
    (void)applyPatternsAndFoldGreedily(funcOp.getOperation(), std::move(patterns));

    // After patterns, run standard Linalg hoisting for redundant transfers.
    linalg::hoistRedundantVectorTransfers(funcOp);
  }

  StringRef getArgument() const override { return "nova-gpu-optimize-transfer"; }
  StringRef getDescription() const override {
    return "Optimize vector.transfer_read/write operations (hoisting, folding).";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUDropVectorUnitDimsPass
//===----------------------------------------------------------------------===//

struct NovaGPUDropVectorUnitDimsPass
    : public PassWrapper<NovaGPUDropVectorUnitDimsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUDropVectorUnitDimsPass)

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);

    (void)applyPatternsAndFoldGreedily(funcOp.getOperation(), std::move(patterns));
  }

  StringRef getArgument() const override { return "nova-gpu-drop-unit-dims"; }
  StringRef getDescription() const override {
    return "Drop unit dimensions from vector operations.";
  }
};
} // namespace

namespace mlir::nova {

std::unique_ptr<Pass> createNovaGPUOptimizeVectorTransferPass() {
  return std::make_unique<NovaGPUOptimizeVectorTransferPass>();
}
void registerNovaGPUOptimizeVectorTransferPass() {
  PassRegistration<NovaGPUOptimizeVectorTransferPass>();
}

std::unique_ptr<Pass> createNovaGPUDropVectorUnitDimsPass() {
  return std::make_unique<NovaGPUDropVectorUnitDimsPass>();
}
void registerNovaGPUDropVectorUnitDimsPass() {
  PassRegistration<NovaGPUDropVectorUnitDimsPass>();
}


} // namespace mlir::nova
