//===- NovaGPUVectorization.cpp - Nova GPU Vectorization Passes -----------===//
//
// Defines the vectorization and hoisting passes for the Nova GPU pipeline,
// following the IREE-style SIMD-to-SIMT strategy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// NovaGPUGenericVectorizationPass
//===----------------------------------------------------------------------===//

struct NovaGPUGenericVectorizationPass
    : public PassWrapper<NovaGPUGenericVectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUGenericVectorizationPass)
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    
    // 1. Vectorize Linalg operations and replace them with the results.
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      rewriter.setInsertionPoint(linalgOp);
      auto result = linalg::vectorize(rewriter, linalgOp);
      if (succeeded(result)) {
        rewriter.replaceOp(linalgOp, result->replacements);
      }
    });

    // 2. Flatten high-dimensional vectors to 1D to facilitate LLVM/GPU lowering.
    {
      RewritePatternSet patterns(context);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // Also lower contraction if any survived.
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
  StringRef getArgument() const override { return "nova-gpu-generic-vectorization"; }
  StringRef getDescription() const override {
    return "Vectorize linalg operations in the Nova GPU pipeline.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUSubsetHoistingPass
//===----------------------------------------------------------------------===//

struct NovaGPUSubsetHoistingPass
    : public PassWrapper<NovaGPUSubsetHoistingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUSubsetHoistingPass)
  void runOnOperation() override {
    auto funcOp = getOperation();
    // Hoist redundant transfer_read/transfer_write out of loops.
    linalg::hoistRedundantVectorTransfers(funcOp);
  }
  StringRef getArgument() const override { return "nova-gpu-subset-hoisting"; }
  StringRef getDescription() const override {
    return "Hoist redundant vector.transfer_read/write out of scf.for loops.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUVectorizeMemrefCopyPass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorizeMemrefCopyPass
    : public PassWrapper<NovaGPUVectorizeMemrefCopyPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorizeMemrefCopyPass)
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    MLIRContext *context = &getContext();
    funcOp.walk([&](memref::CopyOp copyOp) {
      (void)linalg::vectorizeCopy(rewriter, copyOp);
    });

    // Flatten high-dimensional vectors to 1D after bufferization.
    {
      RewritePatternSet patterns(context);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
  StringRef getArgument() const override { return "nova-gpu-vectorize-memref-copy"; }
  StringRef getDescription() const override {
    return "Vectorize memref.copy ops, especially between global and shared memory.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUUnrollToIntrinsicsPass
//===----------------------------------------------------------------------===//

struct NovaGPUUnrollToIntrinsicsPass
    : public PassWrapper<NovaGPUUnrollToIntrinsicsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUUnrollToIntrinsicsPass)
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Simple unrolling to 16x16x16 for Tensor Core compatibility proof-of-concept.
    vector::UnrollVectorOptions options;
    options.setNativeShape({16, 16, 16});
    options.setFilterConstraint([](Operation *op) {
      return success(isa<vector::ContractionOp>(op));
    });
    vector::populateVectorUnrollPatterns(patterns, options);
    
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  StringRef getArgument() const override { return "nova-gpu-unroll-to-intrinsics"; }
  StringRef getDescription() const override {
    return "Unroll vector operations to target-specific intrinsic sizes.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory Functions and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUGenericVectorizationPass() {
  return std::make_unique<NovaGPUGenericVectorizationPass>();
}
void registerNovaGPUGenericVectorizationPass() {
  PassRegistration<NovaGPUGenericVectorizationPass>();
}

std::unique_ptr<Pass> createNovaGPUSubsetHoistingPass() {
  return std::make_unique<NovaGPUSubsetHoistingPass>();
}
void registerNovaGPUSubsetHoistingPass() {
  PassRegistration<NovaGPUSubsetHoistingPass>();
}

std::unique_ptr<Pass> createNovaGPUVectorizeMemrefCopyPass() {
  return std::make_unique<NovaGPUVectorizeMemrefCopyPass>();
}
void registerNovaGPUVectorizeMemrefCopyPass() {
  PassRegistration<NovaGPUVectorizeMemrefCopyPass>();
}

std::unique_ptr<Pass> createNovaGPUUnrollToIntrinsicsPass() {
  return std::make_unique<NovaGPUUnrollToIntrinsicsPass>();
}
void registerNovaGPUUnrollToIntrinsicsPass() {
  PassRegistration<NovaGPUUnrollToIntrinsicsPass>();
}

} // namespace mlir::nova
