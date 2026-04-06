//===- NovaFuseReductionIntoProducer.cpp ----------------------------------===//  
//  
// Wrapper for nova elementwise fusion to prevent matmul from fusing  
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
  
#define DEBUG_TYPE "nova-fuse-element-wise-operations"  
  
using namespace mlir;  
using namespace mlir::linalg;  
  
namespace mlir::nova {  
  
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
      
    // Custom control function that skips matmul consumers  
    ControlFusionFn skipMatmulConsumerFn = [](OpOperand *fusedOperand) {  
      // Check if the CONSUMER is a matmul operation  
      auto consumer = dyn_cast<linalg::GenericOp>(fusedOperand->getOwner());  
      if (!consumer) return false;  
        
      // Skip matmul consumers  
      if (mlir::linalg::isaContractionOpInterface(consumer))  
        return false;  
        
      // Default behavior: fuse only if producer has one use  
      Operation *producer = fusedOperand->get().getDefiningOp();  
      return producer && producer->hasOneUse();  
    };  
      
    // Use standard MLIR fusion patterns with custom control function  
    linalg::populateElementwiseOpsFusionPatterns(patterns, skipMatmulConsumerFn);  
      
    // Use TopDownTraversal for better compile time  
    GreedyRewriteConfig config;  
    config.setUseTopDownTraversal();  
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))  
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