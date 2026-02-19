//===- NovaToArithLowering.cpp - Lower Nova dialect to Arith dialect ------===//
//
// This file implements a pass to lower Nova dialect operations to Arith dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"


#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace nova {

//--------------------------------------constant-----------------------------

struct NovaToArithOp{
  template <typename OpTy>
  static Value maptop(OpTy op,Type resultType,ValueRange input,OpBuilder* builder){
    return mappingArith(op,resultType,input,builder);
  } 

  private:
  template <typename OpTy>
  static Value mappingArith(OpTy op,Type resultType,ValueRange input,OpBuilder* builder){
  return nullptr;
  }
  static Value mappingArith(nova::ConstantOp op,Type resultType,ValueRange input,OpBuilder* builder){
    return builder ->create<arith::ConstantOp>(op.getLoc(),resultType,op.getValue());
  }
   // SCE lOWERING  pattern
  static Value mappingArith(nova::SceOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // Input[0] = logits (e.g., tensor<4x10xf32>)
    // Input[1] = targets (e.g., tensor<4xi32>)
    Value logits = input[0];
    Value targets = input[1];

    // Ensure targets are i32
    auto targetsType = cast<mlir::RankedTensorType>(targets.getType());
    if (isa<mlir::FloatType>(targetsType.getElementType())) {
      auto newTargetsType = mlir::RankedTensorType::get(targetsType.getShape(),
                                                        builder->getI32Type());
      targets =
          builder->create<tosa::CastOp>(op.getLoc(), newTargetsType, targets);
      targetsType = newTargetsType;
    }

    auto restensor = cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();

    auto logitsType = cast<mlir::RankedTensorType>(logits.getType());

    // Ensure logits match the target element type
    if (logitsType.getElementType() != targetElemType) {
      auto newLogitsType =
          mlir::RankedTensorType::get(logitsType.getShape(), targetElemType);
      logits =
          builder->create<tosa::CastOp>(op.getLoc(), newLogitsType, logits);
      logitsType = newLogitsType;
    }

    // Step 1: max_val = reduce_max(logits, dim=-1, keepdims=true)
    int64_t rank = logitsType.getRank();
    int64_t lastDim = rank - 1;
    auto axisAttr = builder->getI32IntegerAttr(lastDim);

    auto maxShape = logitsType.getShape().vec();
    maxShape[lastDim] = 1;
    auto maxValType = mlir::RankedTensorType::get(maxShape, targetElemType);

    Value maxVal = builder->create<tosa::ReduceMaxOp>(op.getLoc(), maxValType,
                                                      logits, axisAttr);

    // Step 2: z_shifted = logits - max_val
    Value zShifted = builder->create<nova::SubOp>(op.getLoc(), logits, maxVal);

    // Step 3: exp_z_shifted = exp(z_shifted)
    Value expZShifted = builder->create<nova::ExpOp>(op.getLoc(), zShifted);

    // Step 4: sum_exp = reduce_sum(exp_z_shifted, dim=-1, keepdims=true)
    Value sumExp = builder->create<tosa::ReduceSumOp>(op.getLoc(), maxValType,
                                                      expZShifted, axisAttr);

    // Step 5: log_sum_exp = log(sum_exp)
    Value logSumExp = builder->create<nova::LogOp>(op.getLoc(), sumExp);

    // Step 6: log_sm_Z = z_shifted - log_sum_exp (log-softmax)
    Value logSmZ =
        builder->create<nova::SubOp>(op.getLoc(), zShifted, logSumExp);

    // Step 7: Gather using linalg.generic since TOSA gather has shape
    // constraints selected_log_probs[i] = log_sm_Z[i, targets[i]]
    // Use lastDim as gather axis (e.g., C)
    Value selectedLogProbs =
        builder->create<nova::GatherOp>(op.getLoc(), logSmZ, targets, lastDim)
            .getResult();

    // Step 8: loss = reduce_mean(selected_log_probs * -1.0)
    // Create -1.0 constant
    auto constType = mlir::RankedTensorType::get({}, targetElemType);
    auto minus1Attr = DenseElementsAttr::get(
        constType, builder->getFloatAttr(targetElemType, -1.0));
    Value minus1 =
        builder->create<nova::ConstantOp>(op.getLoc(), constType, minus1Attr);

    // Multiply selected_log_probs by -1
    Value negLogProbs =
        builder->create<nova::MulOp>(op.getLoc(), selectedLogProbs, minus1);

    // Reduce mean over all dimensions
    auto rk = nova::ReductionKind::MEAN;
    llvm::SmallVector<int64_t, 2> dimensions;
    auto probsType = llvm::cast<RankedTensorType>(negLogProbs.getType());
    for (int64_t i = 0; i < probsType.getRank(); ++i)
      dimensions.push_back(i);

    // Determine the scalar type based on the result type
    auto finalResultType = llvm::cast<RankedTensorType>(resultType);
    auto scalarType =
        RankedTensorType::get({1}, finalResultType.getElementType());

    // Perform reduction to scalar
    Value reducedLoss = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, negLogProbs, scalarType, false, dimensions);

    // Reshape scalar to 1D tensor (as expected by the new return type)
    llvm::SmallVector<int64_t> newShape = {1};
    auto shapeAttrType = RankedTensorType::get({1}, builder->getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeAttrType, newShape);

    Value shapeConst = builder->create<tosa::ConstShapeOp>(
        op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1),
        shapeAttr);

    Value finalLoss = builder->create<tosa::ReshapeOp>(op.getLoc(), resultType,
                                                       reducedLoss, shapeConst);

    return finalLoss;
};

};
template <typename NovaArithOp>
class NovaArithConversionPattern : public OpConversionPattern<NovaArithOp>{
   public :
   using OpConversionPattern<NovaArithOp> :: OpConversionPattern;
   using OpAdaptor = typename NovaArithOp::Adaptor;

   LogicalResult matchAndRewrite(NovaArithOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();

    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if(!resultType)
    return rewriter.notifyMatchFailure(op,"Needs Tensor Type");

    Value result = NovaToArithOp::maptop(op,resultType,operands,&rewriter);
    if (!result)
      return rewriter.notifyMatchFailure(op, "Mapping to Arith failed or returned null");
    rewriter.replaceOp(op,result);
    return success();
   }
};


// Pass Definition

namespace {
struct NovaToArithLoweringPass
    : public PassWrapper<NovaToArithLoweringPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToArithLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "convert-nova-to-arith"; }
  
  StringRef getDescription() const final {
    return "Lower Nova dialect operations to Arith dialect";
  }
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<nova::ConstantOp>();
    target.addIllegalOp<nova::SceOp>();

    RewritePatternSet patterns(&getContext());

    
    populateNovaToArithConversionPatterns(patterns);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void populateNovaToArithConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaArithConversionPattern<nova::ConstantOp>>(
    patterns.getContext()
  );
  patterns.add<NovaArithConversionPattern<nova::SceOp>>(
    patterns.getContext()
  );
}

std::unique_ptr<Pass> createNovaToArithLoweringPass() {
  return std::make_unique<NovaToArithLoweringPass>();
}

// Register the pass
void registerNovaToArithLoweringPass() {
  PassRegistration<NovaToArithLoweringPass>();
}

} // namespace nova
} // namespace mlir