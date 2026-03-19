//===- NovaToArithLowering.cpp - Lower Nova dialect to Arith dialect ------===//
//
// This file implements a pass to lower Nova dialect operations to Arith dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    auto elementsAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (elementsAttr && elementsAttr.isSplat()) {
      auto splatVal = elementsAttr.getSplatValue<Attribute>();
      bool isZero = false;
      if (auto floatAttr = dyn_cast<FloatAttr>(splatVal)) {
        if (floatAttr.getValueAsDouble() == 0.0)
          isZero = true;
      } else if (auto intAttr = dyn_cast<IntegerAttr>(splatVal)) {
        if (intAttr.getInt() == 0)
          isZero = true;
      }

      if (isZero) {
        auto tensorType = cast<RankedTensorType>(resultType);
        Location loc = op.getLoc();
        // 1. Create tensor.empty to represent a destination buffer.
        // Bufferization will later map this to a GPU allocation (addrspace 1).
        Value empty = builder->create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType());

        // 2. Create the scalar zero to fill the tensor with.
        Value zero;
        if (auto floatType = dyn_cast<FloatType>(tensorType.getElementType())) {
          zero = builder->create<arith::ConstantOp>(
              loc, builder->getFloatAttr(floatType, 0.0));
        } else {
          zero = builder->create<arith::ConstantOp>(
              loc, builder->getIntegerAttr(tensorType.getElementType(), 0));
        }

        // 3. Use linalg.fill to zero-initialize the tensor on the device.
        return builder->create<linalg::FillOp>(loc, zero, empty).result();
      }
    }
    return builder ->create<arith::ConstantOp>(op.getLoc(),resultType,op.getValue());
  }
   // SCE lOWERING  pattern
  // Computes: loss = mean(log(sum(exp(logits - max))) + max - logits[targets])
  // The gather reads from the ORIGINAL logits (function argument) to avoid
  // tensor.extract cross-forall dominance issues in the tiling pass.
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

    // Step 1: max_val = reduce_max(logits, dim=-1, keepdims=false)
    int64_t rank = logitsType.getRank();
    int64_t lastDim = rank - 1;\

    // keepdims=true shape for broadcast in sub/exp
    auto keepShape = logitsType.getShape().vec();
    keepShape[lastDim] = 1;
    auto keepType = mlir::RankedTensorType::get(keepShape, targetElemType);
 //h : create nova reduce max
  nova::ReductionKind rk = nova::ReductionKind::MAX;
    Value maxValKeep = builder->create<nova::ReduceOp>(op.getLoc(),rk,
                                                          logits,keepType, true,llvm::ArrayRef<int64_t>{-1},false);

    // Step 2: z_shifted = logits - max_val (broadcast sub)
    Value zShifted =
        builder->create<nova::SubOp>(op.getLoc(), logits, maxValKeep);

    // Step 3: exp(z_shifted)
    Value expZShifted = builder->create<nova::ExpOp>(op.getLoc(), zShifted);

    // Step 4: sum_exp = reduce_sum(exp_z_shifted, dim=-1, keepdims=false)
    // Use keepdims=false for the flat per-sample values
    auto batchShape = logitsType.getShape().vec();
    batchShape.pop_back(); // remove last dim → e.g. [4]
    auto batchType =
        mlir::RankedTensorType::get(batchShape, targetElemType);

    auto rk_sum = nova::ReductionKind::SUM;
    llvm::SmallVector<int64_t, 1> reduceDims = {lastDim};
    Value sumExp = builder->create<nova::ReduceOp>(
        op.getLoc(), rk_sum, expZShifted, batchType, false, reduceDims);



    // Step 6: max_val flat (keepdims=false) → tensor<4xf32>
    auto rk_max = nova::ReductionKind::MAX;
    Value maxValFlat = builder->create<nova::ReduceOp>(
        op.getLoc(), rk_max, logits, batchType, false, reduceDims);

    // Step 7: Gather target class logit: gathered[b,t] = logits[b, t, targets[b,t]]
    // Gather along the LAST axis (the class dimension).
    Value gatheredLogits =
        builder->create<nova::GatherOp>(op.getLoc(), logits, targets, lastDim)
            .getResult();

    // Step 5: log_sum_exp = log(sum_exp) → tensor<4xf32>
    Value logSumExp = builder->create<nova::LogOp>(op.getLoc(), sumExp);
    // Step 8: per_sample_loss = log_sum_exp + max_val - gathered_logits
    Value lseMaxSum =
        builder->create<nova::AddOp>(op.getLoc(), logSumExp, maxValFlat);
    Value perSampleLoss =
        builder->create<nova::SubOp>(op.getLoc(), lseMaxSum, gatheredLogits);

    // Step 9: loss = reduce_mean(per_sample_loss)
    auto rk_mean = nova::ReductionKind::MEAN;
    llvm::SmallVector<int64_t, 1> allDims;
    auto pslType = llvm::cast<RankedTensorType>(perSampleLoss.getType());
    for (int64_t i = 0; i < pslType.getRank(); ++i)
      allDims.push_back(i);

    auto finalResultType = llvm::cast<RankedTensorType>(resultType);
    auto scalarType =
        RankedTensorType::get({1}, finalResultType.getElementType());

    Value reducedLoss = builder->create<nova::ReduceOp>(
        op.getLoc(), rk_mean, perSampleLoss, scalarType, false, allDims);

    // Reshape to match expected return type
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
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
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
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
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