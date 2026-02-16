

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
namespace mlir {
namespace nova {

// functions which will be called inside template
struct NovaOpTosaOp {
  // helper function
  static SmallVector<int64_t>
  shapeFind(Type currType, int64_t axis) // if 2x3,axis=1 is given returns
  {
    SmallVector<int64_t>
        newshape; // paramters=>inputshape(auto) and axis(int32)
    auto rankedType = cast<RankedTensorType>(currType);
    for (int64_t i = 0; i < rankedType.getRank(); ++i) {
      if (i == axis) {
        newshape.push_back(1); // TOSA keeps reduced dimension as size 1
      } else {
        newshape.push_back(rankedType.getDimSize(i));
      }
    }
    return newshape;
  }
  // sigmoid
  static Value mappingtosa(nova::SigmoidOp op, Type resultType,
                           ValueRange input, OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType =
        mlir::RankedTensorType::get(v_type.getShape(), targetElemType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    return builder->create<tosa::SigmoidOp>(op.getLoc(), resultType, v);
  }

  // MAE lowering pattern
  static Value mappingtosa(nova::MaeOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType =
        mlir::RankedTensorType::get(v_type.getShape(), targetElemType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType =
        mlir::RankedTensorType::get(w_type.getShape(), targetElemType);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);
    // loss= reduce_mean(abs(arg0-arg1))
    auto sub = builder->create<nova::SubOp>(op.getLoc(), newVType, v, w);
    auto abs = builder->create<nova::AbsOp>(op.getLoc(), newVType, sub);
    nova::ReductionKind rk = nova::ReductionKind::MEAN;
    // only 2d for now.
    int64_t rank = cast<mlir::ShapedType>(abs.getType()).getRank();
    llvm::SmallVector<int64_t, 1> dimensions;
    if (rank > 0) {
      for (int64_t i = 0; i < rank; ++i) {
        dimensions.push_back(i);
      }
    }
    // Determine scalar type
    auto finalResultType = llvm::cast<RankedTensorType>(resultType);
    auto scalarType =
        RankedTensorType::get({1}, finalResultType.getElementType());

    // Reduce to scalar
    Value reducedLoss = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, abs, scalarType, false, dimensions);

    // Reshape to {1}
    llvm::SmallVector<int64_t> newShape = {1};
    auto shapeAttrType = RankedTensorType::get({1}, builder->getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeAttrType, newShape);

    Value shapeConst = builder->create<tosa::ConstShapeOp>(
        op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1),
        shapeAttr);

    return builder->create<tosa::ReshapeOp>(op.getLoc(), resultType,
                                            reducedLoss, shapeConst);
  }
  // MSE lowering pattern
  static Value mappingtosa(nova::MseOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // loss= reduce_mean(square(arg0-arg1))
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType =
        mlir::RankedTensorType::get(v_type.getShape(), targetElemType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType =
        mlir::RankedTensorType::get(w_type.getShape(), targetElemType);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);

    auto sub = builder->create<nova::SubOp>(op.getLoc(), newVType, v, w);
    mlir::RankedTensorType constType =
        mlir::RankedTensorType::get(v_type.getShape(), targetElemType);
    mlir::DenseElementsAttr constAttr = mlir::DenseElementsAttr::get(
        constType, builder->getFloatAttr(targetElemType, 2.0));
    auto constTwo =
        builder->create<nova::ConstantOp>(op.getLoc(), constType, constAttr);
    auto abs =
        builder->create<nova::PowOp>(op.getLoc(), newVType, sub, constTwo);

    nova::ReductionKind rk = nova::ReductionKind::MEAN;
    int64_t rank = cast<mlir::ShapedType>(abs.getType()).getRank();

    llvm::SmallVector<int64_t, 1> dimensions;
    if (rank > 0) {
      for (int64_t i = 0; i < rank; ++i) {
        dimensions.push_back(i);
      }
    }
    // Determine scalar type
    auto finalResultType = llvm::cast<RankedTensorType>(resultType);
    auto scalarType =
        RankedTensorType::get({1}, finalResultType.getElementType());

    // Reduce to scalar
    Value reducedLoss = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, abs, scalarType, false, dimensions);

    // Reshape to {1}
    llvm::SmallVector<int64_t> newShape = {1};
    auto shapeAttrType = RankedTensorType::get({1}, builder->getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeAttrType, newShape);

    Value shapeConst = builder->create<tosa::ConstShapeOp>(
        op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1),
        shapeAttr);

    return builder->create<tosa::ReshapeOp>(op.getLoc(), resultType,
                                            reducedLoss, shapeConst);
  }
  // CCE lowering pattern
  static Value mappingtosa(nova::CceOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // basic casting logic
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType =
        mlir::RankedTensorType::get(v_type.getShape(), targetElemType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType =
        mlir::RankedTensorType::get(w_type.getShape(), targetElemType);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);
    // step1:creating 1x10^-7  tensor constant
    auto hostVType =
        mlir::RankedTensorType::get(newVType.getShape(), targetElemType);

    auto epiAttr = DenseElementsAttr::get(
        hostVType, builder->getFloatAttr(targetElemType, 1e-7));
    Value epi =
        builder->create<nova::ConstantOp>(op.getLoc(), hostVType, epiAttr);

    // step2: creating one minus epsilon constant
    auto oneminusepiAttr = DenseElementsAttr::get(
        hostVType, builder->getFloatAttr(targetElemType, 1.0));
    Value ones = builder->create<nova::ConstantOp>(op.getLoc(), hostVType,
                                                   oneminusepiAttr);
    Value oneminusepi = builder->create<nova::SubOp>(op.getLoc(), ones, epi);
    // step3:creating compare op
    auto inputShape = cast<mlir::RankedTensorType>(v.getType()).getShape();
    // Get the boolean element type (i1)
    auto boolType = builder->getI1Type();
    auto compareResultType = mlir::RankedTensorType::get(inputShape, boolType);
    auto ck = nova::ComparisonType::LT;
    auto compare = builder->create<nova::CompareOp>(
        op.getLoc(), compareResultType, v, epi, ck);
    auto cp =
        builder->create<tosa::SelectOp>(op.getLoc(), newVType, compare, epi, v);
    // step4:second compare
    auto ck1 = nova::ComparisonType::GT;
    auto compare1 = builder->create<nova::CompareOp>(
        op.getLoc(), compareResultType, cp, oneminusepi, ck1);
    auto cp1 = builder->create<tosa::SelectOp>(op.getLoc(), newVType, compare1,
                                               oneminusepi, cp);
    // step5 : target *log(cp)
    auto log = builder->create<nova::LogOp>(op.getLoc(), cp1);
    auto mul = builder->create<nova::MulOp>(op.getLoc(), log, w);
    // step6:create -1 constant tensor (scalar)
    auto constType = mlir::RankedTensorType::get({}, targetElemType);
    auto minus1Attr = DenseElementsAttr::get(
        constType, builder->getFloatAttr(targetElemType, -1.0));
    Value minus1 =
        builder->create<nova::ConstantOp>(op.getLoc(), constType, minus1Attr);
    // step 7 :reducesum(log result) along expect 0
    auto inputTensorType = cast<mlir::RankedTensorType>(mul.getType());
    int64_t inputRank = inputTensorType.getRank();
    llvm::SmallVector<int64_t, 4> newShape;
    newShape.push_back(inputTensorType.getDimSize(0));
    // reducing along all axis expect zero
    llvm::SmallVector<int64_t, 4> dimensions;
    for (int64_t i = 1; i < inputRank; ++i) {
      dimensions.push_back(i);
      // newShape.push_back(1);
    }
    auto reducedResultType =
        mlir::RankedTensorType::get(newShape, targetElemType);

    nova::ReductionKind rk = nova::ReductionKind::SUM;
    auto reduceres = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, mul, reducedResultType, false, dimensions);
    // Determine scalar type
    auto finalResultType = llvm::cast<RankedTensorType>(resultType);
    auto scalarType =
        RankedTensorType::get({1}, finalResultType.getElementType());

    auto reducemeanres =
        builder->create<nova::ReduceOp>(op.getLoc(), rk, reduceres, scalarType);
    // step 9: mul reduce result and -1
    Value multiplied =
        builder->create<nova::MulOp>(op.getLoc(), reducemeanres, minus1);

    // Reshape to {1}
    llvm::SmallVector<int64_t> newShape1 = {1};
    auto shapeAttrType = RankedTensorType::get({1}, builder->getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeAttrType, newShape1);

    Value shapeConst = builder->create<tosa::ConstShapeOp>(
        op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1),
        shapeAttr);

    return builder->create<tosa::ReshapeOp>(op.getLoc(), resultType, multiplied,
                                            shapeConst);
  }

  // BCE lowering pattern
  static Value mappingtosa(nova::BceOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // basic casting logic
    auto restensor = cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType =
        mlir::RankedTensorType::get(v_type.getShape(), targetElemType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType =
        mlir::RankedTensorType::get(w_type.getShape(), targetElemType);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);

    // step1:creating 1x10^-7  tensor constant
    auto hostVType =
        mlir::RankedTensorType::get(newVType.getShape(), targetElemType);

    auto epiAttr = DenseElementsAttr::get(
        hostVType, builder->getFloatAttr(targetElemType, 1e-7));
    Value epi =
        builder->create<nova::ConstantOp>(op.getLoc(), hostVType, epiAttr);

    // step2: creating one minus epsilon constant
    auto oneminusepiAttr = DenseElementsAttr::get(
        hostVType, builder->getFloatAttr(targetElemType, 1.0));
    Value ones = builder->create<nova::ConstantOp>(op.getLoc(), hostVType,
                                                   oneminusepiAttr);
    Value oneminusepi = builder->create<nova::SubOp>(op.getLoc(), ones, epi);
    // step3:creating compare op
    auto inputShape = cast<mlir::RankedTensorType>(v.getType()).getShape();
    // Get the boolean element type (i1)
    auto boolType = builder->getI1Type();
    auto compareResultType = mlir::RankedTensorType::get(inputShape, boolType);
    auto ck = nova::ComparisonType::LT;
    auto compare = builder->create<nova::CompareOp>(
        op.getLoc(), compareResultType, v, epi, ck);
    auto cp =
        builder->create<tosa::SelectOp>(op.getLoc(), newVType, compare, epi, v);
    // step4:second compare
    auto ck1 = nova::ComparisonType::GT;
    auto compare1 = builder->create<nova::CompareOp>(
        op.getLoc(), compareResultType, cp, oneminusepi, ck1);
    auto cp1 = builder->create<tosa::SelectOp>(op.getLoc(), newVType, compare1,
                                               oneminusepi, cp);
    // step5 : temr1=target *log(cp)
    auto log = builder->create<nova::LogOp>(op.getLoc(), cp1);
    auto term1 = builder->create<nova::MulOp>(op.getLoc(), log, w);

    // step6:find term2=(ones-arg1)*log(ones-clipped predicts)
    // ones-arg1
    auto termonelhs = builder->create<nova::SubOp>(op.getLoc(), ones, w);
    auto termtworhs = builder->create<nova::SubOp>(op.getLoc(), ones, cp1);
    auto termtwologrhs = builder->create<nova::LogOp>(op.getLoc(), termtworhs);
    auto term2 =
        builder->create<nova::MulOp>(op.getLoc(), termonelhs, termtwologrhs);
    // step7 :find sum terms +term1+term2
    auto sumterms = builder->create<nova::AddOp>(op.getLoc(), term1, term2);
    // step 8 :reducemean(sum result) full reduction
    auto inputTensorType = cast<mlir::RankedTensorType>(term1.getType());
    int64_t inputRank = inputTensorType.getRank();
    llvm::SmallVector<int64_t, 4> dimensions;
    for (int64_t i = 0; i < inputRank; ++i) {
      dimensions.push_back(i);
    }
    // Determine scalar type
    auto finalResultType = llvm::cast<RankedTensorType>(resultType);
    auto scalarType =
        RankedTensorType::get({1}, finalResultType.getElementType());

    // reducing along all axis
    auto rk = nova::ReductionKind::MEAN;
    auto reducemeanres = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, sumterms, scalarType, false, dimensions);

    // restore -1 constant for BCE/CCE
    auto constType = mlir::RankedTensorType::get({}, targetElemType);
    auto minus1Attr = DenseElementsAttr::get(
        constType, builder->getFloatAttr(targetElemType, -1.0));
    Value minus1 =
        builder->create<nova::ConstantOp>(op.getLoc(), constType, minus1Attr);
    // multiply by -1
    Value multiplied =
        builder->create<nova::MulOp>(op.getLoc(), reducemeanres, minus1);

    // Reshape to {1}
    llvm::SmallVector<int64_t> newShape1 = {1};
    auto shapeAttrType = RankedTensorType::get({1}, builder->getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeAttrType, newShape1);

    Value shapeConst = builder->create<tosa::ConstShapeOp>(
        op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1),
        shapeAttr);

    return builder->create<tosa::ReshapeOp>(op.getLoc(), resultType, multiplied,
                                            shapeConst);
  }
  // SCE lOWERING  pattern
// static Value mappingtosa(nova::SceOp op, Type resultType, ValueRange input,
//                            OpBuilder *builder) {
//     // Input[0] = logits (e.g., tensor<4x10xf32>)
//     // Input[1] = targets (e.g., tensor<4xi32>)
//     Value logits = input[0];
//     Value targets = input[1];

//     // Ensure targets are i32
//     auto targetsType = cast<mlir::RankedTensorType>(targets.getType());
//     if (isa<mlir::FloatType>(targetsType.getElementType())) {
//       auto newTargetsType = mlir::RankedTensorType::get(targetsType.getShape(),
//                                                         builder->getI32Type());
//       targets =
//           builder->create<tosa::CastOp>(op.getLoc(), newTargetsType, targets);
//       targetsType = newTargetsType;
//     }

//     auto restensor = cast<mlir::RankedTensorType>(resultType);
//     auto targetElemType = restensor.getElementType();

//     auto logitsType = cast<mlir::RankedTensorType>(logits.getType());

//     // Ensure logits match the target element type
//     if (logitsType.getElementType() != targetElemType) {
//       auto newLogitsType =
//           mlir::RankedTensorType::get(logitsType.getShape(), targetElemType);
//       logits =
//           builder->create<tosa::CastOp>(op.getLoc(), newLogitsType, logits);
//       logitsType = newLogitsType;
//     }

//     // Step 1: max_val = reduce_max(logits, dim=-1, keepdims=true)
//     int64_t rank = logitsType.getRank();
//     int64_t lastDim = rank - 1;
//     auto axisAttr = builder->getI32IntegerAttr(lastDim);

//     auto maxShape = logitsType.getShape().vec();
//     maxShape[lastDim] = 1;
//     auto maxValType = mlir::RankedTensorType::get(maxShape, targetElemType);

//     Value maxVal = builder->create<tosa::ReduceMaxOp>(op.getLoc(), maxValType,
//                                                       logits, axisAttr);

//     // Step 2: z_shifted = logits - max_val
//     Value zShifted = builder->create<nova::SubOp>(op.getLoc(), logits, maxVal);

//     // Step 3: exp_z_shifted = exp(z_shifted)
//     Value expZShifted = builder->create<nova::ExpOp>(op.getLoc(), zShifted);

//     // Step 4: sum_exp = reduce_sum(exp_z_shifted, dim=-1, keepdims=true)
//     Value sumExp = builder->create<tosa::ReduceSumOp>(op.getLoc(), maxValType,
//                                                       expZShifted, axisAttr);

//     // Step 5: log_sum_exp = log(sum_exp)
//     Value logSumExp = builder->create<nova::LogOp>(op.getLoc(), sumExp);

//     // Step 6: log_sm_Z = z_shifted - log_sum_exp (log-softmax)
//     Value logSmZ =
//         builder->create<nova::SubOp>(op.getLoc(), zShifted, logSumExp);

//     // Step 7: Gather using linalg.generic since TOSA gather has shape
//     // constraints selected_log_probs[i] = log_sm_Z[i, targets[i]]
//     // Use lastDim as gather axis (e.g., C)
//     Value selectedLogProbs =
//         builder->create<nova::GatherOp>(op.getLoc(), logSmZ, targets, lastDim)
//             .getResult();

//     // Step 8: loss = reduce_mean(selected_log_probs * -1.0)
//     // Create -1.0 constant
//     auto constType = mlir::RankedTensorType::get({}, targetElemType);
//     auto minus1Attr = DenseElementsAttr::get(
//         constType, builder->getFloatAttr(targetElemType, -1.0));
//     Value minus1 =
//         builder->create<nova::ConstantOp>(op.getLoc(), constType, minus1Attr);

//     // Multiply selected_log_probs by -1
//     Value negLogProbs =
//         builder->create<nova::MulOp>(op.getLoc(), selectedLogProbs, minus1);

//     // Reduce mean over all dimensions
//     auto rk = nova::ReductionKind::MEAN;
//     llvm::SmallVector<int64_t, 2> dimensions;
//     auto probsType = llvm::cast<RankedTensorType>(negLogProbs.getType());
//     for (int64_t i = 0; i < probsType.getRank(); ++i)
//       dimensions.push_back(i);

//     // Determine the scalar type based on the result type
//     auto finalResultType = llvm::cast<RankedTensorType>(resultType);
//     auto scalarType =
//         RankedTensorType::get({1}, finalResultType.getElementType());

//     // Perform reduction to scalar
//     Value reducedLoss = builder->create<nova::ReduceOp>(
//         op.getLoc(), rk, negLogProbs, scalarType, false, dimensions);

//     // Reshape scalar to 1D tensor (as expected by the new return type)
//     llvm::SmallVector<int64_t> newShape = {1};
//     auto shapeAttrType = RankedTensorType::get({1}, builder->getIndexType());
//     auto shapeAttr = DenseIntElementsAttr::get(shapeAttrType, newShape);

//     Value shapeConst = builder->create<tosa::ConstShapeOp>(
//         op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1),
//         shapeAttr);

//     Value finalLoss = builder->create<tosa::ReshapeOp>(op.getLoc(), resultType,
//                                                        reducedLoss, shapeConst);

//     return finalLoss;
//   }
//   // Cast lowering pattern
  static Value mappingtosa(nova::CastOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto inputType = cast<RankedTensorType>(input[0].getType());
    auto outputType = cast<RankedTensorType>(resultType);
    auto inputElemType = inputType.getElementType();
    auto outputElemType = outputType.getElementType();

    // Special case: bool (i1) to float (f16, f32, f64)
    // TOSA cast doesn't support i1 to float directly.
    // We do i1 -> i32 -> float
    if (inputElemType.isInteger(1) && isa<FloatType>(outputElemType)) {
      auto i32Type = builder->getI32Type();
      auto intermediateType =
          RankedTensorType::get(inputType.getShape(), i32Type);
      auto intermediateCast = builder->create<tosa::CastOp>(
          op.getLoc(), intermediateType, input[0]);
      return builder->create<tosa::CastOp>(op.getLoc(), outputType,
                                           intermediateCast);
    }

    return builder->create<tosa::CastOp>(op.getLoc(), resultType, input[0]);
  }

  template <typename OpTy>
  static Value maptop(OpTy op, Type resultType, ValueRange input,
                      OpBuilder *builder) {
    return mappingtosa(op, resultType, input, builder);
  }

private:
  template <typename OpTy>
  static Value mappingtosa(OpTy op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    return nullptr;
  }
};

// pattern to convert nova.gelu to seauence of operations
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
struct NovaGeluOpLowering : public OpConversionPattern<mlir::nova::GeluOp> {
  using OpConversionPattern<mlir::nova::GeluOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    // if input is integer, cast to float and update type for following ops
    auto inputType = cast<RankedTensorType>(input.getType());
    if (isa<IntegerType>(inputType.getElementType())) {
      auto newInputType =
          RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
      input = rewriter.create<mlir::tosa::CastOp>(loc, newInputType, input);
      inputType = newInputType;
    }
    // op0 = pow(x, 3)
    Value cst_3 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {3.0f}));
    auto op0 = rewriter.create<mlir::nova::PowOp>(loc, inputType, input, cst_3);
    // op1 = mul(op0, 0.044715)
    Value cst_004 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {4.471500e-02f}));
    auto op1 = rewriter.create<mlir::nova::MulOp>(loc, inputType, op0, cst_004);
    // op2 = add(x, op1)
    auto op2 = rewriter.create<mlir::nova::AddOp>(loc, inputType, input, op1);
    // op3 = mul(op2, sqrt(2/pi))
    Value cst_sqrt2pi = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {0.797884583f}));
    auto op3 =
        rewriter.create<mlir::nova::MulOp>(loc, inputType, op2, cst_sqrt2pi);
    // op4 = tanh(op3)
    auto op4 = rewriter.create<mlir::nova::TanhOp>(loc, inputType, op3);
    // op5 = add(op4 ,1)
    Value cst_1 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {1.0f}));
    auto op5 = rewriter.create<mlir::nova::AddOp>(loc, inputType, op4, cst_1);
    // op6 = mul(x, 0.5)
    Value cst_05 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {0.5f}));
    auto op6 =
        rewriter.create<mlir::nova::MulOp>(loc, inputType, input, cst_05);

    auto op7 = rewriter.create<mlir::nova::MulOp>(loc, inputType, op6, op5);

    rewriter.replaceOp(op, {op7.getResult()});

    return success();
  }
};
// relu lowering
struct NovaReluOpLowering : public OpConversionPattern<mlir::nova::ReluOp> {
  using OpConversionPattern<mlir::nova::ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    Type elementType = inputType.getElementType();

    // Create zero constant tensor with the same shape as input
    Attribute zeroAttr;

    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      APFloat zeroVal = APFloat::getZero(floatType.getFloatSemantics());
      zeroAttr = rewriter.getFloatAttr(floatType, zeroVal);
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      zeroAttr = rewriter.getIntegerAttr(intType, 0);
    } else {
      return failure();
    }
    DenseElementsAttr zeroTensor = DenseElementsAttr::get(inputType, zeroAttr);
    Value zero =
        rewriter.create<mlir::nova::ConstantOp>(loc, inputType, zeroTensor);
    Value result =
        rewriter.create<mlir::nova::MaxOp>(loc, inputType, input, zero);

    rewriter.replaceOp(op, result);
    return success();
  }
};
// creating a  lowering for softmax
struct NovaSoftmaxLoweringPattern
    : public OpConversionPattern<mlir::nova::SoftmaxOp> {
  using OpConversionPattern<mlir::nova::SoftmaxOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto restype = cast<RankedTensorType>(op.getType());
    auto inputType = cast<RankedTensorType>(input.getType());
    auto size = inputType.getShape().size();
    int32_t dimension =
        op.getDimension().has_value() ? op.getDimension().value() : -1;
    if (dimension < 0) {
      //  auto size1=inputType.getShape().size() - 1;
      dimension += size;
    }
    SmallVector<int64_t> dim;
    dim.push_back(dimension);

    auto shape = NovaOpTosaOp::shapeFind(inputType, dimension);
    auto tempresult = RankedTensorType::get(shape, restype.getElementType());
    // creating cast - only if element types differ
    if (inputType.getElementType() != restype.getElementType()) {
      input = rewriter.create<mlir::tosa::CastOp>(loc, restype, input);
    }

    auto axisAttr = rewriter.getI32IntegerAttr(dimension);
    Value op1 = rewriter.create<mlir::tosa::ReduceMaxOp>(loc, tempresult, input,
                                                         axisAttr);

    // step2
    // create a TOSA sub op with input and op1
    Value op2 = rewriter.create<mlir::nova::SubOp>(loc, restype, input, op1);
    // step3
    // create  a TOSA exp op
    Value op3 = rewriter.create<mlir::nova::ExpOp>(loc, restype, op2);
    // step4
    // create a TOSA reduce sum
    Value op4 = rewriter.create<mlir::tosa::ReduceSumOp>(loc, tempresult, op3,
                                                         axisAttr);

    // step 5
    // Explicitly broadcast op4 to match op3's shape for division
    // op3 is tensor<8x128x128xf32>, op4 is tensor<8x128x1xf32>
    // We need to tile op4 along dimension 2 to broadcast it
    SmallVector<int64_t> multiples;
    auto op3Shape = cast<RankedTensorType>(op3.getType()).getShape();
    auto op4Shape = cast<RankedTensorType>(op4.getType()).getShape();
    for (size_t i = 0; i < op3Shape.size(); ++i) {
      if (i < op4Shape.size()) {
        multiples.push_back(op3Shape[i] / op4Shape[i]);
      } else {
        multiples.push_back(1);
      }
    }

    auto shapeType = RankedTensorType::get(
        {static_cast<int64_t>(multiples.size())}, rewriter.getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, multiples);
    Value multiplesConst = rewriter.create<mlir::tosa::ConstShapeOp>(
        loc,
        mlir::tosa::shapeType::get(rewriter.getContext(), multiples.size()),
        shapeAttr);

    Value op4_broadcast =
        rewriter.create<mlir::tosa::TileOp>(loc, restype, op4, multiplesConst);

    // create TOSA div: reciprocal(op4_broadcast) * op3
    Value recip =
        rewriter.create<mlir::nova::ReciprocalOp>(loc, restype, op4_broadcast);
    Value op5 = rewriter.create<mlir::nova::MulOp>(loc, restype, op3, recip);
    rewriter.replaceOp(op, op5);

    return success();
  }
};

// creating a template
template <typename NovaTopTy>
class NovaToTosaLoweringTemplate : public OpConversionPattern<NovaTopTy> {
public:
  using OpConversionPattern<NovaTopTy>::OpConversionPattern;
  using OpAdaptor = typename NovaTopTy::Adaptor; // for getting all meta data
                                                 // dynamically using adaptor
  LogicalResult
  matchAndRewrite(NovaTopTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
    // checking operand is empty or not
    if (operands.empty())
      return rewriter.notifyMatchFailure(
          op, "expected operands for tosa lowering operations");
    // getting resultType
    auto resultType = op.getResult().getType();
    Value result = NovaOpTosaOp::maptop(op, resultType, operands, &rewriter);
    if (!result)
      return rewriter.notifyMatchFailure(op, "failed to map to TOSA operation");

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaConstantToArithConstPattern
    : public OpConversionPattern<nova::ConstantOp> {
  using OpConversionPattern<nova::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ElementsAttr valueAttr = op.getValue();
    DenseElementsAttr value = dyn_cast<DenseElementsAttr>(valueAttr);

    auto outputType = cast<RankedTensorType>(op.getOutput().getType());
    auto hostOutputType = RankedTensorType::get(outputType.getShape(),
                                                outputType.getElementType());
    auto hostValue = value.reshape(hostOutputType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, hostOutputType,
                                                   hostValue);
    return success();
  }
};

// pass definition
namespace {
struct NovaToTosaLoweringPass
    : public PassWrapper<NovaToTosaLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToTosaLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<nova::NovaDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<complex::ComplexDialect>();
    registry.insert<arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "convert-nova-to-tosa"; }

  StringRef getDescription() const final {
    return "Lower Nova dialect operations to Tosa dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           complex::ComplexDialect, arith::ArithDialect>();
    target.addIllegalOp<nova::ConstantOp>();
    target.addIllegalOp<nova::ReluOp>();
    target.addIllegalOp<nova::MseOp>();
    target.addIllegalOp<nova::CceOp>();
    target.addIllegalOp<nova::SigmoidOp>();
    target.addIllegalOp<nova::GeluOp>();
    target.addIllegalOp<nova::SoftmaxOp>();
    target.addIllegalOp<nova::BceOp>();

    target.addIllegalOp<nova::MaeOp>();
    target.addIllegalOp<nova::CastOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    mlir::RewritePatternSet patterns(&getContext());
    mlir::nova::populateNovaToTosaConversionPatterns(patterns);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void populateNovaToTosaConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaReluOpLowering, NovaGeluOpLowering,
               NovaSoftmaxLoweringPattern, NovaConstantToArithConstPattern,
               NovaToTosaLoweringTemplate<nova::MaeOp>,
               NovaToTosaLoweringTemplate<nova::MseOp>,
               NovaToTosaLoweringTemplate<nova::CceOp>,
               NovaToTosaLoweringTemplate<nova::BceOp>,

               NovaToTosaLoweringTemplate<nova::SigmoidOp>,
               NovaToTosaLoweringTemplate<nova::CastOp>>(patterns.getContext());
}

// creating a pointer for this pass
std::unique_ptr<Pass> createNovaToTosaLoweringPass() {
  return std::make_unique<NovaToTosaLoweringPass>();
}

// Register the pass
void registerNovaToTosaLoweringPass() {
  PassRegistration<NovaToTosaLoweringPass>();
}

} // namespace nova
} // namespace mlir
