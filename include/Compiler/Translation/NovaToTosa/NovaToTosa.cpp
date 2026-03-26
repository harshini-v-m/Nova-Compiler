

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
    // op0 = xxx
    Value xsquare=rewriter.create<mlir::nova::MulOp>(loc,inputType,input,input);
    Value op0=rewriter.create<mlir::nova::MulOp>(loc,inputType,xsquare,input);
    // op1 = mul(op0, 0.044715)
    Value cst_004 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {4.471500e-02f}));
    auto op1 = rewriter.create<mlir::nova::MulOp>(loc, inputType, op0, cst_004);
    // op2 = add(x, op1)
    auto op2 = rewriter.create<mlir::nova::AddOp>(loc, inputType, input, op1);
    // op3 = mul(op2, sqrt(2/pi))
    Value cst_sqrt2pi = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {0.7978845608028654f}));
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

struct NovaGeluBackwardPattern
    : public OpConversionPattern<mlir::nova::GeluBackwardOp> {
  using OpConversionPattern<mlir::nova::GeluBackwardOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::GeluBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value grad_out = adaptor.getGradOut();
    Value input = adaptor.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    if (isa<IntegerType>(inputType.getElementType())) {
      auto newInputType =
          RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
      input = rewriter.create<mlir::tosa::CastOp>(loc, newInputType, input);
      grad_out =
          rewriter.create<mlir::tosa::CastOp>(loc, newInputType, grad_out);
      inputType = newInputType;
    }
    Value cst_sqrt2pi = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {0.797884583f}));
    Value cst_004 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {4.471500e-02f}));
    Value cst_05 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {0.5f}));
    Value cst_1 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {1.0f}));
    Value cst_013 = rewriter.create<mlir::nova::ConstantOp>(
        loc, inputType, DenseElementsAttr::get(inputType, {0.134145f}));

   //Finding g(x) = 0.7978845(x+0.044715xx*x)
    //finding g'(x)=0.7978845(1+0.134145xx)

    Value input_square=rewriter.create<mlir::nova::MulOp>(loc, inputType, input, input);
    auto op_xsq_coeff =rewriter.create<mlir::nova::MulOp>(loc, inputType, input_square, cst_013);
    auto op0 = rewriter.create<mlir::nova::MulOp>(loc, inputType, input, input_square);
    auto op1 = rewriter.create<mlir::nova::MulOp>(loc, inputType, op0, cst_004);
    auto op2 = rewriter.create<mlir::nova::AddOp>(loc, inputType, input, op1);
    auto op3 =rewriter.create<mlir::nova::MulOp>(loc, inputType, op2, cst_sqrt2pi);
    auto poly_grad =rewriter.create<mlir::nova::AddOp>(loc, inputType, cst_1, op_xsq_coeff);    
    auto dg =rewriter.create<mlir::nova::MulOp>(loc, inputType, poly_grad, cst_sqrt2pi);
   //fiding first term of grad_input
   // =>0.5*(1+tanh(g(x)))
    //finding second term of grad_input
    // => 0.5 x(1-tanh^2(g(x)) ) *g'(x)
    //finding  1- tanh^2
    auto op4 = rewriter.create<mlir::nova::TanhOp>(loc, inputType, op3);
    auto op4_sq = rewriter.create<mlir::nova::MulOp>(loc, inputType, op4, op4);
    auto one_minus_tanh_sq =rewriter .create<mlir::nova::SubOp>(loc, inputType, cst_1, op4_sq);
    auto t2_a = rewriter.create<mlir::nova::MulOp>(loc, inputType, dg,one_minus_tanh_sq);
    auto term2 =rewriter.create<mlir::nova::MulOp>(loc, inputType, input, t2_a);
    auto secondterm =rewriter.create<mlir::nova::MulOp>(loc, inputType, term2, cst_05);
    auto op5 = rewriter.create<mlir::nova::AddOp>(loc, inputType, op4, cst_1);
    auto term1 =rewriter.create<mlir::nova::MulOp>(loc, inputType, op5, cst_05);

    auto op6 =rewriter.create<mlir::nova::MulOp>(loc, inputType, input, cst_05);
    //adding two terms
    auto d_gelu =rewriter.create<mlir::nova::AddOp>(loc, inputType, term1, secondterm);

 //final cahin result multiply
    auto grad_input =
        rewriter.create<mlir::nova::MulOp>(loc, inputType, grad_out, d_gelu);
    rewriter.replaceOp(op, {grad_input.getResult()});
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

struct NovaSoftmaxLoweringPattern : public OpConversionPattern<mlir::nova::SoftmaxOp> { 
  using OpConversionPattern<mlir::nova::SoftmaxOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::nova::SoftmaxOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // e^(x - max) / reduce<sum>(e^x - max)
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto restype = cast<RankedTensorType>(op.getType());
    auto inputType = cast<RankedTensorType>(input.getType());
    auto size = inputType.getShape().size();

    int32_t dimension = op.getDimension().has_value() ? op.getDimension().value() : -1;
    if (dimension < 0)
      dimension += size;
    SmallVector<int64_t> dim;
    dim.push_back(dimension);

    auto shape = NovaOpTosaOp::shapeFind(inputType, dimension);
    auto tempresult = RankedTensorType::get(shape, restype.getElementType());
    // creating cast - only if element types differ
    if (inputType.getElementType() != restype.getElementType()) {
      input = rewriter.create<mlir::tosa::CastOp>(loc, restype, input);
    }

    // 1. find maximum
    SmallVector<int64_t> axes = {dimension};
    Value op1 = rewriter.create<mlir::nova::ReduceOp>(loc, ReductionKind::MAX, input, tempresult, true, axes);

    // 2. xi - max
    Value op2 = rewriter.create<mlir::nova::SubOp>(loc, restype, input, op1);
    
    // 3. exp(xi - max)
    Value op3 = rewriter.create<mlir::nova::ExpOp>(loc, restype, op2);
    
    // 4. sum(exp(xi - max))
    Value op4 = rewriter.create<mlir::nova::ReduceOp>(loc, ReductionKind::SUM, op3, tempresult, true, axes);

    // 5. 1 / sum
    // create TOSA div: reciprocal(op4) * op3
    // TOSA's native broadcasting will map the 8x1 op4 onto the 8x16 op3.
    // Guard: clamp sum_exp to at least eps to prevent Inf from reciprocal(~0).
    // Without this, when logits have extreme variance, exp() underflows for most
    // entries, making sum_exp ≈ 0 → reciprocal → Inf → NaN in backward pass.
    auto epsType = cast<RankedTensorType>(op4.getType());
    auto epsAttr = DenseElementsAttr::get(epsType, rewriter.getFloatAttr(epsType.getElementType(), 1.0e-7));
    Value epsConst = rewriter.create<mlir::nova::ConstantOp>(loc, epsType, epsAttr);
    Value safeSumExp = rewriter.create<mlir::nova::MaxOp>(loc, epsType, op4, epsConst);
    Value recip = rewriter.create<mlir::nova::ReciprocalOp>(loc, safeSumExp.getType(), safeSumExp);
    
    // 6. mul(reciprocal * each_value)
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

struct NovaLinearOpLowering : public OpConversionPattern<nova::LinearOp> {
  using OpConversionPattern<nova::LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    if (operands.size() != 3) {
      return rewriter.notifyMatchFailure(op, "expected exactly 3 operands");
    }

    Value input = operands[0];
    Value weight = operands[1];
    Value bias = operands[2];

    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto weightType = llvm::dyn_cast<RankedTensorType>(weight.getType());
    auto biasType = llvm::dyn_cast<RankedTensorType>(bias.getType());

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !weightType || !biasType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }

    auto loc = op.getLoc();
    auto elementType = resultType.getElementType();
    int64_t inputRank = inputType.getRank();

    //create nova::matmul and nova::add 
    auto matmulOperation = rewriter.create<mlir::nova::MatmulOp>(loc, input, weight).getResult();
    auto addOperation = rewriter.create<mlir::nova::AddOp>(loc, matmulOperation, bias);
    rewriter.replaceOp(op, addOperation);
    return success();
  }
};


// layer norm lowering with nova operations

struct NovaLayerNormPattern : public OpConversionPattern<nova::LayerNormOp> {
  using OpConversionPattern<nova::LayerNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    //  y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    //
    // Fused lowering: 3 linalg.generics instead of 9+ Nova ops.
    //   Generic 1: mean = sum(x, axis=-1) / D        [reduction]
    //   Generic 2: var  = sum((x-mean)^2, axis=-1)/D  [reduction, reads mean]
    //   Generic 3: y = (x-mean)*rsqrt(var+eps)*gamma+beta  [all-parallel]
    //
    // Generic 3 is purely elementwise and will fuse as a producer into
    // the downstream matmul via the tiling pipeline.
    Location loc = op.getLoc();
    Value x = adaptor.getInput();
    Value gamma = adaptor.getGamma();
    Value beta = adaptor.getBeta();
    auto xType = cast<RankedTensorType>(x.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType = xType.getElementType();
    int64_t rank = xType.getRank();
    int64_t D = xType.getShape().back();
    float eps = 1e-5f;
    float invD = 1.0f / static_cast<float>(D);
    MLIRContext *ctx = rewriter.getContext();

    // Shapes: input [B,T,D], stats [B,T] (squeezed — no keepdims)
    SmallVector<int64_t> statsShape(xType.getShape().begin(),
                                    xType.getShape().end() - 1);
    auto statsType = RankedTensorType::get(statsShape, elemType);

    // Iterator types: parallel(B), parallel(T), reduction(D)
    SmallVector<utils::IteratorType> reductionIterTypes;
    for (int64_t i = 0; i < rank - 1; ++i)
      reductionIterTypes.push_back(utils::IteratorType::parallel);
    reductionIterTypes.push_back(utils::IteratorType::reduction);

    // Indexing maps for reduction: input (b,t,d), output (b,t)
    SmallVector<AffineExpr> inputExprs, statsExprs;
    for (int64_t i = 0; i < rank; ++i)
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    for (int64_t i = 0; i < rank - 1; ++i)
      statsExprs.push_back(rewriter.getAffineDimExpr(i));
    auto inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
    auto statsMap = AffineMap::get(rank, 0, statsExprs, ctx);

    // Identity value for sum reduction
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));

    // ---- Generic 1: mean = sum(x) / D ----
    Value meanEmpty = rewriter.create<tensor::EmptyOp>(
        loc, statsShape, elemType);
    Value meanInit = rewriter.create<linalg::FillOp>(
        loc, zero, meanEmpty).result();

    auto meanGeneric = rewriter.create<linalg::GenericOp>(
        loc, statsType, /*inputs=*/x, /*outputs=*/meanInit,
        SmallVector<AffineMap>{inputMap, statsMap}, reductionIterTypes,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(nl, args[0], args[1]);
          b.create<linalg::YieldOp>(nl, sum);
        });
    // meanSum = sum(x) per row. Keep as sum — fold invD into normalize.
    Value meanSum = meanGeneric.getResult(0);

    Value invDVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, invD));

    // ---- Generic 2: varSum = sum((x - mean)^2) per row ----
    // Computes mean inline as meanSum * invD.
    Value varEmpty = rewriter.create<tensor::EmptyOp>(
        loc, statsShape, elemType);
    Value varInit = rewriter.create<linalg::FillOp>(
        loc, zero, varEmpty).result();

    auto varGeneric = rewriter.create<linalg::GenericOp>(
        loc, statsType,
        /*inputs=*/ValueRange{x, meanSum},
        /*outputs=*/varInit,
        SmallVector<AffineMap>{inputMap, statsMap, statsMap},
        reductionIterTypes,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // args[0] = x[b,t,d], args[1] = meanSum[b,t], args[2] = acc
          Value mean = b.create<arith::MulFOp>(nl, args[1], invDVal);
          Value diff = b.create<arith::SubFOp>(nl, args[0], mean);
          Value sq = b.create<arith::MulFOp>(nl, diff, diff);
          Value s = b.create<arith::AddFOp>(nl, sq, args[2]);
          b.create<linalg::YieldOp>(nl, s);
        });
    // varSum = sum((x-mean)^2). Fold invD into normalize generic.
    Value varSum = varGeneric.getResult(0);

    // ---- Generic 3: y = (x - mean) * rsqrt(var + eps) * gamma + beta ----
    // All-parallel: will fuse as producer into downstream matmul.
    SmallVector<utils::IteratorType> allParallelIters(rank,
        utils::IteratorType::parallel);

    // Indexing maps: x[b,t,d], mean[b,t], var[b,t], gamma[d], beta[d], out[b,t,d]
    SmallVector<AffineExpr> gammaExprs = {rewriter.getAffineDimExpr(rank - 1)};
    auto gammaMap = AffineMap::get(rank, 0, gammaExprs, ctx);

    Value normEmpty = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elemType);

    Value epsVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, eps));

    auto normOp = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        /*inputs=*/ValueRange{x, meanSum, varSum, gamma, beta},
        /*outputs=*/normEmpty,
        SmallVector<AffineMap>{inputMap, statsMap, statsMap,
                               gammaMap, gammaMap, inputMap},
        allParallelIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // args: x, meanSum, varSum, gamma, beta, out_init
          // Compute mean and var inline from their sums.
          Value mean = b.create<arith::MulFOp>(nl, args[1], invDVal);
          Value var = b.create<arith::MulFOp>(nl, args[2], invDVal);
          Value diff = b.create<arith::SubFOp>(nl, args[0], mean);
          Value varEps = b.create<arith::AddFOp>(nl, var, epsVal);
          Value rsqrt = b.create<math::RsqrtOp>(nl, varEps);
          Value norm = b.create<arith::MulFOp>(nl, diff, rsqrt);
          Value scaled = b.create<arith::MulFOp>(nl, norm, args[3]);
          Value result = b.create<arith::AddFOp>(nl, scaled, args[4]);
          b.create<linalg::YieldOp>(nl, result);
        });

    rewriter.replaceOp(op, normOp.getResult(0));
    return success();
  }
};



struct NovaLayerNormBackwardPattern
   : public OpConversionPattern<nova::LayerNormBackwardOp> {
 using OpConversionPattern<nova::LayerNormBackwardOp>::OpConversionPattern;

 LogicalResult
 matchAndRewrite(nova::LayerNormBackwardOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) const override {
   // LN backward: given grad_y, x, gamma, compute dx, dgamma, dbeta.
   //
   // Fused lowering: 5 linalg.generics instead of 15+ Nova ops.
   //   1. mean_reduce:     sum(x)/D → mean[B,T]
   //   2. var_reduce:      sum((x-mean)²)/D → var[B,T]
   //   3. dx_stats_reduce: sum(gy*gamma) and sum(gy*gamma*x_hat) per row → [B,T]
   //   4. dx_elementwise:  dx = rstd*(gy*gamma - mean1/D - x_hat*mean2/D)
   //   5. dgamma_dbeta:    sum(gy*x_hat) and sum(gy) over batch → [D]
   //
   // No intermediate [B,T,D] workspace tensors — all computed inline.
   Location loc = op.getLoc();
   Value gy = adaptor.getGradY();
   Value x = adaptor.getX();
   Value gamma = adaptor.getGamma();

   auto xType = cast<RankedTensorType>(x.getType());
   auto gammaType = cast<RankedTensorType>(gamma.getType());
   auto elemType = xType.getElementType();
   int64_t rank = xType.getRank();
   int64_t D = xType.getShape().back();
   float eps = 1e-5f;
   float invD = 1.0f / static_cast<float>(D);
   MLIRContext *ctx = rewriter.getContext();

   // Common shapes and maps
   SmallVector<int64_t> statsShape(xType.getShape().begin(),
                                   xType.getShape().end() - 1);
   auto statsType = RankedTensorType::get(statsShape, elemType);

   // Indexing maps
   SmallVector<AffineExpr> inputExprs, statsExprs, gammaExprs;
   for (int64_t i = 0; i < rank; ++i)
     inputExprs.push_back(rewriter.getAffineDimExpr(i));
   for (int64_t i = 0; i < rank - 1; ++i)
     statsExprs.push_back(rewriter.getAffineDimExpr(i));
   gammaExprs.push_back(rewriter.getAffineDimExpr(rank - 1));

   auto inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
   auto statsMap = AffineMap::get(rank, 0, statsExprs, ctx);
   auto gammaMap = AffineMap::get(rank, 0, gammaExprs, ctx);

   // Iterator types
   SmallVector<utils::IteratorType> rowReductionIters;
   for (int64_t i = 0; i < rank - 1; ++i)
     rowReductionIters.push_back(utils::IteratorType::parallel);
   rowReductionIters.push_back(utils::IteratorType::reduction);

   SmallVector<utils::IteratorType> allParallelIters(rank,
       utils::IteratorType::parallel);

   // Batch reduction: reduce(B), reduce(T), parallel(D)
   SmallVector<utils::IteratorType> batchReductionIters;
   for (int64_t i = 0; i < rank - 1; ++i)
     batchReductionIters.push_back(utils::IteratorType::reduction);
   batchReductionIters.push_back(utils::IteratorType::parallel);

   Value zero = rewriter.create<arith::ConstantOp>(
       loc, rewriter.getZeroAttr(elemType));
   Value invDVal = rewriter.create<arith::ConstantOp>(
       loc, rewriter.getFloatAttr(elemType, invD));
   Value epsVal = rewriter.create<arith::ConstantOp>(
       loc, rewriter.getFloatAttr(elemType, eps));

   // Stats-level identity map
   SmallVector<AffineExpr> statsIdExprs;
   for (int64_t i = 0; i < rank - 1; ++i)
     statsIdExprs.push_back(rewriter.getAffineDimExpr(i));
   auto statsIdMap = AffineMap::get(rank - 1, 0, statsIdExprs, ctx);
   SmallVector<utils::IteratorType> parallelStatsIters(rank - 1,
       utils::IteratorType::parallel);

   // ---- Generic 1: meanSum = sum(x) per row → [B,T] ----
   // Keep raw sum — fold invD into downstream generics.
   Value meanEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
   Value meanInit = rewriter.create<linalg::FillOp>(loc, zero, meanEmpty).result();
   auto meanSumOp = rewriter.create<linalg::GenericOp>(
       loc, statsType, /*inputs=*/x, /*outputs=*/meanInit,
       SmallVector<AffineMap>{inputMap, statsMap}, rowReductionIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         Value sum = b.create<arith::AddFOp>(nl, args[0], args[1]);
         b.create<linalg::YieldOp>(nl, sum);
       });
   Value meanSum = meanSumOp.getResult(0);

   // ---- Generic 2: varSum = sum((x - meanSum/D)²) per row → [B,T] ----
   Value varEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
   Value varInit = rewriter.create<linalg::FillOp>(loc, zero, varEmpty).result();
   auto varSumOp = rewriter.create<linalg::GenericOp>(
       loc, statsType, ValueRange{x, meanSum}, varInit,
       SmallVector<AffineMap>{inputMap, statsMap, statsMap}, rowReductionIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         Value mean = b.create<arith::MulFOp>(nl, args[1], invDVal);
         Value diff = b.create<arith::SubFOp>(nl, args[0], mean);
         Value sq = b.create<arith::MulFOp>(nl, diff, diff);
         Value s = b.create<arith::AddFOp>(nl, sq, args[2]);
         b.create<linalg::YieldOp>(nl, s);
       });
   Value varSum = varSumOp.getResult(0);

   // ---- dx_stats: two separate row reductions ----
   // Dual-output linalg.generic reductions don't tile correctly.
   // Use single-output generics: compute gy*gamma and gy*gamma*x_hat
   // as elementwise ops, then reduce each with nova.reduce.

   // Compute gy_gamma [B,T,D] and gy_gamma_xhat [B,T,D] in one pass
   Value gyGammaEmpty = rewriter.create<tensor::EmptyOp>(
       loc, xType.getShape(), elemType);
   Value gyGammaXhatEmpty = rewriter.create<tensor::EmptyOp>(
       loc, xType.getShape(), elemType);
   auto dxPreOp = rewriter.create<linalg::GenericOp>(
       loc, TypeRange{xType, xType},
       /*inputs=*/ValueRange{gy, x, meanSum, varSum, gamma},
       /*outputs=*/ValueRange{gyGammaEmpty, gyGammaXhatEmpty},
       SmallVector<AffineMap>{inputMap, inputMap, statsMap, statsMap,
                              gammaMap, inputMap, inputMap},
       allParallelIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         // args: gy, x, meanSum, varSum, gamma, out1_init, out2_init
         Value gyGamma = b.create<arith::MulFOp>(nl, args[0], args[4]);
         Value mean = b.create<arith::MulFOp>(nl, args[2], invDVal);
         Value var = b.create<arith::MulFOp>(nl, args[3], invDVal);
         Value diff = b.create<arith::SubFOp>(nl, args[1], mean);
         Value varEps = b.create<arith::AddFOp>(nl, var, epsVal);
         Value rstd = b.create<math::RsqrtOp>(nl, varEps);
         Value xHat = b.create<arith::MulFOp>(nl, diff, rstd);
         Value gyGammaXhat = b.create<arith::MulFOp>(nl, gyGamma, xHat);
         b.create<linalg::YieldOp>(nl, ValueRange{gyGamma, gyGammaXhat});
       });
   Value gyGammaVal = dxPreOp.getResult(0);
   Value gyGammaXhatVal = dxPreOp.getResult(1);

   // sum(gy*gamma, axis=-1) per row → [B,T]
   Value sum1Empty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
   Value sum1Init = rewriter.create<linalg::FillOp>(loc, zero, sum1Empty).result();
   auto sum1Op = rewriter.create<linalg::GenericOp>(
       loc, statsType, /*inputs=*/gyGammaVal, /*outputs=*/sum1Init,
       SmallVector<AffineMap>{inputMap, statsMap}, rowReductionIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         Value s = b.create<arith::AddFOp>(nl, args[0], args[1]);
         b.create<linalg::YieldOp>(nl, s);
       });
   Value sumGyGamma = sum1Op.getResult(0);

   // sum(gy*gamma*x_hat, axis=-1) per row → [B,T]
   Value sum2Empty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
   Value sum2Init = rewriter.create<linalg::FillOp>(loc, zero, sum2Empty).result();
   auto sum2Op = rewriter.create<linalg::GenericOp>(
       loc, statsType, /*inputs=*/gyGammaXhatVal, /*outputs=*/sum2Init,
       SmallVector<AffineMap>{inputMap, statsMap}, rowReductionIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         Value s = b.create<arith::AddFOp>(nl, args[0], args[1]);
         b.create<linalg::YieldOp>(nl, s);
       });
   Value sumGyGammaXhat = sum2Op.getResult(0);

   // ---- Generic 4: dx — all-parallel elementwise ----
   // dx = rstd * (gy*gamma - sumGyGamma/D - x_hat * sumGyGammaXhat/D)
   Value dxEmpty = rewriter.create<tensor::EmptyOp>(
       loc, xType.getShape(), elemType);
   auto dxOp = rewriter.create<linalg::GenericOp>(
       loc, xType,
       /*inputs=*/ValueRange{gy, x, meanSum, varSum, gamma,
                             sumGyGamma, sumGyGammaXhat},
       /*outputs=*/dxEmpty,
       SmallVector<AffineMap>{inputMap, inputMap, statsMap, statsMap,
                              gammaMap, statsMap, statsMap, inputMap},
       allParallelIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         // args: gy, x, meanSum, varSum, gamma, s1, s2, out_init
         Value gyGamma = b.create<arith::MulFOp>(nl, args[0], args[4]);
         Value mean = b.create<arith::MulFOp>(nl, args[2], invDVal);
         Value var = b.create<arith::MulFOp>(nl, args[3], invDVal);
         Value diff = b.create<arith::SubFOp>(nl, args[1], mean);
         Value varEps = b.create<arith::AddFOp>(nl, var, epsVal);
         Value rstd = b.create<math::RsqrtOp>(nl, varEps);
         Value xHat = b.create<arith::MulFOp>(nl, diff, rstd);
         Value meanS1 = b.create<arith::MulFOp>(nl, args[5], invDVal);
         Value meanS2 = b.create<arith::MulFOp>(nl, args[6], invDVal);
         Value t1 = b.create<arith::SubFOp>(nl, gyGamma, meanS1);
         Value t2 = b.create<arith::MulFOp>(nl, xHat, meanS2);
         Value t3 = b.create<arith::SubFOp>(nl, t1, t2);
         Value dx = b.create<arith::MulFOp>(nl, t3, rstd);
         b.create<linalg::YieldOp>(nl, dx);
       });
   Value dx = dxOp.getResult(0);

   // ---- dgamma and dbeta via Nova reduce ops ----
   // The batch reduction [reduce(B), reduce(T), parallel(D)] doesn't tile
   // correctly as a linalg.generic — the tiling framework treats outer
   // reduction dims as workgroup distribution, losing the accumulation.
   // Use Nova reduce ops which the existing ReduceOpConverter handles
   // correctly with proper indexing maps and initialization.
   //
   // First compute gy * x_hat as an all-parallel generic, then reduce.
   Value gyXhatEmpty = rewriter.create<tensor::EmptyOp>(
       loc, xType.getShape(), elemType);
   auto gyXhatOp = rewriter.create<linalg::GenericOp>(
       loc, xType,
       /*inputs=*/ValueRange{gy, x, meanSum, varSum},
       /*outputs=*/gyXhatEmpty,
       SmallVector<AffineMap>{inputMap, inputMap, statsMap, statsMap, inputMap},
       allParallelIters,
       [&](OpBuilder &b, Location nl, ValueRange args) {
         // args: gy, x, meanSum, varSum, out_init
         Value mean = b.create<arith::MulFOp>(nl, args[2], invDVal);
         Value var = b.create<arith::MulFOp>(nl, args[3], invDVal);
         Value diff = b.create<arith::SubFOp>(nl, args[1], mean);
         Value varEps = b.create<arith::AddFOp>(nl, var, epsVal);
         Value rstd = b.create<math::RsqrtOp>(nl, varEps);
         Value xHat = b.create<arith::MulFOp>(nl, diff, rstd);
         Value gyXhat = b.create<arith::MulFOp>(nl, args[0], xHat);
         b.create<linalg::YieldOp>(nl, gyXhat);
       });
   Value gyXhat = gyXhatOp.getResult(0);

   // dgamma = sum(gy * x_hat, over batch dims [0..rank-2])
   SmallVector<int64_t> batchDims;
   for (int64_t i = 0; i < rank - 1; ++i)
     batchDims.push_back(i);
   Value dgamma = rewriter.create<nova::ReduceOp>(
       loc, nova::ReductionKind::SUM, gyXhat, gammaType,
       /*keepdims=*/false, batchDims).getResult();

   // dbeta = sum(gy, over batch dims)
   Value dbeta = rewriter.create<nova::ReduceOp>(
       loc, nova::ReductionKind::SUM, gy, gammaType,
       /*keepdims=*/false, batchDims).getResult();

   rewriter.replaceOp(op, {dx, dgamma, dbeta});
   return success();
 }
};


struct NovaSceBackwardOpLowering
    : public OpConversionPattern<mlir::nova::SceBackwardOp> {
  using OpConversionPattern<mlir::nova::SceBackwardOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::SceBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value softmaxRes = adaptor.getSoftmax();
    Value targets = adaptor.getTargets();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto logitsType = cast<RankedTensorType>(softmaxRes.getType());
    auto targetsType = cast<RankedTensorType>(targets.getType());
    int64_t rank = logitsType.getRank();
    auto resultElemType = resultType.getElementType();
    auto targetIdxElemType = targetsType.getElementType();

    int64_t dim = op.getDim();
    if (dim < 0)
      dim += rank;

    // 1. Softmax is passed in directly from the forward SceOp result

    // 2. Flattened Probabilities
    int64_t totalel = 1;
    for (auto s : logitsType.getShape()) {
      if (s == ShapedType::kDynamic) return failure();
      totalel *= s;
    }
    auto flatProbType = RankedTensorType::get({totalel}, resultElemType);
    auto probFlatOp = rewriter.create<mlir::nova::ReshapeOp>(loc, flatProbType, softmaxRes);
    Value probFlat = probFlatOp.getResult();

    // 3. Calculate N and flattened offsets
    int64_t B = logitsType.getDimSize(0);
    int64_t C = (rank > 1) ? logitsType.getDimSize(1) : 1;
    if (rank == 3) {
      B = logitsType.getDimSize(0) * logitsType.getDimSize(1);
      C = logitsType.getDimSize(2);
    }

    int64_t N = 1;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != dim) N *= logitsType.getDimSize(i);
    }

    // Force i64 indices throughout
    auto i64Type = rewriter.getI64Type();
    auto offsetsType = RankedTensorType::get({B}, i64Type);

    // Compute row offsets at runtime via linalg.generic: offsets[i] = i * C
    // Avoids embedding a large (B-element) compile-time constant in the IR.
    Value offsetsEmpty = rewriter.create<mlir::tensor::EmptyOp>(loc, llvm::ArrayRef<int64_t>{B}, i64Type);
    Value cConst = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(C));
    auto offsetsMap = mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());
    Value offsetsConst = rewriter.create<mlir::linalg::GenericOp>(
        loc, TypeRange{offsetsType}, ValueRange{}, ValueRange{offsetsEmpty},
        ArrayRef<mlir::AffineMap>{offsetsMap},
        ArrayRef<mlir::utils::IteratorType>{mlir::utils::IteratorType::parallel},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          Value idx = b.create<mlir::linalg::IndexOp>(l, 0);
          Value idxI64 = b.create<mlir::arith::IndexCastOp>(l, i64Type, idx);
          Value offset = b.create<mlir::arith::MulIOp>(l, idxI64, cConst);
          b.create<mlir::linalg::YieldOp>(l, offset);
        }).getResult(0);

    Value targetsI64 = targets;
    if (!targetIdxElemType.isInteger(64)) {
       auto targetsI64Type = RankedTensorType::get(targetsType.getShape(), i64Type);
       targetsI64 = rewriter.create<mlir::tosa::CastOp>(loc, targetsI64Type, targets);
    }

    auto targetFlatType = RankedTensorType::get({B}, i64Type);
    auto targetFlatOp = rewriter.create<mlir::nova::ReshapeOp>(loc, targetFlatType, targetsI64);
    Value targetFlat = targetFlatOp.getResult();

    auto indicesFlatOp = rewriter.create<mlir::nova::AddOp>(loc, targetFlatType, targetFlat, offsetsConst);
    Value indicesFlat = indicesFlatOp.getResult();

    // 4. -1.0 values to add
    auto negOnesType = RankedTensorType::get({B}, resultElemType);
    auto negOnesAttr = DenseElementsAttr::get(negOnesType, rewriter.getFloatAttr(resultElemType, -1.0));
    auto negOnesConstOp = rewriter.create<mlir::nova::ConstantOp>(loc, negOnesType, negOnesAttr);
    Value negOnesConst = negOnesConstOp.getResult();

    // 5. Scatter Add
    auto scatterAddOp = rewriter.create<mlir::nova::ScatterAddOp>(
        loc, flatProbType, probFlat, indicesFlat, negOnesConst, rewriter.getI64IntegerAttr(0));
    Value diffFlat = scatterAddOp.getResult();

    // 6. Reshape back
    auto diffOp = rewriter.create<mlir::nova::ReshapeOp>(loc, logitsType, diffFlat);
    Value diff = diffOp.getResult();


    // 7. Normalization (divide by N)
    auto nInvConstType = RankedTensorType::get({}, resultElemType);
    auto nInvConstAttr = DenseElementsAttr::get(nInvConstType, rewriter.getFloatAttr(resultElemType, 1.0 / N));
    Value nInvConst = rewriter.create<mlir::nova::ConstantOp>(loc, nInvConstType, nInvConstAttr);

    Value finalGrad = rewriter.create<mlir::nova::MulOp>(loc, resultType, diff, nInvConst);

    rewriter.replaceOp(op, finalGrad);
    return success();
  }
};
struct NovaLinearBackwardPattern
    : public OpConversionPattern<mlir::nova::LinearBackwardOp> {
  using OpConversionPattern<mlir::nova::LinearBackwardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::LinearBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value grad_out = adaptor.getGradOut();
    Value x = adaptor.getInput();
    Value w = adaptor.getWeight();

    auto gradOutType = cast<RankedTensorType>(grad_out.getType());
    auto xType = cast<RankedTensorType>(x.getType());
    auto wType = cast<RankedTensorType>(w.getType());

    // 1. grad_input = matmul(grad_out, w^T)
    auto wShape = wType.getShape().vec();
    if (wShape.size() >= 2)
      std::swap(wShape[wShape.size() - 1], wShape[wShape.size() - 2]);
    auto wtType = RankedTensorType::get(wShape, wType.getElementType());
    Value wt = rewriter.create<mlir::nova::TransposeOp>(loc, wtType, w, 
      rewriter.getI32IntegerAttr(-1),
      rewriter.getI32IntegerAttr(-2)).getResult();

    Value grad_input =rewriter.create<mlir::nova::MatmulOp>(loc,RankedTensorType::get(xType.getShape(), xType.getElementType()),
      grad_out, wt).getResult();

    // 2. grad_weight = matmul(x^T, grad_out) then reduced
    auto xShape = xType.getShape().vec();
    if (xShape.size() >= 2)
      std::swap(xShape[xShape.size() - 1], xShape[xShape.size() - 2]);
    auto xtType = RankedTensorType::get(xShape, xType.getElementType());
    Value xt = rewriter.create<mlir::nova::TransposeOp>(
                       loc, xtType, x, rewriter.getI32IntegerAttr(-1),
                       rewriter.getI32IntegerAttr(-2))
                   .getResult();

    // Calculate intermediate batched dw shape
    SmallVector<int64_t> dwBatchedShape;
    for (size_t i = 0; i < xShape.size(); ++i) {
      if (i == xShape.size() - 2) {
        dwBatchedShape.push_back(xShape[xShape.size() - 2]); // C
      } else if (i == xShape.size() - 1) {
        dwBatchedShape.push_back(
            gradOutType.getDimSize(gradOutType.getRank() - 1)); // D
      } else {
        dwBatchedShape.push_back(xShape[i]); // A
      }
    }
    auto dwBatchedType =
        RankedTensorType::get(dwBatchedShape, wType.getElementType());
    Value dw_batched =
        rewriter.create<mlir::nova::MatmulOp>(loc, dwBatchedType, xt, grad_out)
            .getResult();

    Value grad_weight = dw_batched;
    if (dwBatchedType.getRank() > wType.getRank()) {
      SmallVector<int64_t> rdims;
      for (int64_t i = 0; i < dwBatchedType.getRank() - wType.getRank(); ++i) {
        rdims.push_back(i);
      }
      grad_weight =
          rewriter
              .create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::SUM,
                                            dw_batched, wType, false, rdims)
              .getResult();
    }

    // 3. grad_bias = reduce_sum(grad_out, batchDims)
    SmallVector<int64_t> batchDims;
    for (int64_t i = 0; i < gradOutType.getRank() - 1; ++i) {
      batchDims.push_back(i);
    }
    auto gradBiasType = cast<RankedTensorType>(op.getResultTypes()[2]);
    Value grad_bias = rewriter
                          .create<mlir::nova::ReduceOp>(
                              loc, mlir::nova::ReductionKind::SUM, grad_out,
                              gradBiasType, false, batchDims)
                          .getResult();

    rewriter.replaceOp(op, {grad_input, grad_weight, grad_bias});
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
    target.addIllegalOp<nova::SceBackwardOp>();
    target.addIllegalOp<nova::LayerNormOp>();
    target.addIllegalOp<nova::LayerNormBackwardOp>();
    target.addIllegalOp<nova::LinearBackwardOp>();
    target.addIllegalOp<nova::GeluBackwardOp>();
    target.addIllegalOp<nova::MaeOp>();
    target.addIllegalOp<nova::CastOp>();
    target.addIllegalOp<nova::LinearOp>();
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
  patterns
      .add<NovaReluOpLowering, NovaGeluOpLowering, NovaGeluBackwardPattern, NovaLinearOpLowering,
               NovaSoftmaxLoweringPattern, NovaConstantToArithConstPattern,
               NovaSceBackwardOpLowering, NovaLayerNormPattern,
               NovaLayerNormBackwardPattern, NovaLinearBackwardPattern,
               NovaToTosaLoweringTemplate<nova::MaeOp>,
               NovaToTosaLoweringTemplate<nova::MseOp>,
               NovaToTosaLoweringTemplate<nova::CceOp>,
               NovaToTosaLoweringTemplate<nova::BceOp>,
               NovaToTosaLoweringTemplate<nova::SceOp>,
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
