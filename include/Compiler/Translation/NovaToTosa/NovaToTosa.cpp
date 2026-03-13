

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

    SmallVector<int64_t> axes = {dimension};
    Value op1 = rewriter.create<mlir::nova::ReduceOp>(
        loc, ReductionKind::MAX, input, tempresult, /*keepdims=*/true, axes);

    // step2
    // create a TOSA sub op with input and op1
    Value op2 = rewriter.create<mlir::nova::SubOp>(loc, restype, input, op1);
    // step3
    // create  a TOSA exp op
    Value op3 = rewriter.create<mlir::nova::ExpOp>(loc, restype, op2);
    // step4
    // create a TOSA reduce sum
    Value op4 = rewriter.create<mlir::nova::ReduceOp>(
        loc, ReductionKind::SUM, op3, tempresult, /*keepdims=*/true, axes);

    // step 5
    // create TOSA div: reciprocal(op4) * op3
    // TOSA's native broadcasting will map the 8x1 op4 onto the 8x16 op3.
    Value recip =
        rewriter.create<mlir::nova::ReciprocalOp>(loc, op4.getType(), op4);
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
    // getting operands
    Location loc = op.getLoc();
    Value x = adaptor.getInput();
    Value gamma = adaptor.getGamma();
    Value beta = adaptor.getBeta();
    auto xres = cast<RankedTensorType>(op.getType());
    auto xType = cast<RankedTensorType>(x.getType());

    auto elemType = xType.getElementType();
    // setting up
    float eps = 1e-5f;
    int last_dim = xType.getRank()-1;
    // 1.finding row mean
    std::vector<int64_t> meanShape(xType.getShape().begin(),
                                   xType.getShape().end());
    float last_dim_n = meanShape[last_dim];
    meanShape.back() = 1;
    auto meanType = mlir::RankedTensorType::get(meanShape, elemType);
    auto mean_sum = rewriter
                    .create<mlir::nova::ReduceOp>(
                        loc, mlir::nova::ReductionKind::SUM, x, meanType, true,
                        llvm::ArrayRef<int64_t>{-1}, false)
                    .getResult();
    auto scalarType = mlir::RankedTensorType::get({}, xType.getElementType());
    auto n = rewriter.create<mlir::nova::ConstantOp>(loc, scalarType, mlir::DenseElementsAttr::get(scalarType, rewriter.getFloatAttr(xType.getElementType(), (double)(1/last_dim_n))));
   // auto reciprocal = rewriter.create<nova::ReciprocalOp>(loc, n);
    auto mean = rewriter.create<mlir::nova::MulOp>(loc, n, mean_sum).getResult();

    auto x_minus_mean = rewriter.create<mlir::nova::SubOp>(loc, xres, x, mean).getResult();


    // 3. (x - mean)^2
    auto sq_diff =
        rewriter
            .create<mlir::nova::MulOp>(loc, xres, x_minus_mean, x_minus_mean)
            .getResult();
    // 4. Var(x) = Mean((x-mean)^2)
    auto var_sum = rewriter
                   .create<mlir::nova::ReduceOp>(
                       loc, mlir::nova::ReductionKind::SUM, sq_diff, meanType,
                       true, llvm::ArrayRef<int64_t>{-1}, false)
                   .getResult();
    auto var = rewriter.create<mlir::nova::MulOp>(loc, n, var_sum).getResult();

    // 5. sqrt(var + eps)
    auto epsAttr = mlir::DenseElementsAttr::get(
        mlir::cast<mlir::RankedTensorType>(var.getType()), eps);
    auto epsConst =
        rewriter.create<mlir::nova::ConstantOp>(loc, var.getType(), epsAttr)
            .getResult();
    auto var_plus_eps =
        rewriter.create<mlir::nova::AddOp>(loc, var.getType(), var, epsConst)
            .getResult();
    auto std_inv =
        rewriter.create<mlir::nova::RsqrtOp>(loc, var.getType(), var_plus_eps)
            .getResult();
    // 6. Normalize
    auto norm =
        rewriter.create<mlir::nova::MulOp>(loc, xres, x_minus_mean, std_inv)
            .getResult();
    // 7. scale and shift
    // multiply by gamma
    auto res =
        rewriter.create<mlir::nova::MulOp>(loc, xres, norm, gamma).getResult();
    // add beta
    auto result =
        rewriter.create<mlir::nova::AddOp>(loc, xres, res, beta).getResult();
    rewriter.replaceOp(op, result);
    return success();
  }
};



struct NovaLayerNormBackwardPattern
   : public OpConversionPattern<nova::LayerNormBackwardOp> {
 using OpConversionPattern<nova::LayerNormBackwardOp>::OpConversionPattern;

 LogicalResult
 matchAndRewrite(nova::LayerNormBackwardOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) const override {
   Location loc = op.getLoc();
   Value gy = adaptor.getGradY();
   Value x = adaptor.getX();
   Value gamma = adaptor.getGamma();

   auto xType = cast<RankedTensorType>(x.getType());
   auto gammaType = cast<RankedTensorType>(gamma.getType());


   // epsilon
   float eps = 1e-5f;


   // calculate mean
   int64_t last_dim = xType.getRank() - 1;
   llvm::SmallVector<int64_t, 1> dims = {last_dim};
   llvm::SmallVector<int64_t, 4> red_shape(xType.getShape().begin(), xType.getShape().end());
   //create constant of 1/total number of elements 
    
   float ndim=red_shape[last_dim];
   red_shape[last_dim] = 1;
   auto red_type = mlir::RankedTensorType::get(red_shape, xType.getElementType());
   auto scalarType = mlir::RankedTensorType::get({}, xType.getElementType());
   auto dim_const = rewriter.create<mlir::nova::ConstantOp>(loc, scalarType, mlir::DenseElementsAttr::get(scalarType, rewriter.getFloatAttr(xType.getElementType(), (float)(1/ndim))));
   auto sum_x = rewriter.create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::SUM, x, red_type, true,  llvm::ArrayRef<int64_t>{-1}, false);
   auto mean = rewriter.create<mlir::nova::MulOp>(loc,dim_const,sum_x);
   auto meanType = cast<RankedTensorType>(mean.getType());


   // calculate standard deviation
   auto diff = rewriter.create<mlir::nova::SubOp>(loc, x, mean.getResult());
   auto diff2 = rewriter.create<mlir::nova::MulOp>(loc, diff.getResult(), diff.getResult());
   auto sum_sq = rewriter.create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::SUM, diff2.getResult(), red_type, true,  llvm::ArrayRef<int64_t>{-1}, false);
   auto var = rewriter.create<mlir::nova::MulOp>(loc,dim_const,sum_sq);
   auto eps_const = rewriter.create<mlir::nova::ConstantOp>(loc, scalarType, mlir::DenseElementsAttr::get(scalarType, rewriter.getFloatAttr(xType.getElementType(), (double)eps)));
   auto var_eps = rewriter.create<mlir::nova::AddOp>(loc, var.getResult(), eps_const.getResult());
   auto rstd = rewriter.create<mlir::nova::RsqrtOp>(loc, var_eps.getResult());




   // 1. Compute x_hat = (x - mean) * rstd
   Value x_minus_mean =
       rewriter.create<mlir::nova::SubOp>(loc, xType, x, mean).getResult();
   Value x_hat =
       rewriter.create<mlir::nova::MulOp>(loc, xType, x_minus_mean, rstd)
           .getResult();


   // 3. dgamma = sum(gy * x_hat, over batch dims)
   Value gy_xhat =
       rewriter.create<mlir::nova::MulOp>(loc, xType, gy, x_hat).getResult();
   SmallVector<int64_t> batchDims;
   for (int64_t i = 0; i < xType.getRank() - 1; ++i)
     batchDims.push_back(i);
   Value dgamma =
       rewriter
           .create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::SUM,
                                         gy_xhat, gammaType, false, batchDims)
           .getResult();


   // 4. dbeta = sum(gy, over batch dims)
   Value dbeta =
       rewriter
           .create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::SUM,
                                         gy, gammaType, false, batchDims)
           .getResult();


   // 5. dx
   Value gy_gamma =
       rewriter.create<mlir::nova::MulOp>(loc, xType, gy, gamma).getResult();

 //Using reduce sum
   Value sum_gy_gamma =
       rewriter
           .create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::SUM,
                                         gy_gamma, meanType, true,
                                         llvm::ArrayRef<int64_t>{-1})
           .getResult();
  //CReating constant of reciprocal 
  
  Value mean_gy_gamma = rewriter.create<mlir::nova::MulOp>(loc,dim_const,sum_gy_gamma);
   Value gy_gamma_xhat =
       rewriter.create<mlir::nova::MulOp>(loc, xType, gy_gamma, x_hat)
           .getResult();
           
   Value mean_gy_gamma_xhat =
       rewriter
           .create<mlir::nova::ReduceOp>(loc, mlir::nova::ReductionKind::MEAN,
                                         gy_gamma_xhat, meanType, true,
                                         llvm::ArrayRef<int64_t>{-1})
           .getResult();
  // Value mean_gy_gamma_xhat = rewriter.create<mlir::nova::MulOp>(loc,dim_const,sum_gy_gamma_xhat).getResult();


   Value t3 =
       rewriter
           .create<mlir::nova::MulOp>(loc, xType, x_hat, mean_gy_gamma_xhat)
           .getResult();
   Value t4 =
       rewriter.create<mlir::nova::SubOp>(loc, xType, gy_gamma, mean_gy_gamma)
           .getResult();
   Value t5 =
       rewriter.create<mlir::nova::SubOp>(loc, xType, t4, t3).getResult();
   Value dx =
       rewriter.create<mlir::nova::MulOp>(loc, xType, t5, rstd).getResult();


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
    Value logits = adaptor.getLogits();
    Value targets = adaptor.getTargets();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto logitsType = cast<RankedTensorType>(logits.getType());
    auto targetsType = cast<RankedTensorType>(targets.getType());
    int64_t rank = logitsType.getRank();
    auto resultElemType = resultType.getElementType();
    auto targetIdxElemType = targetsType.getElementType();

    int64_t dim = op.getDim();
    if (dim < 0)
      dim += rank;

    // 1. Calculate Softmax
    auto softmaxOp = rewriter.create<mlir::nova::SoftmaxOp>(
        loc, logitsType, logits, rewriter.getI32IntegerAttr(dim));
    Value softmaxRes = softmaxOp.getResult();

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
    std::vector<int64_t> offsets(B);
    for (int64_t i = 0; i < B; ++i) offsets[i] = i * C;
    auto offsetsAttr = DenseIntElementsAttr::get(offsetsType, llvm::ArrayRef<int64_t>(offsets));
    auto offsetsConstOp = rewriter.create<mlir::nova::ConstantOp>(loc, offsetsType, offsetsAttr);
    Value offsetsConst = offsetsConstOp.getResult();

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
