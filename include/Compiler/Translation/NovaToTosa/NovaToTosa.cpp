

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
  static SmallVector<int64_t> shapeFindargmax(Type currType, int64_t axis) {
    SmallVector<int64_t>
        newshape; // paramters=>inputshape(auto) and axis(int32)
    auto rankedType = cast<RankedTensorType>(currType);
    for (int64_t i = 0; i < rankedType.getRank(); ++i) {
      if (i == axis) {
        // newshape.push_back(1); // TOSA keeps reduced dimension as size 1
      } else {
        newshape.push_back(rankedType.getDimSize(i));
      }
    }
    return newshape;
  }

  static int64_t shapeFindforargmax(Type currType) {
    int64_t newshape = 1; // paramters=>inputshape(auto) and axis(int32)
    auto rankedType = cast<RankedTensorType>(currType);
    for (int64_t i = 0; i < rankedType.getRank(); ++i) {
      newshape *= rankedType.getDimSize(i);
    }
    return newshape;
  }
  static Value mappingtosa(nova::MaxOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {

    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

    return builder->create<tosa::MaximumOp>(op.getLoc(), resultType, v, w);
  }

  static Value mappingtosa(nova::MinOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
    return builder->create<tosa::MinimumOp>(op.getLoc(), resultType, v, w);
  }
  static Value mappingtosa(nova::AndOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
    return builder->create<tosa::LogicalAndOp>(op.getLoc(), resultType, v, w);
  }
  static Value mappingtosa(nova::OrOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

    return builder->create<tosa::LogicalOrOp>(op.getLoc(), resultType, v, w);
  }
  // log op
  static Value mappingtosa(nova::LogOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // if complex type use complex.exp

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.exp element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value exp = b.create<complex::LogOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, exp);
          });

      return genericOp.getResult(0);
    }
    // cast operation to result data type
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

    return builder->create<tosa::LogOp>(op.getLoc(), resultType, v);
  }
  // exp op
  static Value mappingtosa(nova::ExpOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // if complex type use complex.exp

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.exp element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value exp = b.create<complex::ExpOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, exp);
          });

      return genericOp.getResult(0);
    }
    // cast operation to result data type
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

    return builder->create<tosa::ExpOp>(op.getLoc(), resultType, v);
  }
  // square op
  static Value mappingtosa(nova::SquareOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto shift = builder->create<tosa::ConstOp>(
        op.getLoc(),
        RankedTensorType::get({1}, builder->getI8Type(),
                              restensor.getEncoding()),
        DenseElementsAttr::get(RankedTensorType::get({1}, builder->getI8Type()),
                               {static_cast<int8_t>(0)}));
    return builder->create<tosa::MulOp>(op.getLoc(), resultType, v, v, shift);
  }
  // sqrt op
  static Value mappingtosa(nova::SqrtOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

    int64_t rank = restensor.getRank();
    SmallVector<int64_t> constShape(rank, 1);
    auto constType = RankedTensorType::get(constShape, builder->getF32Type(),
                                           restensor.getEncoding());
    auto constAttr = DenseElementsAttr::get(constType, {0.5f});
    auto constOp =
        builder->create<tosa::ConstOp>(op.getLoc(), constType, constAttr);

    return builder->create<tosa::PowOp>(op.getLoc(), resultType, v, constOp);
  }
  // abs op
  static Value mappingtosa(nova::AbsOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // if complex type use complex.abs
    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.abs element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value abs = b.create<complex::AbsOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, abs);
          });

      return genericOp.getResult(0);
    }
    return builder->create<tosa::AbsOp>(op.getLoc(), resultType, input[0]);
  }
  static Value mappingtosa(nova::XorOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

    return builder->create<tosa::LogicalXorOp>(op.getLoc(), resultType, v, w);
  }
  static Value mappingtosa(nova::NegOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<IntegerType>(tensorTy.getElementType()) ||
        isa<FloatType>(tensorTy.getElementType())) {
      return builder->create<tosa::NegateOp>(op.getLoc(), resultType, input[0]);
    }
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.neg element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value neg = b.create<complex::NegOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, neg);
          });

      return genericOp.getResult(0);
    }
    return nullptr;
  }
  //=================================
  // TRIGNOMENTARY
  //=================================
  // sin op
  static Value mappingtosa(nova::SinOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // if complex type use complex.exp

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.exp element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value exp = b.create<complex::SinOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, exp);
          });

      return genericOp.getResult(0);
    }
    // cast operation to result data type
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

    return builder->create<tosa::SinOp>(op.getLoc(), resultType, v);
  }
  static Value mappingtosa(nova::CosOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // if complex type use complex.exp

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.exp element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value exp = b.create<complex::CosOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, exp);
          });

      return genericOp.getResult(0);
    }
    // cast operation to result data type
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

    return builder->create<tosa::CosOp>(op.getLoc(), resultType, v);
  }
  // tanh
  static Value mappingtosa(nova::TanhOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // if complex type use complex.exp
    auto tensorTy = llvm::dyn_cast<RankedTensorType>(input[0].getType());
    if (isa<ComplexType>(tensorTy.getElementType())) {
      // Need to use linalg.generic to apply complex.exp element-wise
      auto loc = op.getLoc();
      auto resultTensorType = llvm::cast<RankedTensorType>(resultType);

      Value emptyTensor = builder->create<tensor::EmptyOp>(
          loc, resultTensorType.getShape(), resultTensorType.getElementType(),
          resultTensorType.getEncoding());

      auto identityMap =
          builder->getMultiDimIdentityMap(resultTensorType.getRank());
      SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultTensorType.getRank(), utils::IteratorType::parallel);

      auto genericOp = builder->create<linalg::GenericOp>(
          loc, TypeRange{resultType}, input[0], emptyTensor, indexingMaps,
          iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is complex<f32> (scalar)
            Value exp = b.create<complex::TanhOp>(loc, args[0]);
            b.create<linalg::YieldOp>(loc, exp);
          });

      return genericOp.getResult(0);
    }
    // cast operation to result data type
    auto restensor = dyn_cast<RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

    return builder->create<tosa::TanhOp>(op.getLoc(), resultType, v);
  }
  static Value mappingtosa(nova::ReciprocalOp op, Type resultType,
                           ValueRange input, OpBuilder *builder) {
    return builder->create<tosa::ReciprocalOp>(op.getLoc(), resultType,
                                               input[0]);
  }
  static Value mappingtosa(nova::NotOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restype = dyn_cast<mlir::RankedTensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restype, input[0]);
    return builder->create<tosa::LogicalNotOp>(op.getLoc(), resultType, v);
  }
  static Value mappingtosa(nova::SigmoidOp op, Type resultType,
                           ValueRange input, OpBuilder *builder) {
    return builder->create<tosa::SigmoidOp>(op.getLoc(), resultType, input[0]);
  }

  // MAE lowering pattern
  static Value mappingtosa(nova::MaeOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType = mlir::RankedTensorType::get(
        v_type.getShape(), targetElemType, v_type.getEncoding());
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType = mlir::RankedTensorType::get(
        w_type.getShape(), targetElemType, w_type.getEncoding());
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);
    // loss= reduce_mean(abs(arg0-arg1))
    auto sub = builder->create<tosa::SubOp>(op.getLoc(), newVType, v, w);
    auto abs = builder->create<tosa::AbsOp>(op.getLoc(), newVType, sub);
    nova::ReductionKind rk = nova::ReductionKind::MEAN;
    // only 2d for now.
    int64_t rank = cast<mlir::ShapedType>(abs.getType()).getRank();
    llvm::SmallVector<int64_t, 1> dimensions;
    if (rank > 0) {
      for (int64_t i = 0; i < rank; ++i) {
        dimensions.push_back(i);
      }
    }
    return builder->create<nova::ReduceOp>(op.getLoc(), rk, abs, resultType,
                                           false, dimensions);
  }
  // MSE lowering pattern
  static Value mappingtosa(nova::MseOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // loss= reduce_mean(square(arg0-arg1))
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType = mlir::RankedTensorType::get(
        v_type.getShape(), targetElemType, v_type.getEncoding());
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType = mlir::RankedTensorType::get(
        w_type.getShape(), targetElemType, w_type.getEncoding());
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);
    auto sub = builder->create<tosa::SubOp>(op.getLoc(), newVType, v, w);

    mlir::RankedTensorType constType = mlir::RankedTensorType::get(
        v_type.getShape(), builder->getF32Type(), v_type.getEncoding());
    mlir::DenseElementsAttr constAttr =
        mlir::DenseElementsAttr::get(constType, llvm::ArrayRef<float_t>(2));
    auto constTwo =
        builder->create<tosa::ConstOp>(op.getLoc(), constType, constAttr);
    auto abs =
        builder->create<tosa::PowOp>(op.getLoc(), newVType, sub, constTwo);

    nova::ReductionKind rk = nova::ReductionKind::MEAN;
    int64_t rank = cast<mlir::ShapedType>(abs.getType()).getRank();

    llvm::SmallVector<int64_t, 1> dimensions;
    if (rank > 0) {
      for (int64_t i = 0; i < rank; ++i) {
        dimensions.push_back(i);
      }
    }
    return builder->create<nova::ReduceOp>(op.getLoc(), rk, abs, resultType,
                                           false, dimensions);
  }
  // CCE lowering pattern
  static Value mappingtosa(nova::CceOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // basic casting logic
    auto restensor = dyn_cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType = mlir::RankedTensorType::get(
        v_type.getShape(), targetElemType, v_type.getEncoding());
    auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType, input[0]);
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    auto newWType = mlir::RankedTensorType::get(
        w_type.getShape(), targetElemType, w_type.getEncoding());
    auto w = builder->create<tosa::CastOp>(op.getLoc(), newWType, input[1]);
    // step1:creating 1x10^-7  tensor constant
    auto hostVType = mlir::RankedTensorType::get(
        newVType.getShape(), targetElemType, v_type.getEncoding());
    auto epiAttr =
        DenseElementsAttr::get(hostVType, builder->getF32FloatAttr(0.0000001f));
    Value epi = builder->create<tosa::ConstOp>(op.getLoc(), hostVType, epiAttr);
    // step2: creating one minus epsilon constant
    auto oneminusepiAttr =
        DenseElementsAttr::get(hostVType, builder->getF32FloatAttr(1.0f));
    Value ones =
        builder->create<tosa::ConstOp>(op.getLoc(), hostVType, oneminusepiAttr);
    Value oneminusepi = builder->create<nova::SubOp>(op.getLoc(), ones, epi);
    // step3:creating compare op
    auto inputShape = cast<mlir::RankedTensorType>(v.getType()).getShape();
    // Get the boolean element type (i1)
    auto boolType = builder->getI1Type();
    auto compareResultType =
        mlir::RankedTensorType::get(inputShape, boolType, v_type.getEncoding());
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
    auto constType =
        mlir::RankedTensorType::get({}, targetElemType, v_type.getEncoding());
    auto minus1Attr =
        DenseElementsAttr::get(constType, builder->getF32FloatAttr(-1.0));
    Value minus1 =
        builder->create<tosa::ConstOp>(op.getLoc(), constType, minus1Attr);
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
    auto reducedResultType = mlir::RankedTensorType::get(
        newShape, targetElemType, v_type.getEncoding());

    nova::ReductionKind rk = nova::ReductionKind::SUM;
    auto reduceres = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, mul, reducedResultType, false, dimensions);
    rk = nova::ReductionKind::MEAN;
    auto reducemeanres =
        builder->create<nova::ReduceOp>(op.getLoc(), rk, reduceres, resultType);
    // step 9: mul reduce result and -1
    return builder->create<nova::MulOp>(op.getLoc(), reducemeanres, minus1);
  }

  // BCE lowering pattern
  static Value mappingtosa(nova::BceOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // basic casting logic
    auto restensor = cast<mlir::RankedTensorType>(resultType);
    auto targetElemType = restensor.getElementType();
    auto v_type = cast<mlir::RankedTensorType>(input[0].getType());
    auto newVType = mlir::RankedTensorType::get(
        v_type.getShape(), targetElemType, v_type.getEncoding());
    //  auto v = builder->create<tosa::CastOp>(op.getLoc(), newVType,input[0]);
    auto v = input[0];
    auto w = input[1];
    auto w_type = cast<mlir::RankedTensorType>(input[1].getType());
    [[maybe_unused]] auto newWType = mlir::RankedTensorType::get(
        w_type.getShape(), targetElemType, w_type.getEncoding());
    // auto w = builder->create<tosa::CastOp>(op.getLoc(), newVType,input[1]);
    // step1:creating 1x10^-7  tensor constant
    auto hostVType = mlir::RankedTensorType::get(
        newVType.getShape(), targetElemType, v_type.getEncoding());
    auto epiAttr =
        DenseElementsAttr::get(hostVType, builder->getF32FloatAttr(0.0000001f));
    Value epi = builder->create<tosa::ConstOp>(op.getLoc(), hostVType, epiAttr);
    // step2: creating one minus epsilon constant
    auto oneminusepiAttr =
        DenseElementsAttr::get(hostVType, builder->getF32FloatAttr(1.0f));
    Value ones =
        builder->create<tosa::ConstOp>(op.getLoc(), hostVType, oneminusepiAttr);
    Value oneminusepi = builder->create<nova::SubOp>(op.getLoc(), ones, epi);
    // step3:creating compare op
    auto inputShape = cast<mlir::RankedTensorType>(v.getType()).getShape();
    // Get the boolean element type (i1)
    auto boolType = builder->getI1Type();
    auto compareResultType =
        mlir::RankedTensorType::get(inputShape, boolType, v_type.getEncoding());
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
    // reducing along all axis
    auto rk = nova::ReductionKind::MEAN;
    auto reducemeanres = builder->create<nova::ReduceOp>(
        op.getLoc(), rk, sumterms, resultType, false, dimensions);

    // step9:create -1 constant tensor (scalar)
    auto constType =
        mlir::RankedTensorType::get({}, targetElemType, v_type.getEncoding());
    auto minus1Attr =
        DenseElementsAttr::get(constType, builder->getF32FloatAttr(-1.0));
    Value minus1 =
        builder->create<tosa::ConstOp>(op.getLoc(), constType, minus1Attr);
    // final step: mul reduce result and -1
    return builder->create<nova::MulOp>(op.getLoc(), reducemeanres, minus1);
  }
  // SCE lOWERING  pattern
  static Value mappingtosa(nova::SceOp op, Type resultType, ValueRange input,
                           OpBuilder *builder) {
    // Input[0] = logits (e.g., tensor<4x10xf32>)
    // Input[1] = targets (e.g., tensor<4xi32>)
    Value logits = input[0];
    Value targets = input[1];

    // Ensure targets are i32
    auto targetsType = cast<mlir::RankedTensorType>(targets.getType());
    if (isa<mlir::FloatType>(targetsType.getElementType())) {
      auto newTargetsType = mlir::RankedTensorType::get(
          targetsType.getShape(), builder->getI32Type(),
          targetsType.getEncoding());
      targets =
          builder->create<tosa::CastOp>(op.getLoc(), newTargetsType, targets);
      targetsType = newTargetsType;
    }

    auto logitsType = cast<mlir::RankedTensorType>(logits.getType());

    // Ensure logits are f32
    auto logitsElemType = logitsType.getElementType();
    if (!isa<mlir::FloatType>(logitsElemType) ||
        cast<mlir::FloatType>(logitsElemType).getWidth() != 32) {
      auto newLogitsType = mlir::RankedTensorType::get(
          logitsType.getShape(), builder->getF32Type(),
          logitsType.getEncoding());
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
    auto maxValType = mlir::RankedTensorType::get(
        maxShape, builder->getF32Type(), logitsType.getEncoding());

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
    Value selectedLogProbs =
        builder->create<nova::GatherOp>(op.getLoc(), logSmZ, targets, 1)
            .getResult();

    // Step 8: loss = reduce_mean(selected_log_probs * -1.0)
    // Create -1.0 constant
    auto constType = mlir::RankedTensorType::get({}, builder->getF32Type(),
                                                 logitsType.getEncoding());
    auto minus1Attr =
        DenseElementsAttr::get(constType, builder->getF32FloatAttr(-1.0));
    Value minus1 =
        builder->create<tosa::ConstOp>(op.getLoc(), constType, minus1Attr);

    // Multiply selected_log_probs by -1
    Value negLogProbs =
        builder->create<nova::MulOp>(op.getLoc(), selectedLogProbs, minus1);

    // Reduce mean over batch dimension
    auto rk = nova::ReductionKind::MEAN;
    llvm::SmallVector<int64_t, 1> dimensions = {0};

    Value loss = builder->create<nova::ReduceOp>(op.getLoc(), rk, negLogProbs,
                                                 resultType, false, dimensions);

    return loss;
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
      auto newInputType = RankedTensorType::get(
          inputType.getShape(), rewriter.getF32Type(), inputType.getEncoding());
      input = rewriter.create<mlir::tosa::CastOp>(loc, newInputType, input);
      inputType = newInputType;
    }
    // Helper to get type without device encoding for constants
    auto stripEncoding = [&](RankedTensorType type) -> RankedTensorType {
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   type.getEncoding());
    };
    auto hostInputType = stripEncoding(inputType);

    // op0 = pow(x, 3)
    Value cst_3 = rewriter.create<mlir::nova::ConstantOp>(
        loc, hostInputType, DenseElementsAttr::get(hostInputType, {3.0f}));
    auto op0 = rewriter.create<mlir::tosa::PowOp>(loc, inputType, input, cst_3);
    // op1 = mul(op0, 0.044715)
    Value cst_004 = rewriter.create<mlir::nova::ConstantOp>(
        loc, hostInputType,
        DenseElementsAttr::get(hostInputType, {4.471500e-02f}));
    auto op1 = rewriter.create<mlir::nova::MulOp>(loc, inputType, op0, cst_004);
    // op2 = add(x, op1)
    auto op2 = rewriter.create<mlir::tosa::AddOp>(loc, inputType, input, op1);
    // op3 = mul(op2, sqrt(2/pi))
    Value cst_sqrt2pi = rewriter.create<mlir::nova::ConstantOp>(
        loc, hostInputType,
        DenseElementsAttr::get(hostInputType, {0.797884583f}));
    auto op3 =
        rewriter.create<mlir::nova::MulOp>(loc, inputType, op2, cst_sqrt2pi);
    // op4 = tanh(op3)
    auto op4 = rewriter.create<mlir::tosa::TanhOp>(loc, inputType, op3);
    // op5 = add(op4 ,1)
    Value cst_1 = rewriter.create<mlir::nova::ConstantOp>(
        loc, hostInputType, DenseElementsAttr::get(hostInputType, {1.0f}));
    auto op5 = rewriter.create<mlir::tosa::AddOp>(loc, inputType, op4, cst_1);
    // op6 = mul(x, 0.5)
    Value cst_05 = rewriter.create<mlir::nova::ConstantOp>(
        loc, hostInputType, DenseElementsAttr::get(hostInputType, {0.5f}));
    auto op6 =
        rewriter.create<mlir::nova::MulOp>(loc, inputType, input, cst_05);

    auto op7 = rewriter.create<mlir::nova::MulOp>(loc, inputType, op6, op5);

    rewriter.replaceOp(op, {op7.getResult()});

    return success();
  }
};
// Pattern to convert nova.relu to tosa.relu
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
    auto hostInputType = RankedTensorType::get(
        inputType.getShape(), elementType, inputType.getEncoding());
    DenseElementsAttr zeroTensor =
        DenseElementsAttr::get(hostInputType, zeroAttr);
    Value zero =
        rewriter.create<mlir::nova::ConstantOp>(loc, hostInputType, zeroTensor);
    Value result =
        rewriter.create<mlir::tosa::MaximumOp>(loc, inputType, input, zero);

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
    auto tempresult = RankedTensorType::get(shape, restype.getElementType(),
                                            restype.getEncoding());
    // creating cast - only if element types differ
    if (inputType.getElementType() != restype.getElementType()) {
      input = rewriter.create<mlir::tosa::CastOp>(loc, restype, input);
    }

    auto axisAttr = rewriter.getI32IntegerAttr(dimension);
    Value op1 = rewriter.create<mlir::tosa::ReduceMaxOp>(loc, tempresult, input,
                                                         axisAttr);

    // step2
    // create a TOSA sub op with input and op1
    Value op2 = rewriter.create<mlir::tosa::SubOp>(loc, restype, input, op1);
    // step3
    // create  a TOSA exp op
    Value op3 = rewriter.create<mlir::tosa::ExpOp>(loc, restype, op2);
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
        rewriter.create<mlir::tosa::ReciprocalOp>(loc, restype, op4_broadcast);
    auto shift = rewriter.create<mlir::arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(RankedTensorType::get({1}, rewriter.getI8Type()),
                               rewriter.getI8IntegerAttr(0)));
    Value op5 =
        rewriter.create<mlir::tosa::MulOp>(loc, restype, op3, recip, shift);
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

struct NovaConstantToTosaConstPattern
    : public OpConversionPattern<nova::ConstantOp> {
  using OpConversionPattern<nova::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ElementsAttr valueAttr = op.getValue();
    DenseElementsAttr value = dyn_cast<DenseElementsAttr>(valueAttr);

    auto outputType = cast<RankedTensorType>(op.getOutput().getType());
    auto hostOutputType = RankedTensorType::get(outputType.getShape(),
                                                outputType.getElementType(),
                                                outputType.getEncoding());
    auto hostValue = value.reshape(hostOutputType);
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, hostOutputType, hostValue);
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
  }

  StringRef getArgument() const final { return "convert-nova-to-tosa"; }

  StringRef getDescription() const final {
    return "Lower Nova dialect operations to Tosa dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    target.addIllegalOp<nova::ConstantOp>();
    target.addIllegalOp<nova::ReluOp>();
    target.addIllegalOp<nova::ExpOp>();
    target.addIllegalOp<nova::LogOp>();
    target.addIllegalOp<nova::AbsOp>();
    target.addIllegalOp<nova::MaxOp>();
    target.addIllegalOp<nova::MinOp>();
    target.addIllegalOp<nova::SubOp>();
    target.addIllegalOp<nova::PowOp>();
    target.addIllegalOp<nova::SqrtOp>();
    target.addIllegalOp<nova::SquareOp>();
    target.addIllegalOp<nova::AndOp>();
    target.addIllegalOp<nova::OrOp>();
    target.addIllegalOp<nova::XorOp>();
    target.addIllegalOp<nova::NegOp>();
    target.addIllegalOp<nova::NotOp>();
    target.addIllegalOp<nova::SinOp>();
    target.addIllegalOp<nova::CosOp>();
    target.addIllegalOp<nova::TanhOp>();
    target.addIllegalOp<nova::ReciprocalOp>();
    target.addIllegalOp<nova::MseOp>();
    target.addIllegalOp<nova::CceOp>();
    target.addIllegalOp<nova::SigmoidOp>();
    target.addIllegalOp<nova::GeluOp>();
    target.addIllegalOp<nova::SoftmaxOp>();
    target.addIllegalOp<nova::BceOp>();
    target.addIllegalOp<nova::SceOp>();
    target.addIllegalOp<nova::AddOp>();
    target.addIllegalOp<nova::MaeOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    mlir::RewritePatternSet patterns(&getContext());
    mlir::nova::populateNovaToTosaConversionPatterns(patterns);
    mlir::nova::populateNovaToTosaTemplatePatterns(patterns);
    //  populateNovaToArithConversionPatterns(patterns);
    //   populateNovaToLinalgPatterns(patterns);
    // populateNovaToLinalgPatternsTemplate(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void populateNovaToTosaConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaReluOpLowering, NovaGeluOpLowering,
               NovaSoftmaxLoweringPattern, NovaConstantToTosaConstPattern,
               NovaToTosaLoweringTemplate<nova::MaxOp>, 
               NovaToTosaLoweringTemplate<nova::LogOp>, 
               NovaToTosaLoweringTemplate<nova::AbsOp>, 
               NovaToTosaLoweringTemplate<nova::ExpOp>, 
               NovaToTosaLoweringTemplate<nova::MinOp>, 
               NovaToTosaLoweringTemplate<nova::AndOp>, 
               NovaToTosaLoweringTemplate<nova::SinOp>, 
               NovaToTosaLoweringTemplate<nova::CosOp>, 
               NovaToTosaLoweringTemplate<nova::TanhOp>, 
               NovaToTosaLoweringTemplate<nova::OrOp>, 
               NovaToTosaLoweringTemplate<nova::XorOp>, 
               NovaToTosaLoweringTemplate<nova::NotOp>, 
               NovaToTosaLoweringTemplate<nova::NegOp>, 
               NovaToTosaLoweringTemplate<nova::ReciprocalOp>, 
               NovaToTosaLoweringTemplate<nova::MaeOp>, 
               NovaToTosaLoweringTemplate<nova::MseOp>, 
               NovaToTosaLoweringTemplate<nova::CceOp>, 
               NovaToTosaLoweringTemplate<nova::BceOp>, 
               NovaToTosaLoweringTemplate<nova::SceOp>, 
               NovaToTosaLoweringTemplate<nova::SigmoidOp>, 
               NovaToTosaLoweringTemplate<nova::SquareOp>, 
               NovaToTosaLoweringTemplate<nova::SqrtOp> 
               >(patterns.getContext());
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
