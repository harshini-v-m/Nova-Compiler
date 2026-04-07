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

#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

namespace mlir {
namespace nova {
//===--------------------------------------------------------------------------------------------===//
// Helper Functions
//===--------------------------------------------------------------------------------------------===//
inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}
static Value reduce_to_shape_nova(OpBuilder &builder, Location loc, Value grad, Value target) {
  auto grad_type = llvm::dyn_cast<mlir::RankedTensorType>(grad.getType());
  auto target_type = llvm::dyn_cast<mlir::RankedTensorType>(target.getType());
  if (!grad_type || !target_type)
    return grad;

  auto grad_shape = grad_type.getShape();
  auto target_shape = target_type.getShape();

  if (grad_shape == target_shape)
    return grad;

  int64_t grad_rank = grad_type.getRank();
  int64_t target_rank = target_type.getRank();

  llvm::SmallVector<int64_t, 4> reduce_dims;
  for (int64_t i = 0; i < grad_rank; ++i) {
    int64_t grad_dim = grad_shape[grad_rank - 1 - i];
    int64_t target_dim = (i < target_rank) ? target_shape[target_rank - 1 - i] : 1;
    if (grad_dim != target_dim)
        reduce_dims.push_back(grad_rank - 1 - i);
  }

  if (reduce_dims.empty() && grad_rank == target_rank)
    return grad;
  std::sort(reduce_dims.begin(), reduce_dims.end());

  llvm::SmallVector<int64_t, 4> intermediate_shape;
  for (int64_t i = 0; i < grad_rank; ++i) {
    bool is_reduced = false;
    for (auto d : reduce_dims)
    if (d == i)
        is_reduced = true;
    if (!is_reduced)
        intermediate_shape.push_back(grad_shape[i]);
  }
  if (intermediate_shape.empty()) {
    intermediate_shape.push_back(1);
  }
  auto intermediate_type = mlir::RankedTensorType::get(intermediate_shape, grad_type.getElementType());
  auto reduced = builder.create<mlir::nova::ReduceOp> (
                  loc, mlir::nova::ReductionKind::SUM, grad, 
                  intermediate_type, false,
                  reduce_dims, false);

  return builder.create<mlir::nova::ReshapeOp>(loc, target_type, reduced.getResult()).getResult();
}

//===--------------------------------------------------------------------------------------------===//
// Arithmetic forward and backward operations: add, sub, mul, div, matmul
//===--------------------------------------------------------------------------------------------===//
// nova.cast → linalg.generic using arith element-wise casts (no TOSA dependency).
struct NovaCastOpLowering : public OpConversionPattern<mlir::nova::CastOp> {
  using OpConversionPattern<mlir::nova::CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::nova::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src      = adaptor.getInput();
    auto srcType   = cast<RankedTensorType>(src.getType());
    auto dstType   = cast<RankedTensorType>(op.getType());
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return success();
    }
    auto srcElem = srcType.getElementType();
    auto dstElem = dstType.getElementType();
    int64_t rank = dstType.getRank();
    MLIRContext *ctx = rewriter.getContext();
    auto identMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), dstElem);
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, dstType, /*inputs=*/src, /*outputs=*/outEmpty,
        SmallVector<AffineMap>{identMap, identMap}, iters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value val = args[0];
          Value castVal;
          if (isa<FloatType>(srcElem) && isa<FloatType>(dstElem)) {
            unsigned srcW = cast<FloatType>(srcElem).getWidth();
            unsigned dstW = cast<FloatType>(dstElem).getWidth();
            castVal = srcW < dstW ? b.create<arith::ExtFOp>(nl, dstElem, val).getResult()
                                  : b.create<arith::TruncFOp>(nl, dstElem, val).getResult();
          } else if (isa<IntegerType>(srcElem) && isa<FloatType>(dstElem)) {
            castVal = b.create<arith::SIToFPOp>(nl, dstElem, val).getResult();
          } else if (isa<FloatType>(srcElem) && isa<IntegerType>(dstElem)) {
            castVal = b.create<arith::FPToSIOp>(nl, dstElem, val).getResult();
          } else {
            unsigned srcW = cast<IntegerType>(srcElem).getWidth();
            unsigned dstW = cast<IntegerType>(dstElem).getWidth();
            if (srcW < dstW)
              castVal = b.create<arith::ExtSIOp>(nl, dstElem, val).getResult();
            else if (srcW > dstW)
              castVal = b.create<arith::TruncIOp>(nl, dstElem, val).getResult();
            else
              castVal = val;
          }
          b.create<linalg::YieldOp>(nl, castVal);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};
// struct NovaAddBackwardPattern : public OpConversionPattern<mlir::nova::AddBackwardOp> {
//   using OpConversionPattern<mlir::nova::AddBackwardOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(mlir::nova::AddBackwardOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value g = adaptor.getGradOut();
//     Value gx = reduce_to_shape_nova(rewriter, loc, g, adaptor.getX());
//     Value gy = reduce_to_shape_nova(rewriter, loc, g, adaptor.getY());
//     rewriter.replaceOp(op, {gx, gy});
//     return success();
//   }
// };

// struct NovaSubBackwardPattern : public OpConversionPattern<mlir::nova::SubBackwardOp> {
//   using OpConversionPattern<mlir::nova::SubBackwardOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(mlir::nova::SubBackwardOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value g = adaptor.getGradOut();
//     Value neg_g = rewriter.create<mlir::nova::NegOp>(loc, g.getType(), g).getResult();
//     Value gx = reduce_to_shape_nova(rewriter, loc, g, adaptor.getX());
//     Value gy = reduce_to_shape_nova(rewriter, loc, neg_g, adaptor.getY());
//     rewriter.replaceOp(op, {gx, gy});
//     return success();
//   }
// };

// struct NovaMulBackwardPattern : public OpConversionPattern<mlir::nova::MulBackwardOp> {
//   using OpConversionPattern<mlir::nova::MulBackwardOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(mlir::nova::MulBackwardOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value g = adaptor.getGradOut();
//     Value dx_p = rewriter.create<mlir::nova::MulOp>(loc, g.getType(), g, adaptor.getY()).getResult();
//     Value dy_p = rewriter.create<mlir::nova::MulOp>(loc, g.getType(), g, adaptor.getX()).getResult();
//     Value gx = reduce_to_shape_nova(rewriter, loc, dx_p, adaptor.getX());
//     Value gy = reduce_to_shape_nova(rewriter, loc, dy_p, adaptor.getY());
//     rewriter.replaceOp(op, {gx, gy});
//     return success();
//   }
// };

// struct NovaDivBackwardPattern : public OpConversionPattern<mlir::nova::DivBackwardOp> {
//   using OpConversionPattern<mlir::nova::DivBackwardOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(mlir::nova::DivBackwardOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value g = adaptor.getGradOut();
//     Value x = adaptor.getX();
//     Value y = adaptor.getY();
//     Value inv_y = rewriter.create<mlir::nova::ReciprocalOp>(loc, y.getType(), y).getResult();
//     Value dx_p = rewriter.create<mlir::nova::MulOp>(loc, g.getType(), g, inv_y).getResult();
//     Value y2 = rewriter.create<mlir::nova::MulOp>(loc, y.getType(), y, y).getResult();
//     Value inv_y2 = rewriter.create<mlir::nova::ReciprocalOp>(loc, y.getType(), y2).getResult();
//     Value dy_pre = rewriter.create<mlir::nova::MulOp>(loc, x.getType(), x, inv_y2).getResult();
//     Value dy_mid = rewriter.create<mlir::nova::MulOp>(loc, g.getType(), g, dy_pre).getResult();
//     Value dy_p = rewriter.create<mlir::nova::NegOp>(loc, dy_mid.getType(), dy_mid).getResult();
//     Value gx = reduce_to_shape_nova(rewriter, loc, dx_p, x);
//     Value gy = reduce_to_shape_nova(rewriter, loc, dy_p, y);
//     rewriter.replaceOp(op, {gx, gy});
//     return success();
//   }
// };

// struct NovaMatmulBackwardPattern : public OpConversionPattern<mlir::nova::MatmulBackwardOp> {
//   using OpConversionPattern<mlir::nova::MatmulBackwardOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(mlir::nova::MatmulBackwardOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value g = adaptor.getGradOut();
//     Value x = adaptor.getX();
//     Value y = adaptor.getY();
//     auto xType = llvm::cast<mlir::RankedTensorType>(x.getType());
//     auto yType = llvm::cast<mlir::RankedTensorType>(y.getType());

//     auto xShape = xType.getShape().vec();
//     if (xShape.size() >= 2) std::swap(xShape.back(), xShape[xShape.size()-2]);
//     auto xtType = mlir::RankedTensorType::get(xShape, xType.getElementType());
//     Value xt = rewriter.create<mlir::nova::TransposeOp>(loc, xtType, x, rewriter.getI32IntegerAttr(-1), rewriter.getI32IntegerAttr(-2)).getResult();

//     auto yShape = yType.getShape().vec();
//     if (yShape.size() >= 2) std::swap(yShape.back(), yShape[yShape.size()-2]);
//     auto ytType = mlir::RankedTensorType::get(yShape, yType.getElementType());
//     Value yt = rewriter.create<mlir::nova::TransposeOp>(loc, ytType, y, rewriter.getI32IntegerAttr(-1), rewriter.getI32IntegerAttr(-2)).getResult();

//     Value da = rewriter.create<mlir::nova::MatmulOp>(loc, xType, g, yt).getResult();
//     Value db = rewriter.create<mlir::nova::MatmulOp>(loc, yType, xt, g).getResult();
//     rewriter.replaceOp(op, {da, db});
//     return success();
//   }
// };

//===--------------------------------------------------------------------------------------------===//
// Loss forward and backward operations: mae, mse, cce, bce, sce
//===--------------------------------------------------------------------------------------------===//

struct NovaMaeForwardLowering : public OpConversionPattern<mlir::nova::MaeOp> {
  using OpConversionPattern<mlir::nova::MaeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::MaeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // loss = reduce_mean(abs(arg0 - arg1))                
    Location loc = op.getLoc();
    Value pred = adaptor.getOperands()[0];
    Value tgt  = adaptor.getOperands()[1];

    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();

    auto predType = cast<RankedTensorType>(pred.getType());
    auto absType  = RankedTensorType::get(predType.getShape(), elemType);

    // Cast inputs to result element type if needed
    if (predType.getElementType() != elemType)
      pred = rewriter.create<tosa::CastOp>(loc, absType, pred);
    auto tgtCastType = RankedTensorType::get(
        cast<RankedTensorType>(tgt.getType()).getShape(), elemType);
    if (cast<RankedTensorType>(tgt.getType()).getElementType() != elemType)
      tgt = rewriter.create<tosa::CastOp>(loc, tgtCastType, tgt);

    // Single fused generic: abs(pred - tgt)
    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = absType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<utils::IteratorType> parallelIters(rank,
        utils::IteratorType::parallel);

    Value absEmpty = rewriter.create<tensor::EmptyOp>(
        loc, absType.getShape(), elemType);
    Value absResult = rewriter.create<linalg::GenericOp>(
        loc, absType,
        /*inputs=*/ValueRange{pred, tgt},
        /*outputs=*/absEmpty,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap},
        parallelIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value diff   = b.create<arith::SubFOp>(nl, args[0], args[1]);
          Value absVal = b.create<math::AbsFOp>(nl, diff);
          b.create<linalg::YieldOp>(nl, absVal);
        }).getResult(0);

    // Full reduction mean — stays in Nova dialect
    llvm::SmallVector<int64_t> dimensions;
    for (int64_t i = 0; i < rank; ++i)
      dimensions.push_back(i);

    auto scalarType = RankedTensorType::get({1}, elemType);
    Value mean = rewriter.create<mlir::nova::ReduceOp>(
        loc, mlir::nova::ReductionKind::MEAN, absResult,
        scalarType, /*keepdims=*/false, dimensions).getResult();

    rewriter.replaceOp(op, mean);
    return success();
  }
};

// struct NovaMaeBackwardLowering : public OpConversionPattern<mlir::nova::MaeBackwardOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(mlir::nova::MaeBackwardOp op,
//                   mlir::nova::MaeBackwardOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto operands = adaptor.getOperands();
//     auto resultType = dyn_cast<RankedTensorType>(op.getType());
//     if (!resultType)
//       return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

//     auto floatType = dyn_cast<FloatType>(resultType.getElementType());
//     if (!floatType)
//       return rewriter.notifyMatchFailure(op, "expected float element type");

//     auto predTensorType = cast<RankedTensorType>(op.getPred().getType());
//     int64_t numel = predTensorType.getNumElements();
//     int64_t rank = resultType.getRank();
//     Location loc = op.getLoc();

//     Value out = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
//                                                   floatType);

//     SmallVector<AffineMap> maps;
//     for (Value v : operands)
//       maps.push_back(rewriter.getMultiDimIdentityMap(rank));
//     maps.push_back(rewriter.getMultiDimIdentityMap(rank));

//     auto linalgOp = rewriter.create<linalg::GenericOp>(
//         loc, out.getType(), operands, out, maps,
//         getNParallelLoopsAttrs(rank),
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value grad_out = args[0];
//           Value pred    = args[1];
//           Value target  = args[2];

//           Value scale   = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0 / numel));
//           Value zero    = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 0.0));
//           Value one     = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0));
//           Value neg_one = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, -1.0));

//           Value diff = b.create<arith::SubFOp>(loc, pred, target);
//           Value gt   = b.create<arith::CmpFOp>(
//               loc, arith::CmpFPredicate::OGT, diff, zero);
//           Value lt   = b.create<arith::CmpFOp>(
//               loc, arith::CmpFPredicate::OLT, diff, zero);

//           Value sign = b.create<arith::SelectOp>(
//               loc, gt, one,
//               b.create<arith::SelectOp>(loc, lt, neg_one, zero));

//           Value dx_pre = b.create<arith::MulFOp>(loc, sign, scale);
//           Value result = b.create<arith::MulFOp>(loc, dx_pre, grad_out);
//           b.create<linalg::YieldOp>(loc, result);
//         });

//     rewriter.replaceOp(op, linalgOp->getResults());
//     return success();
//   }
// };

struct NovaMseForwardLowering : public OpConversionPattern<mlir::nova::MseOp> {
  using OpConversionPattern<mlir::nova::MseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::MseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // loss = reduce_mean(square(arg0 - arg1))
    Location loc = op.getLoc();
    Value pred = adaptor.getOperands()[0];
    Value tgt  = adaptor.getOperands()[1];

    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();

    auto predType  = cast<RankedTensorType>(pred.getType());
    auto sqType    = RankedTensorType::get(predType.getShape(), elemType);

    // Cast inputs to result element type if needed
    if (predType.getElementType() != elemType)
      pred = rewriter.create<tosa::CastOp>(loc, sqType, pred);
    auto tgtCastType = RankedTensorType::get(
        cast<RankedTensorType>(tgt.getType()).getShape(), elemType);
    if (cast<RankedTensorType>(tgt.getType()).getElementType() != elemType)
      tgt = rewriter.create<tosa::CastOp>(loc, tgtCastType, tgt);

    // Single fused generic: (pred - tgt)^2
    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = sqType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<utils::IteratorType> parallelIters(rank,
        utils::IteratorType::parallel);

    Value sqEmpty = rewriter.create<tensor::EmptyOp>(
        loc, sqType.getShape(), elemType);
    Value sqResult = rewriter.create<linalg::GenericOp>(
        loc, sqType,
        /*inputs=*/ValueRange{pred, tgt},
        /*outputs=*/sqEmpty,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap},
        parallelIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value diff = b.create<arith::SubFOp>(nl, args[0], args[1]);
          Value sq   = b.create<arith::MulFOp>(nl, diff, diff);
          b.create<linalg::YieldOp>(nl, sq);
        }).getResult(0);

    // Full reduction mean — stays in Nova dialect
    llvm::SmallVector<int64_t> dimensions;
    for (int64_t i = 0; i < rank; ++i)
      dimensions.push_back(i);

    auto scalarType = RankedTensorType::get({1}, elemType);
    Value mean = rewriter.create<mlir::nova::ReduceOp>(
        loc, mlir::nova::ReductionKind::MEAN, sqResult,
        scalarType, /*keepdims=*/false, dimensions).getResult();

    rewriter.replaceOp(op, mean);
    return success();
  }
};

// struct NovaMseBackwardLowering : public OpConversionPattern<mlir::nova::MseBackwardOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(nova::MseBackwardOp op,
//                   nova::MseBackwardOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto operands = adaptor.getOperands();
//     auto resultType = dyn_cast<RankedTensorType>(op.getType());
//     if (!resultType)
//       return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

//     auto floatType = dyn_cast<FloatType>(resultType.getElementType());
//     if (!floatType)
//       return rewriter.notifyMatchFailure(op, "expected float element type");

//     auto predTensorType = cast<RankedTensorType>(op.getPred().getType());
//     int64_t numel = predTensorType.getNumElements();
//     int64_t rank = resultType.getRank();
//     Location loc = op.getLoc();

//     Value out = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
//                                                   floatType);

//     SmallVector<AffineMap> maps;
//     for (size_t i = 0; i < operands.size(); ++i)
//       maps.push_back(rewriter.getMultiDimIdentityMap(rank));
//     maps.push_back(rewriter.getMultiDimIdentityMap(rank));

//     auto linalgOp = rewriter.create<linalg::GenericOp>(
//         loc, out.getType(), operands, out, maps,
//         getNParallelLoopsAttrs(rank),
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value grad_out = args[0];
//           Value pred     = args[1];
//           Value target   = args[2];

//           Value two   = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 2.0));
//           Value scale = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0 / numel));

//           Value diff         = b.create<arith::SubFOp>(loc, pred, target);
//           Value scaled_diff  = b.create<arith::MulFOp>(loc, diff, scale);
//           Value scaled_diff2 = b.create<arith::MulFOp>(loc, scaled_diff, two);
//           Value result       = b.create<arith::MulFOp>(loc, scaled_diff2, grad_out);
//           b.create<linalg::YieldOp>(loc, result);
//         });

//     rewriter.replaceOp(op, linalgOp->getResults());
//     return success();
//   }
// };

struct NovaCceForwardLowering : public OpConversionPattern<mlir::nova::CceOp> {
  using OpConversionPattern<mlir::nova::CceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::CceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Generic 1: term[i] = yA * log(clip(yP, eps, 1-eps))
    // nova.reduce MEAN
    // Generic 2: result = -mean  (negates the single scalar element)
    // loss = -reduce_mean(yA * log(clip(yP, eps, 1-eps)))
    Location loc = op.getLoc();
    Value pred = adaptor.getOperands()[0]; // yP
    Value tgt  = adaptor.getOperands()[1]; // yA

    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();

    auto predType = cast<RankedTensorType>(pred.getType());
    auto inputType = RankedTensorType::get(predType.getShape(), elemType);

    // Cast inputs to result element type if needed
    if (predType.getElementType() != elemType)
      pred = rewriter.create<tosa::CastOp>(loc, inputType, pred);
    auto tgtCastType = RankedTensorType::get(
        cast<RankedTensorType>(tgt.getType()).getShape(), elemType);
    if (cast<RankedTensorType>(tgt.getType()).getElementType() != elemType)
      tgt = rewriter.create<tosa::CastOp>(loc, tgtCastType, tgt);

    // Scalar clipping constants — outside the generic body
    Value epsVal    = rewriter.create<arith::ConstantOp>(loc,
        rewriter.getFloatAttr(elemType, 1e-7));
    Value oneepsVal = rewriter.create<arith::ConstantOp>(loc,
        rewriter.getFloatAttr(elemType, 1.0 - 1e-7));

    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = inputType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<utils::IteratorType> parallelIters(rank,
        utils::IteratorType::parallel);

    // Generic 1: term[i] = yA * log(clip(yP, eps, 1-eps))
    Value termEmpty = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), elemType);
    Value termResult = rewriter.create<linalg::GenericOp>(
        loc, inputType,
        /*inputs=*/ValueRange{pred, tgt},
        /*outputs=*/termEmpty,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap},
        parallelIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value lo      = b.create<arith::MaximumFOp>(nl, args[0], epsVal);
          Value clipped = b.create<arith::MinimumFOp>(nl, lo, oneepsVal);
          Value logVal  = b.create<math::LogOp>(nl, clipped);
          Value term    = b.create<arith::MulFOp>(nl, logVal, args[1]);
          b.create<linalg::YieldOp>(nl, term);
        }).getResult(0);

    // nova.reduce MEAN — full reduction, stays in Nova dialect → shape {1}
    llvm::SmallVector<int64_t> dimensions;
    for (int64_t i = 0; i < rank; ++i)
      dimensions.push_back(i);

    auto scalarType = RankedTensorType::get({1}, elemType);
    Value mean = rewriter.create<mlir::nova::ReduceOp>(
        loc, mlir::nova::ReductionKind::MEAN, termResult,
        scalarType, /*keepdims=*/false, dimensions).getResult();

    // Generic 2: negate the single scalar element
    auto scalarIdentity = AffineMap::getMultiDimIdentityMap(1, ctx);
    SmallVector<utils::IteratorType> scalarParallel(1,
        utils::IteratorType::parallel);
    Value negEmpty = rewriter.create<tensor::EmptyOp>(loc,
        ArrayRef<int64_t>{1}, elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        loc, scalarType,
        /*inputs=*/mean,
        /*outputs=*/negEmpty,
        SmallVector<AffineMap>{scalarIdentity, scalarIdentity},
        scalarParallel,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value neg = b.create<arith::NegFOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, neg);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// struct NovaCceBackwardLowering : public OpConversionPattern<nova::CceBackwardOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(nova::CceBackwardOp op,
//                   nova::CceBackwardOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto operands = adaptor.getOperands();
//     auto resultType = dyn_cast<RankedTensorType>(op.getType());
//     if (!resultType)
//       return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

//     auto floatType = dyn_cast<FloatType>(resultType.getElementType());
//     if (!floatType)
//       return rewriter.notifyMatchFailure(op, "expected float element type");

//     auto inputTensorType = cast<RankedTensorType>(op.getInput().getType());
//     int64_t batch_size = inputTensorType.getShape()[0];
//     int64_t rank = resultType.getRank();
//     Location loc = op.getLoc();

//     Value out = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
//                                                   floatType);

//     SmallVector<AffineMap> maps;
//     for (size_t i = 0; i < operands.size(); ++i)
//       maps.push_back(rewriter.getMultiDimIdentityMap(rank));
//     maps.push_back(rewriter.getMultiDimIdentityMap(rank));

//     auto linalgOp = rewriter.create<linalg::GenericOp>(
//         loc, out.getType(), operands, out, maps,
//         getNParallelLoopsAttrs(rank),
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value grad_out = args[0];
//           Value pred     = args[1];
//           Value target   = args[2];

//           Value scale      = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0 / batch_size));
//           Value eps        = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1e-7));
//           Value one        = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0));
//           Value zero       = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 0.0));
//           Value oneminuseps = b.create<arith::SubFOp>(loc, one, eps);

//           Value ge_eps     = b.create<arith::CmpFOp>(
//               loc, arith::CmpFPredicate::OGE, pred, eps);
//           Value le_ome     = b.create<arith::CmpFOp>(
//               loc, arith::CmpFPredicate::OLE, pred, oneminuseps);
//           Value valid      = b.create<arith::AndIOp>(loc, ge_eps, le_ome);

//           Value neg_target = b.create<arith::NegFOp>(loc, target);
//           Value raw_grad   = b.create<arith::DivFOp>(loc, neg_target, pred);

//           Value masked_grad = b.create<arith::SelectOp>(loc, valid, raw_grad, zero);
//           Value dx_pre      = b.create<arith::MulFOp>(loc, masked_grad, scale);
//           Value result      = b.create<arith::MulFOp>(loc, dx_pre, grad_out);
//           b.create<linalg::YieldOp>(loc, result);
//         });

//     rewriter.replaceOp(op, linalgOp->getResults());
//     return success();
//   }
// };
													
struct NovaBceForwardLowering : public OpConversionPattern<mlir::nova::BceOp> {
  using OpConversionPattern<mlir::nova::BceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::BceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Generic 1: bce[i] = yA*log(clip(yP)) + (1-yA)*log(1-clip(yP))
    // nova.reduce MEAN
    // Generic 2: result = -mean  (negates the single scalar element)
    // loss = -reduce_mean(yA*log(clip(yP)) + (1-yA)*log(1-clip(yP)))
    Location loc = op.getLoc();
    Value pred = adaptor.getOperands()[0]; // yP
    Value tgt  = adaptor.getOperands()[1]; // yA

    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();

    auto predType = cast<RankedTensorType>(pred.getType());
    auto inputType = RankedTensorType::get(predType.getShape(), elemType);

    // Cast inputs to result element type if needed
    if (predType.getElementType() != elemType)
      pred = rewriter.create<tosa::CastOp>(loc, inputType, pred);
    auto tgtCastType = RankedTensorType::get(
        cast<RankedTensorType>(tgt.getType()).getShape(), elemType);
    if (cast<RankedTensorType>(tgt.getType()).getElementType() != elemType)
      tgt = rewriter.create<tosa::CastOp>(loc, tgtCastType, tgt);

    // Scalar clipping constants — outside the generic body
    Value epsVal    = rewriter.create<arith::ConstantOp>(loc,
        rewriter.getFloatAttr(elemType, 1e-7));
    Value oneepsVal = rewriter.create<arith::ConstantOp>(loc,
        rewriter.getFloatAttr(elemType, 1.0 - 1e-7));
    Value oneVal    = rewriter.create<arith::ConstantOp>(loc,
        rewriter.getFloatAttr(elemType, 1.0));

    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = inputType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<utils::IteratorType> parallelIters(rank,
        utils::IteratorType::parallel);

    // Generic 1: bce[i] = yA*log(clip(yP)) + (1-yA)*log(1-clip(yP))
    Value bceEmpty = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), elemType);
    Value bceResult = rewriter.create<linalg::GenericOp>(
        loc, inputType,
        /*inputs=*/ValueRange{pred, tgt},
        /*outputs=*/bceEmpty,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap},
        parallelIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // clip yP to [eps, 1-eps]
          Value lo      = b.create<arith::MaximumFOp>(nl, args[0], epsVal);
          Value clipped = b.create<arith::MinimumFOp>(nl, lo, oneepsVal);
          // term1 = yA * log(clipped)
          Value log1    = b.create<math::LogOp>(nl, clipped);
          Value term1   = b.create<arith::MulFOp>(nl, args[1], log1);
          // term2 = (1 - yA) * log(1 - clipped)
          Value oneMtgt = b.create<arith::SubFOp>(nl, oneVal, args[1]);
          Value oneMcp  = b.create<arith::SubFOp>(nl, oneVal, clipped);
          Value log2    = b.create<math::LogOp>(nl, oneMcp);
          Value term2   = b.create<arith::MulFOp>(nl, oneMtgt, log2);
          Value bce     = b.create<arith::AddFOp>(nl, term1, term2);
          b.create<linalg::YieldOp>(nl, bce);
        }).getResult(0);

    // nova.reduce MEAN — full reduction, stays in Nova dialect → shape {1}
    llvm::SmallVector<int64_t> dimensions;
    for (int64_t i = 0; i < rank; ++i)
      dimensions.push_back(i);

    auto scalarType = RankedTensorType::get({1}, elemType);
    Value mean = rewriter.create<mlir::nova::ReduceOp>(
        loc, mlir::nova::ReductionKind::MEAN, bceResult,
        scalarType, /*keepdims=*/false, dimensions).getResult();

    // Generic 2: negate the single scalar element
    auto scalarIdentity = AffineMap::getMultiDimIdentityMap(1, ctx);
    SmallVector<utils::IteratorType> scalarParallel(1,
        utils::IteratorType::parallel);
    Value negEmpty = rewriter.create<tensor::EmptyOp>(loc,
        ArrayRef<int64_t>{1}, elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        loc, scalarType,
        /*inputs=*/mean,
        /*outputs=*/negEmpty,
        SmallVector<AffineMap>{scalarIdentity, scalarIdentity},
        scalarParallel,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value neg = b.create<arith::NegFOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, neg);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// struct NovaBceBackwardLowering : public OpConversionPattern<nova::BceBackwardOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(nova::BceBackwardOp op,
//                   nova::BceBackwardOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto operands = adaptor.getOperands();
//     auto resultType = dyn_cast<RankedTensorType>(op.getType());
//     if (!resultType)
//       return rewriter.notifyMatchFailure(op, "expected ranked tensor result");


//     auto floatType = dyn_cast<FloatType>(resultType.getElementType());
//     if (!floatType)
//       return rewriter.notifyMatchFailure(op, "expected float element type");

//     auto predTensorType = cast<RankedTensorType>(op.getPred().getType());
//     int64_t numel = predTensorType.getNumElements();
//     int64_t rank = resultType.getRank();
//     Location loc = op.getLoc();

//     Value out = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
//                                                   floatType);

//     SmallVector<AffineMap> maps;
//     for (size_t i = 0; i < operands.size(); ++i)
//       maps.push_back(rewriter.getMultiDimIdentityMap(rank));
//     maps.push_back(rewriter.getMultiDimIdentityMap(rank));

//     auto linalgOp = rewriter.create<linalg::GenericOp>(
//         loc, out.getType(), operands, out, maps,
//         getNParallelLoopsAttrs(rank),
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value grad_out = args[0];
//           Value pred     = args[1];
//           Value target   = args[2];

//           Value scale      = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0 / numel));
//           Value eps        = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1e-7));
//           Value one        = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(floatType, 1.0));
//           Value oneminuseps = b.create<arith::SubFOp>(loc, one, eps);

//           Value p_clamped  = b.create<arith::MaximumFOp>(loc, pred, eps);
//           Value p_clipped  = b.create<arith::MinimumFOp>(loc, p_clamped, oneminuseps);

//           Value num        = b.create<arith::SubFOp>(loc, p_clipped, target);
//           Value oneminusp  = b.create<arith::SubFOp>(loc, one, p_clipped);
//           Value denom      = b.create<arith::MulFOp>(loc, p_clipped, oneminusp);
//           Value raw_grad   = b.create<arith::DivFOp>(loc, num, denom);

//           Value dx_pre     = b.create<arith::MulFOp>(loc, raw_grad, scale);
//           Value result     = b.create<arith::MulFOp>(loc, dx_pre, grad_out);
//           b.create<linalg::YieldOp>(loc, result);
//         });

//     rewriter.replaceOp(op, linalgOp->getResults());
//     return success();
//   }
// };
			
struct NovaSceForwardLowering : public OpConversionPattern<mlir::nova::SceOp> {
  using OpConversionPattern<mlir::nova::SceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::SceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Generic 1 : maxVal   = max(logits, axis=lastDim)               [reduction]
    // Generic 2 : sumExp   = sum(exp(logits - maxVal), axis=lastDim)  [fused reduction]
    // Generic 3 : softmax  = exp(logits - maxVal) / clamp(sumExp,eps) [all-parallel]
    // nova.gather(softmax, targets, axis=lastDim) -> [batchShape]
    // Generic 4 : psl      = -log(clamp(gathered, eps, 1))            [all-parallel]
    // nova.reduce MEAN over batchShape -> {1}
    Location loc = op.getLoc();
    Value logits  = adaptor.getLogits();
    Value targets = adaptor.getTargets();

    auto softmaxResultType = cast<RankedTensorType>(op.getSoftmax().getType());
    auto lossResultType    = cast<RankedTensorType>(op.getResult().getType());
    auto elemType          = softmaxResultType.getElementType();

    auto targetsType = cast<RankedTensorType>(targets.getType());
    if (!targetsType.getElementType().isInteger(32)) {
      auto i32Type = RankedTensorType::get(targetsType.getShape(),
                                           rewriter.getI32Type());
      targets     = rewriter.create<tosa::CastOp>(loc, i32Type, targets);
      targetsType = cast<RankedTensorType>(targets.getType());
    }

    auto logitsType = cast<RankedTensorType>(logits.getType());
    if (logitsType.getElementType() != elemType) {
      auto castType = RankedTensorType::get(logitsType.getShape(), elemType);
      logits    = rewriter.create<tosa::CastOp>(loc, castType, logits);
      logitsType = cast<RankedTensorType>(logits.getType());
    }

    MLIRContext *ctx = rewriter.getContext();
    int64_t rank     = logitsType.getRank();
    int64_t lastDim  = rank - 1;

    SmallVector<AffineExpr> inputExprs, statsExprs;
    for (int64_t i = 0; i < rank; ++i)
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    for (int64_t i = 0; i < rank; ++i)
      if (i != lastDim) statsExprs.push_back(rewriter.getAffineDimExpr(i));

    auto inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
    auto statsMap = AffineMap::get(rank, 0, statsExprs, ctx);

    SmallVector<utils::IteratorType> redIters;
    for (int64_t i = 0; i < rank; ++i)
      redIters.push_back(i == lastDim ? utils::IteratorType::reduction
                                      : utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> parIters(rank, utils::IteratorType::parallel);

    SmallVector<int64_t> statsShape;
    for (int64_t i = 0; i < rank; ++i)
      if (i != lastDim) statsShape.push_back(logitsType.getDimSize(i));
    auto statsType = RankedTensorType::get(statsShape, elemType);
    auto batchType = RankedTensorType::get(statsShape, elemType);
    int64_t batchRank = static_cast<int64_t>(statsShape.size());

    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(
                 elemType, -std::numeric_limits<float>::infinity()));
    Value fZero  = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value epsVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 1.0e-7f));
    Value fOne   = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 1.0f));

    Value maxEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
    Value maxInit  = rewriter.create<linalg::FillOp>(loc, negInf, maxEmpty).result();
    Value maxVal   = rewriter.create<linalg::GenericOp>(
        loc, statsType,
        /*inputs=*/logits, /*outputs=*/maxInit,
        SmallVector<AffineMap>{inputMap, statsMap},
        redIters,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value bb= b.create<arith::MaximumFOp>(nl, args[0], args[1]);
          b.create<linalg::YieldOp>(nl, bb);
        }).getResult(0);

    Value sumEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
    Value sumInit  = rewriter.create<linalg::FillOp>(loc, fZero, sumEmpty).result();
    Value sumExp   = rewriter.create<linalg::GenericOp>(
        loc, statsType,
        /*inputs=*/ValueRange{logits, maxVal},
        /*outputs=*/sumInit,
        SmallVector<AffineMap>{inputMap, statsMap, statsMap},
        redIters,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value shifted = b.create<arith::SubFOp>(nl, args[0], args[1]);
          Value expV    = b.create<math::ExpOp>(nl, shifted);
          Value sumV    = b.create<arith::AddFOp>(nl, args[2], expV);
          b.create<linalg::YieldOp>(nl, sumV);
        }).getResult(0);

    // Generic 3: logSumExp = log(max(sumExp, eps))
    auto batchIdentity = AffineMap::getMultiDimIdentityMap(batchRank, ctx);
    SmallVector<utils::IteratorType> batchParallel(
        batchRank, utils::IteratorType::parallel);
    Value logSumExp = rewriter.create<linalg::GenericOp>(
        loc, statsType,
        /*inputs=*/sumExp,
        /*outputs=*/rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType).getResult(),
        SmallVector<AffineMap>{batchIdentity, batchIdentity},
        batchParallel,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value safeSum = b.create<arith::MaximumFOp>(nl, args[0], epsVal);
          Value logSum  = b.create<math::LogOp>(nl, safeSum);
          b.create<linalg::YieldOp>(nl, logSum);
        }).getResult(0);

    // Generic 4: softmax = exp(logits - maxVal) / sumExp
    Value smEmpty = rewriter.create<tensor::EmptyOp>(
        loc, softmaxResultType.getShape(), elemType);
    Value softmax = rewriter.create<linalg::GenericOp>(
        loc, softmaxResultType,
        /*inputs=*/ValueRange{logits, maxVal, sumExp},
        /*outputs=*/smEmpty,
        SmallVector<AffineMap>{inputMap, statsMap, statsMap, inputMap},
        parIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value shifted = b.create<arith::SubFOp>(nl, args[0], args[1]);
          Value expV    = b.create<math::ExpOp>(nl, shifted);
          Value safeSum = b.create<arith::MaximumFOp>(nl, args[2], epsVal);
          Value divV    = b.create<arith::DivFOp>(nl, expV, safeSum);
          b.create<linalg::YieldOp>(nl, divV);
        }).getResult(0);

    // Stable Loss Calculation:
    // loss = logSumExp - (gatheredLogits - gatheredMaxVal)
    Value gatheredLogits = rewriter.create<nova::GatherOp>(
        loc, logits, targets, lastDim).getResult();

    Value pslEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
    Value perSampleLoss = rewriter.create<linalg::GenericOp>(
        loc, statsType,
        /*inputs=*/ValueRange{gatheredLogits, maxVal, logSumExp},
        /*outputs=*/pslEmpty,
        SmallVector<AffineMap>{batchIdentity, batchIdentity, batchIdentity, batchIdentity},
        batchParallel,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // loss = logSumExp - (logits - maxVal)
          Value shifted = b.create<arith::SubFOp>(nl, args[0], args[1]);
          Value loss    = b.create<arith::SubFOp>(nl, args[2], shifted);
          b.create<linalg::YieldOp>(nl, loss);
        }).getResult(0);

    llvm::SmallVector<int64_t> allDims;
    for (int64_t i = 0; i < batchRank; ++i)
      allDims.push_back(i);

    auto scalarType = RankedTensorType::get({1}, elemType);
    Value reducedLoss = rewriter.create<nova::ReduceOp>(
        loc, nova::ReductionKind::MEAN, perSampleLoss,
        scalarType, /*keepdims=*/false, allDims, /*ignore_nan=*/false);

    Value finalLoss = reducedLoss;
    if (cast<RankedTensorType>(reducedLoss.getType()) != lossResultType) {
      auto shapeAttrType = RankedTensorType::get(
          {lossResultType.getRank()}, rewriter.getIndexType());
      auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
          loc,
          mlir::tosa::shapeType::get(rewriter.getContext(),
                                     lossResultType.getRank()),
          DenseIntElementsAttr::get(shapeAttrType, lossResultType.getShape()));
      finalLoss = rewriter.create<tosa::ReshapeOp>(
          loc, lossResultType, reducedLoss, shapeConst);
    }

    rewriter.replaceOp(op, {softmax, finalLoss});
    return success();
  }
};
struct NovaSceBackwardLowering : public OpConversionPattern<mlir::nova::SceBackwardOp> {
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

//===--------------------------------------------------------------------------------------------===//
// Activation forward and backward lowerings: Sigmoid, Tanh, Gelu, Softmax
//===--------------------------------------------------------------------------------------------===//

struct NovaSigmoidForwardLowering : public OpConversionPattern<mlir::nova::SigmoidOp> {
  using OpConversionPattern<mlir::nova::SigmoidOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // sigmoid(x) = 1 / (1 + exp(-x))
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();

    auto resultType = cast<RankedTensorType>(op.getType());
    auto inputType = cast<RankedTensorType>(input.getType());
    auto dstElemType = resultType.getElementType();
    int64_t rank = resultType.getRank();
    MLIRContext *ctx = rewriter.getContext();

    // Identity affine maps (all dims, all-parallel — purely elementwise)
    SmallVector<AffineExpr> dimExprs;
    for (int64_t i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineDimExpr(i));
    auto identMap = AffineMap::get(rank, 0, dimExprs, ctx);
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), dstElemType);

    auto sigmoidGeneric = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        /*inputs=*/input, /*outputs=*/outEmpty,
        SmallVector<AffineMap>{identMap, identMap}, iters,
        [&](OpBuilder &b, Location nl, ValueRange args) {

          Value x = args[0];
          if (inputType.getElementType() != resultType.getElementType()) {
            input = rewriter.create<mlir::tosa::CastOp>(loc, resultType, input);
          }
          // sigmoid(x) = 1 / (1 + exp(-x))
          Value one    = b.create<arith::ConstantOp>(nl, b.getFloatAttr(dstElemType, 1.0));
          Value negX   = b.create<arith::NegFOp>(nl, x);
          Value expNeg = b.create<math::ExpOp>(nl, negX);
          Value denom  = b.create<arith::AddFOp>(nl, one, expNeg);
          Value result = b.create<arith::DivFOp>(nl, one, denom);
          b.create<linalg::YieldOp>(nl, result);
        });

    rewriter.replaceOp(op, sigmoidGeneric.getResult(0));
    return success();
  }
};

// struct NovaSigmoidBackwardLowering : public OpConversionPattern<mlir::nova::SigmoidBackwardOp> {  
//   using OpConversionPattern<mlir::nova::SigmoidBackwardOp>::OpConversionPattern;  
    
//   LogicalResult  
//   matchAndRewrite(mlir::nova::SigmoidBackwardOp op, OpAdaptor adaptor,  
//                   ConversionPatternRewriter &rewriter) const override {  
//     // sigmoid_backward: grad_input = grad_out * output * (1 - output)  
//     Location loc = op.getLoc();  
//     Value grad_out = adaptor.getGradOut();  
//     Value output = adaptor.getOutput();  
  
//     auto resultType = cast<RankedTensorType>(op.getType());   
//     auto dstElemType = resultType.getElementType();  
//     int64_t rank = resultType.getRank();  
//     MLIRContext *ctx = rewriter.getContext();  
  
//     // Identity affine maps (all dims, all-parallel — purely elementwise)  
//     SmallVector<AffineExpr> dimExprs;  
//     for (int64_t i = 0; i < rank; ++i)  
//       dimExprs.push_back(rewriter.getAffineDimExpr(i));  
//     auto identMap = AffineMap::get(rank, 0, dimExprs, ctx);  
//     SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);  
  
//     Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), dstElemType);  
  
//     auto sigmoidBackwardGeneric = rewriter.create<linalg::GenericOp>(  
//         loc, resultType,  
//         /*inputs=*/ValueRange{grad_out, output}, /*outputs=*/outEmpty,  
//         SmallVector<AffineMap>{identMap, identMap, identMap}, iters,  
//         [&](OpBuilder &b, Location nl, ValueRange args) {  
//           // args[0]=grad_out, args[1]=output  
//           // sigmoid_derivative = output * (1 - output)
//           // grad_input = grad_out * sigmoid_derivative
//           Value one = b.create<arith::ConstantOp>(nl, b.getFloatAttr(dstElemType, 1.0));  
//           Value one_minus_output = b.create<arith::SubFOp>(nl, one, args[1]);  
//           Value sigmoid_derivative = b.create<arith::MulFOp>(nl, args[1], one_minus_output);  
//           Value grad_input = b.create<arith::MulFOp>(nl, args[0], sigmoid_derivative);  
//           b.create<linalg::YieldOp>(nl, grad_input);  
//         });  
  
//     rewriter.replaceOp(op, sigmoidBackwardGeneric.getResult(0));  
//     return success();  
//   }  
// };

// struct NovaTanhBackwardLowering : public OpConversionPattern<mlir::nova::TanhBackwardOp> { 
//   using OpConversionPattern<mlir::nova::TanhBackwardOp>::OpConversionPattern; 
    
//   LogicalResult  
//   matchAndRewrite(mlir::nova::TanhBackwardOp op, OpAdaptor adaptor,  
//                   ConversionPatternRewriter &rewriter) const override {  
//     // tanh_backward: grad_input = grad_out * (1 - output^2)  
//     Location loc = op.getLoc();  
//     Value grad_out = adaptor.getGradOut();  
//     Value output = adaptor.getOutput();  
  
//     auto resultType = cast<RankedTensorType>(op.getType());  
//     auto dstElemType = resultType.getElementType();  
//     int64_t rank = resultType.getRank();  
//     MLIRContext *ctx = rewriter.getContext();  
  
//     // Identity affine maps (all dims, all-parallel — purely elementwise)  
//     SmallVector<AffineExpr> dimExprs;  
//     for (int64_t i = 0; i < rank; ++i)  
//       dimExprs.push_back(rewriter.getAffineDimExpr(i));  
//     auto identMap = AffineMap::get(rank, 0, dimExprs, ctx);  
//     SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);  
  
//     Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), dstElemType);  
  
//     auto tanhBackwardGeneric = rewriter.create<linalg::GenericOp>(  
//         loc, resultType,  
//         /*inputs=*/ValueRange{grad_out, output}, /*outputs=*/outEmpty,  
//         SmallVector<AffineMap>{identMap, identMap, identMap}, iters,  
//         [&](OpBuilder &b, Location nl, ValueRange args) {  
//           // args[0]=grad_out, args[1]=output  
//           // tanh_derivative = 1 - output^2
//           // grad_input = grad_out * tanh_derivative
//           Value one = b.create<arith::ConstantOp>(nl, b.getFloatAttr(dstElemType, 1.0));  
//           Value output_sq = b.create<arith::MulFOp>(nl, args[1], args[1]);  
//           Value one_minus_output_sq = b.create<arith::SubFOp>(nl, one, output_sq);  
//           Value grad_input = b.create<arith::MulFOp>(nl, args[0], one_minus_output_sq);  
//           b.create<linalg::YieldOp>(nl, grad_input);  
//         });  
  
//     rewriter.replaceOp(op, tanhBackwardGeneric.getResult(0));  
//     return success();  
//   }  
// };

struct NovaGeluForwardPattern : public OpConversionPattern<mlir::nova::GeluOp> {
  using OpConversionPattern<mlir::nova::GeluOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();

    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    MLIRContext *ctx = rewriter.getContext();

    SmallVector<AffineExpr> dimExprs;
    for (int64_t i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineDimExpr(i));
    auto identMap = AffineMap::get(rank, 0, dimExprs, ctx);
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    // if input is integer, cast to float and update type for following ops
    if (isa<IntegerType>(inputType.getElementType())) {
      auto newInputType = RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
      input = rewriter.create<mlir::tosa::CastOp>(loc, newInputType, input);
      inputType = newInputType;
    }
    auto elemType = inputType.getElementType();
    Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(), elemType);

    auto geluGeneric = rewriter.create<linalg::GenericOp>(
        loc, inputType, /*inputs=*/input, /*outputs=*/outEmpty,
        SmallVector<AffineMap>{identMap, identMap}, iters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value x = args[0];
          Value cst_004 = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 4.471500e-02f));
          Value cst_sqrt2pi = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 0.7978845608028654f));
          Value cst_1  = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 1.0f));
          Value cst_2 = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 0.5f));

          // x^3
          Value x2     = b.create<arith::MulFOp>(nl, x, x);
          Value x3     = b.create<arith::MulFOp>(nl, x2, x);
          // 0.044715 * x^3
          Value cubic  = b.create<arith::MulFOp>(nl, cst_004, x3);
          // x + 0.044715 * x^3
          Value inner  = b.create<arith::AddFOp>(nl, x, cubic);
          // sqrt(2/pi) * inner
          Value scaled = b.create<arith::MulFOp>(nl, cst_sqrt2pi, inner);
          // tanh(scaled)
          Value tanhV  = b.create<math::TanhOp>(nl, scaled);
          // 1 + tanh(...)
          Value onePT  = b.create<arith::AddFOp>(nl, cst_1, tanhV);
          // 0.5 * x * (1 + tanh(...))
          Value halfX  = b.create<arith::MulFOp>(nl, cst_2, x);
          Value result = b.create<arith::MulFOp>(nl, halfX, onePT);
          b.create<linalg::YieldOp>(nl, result);
        });

    rewriter.replaceOp(op, geluGeneric.getResult(0));
    return success();
  }
};

struct NovaGeluBackwardPattern : public OpConversionPattern<mlir::nova::GeluBackwardOp> {
  using OpConversionPattern<mlir::nova::GeluBackwardOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::nova::GeluBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // d_gelu = grad_out * (term1 + term2)
    // term1 = 0.5 * (1 + tanh(g(x)))
    // term2 = 0.5 * x * (1 - tanh^2(g(x))) * g'(x)
    // g(x)  = sqrt2pi * (x + 0.044715 * x^3)
    // g'(x) = sqrt2pi * (1 + 0.134145 * x^2)
    Location loc = op.getLoc();
    Value input    = adaptor.getInput();
    Value grad_out = adaptor.getGradOut();
    
    auto inputType = cast<RankedTensorType>(input.getType());
    if (isa<IntegerType>(inputType.getElementType())) {
      auto newInputType = RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
      input = rewriter.create<mlir::tosa::CastOp>(loc, newInputType, input);
      grad_out = rewriter.create<mlir::tosa::CastOp>(loc, newInputType, grad_out);
      inputType = newInputType;
    }

    auto elemType  = inputType.getElementType();
    int64_t rank   = inputType.getRank();
    MLIRContext *ctx = rewriter.getContext();

    SmallVector<AffineExpr> dimExprs;
    for (int64_t i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineDimExpr(i));
    auto identMap = AffineMap::get(rank, 0, dimExprs, ctx);
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    // Two-input all-parallel generic: (input, grad_out) -> grad_input
    Value outEmpty = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), elemType);

    // identMap used for both inputs and output
    auto geluBwdGeneric = rewriter.create<linalg::GenericOp>(
        loc, inputType,
        /*inputs=*/ValueRange{input, grad_out}, /*outputs=*/outEmpty,
        SmallVector<AffineMap>{identMap, identMap, identMap}, iters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value x  = args[0];
          Value go = args[1];

          Value cst_sqrt2pi = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 0.797884583f));
          Value c004     = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 4.471500e-02f));
          Value c013     = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 0.134145f));
          Value c05      = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 0.5f));
          Value c1       = b.create<arith::ConstantOp>(nl, b.getFloatAttr(elemType, 1.0f));

          // g(x) = sqrt2pi * (x + 0.044715 * x^3)
          Value x2     = b.create<arith::MulFOp>(nl, x, x);
          Value x3     = b.create<arith::MulFOp>(nl, x2, x);
          Value cubic  = b.create<arith::MulFOp>(nl, c004, x3);
          Value inner  = b.create<arith::AddFOp>(nl, x, cubic);
          Value gx     = b.create<arith::MulFOp>(nl, cst_sqrt2pi, inner);

          // g'(x) = sqrt2pi * (1 + (0.134145 * x^2))
          Value xsq_c  = b.create<arith::MulFOp>(nl, x2, c013);
          Value polyG  = b.create<arith::AddFOp>(nl, c1, xsq_c);
          Value dgx    = b.create<arith::MulFOp>(nl, polyG, cst_sqrt2pi);

          // tanh(g(x)), tanh^2
          Value tanhG  = b.create<math::TanhOp>(nl, gx);
          Value tanh2  = b.create<arith::MulFOp>(nl, tanhG, tanhG);

          // term1 = 0.5 * (1 + tanh(g(x)))
          Value onePT  = b.create<arith::AddFOp>(nl, c1, tanhG);
          Value term1  = b.create<arith::MulFOp>(nl, c05, onePT);

          // term2 = 0.5 * x * (1 - tanh^2) * g'(x)
          Value omT2   = b.create<arith::SubFOp>(nl, c1, tanh2);
          Value t2a    = b.create<arith::MulFOp>(nl, dgx, omT2);
          Value t2b    = b.create<arith::MulFOp>(nl, x, t2a);
          Value term2  = b.create<arith::MulFOp>(nl, c05, t2b);

          // grad_input = grad_out * (term1 + term2)
          Value dGelu  = b.create<arith::AddFOp>(nl, term1, term2);
          Value result = b.create<arith::MulFOp>(nl, go, dGelu);
          b.create<linalg::YieldOp>(nl, result);
        });

    rewriter.replaceOp(op, geluBwdGeneric.getResult(0));
    return success();
  }
};

struct NovaSoftmaxForwardPattern : public OpConversionPattern<mlir::nova::SoftmaxOp> {
  using OpConversionPattern<mlir::nova::SoftmaxOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::nova::SoftmaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // softmax(x) = exp(x - max(x, axis=dim)) / sum(exp(x - max(x, axis=dim)), axis=dim)
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType  = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = inputType.getElementType();
    int64_t rank    = inputType.getRank();
    MLIRContext *ctx = rewriter.getContext();
int64_t dim = op.getDimension().has_value()  
              ? static_cast<int64_t>(op.getDimension().value())   
              : static_cast<int64_t>(-1);
    if (dim > rank) dim = rank - 1;
    if (dim < 0) dim += rank;
    if (inputType.getElementType() != resultType.getElementType()) {
      input     = rewriter.create<mlir::tosa::CastOp>(loc, resultType, input);
      inputType = resultType;
      elemType  = resultType.getElementType();
    }

    // Stats shape: input with axis `dim` squeezed
    SmallVector<int64_t> statsShape;
    for (int64_t i = 0; i < rank; ++i)
      if (i != dim) statsShape.push_back(inputType.getDimSize(i));
    auto statsType = RankedTensorType::get(statsShape, elemType);

    // Reduction iterators: reduction on `dim`, parallel elsewhere
    SmallVector<utils::IteratorType> redIters;
    for (int64_t i = 0; i < rank; ++i)
      redIters.push_back(i == dim ? utils::IteratorType::reduction
                                  : utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> parIters(rank, utils::IteratorType::parallel);

    // Affine maps
    SmallVector<AffineExpr> inputExprs, statsExprs;
    for (int64_t i = 0; i < rank; ++i)
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    for (int64_t i = 0; i < rank; ++i)
      if (i != dim) statsExprs.push_back(rewriter.getAffineDimExpr(i));
    auto inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
    auto statsMap = AffineMap::get(rank, 0, statsExprs, ctx);

    // ---- Generic 1: maxVal = max(x, axis=dim) ----
    Value negInf   = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value maxEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
    Value maxInit  = rewriter.create<linalg::FillOp>(loc, negInf, maxEmpty).result();
    auto maxGeneric = rewriter.create<linalg::GenericOp>(
        loc, statsType, /*inputs=*/input, /*outputs=*/maxInit,
        SmallVector<AffineMap>{inputMap, statsMap}, redIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // maxVal = max(x, axis=dim)
          Value maxVal = b.create<arith::MaximumFOp>(nl, args[0], args[1]);
          b.create<linalg::YieldOp>(nl, maxVal);
        });
    Value maxVal = maxGeneric.getResult(0);

    // ---- Generic 2 (fused): expShifted[full] + sumExp[reduction] — single pass ----
    // maxVal is broadcast via statsMap across the reduction dim
    Value expEmpty = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(), elemType);
    Value sumEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);
    Value zero     = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
    Value sumInit  = rewriter.create<linalg::FillOp>(loc, zero, sumEmpty).result();

    auto fusedGeneric = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{inputType, statsType},
        /*inputs=*/ValueRange{input, maxVal},
        /*outputs=*/ValueRange{expEmpty, sumInit},
        SmallVector<AffineMap>{inputMap, statsMap, inputMap, statsMap},
        redIters,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // args[0]=x, args[1]=maxVal(broadcast), args[2]=exp_out, args[3]=sum_acc
          // shifted = x - maxVal, expShifted = exp(shifted)
          // sumExp = sum(expShifted, axis=dim)
          Value shifted = b.create<arith::SubFOp>(nl, args[0], args[1]);
          Value expV    = b.create<math::ExpOp>(nl, shifted);
          Value sumExp  = b.create<arith::AddFOp>(nl, expV, args[3]);
          b.create<linalg::YieldOp>(nl, ValueRange{expV, sumExp});
        });
    Value expShifted = fusedGeneric.getResult(0);
    Value sumExp     = fusedGeneric.getResult(1);

    // ---- Generic 3: out = expShifted / clamp(sumExp, eps) [all-parallel] ----
    Value epsVal   = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0e-7f));
    Value valOne   = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 1.0f));
    Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
    auto normGeneric = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        /*inputs=*/ValueRange{expShifted, sumExp}, /*outputs=*/outEmpty,
        SmallVector<AffineMap>{inputMap, statsMap, inputMap},
        parIters,   // all-parallel, not redIters
        [&](OpBuilder &b, Location nl, ValueRange args) {
          // args[0]=expShifted, args[1]=sumExp(broadcast via statsMap)
          // safeSum = max(sumExp, eps)
          // recip = 1/safeSum
          // result = expShifted * recip
          Value safeSum = b.create<arith::MaximumFOp>(nl, args[1], epsVal);
          Value recip  = b.create<arith::DivFOp>(nl, valOne, safeSum);
          Value result  = b.create<arith::MulFOp>(nl, args[0], recip);
          b.create<linalg::YieldOp>(nl, result);
        });

    rewriter.replaceOp(op, normGeneric.getResult(0));
    return success();
  }
};

// struct NovaSoftmaxBackwardPattern : public OpConversionPattern<mlir::nova::SoftmaxBackwardOp> {  
//   using OpConversionPattern<mlir::nova::SoftmaxBackwardOp>::OpConversionPattern;  
    
//   LogicalResult matchAndRewrite(mlir::nova::SoftmaxBackwardOp op, OpAdaptor adaptor,  
//                                 ConversionPatternRewriter &rewriter) const override {  
//     // d_softmax = grad_y - sum(grad_y, axis=dim) * output
//     Location loc = op.getLoc();  
//     Value grad_y = adaptor.getGradOut();  
//     Value output = adaptor.getOutput();  
      
//     auto gradType = cast<RankedTensorType>(grad_y.getType());  
//     auto elemType = gradType.getElementType();  
//     int64_t rank = gradType.getRank();  
//     MLIRContext *ctx = rewriter.getContext();  
  
//     // Resolve dimension (support negative indexing)  
//     int64_t dim = op.getDimension().has_value() ? static_cast<int64_t>(op.getDimension().value()) : -1;  
//     if (dim < 0) dim += rank;  
  
//     // Stats shape: input with axis `dim` squeezed  
//     SmallVector<int64_t> statsShape;  
//     for (int64_t i = 0; i < rank; ++i)  
//       if (i != dim) statsShape.push_back(gradType.getDimSize(i));  
//     auto statsType = RankedTensorType::get(statsShape, elemType);  
  
//     // Iterator types  
//     SmallVector<utils::IteratorType> redIters;  
//     for (int64_t i = 0; i < rank; ++i)  
//       redIters.push_back(i == dim ? utils::IteratorType::reduction  
//                                   : utils::IteratorType::parallel);  
//     SmallVector<utils::IteratorType> parIters(rank, utils::IteratorType::parallel);  
  
//     // Affine maps  
//     SmallVector<AffineExpr> inputExprs, statsExprs;  
//     for (int64_t i = 0; i < rank; ++i)  
//       inputExprs.push_back(rewriter.getAffineDimExpr(i));  
//     for (int64_t i = 0; i < rank; ++i)  
//       if (i != dim) statsExprs.push_back(rewriter.getAffineDimExpr(i));  
//     auto inputMap = AffineMap::get(rank, 0, inputExprs, ctx);  
//     auto statsMap = AffineMap::get(rank, 0, statsExprs, ctx);  
  
//     // ---- Single Generic: sumDiff = sum(grad_y - output, axis=dim) [reduction] ----  
//     Value sumEmpty = rewriter.create<tensor::EmptyOp>(loc, statsShape, elemType);  
//     Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));  
//     Value sumInit = rewriter.create<linalg::FillOp>(loc, zero, sumEmpty).result();  
            
//     auto sumDiffGeneric = rewriter.create<linalg::GenericOp>(  
//         loc, statsType,  
//         /*inputs=*/ValueRange{grad_y, output},  
//         /*outputs=*/sumInit,  
//         SmallVector<AffineMap>{inputMap, inputMap, statsMap},  
//         redIters,  
//         [&](OpBuilder &b, Location nl, ValueRange args) {  
//           // args[0]=grad_y, args[1]=output, args[2]=sum_acc  
//           // diffVal = grad_y - output
//           // sumDiff = sum(diff, axis=dim)
//           Value diffVal = b.create<arith::SubFOp>(nl, args[0], args[1]);  
//           Value sumDiff = b.create<arith::AddFOp>(nl, diffVal, args[2]);  
//           b.create<linalg::YieldOp>(nl, sumDiff);  
//         });  
//     Value sumDiff = sumDiffGeneric.getResult(0);  
  
//     // ---- Generic 3: grad_input = (grad_y - sumDiff) * output [all-parallel] ----  
//     Value gradEmpty = rewriter.create<tensor::EmptyOp>(loc, gradType.getShape(), elemType);  
//     auto gradInputGeneric = rewriter.create<linalg::GenericOp>(  
//         loc, gradType,  
//         /*inputs=*/ValueRange{grad_y, sumDiff, output},  
//         /*outputs=*/gradEmpty,  
//         SmallVector<AffineMap>{inputMap, statsMap, inputMap, inputMap},  
//         parIters,  
//         [&](OpBuilder &b, Location nl, ValueRange args) {  
//           // args[0]=grad_y, args[1]=sumDiff(broadcast), args[2]=output  
//           // grad_input = (grad_y - sumDiff) * output
//           Value adjusted = b.create<arith::SubFOp>(nl, args[0], args[1]);  
//           Value gradVal = b.create<arith::MulFOp>(nl, adjusted, args[2]);  
//           b.create<linalg::YieldOp>(nl, gradVal);  
//         });
  
//     rewriter.replaceOp(op, gradInputGeneric.getResult(0));  
//     return success();  
//   }  
// };

//===------------------------------------------------------------------------------------------===//
// Special operation lowerings: linear and layernorm
//===------------------------------------------------------------------------------------------===//

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

    //create nova::matmul and nova::add 
    auto matmulOperation = rewriter.create<mlir::nova::MatmulOp>(loc, input, weight).getResult();
    auto addOperation = rewriter.create<mlir::nova::AddOp>(loc, matmulOperation, bias);
    rewriter.replaceOp(op, addOperation);
    return success();
  }
};

struct NovaLinearBackwardPattern : public OpConversionPattern<mlir::nova::LinearBackwardOp> {
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
    SmallVector<int64_t> statsShape(xType.getShape().begin(), xType.getShape().end() - 1);
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

struct NovaLayerNormBackwardPattern : public OpConversionPattern<nova::LayerNormBackwardOp> {
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
struct NovaGatherOpLowering : public OpConversionPattern<nova::GatherOp> {
  using OpConversionPattern<nova::GatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value indices = adaptor.getIndices();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto indicesType = cast<RankedTensorType>(indices.getType());
    int64_t indicesRank = indicesType.getRank();
    int64_t resRank = resultType.getRank();
    int64_t axis = op.getAxis();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    if (axis < 0)
      axis += inputRank;
    int64_t batchDims = 0;
    int64_t maxBatchDims = std::min<int64_t>(axis, indicesRank);
    for (int64_t i = 0; i < maxBatchDims; ++i) {
      bool inputDyn = inputType.getDimSize(i) == ShapedType::kDynamic;
      bool indexDyn = indicesType.getDimSize(i) == ShapedType::kDynamic;
      if (!inputDyn && !indexDyn && inputType.getDimSize(i) != indicesType.getDimSize(i)) {
        break;
      }
      batchDims++;
    }
    // Map for indices: first batchDims map to loops [0, batchDims).
    // Remaining indices map to loops [axis, axis + indicesRank - batchDims).
    SmallVector<AffineExpr> indicesExprs;
    for (int i = 0; i < batchDims; ++i) {
      indicesExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int i = 0; i < indicesRank - batchDims; ++i) {
      indicesExprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }
    auto indicesMap =
        AffineMap::get(resRank, 0, indicesExprs, rewriter.getContext());
    // Constant map for input: makes the data dependency on `input` explicit
    // so tiling passes can see it (the actual read is via tensor.extract).
    SmallVector<AffineExpr> inputConstExprs;
    for (int64_t i = 0; i < inputRank; ++i)
      inputConstExprs.push_back(rewriter.getAffineConstantExpr(0));
    auto inputMap =
        AffineMap::get(resRank, 0, inputConstExprs, rewriter.getContext());
    // Map for output is identity
    auto outMap = rewriter.getMultiDimIdentityMap(resRank);
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(indicesMap);
    indexingMaps.push_back(inputMap);
    indexingMaps.push_back(outMap);
    SmallVector<utils::IteratorType> iteratorTypes(
        resRank, utils::IteratorType::parallel);
    Type indicesElemType = indicesType.getElementType();
    // Pass both indices and input as inputs; input is needed to make the
    // dependency visible to tiling passes (the block arg is unused).
    SmallVector<Value> inputs = {indices, input};
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, inputs, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location l, ValueRange args) {
          Value indexVal = args[0]; // from indices
          // args[1] is the dummy input element (unused)
          if (llvm::isa<FloatType>(indicesElemType)) {
            indexVal = b.create<arith::FPToSIOp>(l, b.getI32Type(), indexVal);
          } else if (auto intType =
                         llvm::dyn_cast<IntegerType>(indicesElemType)) {
            // Widen narrow integer indices (e.g. i16) to i32 before
            // IndexCastOp. MLIR's i16 is signless — IndexCastOp may
            // sign-extend, mapping values > 32767 to negative indices.
            // Use zero-extension (ExtUIOp) to preserve the unsigned
            // range 0..65535, which covers vocab sizes up to 65536.
            if (intType.getWidth() < 32) {
              indexVal =
                  b.create<arith::ExtUIOp>(l, b.getI32Type(), indexVal);
            }
          }
          Value classIdx =
              b.create<arith::IndexCastOp>(l, b.getIndexType(), indexVal);
          SmallVector<Value> extractionIndices;
          // Before axis
          for (int64_t i = 0; i < axis; ++i) {
            extractionIndices.push_back(b.create<linalg::IndexOp>(l, i));
          }
          // At axis
          extractionIndices.push_back(classIdx);
          // After axis
          for (int64_t i = axis + 1; i < inputRank; ++i) {
            extractionIndices.push_back(
                b.create<linalg::IndexOp>(l, i + indicesRank - batchDims - 1));
          }
          Value extracted =
              b.create<tensor::ExtractOp>(l, input, extractionIndices);
          b.create<linalg::YieldOp>(l, extracted);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
//===-----------------------------------------------------------------------------------------===//
// Exponents and logarithms lowering patterns: exp2, log2, log10
//===-----------------------------------------------------------------------------------------===//

struct NovaExp2LoweringPattern : public OpConversionPattern<nova::Exp2Op> {
  using OpConversionPattern<nova::Exp2Op>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::Exp2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value exp2= b.create<math::Exp2Op>(nl, args[0]);
          b.create<linalg::YieldOp>(nl,  exp2);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaLog2LoweringPattern : public OpConversionPattern<nova::Log2Op> {
  using OpConversionPattern<nova::Log2Op>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::Log2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value log2 = b.create<math::Log2Op>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, log2);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaLog10LoweringPattern : public OpConversionPattern<nova::Log10Op> {
  using OpConversionPattern<nova::Log10Op>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::Log10Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
            Value log10 = b.create<math::Log10Op>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, log10);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===-----------------------------------------------------------------------------------------===//
// Trigonometric functions lowering patterns: sin, cos, tan, asin, acos, atan, 
// sinh, cosh, tanh, asinh, acosh, atanh
//===-----------------------------------------------------------------------------------------===//

struct NovaSinLoweringPattern : public OpConversionPattern<nova::SinOp> {
  using OpConversionPattern<nova::SinOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::SinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value sin = b.create<math::SinOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, sin);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaCosLoweringPattern : public OpConversionPattern<nova::CosOp> {
  using OpConversionPattern<nova::CosOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::CosOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value cos = b.create<math::CosOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, cos);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaTanLoweringPattern : public OpConversionPattern<nova::TanOp> {
  using OpConversionPattern<nova::TanOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::TanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value tan = b.create<math::TanOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, tan);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAsinLoweringPattern : public OpConversionPattern<nova::AsinOp> {
  using OpConversionPattern<nova::AsinOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AsinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value asin = b.create<math::AsinOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, asin);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAcosLoweringPattern : public OpConversionPattern<nova::AcosOp> {
  using OpConversionPattern<nova::AcosOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AcosOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value acos = b.create<math::AcosOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, acos);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAtanLoweringPattern : public OpConversionPattern<nova::AtanOp> {
  using OpConversionPattern<nova::AtanOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AtanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value atan = b.create<math::AtanOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, atan);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaSinhLoweringPattern : public OpConversionPattern<nova::SinhOp> {
  using OpConversionPattern<nova::SinhOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::SinhOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value sinh = b.create<math::SinhOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, sinh);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaCoshLoweringPattern : public OpConversionPattern<nova::CoshOp> {
  using OpConversionPattern<nova::CoshOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::CoshOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value cosh = b.create<math::CoshOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, cosh);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAsinhLoweringPattern : public OpConversionPattern<nova::AsinhOp> {
  using OpConversionPattern<nova::AsinhOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AsinhOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value asinh = b.create<math::AsinhOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, asinh);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAcoshLoweringPattern : public OpConversionPattern<nova::AcoshOp> {
  using OpConversionPattern<nova::AcoshOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AcoshOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value acosh = b.create<math::AcoshOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl, acosh);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAtanhLoweringPattern : public OpConversionPattern<nova::AtanhOp> {
  using OpConversionPattern<nova::AtanhOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AtanhOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value atanh = b.create<math::AtanhOp>(nl, args[0]);
          b.create<linalg::YieldOp>(nl,atanh);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===-----------------------------------------------------------------------------------------===//
// Logical operations lowering patterns: not, and, or
//===-----------------------------------------------------------------------------------------===//
struct NovaNotLoweringPattern : public OpConversionPattern<nova::NotOp> {
  using OpConversionPattern<nova::NotOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::NotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          Value x = args[0];
          Value cmp;
          if (isa<IntegerType>(x.getType())) {
            Value zero = b.create<arith::ConstantIntOp>(nl, 0,
                cast<IntegerType>(x.getType()).getWidth());
            cmp = b.create<arith::CmpIOp>(nl, arith::CmpIPredicate::eq, x, zero);
          } else {
            APFloat zeroVal(cast<FloatType>(x.getType()).getFloatSemantics(), 0);
            Value zero = b.create<arith::ConstantFloatOp>(nl,
                cast<FloatType>(x.getType()), zeroVal);
            cmp = b.create<arith::CmpFOp>(nl, arith::CmpFPredicate::OEQ, x, zero);
          }
          Value notX = b.create<arith::XOrIOp>(nl, cmp, b.create<arith::ConstantIntOp>(nl, 1, 1));
          b.create<linalg::YieldOp>(nl, notX);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaAndLoweringPattern : public OpConversionPattern<nova::AndOp> {
  using OpConversionPattern<nova::AndOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::AndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value lhsIn = adaptor.getOperands()[0];
    Value rhsIn = adaptor.getOperands()[1];
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ValueRange{lhsIn, rhsIn}, out,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          auto toBool = [&](Value v) -> Value {
            if (isa<FloatType>(v.getType())) {
              APFloat z(cast<FloatType>(v.getType()).getFloatSemantics(), 0);
              Value zero = b.create<arith::ConstantFloatOp>(nl, cast<FloatType>(v.getType()), z);
              return b.create<arith::CmpFOp>(nl, arith::CmpFPredicate::UNE, v, zero);
            }
            return v;
          };
          Value andVal = b.create<arith::AndIOp>(nl, toBool(args[0]), toBool(args[1]));
          b.create<linalg::YieldOp>(nl,andVal);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaOrLoweringPattern : public OpConversionPattern<nova::OrOp> {
  using OpConversionPattern<nova::OrOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::OrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value lhsIn = adaptor.getOperands()[0];
    Value rhsIn = adaptor.getOperands()[1];
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ValueRange{lhsIn, rhsIn}, out,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          auto toBool = [&](Value v) -> Value {
            if (isa<FloatType>(v.getType())) {
              APFloat z(cast<FloatType>(v.getType()).getFloatSemantics(), 0);
              Value zero = b.create<arith::ConstantFloatOp>(nl, cast<FloatType>(v.getType()), z);
              return b.create<arith::CmpFOp>(nl, arith::CmpFPredicate::UNE, v, zero);
            }
            return v;
          };
          Value orVal = b.create<arith::OrIOp>(nl, toBool(args[0]), toBool(args[1]));
          b.create<linalg::YieldOp>(nl,orVal);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaXorLoweringPattern : public OpConversionPattern<nova::XorOp> {
  using OpConversionPattern<nova::XorOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::XorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value lhsIn = adaptor.getOperands()[0];
    Value rhsIn = adaptor.getOperands()[1];
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ValueRange{lhsIn, rhsIn}, out,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap}, parallel,
        [](OpBuilder &b, Location nl, ValueRange args) {
          auto toBool = [&](Value v) -> Value {
            if (isa<FloatType>(v.getType())) {
              APFloat z(cast<FloatType>(v.getType()).getFloatSemantics(), 0);
              Value zero = b.create<arith::ConstantFloatOp>(nl, cast<FloatType>(v.getType()), z);
              return b.create<arith::CmpFOp>(nl, arith::CmpFPredicate::UNE, v, zero);
            }
            return v;
          };
          Value xorVal = b.create<arith::XOrIOp>(nl, toBool(args[0]), toBool(args[1]));
          b.create<linalg::YieldOp>(nl,xorVal);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaSignLoweringPattern : public OpConversionPattern<nova::SignOp> {
  using OpConversionPattern<nova::SignOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::SignOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, adaptor.getOperands()[0], out,
        SmallVector<AffineMap>{identityMap, identityMap}, parallel,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value x = args[0];
          Value res;
          if (isa<FloatType>(x.getType())) {
            Value zero  = b.create<arith::ConstantOp>(nl, elemType, b.getFloatAttr(elemType, 0.0));
            Value isPos = b.create<arith::CmpFOp>(nl, arith::CmpFPredicate::OGT, x, zero);
            Value isNeg = b.create<arith::CmpFOp>(nl, arith::CmpFPredicate::OLT, x, zero);
            Value sPos  = b.create<arith::SIToFPOp>(nl, elemType, isPos);
            Value sNeg  = b.create<arith::SIToFPOp>(nl, elemType, isNeg);
            res = b.create<arith::SubFOp>(nl, sPos, sNeg);
          } else {
            Value zero  = b.create<arith::ConstantOp>(nl, x.getType(), b.getIntegerAttr(x.getType(), 0));
            Value isPos = b.create<arith::CmpIOp>(nl, arith::CmpIPredicate::sgt, x, zero);
            Value isNeg = b.create<arith::CmpIOp>(nl, arith::CmpIPredicate::slt, x, zero);
            Value sPos  = b.create<arith::SIToFPOp>(nl, elemType, isPos);
            Value sNeg  = b.create<arith::SIToFPOp>(nl, elemType, isNeg);
            res = b.create<arith::SubFOp>(nl, sPos, sNeg);
          }
          b.create<linalg::YieldOp>(nl, res);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NovaCompareLoweringPattern : public OpConversionPattern<nova::CompareOp> {
  using OpConversionPattern<nova::CompareOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(nova::CompareOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType   = resultType.getElementType();
    int64_t rank    = resultType.getRank();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    nova::ComparisonType cmpKind = op.getKind();
    Value out = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), elemType);
    Value lhsIn = adaptor.getOperands()[0];
    Value rhsIn = adaptor.getOperands()[1];
    Value result = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ValueRange{lhsIn, rhsIn}, out,
        SmallVector<AffineMap>{identityMap, identityMap, identityMap}, parallel,
        [&](OpBuilder &b, Location nl, ValueRange args) {
          Value lhs = args[0], rhs = args[1];
          Value cmp;
          if (isa<FloatType>(lhs.getType())) {
            arith::CmpFPredicate pred;
            switch (cmpKind) {
              case nova::ComparisonType::EQ: pred = arith::CmpFPredicate::OEQ; break;
              case nova::ComparisonType::NEQ: pred = arith::CmpFPredicate::UNE; break;
              case nova::ComparisonType::LT: pred = arith::CmpFPredicate::OLT; break;
              case nova::ComparisonType::LE: pred = arith::CmpFPredicate::OLE; break;
              case nova::ComparisonType::GT: pred = arith::CmpFPredicate::OGT; break;
              case nova::ComparisonType::GE: pred = arith::CmpFPredicate::OGE; break;
            }
            cmp = b.create<arith::CmpFOp>(nl, pred, lhs, rhs);
          } else {
            arith::CmpIPredicate pred;
            switch (cmpKind) {
              case nova::ComparisonType::EQ: pred = arith::CmpIPredicate::eq;  break;
              case nova::ComparisonType::NEQ: pred = arith::CmpIPredicate::ne;  break;
              case nova::ComparisonType::LT: pred = arith::CmpIPredicate::slt; break;
              case nova::ComparisonType::LE: pred = arith::CmpIPredicate::sle; break;
              case nova::ComparisonType::GT: pred = arith::CmpIPredicate::sgt; break;
              case nova::ComparisonType::GE: pred = arith::CmpIPredicate::sge; break;
            }
            cmp = b.create<arith::CmpIOp>(nl, pred, lhs, rhs);
          }
          b.create<linalg::YieldOp>(nl, cmp);
        }).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};





// pass definition
namespace {
struct NovaToLinalgGenericLoweringPass : public PassWrapper<NovaToLinalgGenericLoweringPass, OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToLinalgGenericLoweringPass)

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
    ConversionTarget target(getContext());

    target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           complex::ComplexDialect, arith::ArithDialect>();


    target.addIllegalOp<nova::MaeOp>();

    target.addIllegalOp<nova::MseOp>();

    target.addIllegalOp<nova::BceOp>();
    
    target.addIllegalOp<nova::CceOp>();

    target.addIllegalOp<nova::SceOp>();
    target.addIllegalOp<nova::SceBackwardOp>();

    target.addIllegalOp<nova::SigmoidOp>();


    target.addIllegalOp<nova::GeluOp>();
    target.addIllegalOp<nova::GeluBackwardOp>();

    target.addIllegalOp<nova::SoftmaxOp>();

    target.addIllegalOp<nova::LinearOp>();
    target.addIllegalOp<nova::LinearBackwardOp>();
    target.addIllegalOp<nova::GatherOp>();

    target.addIllegalOp<nova::LayerNormOp>();
    target.addIllegalOp<nova::LayerNormBackwardOp>();

    target.addIllegalOp<nova::Exp2Op>();
    target.addIllegalOp<nova::Log2Op>();
    target.addIllegalOp<nova::Log10Op>();
    target.addIllegalOp<nova::CastOp>();

    target.addIllegalOp<nova::SinOp>();
    target.addIllegalOp<nova::CosOp>();
    target.addIllegalOp<nova::TanOp>();
    target.addIllegalOp<nova::AsinOp>();
    target.addIllegalOp<nova::AcosOp>();
    target.addIllegalOp<nova::AtanOp>();
    target.addIllegalOp<nova::SinhOp>();
    target.addIllegalOp<nova::CoshOp>();
    target.addIllegalOp<nova::AsinhOp>();
    target.addIllegalOp<nova::AcoshOp>();
    target.addIllegalOp<nova::AtanhOp>();

    target.addIllegalOp<nova::NotOp>();
    target.addIllegalOp<nova::AndOp>();
    target.addIllegalOp<nova::OrOp>();
    target.addIllegalOp<nova::XorOp>();
    target.addIllegalOp<nova::SignOp>();
    target.addIllegalOp<nova::CompareOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    mlir::RewritePatternSet patterns(&getContext());
    mlir::nova::populateNovaToLinalgGenericConversionPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void populateNovaToLinalgGenericConversionPatterns(RewritePatternSet &patterns) {
  patterns.add< 
      NovaMaeForwardLowering, 
      NovaMseForwardLowering, 
      NovaBceForwardLowering, 
      NovaCceForwardLowering, 
      NovaSceForwardLowering, NovaSceBackwardLowering,
      NovaSigmoidForwardLowering,
      NovaGatherOpLowering,
      NovaGeluForwardPattern, NovaGeluBackwardPattern, 
      NovaSoftmaxForwardPattern, 
      NovaLinearOpLowering, NovaLinearBackwardPattern,
      NovaLayerNormPattern, NovaLayerNormBackwardPattern,
      NovaExp2LoweringPattern, NovaLog2LoweringPattern, NovaLog10LoweringPattern,
      NovaSinLoweringPattern, NovaCosLoweringPattern, NovaTanLoweringPattern, 
      NovaAsinLoweringPattern, NovaAcosLoweringPattern, NovaAtanLoweringPattern, 
      NovaSinhLoweringPattern, NovaCoshLoweringPattern, NovaAsinhLoweringPattern, 
      NovaAcoshLoweringPattern, NovaAtanhLoweringPattern,
      NovaNotLoweringPattern , NovaAndLoweringPattern, NovaOrLoweringPattern, 
      NovaXorLoweringPattern, NovaSignLoweringPattern, NovaCompareLoweringPattern,NovaCastOpLowering
      >(patterns.getContext());
}

// creating a pointer for this pass
std::unique_ptr<Pass> createNovaToLinalgGenericLoweringPass() {
  return std::make_unique<NovaToLinalgGenericLoweringPass>();
}

// Register the pass
void registerNovaToLinalgGenericLoweringPass() {
  PassRegistration<NovaToLinalgGenericLoweringPass>();
}   

} // namespace nova
} // namespace mlir
