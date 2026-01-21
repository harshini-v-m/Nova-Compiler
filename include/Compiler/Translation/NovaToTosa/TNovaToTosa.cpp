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

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
// base function
namespace mlir {
namespace nova {
struct NovaOpTosaOp {
  static Value castElementType(OpBuilder *builder, Location loc, Value val, Type targetElemTy) {
    auto rankedTy = dyn_cast<RankedTensorType>(val.getType());
    if (!rankedTy || rankedTy.getElementType() == targetElemTy) return val;
    auto newTy = RankedTensorType::get(rankedTy.getShape(), targetElemTy, rankedTy.getEncoding());
    return builder->create<tosa::CastOp>(loc, newTy, val);
  }

  static Value matchRank(OpBuilder *builder, Location loc, Value val, int64_t targetRank) {
    auto rankedTy = cast<RankedTensorType>(val.getType());
    int64_t currentRank = rankedTy.getRank();
    if (currentRank == targetRank) return val;

    SmallVector<int64_t> newShape;
    for (int64_t i = 0; i < targetRank - currentRank; ++i) {
      newShape.push_back(1);
    }
    for (int64_t dim : rankedTy.getShape()) {
      newShape.push_back(dim);
    }

    auto newTy = RankedTensorType::get(newShape, rankedTy.getElementType(), rankedTy.getEncoding());
    
    // TOSA Reshape requires a shape constant
    auto shapeType = RankedTensorType::get({targetRank}, builder->getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
    Value shapeConst = builder->create<tosa::ConstShapeOp>(
        loc, tosa::shapeType::get(builder->getContext(), targetRank), shapeAttr);

    return builder->create<tosa::ReshapeOp>(loc, newTy, val, shapeConst);
  }

  static Value rmappingtosa(nova::AddOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {
    auto restensor = cast<RankedTensorType>(resultType);
    auto targetElemTy = restensor.getElementType();
    int64_t targetRank = restensor.getRank();
    auto v = castElementType(builder, op.getLoc(), input[0], targetElemTy);
    auto w = castElementType(builder, op.getLoc(), input[1], targetElemTy);
    v = matchRank(builder, op.getLoc(), v, targetRank);
    w = matchRank(builder, op.getLoc(), w, targetRank);
    return builder->create<tosa::AddOp>(op.getLoc(), resultType, v, w);
  }

  static Value rmappingtosa(nova::SubOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {
    auto restensor = cast<RankedTensorType>(resultType);
    auto targetElemTy = restensor.getElementType();
    int64_t targetRank = restensor.getRank();
    auto v = castElementType(builder, op.getLoc(), input[0], targetElemTy);
    auto w = castElementType(builder, op.getLoc(), input[1], targetElemTy);
    v = matchRank(builder, op.getLoc(), v, targetRank);
    w = matchRank(builder, op.getLoc(), w, targetRank);
    return builder->create<tosa::SubOp>(op.getLoc(), resultType, v, w);
  }
  // static Value rmappingtosa(nova::MulOp op, Type resultType, ValueRange input,
  //                           OpBuilder *builder) {
  //   auto restensor = cast<mlir::RankedTensorType>(resultType);
  //   auto targetElemTy = restensor.getElementType();
  //   int64_t targetRank = restensor.getRank();
  //   auto v = castElementType(builder, op.getLoc(), input[0], targetElemTy);
  //   auto w = castElementType(builder, op.getLoc(), input[1], targetElemTy);
  //   v = matchRank(builder, op.getLoc(), v, targetRank);
  //   w = matchRank(builder, op.getLoc(), w, targetRank);
    
  //   auto shift = builder->create<mlir::arith::ConstantOp>(
  //       op.getLoc(),
  //       DenseElementsAttr::get(RankedTensorType::get({1}, builder->getI8Type()),
  //                              builder->getI8IntegerAttr(0)));
  //   return builder->create<tosa::MulOp>(op.getLoc(), resultType, v, w, shift);
  // }
  static Value rmappingtosa(nova::PowOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {
    auto restensor = cast<mlir::RankedTensorType>(resultType);
    auto targetElemTy = restensor.getElementType();
    int64_t targetRank = restensor.getRank();
    auto v = castElementType(builder, op.getLoc(), input[0], targetElemTy);
    auto w = castElementType(builder, op.getLoc(), input[1], targetElemTy);
    v = matchRank(builder, op.getLoc(), v, targetRank);
    w = matchRank(builder, op.getLoc(), w, targetRank);
    return builder->create<tosa::PowOp>(op.getLoc(), resultType, v, w);
  }




  template <typename OpTy>
  static Value rmaptop(OpTy op, Type resultType, ValueRange input,
                       OpBuilder *builder) {
    return rmappingtosa(op, resultType, input, builder);
  }

private:
  template <typename OpTy>
  static Value rmappingtosa(OpTy op, Type resultType, ValueRange input,
                            OpBuilder *builder) {
    return nullptr;
  }
};

// creating a template
template <typename NovaTopTy>
class NovaToTosaLoweringTemplater : public OpConversionPattern<NovaTopTy> {
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
    Value result = NovaOpTosaOp::rmaptop(op, resultType, operands, &rewriter);
    if (!result)
      return rewriter.notifyMatchFailure(op, "failed to map to TOSA operation");

    rewriter.replaceOp(op, result);
    return success();
  }
};
void populateNovaToTosaTemplatePatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<NovaToTosaLoweringTemplater<nova::AddOp>,
               NovaToTosaLoweringTemplater<nova::SubOp>,
              //  NovaToTosaLoweringTemplater<nova::MulOp>,
               NovaToTosaLoweringTemplater<nova::PowOp>>(
      patterns.getContext());
}
} // namespace nova
} // namespace mlir