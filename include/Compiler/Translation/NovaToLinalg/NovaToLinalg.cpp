#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

using namespace mlir;
using namespace mlir::nova;
using namespace mlir::bufferization;

namespace mlir {
namespace nova {
  
//HELPER FUNCTIONS
inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}
Value broadcastToShape(OpBuilder &b, Location loc,
                               Value input, ArrayRef<int64_t> targetShape) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return input;

  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank  = static_cast<int64_t>(inputShape.size());
  int64_t targetRank = static_cast<int64_t>(targetShape.size());

  if (inputRank == targetRank &&
      std::equal(inputShape.begin(), inputShape.end(), targetShape.begin()))
    return input;

  int64_t rankDiff = targetRank - inputRank;
  SmallVector<AffineExpr> inputExprs;

  if (inputRank < targetRank) {
    // Rank mismatch: align input dims to the trailing dims of the target.
    // e.g. input rank=1, target rank=3 → input[0] maps to d2
    for (int64_t i = 0; i < inputRank; ++i)
      inputExprs.push_back(b.getAffineDimExpr(rankDiff + i));
  } else {
    // Same rank, shape mismatch: size-1 dims use constant-0 (broadcast),
    // all other dims pass through as identity.
    // e.g. tensor<1x10xf32> → tensor<16x10xf32>: d0 → 0, d1 → d1
    for (int64_t i = 0; i < targetRank; ++i) {
      if (inputShape[i] == 1 && targetShape[i] != 1)
        inputExprs.push_back(b.getAffineConstantExpr(0));
      else
        inputExprs.push_back(b.getAffineDimExpr(i));
    }
  }

  auto inputMap  = AffineMap::get(targetRank, 0, inputExprs, b.getContext());
  auto outputMap = AffineMap::getMultiDimIdentityMap(targetRank, b.getContext());

  Value initTensor = b.create<tensor::EmptyOp>(
      loc, targetShape, inputType.getElementType());

  SmallVector<utils::IteratorType> iters(targetRank,
                                         utils::IteratorType::parallel);

  return b.create<linalg::GenericOp>(
             loc, TypeRange{initTensor.getType()},
             input, initTensor,
             SmallVector<AffineMap>{inputMap, outputMap}, iters,
             [](OpBuilder &b, Location loc, ValueRange args) {
               b.create<linalg::YieldOp>(loc, args[0]);
             })
           .getResult(0);
}

// Conversion Patterns
//Arithmetic ops
//ADD OP - > LINALG.GENERIC
struct NovaOpToStdScalarOp {
  template <typename OpTy>
  // kind of main function
  static Value mapOp(OpTy op, Type resultType, ArrayRef<Value> args,
                     OpBuilder *builder) {
    return mapOpImpl(op, resultType, args, builder);
  }

  // default function to return null ptr if the operation didn't match
private:
  template <typename OpTy>
  static Value mapOpImpl(OpTy op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    return nullptr;
  }
  // add operation
  static Value mapOpImpl(nova::AddOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::AddFOp>(op.getLoc(), args[0], args[1]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::AddIOp>(op.getLoc(), args[0], args[1]);
    return nullptr;
  }};

//SUB OP - > LINALG.GENERIC
template <typename NovaOpTy>
class NovaToLinalgElementwiseConverter : public OpConversionPattern<NovaOpTy> {
public:
  using OpConversionPattern<NovaOpTy>::OpConversionPattern; // creates a
                                                            // constructor
  using OpAdaptor = typename NovaOpTy::Adaptor; // for getting data type
                                                // dynamically using adaptor

  LogicalResult
  matchAndRewrite(NovaOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    if (operands.empty())
      return rewriter.notifyMatchFailure(
          op, "expected operands for linalg lowering operations");
    // checking if operand is ranked tensortype
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    // each element type
    auto resultDataType = resultType.getElementType();
    
    // Check for in_place attribute for DPS (Destination-Passing Style)
    bool isInPlace = op->template hasAttrOfType<BoolAttr>("in_place") &&
                     op->template getAttrOfType<BoolAttr>("in_place").getValue();

    SmallVector<Value> insOperands;
    Value out;
    if (isInPlace && operands.size() == 2) {
      insOperands = {operands[1]}; // rhs is the explicit input (the new gradient values)
      out = operands[0];           // lhs is the destination (the persistent parameter buffer)
    } else {
      insOperands = operands;
      out = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), resultType.getShape(), resultDataType);
    }

    // Prepare affine maps
    int64_t rank = resultType.getRank();
    SmallVector<AffineMap> maps;
    for (Value v : insOperands) {
      auto vType = cast<RankedTensorType>(v.getType());
      auto vShape = vType.getShape();
      auto vRank = vType.getRank();
      SmallVector<AffineExpr> exprs;
      // Standard broadcasting: align right
      for (int64_t i = 0; i < vRank; ++i) {
        if (vShape[i] == 1) {
          exprs.push_back(rewriter.getAffineConstantExpr(0));
        } else {
          exprs.push_back(rewriter.getAffineDimExpr(i + (rank - vRank)));
        }
      }
      maps.push_back(AffineMap::get(rank, 0, exprs, rewriter.getContext()));
    }
    // Output map
    maps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Create Linalg generic
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), out.getType(), insOperands, out, maps,
        getNParallelLoopsAttrs(rank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Type elemType = getElementTypeOrSelf(out);
          SmallVector<Value> argVec(args.begin(), args.end());
          // call our custom lowering functions
          Value inner = NovaOpToStdScalarOp::mapOp(op, elemType, argVec, &b);
          if (!inner)
            return; // op failed to map
          b.create<linalg::YieldOp>(loc, inner);
        });

    // Propagate in_place intent so OneShotBufferize can alias the output
    // buffer without inserting a defensive copy when lhs has a single use.
    if (isInPlace)
      linalgOp->setAttr("nova.in_place", rewriter.getUnitAttr());

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};
// struct NovaAddOpLowering : public OpConversionPattern<nova::AddOp> {
//   using OpConversionPattern<nova::AddOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(nova::AddOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
//     if (!resultType)
//       return failure();

//     Location loc = op.getLoc();
//     Value input = adaptor.getLhs();
//     Value other = adaptor.getRhs();
//     input = broadcastToShape(rewriter, loc, input, resultType.getShape());
//     other = broadcastToShape(rewriter, loc, other, resultType.getShape());

//     // Create empty output tensor
//     Value emptyTensor = rewriter.create<tensor::EmptyOp>(
//         loc, resultType.getShape(), resultType.getElementType());

//     int64_t rank = resultType.getRank();
//     AffineMap identityMap =
//         AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
//     SmallVector<AffineMap> indexingMaps = {identityMap, identityMap,
//                                            identityMap};
//     SmallVector<utils::IteratorType> iteratorTypes(rank,
//                                                    utils::IteratorType::parallel);

//     auto genericOp = rewriter.create<linalg::GenericOp>(
//         loc, TypeRange{resultType},
//         ValueRange{input, other}, ValueRange{emptyTensor},
//         indexingMaps, iteratorTypes,
//         [](OpBuilder &b, Location loc, ValueRange args) {
//           //check if int lower to arith addiop
//           if (isa<IntegerType>(args[0].getType())) {
//             Value result =
//                 b.create<arith::AddIOp>(loc, args[0], args[1]);
//             b.create<linalg::YieldOp>(loc, result);
//             return;
//           }
//           Value result =
//               b.create<arith::AddFOp>(loc, args[0], args[1]);
//           b.create<linalg::YieldOp>(loc, result);
//         });

//     rewriter.replaceOp(op, genericOp.getResults());
//     return success();
//   }
// };

//SUB OP - > LINALG.SUB
struct NovaSubOpLowering : public OpConversionPattern<nova::SubOp> {
  using OpConversionPattern<nova::SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value other = adaptor.getRhs();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    input = broadcastToShape(rewriter, loc, input, resultType.getShape());
    other = broadcastToShape(rewriter, loc, other, resultType.getShape());   
    // Identity maps for linalg.sub
    auto ResOp = rewriter.create<linalg::SubOp>(
        loc, TypeRange{resultType},ValueRange{input, other},ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//MUL OP - > LINALG.MUL
struct NovaMulOpLowering : public OpConversionPattern<nova::MulOp> {
  using OpConversionPattern<nova::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value other = adaptor.getRhs();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
        
    // Identity maps for linalg.mul
    auto ResOp = rewriter.create<linalg::MulOp>(
        loc, TypeRange{resultType},ValueRange{input, other},ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//DIV OP - > LINALG.DIV
struct NovaDivOpLowering : public OpConversionPattern<nova::DivOp> {
  using OpConversionPattern<nova::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value other = adaptor.getRhs();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
        
    // Identity maps for linalg.div
    auto ResOp = rewriter.create<linalg::DivOp>(
        loc, TypeRange{resultType},ValueRange{input, other},ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//MIN OP - > LINALG.MIN
struct NovaMinOpLowering : public OpConversionPattern<nova::MinOp> {
  using OpConversionPattern<nova::MinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value other = adaptor.getRhs();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
        
    // Identity maps for linalg.min
    auto ResOp = rewriter.create<linalg::MinOp>(
        loc, TypeRange{resultType},ValueRange{input, other},ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//MAX OP - > LINALG.MAX
struct NovaMaxOpLowering : public OpConversionPattern<nova::MaxOp> {
  using OpConversionPattern<nova::MaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value other = adaptor.getRhs();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
        
    // Identity maps for linalg.max
    auto ResOp = rewriter.create<linalg::MaxOp>(
        loc, TypeRange{resultType},ValueRange{input, other},ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//POW OP - > LINALG.POW
struct NovaPowOpLowering : public OpConversionPattern<nova::PowOp> {
  using OpConversionPattern<nova::PowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value other = adaptor.getRhs();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    Type elemType = resultType.getElementType();

    if (isa<FloatType>(elemType)) {
      // Float path: use linalg.powf named op
      auto ResOp = rewriter.create<linalg::PowFOp>(
          loc, TypeRange{resultType}, ValueRange{input, other},
          ValueRange{emptyTensor});
      rewriter.replaceOp(op, ResOp.getResults());
    } else {
      // Integer path: lower to linalg.generic with math.ipowi
      int64_t rank = resultType.getRank();
      SmallVector<AffineMap> indexingMaps(
          3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
      SmallVector<utils::IteratorType> iteratorTypes(
          rank, utils::IteratorType::parallel);

      auto ResOp = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{resultType}, ValueRange{input, other},
          ValueRange{emptyTensor}, indexingMaps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value result = b.create<math::IPowIOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, result);
          });
      rewriter.replaceOp(op, ResOp.getResults());
    }
    return success();
  }
};

//ABS OP - > LINALG.ABS
struct NovaAbsOpLowering : public OpConversionPattern<nova::AbsOp> {
  using OpConversionPattern<nova::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Identity maps for linalg.abs
    auto ResOp = rewriter.create<linalg::AbsOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//EXP OP - > LINALG.EXP
struct NovaExpOpLowering : public OpConversionPattern<nova::ExpOp> {
  using OpConversionPattern<nova::ExpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Identity maps for linalg.abs
    auto ResOp = rewriter.create<linalg::ExpOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//LOG OP - > LINALG.LOG
struct NovaLogOpLowering : public OpConversionPattern<nova::LogOp> {
  using OpConversionPattern<nova::LogOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Identity maps for linalg.abs
    auto ResOp = rewriter.create<linalg::LogOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//NEG OP - > LINALG.NEGF (ONLY FOR FLOATS, INTS LOWERED TO MUL WITH -1)
struct NovaNegOpLowering : public OpConversionPattern<nova::NegOp> {
  using OpConversionPattern<nova::NegOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Type elemType = resultType.getElementType();
    if (isa<FloatType>(elemType)) {
      // Float path: use linalg.negf named op
      auto ResOp = rewriter.create<linalg::NegFOp>(
          loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
      rewriter.replaceOp(op, ResOp.getResults());
    } else {
      // Integer path: lower to linalg.generic with multiplication by -1
      auto negOne = rewriter.create<arith::ConstantOp>(
          loc, resultType.getElementType(),
          rewriter.getIntegerAttr(resultType.getElementType(), -1));
      auto ResOp = rewriter.create<linalg::MulOp>(
          loc, TypeRange{resultType}, ValueRange{input, negOne},
          ValueRange{emptyTensor});
      rewriter.replaceOp(op, ResOp.getResults());
    }
    return success();
  }
};

//RECIPROCAL OP - > LINALG.RECRIPROCAL 
struct NovaReciprocalOpLowering : public OpConversionPattern<nova::ReciprocalOp> {
  using OpConversionPattern<nova::ReciprocalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ReciprocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Identity maps for linalg.reciprocal
    auto ResOp = rewriter.create<linalg::ReciprocalOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//SQRT OP - > LINALG.SQRT
struct NovaSqrtOpLowering : public OpConversionPattern<nova::SqrtOp> {
  using OpConversionPattern<nova::SqrtOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    // Identity maps for linalg.sqrt
    auto ResOp = rewriter.create<linalg::SqrtOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//SQUARE OP - > LINALG.SQUARE
struct NovaSquareOpLowering : public OpConversionPattern<nova::SquareOp> {
  using OpConversionPattern<nova::SquareOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::SquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    // Identity maps for linalg.square
    auto ResOp = rewriter.create<linalg::SquareOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//TANH OP - > LINALG.TANH
struct NovaTanhOpLowering : public OpConversionPattern<nova::TanhOp> {
  using OpConversionPattern<nova::TanhOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::TanhOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    // Identity maps for linalg.tanh
    auto ResOp = rewriter.create<linalg::TanhOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//RSQRT OP - > LINALG.RSQRT
struct NovaRsqrtOpLowering : public OpConversionPattern<nova::RsqrtOp> {
  using OpConversionPattern<nova::RsqrtOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    // Identity maps for linalg.rsqrt
    auto ResOp = rewriter.create<linalg::RsqrtOp>(
        loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyTensor});
    rewriter.replaceOp(op, ResOp.getResults());
    return success();
  }
};

//BROADCAST_IN_DIM OP - > LINALG.BROADCAST
struct NovaBroadcastInDimOpLowering
    : public OpConversionPattern<nova::BroadcastInDimOp> {
  using OpConversionPattern<nova::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperand();

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked tensor result type");
    }
    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked tensor input type");
    }
    auto loc = op.getLoc();
    auto dimsAttr = op.getBroadcastDimensions();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Build affine map for input
    SmallVector<AffineExpr> inputExprs;
    for (auto [inputIdx, dimAttr] :
         llvm::enumerate(dimsAttr.getAsValueRange<IntegerAttr>())) {
      int64_t outputDim = dimAttr.getSExtValue();
      int64_t inputSize = inputType.getDimSize(inputIdx);
      int64_t outputSize = resultType.getDimSize(outputDim);

      // If broadcasting dimension (1 -> N), use constant 0
      if (inputSize == 1 && outputSize != 1) {
        inputExprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        inputExprs.push_back(rewriter.getAffineDimExpr(outputDim));
      }
    }

    // Build affine map for output (identity)
    SmallVector<AffineExpr> outputExprs;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }

    auto inputMap = AffineMap::get(resultType.getRank(), 0, inputExprs,
                                   rewriter.getContext());
    auto outputMap = AffineMap::get(resultType.getRank(), 0, outputExprs,
                                    rewriter.getContext());

    SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    // Create linalg.generic for broadcast
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, input, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

// Helper function to broadcast for matmul
Value broadcastTensor(ConversionPatternRewriter &rewriter, Location loc,
                             Value input, ArrayRef<int64_t> targetShape) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  auto inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  int64_t targetRank = targetShape.size();

  // If shapes already match, no broadcast needed
  if (inputRank == targetRank &&
      std::equal(inputShape.begin(), inputShape.end(), targetShape.begin())) {
    return input;
  }

  Value current = input;

  // Handle rank expansion ( [M, K] -> [1, M, K])
  if (inputRank < targetRank) {
    int64_t rankDiff = targetRank - inputRank;

    // Build the expanded shape with leading 1s
    SmallVector<int64_t> expandedShape;
    for (int64_t i = 0; i < rankDiff; ++i) {
      expandedShape.push_back(1);
    }
    expandedShape.append(inputShape.begin(), inputShape.end());

    // Build reassociation indices for tensor.expand_shape
    SmallVector<ReassociationIndices> reassociation;

    // all the new dimensions (leading 1s) plus the first original dimension
    ReassociationIndices firstGroup;
    for (int64_t i = 0; i <= rankDiff; ++i) {
      firstGroup.push_back(i);
    }
    reassociation.push_back(firstGroup);

    // Remaining dimensions map 1:1
    for (int64_t i = 1; i < inputRank; ++i) {
      reassociation.push_back({rankDiff + i});
    }

    auto expandedType =
        RankedTensorType::get(expandedShape, inputType.getElementType());
    current = rewriter.create<tensor::ExpandShapeOp>(loc, expandedType, current,
                                                     reassociation);

    inputType = expandedType;
    inputShape = expandedShape;
    inputRank = targetRank;
  }

  // Handle broadcasting dimensions ([1, M, K] -> [B, M, K])
  bool needsBroadcast = false;
  for (int64_t i = 0; i < targetRank; ++i) {
    if (inputShape[i] != targetShape[i]) {
      needsBroadcast = true;
      break;
    }
  }

  if (!needsBroadcast) {
    return current;
  }

  // Build affine map for broadcasting
  SmallVector<AffineExpr> inputExprs;
  for (int64_t i = 0; i < targetRank; ++i) {
    int64_t inputSize = inputShape[i];
    int64_t targetSize = targetShape[i];

    // If broadcasting dimension (1 -> N), use constant 0
    if (inputSize == 1 && targetSize != 1) {
      inputExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }

  // Build affine map for output (identity)
  SmallVector<AffineExpr> outputExprs;
  for (int64_t i = 0; i < targetRank; ++i) {
    outputExprs.push_back(rewriter.getAffineDimExpr(i));
  }

  auto inputMap =
      AffineMap::get(targetRank, 0, inputExprs, rewriter.getContext());
  auto outputMap =
      AffineMap::get(targetRank, 0, outputExprs, rewriter.getContext());

  // Create empty output tensor
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, targetShape, inputType.getElementType());

  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
  SmallVector<utils::IteratorType> iteratorTypes(targetRank,
                                                 utils::IteratorType::parallel);

  // Create linalg.generic for broadcast
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{emptyTensor.getType()}, current, emptyTensor, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      });

  return genericOp.getResult(0);
}

// Fuses nova.transpose(A) + nova.matmul(A^T, B) → linalg.matmul_transpose_a(A, B).
// This prevents the explicit linalg.generic transpose from being created, so the
// K-tiling pass sees A in [K,M] layout and the shared-memory promotion copy reads
// M-contiguous rows — enabling cp.async.16 instead of scalar ld.global.b32.
// Only fires for 2D matmul where the LHS is exclusively consumed by this matmul
// (hasOneUse guard), which is always true for the backward dB = A^T × grad case.
struct NovaMatmulTransposeAFusionLowering
    : public OpConversionPattern<nova::MatmulOp> {
  NovaMatmulTransposeAFusionLowering(MLIRContext *ctx)
      : OpConversionPattern<nova::MatmulOp>(ctx, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(nova::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType || resultType.getRank() != 2)
      return failure();

    // LHS must be nova.transpose with a last-two-dims swap
    auto transpOp = op.getOperand(0).getDefiningOp<nova::TransposeOp>();
    if (!transpOp || !transpOp->hasOneUse())
      return failure();

    auto inputType = cast<RankedTensorType>(transpOp.getInput().getType());
    int64_t rank = inputType.getRank();
    int64_t ax1 = rank - 1, ax2 = rank - 2;
    if (auto a = transpOp->getAttrOfType<IntegerAttr>("axes1"))
      ax1 = a.getInt() < 0 ? a.getInt() + rank : a.getInt();
    if (auto a = transpOp->getAttrOfType<IntegerAttr>("axes2"))
      ax2 = a.getInt() < 0 ? a.getInt() + rank : a.getInt();
    if (!((ax1 == rank - 1 && ax2 == rank - 2) ||
          (ax1 == rank - 2 && ax2 == rank - 1)))
      return failure();

    Location loc = op.getLoc();
    // origA is A in [K, M] layout — keep it un-transposed.
    // Use the remapped form in case origA is itself a converted Nova op result.
    Value origA = transpOp.getInput();
    if (Value remapped = rewriter.getRemappedValue(origA))
      origA = remapped;
    Value rhs = adaptor.getOperands()[1];

    Value cst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Value outputTensor =
        rewriter.create<linalg::FillOp>(loc, cst, emptyTensor).getResult(0);

    // linalg.matmul_transpose_a: A[K,M], B[K,N] → C[M,N] = A^T × B
    rewriter.replaceOpWithNewOp<linalg::MatmulTransposeAOp>(
        op, ValueRange{origA, rhs}, outputTensor);
    return success();
  }
};

struct NovaMatmulOpLowering : public OpConversionPattern<nova::MatmulOp> {
  using OpConversionPattern<nova::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();

    if (operands.size() != 2) {
      return rewriter.notifyMatchFailure(op, "expected exactly 2 operands");
    }

    Value lhs = operands[0];
    Value rhs = operands[1];

    auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());

    // Get result type and create empty output tensor
    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }

    auto resultShape = resultType.getShape();
    auto newShape = resultShape.drop_back(2);
    int64_t resultRank = resultShape.size();

    // For rank > 3: flatten batch dimensions
    if (resultRank > 3) {
      // Broadcast LHS to match result batch dimensions
      SmallVector<int64_t> lhsTargetShape(newShape.begin(), newShape.end());
      lhsTargetShape.push_back(lhsType.getShape()[lhsType.getRank() - 2]);
      lhsTargetShape.push_back(lhsType.getShape()[lhsType.getRank() - 1]);
      lhs = broadcastTensor(rewriter, op.getLoc(), lhs, lhsTargetShape);
      lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());

      // Broadcast RHS to match result batch dimensions
      SmallVector<int64_t> rhsTargetShape(newShape.begin(), newShape.end());
      rhsTargetShape.push_back(rhsType.getShape()[rhsType.getRank() - 2]);
      rhsTargetShape.push_back(rhsType.getShape()[rhsType.getRank() - 1]);
      rhs = broadcastTensor(rewriter, op.getLoc(), rhs, rhsTargetShape);
      rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());

      int64_t N = 1;
      for (int64_t i = 0; i < (resultRank - 2); i++) {
        N *= resultType.getShape()[i];
      }

      int64_t M = resultType.getShape()[resultRank - 2];
      int64_t K = lhsType.getShape()[lhsType.getRank() - 1];
      int64_t N_cols = resultType.getShape()[resultRank - 1];

      SmallVector<int64_t> rank3_lhs_shape({N, M, K});
      SmallVector<int64_t> rank3_rhs_shape({N, K, N_cols});
      SmallVector<int64_t> rank3_output_shape({N, M, N_cols});

      auto rank3LhsType =
          RankedTensorType::get(rank3_lhs_shape, lhsType.getElementType());
      auto rank3RhsType =
          RankedTensorType::get(rank3_rhs_shape, rhsType.getElementType());

      // Build reassociation to collapse batch dimensions into one
      SmallVector<ReassociationIndices> lhsReassociation;
      ReassociationIndices batchIndices;
      for (int64_t i = 0; i < resultRank - 2; ++i) {
        batchIndices.push_back(i); // All batch dims collapse to one
      }
      lhsReassociation.push_back(batchIndices);
      lhsReassociation.push_back({resultRank - 2}); // M dimension
      lhsReassociation.push_back({resultRank - 1}); // K dimension

      SmallVector<ReassociationIndices> rhsReassociation;
      ReassociationIndices rhsBatchIndices;
      for (int64_t i = 0; i < resultRank - 2; ++i) {
        rhsBatchIndices.push_back(i); // All batch dims collapse to one
      }
      rhsReassociation.push_back(rhsBatchIndices);
      rhsReassociation.push_back({resultRank - 2}); // K dimension
      rhsReassociation.push_back({resultRank - 1}); // N dimension

      Value lhsCollapsed = rewriter.create<tensor::CollapseShapeOp>(
          op.getLoc(), rank3LhsType, lhs, lhsReassociation);
      Value rhsCollapsed = rewriter.create<tensor::CollapseShapeOp>(
          op.getLoc(), rank3RhsType, rhs, rhsReassociation);

      Value cst = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getZeroAttr(resultType.getElementType()));
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), rank3_output_shape, resultType.getElementType());
      Value outputTensor =
          rewriter.create<linalg::FillOp>(op.getLoc(), cst, emptyTensor)
              .getResult(0);

      Value matmul3D =
          rewriter
              .create<linalg::BatchMatmulOp>(
                  op.getLoc(), ValueRange{lhsCollapsed, rhsCollapsed},
                  outputTensor)
              .getResult(0);

      // Build reassociation to expand N back to original batch dimensions
      SmallVector<ReassociationIndices> resultReassociation;
      ReassociationIndices expandedBatchIndices;
      for (int64_t i = 0; i < resultRank - 2; ++i) {
        expandedBatchIndices.push_back(i); // N expands to all batch dims
      }
      resultReassociation.push_back(expandedBatchIndices);
      resultReassociation.push_back({resultRank - 2}); // M dimension
      resultReassociation.push_back({resultRank - 1}); // N dimension

      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          op, resultType, matmul3D, resultReassociation);
      return success();
    }

    // Handle broadcasting for batch matmul
    if (resultRank == 3) {
      // Result shape is [B, M, N]
      int64_t batchDim = resultShape[0];

      // Broadcast lhs to [B, M, K] if needed
      SmallVector<int64_t> lhsTargetShape;
      if (lhsType.getRank() == 2) {
        // [M, K] -> [B, M, K]
        lhsTargetShape = {batchDim, lhsType.getShape()[0],
                          lhsType.getShape()[1]};
      } else if (lhsType.getRank() == 3) {
        // [B', M, K] -> [B, M, K] (B' might be 1 or different)
        lhsTargetShape = {batchDim, lhsType.getShape()[1],
                          lhsType.getShape()[2]};
      }

      if (!lhsTargetShape.empty()) {
        lhs = broadcastTensor(rewriter, op.getLoc(), lhs, lhsTargetShape);
      }

      // Broadcast rhs to [B, K, N] if needed
      SmallVector<int64_t> rhsTargetShape;
      if (rhsType.getRank() == 2) {
        rhsTargetShape = {batchDim, rhsType.getShape()[0],
                          rhsType.getShape()[1]};
      } else if (rhsType.getRank() == 3) {
        rhsTargetShape = {batchDim, rhsType.getShape()[1],
                          rhsType.getShape()[2]};
      }

      if (!rhsTargetShape.empty()) {
        rhs = broadcastTensor(rewriter, op.getLoc(), rhs, rhsTargetShape);
      }
    }

    // create a constant zero
    Value cst = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(resultType.getElementType()));
    // Create an empty tensor for the output
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    // create a fill op to initialize the output tensor to zero
    Value outputTensor =
        rewriter.create<linalg::FillOp>(op.getLoc(), cst, emptyTensor)
            .getResult(0);

    // batch matmul
    if (resultType.getRank() == 3) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, ValueRange{lhs, rhs}, outputTensor);
      return success();
    }
    // Create linalg.matmul with inputs and output
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, ValueRange{lhs, rhs}, // inputs
        outputTensor);            // outputs
    return success();
  }
};

// RELU OP - > LINALG.MAX with 0
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

struct NovaScatterAddOpLowering
    : public OpConversionPattern<nova::ScatterAddOp> {
  using OpConversionPattern<nova::ScatterAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ScatterAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elementTy = resultType.getElementType();
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value indices = adaptor.getIndices();
    Value src = adaptor.getSrc();
    int64_t axis = op.getAxis();
    int64_t inputRank = resultType.getRank();

    if (axis < 0)
      axis += inputRank;

    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcShape = srcType.getShape();
    int64_t srcRank = srcType.getRank();

    // Normalize index element type: cast float indices to i32 before
    // bufferization. Narrow integer widening (e.g. i16→i32) is handled
    // after the scalar load via ExtUIOp (see below).
    auto indicesType = cast<RankedTensorType>(indices.getType());
    auto indicesElemType = indicesType.getElementType();
    Value processedIndices = indices;
    if (llvm::isa<FloatType>(indicesElemType)) {
      auto i32IndicesType =
          RankedTensorType::get(indicesType.getShape(), rewriter.getI32Type());
      processedIndices =
          rewriter.create<tosa::CastOp>(loc, i32IndicesType, indices);
    }
    auto processedIndicesType =
        cast<RankedTensorType>(processedIndices.getType());

    // Bufferize all operands to MemRef.
    Value inputMem =
        rewriter
            .create<ToBufferOp>(loc,
                                MemRefType::get(resultType.getShape(), elementTy),
                                input, /*read_only=*/false)
            .getResult();
    Value srcMem =
        rewriter
            .create<ToBufferOp>(loc,
                                MemRefType::get(srcType.getShape(),
                                                srcType.getElementType()),
                                src, /*read_only=*/true)
            .getResult();
    Value indicesMem =
        rewriter
            .create<ToBufferOp>(loc,
                                MemRefType::get(processedIndicesType.getShape(),
                                                processedIndicesType.getElementType()),
                                processedIndices, /*read_only=*/true)
            .getResult();

    // Two-level GPU parallel loop: outer forall maps to blocks (all dims except
    // innermost), inner forall maps to threads (innermost dim). This gives each
    // block multiple threads instead of 1 thread per element.
    constexpr int64_t kMaxThreadsPerBlock = 1024;
    int64_t innerDim = srcRank - 1;
    int64_t innerSize = srcShape[innerDim]; // may be dynamic

    // Outer forall: one block dimension per source dim except the innermost.
    SmallVector<OpFoldResult> outerLbs, outerUbs, outerSteps;
    SmallVector<Attribute> blockMapping;
    for (int64_t i = 0; i < srcRank - 1; ++i) {
      outerLbs.push_back(rewriter.getIndexAttr(0));
      outerSteps.push_back(rewriter.getIndexAttr(1));
      if (srcShape[i] == ShapedType::kDynamic)
        outerUbs.push_back(
            rewriter.create<memref::DimOp>(loc, srcMem, i).getResult());
      else
        outerUbs.push_back(rewriter.getIndexAttr(srcShape[i]));
      gpu::MappingId id = (i == 0) ? gpu::MappingId::DimX
                        : (i == 1) ? gpu::MappingId::DimY
                                   : gpu::MappingId::DimZ;
      blockMapping.push_back(
          gpu::GPUBlockMappingAttr::get(rewriter.getContext(), id));
    }

    // 1-D case: tile the single dim into ceil(N/1024) blocks × min(N,1024)
    // threads to stay within CUDA's 1024-thread-per-block limit.
    if (srcRank == 1) {
      int64_t numBlocks = 1;
      if (innerSize != ShapedType::kDynamic && innerSize > kMaxThreadsPerBlock) {
        numBlocks = (innerSize + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
        innerSize = kMaxThreadsPerBlock;
      }
      outerLbs.push_back(rewriter.getIndexAttr(0));
      outerUbs.push_back(rewriter.getIndexAttr(numBlocks));
      outerSteps.push_back(rewriter.getIndexAttr(1));
      blockMapping.push_back(gpu::GPUBlockMappingAttr::get(
          rewriter.getContext(), gpu::MappingId::DimX));
    }

    auto outerForall = rewriter.create<scf::ForallOp>(
        loc, outerLbs, outerUbs, outerSteps, ValueRange{},
        rewriter.getArrayAttr(blockMapping));

    {
      Block *outerBody = outerForall.getBody();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(outerBody, outerBody->without_terminator().begin());

      // Capture outer IVs directly from block args — getInductionVars() on a
      // ConversionPatternRewriter-created op can return remapped values after
      // the insertion point moves into a nested body.
      SmallVector<Value> outerIVsSaved(outerBody->getArguments().begin(),
                                       outerBody->getArguments().end());

      // Inner forall: thread-mapped over the innermost dim.
      SmallVector<OpFoldResult> innerUbs;
      if (innerSize == ShapedType::kDynamic)
        innerUbs.push_back(rewriter.create<memref::DimOp>(
            loc, srcMem,
            rewriter.create<arith::ConstantIndexOp>(loc, innerDim)).getResult());
      else
        innerUbs.push_back(rewriter.getIndexAttr(innerSize));

      auto innerForall = rewriter.create<scf::ForallOp>(
          loc,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0)}, innerUbs,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)}, ValueRange{},
          rewriter.getArrayAttr(SmallVector<Attribute>{
              gpu::GPUThreadMappingAttr::get(rewriter.getContext(),
                                             gpu::MappingId::LinearDim0)}));

      Block *innerBody = innerForall.getBody();
      rewriter.setInsertionPoint(innerBody, innerBody->without_terminator().begin());

      // Compose IVs: outer IVs (blocks) + inner IV (thread within block).
      SmallVector<Value> ivs;
      if (srcRank == 1) {
        // 1D: elemIdx = blockIV * kMaxThreadsPerBlock + threadIV,
        // guarded against the last-block tail when N % 1024 != 0.
        Value elemIdx = rewriter.create<arith::AddIOp>(
            loc,
            rewriter.create<arith::MulIOp>(
                loc, outerIVsSaved[0],
                rewriter.create<arith::ConstantIndexOp>(loc, kMaxThreadsPerBlock)),
            innerForall.getInductionVars()[0]);
        Value inBounds = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, elemIdx,
            rewriter.create<arith::ConstantIndexOp>(loc, srcShape[0]));
        auto ifOp = rewriter.create<scf::IfOp>(loc, inBounds,
                                               /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        ivs.push_back(elemIdx);
      } else {
        for (Value iv : outerIVsSaved)
          ivs.push_back(iv);
        ivs.push_back(innerForall.getInductionVars()[0]);
      }

      // Load the scatter index, then widen narrow integers to i32 via
      // zero-extension (preserves unsigned range, e.g. i16 [0..65535]).
      Value idxVal =
          rewriter.create<memref::LoadOp>(loc, indicesMem, ValueRange{ivs[axis]});
      if (auto intType = llvm::dyn_cast<IntegerType>(indicesElemType))
        if (intType.getWidth() < 32)
          idxVal = rewriter.create<arith::ExtUIOp>(
              loc, rewriter.getI32Type(), idxVal);
      Value targetIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), idxVal);

      Value val = rewriter.create<memref::LoadOp>(loc, srcMem, ivs);

      SmallVector<Value> dstCoords;
      for (int64_t d = 0; d < srcRank; ++d)
        dstCoords.push_back(d == axis ? targetIdx : ivs[d]);

      // Bounds check: only issue atomic_rmw when targetIdx ∈ [0, inputShape[axis]).
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value axisSize =
          (resultType.getShape()[axis] == ShapedType::kDynamic)
              ? rewriter.create<memref::DimOp>(loc, inputMem, axis).getResult()
              : rewriter
                    .create<arith::ConstantIndexOp>(
                        loc, resultType.getShape()[axis])
                    .getResult();
      Value inBounds = rewriter.create<arith::AndIOp>(
          loc,
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                         targetIdx, zero),
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                         targetIdx, axisSize));
      auto ifOp = rewriter.create<scf::IfOp>(loc, inBounds,
                                             /*withElseRegion=*/false);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());

      arith::AtomicRMWKind kind = llvm::isa<FloatType>(elementTy)
                                      ? arith::AtomicRMWKind::addf
                                      : arith::AtomicRMWKind::addi;
      rewriter.create<memref::AtomicRMWOp>(loc, kind, val, inputMem, dstCoords);
    }

    rewriter.setInsertionPointAfter(outerForall);
    Value resultTensor =
        rewriter.create<ToTensorOp>(loc, resultType, inputMem, /*restrict=*/true)
            .getResult();
    rewriter.replaceOp(op, resultTensor);
    return success();
  }
};

struct NovaTransposeOpLowering : public OpConversionPattern<mlir::nova::TransposeOp> {
  using OpConversionPattern<mlir::nova::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::nova::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    auto resultShape = resultType.getShape();
    int64_t rank = resultShape.size();

    int64_t axes1 = rank - 1;
    int64_t axes2 = rank - 2;

    if (auto attr = op->getAttrOfType<IntegerAttr>("axes1"))
      axes1 = attr.getInt();
    if (auto attr = op->getAttrOfType<IntegerAttr>("axes2"))
      axes2 = attr.getInt();

    if (axes1 < 0)
      axes1 += rank;
    if (axes2 < 0)
      axes2 += rank;

    llvm::SmallVector<int64_t> perms;
    for (int64_t i = 0; i < rank; i++) {
      if (i == axes1)
        perms.push_back(axes2);
      else if (i == axes2)
        perms.push_back(axes1);
      else
        perms.push_back(i);
    }

    Location loc = op.getLoc();  
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());  
  
    // Input map: identity — iterate over input's natural (row-major) layout  
    // so that the global memory read is coalesced.  
    auto inputMap = rewriter.getMultiDimIdentityMap(rank);  
  
    // Output map: inverse permutation of perms.  
    // perms[i] = j means output_dim_i comes from input_dim_j.  
    // inversePerms[j] = i means input_dim_j maps to output_dim_i.  
    // The output map is (d0, d1, ...) -> (d_{inversePerms[0]}, d_{inversePerms[1]}, ...)  
    SmallVector<int64_t> inversePerms(rank);  
    for (int64_t i = 0; i < rank; ++i)  
      inversePerms[perms[i]] = i;  
  
    SmallVector<AffineExpr> outputExprs;  
    for (int64_t i = 0; i < rank; ++i)  
      outputExprs.push_back(rewriter.getAffineDimExpr(inversePerms[i]));  
    auto outputMap = AffineMap::get(rank, 0, outputExprs, rewriter.getContext());  
  
    SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};  
    SmallVector<utils::IteratorType> iteratorTypes(  
        rank, utils::IteratorType::parallel);  
  
    // Output tensor has the transposed (result) shape.  
    Value permutedInit = rewriter.create<tensor::EmptyOp>(  
        loc, resultShape, resultType.getElementType());  
  
    // Create linalg.generic: body just yields the input element.  
    auto genericOp = rewriter.create<linalg::GenericOp>(  
    loc, /*resultTypes=*/TypeRange{resultType},  
    /*inputs=*/ValueRange{adaptor.getInput()},   // ← wrap in ValueRange  
    /*outputs=*/permutedInit,                     // ← now a Value, not EmptyOp  
    SmallVector<AffineMap>{inputMap, outputMap},  
    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),  
    [&](OpBuilder &b, Location nl, ValueRange args) {  
      b.create<linalg::YieldOp>(nl, args[0]);  
    });  
  
rewriter.replaceOp(op, genericOp.getResults()); 
    return success();

  }
};
struct NovaToDeviceOpLowering : public OpConversionPattern<nova::ToDeviceOp> {
  using OpConversionPattern<nova::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ToDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Identity maps for linalg.generic
    auto identityMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, input, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct NovaRandomOpLowering : public OpConversionPattern<nova::Rndm2DOp> {
  using OpConversionPattern<nova::Rndm2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::Rndm2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resulttype = cast<mlir::RankedTensorType>(op.getType());
    if (!resulttype) {
      return failure();
    }
    auto loc = op.getLoc();
    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resulttype.getShape(), resulttype.getElementType());
    ValueRange args = op.getOperands();
    // seed= (min+max)^max
    Value sum = rewriter.create<arith::AddFOp>(loc, args[0], args[1]);
    Value mySeed = rewriter.create<math::PowFOp>(loc, sum, args[1]);
    Value seedVal =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI32Type(), mySeed);

    auto linalgop = rewriter.create<linalg::FillRng2DOp>(
        loc, ValueRange{args[0], args[1], seedVal}, ValueRange{output});

    rewriter.replaceOp(op, linalgop);
    return success();
  }
};

//----------------------------------------------------------------
//                          Argmin
//----------------------------------------------------------------
static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<nova::ArgMinOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       cast<FloatType>(elementTy).getFloatSemantics(), false));

  if (isa<nova::ArgMinOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}


class ArgMinConverter : public OpRewritePattern<nova::ArgMinOp> {
public:
  using OpRewritePattern<nova::ArgMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ArgMinOp argminOp,
                                PatternRewriter &rewriter) const final {
    auto loc = argminOp.getLoc();
    Value input = argminOp.getInput();
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<RankedTensorType>(argminOp.getType());
    auto inElementTy = inputTy.getElementType();
    auto outElementTy = resultTy.getElementType();
    int axis = static_cast<int>(argminOp.getDimension().value());
    auto resultMinTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

    if (!isa<IntegerType>(outElementTy))
      return rewriter.notifyMatchFailure(
          argminOp,
          "nova.arg_min to linalg.* requires integer-like result type");

    // Use i64 for the index accumulator inside the generic to avoid an
    // index-to-i32 trunc in the inner loop, which triggers an LLVM
    // LoopStrengthReduce bug on NVPTX when the reduction is the innermost dim.
    auto accIdxTy = rewriter.getI64Type();
    auto accResultTy = RankedTensorType::get(resultTy.getShape(), accIdxTy);

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index (using i64).
    auto emptyTensorIdx = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       accIdxTy, dynDims)
                              .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(accIdxTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running min.
    auto emptyTensorMin = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       inElementTy, dynDims)
                              .getResult();
    auto fillValueMinAttr =
        createInitialValueForReduceOp(argminOp, inElementTy, rewriter);

    if (!fillValueMinAttr)
      return rewriter.notifyMatchFailure(
          argminOp, "unsupported nova.argmin element type");

    auto fillValueMin =
        rewriter.create<arith::ConstantOp>(loc, fillValueMinAttr);
    auto filledTensorMin =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueMin},
                                    ValueRange{emptyTensorMin})
            .result();

    // We need to reduce along the arg-min axis, with parallel operations along
    // the rest.
    SmallVector<utils::IteratorType, 4> iteratorTypes;
    iteratorTypes.resize(inputTy.getRank(), utils::IteratorType::parallel);
    iteratorTypes[axis] = utils::IteratorType::reduction;

    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    for (int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
      srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (axis != i)
        dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }

    bool didEncounterError = false;
    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs},
                                             rewriter.getContext());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({accResultTy, resultMinTy}), input,
        ValueRange({filledTensorIdx, filledTensorMin}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, accIdxTy,
              rewriter.create<linalg::IndexOp>(loc, axis));

          Value predicate;
          if (isa<FloatType>(inElementTy)) {
            if (argminOp.getIgnoreNan()) {
              // Only update index & min value for non NaN values. If all
              // values are NaNs, the initial index will be return which is 0.
              predicate = rewriter.create<arith::CmpFOp>(
                  nestedLoc, arith::CmpFPredicate::OLT, newValue, oldValue);
            } else {
              // Update min value if either of the following is true:
              // - new value is bigger
              // - cur min is not NaN and new value is NaN
              Value lt = rewriter.create<arith::CmpFOp>(
                  nestedLoc, arith::CmpFPredicate::ULT, newValue, oldValue);
              Value oldNonNaN = rewriter.create<arith::CmpFOp>(
                  nestedLoc, arith::CmpFPredicate::ORD, oldValue, oldValue);
              predicate = rewriter.create<arith::AndIOp>(
                  nestedLoc, rewriter.getI1Type(), lt, oldNonNaN);
            }
          } else if (isa<IntegerType>(inElementTy)) {
            predicate = rewriter.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::slt, newValue, oldValue);
          } else {
            didEncounterError = true;
            return;
          }

          auto resultMin = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultIndex, resultMin}));
        });

    if (didEncounterError)
      return rewriter.notifyMatchFailure(
          argminOp, "unsupported nova.argmin element type");

    // Truncate the i64 index result back to the original output type using a
    // linalg.generic so that the bufferizer can handle the tensor operation.
    Value idxResult = linalgOp.getResult(0);
    if (accIdxTy != outElementTy) {
      auto emptyOut = rewriter.create<tensor::EmptyOp>(
          loc, resultTy.getShape(), outElementTy, dynDims);
      SmallVector<AffineMap> truncMaps(
          2, rewriter.getMultiDimIdentityMap(resultTy.getRank()));
      SmallVector<utils::IteratorType> truncIters(resultTy.getRank(),
                                                  utils::IteratorType::parallel);
      auto truncOp = rewriter.create<linalg::GenericOp>(
          loc, resultTy, idxResult, ValueRange{emptyOut.getResult()},
          truncMaps, truncIters,
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value truncated = b.create<arith::TruncIOp>(l, outElementTy, args[0]);
            b.create<linalg::YieldOp>(l, truncated);
          });
      idxResult = truncOp.getResult(0);
    }
    rewriter.replaceOp(argminOp, idxResult);
    return success();
  }
};

//----------------------------------------------------------------
//                          ArgMax
//----------------------------------------------------------------
static TypedAttr createInitialValueForArgMaxOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<nova::ArgmaxOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy,
        APFloat::getInf(cast<FloatType>(elementTy).getFloatSemantics(), true));

  if (isa<nova::ArgmaxOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}

class ArgMaxConverter : public OpRewritePattern<nova::ArgmaxOp> {
public:
  using OpRewritePattern<nova::ArgmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ArgmaxOp argmaxOp,
                                PatternRewriter &rewriter) const final {
    auto loc = argmaxOp.getLoc();
    Value input = argmaxOp.getInput();

    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<RankedTensorType>(argmaxOp.getType());
    auto inElementTy = inputTy.getElementType();
    auto outElementTy = resultTy.getElementType();
    int axis = static_cast<int>(argmaxOp.getDimension().value_or(0));
    if (axis < 0)
      axis += inputTy.getRank();

    auto resultMaxTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

    if (!isa<IntegerType>(outElementTy))
      return rewriter.notifyMatchFailure(
          argmaxOp,
          "nova.argmax to linalg.* requires integer-like result type");

    // Use i64 for the index accumulator inside the generic to avoid an
    // index-to-i32 trunc in the inner loop, which triggers an LLVM
    // LoopStrengthReduce bug on NVPTX when the reduction is the innermost dim.
    auto accIdxTy = rewriter.getI64Type();
    auto accResultTy = RankedTensorType::get(resultTy.getShape(), accIdxTy);

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index (using i64).
    auto emptyTensorIdx = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       accIdxTy, dynDims)
                              .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(accIdxTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running max.
    auto emptyTensorMax = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       inElementTy, dynDims)
                              .getResult();

    auto fillValueMaxAttr =
        createInitialValueForArgMaxOp(argmaxOp, inElementTy, rewriter);
    auto fillValueMax =
        rewriter.create<arith::ConstantOp>(loc, fillValueMaxAttr);

    auto filledTensorMax =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueMax},
                                    ValueRange{emptyTensorMax})
            .result();

    // We need to reduce along the arg-max axis, with parallel operations
    // along the rest.
    SmallVector<utils::IteratorType, 4> iteratorTypes;
    iteratorTypes.resize(inputTy.getRank(), utils::IteratorType::parallel);
    iteratorTypes[axis] = utils::IteratorType::reduction;

    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    for (int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
      srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (axis != i)
        dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }

    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs},
                                             rewriter.getContext());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({accResultTy, resultMaxTy}), input,
        ValueRange({filledTensorIdx, filledTensorMax}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, accIdxTy,
              rewriter.create<linalg::IndexOp>(loc, axis));

          Value predicate;
          if (isa<FloatType>(inElementTy)) {
            // For Max: newValue > oldValue
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          } else if (isa<IntegerType>(inElementTy)) {
            predicate = rewriter.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::sgt, newValue, oldValue);
          }

          auto resultMax = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultIndex, resultMax}));
        });

    // Truncate the i64 index result back to the original output type using a
    // linalg.generic so that the bufferizer can handle the tensor operation.
    Value idxResult = linalgOp.getResult(0);
    if (accIdxTy != outElementTy) {
      auto emptyOut = rewriter.create<tensor::EmptyOp>(
          loc, resultTy.getShape(), outElementTy, dynDims);
      SmallVector<AffineMap> truncMaps(
          2, rewriter.getMultiDimIdentityMap(resultTy.getRank()));
      SmallVector<utils::IteratorType> truncIters(resultTy.getRank(),
                                                  utils::IteratorType::parallel);
      auto truncOp = rewriter.create<linalg::GenericOp>(
          loc, resultTy, idxResult, ValueRange{emptyOut.getResult()},
          truncMaps, truncIters,
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value truncated = b.create<arith::TruncIOp>(l, outElementTy, args[0]);
            b.create<linalg::YieldOp>(l, truncated);
          });
      idxResult = truncOp.getResult(0);
    }
    rewriter.replaceOp(argmaxOp, idxResult);
    return success();
  }
};

//----------------------------------------------------------------
//                          ReduceOp
//----------------------------------------------------------------
// Helper to get identity value for reduction operations
static TypedAttr getReduceIdentity(nova::ReductionKind kind, Type elemType,
                                   PatternRewriter &rewriter) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    switch (kind) {
    case nova::ReductionKind::SUM:
    case nova::ReductionKind::MEAN:
      return rewriter.getFloatAttr(elemType, 0.0);
    case nova::ReductionKind::PRODUCT:
      return rewriter.getFloatAttr(elemType, 1.0);
    case nova::ReductionKind::MAX:
      return rewriter.getFloatAttr(
          elemType,
          APFloat::getInf(floatType.getFloatSemantics(), /*neg=*/true));
    case nova::ReductionKind::MIN:
      return rewriter.getFloatAttr(
          elemType,
          APFloat::getInf(floatType.getFloatSemantics(), /*neg=*/false));
    default:
      return {};
    }
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    unsigned bw = intType.getWidth();
    switch (kind) {
    case nova::ReductionKind::SUM:
    case nova::ReductionKind::MEAN:
      return rewriter.getIntegerAttr(elemType, 0);
    case nova::ReductionKind::PRODUCT:
      return rewriter.getIntegerAttr(elemType, 1);
    case nova::ReductionKind::MAX:
      return rewriter.getIntegerAttr(elemType, APInt::getSignedMinValue(bw));
    case nova::ReductionKind::MIN:
      return rewriter.getIntegerAttr(elemType, APInt::getSignedMaxValue(bw));
    case nova::ReductionKind::ALL:
      return rewriter.getIntegerAttr(elemType, 1);
    case nova::ReductionKind::ANY:
      return rewriter.getIntegerAttr(elemType, 0);
    }
  }
  return {};
}

// Helper to create reduction combiner operation
static Value createReduceCombiner(OpBuilder &b, Location loc,
                                  nova::ReductionKind kind, Value lhs,
                                  Value rhs, Type elemType) {
  switch (kind) {
  case nova::ReductionKind::SUM:
  case nova::ReductionKind::MEAN:
    return isa<FloatType>(elemType)
               ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::AddIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::PRODUCT:
    return isa<FloatType>(elemType)
               ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MulIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::MAX:
    return isa<FloatType>(elemType)
               ? b.create<arith::MaximumFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::MIN:
    return isa<FloatType>(elemType)
               ? b.create<arith::MinimumFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MinSIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::ALL:
    return b.create<arith::AndIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::ANY:
    return b.create<arith::OrIOp>(loc, lhs, rhs).getResult();
  }
  return nullptr;
}

// Linalg.generic based lowering for keepdims=true
static LogicalResult
lowerWithLinalgGeneric(nova::ReduceOp op, PatternRewriter &rewriter,
                       Location loc, Value input, RankedTensorType inputType,
                       RankedTensorType resultType, Type elemType, int64_t rank,
                       nova::ReductionKind kind, SmallVector<int64_t> &axes) {

  llvm::sort(axes);
  llvm::SmallDenseSet<int64_t> axisSet(axes.begin(), axes.end());

  // Handle ALL/ANY: cast to i1 first
  Type reductionElemType = elemType;
  Value current = input;
  if (kind == nova::ReductionKind::ALL || kind == nova::ReductionKind::ANY) {
    reductionElemType = rewriter.getI1Type();
    if (elemType != reductionElemType) {
      auto boolType = RankedTensorType::get(inputType.getShape(), reductionElemType);
      current = rewriter.create<tosa::CastOp>(loc, boolType, current);
    }
  }

  // Compute total reduced elements for MEAN
  int64_t totalReducedElements = 1;
  for (int64_t axis : axes) {
    totalReducedElements *= inputType.getDimSize(axis);
  }

  // Get identity value
  auto identityAttr = getReduceIdentity(kind, reductionElemType, rewriter);
  if (!identityAttr)
    return rewriter.notifyMatchFailure(op, "unsupported reduction kind");
  Value identity = rewriter.create<arith::ConstantOp>(loc, identityAttr);

  // Compute SQUEEZED shape (completely remove reduced dims)
  // This allows the indexing map to be a valid permuted projection.
  SmallVector<int64_t> squeezedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!axisSet.contains(i))
      squeezedShape.push_back(inputType.getDimSize(i));
  }
  auto squeezedType = RankedTensorType::get(squeezedShape, reductionElemType);

  // Create squeezed output tensor
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, squeezedShape, reductionElemType, ValueRange{});
  Value filledTensor = rewriter.create<linalg::FillOp>(loc, identity, emptyTensor).result();

  // Map logic remains similar, but output rank is now (rank - axes.size())
  SmallVector<int64_t> parallelAxes;
  SmallVector<int64_t> reductionAxes;
  for (int64_t i = 0; i < rank; ++i) {
    if (axisSet.contains(i)) reductionAxes.push_back(i);
    else parallelAxes.push_back(i);
  }

  // logicalToLoop[logical_dim] = generic_loop_index
  SmallVector<int64_t> logicalToLoop(rank);
  for (size_t i = 0; i < parallelAxes.size(); ++i)
    logicalToLoop[parallelAxes[i]] = i;
  for (size_t i = 0; i < reductionAxes.size(); ++i)
    logicalToLoop[reductionAxes[i]] = parallelAxes.size() + i;

  SmallVector<AffineExpr> inputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    inputExprs.push_back(rewriter.getAffineDimExpr(logicalToLoop[i]));
  }

  //  Only add DimExprs for parallel dims (results in rank N-K)
  SmallVector<AffineExpr> outputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    if (!axisSet.contains(i)) {
      outputExprs.push_back(rewriter.getAffineDimExpr(logicalToLoop[i]));
    }
  }

  SmallVector<utils::IteratorType> iteratorTypes;
  for (size_t i = 0; i < parallelAxes.size(); ++i)
    iteratorTypes.push_back(utils::IteratorType::parallel);
  for (size_t i = 0; i < reductionAxes.size(); ++i)
    iteratorTypes.push_back(utils::IteratorType::reduction);

  auto inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
  auto outputMap = AffineMap::get(rank, 0, outputExprs, rewriter.getContext());

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, squeezedType, current, filledTensor,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value result = createReduceCombiner(b, nestedLoc, kind, args[0], args[1], reductionElemType);
        b.create<linalg::YieldOp>(nestedLoc, result);
      });

  Value reduced = genericOp.getResult(0);

  //  divide by total reduced elements
  if (kind == nova::ReductionKind::MEAN && isa<FloatType>(reductionElemType)) {
    double divisor = static_cast<double>(totalReducedElements);
    // Scalar constant captured by the region – no extra tensor needed.
    Value divisorVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(reductionElemType, divisor));

    int64_t squeezedRank = static_cast<int64_t>(squeezedShape.size());
    // Single identity map: every loop index maps to the same output element.
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(squeezedRank);

    auto divOp = rewriter.create<linalg::GenericOp>(
        loc, squeezedType,
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{reduced},
        /*indexingMaps=*/SmallVector<AffineMap>{identityMap},
        /*iteratorTypes=*/getNParallelLoopsAttrs(squeezedRank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          // args[0] is the accumulated sum element (from outs buffer).
          Value res = b.create<arith::DivFOp>(nestedLoc, args[0], divisorVal);
          b.create<linalg::YieldOp>(nestedLoc, res);
        });
    reduced = divOp.getResult(0);
  }

  //  Reshape back to the resultType (handles keepdims=true or false)
  if (cast<RankedTensorType>(reduced.getType()) != resultType) {
      auto shapeType = RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType());
      auto shapeAttr = DenseIntElementsAttr::get(shapeType, resultType.getShape());
      auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
          loc, mlir::tosa::shapeType::get(rewriter.getContext(), resultType.getRank()), shapeAttr);
      reduced = rewriter.create<tosa::ReshapeOp>(loc, resultType, reduced, shapeConst);
  }

  rewriter.replaceOp(op, reduced);
  return success();
}
static LogicalResult
lowerFullReduceMeanToSCF(nova::ReduceOp op, PatternRewriter &rewriter,
                         Location loc, Value input, RankedTensorType inputType,
                         RankedTensorType resultType, Type elemType,
                         int64_t rank, SmallVector<int64_t> &axes) {

  assert(static_cast<int64_t>(axes.size()) == rank &&
         "lowerFullReduceMeanToSCF expects a full reduction over all axes");
  assert(isa<FloatType>(elemType) &&
         "full-reduction MEAN SCF path only supports float types");

  llvm::sort(axes);

  int64_t totalElems = 1;
  for (int64_t d : axes)
    totalElems *= inputType.getDimSize(d);

  Value fZero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(elemType, 0.0));

  auto scalarTensorType = RankedTensorType::get({}, elemType);

  auto inputMemType = MemRefType::get(inputType.getShape(), elemType);
  Value inputMem = rewriter.create<ToBufferOp>(
      loc, inputMemType, input, /*read_only=*/true).getResult();

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  
  // We distribute across a fixed number of blocks (e.g., 64)
  int64_t numBlocks = 64;
  int64_t elemsPerBlock = (totalElems + numBlocks - 1) / numBlocks;

  Value cNumBlocks = rewriter.create<arith::ConstantIndexOp>(loc, numBlocks);

  // Allocate an intermediate memref for the 64 partial sums
  auto partialMemType = MemRefType::get({numBlocks}, elemType);
  Value partialMem = rewriter.create<memref::AllocOp>(loc, partialMemType);

  SmallVector<Attribute> mapping;
  mapping.push_back(gpu::GPUBlockMappingAttr::get(rewriter.getContext(), gpu::MappingId::DimX));
  ArrayAttr mappingAttr = rewriter.getArrayAttr(mapping);

  // --- PASS 1: Partial Reduction (Distributed across blocks) ---
  auto forallOp1 = rewriter.create<scf::ForallOp>(
      loc,
      /*lbs=*/SmallVector<OpFoldResult>{getAsOpFoldResult(c0)},
      /*ubs=*/SmallVector<OpFoldResult>{getAsOpFoldResult(cNumBlocks)},
      /*steps=*/SmallVector<OpFoldResult>{getAsOpFoldResult(c1)},
      /*outputs=*/ValueRange{},  
      /*mapping=*/mappingAttr);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forallOp1.getBody());

    Value blockId = forallOp1.getInductionVar(0);

    // Compute start and end linear indices for this block
    Value cElemsPerBlock = rewriter.create<arith::ConstantIndexOp>(loc, elemsPerBlock);
    Value startIdx = rewriter.create<arith::MulIOp>(loc, blockId, cElemsPerBlock);
    Value endIdxUnclamped = rewriter.create<arith::AddIOp>(loc, startIdx, cElemsPerBlock);
    Value cTotalElems = rewriter.create<arith::ConstantIndexOp>(loc, totalElems);
    
    // clamp endIdx to totalElems
    Value isExceeding = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, endIdxUnclamped, cTotalElems);
    Value endIdx = rewriter.create<arith::SelectOp>(loc, isExceeding, cTotalElems, endIdxUnclamped);

    // Sequential loop within the block over its designated chunk
    auto forOp = rewriter.create<scf::ForOp>(
        loc, startIdx, endIdx, c1, ValueRange{fZero});
    {
      OpBuilder::InsertionGuard gLevel(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value linearIdx = forOp.getInductionVar();
      Value currentSum = forOp.getRegionIterArgs()[0];

      // Delinearize 'linearIdx' into multi-dimensional indices 'ivs'
      SmallVector<Value> ivs(rank);
      Value rem = linearIdx;
      for (int64_t d = rank - 1; d >= 0; --d) {
        int64_t dimSize = inputType.getDimSize(d);
        Value cDimSize = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
        if (d == 0) {
          ivs[d] = rem;
        } else {
          ivs[d] = rewriter.create<arith::RemSIOp>(loc, rem, cDimSize);
          rem = rewriter.create<arith::DivSIOp>(loc, rem, cDimSize);
        }
      }

      Value elem = rewriter.create<memref::LoadOp>(loc, inputMem, ivs);
      Value nextSum = rewriter.create<arith::AddFOp>(loc, currentSum, elem);
      rewriter.create<scf::YieldOp>(loc, nextSum);
    }
    Value partialSum = forOp.getResult(0);

    // Store partial sum in the intermediate buffer
    rewriter.create<memref::StoreOp>(loc, partialSum, partialMem, ValueRange{blockId});
  }

  // --- PASS 2: Final Reduction (Single Block) ---
  auto scalarMemType = MemRefType::get({}, elemType);
  Value finalMem = rewriter.create<memref::AllocOp>(loc, scalarMemType);

  auto forallOp2 = rewriter.create<scf::ForallOp>(
      loc,
      /*lbs=*/SmallVector<OpFoldResult>{getAsOpFoldResult(c0)},
      /*ubs=*/SmallVector<OpFoldResult>{getAsOpFoldResult(c1)},
      /*steps=*/SmallVector<OpFoldResult>{getAsOpFoldResult(c1)},
      /*outputs=*/ValueRange{},  
      /*mapping=*/mappingAttr);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forallOp2.getBody());

    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, cNumBlocks, c1, ValueRange{fZero});
    {
      OpBuilder::InsertionGuard gLevel(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value idx = forOp.getInductionVar();
      Value currentSum = forOp.getRegionIterArgs()[0];
      Value partialElem = rewriter.create<memref::LoadOp>(loc, partialMem, ValueRange{idx});
      Value nextSum = rewriter.create<arith::AddFOp>(loc, currentSum, partialElem);
      rewriter.create<scf::YieldOp>(loc, nextSum);
    }
    Value totalSum = forOp.getResult(0);

    Value divisorVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, static_cast<double>(totalElems)));
    Value mean = rewriter.create<arith::DivFOp>(loc, totalSum, divisorVal);
    rewriter.create<memref::StoreOp>(loc, mean, finalMem, ValueRange{});
  }

  // Convert final scalar memref back to tensor
  Value resultTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, scalarTensorType, finalMem, /*restrict=*/true).getResult();

  // Reshape to final result shape if needed
  Value result = resultTensor;
  if (cast<RankedTensorType>(result.getType()) != resultType) {
    auto shapeType = RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, resultType.getShape());
    auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
        loc, mlir::tosa::shapeType::get(rewriter.getContext(), resultType.getRank()), shapeAttr);
    result = rewriter.create<tosa::ReshapeOp>(loc, resultType, result, shapeConst);
  }

  // Deallocate intermediate buffer (memref memory management)
  // rewriter.create<memref::DeallocOp>(loc, partialMem); // Optional standard cleanup, often omitted in MLIR tensor passes till bufferization finalization, but good practice.

  rewriter.replaceOp(op, result);
  return success();
}

class ReduceOpConverter : public OpRewritePattern<nova::ReduceOp> {
public:
  using OpRewritePattern<nova::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elemType = inputType.getElementType();
    int64_t rank = inputType.getRank();

    nova::ReductionKind kind = op.getKind();

    // Collect reduction axes
    SmallVector<int64_t> axes;
    if (auto dims = op.getDimension()) {
      for (auto attr : *dims) {
        int64_t axis = cast<IntegerAttr>(attr).getInt();
        if (axis < 0)
          axis += rank;
        axes.push_back(axis);
      }
    } else {
      // Reduce all dimensions
      for (int64_t i = 0; i < rank; ++i)
        axes.push_back(i);
    }

    // Full-reduction MEAN → scf.forall (GPU block) + nested scf.for loops.
    // All other kinds and all partial reductions go through linalg.generic.
    bool isFullReduction = (static_cast<int64_t>(axes.size()) == rank);
    if (kind == nova::ReductionKind::MEAN && isFullReduction &&
        isa<FloatType>(elemType)) {
      return lowerFullReduceMeanToSCF(op, rewriter, loc, input, inputType,
                                      resultType, elemType, rank, axes);
    }

    // Use linalg.generic path for everything else (partial reductions,
    // other reduction kinds, non-float MEAN, …).
    return lowerWithLinalgGeneric(op, rewriter, loc, input, inputType,
                                  resultType, elemType, rank, kind, axes);

    // Sort axes in descending order so we can reduce from higher dims first
    // (this prevents axis index shifting issues)
    llvm::sort(axes, std::greater<int64_t>());

    Value current = input;
    int64_t totalReducedElements = 1;

    // Reduce each axis using TOSA reduce ops
    for (int64_t axis : axes) {
      auto currentType = cast<RankedTensorType>(current.getType());
      Type currentElemType = currentType.getElementType();

      // Compute intermediate result shape (reduced axis becomes 1)
      SmallVector<int64_t> resultShape =
          llvm::to_vector(currentType.getShape());
      totalReducedElements *= resultShape[axis];
      resultShape[axis] = 1;

      auto axisAttr = rewriter.getI32IntegerAttr(axis);

      switch (kind) {
      case nova::ReductionKind::SUM:
      case nova::ReductionKind::MEAN: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceSumOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::MAX: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceMaxOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::MIN: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceMinOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::PRODUCT: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceProductOp>(loc, intermediateType,
                                                         current, axisAttr);
        break;
      }
      case nova::ReductionKind::ALL: {
        // ALL requires boolean input - cast to i1 if needed
        Type i1Type = rewriter.getI1Type();
        if (currentElemType != i1Type) {
          auto boolType = RankedTensorType::get(currentType.getShape(), i1Type);
          current = rewriter.create<tosa::CastOp>(loc, boolType, current);
        }
        auto intermediateType = RankedTensorType::get(resultShape, i1Type);
        current = rewriter.create<tosa::ReduceAllOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::ANY: {
        // ANY requires boolean input - cast to i1 if needed
        Type i1Type = rewriter.getI1Type();
        if (currentElemType != i1Type) {
          auto boolType = RankedTensorType::get(currentType.getShape(), i1Type);
          current = rewriter.create<tosa::CastOp>(loc, boolType, current);
        }
        auto intermediateType = RankedTensorType::get(resultShape, i1Type);
        current = rewriter.create<tosa::ReduceAnyOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      }
    }

    // Handle MEAN: divide by product of reduced dimensions
    if (kind == nova::ReductionKind::MEAN) {
      auto currentType = cast<RankedTensorType>(current.getType());

      // Create scalar divisor with shape [1, 1, ...] matching current rank
      SmallVector<int64_t> divisorShape(currentType.getRank(), 1);
      auto divisorType = RankedTensorType::get(divisorShape, elemType);

      Value divisor;
      if (llvm::isa<FloatType>(elemType)) {
        auto attr = DenseElementsAttr::get(
            divisorType,
            rewriter.getFloatAttr(elemType, (double)totalReducedElements));
        divisor = rewriter.create<tosa::ConstOp>(loc, divisorType, attr);
      } else {
        auto attr = DenseElementsAttr::get(
            divisorType,
            rewriter.getIntegerAttr(elemType, totalReducedElements));
        divisor = rewriter.create<tosa::ConstOp>(loc, divisorType, attr);
      }

      // Use tosa.reciprocal + tosa.mul instead of division for better
      // performance
      auto reciprocal =
          rewriter.create<tosa::ReciprocalOp>(loc, divisorType, divisor);

      // Create shift tensor for mul
      auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
      auto shiftAttr =
          DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
      auto shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

      current = rewriter.create<tosa::MulOp>(loc, currentType, current,
                                             reciprocal, shift);
    }

    // Reshape to final result shape if needed
    // If keepdims=false, we need to drop the size-1 dimensions
    auto currentType = cast<RankedTensorType>(current.getType());
    if (currentType.getShape() != resultType.getShape()) {
      auto shapeType = RankedTensorType::get({resultType.getRank()},
                                             rewriter.getIndexType());
      auto shapeAttr =
          DenseIntElementsAttr::get(shapeType, resultType.getShape());
      auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
          loc,
          mlir::tosa::shapeType::get(rewriter.getContext(),
                                     resultType.getRank()),
          shapeAttr);
      current = rewriter.create<tosa::ReshapeOp>(loc, resultType, current,
                                                 shapeConst);
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

struct AdamOpConverter : public OpConversionPattern<::mlir::nova::AdamOp> {
  using OpConversionPattern<::mlir::nova::AdamOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mlir::nova::AdamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value param = adaptor.getParam();
    Value m = adaptor.getM();
    Value v = adaptor.getV();
    Value grad = adaptor.getGrad();

    // Load hyperparameters
    double beta1 = op.getBeta1().convertToDouble();
    double beta2 = op.getBeta2().convertToDouble();
    double epsilon = op.getEpsilon().convertToDouble();
    double lr = op.getLr().convertToDouble();
    int64_t t = op.getT();

    // Team Logic: Compute alpha_eff and bias corrections
    // Note: t is assumed to be the already-incremented step count (1, 2, ...)
    double bias_corr1 = 1.0 - std::pow(beta1, t);
    double bias_corr2 = 1.0 - std::pow(beta2, t);

    // alpha_eff = alpha * sqrt(bias_corr2) / bias_corr1
    double alpha_eff = lr * std::sqrt(bias_corr2) / bias_corr1;
    double sqrt_bias_corr2 = std::sqrt(bias_corr2);

    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    auto elementType = resultType.getElementType();

    // Create separate empty tensors for results
    Value empty_param = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);
    Value empty_m = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                     elementType);
    Value empty_v = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                     elementType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType, resultType, resultType},
        ValueRange{param, m, v, grad},
        ValueRange{empty_param, empty_m, empty_v},
        SmallVector<AffineMap>(
            7, rewriter.getMultiDimIdentityMap(resultType.getRank())),
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value current_p = args[0];
          Value current_m = args[1];
          Value current_v = args[2];
          Value current_g = args[3];

          // Constants based on team implementation
          Value c_beta1 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, beta1));
          Value c_beta2 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, beta2));
          Value c_one_minus_beta1 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, 1.0 - beta1));
          Value c_one_minus_beta2 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, 1.0 - beta2));
          Value c_alpha_eff = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, alpha_eff));
          Value c_epsilon = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, epsilon));
          Value c_sqrt_bias_corr2 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, sqrt_bias_corr2));

          // 1. Update first moment: m = beta1*m + (1-beta1)*grad
          Value m_term1 = b.create<arith::MulFOp>(loc, current_m, c_beta1);
          Value m_term2 =
              b.create<arith::MulFOp>(loc, current_g, c_one_minus_beta1);
          Value m_new = b.create<arith::AddFOp>(loc, m_term1, m_term2);

          // 2. Update second moment: v = beta2*v + (1-beta2)*grad^2
          Value g_sq = b.create<arith::MulFOp>(loc, current_g, current_g);
          Value v_term1 = b.create<arith::MulFOp>(loc, current_v, c_beta2);
          Value v_term2 = b.create<arith::MulFOp>(loc, g_sq, c_one_minus_beta2);
          Value v_new = b.create<arith::AddFOp>(loc, v_term1, v_term2);

          // 3. Update parameter: param -= alpha_eff * m / (sqrt(v) + epsilon *
          // sqrt(bias_corr2))
          Value sqrt_v = b.create<math::SqrtOp>(loc, v_new);
          Value eps_term =
              b.create<arith::MulFOp>(loc, c_epsilon, c_sqrt_bias_corr2);
          Value denom = b.create<arith::AddFOp>(loc, sqrt_v, eps_term);

          Value num = b.create<arith::MulFOp>(loc, c_alpha_eff, m_new);
          Value update = b.create<arith::DivFOp>(loc, num, denom);

          Value p_new = b.create<arith::SubFOp>(loc, current_p, update);

          b.create<linalg::YieldOp>(loc, ValueRange{p_new, m_new, v_new});
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct ReshapeOpConverter : public OpConversionPattern<nova::ReshapeOp> {
  using OpConversionPattern<nova::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto resultType = cast<RankedTensorType>(op.getType());
    Location loc = op.getLoc();

    // Create a constant tensor for the output shape
    auto resultShape = resultType.getShape();
    auto shapeType =
        RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType());

    // Convert shape to attribute
    auto shapeAttr = rewriter.getIndexTensorAttr(resultShape);
    auto shapeConst =
        rewriter.create<arith::ConstantOp>(loc, shapeType, shapeAttr);

    // Create tensor.reshape
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType, input,
                                                   shapeConst);
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

void populateNovaToLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<
  NovaAbsOpLowering,
  NovaToLinalgElementwiseConverter<nova::AddOp>,
  AdamOpConverter,ArgMaxConverter,
  ArgMinConverter,NovaBroadcastInDimOpLowering,NovaConstantToArithConstPattern,
  NovaDivOpLowering,NovaExpOpLowering,
  NovaLogOpLowering,NovaMatmulTransposeAFusionLowering,NovaMatmulOpLowering,
  NovaMaxOpLowering,NovaMinOpLowering,
  NovaMulOpLowering,NovaNegOpLowering,
  NovaPowOpLowering,NovaRandomOpLowering,
  NovaReciprocalOpLowering,ReduceOpConverter,
  NovaReluOpLowering,ReshapeOpConverter,
  NovaScatterAddOpLowering,NovaSqrtOpLowering,
  NovaSquareOpLowering,NovaSubOpLowering,
  NovaTanhOpLowering,NovaToDeviceOpLowering,
  NovaTransposeOpLowering>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// NovaToLinalg Pass (structural ops only)
//===----------------------------------------------------------------------===//

struct NovaToLinalgPass
    : public PassWrapper<NovaToLinalgPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, tensor::TensorDialect,
                arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, gpu::GPUDialect>();
  }

  StringRef getArgument() const final { return "convert-nova-to-linalg-named"; }

  StringRef getDescription() const final {
    return "Lower Nova structural ops (matmul, gather, transpose, etc.) to "
           "Linalg";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();
    ConversionTarget target(*context);

    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::bufferization::BufferizationDialect>();
    target.addIllegalDialect<mlir::nova::NovaDialect>();

    // Only the structural ops this file has patterns for are illegal.
    target.addIllegalOp<nova::AbsOp>();
    target.addIllegalOp<nova::AddOp>();
    target.addIllegalOp<nova::AdamOp>();
    target.addIllegalOp<nova::ArgmaxOp>();
    target.addIllegalOp<nova::ArgMinOp>();
    target.addIllegalOp<nova::BroadcastInDimOp>();
    target.addIllegalOp<nova::ConstantOp>();
    target.addIllegalOp<nova::DivOp>();
    target.addIllegalOp<nova::ExpOp>();
    target.addIllegalOp<nova::LogOp>();
    target.addIllegalOp<nova::MatmulOp>();
    target.addIllegalOp<nova::MaxOp>();
    target.addIllegalOp<nova::MinOp>();
    target.addIllegalOp<nova::MulOp>();
    target.addIllegalOp<nova::NegOp>();
    target.addIllegalOp<nova::PowOp>();
    target.addIllegalOp<nova::ReciprocalOp>();
    target.addIllegalOp<nova::ReduceOp>();
    target.addIllegalOp<nova::ReluOp>();
    target.addIllegalOp<nova::ReshapeOp>();
    target.addIllegalOp<nova::RsqrtOp>();
    target.addIllegalOp<nova::Rndm2DOp>();
    target.addIllegalOp<nova::ScatterAddOp>();
    target.addIllegalOp<nova::SqrtOp>();
    target.addIllegalOp<nova::SquareOp>();
    target.addIllegalOp<nova::SubOp>();
    target.addIllegalOp<nova::TanhOp>();
    target.addIllegalOp<nova::ToDeviceOp>();
    target.addIllegalOp<nova::TransposeOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    populateNovaToLinalgPatterns(patterns);
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

void registerNovaToLinalgNamedPass() { PassRegistration<NovaToLinalgPass>(); }

std::unique_ptr<Pass> createNovaToLinalgNamedPass() {
  return std::make_unique<NovaToLinalgPass>();
}

} // namespace nova
} // namespace mlir