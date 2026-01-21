#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
namespace mlir {
namespace nova {

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
using namespace mlir;
inline bool isScalar(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  return !type || type.getRank() == 0;
}

inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

Value createIntToFloatCast(Value input, Type targetType, OpBuilder &builder,
                           Location loc) {
  Type inputType = input.getType();
  if (auto inputTensorType = dyn_cast<RankedTensorType>(inputType)) {
    if (isa<IntegerType>(inputTensorType.getElementType())) {
      Value out = builder.create<tensor::EmptyOp>(
          loc, inputTensorType.getShape(), targetType, inputTensorType.getEncoding());
      int64_t rank = inputTensorType.getRank();
      AffineMap idMap =
          AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
      SmallVector<AffineMap> maps = {idMap, idMap};
      auto linalgOp = builder.create<linalg::GenericOp>(
          loc, out.getType(), ValueRange{input}, ValueRange{out},
          maps, getNParallelLoopsAttrs(rank),
          [&](OpBuilder &b, Location innerLoc, ValueRange args) {
            Value innerInputScalar = args[0];
            Value inner = b.create<arith::UIToFPOp>(innerLoc, targetType,
                                                    innerInputScalar);
            b.create<linalg::YieldOp>(innerLoc, inner);
          });
      return linalgOp.getResult(0);
    }
  }
  return input;
}

struct NovaOpLinalOp {
  template <typename OpTy>
  static Value mapop(OpTy op, Type resultType, ValueRange args, Value init,
                     OpBuilder &builder) {
    return mapOpImpl(op, resultType, args, init, builder);
  }

private:
  template <typename OpTy>
  static Value mapOpImpl(OpTy op, Type resultType, ValueRange args, Value init,
                         OpBuilder &builder) {
    return nullptr;
  }

  // static Value mapOpImpl(nova::AddOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::AddOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0], args[1]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
  //   static Value mapOpImpl(nova::SubOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::SubOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0], args[1]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
    static Value mapOpImpl(nova::MulOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::MulOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0], args[1]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
      static Value mapOpImpl(nova::DivOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::DivOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0], args[1]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    //     static Value mapOpImpl(nova::PowOp op, Type resultType, ValueRange args,
    //                      Value init, OpBuilder &builder) {
    // Location loc = op.getLoc();
    // RankedTensorType resultTensorType = dyn_cast<RankedTensorType>(resultType);
    
    // if (!resultTensorType) {
    //     op.emitOpError("PowOp expects ranked tensor result type.");
    //     return nullptr;
    // }

//     Type targetFloatType = builder.getF32Type();
//     Value lhs = args[0];
//     Value rhs = args[1];

//     if(isa<IntegerType>(lhs.getType())){
//         lhs = createIntToFloatCast(lhs, targetFloatType, builder, loc);
//     }
//     if(isa<IntegerType>(rhs.getType())){
//         rhs = createIntToFloatCast(rhs, targetFloatType, builder, loc);
//     }

//     auto linalgop = builder.create<linalg::PowFOp>(
//         loc, 
//         resultType,
//         ValueRange{lhs, rhs},
//         ValueRange{init}
//     );

//     return linalgop.getResult(0);
// }
  // static Value mapOpImpl(nova::SquareOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::SquareOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
  //   static Value mapOpImpl(nova::SqrtOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::SqrtOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
  //   static Value mapOpImpl(nova::ReciprocalOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::ReciprocalOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
  //     static Value mapOpImpl(nova::ExpOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::ExpOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
  //   static Value mapOpImpl(nova::LogOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::LogOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
    
  //     static Value mapOpImpl(nova::TanhOp op, Type resultType, ValueRange args,
  //                        Value init, OpBuilder &builder) {
  //   auto linalgop = builder.create<linalg::TanhOp>(
  //       op.getLoc(), 
  //       resultType,
  //       ValueRange{args[0]},
  //       ValueRange{init});

  //   return linalgop.getResult(0);
  // }
static Value mapOpImpl(nova::Rndm2DOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
auto loc=op.getLoc();
// seed= (min+max)^max
    Value sum = builder.create<arith::AddFOp>(loc, args[0], args[1]);
    Value mySeed = builder.create<math::PowFOp>(loc, sum, args[1]);
    Value seedVal = builder.create<arith::FPToSIOp>(loc, builder.getI32Type(), mySeed);

auto linalgop = builder.create<linalg::FillRng2DOp>(
    loc,
    ValueRange{args[0],args[1],seedVal},
    ValueRange{init}
);

    return linalgop.getResult(0);
  }
};

template <typename NovaOpTy>
class NovaToLinalgNamedopsConverter : public OpConversionPattern<NovaOpTy> {
public:
  using OpConversionPattern<NovaOpTy>::OpConversionPattern;
  using OpAdaptor = typename NovaOpTy::Adaptor;

  LogicalResult matchAndRewrite(
      NovaOpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    auto resultType = op.getResult().getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    Value init = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType(), tensorType.getEncoding());

    Value result = NovaOpLinalOp::mapop(op, resultType, operands, init, rewriter);

    if (!result)
      return rewriter.notifyMatchFailure(op, "mapping failed");

    rewriter.replaceOp(op, result);
    return success();
  }
};
void populateNovaToLinalgNamedPatterns(RewritePatternSet &patterns) {
  patterns.add<
            NovaToLinalgNamedopsConverter<nova::MulOp>, 
             NovaToLinalgNamedopsConverter<nova::DivOp>, 
            // NovaToLinalgNamedopsConverter<nova::SqrtOp>,
           //  NovaToLinalgNamedopsConverter<nova::ExpOp>,
            // NovaToLinalgNamedopsConverter<nova::LogOp>,
             //NovaToLinalgNamedopsConverter<nova::TanhOp>,
           //  NovaToLinalgNamedopsConverter<nova::PowOp>,
          //   NovaToLinalgNamedopsConverter<nova::MinOp>,
           //   NovaToLinalgNamedopsConverter<nova::MaxOp>,
           //  NovaToLinalgNamedopsConverter<nova::ReciprocalOp>,
            //NovaToLinalgNamedopsConverter<nova::SquareOp>,
            NovaToLinalgNamedopsConverter<nova::Rndm2DOp> 

  >(patterns.getContext());
}

} 
}