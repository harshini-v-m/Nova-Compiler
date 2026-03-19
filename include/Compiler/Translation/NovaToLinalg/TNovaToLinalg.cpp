#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir {
namespace nova {

// Helper Utilities
inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}
//-----------------------------------------------
// For getting Type builder
//-----------------------------------------------
using namespace mlir;
static mlir::IntegerType getinttype(int bw, OpBuilder *builder) {
  switch (bw) {
  case 64:
    return builder->getI64Type();
  case 32:
    return builder->getI32Type();
  }
  return nullptr;
}

static mlir::FloatType getfloattype(int bw, OpBuilder *builder) {
  switch (bw) {
  case 64:
    return builder->getF64Type();
  case 32:
    return builder->getF32Type();
  }
  return nullptr;
}
static std::optional<arith::CmpIPredicate>
getArithCmpiPredicate(nova::ComparisonType type) {
  switch (type) {
  case nova::ComparisonType::EQ:
    return arith::CmpIPredicate::eq;
  case nova::ComparisonType::NEQ:
    return arith::CmpIPredicate::ne;
  case nova::ComparisonType::LT:
    return arith::CmpIPredicate::slt;
  case nova::ComparisonType::GT:
    return arith::CmpIPredicate::sgt;
  case nova::ComparisonType::LE:
    return arith::CmpIPredicate::sle;
  case nova::ComparisonType::GE:
    return arith::CmpIPredicate::sge;
  }
  return std::nullopt;
}
static std::optional<arith::CmpFPredicate>
getArithCmpfPredicate(nova::ComparisonType type) {
  switch (type) {
  case nova::ComparisonType::EQ:
    return arith::CmpFPredicate::UEQ;
  case nova::ComparisonType::NEQ:
    return arith::CmpFPredicate::UNE;
  case nova::ComparisonType::LT:
    return arith::CmpFPredicate::ULT;
  case nova::ComparisonType::GT:
    return arith::CmpFPredicate::UGT;
  case nova::ComparisonType::LE:
    return arith::CmpFPredicate::ULE;
  case nova::ComparisonType::GE:
    return arith::CmpFPredicate::UGE;
  }
  return std::nullopt;
}
// function to select operation
static Value opdispatcher(nova::CompareOp op, Value lhs, Value rhs,
                          OpBuilder *builder) {
  nova::ComparisonType compareType = op.getKind();
  if (isa<IntegerType>(lhs.getType())) {
    std::optional<arith::CmpIPredicate> arithPred =
        getArithCmpiPredicate(compareType);
    return builder->create<arith::CmpIOp>(op.getLoc(), *arithPred, lhs, rhs);
  }
  if (isa<FloatType>(lhs.getType())) {
    std::optional<arith::CmpFPredicate> arithPred =
        getArithCmpfPredicate(compareType);
    return builder->create<arith::CmpFOp>(op.getLoc(), *arithPred, lhs, rhs);
  }
  return nullptr;
}
// TYPE PROMOTION LOWERING
template <typename top>
static Value
CompareTypePromotionLowering(top op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder) { // need to find parameters
  // 1..fiding dtype
  auto flhstype = dyn_cast<mlir::FloatType>(args[0].getType());
  auto frhstype = dyn_cast<mlir::FloatType>(args[1].getType());
  auto ilhstype = dyn_cast<mlir::IntegerType>(args[0].getType());
  auto irhstype = dyn_cast<mlir::IntegerType>(args[1].getType());
  Value v;
  // checking if lhs and rhs are same
  if (isa<FloatType>(args[0].getType()) && isa<FloatType>(args[1].getType())) {
    // check both bitwidth
    auto lhsbw = flhstype.getWidth();
    auto rhsbw = frhstype.getWidth();
    // selecting bigger one
    if (lhsbw == rhsbw)
      return opdispatcher(op, args[0], args[1], builder);
    else if (lhsbw > rhsbw) {
      v = builder->create<arith::ExtFOp>(op.getLoc(),
                                         getfloattype(lhsbw, builder), args[1]);
      return opdispatcher(op, args[0], v, builder);
    } else {
      v = builder->create<arith::ExtFOp>(op.getLoc(),
                                         getfloattype(rhsbw, builder), args[0]);
      return opdispatcher(op, v, args[1], builder);
    }
  } else if (isa<IntegerType>(args[0].getType()) &&
             isa<FloatType>(args[1].getType())) {

    auto lhsbw = ilhstype.getWidth();
    auto rhsbw = frhstype.getWidth();
    if (lhsbw == rhsbw) {
      v = builder->create<arith::SIToFPOp>(
          op.getLoc(), getfloattype(rhsbw, builder), args[0]);
      return opdispatcher(op, v, args[1], builder);
    } else if (lhsbw > rhsbw) {
      v = builder->create<arith::ExtFOp>(op.getLoc(),
                                         getfloattype(lhsbw, builder), args[1]);
      auto lhs = builder->create<arith::SIToFPOp>(
          op.getLoc(), getfloattype(lhsbw, builder), args[0]);
      return opdispatcher(op, lhs, v, builder);
    } else {
      v = builder->create<arith::ExtSIOp>(op.getLoc(),
                                          getinttype(rhsbw, builder), args[0]);
      auto lhs = builder->create<arith::SIToFPOp>(
          op.getLoc(), getfloattype(rhsbw, builder), v);
      return opdispatcher(op, lhs, args[1], builder);
    }
  }
  // lhs if float and rhs is int
  else if (isa<FloatType>(args[0].getType()) &&
           isa<IntegerType>(args[1].getType())) {
    auto lhsbw = flhstype.getWidth();
    auto rhsbw = irhstype.getWidth();
    if (lhsbw == rhsbw) {
      v = builder->create<arith::SIToFPOp>(
          op.getLoc(), getfloattype(lhsbw, builder), args[1]);
      return opdispatcher(op, args[0], v, builder);
    } else if (lhsbw > rhsbw) {
      v = builder->create<arith::ExtSIOp>(op.getLoc(),
                                          getinttype(lhsbw, builder), args[1]);
      auto rhs = builder->create<arith::SIToFPOp>(
          op.getLoc(), getfloattype(lhsbw, builder), v);
      return opdispatcher(op, args[0], rhs, builder);
    } else {
      v = builder->create<arith::ExtFOp>(op.getLoc(),
                                         getfloattype(rhsbw, builder), args[0]);
      auto rhs = builder->create<arith::SIToFPOp>(
          op.getLoc(), getfloattype(rhsbw, builder), args[1]);
      return opdispatcher(op, v, rhs, builder);
    }
  }

  else if (isa<IntegerType>(args[0].getType()) &&
           isa<IntegerType>(args[1].getType())) {
    auto lhsbw = ilhstype.getWidth();
    auto rhsbw = irhstype.getWidth();
    if (lhsbw == rhsbw) {
      return opdispatcher(op, args[0], args[1], builder);
    }

    else if (lhsbw > rhsbw) {
      v = builder->create<arith::ExtSIOp>(op.getLoc(),
                                          getinttype(lhsbw, builder), args[1]);
      return opdispatcher(op, args[0], v, builder);
    }

    else {
      v = builder->create<arith::ExtSIOp>(op.getLoc(),
                                          getinttype(rhsbw, builder), args[0]);
      return opdispatcher(op, v, args[1], builder);
    }
  } else {
    return opdispatcher(op, args[0], args[1], builder);
  }
}
// Scalar Operation Mapper

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
  }
  // sub operation
  static Value mapOpImpl(nova::SubOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::SubFOp>(op.getLoc(), args[0], args[1]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::SubIOp>(op.getLoc(), args[0], args[1]);

    return nullptr;
  }

  // mul operation
  static Value mapOpImpl(nova::MulOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::MulFOp>(op.getLoc(), args[0], args[1]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::MulIOp>(op.getLoc(), args[0], args[1]);
    return nullptr;
  }
  // div operation
  static Value mapOpImpl(nova::DivOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::DivFOp>(op.getLoc(), args[0], args[1]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::DivSIOp>(op.getLoc(), args[0], args[1]);
    return nullptr;
  }
  // max operation
  static Value mapOpImpl(MaxOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::MaximumFOp>(op.getLoc(), args[0], args[1]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::MaxSIOp>(op.getLoc(), args[0], args[1]);
    return nullptr;
  }
  // min operation
  static Value mapOpImpl(MinOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::MinimumFOp>(op.getLoc(), args[0], args[1]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::MinSIOp>(op.getLoc(), args[0], args[1]);
    return nullptr;
  }
  // abs operation
  static Value mapOpImpl(nova::AbsOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::AbsFOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(resultType))
      return builder->create<math::AbsIOp>(op.getLoc(), args[0]);
    return nullptr;
  }
  // log operation
  static Value mapOpImpl(nova::LogOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::LogOp>(op.getLoc(), args[0]);
    return nullptr;
  }
  // exp operation
  static Value mapOpImpl(nova::ExpOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::ExpOp>(op.getLoc(), args[0]);
    return nullptr;
  }

  // square operation
  static Value mapOpImpl(nova::SquareOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::MulFOp>(op.getLoc(), args[0], args[0]);
    if (isa<IntegerType>(resultType))
      return builder->create<arith::MulIOp>(op.getLoc(), args[0], args[0]);
    return nullptr;
  }
  // neg operation
  static Value mapOpImpl(nova::NegOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<arith::NegFOp>(op.getLoc(), args[0]);
    return nullptr;
  }
  // tanh operation
  static Value mapOpImpl(nova::TanhOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::TanhOp>(op.getLoc(), args[0]);
    return nullptr;
  }
  // rsqrt operation
  static Value mapOpImpl(nova::RsqrtOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::RsqrtOp>(op.getLoc(), args[0]);
    return nullptr;
  }
  // sqrt operation
  static Value mapOpImpl(nova::SqrtOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::SqrtOp>(op.getLoc(), args[0]);
    return nullptr;
  }
  // reciprocal operation
  static Value mapOpImpl(nova::ReciprocalOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(resultType)) {
      auto floatType = cast<FloatType>(resultType);
      Value one = builder->create<arith::ConstantOp>(
          op.getLoc(), floatType, builder->getFloatAttr(floatType, 1.0));
      return builder->create<arith::DivFOp>(op.getLoc(), one, args[0]);
    }
    return nullptr;
  }

  // mod operation
  static Value mapOpImpl(nova::ModOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType)) {
      auto flhstype = dyn_cast<mlir::FloatType>(args[0].getType());
      auto frhstype = dyn_cast<mlir::FloatType>(args[1].getType());
      auto ilhstype = dyn_cast<mlir::IntegerType>(args[0].getType());
      auto irhstype = dyn_cast<mlir::IntegerType>(args[1].getType());
      Value v;
      if (isa<FloatType>(args[0].getType()) &&
          isa<FloatType>(args[1].getType())) {
        // check both bitwidth
        auto lhsbw = flhstype.getWidth();
        auto rhsbw = frhstype.getWidth();
        // selecting bigger one
        if (lhsbw == rhsbw)
          return builder->create<arith::RemFOp>(op.getLoc(), args[0], args[1]);
        else if (lhsbw > rhsbw) {
          v = builder->create<arith::ExtFOp>(
              op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          return builder->create<arith::RemFOp>(op.getLoc(), args[0], v);
        } else {
          v = builder->create<arith::ExtFOp>(
              op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          return builder->create<arith::RemFOp>(op.getLoc(), v, args[1]);
        }
      } else if (isa<IntegerType>(args[0].getType()) &&
                 isa<FloatType>(args[1].getType())) {

        auto lhsbw = ilhstype.getWidth();
        auto rhsbw = frhstype.getWidth();
        if (lhsbw == rhsbw) {
          v = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          return builder->create<arith::RemFOp>(op.getLoc(), v, args[1]);
        } else if (lhsbw > rhsbw) {
          v = builder->create<arith::ExtFOp>(
              op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          auto lhs = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(lhsbw, builder), args[0]);
          return builder->create<arith::RemFOp>(op.getLoc(), lhs, v);
        } else {
          v = builder->create<arith::ExtSIOp>(
              op.getLoc(), getinttype(rhsbw, builder), args[0]);
          auto lhs = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(rhsbw, builder), v);
          return builder->create<arith::RemFOp>(op.getLoc(), lhs, args[1]);
        }
      } else if (isa<FloatType>(args[0].getType()) &&
                 isa<IntegerType>(args[1].getType())) {
        auto lhsbw = flhstype.getWidth();
        auto rhsbw = irhstype.getWidth();
        if (lhsbw == rhsbw) {
          v = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          return builder->create<arith::RemFOp>(op.getLoc(), args[0], v);
        } else if (lhsbw > rhsbw) {
          v = builder->create<arith::ExtSIOp>(
              op.getLoc(), getinttype(lhsbw, builder), args[1]);
          auto rhs = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(lhsbw, builder), v);
          return builder->create<arith::RemFOp>(op.getLoc(), args[0], rhs);
        } else {
          v = builder->create<arith::ExtFOp>(
              op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          auto rhs = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(rhsbw, builder), args[1]);
          return builder->create<arith::RemFOp>(op.getLoc(), v, rhs);
        }
      }

      else if (isa<IntegerType>(args[0].getType()) &&
               isa<IntegerType>(args[1].getType())) {
        auto lhsbw = ilhstype.getWidth();
        auto rhsbw = irhstype.getWidth();
        if (lhsbw == rhsbw) {
          v = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          auto w = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          return builder->create<arith::RemFOp>(op.getLoc(), v, w);
        }

        else if (lhsbw > rhsbw) {
          v = builder->create<arith::ExtSIOp>(
              op.getLoc(), getinttype(lhsbw, builder), args[1]);
          auto r = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(lhsbw, builder), args[0]);
          auto w = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(lhsbw, builder), v);
          return builder->create<arith::RemFOp>(op.getLoc(), r, w);
        }

        else {
          v = builder->create<arith::ExtSIOp>(
              op.getLoc(), getinttype(rhsbw, builder), args[0]);
          auto r = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(rhsbw, builder), v);
          auto w = builder->create<arith::BitcastOp>(
              op.getLoc(), getfloattype(rhsbw, builder), args[1]);
          return builder->create<arith::RemFOp>(op.getLoc(), r, w);
        }
      }
    }
    return nullptr;
  }
  //--------------------------------------------------------
  // EXPONENTS
  //-----------------------------------------------------------

  // exp2 operaton
  static Value mapOpImpl(nova::Exp2Op op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::Exp2Op>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::Exp2Op>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }

  //----------------------------------------------------------------
  // log2 operaton
  static Value mapOpImpl(nova::Log2Op op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::Log2Op>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::Log2Op>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // log10 operaton
  static Value mapOpImpl(nova::Log10Op op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::Log10Op>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::Log10Op>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }

  // tan operation
  static Value mapOpImpl(nova::TanOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::TanOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::TanOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    if (isa<ComplexType>(args[0].getType())) {
      return builder->create<complex::TanOp>(op.getLoc(), args[0]);
    }
    return nullptr;
  }
  // cosop
  static Value mapOpImpl(nova::CosOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::CosOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::CosOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    if (isa<ComplexType>(args[0].getType())) {
      return builder->create<complex::CosOp>(op.getLoc(), args[0]);
    }
    return nullptr;
  }
  // sinop
  static Value mapOpImpl(nova::SinOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::SinOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::SinOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    if (isa<ComplexType>(args[0].getType())) {
      return builder->create<complex::SinOp>(op.getLoc(), args[0]);
    }
    return nullptr;
  }
  // asin operation
  static Value mapOpImpl(nova::AsinOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::AsinOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::AsinOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // acos operation
  static Value mapOpImpl(nova::AcosOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::AcosOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::AcosOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // atan operation
  static Value mapOpImpl(nova::AtanOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::AtanOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::AtanOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));

    return nullptr;
  }
  // sinh operation
  static Value mapOpImpl(nova::SinhOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::SinhOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::SinhOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // cosh operation
  static Value mapOpImpl(nova::CoshOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::CoshOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::CoshOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }

  // asinh operation
  static Value mapOpImpl(nova::AsinhOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::AsinhOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::AsinhOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // acosh operation
  static Value mapOpImpl(nova::AcoshOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::AcoshOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::AcoshOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // atanh operation
  static Value mapOpImpl(nova::AtanhOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    if (isa<FloatType>(args[0].getType()))
      return builder->create<math::AtanhOp>(op.getLoc(), args[0]);
    if (isa<IntegerType>(args[0].getType()))
      return builder->create<math::AtanhOp>(
          op.getLoc(), builder->create<arith::SIToFPOp>(
                           op.getLoc(), builder->getF32Type(), args[0]));
    return nullptr;
  }
  // pow operation
  static Value mapOpImpl(nova::PowOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    // if int use arith.powi else use math.powf
    if (isa<IntegerType>(args[0].getType())) {
      return builder->create<math::IPowIOp>(op.getLoc(), args[0], args[1]);
    } else {
      return builder->create<math::PowFOp>(op.getLoc(), args[0], args[1]);
    }
  }

  // and operation
  static Value mapOpImpl(nova::AndOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    Value lhs = args[0];
    Value rhs = args[1];
    Location loc = op.getLoc();

    if (isa<FloatType>(lhs.getType())) {
      Value zero = builder->create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(lhs.getType()),
          APFloat::getZero(cast<FloatType>(lhs.getType()).getFloatSemantics()));
      lhs = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, lhs,
                                           zero);
    }
    if (isa<FloatType>(rhs.getType())) {
      Value zero = builder->create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(rhs.getType()),
          APFloat::getZero(cast<FloatType>(rhs.getType()).getFloatSemantics()));
      rhs = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, rhs,
                                           zero);
    }

    Value res = builder->create<arith::AndIOp>(loc, lhs, rhs);
    return res;
  }

  // or operation
  static Value mapOpImpl(nova::OrOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    Value lhs = args[0];
    Value rhs = args[1];
    Location loc = op.getLoc();

    if (isa<FloatType>(lhs.getType())) {
      Value zero = builder->create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(lhs.getType()),
          APFloat::getZero(cast<FloatType>(lhs.getType()).getFloatSemantics()));
      lhs = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, lhs,
                                           zero);
    }
    if (isa<FloatType>(rhs.getType())) {
      Value zero = builder->create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(rhs.getType()),
          APFloat::getZero(cast<FloatType>(rhs.getType()).getFloatSemantics()));
      rhs = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, rhs,
                                           zero);
    }

    Value res = builder->create<arith::OrIOp>(loc, lhs, rhs);
    return res;
  }

  // xor operation
  static Value mapOpImpl(nova::XorOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    Value lhs = args[0];
    Value rhs = args[1];
    Location loc = op.getLoc();

    if (isa<FloatType>(lhs.getType())) {
      Value zero = builder->create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(lhs.getType()),
          APFloat::getZero(cast<FloatType>(lhs.getType()).getFloatSemantics()));
      lhs = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, lhs,
                                           zero);
    }
    if (isa<FloatType>(rhs.getType())) {
      Value zero = builder->create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(rhs.getType()),
          APFloat::getZero(cast<FloatType>(rhs.getType()).getFloatSemantics()));
      rhs = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, rhs,
                                           zero);
    }

    Value res = builder->create<arith::XOrIOp>(loc, lhs, rhs);
    return res;
  }

  // not operation
  static Value mapOpImpl(nova::NotOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    Value input = args[0];
    Location loc = op.getLoc();
    Type inputType = input.getType();

    if (auto integerType = dyn_cast<IntegerType>(inputType)) {
      Value zero =
          builder->create<arith::ConstantIntOp>(loc, 0, integerType.getWidth());
      return builder->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            input, zero);

    } else if (auto floatType = dyn_cast<FloatType>(inputType)) {
      APFloat zeroVal(floatType.getFloatSemantics(), 0);
      Value zeroConstant =
          builder->create<arith::ConstantFloatOp>(loc, floatType, zeroVal);
      return builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                            input, zeroConstant);
    }
    return nullptr;
  }
  // Compare operation
  static Value mapOpImpl(nova::CompareOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    // assume example  if compareType is eq of nova dialect them arthpred will
    // be eq of arith dialect
    return CompareTypePromotionLowering(op, resultType, args, builder);
  }

  // sign operation
  static Value mapOpImpl(nova::SignOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    mlir::Value input = args[0];
    mlir::Location loc = op.getLoc();
    if (auto floatType = llvm::dyn_cast<mlir::FloatType>(input.getType())) {
      // Get 1.0 constant of the correct type
      mlir::Value zeroF = builder->create<mlir::arith::ConstantOp>(
          loc, floatType, builder->getFloatAttr(floatType, 0.0));

      mlir::Value greaterThanZero = builder->create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGT, input, zeroF);

      mlir::Value signPos = builder->create<mlir::arith::SIToFPOp>(
          loc, resultType, greaterThanZero);

      mlir::Value lessThanZero = builder->create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OLT, input, zeroF);

      mlir::Value signNeg =
          builder->create<mlir::arith::SIToFPOp>(loc, resultType, lessThanZero);

      return builder->create<mlir::arith::SubFOp>(loc, signPos, signNeg);
    } else if (auto intType =
                   llvm::dyn_cast<mlir::IntegerType>(input.getType())) {

      mlir::Value zero = builder->create<mlir::arith::ConstantOp>(
          loc, intType, builder->getIntegerAttr(intType, 0));
      mlir::Value greaterThanZero = builder->create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, input, zero);
      mlir::Value signPos = builder->create<mlir::arith::SIToFPOp>(
          loc, resultType, greaterThanZero);
      mlir::Value lessThanZero = builder->create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, input, zero);
      mlir::Value signNeg =
          builder->create<mlir::arith::SIToFPOp>(loc, resultType, lessThanZero);

      return builder->create<mlir::arith::SubFOp>(loc, signPos, signNeg);
    }
    return nullptr;
  }
  // Relu backward operation
  static Value mapOpImpl(nova::ReluBackwardOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    Value grad_out = args[0];
    Value input = args[1];
    Location loc = op.getLoc();
    Type elemType = input.getType();

    if (auto floatType = dyn_cast<FloatType>(elemType)) {
      Value zero = builder->create<arith::ConstantOp>(
          loc, builder->getFloatAttr(floatType, 0.0));
      Value mask = builder->create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OGT, input, zero);
      Value maskCast = builder->create<arith::UIToFPOp>(loc, floatType, mask);
      return builder->create<arith::MulFOp>(loc, grad_out, maskCast);
    }
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      Value zero = builder->create<arith::ConstantOp>(
          loc, builder->getIntegerAttr(intType, 0));
      Value mask = builder->create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, input, zero);
      Value maskCast = builder->create<arith::ExtUIOp>(loc, intType, mask);
      return builder->create<arith::MulIOp>(loc, grad_out, maskCast);
    }
    return nullptr;
  }

  // MSE backward operation
  static Value mapOpImpl(nova::MseBackwardOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    Value grad_out = args[0];
    Value pred = args[1];
    Value target = args[2];
    Location loc = op.getLoc();
    Type elemType = pred.getType();

    if (auto floatType = dyn_cast<FloatType>(elemType)) {
      Value two = builder->create<arith::ConstantOp>(
          loc, builder->getFloatAttr(floatType, 2.0));
      auto pred_type = cast<RankedTensorType>(op.getPred().getType());
      int64_t numel = pred_type.getNumElements();
      Value scale = builder->create<arith::ConstantOp>(
          loc, builder->getFloatAttr(floatType, 1.0 / numel));
      
      Value diff = builder->create<arith::SubFOp>(loc, pred, target);
      Value scaled_diff = builder->create<arith::MulFOp>(loc, diff, scale);
      Value scaled_diff_2 = builder->create<arith::MulFOp>(loc, scaled_diff, two);
      return builder->create<arith::MulFOp>(loc, scaled_diff_2, grad_out);
    }
    return nullptr;
  }

  // MAE backward operation
  static Value mapOpImpl(nova::MaeBackwardOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    Value grad_out = args[0];
    Value pred = args[1];
    Value target = args[2];
    Location loc = op.getLoc();
    Type elemType = pred.getType();

    if (auto floatType = dyn_cast<FloatType>(elemType)) {
      auto pred_type = cast<RankedTensorType>(op.getPred().getType());
      int64_t numel = pred_type.getNumElements();
      Value scale = builder->create<arith::ConstantOp>(
          loc, builder->getFloatAttr(floatType, 1.0 / numel));
      
      Value diff = builder->create<arith::SubFOp>(loc, pred, target);
      Value zero = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 0.0));
      Value one = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 1.0));
      Value neg_one = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, -1.0));
      
      Value gt = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, diff, zero);
      Value lt = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, diff, zero);
      
      Value sign = builder->create<arith::SelectOp>(loc, gt, one, 
                     builder->create<arith::SelectOp>(loc, lt, neg_one, zero));
      
      Value dx_pre = builder->create<arith::MulFOp>(loc, sign, scale);
      return builder->create<arith::MulFOp>(loc, dx_pre, grad_out);
    }
    return nullptr;
  }

  // BCE backward operation
  static Value mapOpImpl(nova::BceBackwardOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    Value grad_out = args[0];
    Value pred = args[1];
    Value target = args[2];
    Location loc = op.getLoc();
    Type elemType = pred.getType();

    if (auto floatType = dyn_cast<FloatType>(elemType)) {
      auto pred_type = cast<RankedTensorType>(op.getPred().getType());
      int64_t numel = pred_type.getNumElements();
      Value scale = builder->create<arith::ConstantOp>(
          loc, builder->getFloatAttr(floatType, 1.0 / numel));
      
      Value eps = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 1e-7));
      Value one = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 1.0));
      Value oneminuseps = builder->create<arith::SubFOp>(loc, one, eps);
      
      Value p_clamped_low = builder->create<arith::MaximumFOp>(loc, pred, eps);
      Value p_clipped = builder->create<arith::MinimumFOp>(loc, p_clamped_low, oneminuseps);
      
      Value num = builder->create<arith::SubFOp>(loc, p_clipped, target);
      Value oneminusp = builder->create<arith::SubFOp>(loc, one, p_clipped);
      Value denom = builder->create<arith::MulFOp>(loc, p_clipped, oneminusp);
      Value raw_grad = builder->create<arith::DivFOp>(loc, num, denom);
      
      Value dx_pre = builder->create<arith::MulFOp>(loc, raw_grad, scale);
      return builder->create<arith::MulFOp>(loc, dx_pre, grad_out);
    }
    return nullptr;
  }

  // CCE backward operation
  static Value mapOpImpl(nova::CceBackwardOp op, Type resultType,
                         ArrayRef<Value> args, OpBuilder *builder) {
    Value grad_out = args[0];
    Value pred = args[1];
    Value target = args[2];
    Location loc = op.getLoc();
    Type elemType = pred.getType();

    if (auto floatType = dyn_cast<FloatType>(elemType)) {
      auto pred_type = cast<RankedTensorType>(op.getInput().getType());
      int64_t batch_size = pred_type.getShape()[0];
      Value scale = builder->create<arith::ConstantOp>(
          loc, builder->getFloatAttr(floatType, 1.0 / batch_size));
      
      Value eps = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 1e-7));
      Value one = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 1.0));
      Value oneminuseps = builder->create<arith::SubFOp>(loc, one, eps);
      Value zero = builder->create<arith::ConstantOp>(loc, builder->getFloatAttr(floatType, 0.0));
      
      Value ge_eps = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, pred, eps);
      Value le_ome = builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, pred, oneminuseps);
      Value valid = builder->create<arith::AndIOp>(loc, ge_eps, le_ome);
      
      Value neg_target = builder->create<arith::NegFOp>(loc, target);
      Value raw_grad = builder->create<arith::DivFOp>(loc, neg_target, pred);
      
      Value masked_grad = builder->create<arith::SelectOp>(loc, valid, raw_grad, zero);
      Value dx_pre = builder->create<arith::MulFOp>(loc, masked_grad, scale);
      return builder->create<arith::MulFOp>(loc, dx_pre, grad_out);
    }
    return nullptr;
  }
};

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

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

//---------------=-------------------=------------------=---------------------
// Pass Definition

struct NovaElementwiseToLinalgPass
    : public PassWrapper<NovaElementwiseToLinalgPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaElementwiseToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, tosa::TosaDialect, func::FuncDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect,
                    scf::SCFDialect>();
  }

  StringRef getArgument() const final { return "nova-elementwise-to-linalg"; }

  StringRef getDescription() const final {
    return "Lower Nova elementwise operations to Linalg generic ops";
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
    target.addLegalDialect<tosa::TosaDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::bufferization::BufferizationDialect>();

    // Mark only the elementwise Nova ops that we have patterns for as illegal.
    // Structural ops (nova.reduce, nova.matmul, etc.) remain legal here.
    target.addIllegalOp<ModOp, Exp2Op, Log2Op, Log10Op, TanOp, AsinOp, AcosOp,
                        AtanOp, SinhOp, CoshOp, AsinhOp, AcoshOp, AtanhOp,
                        SinOp, CosOp, CompareOp, NotOp, AndOp, OrOp, XorOp,
                        PowOp, SignOp, AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp,
                        AbsOp, LogOp, ExpOp, SquareOp, NegOp, TanhOp, SqrtOp,
                        ReciprocalOp, nova::RsqrtOp, ReluBackwardOp, 
                        MseBackwardOp, MaeBackwardOp, BceBackwardOp, CceBackwardOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    RewritePatternSet patterns(context);
    populateNovaToLinalgPatternsTemplate(patterns);
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Registration & Pattern Population
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaElementwiseToLinalgPass() {
  return std::make_unique<NovaElementwiseToLinalgPass>();
}

void registerNovaElementwiseToLinalgPass() {
  PassRegistration<NovaElementwiseToLinalgPass>();
}

void populateNovaToLinalgPatternsTemplate(RewritePatternSet &patterns) {
  // Use generic converters for pointwise ops
  patterns.add<NovaToLinalgElementwiseConverter<ModOp>,
               NovaToLinalgElementwiseConverter<Exp2Op>,
               NovaToLinalgElementwiseConverter<Log2Op>,
               NovaToLinalgElementwiseConverter<Log10Op>,
               NovaToLinalgElementwiseConverter<TanOp>,
               NovaToLinalgElementwiseConverter<AsinOp>,
               NovaToLinalgElementwiseConverter<AcosOp>,
               NovaToLinalgElementwiseConverter<AtanOp>,
               NovaToLinalgElementwiseConverter<SinhOp>,
               NovaToLinalgElementwiseConverter<CoshOp>,
               NovaToLinalgElementwiseConverter<AsinhOp>,
               NovaToLinalgElementwiseConverter<AcoshOp>,
               NovaToLinalgElementwiseConverter<AtanhOp>,
               NovaToLinalgElementwiseConverter<SinOp>,
               NovaToLinalgElementwiseConverter<CosOp>,
               NovaToLinalgElementwiseConverter<CompareOp>,
               NovaToLinalgElementwiseConverter<NotOp>,
               NovaToLinalgElementwiseConverter<AndOp>,
               NovaToLinalgElementwiseConverter<OrOp>,
               NovaToLinalgElementwiseConverter<XorOp>,
               NovaToLinalgElementwiseConverter<PowOp>,
               NovaToLinalgElementwiseConverter<SignOp>,
               NovaToLinalgElementwiseConverter<AddOp>,
               NovaToLinalgElementwiseConverter<ReluBackwardOp>,
               NovaToLinalgElementwiseConverter<SubOp>,
               NovaToLinalgElementwiseConverter<MulOp>,
               NovaToLinalgElementwiseConverter<DivOp>,
               NovaToLinalgElementwiseConverter<MaxOp>,
               NovaToLinalgElementwiseConverter<MinOp>,
               NovaToLinalgElementwiseConverter<AbsOp>,
               NovaToLinalgElementwiseConverter<LogOp>,
               NovaToLinalgElementwiseConverter<ExpOp>,
               NovaToLinalgElementwiseConverter<nova::RsqrtOp>,
               NovaToLinalgElementwiseConverter<SquareOp>,
               NovaToLinalgElementwiseConverter<NegOp>,
               NovaToLinalgElementwiseConverter<nova::TanhOp>,
               NovaToLinalgElementwiseConverter<nova::SqrtOp>,
               NovaToLinalgElementwiseConverter<nova::ReciprocalOp>,
               NovaToLinalgElementwiseConverter<MseBackwardOp>,
               NovaToLinalgElementwiseConverter<MaeBackwardOp>,
               NovaToLinalgElementwiseConverter<BceBackwardOp>,
               NovaToLinalgElementwiseConverter<CceBackwardOp>>(
      patterns.getContext());
}

} // namespace nova
} // namespace mlir