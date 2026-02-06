#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
  // rsqrt operation
  static Value mapOpImpl(RsqrtOp op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder) {
    if (isa<FloatType>(resultType))
      return builder->create<math::RsqrtOp>(op.getLoc(), args[0]);
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
      Value one = builder->create<arith::ConstantFloatOp>(
          op.getLoc(), cast<FloatType>(resultType), APFloat(1.0f));
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

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index.
    auto emptyTensorIdx =
        rewriter
            .create<tensor::EmptyOp>(loc, resultTy.getShape(), outElementTy,
                                     dynDims)
            .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(outElementTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running min.
    auto emptyTensorMin =
        rewriter
            .create<tensor::EmptyOp>(loc, resultTy.getShape(), inElementTy,
                                     dynDims)
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
        loc, ArrayRef<Type>({resultTy, resultMinTy}), input,
        ValueRange({filledTensorIdx, filledTensorMin}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
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

    rewriter.replaceOp(argminOp, linalgOp.getResult(0));
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

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index.
    auto emptyTensorIdx =
        rewriter
            .create<tensor::EmptyOp>(loc, resultTy.getShape(), outElementTy,
                                     dynDims)
            .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(outElementTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running max.
    auto emptyTensorMax =
        rewriter
            .create<tensor::EmptyOp>(loc, resultTy.getShape(), inElementTy,
                                     dynDims)
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
        loc, ArrayRef<Type>({resultTy, resultMaxTy}), input,
        ValueRange({filledTensorIdx, filledTensorMax}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
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

    rewriter.replaceOp(argmaxOp, linalgOp.getResult(0));
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
      auto boolType = RankedTensorType::get(inputType.getShape(),
                                            reductionElemType);
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

  // Compute keepdims output shape (reduced dims become 1)
  SmallVector<int64_t> keepdimsShape;
  for (int64_t i = 0; i < rank; ++i) {
    keepdimsShape.push_back(axisSet.contains(i) ? 1 : inputType.getDimSize(i));
  }

  // Create output tensor with keepdims shape 
  auto keepdimsType =
      RankedTensorType::get(keepdimsShape, reductionElemType);
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, keepdimsShape, reductionElemType, ValueRange{});
  Value filledTensor =
      rewriter.create<linalg::FillOp>(loc, identity, emptyTensor).result();

  // Build indexing maps
  // We want iterators: [parallel_0, ..., parallel_M, reduction_0, ...,
  // reduction_K]
  SmallVector<int64_t> parallelAxes;
  SmallVector<int64_t> reductionAxes;
  for (int64_t i = 0; i < rank; ++i) {
    if (axisSet.contains(i))
      reductionAxes.push_back(i);
    else
      parallelAxes.push_back(i);
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

  SmallVector<AffineExpr> outputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    if (axisSet.contains(i)) {
      outputExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
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
      loc, keepdimsType, current, filledTensor,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value result = createReduceCombiner(b, nestedLoc, kind, args[0],
                                            args[1], reductionElemType);
        b.create<linalg::YieldOp>(nestedLoc, result);
      });

  Value reduced = genericOp.getResult(0);

  // Handle MEAN: divide by total reduced elements
  if (kind == nova::ReductionKind::MEAN && isa<FloatType>(reductionElemType)) {
    double divisor = static_cast<double>(totalReducedElements);
    Value divisorVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(reductionElemType, divisor));

    Value divisorTensor = rewriter.create<tensor::EmptyOp>(
        loc, keepdimsShape, reductionElemType, ValueRange{});
    Value filledDivisor =
        rewriter.create<linalg::FillOp>(loc, divisorVal, divisorTensor)
            .result();

    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, keepdimsShape, reductionElemType, ValueRange{});

    auto divOp = rewriter.create<linalg::GenericOp>(
        loc, keepdimsType, ValueRange{reduced, filledDivisor}, outputTensor,
        maps, getNParallelLoopsAttrs(rank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = b.create<arith::DivFOp>(nestedLoc, args[0], args[1]);
          b.create<linalg::YieldOp>(nestedLoc, result);
        });
    reduced = divOp.getResult(0);
  }

  // Final type cast/reshape if needed
  if (cast<RankedTensorType>(reduced.getType()) != resultType) {
    auto reducedType = cast<RankedTensorType>(reduced.getType());
    if (reducedType.getRank() != resultType.getRank()) {
      auto shapeType = RankedTensorType::get({resultType.getRank()},
                                             rewriter.getIndexType());
      auto shapeAttr =
          DenseIntElementsAttr::get(shapeType, resultType.getShape());
      auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
          loc,
          mlir::tosa::shapeType::get(rewriter.getContext(),
                                     resultType.getRank()),
          shapeAttr);
      reduced = rewriter.create<tosa::ReshapeOp>(loc, resultType, reduced,
                                                 shapeConst);
    } else {
      reduced = rewriter.create<tensor::CastOp>(loc, resultType, reduced);
    }
  }

  rewriter.replaceOp(op, reduced);
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

    // Use linalg.generic path 
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
        auto intermediateType = RankedTensorType::get(
            resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceSumOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::MAX: {
        auto intermediateType = RankedTensorType::get(
            resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceMaxOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::MIN: {
        auto intermediateType = RankedTensorType::get(
            resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceMinOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::PRODUCT: {
        auto intermediateType = RankedTensorType::get(
            resultShape, currentElemType);
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
        auto intermediateType =
            RankedTensorType::get(resultShape, i1Type);
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
        auto intermediateType =
            RankedTensorType::get(resultShape, i1Type);
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
    Value empty_m = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);
    Value empty_v = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);

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
    // Create output tensor
    Value out = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultDataType);

    // Prepare affine maps
    int64_t rank = resultType.getRank();
    SmallVector<AffineMap> maps;
    for (Value v : operands) {
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
    maps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Create Linalg generic
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), out.getType(), operands, out, maps,
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

//---------------=-------------------=------------------=---------------------
// Pass Definition

struct NovaToLinalgLoweringPassTemplate
    : public PassWrapper<NovaToLinalgLoweringPassTemplate,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToLinalgLoweringPassTemplate)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, tosa::TosaDialect, func::FuncDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect,
                    scf::SCFDialect>();
  }

  StringRef getArgument() const final { return "convert-nova-to-linalg"; }

  StringRef getDescription() const final {
    return "Lower Nova dialect operations to Linalg dialect";
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
    target.addIllegalDialect<nova::NovaDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    RewritePatternSet patterns(context);
    populateNovaToLinalgPatterns(patterns);
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

std::unique_ptr<Pass> createNovaToLinalgLoweringPass() {
  return std::make_unique<NovaToLinalgLoweringPassTemplate>();
}

void regsiterNovaToLinalgLoweringTemplatePass() {
  PassRegistration<NovaToLinalgLoweringPassTemplate>();
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
               NovaToLinalgElementwiseConverter<TanhOp>,
               NovaToLinalgElementwiseConverter<SqrtOp>,
               NovaToLinalgElementwiseConverter<ReciprocalOp>, ArgMinConverter,
               ArgMaxConverter, ReduceOpConverter, AdamOpConverter,
               ReshapeOpConverter>(patterns.getContext());
}

} // namespace nova
} // namespace mlir