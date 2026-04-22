#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::nova::vec_ext;

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

LogicalResult ToLayoutOp::verify() {
  return getLayout().isValidLayout(getInput().getType(), getLoc());
}

//===----------------------------------------------------------------------===//
// ToSIMDOp / ToSIMTOp folder
//===----------------------------------------------------------------------===//

// to_simd(to_simt(x)) -> x
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>())
    return simtOp.getOperand();
  return {};
}

// to_simt(to_simd(x)) -> x
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>())
    return simdOp.getOperand();
  return {};
}

#define GET_OP_CLASSES
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtOps.cpp.inc"
