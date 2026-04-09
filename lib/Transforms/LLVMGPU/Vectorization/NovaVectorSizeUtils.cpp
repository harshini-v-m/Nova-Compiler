//===- NovaVectorSizeUtils.cpp - Vector size inference for dynamic shapes -===//
//
// Ported from IREE's Codegen/Utils/Utils.cpp (inferSizesFromIR family).
//
// The key function is inferNovaVectorSizes(LinalgOp):
//
//   For each iteration dim of the op:
//     1. mapIterationSpaceDimToAllOperandDims → get (operand, dim) pairs.
//     2. Read the static shape from the first operand's type.
//        If static  → use directly.
//        If dynamic → computeNovaVectorDimBound (ValueBounds UB analysis).
//   Returns the complete vector sizes or nullopt if any dim fails.
//
// computeNovaVectorDimBound uses MLIR's ValueBoundsConstraintSet to walk
// the use-def chain (extract_slice sizes, affine.min, scf.for iter_args)
// and return a constant closed upper bound for that dimension.
//
// This lets GenericVectorization handle tensor<1x128x?xf32> GELU tiles:
//   dim 2 (?) → ValueBounds walks to tensor.extract_slice size
//             → size = min(64, 5034 - offset) → UB = 64
//   vectorSizes = [1, 128, 64]
//   linalg::vectorize(rewriter, geluOp, [1,128,64]) → vector.mask { ... }
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaVectorSizeUtils.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::nova;

//===----------------------------------------------------------------------===//
// computeNovaVectorDimBound
//===----------------------------------------------------------------------===//

FailureOr<int64_t> mlir::nova::computeNovaVectorDimBound(Value shapedValue,
                                                          unsigned dimNum) {
  // Use ValueBoundsConstraintSet to compute a closed upper bound.
  // Variable(shapedValue, dimNum) encodes "the size of dim dimNum of
  // shapedValue", which the constraint set resolves by traversing
  // tensor.extract_slice, scf.for iter_args, affine.min, etc.
  return ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB,
      ValueBoundsConstraintSet::Variable(shapedValue,
                                         static_cast<int64_t>(dimNum)),
      /*stopCondition=*/nullptr,
      /*closedUB=*/true);
}

//===----------------------------------------------------------------------===//
// inferNovaVectorSizes
//===----------------------------------------------------------------------===//

std::optional<NovaVectorizationTileSizes>
mlir::nova::inferNovaVectorSizes(linalg::LinalgOp linalgOp) {
  NovaVectorizationTileSizes result;
  unsigned numDims = linalgOp.getNumLoops();

  for (unsigned dim = 0; dim < numDims; ++dim) {
    // Map this iteration dim to (operand, operandDim) pairs.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty())
      return std::nullopt;

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: static shape.
    int64_t dimSize =
        cast<ShapedType>(firstOperand.getType()).getShape()[firstOperandDim];
    if (!ShapedType::isDynamic(dimSize)) {
      result.vectorSizes.push_back(dimSize);
      result.vectorScalableFlags.push_back(false);
      continue;
    }

    // Dynamic dim: try ValueBounds on each (operand, dim) pair until one
    // succeeds (mirrors IREE's loop over operandDimPairs).
    FailureOr<int64_t> maybeBound = failure();
    for (auto [operand, operandDim] : operandDimPairs) {
      maybeBound = computeNovaVectorDimBound(operand, operandDim);
      if (succeeded(maybeBound))
        break;
    }

    if (failed(maybeBound))
      return std::nullopt;

    result.vectorSizes.push_back(*maybeBound);
    result.vectorScalableFlags.push_back(false);
  }

  return result;
}
