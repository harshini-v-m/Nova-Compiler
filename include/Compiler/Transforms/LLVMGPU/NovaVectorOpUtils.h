//===- NovaVectorOpUtils.h - Vector contraction op analysis ---------------===//
//
// Utilities for analyzing vector.contract indexing maps to extract MNK dims.
//
// VectorContractOpInfo mirrors IREE's VectorContractOpInfo and is used by
// distribution patterns that need to map iteration-space dims to operand dims.
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_NOVA_VECTOR_OP_UTILS_H
#define NOVA_TRANSFORMS_LLVMGPU_NOVA_VECTOR_OP_UTILS_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::nova {

/// Captures the M/N/K dimension positions for a vector.contract.
/// Inferred from the contract's indexing maps via linalg::inferContractionDims.
struct VectorContractOpInfo {
  SmallVector<unsigned> lhsMDims;
  SmallVector<unsigned> rhsNDims;
  SmallVector<unsigned> lhsKDim;
  SmallVector<unsigned> rhsKDim;
  SmallVector<unsigned> outMDims;
  SmallVector<unsigned> outNDims;

  SmallVector<unsigned> lhsUnitDims;
  SmallVector<unsigned> rhsUnitDims;
  SmallVector<unsigned> accUnitDims;

  linalg::ContractionDimensions contractionDims;

  std::pair<int, int> getOperandMNIndex() const;
  std::pair<int, int> getOperandKIndex() const;
  std::pair<int, int> getResultMNIndex() const;

  static FailureOr<VectorContractOpInfo>
  inferFromIndexingMaps(ArrayRef<AffineMap> maps);
};

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_NOVA_VECTOR_OP_UTILS_H
