//===- NovaVectorSizeUtils.h - Vector size inference for dynamic shapes ---===//
//
// Utilities for inferring vector sizes from IR for use in masked vectorization.
//
// Ported from IREE's Codegen/Utils/Utils.h (inferSizesFromIR family).
// Used by GenericVectorization to handle ops with dynamic shapes (e.g. GELU
// on tensor<1x128x?xf32> boundary tiles) via vector.mask-based vectorization.
//
// Key entry point: inferNovaVectorSizes(linalg::LinalgOp)
//   - For each iteration dim: use static shape if known, else ValueBounds UB.
//   - Returns nullopt if any dim bound cannot be determined.
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_TILINGPASSES_NOVAVECTORSIZEUTILS_H_
#define NOVA_TRANSFORMS_LLVMGPU_TILINGPASSES_NOVAVECTORSIZEUTILS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::nova {

/// Holds the inferred vector sizes and scalable flags for a LinalgOp.
/// destShape mirrors IREE's VectorizationTileSizes.destShape — only populated
/// when inferring from an OpResult (pack/unpack). For LinalgOp inference we
/// only need vectorSizes.
struct NovaVectorizationTileSizes {
  SmallVector<int64_t> vectorSizes;
  SmallVector<bool> vectorScalableFlags; // always false on NVIDIA (no SVE)
};

/// Computes the closed upper bound for dimension `dimNum` of `shapedValue`
/// using ValueBoundsConstraintSet analysis.
///
/// Walks the use-def chain (tensor.extract_slice sizes, scf.for iter_args,
/// affine.min expressions) to derive a constant bound.
///
/// Returns failure() if no constant bound can be determined.
FailureOr<int64_t> computeNovaVectorDimBound(Value shapedValue,
                                              unsigned dimNum);

/// Infers vector sizes for `linalgOp` by inspecting its iteration space.
///
/// For each iteration dimension:
///   - If the corresponding operand dimension is statically known → use it.
///   - If dynamic → call computeNovaVectorDimBound via ValueBounds analysis.
///
/// Returns nullopt if any dimension's bound cannot be determined.
///
/// No scalable dims: NVIDIA targets don't use SVE, so vectorScalableFlags
/// is always all-false.
std::optional<NovaVectorizationTileSizes>
inferNovaVectorSizes(linalg::LinalgOp linalgOp);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_TILINGPASSES_NOVAVECTORSIZEUTILS_H_
