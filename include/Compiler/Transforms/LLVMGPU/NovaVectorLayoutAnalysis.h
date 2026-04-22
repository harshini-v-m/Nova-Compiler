//===- NovaVectorLayoutAnalysis.h - Layout propagation analysis -----------===//
//
// Fixpoint layout propagation for vector distribution.
//
// Seeds from nova_vector_ext.to_layout anchor ops inserted by
// ConfigureTensorLayouts, then propagates NestedLayoutAttr constraints
// forward (value→users) and backward (value→defining op) until stable.
//
// Mirrors IREE's VectorLayoutAnalysis.cpp /
// iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_VECTOR_LAYOUT_ANALYSIS_H
#define NOVA_TRANSFORMS_LLVMGPU_VECTOR_LAYOUT_ANALYSIS_H

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::nova {

/// Propagate layout information through the IR rooted at `root`.
///
/// Anchors are all nova_vector_ext.to_layout ops found in `root`. Their
/// layout attributes seed the analysis. From there, layouts propagate
/// forward (through users) and backward (through defining ops) until
/// no more constraints can be inferred (fixpoint).
///
/// On success, `layouts` maps every vector Value that received a layout to
/// its VectorLayoutInterface. Values with no layout are absent from the map.
mlir::LogicalResult propagateVectorLayoutInfo(
    mlir::Operation *root,
    llvm::MapVector<mlir::Value,
                    mlir::nova::vec_ext::VectorLayoutInterface> &layouts);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_VECTOR_LAYOUT_ANALYSIS_H
