#ifndef NOVA_VECTOR_LAYOUT_ATTR_H
#define NOVA_VECTOR_LAYOUT_ATTR_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::nova {

/// Describes how a vector is distributed across the GPU thread hierarchy.
///
/// Stored as a DictionaryAttr on ops under the key "nova.vector_layout".
/// Replaces IREE's NestedLayoutAttr / ToLayoutOp pair without requiring the
/// VectorExt dialect.
///
/// The five count arrays describe how each logical vector dimension is tiled
/// across the GPU hierarchy (outermost to innermost):
///   subgroupCounts  — partitioning across subgroups (IREE: "subgroup" level)
///   batchCounts     — independent outer batches within each subgroup
///   outerCounts     — outer unroll factor per thread
///   threadCounts    — partitioning across threads within a subgroup
///   elementCounts   — elements per thread (innermost, maps to a register)
///
/// The two stride arrays determine the linear layout within each level:
///   subgroupStrides — stride of each dim in the flat subgroup index space
///   threadStrides   — stride of each dim in the flat thread index space
///
/// sharedMemory=true marks that the value should be routed through LDS
/// (replaces ToLayoutOp's {shared_memory_conversion = true}).
///
/// Example for LHS of batch_matmul [8, 1024, 384] x [8, 384, 1536]:
///   subgroupCounts  = [8, 64, 1]
///   batchCounts     = [1, 1, 24]
///   outerCounts     = [1, 1,  1]
///   threadCounts    = [1, 16,  1]
///   elementCounts   = [1,  1, 16]
///   subgroupStrides = [192, 1, 0]
///   threadStrides   = [0,   1, 0]
///   sharedMemory    = true
struct NovaVectorLayout {
  llvm::SmallVector<int64_t> subgroupCounts;
  llvm::SmallVector<int64_t> batchCounts;
  llvm::SmallVector<int64_t> outerCounts;
  llvm::SmallVector<int64_t> threadCounts;
  llvm::SmallVector<int64_t> elementCounts;
  llvm::SmallVector<int64_t> subgroupStrides;
  llvm::SmallVector<int64_t> threadStrides;
  bool sharedMemory = false;

  /// Serialize to a DictionaryAttr for storage on an op.
  DictionaryAttr toAttr(MLIRContext *ctx) const;

  /// Deserialize from a DictionaryAttr.  Returns failure() if any required
  /// key is missing or has the wrong type.
  static FailureOr<NovaVectorLayout> fromAttr(DictionaryAttr attr);

  /// Project this full-rank layout through an indexing map to produce the
  /// operand-specific layout.  Replaces NestedLayoutAttr::apply(AffineMap).
  ///
  /// @param map      AffineMap from full iteration space to operand dims.
  ///                 Must consist entirely of AffineDimExpr results (simple
  ///                 permutations/projections — no arithmetic expressions).
  /// @param fullRank Number of dims in the full iteration space (= rank of
  ///                 this layout's count arrays).
  NovaVectorLayout apply(AffineMap map, int fullRank) const;

  bool isValid() const { return !subgroupCounts.empty(); }
};

/// Attribute key under which NovaVectorLayout is stored on ops.
static constexpr StringLiteral kNovaLayoutAttr = "nova.vector_layout";

/// Return the NovaVectorLayout stored on @p op, or an invalid (empty) layout
/// if none is present or the stored attribute is malformed.
NovaVectorLayout getVectorLayout(Operation *op);

/// Attach @p layout to @p op, overwriting any previously stored layout.
void setVectorLayout(Operation *op, const NovaVectorLayout &layout);

/// Return true iff @p op has a "nova.vector_layout" attribute.
bool hasVectorLayout(Operation *op);

} // namespace mlir::nova

#endif // NOVA_VECTOR_LAYOUT_ATTR_H
