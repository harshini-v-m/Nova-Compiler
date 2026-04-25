//===- NovaGPULoweringConfigUtils.h - Lowering Config helpers ----*- C++ -*-===//
//
// Utilities for reading and writing the Nova GPU lowering configuration
// dictionary attribute attached to linalg ops.
//
// The config is stored as a DictionaryAttr under the "lowering_config" key
// on each linalg op. The dict uses the following well-known string keys:
//
//   kWorkgroupKey     "workgroup"         — I64ArrayAttr of tile sizes
//   kReductionKey     "reduction"         — I64ArrayAttr of tile sizes
//   kThreadKey        "thread"            — I64ArrayAttr of tile sizes
//   kSubgroupKey      "subgroup"          — I64ArrayAttr of tile sizes
//   kMmaKindKey       "mma_kind"          — I32Attr (NVMMAIntrinsic value)
//   kPromotedOpsKey   "promoted_operands" — I64ArrayAttr of operand indices
//
// Mirrors IREE's GPULoweringConfigUtils.h but self-contained in Nova.
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_NOVAGPULOWERINGCONFIGUTILS_H_
#define NOVA_TRANSFORMS_LLVMGPU_NOVAGPULOWERINGCONFIGUTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir::nova {

// Well-known key names for the LoweringConfig DictionaryAttr.
constexpr llvm::StringLiteral kWorkgroupKey     = "workgroup";
constexpr llvm::StringLiteral kReductionKey     = "reduction";
constexpr llvm::StringLiteral kThreadKey        = "thread";
constexpr llvm::StringLiteral kSubgroupKey      = "subgroup";
constexpr llvm::StringLiteral kWgSubgroupKey    = "wg_subgroup";
constexpr llvm::StringLiteral kMmaKindKey       = "mma_kind";
constexpr llvm::StringLiteral kPromotedOpsKey   = "promoted_operands";
constexpr llvm::StringLiteral kPaddingKey       = "padding";
constexpr llvm::StringLiteral kDerivedThreadKey = "derived_thread";
constexpr llvm::StringLiteral kTargetThreadsKey = "target_threads";
/// Marker attribute placed on linalg.copy ops created by promoteOperandToShared.
/// InferMemorySpace uses this to unconditionally classify the copy destination
/// as workgroup memory, bypassing the isCrossThreadAccess heuristic which cannot
/// detect cross-thread reads when the matmul consumes the copy result as an input
/// (rather than a DPS init).  Survives K-tiling and Thread-tiling because MLIR's
/// tiling infrastructure clones op attributes onto tiled successors.
constexpr llvm::StringLiteral kPromoteToWorkgroupAttr = "nova.promote_to_workgroup";

// Attribute name attached to linalg ops for the config dict.
constexpr llvm::StringLiteral kLoweringConfigAttrName = "lowering_config";

//===----------------------------------------------------------------------===//
// Config dict getters / setters
//===----------------------------------------------------------------------===//

/// Reads a tiling level (e.g., "workgroup" or "reduction") from a
/// LoweringConfig dict. Returns an empty vector if the key is absent.
SmallVector<int64_t> getLoweringConfigTileSizes(DictionaryAttr config,
                                                llvm::StringRef levelKey);

/// Reads the mma_kind integer from a LoweringConfig dict.
/// Returns 0 (NVMMAIntrinsic::NONE) if the key is absent.
int32_t getMmaKindRaw(DictionaryAttr config);

/// Append a tiling level array to an in-progress attrs list.
void setLoweringConfigTileSizes(MLIRContext *ctx,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                llvm::StringRef levelKey,
                                ArrayRef<int64_t> sizes);

/// Append the mma_kind integer to an in-progress attrs list.
void setMmaKindRaw(MLIRContext *ctx,
                   SmallVectorImpl<NamedAttribute> &attrs,
                   int32_t intrinsicValue);

/// Append promoted operand indices to an in-progress attrs list.
void appendPromotedOperandsList(MLIRContext *ctx,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                ArrayRef<int64_t> operands);

/// Read promoted operand indices from a LoweringConfig dict.
/// Returns std::nullopt if the key is absent.
std::optional<SmallVector<int64_t>>
getPromotedOperandList(DictionaryAttr config);

/// Read padding sizes from a LoweringConfig dict.
/// Returns std::nullopt if the key is absent.
std::optional<SmallVector<int64_t>>
getPaddingList(DictionaryAttr config);

/// Append padding sizes to an in-progress attrs list.
void appendPaddingList(MLIRContext *ctx,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       ArrayRef<int64_t> padding);

//===----------------------------------------------------------------------===//
// Op-level helpers
//===----------------------------------------------------------------------===//

/// Returns true if the config has the "derived_thread" marker, meaning
/// thread tile sizes should be computed at tiling time from the op's
/// loop ranges and the stored target_threads count.
bool isDerivedThreadConfig(DictionaryAttr config);

/// Reads the target thread count from a derived-thread config.
/// Returns 0 if the key is absent.
int64_t getTargetThreadCount(DictionaryAttr config);

/// Computes thread tile sizes for a copy op so that the resulting
/// scf.forall trip count equals exactly `targetThreads`.
/// `loopRanges` are the op's static loop ranges (after K-tiling).
/// `elemBitWidth` is used to pick the vectorization width (128-bit loads).
SmallVector<int64_t> deriveThreadTileSizes(ArrayRef<int64_t> loopRanges,
                                           int64_t targetThreads,
                                           unsigned elemBitWidth);

/// Retrieve the lowering_config DictionaryAttr from an op, if present.
DictionaryAttr getLoweringConfig(Operation *op);

/// Attach a lowering_config DictionaryAttr to an op.
/// Replaces any existing "lowering_config" attribute.
void setLoweringConfig(Operation *op, DictionaryAttr configDict);

/// Build and attach a LoweringConfig from components.
/// Convenience wrapper around the above two functions.
void setMatmulLoweringConfigAttrs(Operation *op,
                                  MLIRContext *ctx,
                                  ArrayRef<int64_t> workgroupTiles,
                                  ArrayRef<int64_t> reductionTiles,
                                  ArrayRef<int64_t> threadTiles,
                                  ArrayRef<int64_t> subgroupTiles,
                                  int32_t mmaKindValue,
                                  ArrayRef<int64_t> promotedOperands,
                                  ArrayRef<int64_t> paddingSizes = {},
                                  ArrayRef<int64_t> wgSubgroupTiles = {});

/// Remove the lowering_config attribute from an op.
/// Used to strip configs from non-root ops after they fuse into a root's
/// scf.forall (defense-in-depth against stale configs at thread tiling).
void removeLoweringConfig(Operation *op);

/// Returns true if the op represents a matmul-like contraction, even if it is
/// a linalg.generic that doesn't strictly implement ContractionOpInterface.
/// Broadened to recognize patterns with multiple batch dimensions.
bool isTrueContraction(Operation *op);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_NOVAGPULOWERINGCONFIGUTILS_H_
