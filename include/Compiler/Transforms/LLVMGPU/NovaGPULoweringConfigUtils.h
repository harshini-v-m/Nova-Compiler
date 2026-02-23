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
constexpr llvm::StringLiteral kMmaKindKey       = "mma_kind";
constexpr llvm::StringLiteral kPromotedOpsKey   = "promoted_operands";

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

//===----------------------------------------------------------------------===//
// Op-level helpers
//===----------------------------------------------------------------------===//

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
                                  int32_t mmaKindValue,
                                  ArrayRef<int64_t> promotedOperands);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_NOVAGPULOWERINGCONFIGUTILS_H_
