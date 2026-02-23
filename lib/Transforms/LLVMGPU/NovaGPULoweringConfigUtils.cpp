//===- NovaGPULoweringConfigUtils.cpp - Lowering Config helpers -----------===//
//
// Implements utilities for reading and writing the Nova GPU lowering
// configuration DictionaryAttr attached to linalg ops.
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Read helpers
//===----------------------------------------------------------------------===//

SmallVector<int64_t> getLoweringConfigTileSizes(DictionaryAttr config,
                                                llvm::StringRef levelKey) {
  if (!config)
    return {};
  auto attr = config.get(levelKey);
  if (!attr)
    return {};
  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr)
    return {};
  SmallVector<int64_t> sizes;
  for (Attribute a : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(a);
    if (!intAttr) return {};
    sizes.push_back(intAttr.getInt());
  }
  return sizes;
}

int32_t getMmaKindRaw(DictionaryAttr config) {
  if (!config)
    return 0;
  auto attr = config.get(kMmaKindKey);
  if (!attr)
    return 0;
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return 0;
  return static_cast<int32_t>(intAttr.getInt());
}

std::optional<SmallVector<int64_t>>
getPromotedOperandList(DictionaryAttr config) {
  if (!config)
    return std::nullopt;
  auto attr = config.get(kPromotedOpsKey);
  if (!attr)
    return std::nullopt;
  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr)
    return std::nullopt;
  SmallVector<int64_t> ops;
  for (Attribute a : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(a);
    if (!intAttr) return std::nullopt;
    ops.push_back(intAttr.getInt());
  }
  return ops;
}

//===----------------------------------------------------------------------===//
// Write helpers
//===----------------------------------------------------------------------===//

void setLoweringConfigTileSizes(MLIRContext *ctx,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                llvm::StringRef levelKey,
                                ArrayRef<int64_t> sizes) {
  Builder b(ctx);
  attrs.emplace_back(StringAttr::get(ctx, levelKey),
                     b.getI64ArrayAttr(sizes));
}

void setMmaKindRaw(MLIRContext *ctx,
                   SmallVectorImpl<NamedAttribute> &attrs,
                   int32_t intrinsicValue) {
  Builder b(ctx);
  attrs.emplace_back(StringAttr::get(ctx, kMmaKindKey),
                     b.getI32IntegerAttr(intrinsicValue));
}

void appendPromotedOperandsList(MLIRContext *ctx,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                ArrayRef<int64_t> operands) {
  Builder b(ctx);
  attrs.emplace_back(StringAttr::get(ctx, kPromotedOpsKey),
                     b.getI64ArrayAttr(operands));
}

//===----------------------------------------------------------------------===//
// Op-level helpers
//===----------------------------------------------------------------------===//

DictionaryAttr getLoweringConfig(Operation *op) {
  if (!op)
    return {};
  auto attr = op->getAttr(kLoweringConfigAttrName);
  if (!attr)
    return {};
  if (auto dict = dyn_cast<DictionaryAttr>(attr))
    return dict;
  // Fallback for wrapped attribute if we start using it.
  // Assuming Nova::LoweringConfigAttr is generated.
  // For now, staying with DictionaryAttr as it's what the test showed.
  return {};
}

void setLoweringConfig(Operation *op, DictionaryAttr configDict) {
  op->setAttr(kLoweringConfigAttrName, configDict);
}

void setMatmulLoweringConfigAttrs(Operation *op,
                                  MLIRContext *ctx,
                                  ArrayRef<int64_t> workgroupTiles,
                                  ArrayRef<int64_t> reductionTiles,
                                  int32_t mmaKindValue,
                                  ArrayRef<int64_t> promotedOperands) {
  SmallVector<NamedAttribute> attrs;
  setLoweringConfigTileSizes(ctx, attrs, kWorkgroupKey, workgroupTiles);
  setLoweringConfigTileSizes(ctx, attrs, kReductionKey, reductionTiles);
  setMmaKindRaw(ctx, attrs, mmaKindValue);
  appendPromotedOperandsList(ctx, attrs, promotedOperands);
  auto configDict = DictionaryAttr::get(ctx, attrs);
  setLoweringConfig(op, configDict);
}

} // namespace mlir::nova
