//===- NovaGPULoweringConfigUtils.cpp - Lowering config attribute helpers -===//
//
// Implements read and write helpers for the `lowering_config` DictionaryAttr
// attached to linalg ops by the SelectLoweringStrategy pass.
//
// CORE LOGIC: This attribute is the mechanism through which all tiling passes
// (Reduction, Thread, Subgroup) share configuration.  getLoweringConfig() and
// setMatmulLoweringConfigAttrs() are called on virtually every linalg op
// during the pipeline.  Incorrect reads here cause silent wrong-tile-size
// bugs that are very hard to debug at the PTX level.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"

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

std::optional<SmallVector<int64_t>>
getPaddingList(DictionaryAttr config) {
  if (!config)
    return std::nullopt;
  auto attr = config.get(kPaddingKey);
  if (!attr)
    return std::nullopt;
  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr)
    return std::nullopt;
  SmallVector<int64_t> sizes;
  for (Attribute a : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(a);
    if (!intAttr) return std::nullopt;
    sizes.push_back(intAttr.getInt());
  }
  return sizes;
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

void appendPaddingList(MLIRContext *ctx,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       ArrayRef<int64_t> padding) {
  Builder b(ctx);
  attrs.emplace_back(StringAttr::get(ctx, kPaddingKey),
                     b.getI64ArrayAttr(padding));
}

void appendPromotedOperandsList(MLIRContext *ctx,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                ArrayRef<int64_t> operands) {
  Builder b(ctx);
  attrs.emplace_back(StringAttr::get(ctx, kPromotedOpsKey),
                     b.getI64ArrayAttr(operands));
}

//===----------------------------------------------------------------------===//
// Derived thread config helpers
//===----------------------------------------------------------------------===//

bool isDerivedThreadConfig(DictionaryAttr config) {
  if (!config)
    return false;
  auto attr = config.get(kDerivedThreadKey);
  if (!attr)
    return false;
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  return false;
}

int64_t getTargetThreadCount(DictionaryAttr config) {
  if (!config)
    return 0;
  auto attr = config.get(kTargetThreadsKey);
  if (!attr)
    return 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  return 0;
}

SmallVector<int64_t> deriveThreadTileSizes(ArrayRef<int64_t> loopRanges,
                                           int64_t targetThreads,
                                           unsigned elemBitWidth) {
  int64_t rank = loopRanges.size();
  SmallVector<int64_t> tileSizes(rank, 1);
  if (targetThreads <= 0 || rank == 0)
    return tileSizes;

  // Total elements in the copy.
  int64_t flatTrips = 1;
  for (int64_t r : loopRanges)
    flatTrips *= r;

  // If not evenly divisible, fall back to all-ones (each thread gets 1 elem).
  if (flatTrips % targetThreads != 0)
    return tileSizes;

  // Per-thread work = total elements / target thread count.
  int64_t perThread = flatTrips / targetThreads;

  // Max vector width from 128-bit loads (e.g., 4 for f32, 8 for f16).
  int64_t maxVec = std::max<int64_t>(1, 128 / elemBitWidth);

  // Distribute per-thread work across dims, innermost first.
  // Innermost dim gets up to maxVec elements (for coalesced vector loads).
  // Remaining per-thread work is spread to outer dims.
  int64_t remaining = perThread;
  for (int64_t d = rank - 1; d >= 0 && remaining > 1; --d) {
    int64_t range = loopRanges[d];
    // For innermost dim, cap at maxVec for vector load width.
    int64_t maxTile = (d == rank - 1) ? std::min(maxVec, range)
                                      : range;
    // Find largest tile that divides both remaining and range.
    int64_t tile = std::min(maxTile, remaining);
    while (tile > 1 && (range % tile != 0 || remaining % tile != 0))
      --tile;
    tileSizes[d] = tile;
    remaining /= tile;
  }

  // Verify: product(loopRanges) / product(tileSizes) == targetThreads.
  int64_t actualThreads = 1;
  for (int64_t d = 0; d < rank; ++d)
    actualThreads *= (loopRanges[d] / tileSizes[d]);

  if (actualThreads != targetThreads) {
    // Fallback: couldn't achieve exact match, use all-ones.
    // This means trip count = flatTrips, which won't fuse but is safe.
    return SmallVector<int64_t>(rank, 1);
  }

  return tileSizes;
}

//===----------------------------------------------------------------------===//
// Op-level helpers
//===----------------------------------------------------------------------===//

// CORE LOGIC: Reads the lowering_config DictionaryAttr from `op`.
// Returns an empty DictionaryAttr (evaluates to false) when not present.
// All tiling passes call this to determine tile sizes; a missing config
// causes fallback to heuristic values.
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

void removeLoweringConfig(Operation *op) {
  op->removeAttr(kLoweringConfigAttrName);
}

// CORE LOGIC: Builds a complete lowering_config DictionaryAttr and attaches
// it to `op`. Called by NovaKernelConfig for every matmul/reduction op
// during the SelectLoweringStrategy pass. Must match the read layout
// expected by getLoweringConfigTileSizes (direct loop-index mapping).
void setMatmulLoweringConfigAttrs(Operation *op,
                                  MLIRContext *ctx,
                                  ArrayRef<int64_t> workgroupTiles,
                                  ArrayRef<int64_t> reductionTiles,
                                  ArrayRef<int64_t> threadTiles,
                                  ArrayRef<int64_t> subgroupTiles,
                                  int32_t mmaKindValue,
                                  ArrayRef<int64_t> promotedOperands,
                                  ArrayRef<int64_t> paddingSizes,
                                  ArrayRef<int64_t> wgSubgroupTiles) {
  SmallVector<NamedAttribute> attrs;
  setLoweringConfigTileSizes(ctx, attrs, kWorkgroupKey, workgroupTiles);
  setLoweringConfigTileSizes(ctx, attrs, kReductionKey, reductionTiles);
  setLoweringConfigTileSizes(ctx, attrs, kThreadKey, threadTiles);
  setLoweringConfigTileSizes(ctx, attrs, kSubgroupKey, subgroupTiles);
  if (!wgSubgroupTiles.empty())
    setLoweringConfigTileSizes(ctx, attrs, kWgSubgroupKey, wgSubgroupTiles);
  setMmaKindRaw(ctx, attrs, mmaKindValue);
  appendPromotedOperandsList(ctx, attrs, promotedOperands);
  if (!paddingSizes.empty())
    appendPaddingList(ctx, attrs, paddingSizes);
  auto configDict = DictionaryAttr::get(ctx, attrs);
  setLoweringConfig(op, configDict);
}

bool isTrueContraction(Operation *op) {
  if (isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::MatvecOp,
          linalg::VecmatOp, linalg::BatchMatvecOp>(op))
    return true;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;
  if (linalg::isaContractionOpInterface(linalgOp)) return true;
  
  auto indexMaps = linalgOp.getIndexingMapsArray();
  if (indexMaps.size() < 3) return false;
  auto iterTypes = linalgOp.getIteratorTypesArray();
  unsigned numLoops = iterTypes.size();
  bool foundLhsOnly = false;
  bool foundRhsOnly = false;
  for (unsigned i = 0; i < numLoops; ++i) {
    if (iterTypes[i] != utils::IteratorType::parallel) continue;
    bool inInput0 = indexMaps[0].isFunctionOfDim(i);
    bool inInput1 = indexMaps[1].isFunctionOfDim(i);
    // Unique parallel dim for LHS or RHS suggests a contraction pattern.
    if (inInput0 && !inInput1) foundLhsOnly = true;
    if (!inInput0 && inInput1) foundRhsOnly = true;
  }
  return foundLhsOnly && foundRhsOnly;
}

} // namespace mlir::nova
