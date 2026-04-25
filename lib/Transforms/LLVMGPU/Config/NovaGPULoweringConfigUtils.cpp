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
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
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

SmallVector<int64_t, 3> getMMAShape(int32_t mmaKindRaw) {
  switch (static_cast<NVMMAIntrinsicValues>(mmaKindRaw)) {
  case NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16:
  case NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16:
    return {16, 8, 16};
  case NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8:
    return {16, 8, 8};
  case NVMMAIntrinsicValues::WMMA_TF32_16x16x8:
    return {16, 16, 8};
  case NVMMAIntrinsicValues::WMMA_F32_16x16x16:
  case NVMMAIntrinsicValues::WMMA_F16_16x16x16:
    return {16, 16, 16};
  case NVMMAIntrinsicValues::NONE:
  default:
    // Default fallback for NVIDIA Tensor Cores if no specific kind matched.
    return {16, 16, 16};
  }
}

SmallVector<int64_t> getConfigField(DictionaryAttr config,
                                    llvm::StringRef levelKey) {
  return getLoweringConfigTileSizes(config, levelKey);
}

SmallVector<int64_t> getPromotedOperands(DictionaryAttr config) {
  auto optList = getPromotedOperandList(config);
  return optList.has_value() ? *optList : SmallVector<int64_t>{};
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

int64_t getBlockDim(DictionaryAttr config) {
  if (!config)
    return 0;
  auto attr = config.get(kBlockDimKey);
  if (!attr)
    return 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  return 0;
}

SmallVector<int64_t> deriveThreadTileSizes(ArrayRef<int64_t> loopRanges,
                                           int64_t targetThreads,
                                           unsigned elemBitWidth,
                                           int64_t blockDim) {
  int64_t rank = loopRanges.size();
  SmallVector<int64_t> tileSizes(rank, 1);
  if (targetThreads <= 0 || rank == 0)
    return tileSizes;

  // Total elements in the copy.
  int64_t flatTrips = 1;
  for (int64_t r : loopRanges)
    flatTrips *= r;

  // Max vector width from 128-bit loads (e.g., 4 for f32, 8 for f16).
  // For cp.async, this is also the max elements per single instruction
  // (cp.async.16 = 16 bytes = 4 f32 = 8 f16).
  int64_t maxVec = std::max<int64_t>(1, 128 / elemBitWidth);

  // When blockDim is known, cap targetThreads so that each thread gets at
  // least maxVec elements — this ensures innermost tile == maxVec, giving a
  // single cp.async.16 instruction per row rather than a scalar ld/st loop.
  // Example: Copy A [1,64,64], blockDim=256, maxVec=4 (f32)
  //   uncapped: targetThreads=128 → perThread=32 → tile=[1,8,4] (8 cp.async)
  //   capped:   targetThreads=min(128, 4096/4)=min(128,1024)=128 → same
  //   but perThread clamped to max(maxVec, flatTrips/blockDim)=max(4,16)=16
  //   → targetThreads = 4096/16 = 256 → tile=[1,4,4] (4 cp.async per thread)
  //
  // The key invariant: innermost tile is always exactly maxVec.  Outer tiles
  // absorb any remaining perThread work so ConvertMemRefToGpu can emit one
  // cp.async per row with dstElements=maxVec, iterating over outer dims.
  int64_t effectiveThreads = targetThreads;
  if (blockDim > 0) {
    // Minimum perThread so we don't exceed blockDim threads for this copy.
    int64_t minPerThread = (flatTrips + blockDim - 1) / blockDim;
    // Clamp perThread to be at least maxVec (so innermost tile == maxVec)
    // and at least minPerThread (so thread count stays within blockDim).
    int64_t perThreadFloor = std::max(maxVec, minPerThread);
    // Round up to a multiple of maxVec so innermost dim divides cleanly.
    perThreadFloor = ((perThreadFloor + maxVec - 1) / maxVec) * maxVec;
    // Recompute effective thread count from the floored perThread.
    if (flatTrips % perThreadFloor == 0)
      effectiveThreads = flatTrips / perThreadFloor;
  }

  // If not evenly divisible, fall back to all-ones (each thread gets 1 elem).
  if (flatTrips % effectiveThreads != 0)
    return tileSizes;

  // Per-thread work = total elements / effective thread count.
  int64_t perThread = flatTrips / effectiveThreads;

  // Prefer perThread >= maxVec so the innermost tile reaches maxVec (128-bit
  // load width).  If the current perThread is smaller than maxVec, reduce
  // effectiveThreads until perThread hits maxVec — as long as the resulting
  // thread count stays within [warpSize, targetThreads] and divides evenly.
  // Example: loopRanges=[1,4,128], target=256, maxVec=4 →
  //   perThread=2 < 4 → try effectiveThreads=128 → perThread=4=maxVec ✓
  if (perThread < maxVec) {
    constexpr int64_t kWarpSize = 32;
    int64_t preferred = flatTrips / maxVec;
    if (preferred >= kWarpSize && preferred <= targetThreads &&
        flatTrips % preferred == 0) {
      effectiveThreads = preferred;
      perThread = maxVec;
    }
  }

  // Distribute per-thread work across dims, innermost first.
  // Innermost dim is capped at maxVec for a single cp.async.16 instruction.
  // Remaining per-thread work is spread into outer dims (these become the
  // loop bounds in ConvertMemRefToGpu's generated scf.for).
  int64_t remaining = perThread;
  for (int64_t d = rank - 1; d >= 0 && remaining > 1; --d) {
    int64_t range = loopRanges[d];
    int64_t maxTile = (d == rank - 1) ? std::min(maxVec, range) : range;
    int64_t tile = std::min(maxTile, remaining);
    while (tile > 1 && (range % tile != 0 || remaining % tile != 0))
      --tile;
    tileSizes[d] = tile;
    remaining /= tile;
  }

  // Verify: product(loopRanges) / product(tileSizes) == effectiveThreads.
  int64_t actualThreads = 1;
  for (int64_t d = 0; d < rank; ++d)
    actualThreads *= (loopRanges[d] / tileSizes[d]);

  if (actualThreads != effectiveThreads) {
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


//===----------------------------------------------------------------------===//
// MMA single-subgroup layout table
//
// PTX ISA data for each Nova MMA intrinsic.
// Operand index: 0=LHS(A), 1=RHS(B), 2=ACC(C/D)
//
// Each entry: outer={outerDim, innerDim}, thread={outerDim, innerDim},
//             tstrides={outerDim, innerDim}, element={outerDim, innerDim}
//
// For mma.sync m16n8k16 / m16n8k8: outerDim=M, innerDim=K for LHS; K/N for RHS; M/N for ACC.
// Sources: PTX ISA, NVIDIA CUDA Programming Guide,
//          IREE IREEGPUAttrs.cpp getSingleSubgroupLayout().
//===----------------------------------------------------------------------===//

NovaMMASingleSubgroupLayout getNovaSubgroupLayout(int32_t mmaKind,
                                                  int operandIndex) {
  using V = NVMMAIntrinsicValues;
  auto kind = static_cast<V>(mmaKind);

  switch (kind) {
  // ── mma.sync m16n8k16 (F16 and BF16 — identical thread layout) ───────────
  case V::MMA_SYNC_F16_16x8x16:
  case V::MMA_SYNC_BF16_16x8x16:
    switch (operandIndex) {
    case 0: // LHS A [M=16, K=16]
      return {/*outer=*/{2, 2}, /*thread=*/{8, 4}, /*tstrides=*/{4, 1},
              /*element=*/{1, 2}};
    case 1: // RHS B [K=16, N=8]
      return {/*outer=*/{2, 1}, /*thread=*/{4, 8}, /*tstrides=*/{1, 4},
              /*element=*/{2, 1}};
    case 2: // ACC C [M=16, N=8]
      return {/*outer=*/{2, 1}, /*thread=*/{8, 4}, /*tstrides=*/{4, 1},
              /*element=*/{1, 2}};
    default: return {};
    }

  // ── mma.sync m16n8k8 (TF32) ──────────────────────────────────────────────
  case V::MMA_SYNC_TF32_16x8x8:
    switch (operandIndex) {
    case 0: // LHS A [M=16, K=8]
      return {/*outer=*/{2, 1}, /*thread=*/{8, 4}, /*tstrides=*/{4, 1},
              /*element=*/{1, 2}};
    case 1: // RHS B [K=8, N=8]
      return {/*outer=*/{1, 1}, /*thread=*/{4, 8}, /*tstrides=*/{1, 4},
              /*element=*/{2, 1}};
    case 2: // ACC C [M=16, N=8]
      return {/*outer=*/{2, 1}, /*thread=*/{8, 4}, /*tstrides=*/{4, 1},
              /*element=*/{1, 2}};
    default: return {};
    }

  // ── wmma m16n16k16 (Volta/Turing F16/F32) ────────────────────────────────
  case V::WMMA_F32_16x16x16:
  case V::WMMA_F16_16x16x16:
    switch (operandIndex) {
    case 0: // LHS A [M=16, K=16]
      return {/*outer=*/{1, 1}, /*thread=*/{16, 2}, /*tstrides=*/{2, 1},
              /*element=*/{1, 8}};
    case 1: // RHS B [K=16, N=16]
      return {/*outer=*/{1, 1}, /*thread=*/{2, 16}, /*tstrides=*/{1, 2},
              /*element=*/{8, 1}};
    case 2: // ACC C [M=16, N=16]
      return {/*outer=*/{1, 1}, /*thread=*/{8, 4}, /*tstrides=*/{4, 1},
              /*element=*/{2, 4}};
    default: return {};
    }

  // ── wmma m16n16k8 (Volta/Turing TF32) ────────────────────────────────────
  case V::WMMA_TF32_16x16x8:
    switch (operandIndex) {
    case 0: // LHS A [M=16, K=8]
      return {/*outer=*/{1, 1}, /*thread=*/{16, 1}, /*tstrides=*/{1, 0},
              /*element=*/{1, 8}};
    case 1: // RHS B [K=8, N=16]
      return {/*outer=*/{1, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{8, 1}};
    case 2: // ACC C [M=16, N=16]
      return {/*outer=*/{1, 1}, /*thread=*/{8, 4}, /*tstrides=*/{4, 1},
              /*element=*/{2, 4}};
    default: return {};
    }

  default:
    return {};
  }
}

NovaMNKShape getNovaMMKShape(int32_t mmaKind) {
  using V = NVMMAIntrinsicValues;
  switch (static_cast<V>(mmaKind)) {
  case V::MMA_SYNC_F16_16x8x16:
  case V::MMA_SYNC_BF16_16x8x16:
    return {16, 8, 16};
  case V::MMA_SYNC_TF32_16x8x8:
    return {16, 8, 8};
  case V::WMMA_F32_16x16x16:
  case V::WMMA_F16_16x16x16:
    return {16, 16, 16};
  case V::WMMA_TF32_16x16x8:
    return {16, 16, 8};
  default:
    return {0, 0, 0};
  }
}

} // namespace mlir::nova
