//===- NovaKernelConfig.cpp - GPU matmul config heuristic -----------------===//
//
// Implements the heuristic that picks workgroup tile sizes, MMA intrinsic,
// and promoted operands for linalg contraction ops, then attaches a
// "lowering_config" DictionaryAttr to each op.
//
// The strategy (mirroring IREE's KernelConfig.cpp):
//
// For MMA-capable targets (Volta+):
//   1. Select the best MMA intrinsic for the element types.
//   2. Pick subgroup counts (numSubgroupsM, numSubgroupsN) = (2, 2) giving 4
//      subgroups per workgroup.
//   3. Pick per-subgroup tile counts (subgroupTilesM = subgroupTilesN = 4).
//   4. workgroupTileM = mmaM * numSubgroupsM * subgroupTilesM
//      workgroupTileN = mmaN * numSubgroupsN * subgroupTilesN
//   5. reductionStepK = mmaK * kTilesPerStep (kTilesPerStep = 2 by default).
//   6. Build config dict and attach.
//
// For SIMT fallback (no MMA):
//   Use IREE's SIMT table:
//   [{128,64,8},{32,8,4}], [{32,128,32},{32,8,1}], ...
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaKernelConfig.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "nova-kernel-config"

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// SIMT fallback tile table
//===----------------------------------------------------------------------===//

struct SimtTilePair {
  std::array<int64_t, 3> tileMNK;   // Workgroup tile for M, N, K
  std::array<int64_t, 3> workgroup; // Thread block dim (x, y, z)
};

// Mirrors IREE's getMatmulConfig() table. Listed from largest to smallest;
// we pick the first one whose tile sizes divide the problem dimensions.
static constexpr SimtTilePair kSimtTable[] = {
    {{128,  64,  8}, {16,  8, 1}},
    {{ 32, 128, 32}, {32,  8, 1}},
    {{128,  32, 32}, {16, 16, 1}},
    {{ 16, 256, 32}, {64,  2, 1}},
    {{ 64,  64, 32}, {16,  8, 1}},
    {{ 32,  64,  8}, {16,  4, 1}},
    {{ 16,  64,  4}, {16,  2, 1}},
    {{  1, 128,  8}, {32,  1, 1}},
};

//===----------------------------------------------------------------------===//
// Helper: infer M/N/K dims from contraction op
//===----------------------------------------------------------------------===//

struct MatmulDims {
  int64_t M = -1, N = -1, K = -1;
  bool valid() const { return M > 0 && N > 0 && K > 0; }
};

static MatmulDims inferMatmulDims(linalg::LinalgOp op) {
  auto contractionDims = mlir::linalg::inferContractionDims(op);
  if (failed(contractionDims))
    return {};
  if (contractionDims->m.empty() || contractionDims->n.empty() ||
      contractionDims->k.empty())
    return {};

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  if (ShapedType::isDynamic(bounds[mDim]) ||
      ShapedType::isDynamic(bounds[nDim]) ||
      ShapedType::isDynamic(bounds[kDim]))
    return {};

  return {bounds[mDim], bounds[nDim], bounds[kDim]};
}

//===----------------------------------------------------------------------===//
// MMA-based config selection
//===----------------------------------------------------------------------===//

/// Attempts to build a LoweringConfig using MMA intrinsics from `target`.
/// Returns failure() if no suitable intrinsic found.
static LogicalResult trySetMMAConfig(linalg::LinalgOp matmul,
                                     const NVIDIATargetInfo &target,
                                     const MatmulDims &dims,
                                     int numLoops) {
  // Determine element types.
  Type lhsType = getElementTypeOrSelf(matmul.getDpsInputOperand(0)->get());
  Type rhsType = getElementTypeOrSelf(matmul.getDpsInputOperand(1)->get());
  Type accType = getElementTypeOrSelf(matmul.getDpsInitOperand(0)->get());

  int32_t lhsKind = typeToElementKind(lhsType);
  int32_t rhsKind = typeToElementKind(rhsType);
  int32_t accKind = typeToElementKind(accType);
  if (lhsKind < 0 || rhsKind < 0 || accKind < 0)
    return failure();

  NVMMAIntrinsicValues intrinsic =
      selectMMAIntrinsic(target, lhsKind, rhsKind, accKind);
  if (intrinsic == NVMMAIntrinsicValues::NONE)
    return failure();

  // Find the intrinsic info to get tile shapes.
  const NVMMAIntrinsicInfo *info = nullptr;
  for (const auto &i : target.mmaIntrinsics) {
    if (i.intrinsic == intrinsic) {
      info = &i;
      break;
    }
  }
  if (!info)
    return failure();

  // Pick subgroup layout: 2×2 subgroups per workgroup, 4 MMA tiles per subgroup.
  // This gives workgroup tile of mmaM*2*4 = 128 (for 16x16 WMMA),
  //                              mmaN*2*4 = 128.
  constexpr int64_t kNumSubgroupsM      = 2;
  constexpr int64_t kNumSubgroupsN      = 2;
  constexpr int64_t kSubgroupTilesM     = 4;
  constexpr int64_t kSubgroupTilesN     = 4;
  constexpr int64_t kKTilesPerStep      = 2;

  int64_t wgM = info->mSize * kNumSubgroupsM * kSubgroupTilesM;
  int64_t wgN = info->nSize * kNumSubgroupsN * kSubgroupTilesN;
  int64_t kStep = info->kSize * kKTilesPerStep;

  // Clamp tile to problem size.
  wgM   = std::min(wgM, dims.M);
  wgN   = std::min(wgN, dims.N);
  kStep = std::min(kStep, dims.K);

  // Ensure thread count (workgroupSize) doesn't exceed 1024.
  // Thread count = (wgM / 8) * (wgN / 8) since thread tiles are 8x8.
  // Target: 256 threads = (16 * 16) for optimal RTX 3060 occupancy.
  while ((wgM / 8) * (wgN / 8) > 1024 && wgN > 8) {
    wgN /= 2;
  }

  // Build per-loop tile size arrays (remaining dims tiled to 1 or 0).
  auto contractionDims = mlir::linalg::inferContractionDims(matmul);
  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  // Tile all outer M dims to 1 (inner = wgM).
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTiles[m] = 1;
    threadTiles[m] = 1;
    subgroupTiles[m] = 1;
  }
  // Tile all outer N dims to 1 (inner = wgN).
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTiles[n] = 1;
    threadTiles[n] = 1;
    subgroupTiles[n] = 1;
  }
  // Tile all outer K dims to 1 (inner = kStep).
  for (int64_t k : llvm::drop_end(contractionDims->k))
    reductionTiles[k] = 1;
  // Tile batch dims to 1.
  for (int64_t b : contractionDims->batch) {
    workgroupTiles[b] = 1;
    threadTiles[b] = 1;
    subgroupTiles[b] = 1;
  }

  workgroupTiles[contractionDims->m.back()] = wgM;
  workgroupTiles[contractionDims->n.back()] = wgN;
  reductionTiles[contractionDims->k.back()] = kStep;
  // Thread tile = 8: gives (wgM/8) * (wgN/8) = 16*16 = 256 threads for 128x128.
  threadTiles[contractionDims->m.back()] = 8;
  threadTiles[contractionDims->n.back()] = 8;
  subgroupTiles[contractionDims->m.back()] = 16;
  subgroupTiles[contractionDims->n.back()] = 16;

  LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] MMA config: "
                           << "wgM=" << wgM << " wgN=" << wgN
                           << " kStep=" << kStep
                           << " intrinsic=" << (int)intrinsic << "\n");

  MLIRContext *ctx = matmul.getContext();
  // Promoted operands: always inputs 0 (A) and 1 (B).
  SmallVector<int64_t> promotedOps = {0, 1};
  setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(intrinsic),
                               promotedOps);
  return success();
}

//===----------------------------------------------------------------------===//
// SIMT fallback config selection
//===----------------------------------------------------------------------===//

static void setSimtConfig(linalg::LinalgOp matmul,
                          const MatmulDims &dims,
                          int numLoops) {
  // Pick first table entry whose M,N tile divides the problem dims.
  const SimtTilePair *chosen = nullptr;
  for (const auto &entry : kSimtTable) {
    if (dims.M % entry.tileMNK[0] == 0 &&
        dims.N % entry.tileMNK[1] == 0) {
      chosen = &entry;
      break;
    }
  }

  // If nothing divides, pick the last (smallest) entry.
  if (!chosen)
    chosen = &kSimtTable[std::size(kSimtTable) - 1];

  int64_t wgM = chosen->tileMNK[0];
  int64_t wgN = chosen->tileMNK[1];

  // Ensure thread count stays <= 1024.
  // Thread tiles are 8x8 below, targeting 256 threads per block.
  while ((wgM / 8) * (wgN / 8) > 1024 && wgN > 8) {
    wgN /= 2;
  }

  auto contractionDims = mlir::linalg::inferContractionDims(matmul);

  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  for (int64_t b : contractionDims->batch) {
    workgroupTiles[b] = 1;
    threadTiles[b] = 1;
    subgroupTiles[b] = 1;
  }
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTiles[m] = 1;
    threadTiles[m] = 1;
    subgroupTiles[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTiles[n] = 1;
    threadTiles[n] = 1;
    subgroupTiles[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims->k)) reductionTiles[k] = 1;

  workgroupTiles[contractionDims->m.back()] = wgM;
  workgroupTiles[contractionDims->n.back()] = wgN;
  reductionTiles[contractionDims->k.back()] = chosen->tileMNK[2];
  // Thread tile = 8: gives (wgM/8) * (wgN/8) threads, targeting 256 per block.
  threadTiles[contractionDims->m.back()] = 8;
  threadTiles[contractionDims->n.back()] = 8;
  subgroupTiles[contractionDims->m.back()] = 16;
  subgroupTiles[contractionDims->n.back()] = 16;

  LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] SIMT fallback config: "
                           << "M=" << chosen->tileMNK[0]
                           << " N=" << chosen->tileMNK[1]
                           << " K=" << chosen->tileMNK[2] << "\n");

  MLIRContext *ctx = matmul.getContext();
  // No MMA → no promoted operands config (promotion pass uses its own heuristic).
  setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                               /*promotedOperands=*/{0, 1});
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult setMatmulLoweringConfig(linalg::LinalgOp matmul,
                                      const NVIDIATargetInfo &target) {
  MatmulDims dims = inferMatmulDims(matmul);
  if (!dims.valid())
    return failure();

  int numLoops = matmul.getNumLoops();

  // Try MMA-based config first.
  if (!target.mmaIntrinsics.empty()) {
    if (succeeded(trySetMMAConfig(matmul, target, dims, numLoops)))
      return success();
  }

  // Fall back to SIMT tile table.
  setSimtConfig(matmul, dims, numLoops);
  return success();
}

//===----------------------------------------------------------------------===//
// Tile-and-fuse config for non-contraction ops (reductions, elementwise)
//
// Mirrors IREE's setTileAndFuseLoweringConfig from ConfigUtils.cpp.
// For linalg.generic ops with reduction iterators (softmax, layer norm, sum):
//   - Parallel dims → workgroup tiles (distribute across blocks)
//   - Reduction dims → small tiling factor for vectorization
//   - Thread tiles → distribute parallel work across threads in a block
//===----------------------------------------------------------------------===//

/// Returns a small tiling factor for a reduction dimension.
/// Mirrors IREE's getReductionTilingFactor from Utils.cpp.
static int64_t getReductionTilingFactor(int64_t dimSize) {
  if (dimSize <= 0 || ShapedType::isDynamic(dimSize))
    return 1;
  if (dimSize % 4 == 0) return 4;
  if (dimSize % 2 == 0) return 2;
  // Try small prime factors.
  static constexpr int primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  for (int p : primes) {
    if (dimSize % p == 0) return p;
  }
  return 1;
}

LogicalResult setReductionLoweringConfig(linalg::LinalgOp op,
                                         const NVIDIATargetInfo &target) {
  // Must have at least one reduction iterator.
  auto iterTypes = op.getIteratorTypesArray();
  bool hasReduction = llvm::any_of(iterTypes, linalg::isReductionIterator);
  if (!hasReduction)
    return failure();

  int numLoops = op.getNumLoops();
  SmallVector<int64_t> loopBounds = op.getStaticLoopRanges();
  if (loopBounds.size() != static_cast<size_t>(numLoops))
    return failure();

  // Reject if any loop bound is dynamic.
  for (int64_t b : loopBounds) {
    if (ShapedType::isDynamic(b))
      return failure();
  }

  // Collect parallel and reduction dims.
  SmallVector<unsigned> parallelDims, reductionDims;
  for (int i = 0; i < numLoops; ++i) {
    if (linalg::isParallelIterator(iterTypes[i]))
      parallelDims.push_back(i);
    else if (linalg::isReductionIterator(iterTypes[i]))
      reductionDims.push_back(i);
  }

  // --- Workgroup tile sizes (distribute parallel dims across blocks) ---
  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  // Distribute parallel dims: tile innermost parallel dims.
  // Use 128 for the two innermost parallel dims (matching matmul workgroup
  // tile), 1 for batch/outer dims.
  // Target 256 threads per block total.

  // Count active (non-batch) parallel dims to compute per-dim thread count.
  int numActiveParallel = std::min((int)parallelDims.size(), 2);
  // For 2 active dims: 16 threads each → 256 total.
  // For 1 active dim: 256 threads from that dim.
  int64_t perDimThreadTarget = (numActiveParallel >= 2) ? 16 : 256;

  int parallelCount = 0;
  for (int i = parallelDims.size() - 1; i >= 0; --i) {
    unsigned dim = parallelDims[i];
    if (parallelCount < 2) {
      // Clamp to problem size.
      int64_t wgTile = std::min((int64_t)128, loopBounds[dim]);
      workgroupTiles[dim] = wgTile;
      // Thread tile: target perDimThreadTarget threads from this dim.
      int64_t threadTile = std::max((int64_t)1, wgTile / perDimThreadTarget);
      // Ensure thread tile divides workgroup tile.
      while (threadTile > 1 && wgTile % threadTile != 0)
        --threadTile;
      threadTiles[dim] = threadTile;
    } else {
      // Outer/batch dims: tile to 1.
      workgroupTiles[dim] = 1;
      threadTiles[dim] = 1;
    }
    ++parallelCount;
  }

  // --- Reduction tile sizes ---
  for (unsigned dim : reductionDims) {
    reductionTiles[dim] = getReductionTilingFactor(loopBounds[dim]);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-kernel-config] Reduction config for "
                 << op->getName() << ": workgroup=[";
    llvm::interleaveComma(workgroupTiles, llvm::dbgs());
    llvm::dbgs() << "] reduction=[";
    llvm::interleaveComma(reductionTiles, llvm::dbgs());
    llvm::dbgs() << "] thread=[";
    llvm::interleaveComma(threadTiles, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  MLIRContext *ctx = op.getContext();
  setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                               /*promotedOperands=*/{});
  return success();
}

//===----------------------------------------------------------------------===//
// Config for elementwise-only ops (all parallel, no reduction)
//===----------------------------------------------------------------------===//

LogicalResult setElementwiseLoweringConfig(linalg::LinalgOp op,
                                           const NVIDIATargetInfo &target) {
  auto iterTypes = op.getIteratorTypesArray();
  // Must be all-parallel (no reductions).
  if (!llvm::all_of(iterTypes, linalg::isParallelIterator))
    return failure();

  int numLoops = op.getNumLoops();
  SmallVector<int64_t> loopBounds = op.getStaticLoopRanges();
  if (loopBounds.size() != static_cast<size_t>(numLoops))
    return failure();
  for (int64_t b : loopBounds) {
    if (ShapedType::isDynamic(b))
      return failure();
  }

  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  // Target 256 threads per block total.
  int numActiveParallel = std::min(numLoops, 2);
  // For 2 active dims: 16 threads each → 256 total.
  // For 1 active dim: 256 threads from that dim.
  int64_t perDimThreadTarget = (numActiveParallel >= 2) ? 16 : 256;

  int parallelCount = 0;
  for (int i = numLoops - 1; i >= 0; --i) {
    if (parallelCount < 2) {
      int64_t wgTile = std::min((int64_t)128, loopBounds[i]);
      workgroupTiles[i] = wgTile;
      int64_t threadTile = std::max((int64_t)1, wgTile / perDimThreadTarget);
      while (threadTile > 1 && wgTile % threadTile != 0)
        --threadTile;
      threadTiles[i] = threadTile;
    } else {
      workgroupTiles[i] = 1;
      threadTiles[i] = 1;
    }
    ++parallelCount;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-kernel-config] Elementwise config for "
                 << op->getName() << ": workgroup=[";
    llvm::interleaveComma(workgroupTiles, llvm::dbgs());
    llvm::dbgs() << "] thread=[";
    llvm::interleaveComma(threadTiles, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  MLIRContext *ctx = op.getContext();
  setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                               /*promotedOperands=*/{});
  return success();
}


void initNovaGPULaunchConfig(mlir::func::FuncOp funcOp,
                              const NVIDIATargetInfo &target) {
  // Priority-based root operation selection, mirroring IREE's initGPULaunchConfig
  // (KernelConfig.cpp lines 2449-2518):
  //
  //   1. Named contraction ops (matmul, batch_matmul, matmul_transpose_b)
  //   2. linalg.generic with reduction iterators (softmax, layer norm, sum, ...)
  //   3. linalg.generic all-parallel (elementwise ops not fused into a contraction)
  //
  // Each op gets a lowering_config attribute that downstream tiling passes read.

  funcOp.walk([&](linalg::LinalgOp op) {
    // Skip ops that already have a config.
    if (getLoweringConfig(op.getOperation()))
      return;

    // Priority 1: Named contraction ops → matmul config (MMA or SIMT).
    if (linalg::isaContractionOpInterface(op)) {
      (void)setMatmulLoweringConfig(op, target);
      return;
    }

    // Priority 2: Generic ops with reduction iterators.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation())) {
      if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
        (void)setReductionLoweringConfig(op, target);
        return;
      }
    }

    // Priority 3: All-parallel generic ops (standalone elementwise).
    // These usually get fused as epilogues of contractions during workgroup
    // tiling, but standalone ones need a config to be tiled properly.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation())) {
      (void)setElementwiseLoweringConfig(op, target);
      return;
    }
  });
}

} // namespace mlir::nova
