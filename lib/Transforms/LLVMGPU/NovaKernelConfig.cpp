//===- NovaKernelConfig.cpp - GPU kernel config heuristic -----------------===//
//
// Implements the heuristic that picks workgroup tile sizes, thread tiles,
// MMA intrinsic, and promoted operands for linalg compute ops, then attaches
// a "lowering_config" DictionaryAttr to each root op.
//
// Config dispatch (mirroring IREE's KernelConfig.cpp TileAndFuse path):
//
//   1. Contractions (setContractConfig):
//      a. Try MMA-based config (Volta+): 2×2 subgroups, 4 MMA tiles each.
//      b. Fall back to SIMT tile table (8 entries from IREE).
//      Thread tiles = workgroupTile / workgroupSize (like IREE).
//
//   2. Default (setDefaultConfig): For reductions and elementwise ops.
//      Mirrors IREE's setRootDefaultConfig:
//      - workgroupThreads = kTargetThreadsPerBlock = 256
//      - vectorSize = 4 (128-bit loads for f32)
//      - inner parallel dim tile = threads * vectorSize (up to 1024)
//      - outer parallel dims: remaining thread budget distributed inward->outward
//      - reduction dims tile = 4
//      - thread tile for inner dim = vectorSize
//
//   3. initNovaGPULaunchConfig: Single-root model per compute cluster.
//      Walks all linalg ops, finds roots, stamps configs.
//      Root ops = true contractions + any op with a reduction iterator.
//      Non-root ops (prologues, epilogues, fills) fuse into their root.
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaKernelConfig.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "nova-kernel-config"

namespace mlir::nova {

/// Preferred maximum threads per block for this target (Ampere RTX 3060).
/// CUDA's hard limit is 1024; we cap at 256 to leave register file headroom
/// and improve occupancy on the 28-SM GA106 die.
// [Doc5] Changed from kMaxThreadsPerBlock=1024 → kTargetThreadsPerBlock=256
static constexpr int64_t kTargetThreadsPerBlock = 256;

/// Warp size for NVIDIA GPUs.
static constexpr int64_t kWarpSize = 32;

//===----------------------------------------------------------------------===//
// SIMT fallback tile table (from IREE's getMatmulConfig)
//===----------------------------------------------------------------------===//

struct SimtTilePair {
  std::array<int64_t, 3> tileMNK;   // Workgroup tile for M, N, K
  std::array<int64_t, 3> workgroup; // Thread block dim (x, y, z)
};

// Mirrors IREE's getMatmulConfig() table. Listed from largest to smallest;
// we pick the first one whose tile sizes divide the problem dimensions.
static constexpr SimtTilePair kSimtTable[] = {
    {{ 32, 128, 32}, {32,  8, 1}},  // 256 threads
    {{128,  64,  8}, {16,  8, 1}},  // 128 threads
    {{ 16, 256, 32}, {64,  2, 1}},  // 128 threads
    {{  8,  32, 32}, { 8,  8, 1}},  //  64 threads
    {{ 32, 128,  4}, {32,  8, 1}},  // 256 threads
    {{  8, 128,  4}, {32,  1, 1}},  //  32 threads
    {{ 16,  64,  4}, {16,  2, 1}},  //  32 threads
    {{  1, 128,  8}, {32,  1, 1}},  //  32 threads
};

//===----------------------------------------------------------------------===//
// clampThreadTilesToMaxThreads — enforce target thread limit
//===----------------------------------------------------------------------===//

static void clampThreadTilesToMaxThreads(ArrayRef<int64_t> workgroupTiles,
                                         SmallVectorImpl<int64_t> &threadTiles) {
  int n = (int)threadTiles.size();

  auto computeTotal = [&]() -> int64_t {
    int64_t total = 1;
    for (int i = 0; i < n; ++i) {
      if (threadTiles[i] <= 0) continue;
      if (i >= (int)workgroupTiles.size() || workgroupTiles[i] <= 0) continue;
      int64_t trips = (workgroupTiles[i] + threadTiles[i] - 1) / threadTiles[i];
      total *= trips;
    }
    return total;
  };

  // [Doc5] Uses kTargetThreadsPerBlock (256) instead of kMaxThreadsPerBlock (1024)
  while (computeTotal() > kTargetThreadsPerBlock) {
    int worstDim = -1;
    int64_t worstTrips = 0;
    for (int i = 0; i < n; ++i) {
      if (threadTiles[i] <= 0) continue;
      if (i >= (int)workgroupTiles.size() || workgroupTiles[i] <= 0) continue;
      int64_t trips = (workgroupTiles[i] + threadTiles[i] - 1) / threadTiles[i];
      if (trips > worstTrips) {
        worstTrips = trips;
        worstDim = i;
      }
    }
    if (worstDim < 0)
      break;
    threadTiles[worstDim] *= 2;
    if (worstDim < (int)workgroupTiles.size())
      threadTiles[worstDim] = std::min(threadTiles[worstDim],
                                        workgroupTiles[worstDim]);
    if (computeTotal() > kTargetThreadsPerBlock && worstTrips <= 1)
      break;
  }
}

//===----------------------------------------------------------------------===//
// Helper: compute padding sizes from tile config
//===----------------------------------------------------------------------===//

/// Builds padding sizes from workgroup and reduction tile arrays.
/// For parallel dims: pad to workgroupTile (or 1 if untiled).
/// For reduction dims: pad to reductionTile (or 1 if untiled).
/// Mirrors the logic IREE stores in its "padding" config key.
static SmallVector<int64_t>
computePaddingSizes(linalg::LinalgOp op,
                    ArrayRef<int64_t> workgroupTiles,
                    ArrayRef<int64_t> reductionTiles) {
  int numLoops = op.getNumLoops();
  SmallVector<int64_t> padding(numLoops, 1);
  auto iterTypes = op.getIteratorTypesArray();

  for (int i = 0; i < numLoops; ++i) {
    if (linalg::isParallelIterator(iterTypes[i])) {
      int64_t tile = (i < (int)workgroupTiles.size()) ? workgroupTiles[i] : 0;
      if (tile > 0)
        padding[i] = tile;
    } else {
      int64_t tile = (i < (int)reductionTiles.size()) ? reductionTiles[i] : 0;
      if (tile > 0)
        padding[i] = tile;
    }
  }
  return padding;
}

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

static LogicalResult trySetMMAConfig(linalg::LinalgOp matmul,
                                     const NVIDIATargetInfo &target,
                                     const MatmulDims &dims,
                                     int numLoops) {
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

  const NVMMAIntrinsicInfo *info = nullptr;
  for (const auto &i : target.mmaIntrinsics) {
    if (i.intrinsic == intrinsic) {
      info = &i;
      break;
    }
  }
  if (!info)
    return failure();

  // 2x2 subgroups per workgroup, 4 MMA tiles per subgroup.
  constexpr int64_t kNumSubgroupsM  = 2;
  constexpr int64_t kNumSubgroupsN  = 2;
  constexpr int64_t kSubgroupTilesM = 4;
  constexpr int64_t kSubgroupTilesN = 4;
  constexpr int64_t kKTilesPerStep  = 2;

  int64_t wgM = info->mSize * kNumSubgroupsM * kSubgroupTilesM;
  int64_t wgN = info->nSize * kNumSubgroupsN * kSubgroupTilesN;
  int64_t kStep = info->kSize * kKTilesPerStep;

  wgM   = std::min(wgM, dims.M);
  wgN   = std::min(wgN, dims.N);
  kStep = std::min(kStep, dims.K);

  int64_t threadTileM = 8;
  int64_t threadTileN = 8;

  // [Doc5] Uses kTargetThreadsPerBlock (256) instead of kMaxThreadsPerBlock (1024)
  while ((wgM / threadTileM) * (wgN / threadTileN) > kTargetThreadsPerBlock &&
         wgN > threadTileN) {
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
  for (int64_t k : llvm::drop_end(contractionDims->k))
    reductionTiles[k] = 1;

  workgroupTiles[contractionDims->m.back()] = wgM;
  workgroupTiles[contractionDims->n.back()] = wgN;
  reductionTiles[contractionDims->k.back()] = kStep;
  threadTiles[contractionDims->m.back()] = threadTileM;
  threadTiles[contractionDims->n.back()] = threadTileN;
  subgroupTiles[contractionDims->m.back()] = 16;
  subgroupTiles[contractionDims->n.back()] = 16;

  LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] MMA config: "
                           << "wgM=" << wgM << " wgN=" << wgN
                           << " kStep=" << kStep
                           << " intrinsic=" << (int)intrinsic << "\n");

  SmallVector<int64_t> padding =
      computePaddingSizes(matmul, workgroupTiles, reductionTiles);

  MLIRContext *ctx = matmul.getContext();
  SmallVector<int64_t> promotedOps = {0, 1};
  setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(intrinsic),
                               promotedOps, padding);
  return success();
}

//===----------------------------------------------------------------------===//
// SIMT fallback config (mirrors IREE's setContractConfig)
//===----------------------------------------------------------------------===//

static LogicalResult setSimtConfig(linalg::LinalgOp matmul,
                                   const MatmulDims &dims,
                                   int numLoops) {
  const SimtTilePair *chosen = nullptr;
  for (const auto &entry : kSimtTable) {
    if (dims.M % entry.tileMNK[0] == 0 &&
        dims.N % entry.tileMNK[1] == 0 &&
        dims.K % entry.tileMNK[2] == 0) {
      chosen = &entry;
      break;
    }
  }

  if (!chosen) {
    for (const auto &entry : kSimtTable) {
      if (dims.M % entry.tileMNK[0] == 0 &&
          dims.N % entry.tileMNK[1] == 0) {
        chosen = &entry;
        break;
      }
    }
  }

  if (!chosen)
    chosen = &kSimtTable[0];

  int64_t tileM = chosen->tileMNK[0];
  int64_t tileN = chosen->tileMNK[1];
  int64_t tileK = chosen->tileMNK[2];
  int64_t wgX = chosen->workgroup[0];
  int64_t wgY = chosen->workgroup[1];

  while (tileK > 1 && dims.K % tileK != 0)
    tileK >>= 1;

  int64_t threadTileM = std::max((int64_t)1, tileM / wgX);
  int64_t threadTileN = std::max((int64_t)1, tileN / wgY);

  auto contractionDims = mlir::linalg::inferContractionDims(matmul);

  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  for (int64_t b : contractionDims->batch) {
    workgroupTiles[b] = 1;
    threadTiles[b] = 1;
  }
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTiles[m] = 1;
    threadTiles[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTiles[n] = 1;
    threadTiles[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims->k))
    reductionTiles[k] = 1;

  workgroupTiles[contractionDims->m.back()] = tileM;
  workgroupTiles[contractionDims->n.back()] = tileN;
  reductionTiles[contractionDims->k.back()] = tileK;
  threadTiles[contractionDims->m.back()] = threadTileM;
  threadTiles[contractionDims->n.back()] = threadTileN;

  LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] SIMT config: "
                           << "tileM=" << tileM << " tileN=" << tileN
                           << " tileK=" << tileK
                           << " threadM=" << threadTileM
                           << " threadN=" << threadTileN << "\n");

  SmallVector<int64_t> padding =
      computePaddingSizes(matmul, workgroupTiles, reductionTiles);

  MLIRContext *ctx = matmul.getContext();
  setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                               /*promotedOperands=*/{0, 1}, padding);
  return success();
}

//===----------------------------------------------------------------------===//
// setContractConfig — Public API for contractions
//===----------------------------------------------------------------------===//

LogicalResult setContractConfig(linalg::LinalgOp op,
                                const NVIDIATargetInfo &target) {
  if (!linalg::isaContractionOpInterface(op))
    return failure();

  if (op.getNumParallelLoops() < 2)
    return failure();

  MatmulDims dims = inferMatmulDims(op);
  if (!dims.valid())
    return failure();

  if (dims.M == 1 || dims.N == 1)
    return failure();

  int numLoops = op.getNumLoops();

  if (dims.M * dims.N <= kWarpSize) {
    SmallVector<int64_t> workgroupTiles(numLoops, 0);
    SmallVector<int64_t> reductionTiles(numLoops, 0);
    SmallVector<int64_t> threadTiles(numLoops, 0);
    SmallVector<int64_t> subgroupTiles(numLoops, 0);

    auto contractionDims = mlir::linalg::inferContractionDims(op);
    workgroupTiles[contractionDims->m.back()] = dims.M;
    workgroupTiles[contractionDims->n.back()] = dims.N;
    reductionTiles[contractionDims->k.back()] = 4;
    threadTiles[contractionDims->m.back()] = 1;
    threadTiles[contractionDims->n.back()] = 1;
    for (int64_t b : contractionDims->batch) {
      workgroupTiles[b] = 1;
      threadTiles[b] = 1;
    }

    SmallVector<int64_t> padding =
        computePaddingSizes(op, workgroupTiles, reductionTiles);

    MLIRContext *ctx = op.getContext();
    setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                                 workgroupTiles, reductionTiles,
                                 threadTiles, subgroupTiles,
                                 static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                                 /*promotedOperands=*/{}, padding);
    return success();
  }

  if (!target.mmaIntrinsics.empty()) {
    if (succeeded(trySetMMAConfig(op, target, dims, numLoops)))
      return success();
  }

  return setSimtConfig(op, dims, numLoops);
}

//===----------------------------------------------------------------------===//
// setDefaultConfig — For reductions and elementwise (mirrors IREE)
//===----------------------------------------------------------------------===//

LogicalResult setDefaultConfig(linalg::LinalgOp op,
                               const NVIDIATargetInfo &target) {
  int numLoops = op.getNumLoops();
  SmallVector<int64_t> loopBounds = op.getStaticLoopRanges();
  if (loopBounds.size() != static_cast<size_t>(numLoops))
    return failure();

  for (int64_t b : loopBounds) {
    if (ShapedType::isDynamic(b))
      return failure();
  }

  auto iterTypes = op.getIteratorTypesArray();

  SmallVector<unsigned> parallelDims, reductionDims;
  for (int i = 0; i < numLoops; ++i) {
    if (linalg::isParallelIterator(iterTypes[i]))
      parallelDims.push_back(i);
    else if (linalg::isReductionIterator(iterTypes[i]))
      reductionDims.push_back(i);
  }

  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  // [Doc5] kWorkgroupThreads now mirrors kTargetThreadsPerBlock (256)
  constexpr int64_t kWorkgroupThreads = kTargetThreadsPerBlock;
  constexpr int64_t kVectorSize = 4;

  if (parallelDims.empty() && reductionDims.empty()) {
    // No loops at all (e.g., rank-0 fill). Nothing to tile.
  } else if (parallelDims.empty() && !reductionDims.empty()) {
    // Full reduction (all dims are reduction, no parallel dims).
    // Distribute one reduction dim across 32 threads (one warp) using
    // partial reduction. The thread tiling pass uses
    // PartialReductionOuterParallel to correctly split + merge partials,
    // and NovaWarpShuffleReduction converts the merge into shfl.bfly.
    constexpr int64_t kWarpThreads = 32;

    // Score each reduction dim: maximize thread count, then minimize
    // threadTile (prefer threadTile=1 so each thread's slice is trivial
    // along the split dim — avoids inner-reduction edge cases in
    // PartialReductionOuterParallel).
    unsigned bestRedDim = reductionDims[0];
    int64_t bestNumThreads = 0;
    int64_t bestThreadTile = INT64_MAX;
    for (unsigned dim : reductionDims) {
      int64_t bound = loopBounds[dim];
      int64_t nt = std::min(kWarpThreads, bound);
      while (nt > 1 && bound % nt != 0)
        nt /= 2;
      int64_t tt = (nt > 1) ? bound / nt : bound;
      if (nt > bestNumThreads ||
          (nt == bestNumThreads && tt < bestThreadTile)) {
        bestRedDim = dim;
        bestNumThreads = nt;
        bestThreadTile = tt;
      }
    }

    if (bestNumThreads > 1) {
      threadTiles[bestRedDim] = bestThreadTile;
      workgroupTiles[bestRedDim] = loopBounds[bestRedDim];
    } else {
      // Dim too small to parallelize — keep it sequential.
      int64_t bound = loopBounds[bestRedDim];
      if (bound % 4 == 0) reductionTiles[bestRedDim] = 4;
      else if (bound % 2 == 0) reductionTiles[bestRedDim] = 2;
      else reductionTiles[bestRedDim] = 1;
    }

    // Remaining reduction dims stay sequential.
    for (unsigned dim : reductionDims) {
      if (dim == bestRedDim) continue;
      int64_t bound = loopBounds[dim];
      if (bound % 4 == 0) reductionTiles[dim] = 4;
      else if (bound % 2 == 0) reductionTiles[dim] = 2;
      else reductionTiles[dim] = 1;
    }
  } else if (!reductionDims.empty() && parallelDims.size() >= 2 &&
             reductionDims.back() == (unsigned)(numLoops - 1)) {
    // ROW-REDUCTION pattern: multiple parallel dims + innermost reduction.
    // Example: LN mean/var: parallel(B), parallel(T), reduction(D).
    //
    // Strategy: tile outer parallel dims so each workgroup handles a small
    // number of rows. The reduction is done SEQUENTIALLY within each thread
    // (via reductionTiles), NOT distributed across threads (threadTiles=0
    // for reduction dims). This is correct because reduction needs
    // cooperative accumulation — parallel thread slices would each compute
    // partial sums without combining them.
    unsigned innerParallelDim = parallelDims.back();
    int64_t innerSize = loopBounds[innerParallelDim];

    int64_t innerTile = std::min(innerSize, (int64_t)(kWorkgroupThreads));
    while (innerTile > 1 && innerSize % innerTile != 0)
      innerTile /= 2;
    workgroupTiles[innerParallelDim] = innerTile;
    threadTiles[innerParallelDim] = 1;

    for (unsigned dim : parallelDims) {
      if (dim == innerParallelDim) continue;
      workgroupTiles[dim] = 1;
    }

    // DO NOT set threadTiles on reduction dims — that would create a
    // parallel forall where each thread computes a partial sum without
    // any cross-thread combination.
    for (unsigned dim : reductionDims) {
      int64_t bound = loopBounds[dim];
      if (bound % 4 == 0) reductionTiles[dim] = 4;
      else if (bound % 2 == 0) reductionTiles[dim] = 2;
      else reductionTiles[dim] = 1;
    }
  } else {
    // Has parallel dims — distribute across workgroup.
    // Innermost parallel dim gets the bulk of threads * vectorization.
    unsigned innerParallelDim = parallelDims.back();
    int64_t innerSize = loopBounds[innerParallelDim];

    // Two-pass search: prefer vectorSize >= 2 (meaningful thread tiles that
    // actually create thread-level foralls). vectorSize=1 means threadTile=1,
    // which the thread tiling pass may treat as degenerate (especially when
    // the op is fused into a block forall with different dims, causing
    // clampThreadTilesToMaxThreads to overcorrect down to 1 thread).
    int64_t bestThreads = 0;
    int64_t bestVS = 1;
    // Pass 1: only vs >= 2.
    for (int64_t vs = kVectorSize; vs >= 2; vs /= 2) {
      for (int64_t thr = kWorkgroupThreads; thr >= kWarpSize; thr /= 2) {
        int64_t tile = thr * vs;
        if (tile <= innerSize && innerSize % tile == 0) {
          if (thr > bestThreads || (thr == bestThreads && vs > bestVS)) {
            bestThreads = thr;
            bestVS = vs;
          }
          break;
        }
      }
    }
    // Pass 2: fallback to vs=1 only if no vs>=2 combo found.
    if (bestThreads == 0) {
      for (int64_t thr = kWorkgroupThreads; thr >= kWarpSize; thr /= 2) {
        if (thr <= innerSize && innerSize % thr == 0) {
          bestThreads = thr;
          bestVS = 1;
          break;
        }
      }
    }
    // Fallback for small/non-power-of-2 dims: tile the entire dim (1 block).
    if (bestThreads == 0) {
      bestVS = kVectorSize;
      while (bestVS > 1 && innerSize % bestVS != 0)
        bestVS /= 2;
      bestThreads = std::min(innerSize / bestVS, kWorkgroupThreads);
    }
    int64_t vectorSize = bestVS;
    int64_t innerTile = std::min(bestThreads * bestVS, innerSize);

    workgroupTiles[innerParallelDim] = innerTile;
    threadTiles[innerParallelDim] = vectorSize;

    // Outer parallel dims: distribute the remaining thread budget inward->outward
    // so that each block handles multiple rows/slices rather than a single one.
    //
    // Without this, a [B, M, N] all-parallel op (e.g. LayerNorm elementwise)
    // with innerTile covering only the N dim produces workgroupTile_M=1, which
    // multiplies the grid Y dimension by M (e.g. 1024) — spawning orders of
    // magnitude more blocks than needed and blowing the SRAM per-block budget
    // when shared-memory promotion is active.
    //
    // threadTile=0 on budget-exhausted dims means the thread tiling pass skips
    // them entirely — they are fully block-distributed with no thread forall.
    // Setting threadTile=1 instead would create threads along these dims, which
    // is catastrophic when the op is fused into a block forall that doesn't
    // tile these dims (the thread count explodes).
    int64_t remainingBudget = kWorkgroupThreads / bestThreads;
    for (int i = (int)parallelDims.size() - 2; i >= 0; --i) {
      unsigned dim = parallelDims[i];
      int64_t dimSize = loopBounds[dim];
      if (remainingBudget > 1 && dimSize > 1) {
        int64_t tile = std::min(dimSize, remainingBudget);
        while (tile > 1 && dimSize % tile != 0)
          tile /= 2;
        workgroupTiles[dim] = tile;
        threadTiles[dim] = 1;
        remainingBudget /= tile;
      } else {
        workgroupTiles[dim] = 1;
        threadTiles[dim] = 0;
      }
    }

    for (unsigned dim : reductionDims) {
      int64_t bound = loopBounds[dim];
      if (bound % 4 == 0) reductionTiles[dim] = 4;
      else if (bound % 2 == 0) reductionTiles[dim] = 2;
      else reductionTiles[dim] = 1;
    }

    clampThreadTilesToMaxThreads(workgroupTiles, threadTiles);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-kernel-config] Default config for "
                 << op->getName() << ": workgroup=[";
    llvm::interleaveComma(workgroupTiles, llvm::dbgs());
    llvm::dbgs() << "] reduction=[";
    llvm::interleaveComma(reductionTiles, llvm::dbgs());
    llvm::dbgs() << "] thread=[";
    llvm::interleaveComma(threadTiles, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  SmallVector<int64_t> padding =
      computePaddingSizes(op, workgroupTiles, reductionTiles);

  MLIRContext *ctx = op.getContext();
  SmallVector<int64_t> promotedOps;
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  auto maps = linalgOp.getIndexingMapsArray();
  unsigned numInputs = linalgOp.getNumDpsInputs();
  for (unsigned i = 0; i < numInputs; ++i) {
    if (maps[i].getNumResults() < (unsigned)numLoops)
      promotedOps.push_back(i);
  }
  setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                               promotedOps, padding);
  return success();
}

//===----------------------------------------------------------------------===//
// initNovaGPULaunchConfig — Main entry point
//===----------------------------------------------------------------------===//

/// Returns true if `op` is a "true" contraction (matmul-like), not just any op
/// that isaContractionOpInterface might match (e.g., elementwise dot-products
/// like C[i,j] = sum_k A[i,j,k]*B[i,j,k] also match contraction interface but
/// are semantically reductions, not matmuls).
static bool isTrueContraction(Operation *op) {
  if (isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::MatvecOp,
          linalg::VecmatOp, linalg::BatchMatvecOp>(op))
    return true;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp))
    return false;
  auto indexMaps = linalgOp.getIndexingMapsArray();
  if (indexMaps.size() < 3)
    return false;
  auto iterTypes = linalgOp.getIteratorTypesArray();
  unsigned numLoops = iterTypes.size();
  for (unsigned i = 0; i < numLoops; ++i) {
    if (iterTypes[i] != utils::IteratorType::parallel)
      continue;
    bool inInput0 = indexMaps[0].isFunctionOfDim(i);
    bool inInput1 = indexMaps[1].isFunctionOfDim(i);
    if (inInput0 != inInput1)
      return true;
  }
  return false;
}

/// Returns true if `op` is a "root" op that should get its own lowering_config
/// and be independently tiled into its own scf.forall. Roots are:
///   - True contractions (matmul-like ops)
///   - Ops with reduction iterator types (e.g., LN mean/var, softmax reduce)
/// All other ops (elementwise, broadcasts) should fuse into a root's forall.
static bool isRootOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  if (isTrueContraction(op))
    return true;
  for (auto iterType : linalgOp.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel)
      return true;
  }
  return false;
}

/// Returns true if `op` is a purely elementwise (all-parallel) consumer of a
/// root op. Such ops are "epilogues" (bias add, relu, etc.) and should NOT get
/// independent lowering configs — they fuse into the root's scf.forall as
/// consumers during TileDispatch.
///
/// IMPORTANT: Consumers with reduction iterators (e.g., batch-reduce for
/// grad_weight) are NOT epilogues — they need their own configs.
///
/// [Doc5] Added projected permutation guard on all indexing maps before the
/// operand chain walk. Prevents incorrect fusion when maps have transposes or
/// non-trivial reindexing that would cause consumer fusion to silently fail.
static bool isEpilogue(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  // Only all-parallel ops can safely fuse as epilogues.
  for (auto iterType : linalgOp.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel)
      return false;
  }
  // [Doc5] All indexing maps must be projected permutations to safely fuse.
  // Transposes or complex maps will fail consumer fusion in TileDispatch.
  for (AffineMap map : linalgOp.getIndexingMapsArray()) {
    if (!map.isProjectedPermutation())
      return false;
  }
  // Check if any operand traces to a root op (up to 2 levels deep).
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;
    if (isRootOp(defOp))
      return true;
    // One level deeper: epilogue chains like root -> bias_add -> relu.
    for (Value innerOp : defOp->getOperands()) {
      Operation *innerDef = innerOp.getDefiningOp();
      if (!innerDef)
        continue;
      if (isRootOp(innerDef))
        return true;
    }
  }
  return false;
}

/// Returns true if `op` is a direct producer (input) to any root op
/// AND the root accesses this operand through a projected permutation
/// (no transpose/reindex), so tile-based producer fusion is feasible.
///
/// If the root reindexes this operand (e.g., the op produces [8,1024,384]
/// but the root accesses it as [8,384,1024] via a transposed indexing map),
/// producer fusion will fail during TileDispatch, and the op will be left
/// un-tiled -> serialized to 1 workgroup in Phase 2. In that case we must
/// NOT skip the config — the op needs its own lowering_config.
static bool isPrologue(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  // Only all-parallel ops can safely fuse as prologues.
  for (auto iterType : linalgOp.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel)
      return false;
  }
  // Check if any result feeds a root via projected permutation.
  for (OpResult result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!isRootOp(user))
        continue;
      auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user);
      if (!consumerLinalgOp)
        continue;
      for (OpOperand &operand : consumerLinalgOp->getOpOperands()) {
        if (operand.get() != result)
          continue;
        AffineMap map = consumerLinalgOp.getMatchingIndexingMap(&operand);
        if (map.isProjectedPermutation() && !map.isEmpty())
          return true;
        break;
      }
    }
  }
  return false;
}

void initNovaGPULaunchConfig(mlir::func::FuncOp funcOp,
                              const NVIDIATargetInfo &target) {
  // Walk all linalg ops and attach configs. Unlike IREE (which has one root
  // per dispatch), Nova has multiple compute clusters in a single function.
  // We stamp configs only on ROOT ops:
  //   - Contractions get contraction config
  //   - Other root ops (reductions) get default config
  //
  // Non-root ops (prologues, epilogues, fills) do NOT get configs — they
  // fuse into their root's scf.forall via producer/consumer fusion.
  funcOp.walk([&](linalg::LinalgOp op) {
    // Skip ops that already have a config.
    if (getLoweringConfig(op.getOperation()))
      return;

    // FillOps always fuse as DPS init — never need a config.
    if (isa<linalg::FillOp>(op.getOperation()))
      return;

    // Epilogues (all-parallel consumers of roots) fuse as consumers.
    if (isEpilogue(op.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping epilogue: "
                               << op->getName() << "\n");
      return;
    }

    // Prologues (all-parallel producers feeding roots via projected
    // permutation) fuse as producers — no independent config needed.
    if (isPrologue(op.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping prologue: "
                               << op->getName() << "\n");
      return;
    }

    // Root ops: contractions get contraction config, others get default.
    if (linalg::isaContractionOpInterface(op)) {
      if (succeeded(setContractConfig(op, target)))
        return;
      // If contraction config fails (matvec, dynamic), fall through to default.
    }

    if (isa<linalg::GenericOp>(op.getOperation())) {
      // Epilogues and prologues are already handled by the generic isEpilogue /
      // isPrologue checks above — no separate contraction-specific pass needed.
      // Any generic that reaches here is a true root (reduction or standalone
      // elementwise not fused into any contraction) → give it a default config.
      (void)setDefaultConfig(op, target);
      return;
    }

    (void)setDefaultConfig(op, target);
  });
}

} // namespace mlir::nova