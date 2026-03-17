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
//      - workgroupThreads = 2 * warpSize = 64
//      - vectorSize = 4 (128-bit loads for f32)
//      - inner parallel dim tile = threads * vectorSize = 256
//      - reduction dims tile = 4
//      - thread tile for inner dim = vectorSize
//
//   3. initNovaGPULaunchConfig: Single-root model per compute cluster.
//      Walks all linalg ops, finds roots, stamps configs.
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

/// CUDA's maximum number of threads per block.
static constexpr int64_t kMaxThreadsPerBlock = 1024;

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
// clampThreadTilesToMaxThreads — enforce CUDA 1024-thread limit
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

  while (computeTotal() > kMaxThreadsPerBlock) {
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
    if (computeTotal() > kMaxThreadsPerBlock && worstTrips <= 1)
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

  // 2×2 subgroups per workgroup, 4 MMA tiles per subgroup.
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

  // Thread count = (wgM / threadTileM) * (wgN / threadTileN).
  // Use subgroup-derived thread tiles: wgM / (numSubgroupsM * subgroupTilesM)
  // and wgN / (numSubgroupsN * subgroupTilesN) gives per-thread work.
  // But for the thread forall, use 8×8 to get 256 threads for 128×128.
  int64_t threadTileM = 8;
  int64_t threadTileN = 8;

  // Ensure thread count doesn't exceed 1024.
  while ((wgM / threadTileM) * (wgN / threadTileN) > kMaxThreadsPerBlock &&
         wgN > threadTileN) {
    wgN /= 2;
  }

  auto contractionDims = mlir::linalg::inferContractionDims(matmul);
  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);

  // Batch dims → 1.
  for (int64_t b : contractionDims->batch) {
    workgroupTiles[b] = 1;
    threadTiles[b] = 1;
    subgroupTiles[b] = 1;
  }
  // Outer M dims → 1.
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTiles[m] = 1;
    threadTiles[m] = 1;
    subgroupTiles[m] = 1;
  }
  // Outer N dims → 1.
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTiles[n] = 1;
    threadTiles[n] = 1;
    subgroupTiles[n] = 1;
  }
  // Outer K dims → 1.
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
  // Pick first table entry whose M,N tile divides the problem dims.
  const SimtTilePair *chosen = nullptr;
  for (const auto &entry : kSimtTable) {
    if (dims.M % entry.tileMNK[0] == 0 &&
        dims.N % entry.tileMNK[1] == 0 &&
        dims.K % entry.tileMNK[2] == 0) {
      chosen = &entry;
      break;
    }
  }

  // If nothing divides perfectly, try M,N alignment only.
  if (!chosen) {
    for (const auto &entry : kSimtTable) {
      if (dims.M % entry.tileMNK[0] == 0 &&
          dims.N % entry.tileMNK[1] == 0) {
        chosen = &entry;
        break;
      }
    }
  }

  // Last resort: pick the first entry.
  if (!chosen)
    chosen = &kSimtTable[0];

  int64_t tileM = chosen->tileMNK[0];
  int64_t tileN = chosen->tileMNK[1];
  int64_t tileK = chosen->tileMNK[2];
  int64_t wgX = chosen->workgroup[0];
  int64_t wgY = chosen->workgroup[1];

  // Align K tile to actual K size if not evenly divisible.
  while (tileK > 1 && dims.K % tileK != 0)
    tileK >>= 1;

  // Thread tiles: workgroup tile / workgroup size (like IREE).
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

  // Must have at least 2 parallel dims (M and N).
  if (op.getNumParallelLoops() < 2)
    return failure();

  MatmulDims dims = inferMatmulDims(op);
  if (!dims.valid())
    return failure();

  // Reject matvec (one of M,N == 1) — should go through reduction pipeline.
  if (dims.M == 1 || dims.N == 1)
    return failure();

  int numLoops = op.getNumLoops();

  // Very small matmul (M*N <= warpSize): scalar-per-thread config.
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

  // Try MMA-based config first.
  if (!target.mmaIntrinsics.empty()) {
    if (succeeded(trySetMMAConfig(op, target, dims, numLoops)))
      return success();
  }

  // Fall back to SIMT tile table.
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

  // Reject if any loop bound is dynamic.
  for (int64_t b : loopBounds) {
    if (ShapedType::isDynamic(b))
      return failure();
  }

  auto iterTypes = op.getIteratorTypesArray();

  // Classify parallel and reduction dims.
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

  // Mirroring IREE's setRootDefaultConfig:
  // - workgroupThreads = 2 * warpSize = 64
  // - vectorSize = 4 (128-bit loads for f32)
  // - innermost parallel dim tile = workgroupThreads * vectorSize = 256
  // - other parallel dims: 1 per thread
  // - reduction dims: tile = 4
  constexpr int64_t kWorkgroupThreads = 2 * kWarpSize;  // 64
  constexpr int64_t kVectorSize = 4;

  if (parallelDims.empty()) {
    // Full reduction (all dims are reduction, no parallel dims).
    // Set all workgroup tiles to 0 — the dispatch pass will wrap in a
    // single-block forall. Set reduction tiles for vectorization.
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

    // Innermost: threads * vectorSize elements.
    int64_t vectorSize = kVectorSize;
    int64_t innerTile = kWorkgroupThreads * vectorSize;

    // Adjust vectorSize if inner dim doesn't support it.
    while (vectorSize > 1 && innerSize % (kWorkgroupThreads * vectorSize) != 0)
      vectorSize /= 2;
    innerTile = kWorkgroupThreads * vectorSize;

    // Clamp to actual size.
    innerTile = std::min(innerTile, innerSize);

    workgroupTiles[innerParallelDim] = innerTile;
    threadTiles[innerParallelDim] = vectorSize;

    // Other parallel dims (outer): tile to 1, thread tile = 1.
    for (int i = (int)parallelDims.size() - 2; i >= 0; --i) {
      unsigned dim = parallelDims[i];
      workgroupTiles[dim] = 1;
      threadTiles[dim] = 1;
    }

    // Reduction dims: tile = 4 for vectorized loads.
    for (unsigned dim : reductionDims) {
      int64_t bound = loopBounds[dim];
      if (bound % 4 == 0) reductionTiles[dim] = 4;
      else if (bound % 2 == 0) reductionTiles[dim] = 2;
      else reductionTiles[dim] = 1;
    }

    // Clamp total thread count to <= 1024.
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
  setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                               /*promotedOperands=*/{}, padding);
  return success();
}

//===----------------------------------------------------------------------===//
// initNovaGPULaunchConfig — Main entry point
//===----------------------------------------------------------------------===//

/// Returns true if `op` is a purely elementwise (all-parallel) consumer of a
/// contraction op. Such ops are "epilogues" (bias add, relu, etc.) and should
/// NOT get independent lowering configs — they fuse into the contraction's
/// scf.forall as consumers during TileDispatch.
///
/// IMPORTANT: Consumers with reduction iterators (e.g., batch-reduce for
/// grad_weight) are NOT epilogues — they need their own configs. Fusing a
/// dimension-reducing consumer into a contraction's forall creates overlapping
/// writes when the consumer's output rank < forall dims.
static bool isContractionEpilogue(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  // Only all-parallel ops can safely fuse as epilogues.
  for (auto iterType : linalgOp.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel)
      return false;
  }
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;
    auto linalgDef = dyn_cast<linalg::LinalgOp>(defOp);
    if (linalgDef && linalg::isaContractionOpInterface(linalgDef))
      return true;
    // Also check one level deeper: epilogue chains like
    // matmul → bias_add → relu (relu's input is bias_add, not matmul).
    if (auto genericDef = dyn_cast<linalg::GenericOp>(defOp)) {
      for (Value innerOp : genericDef->getOperands()) {
        Operation *innerDef = innerOp.getDefiningOp();
        if (!innerDef)
          continue;
        auto innerLinalg = dyn_cast<linalg::LinalgOp>(innerDef);
        if (innerLinalg && linalg::isaContractionOpInterface(innerLinalg))
          return true;
      }
    }
  }
  return false;
}

/// Returns true if `op` is a direct producer (input) to a contraction op.
/// Such ops are "prologues" (weight broadcast, input transform, etc.) and
/// should NOT get independent lowering configs — they fuse into the
/// contraction's scf.forall as producers during TileDispatch.
static bool isContractionPrologue(Operation *op) {
  for (OpResult result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      auto linalgUser = dyn_cast<linalg::LinalgOp>(user);
      if (linalgUser && linalg::isaContractionOpInterface(linalgUser))
        return true;
    }
  }
  return false;
}

void initNovaGPULaunchConfig(mlir::func::FuncOp funcOp,
                              const NVIDIATargetInfo &target) {
  // Walk all linalg ops and attach configs. Unlike IREE (which has one root
  // per dispatch), Nova has multiple compute clusters in a single function.
  // We stamp configs on each root independently:
  //   1. Contractions get contraction config
  //   2. Generics that are NOT contraction epilogues get default config
  //
  // Epilogue ops (bias, relu, etc. following a contraction) do NOT get
  // configs — they fuse into the contraction's forall via consumer fusion.

  funcOp.walk([&](linalg::LinalgOp op) {
    // Skip ops that already have a config.
    if (getLoweringConfig(op.getOperation()))
      return;

    // Priority 1: Contraction ops → MMA or SIMT config.
    if (linalg::isaContractionOpInterface(op)) {
      if (succeeded(setContractConfig(op, target)))
        return;
      // If contraction config fails (matvec, dynamic), fall through to default.
    }

    // Priority 2: Any linalg.generic → default config, unless it's an
    // epilogue/prologue of a contraction (those fuse as consumers/producers,
    // no independent config needed).
    if (isa<linalg::GenericOp>(op.getOperation())) {
      if (isContractionEpilogue(op.getOperation())) {
        LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping epilogue: "
                                 << op->getName() << "\n");
        return;
      }
      if (isContractionPrologue(op.getOperation())) {
        LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping prologue: "
                                 << op->getName() << "\n");
        return;
      }
      (void)setDefaultConfig(op, target);
      return;
    }
  });
}

} // namespace mlir::nova
