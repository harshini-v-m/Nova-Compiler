

#include "Passes.h"
#include "NovaGPUTileAndFuseUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-tiling"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Tiling Level Enum — mirrors IREE::GPU::TilingLevel
//===----------------------------------------------------------------------===//

enum class NovaTilingLevel {
  Reduction, // K dimension → scf.for (sequential)
  Thread,    // M/N per-thread → scf.forall + thread mapping
  Subgroup,  // M/N per-warp  → scf.forall + warp mapping
};

// Maximum threads per block (CUDA hardware limit).
static constexpr int64_t kMaxThreadsPerBlock = 1024;

/// Returns the config key string for a given tiling level.
static StringRef getLevelKey(NovaTilingLevel level) {
  switch (level) {
  case NovaTilingLevel::Reduction:
    return kReductionKey;
  case NovaTilingLevel::Thread:
    return kThreadKey;
  case NovaTilingLevel::Subgroup:
    return kSubgroupKey;
  }
  llvm_unreachable("unknown tiling level");
}

/// Adjusts threadTiles so that the product of (actualDims[i] / threadTiles[i])
/// over all active dims is at most kMaxThreadsPerBlock.
static void clampThreadTilesToMaxThreads(ArrayRef<int64_t> actualDims,
                                         SmallVectorImpl<int64_t> &threadTiles) {
  int n = (int)threadTiles.size();
  auto computeTotal = [&]() -> int64_t {
    int64_t total = 1;
    for (int i = 0; i < n; ++i) {
      if (threadTiles[i] <= 0) continue;
      if (i >= (int)actualDims.size() || actualDims[i] <= 0 ||
          ShapedType::isDynamic(actualDims[i])) continue;
      total *= (actualDims[i] + threadTiles[i] - 1) / threadTiles[i];
    }
    return total;
  };
  while (computeTotal() > kMaxThreadsPerBlock) {
    int worstDim = -1;
    int64_t worstTrips = 0;
    for (int i = 0; i < n; ++i) {
      if (threadTiles[i] <= 0) continue;
      if (i >= (int)actualDims.size() || actualDims[i] <= 0 ||
          ShapedType::isDynamic(actualDims[i])) continue;
      int64_t trips = (actualDims[i] + threadTiles[i] - 1) / threadTiles[i];
      if (trips > worstTrips) {
        worstTrips = trips;
        worstDim = i;
      }
    }
    if (worstDim < 0 || worstTrips <= 1) break;
    threadTiles[worstDim] *= 2;
  }
}

//===----------------------------------------------------------------------===//
// getTiledOps — config-driven, mirrors IREE's getTiledOps()
//
// Collects all linalg ops that have non-zero tile sizes at the requested
// level in their lowering_config. No heuristic checks needed.
//===----------------------------------------------------------------------===//

static SmallVector<TilingInterface>
getTiledOps(func::FuncOp funcOp, NovaTilingLevel tilingLevel) {
  SmallVector<TilingInterface> targets;
  StringRef levelKey = getLevelKey(tilingLevel);

  funcOp.walk([&](TilingInterface target) {
    // Only collect linalg ops — config propagation may accidentally stamp
    // lowering_config on non-linalg TilingInterface ops (e.g. extract_slice)
    // whose TilingInterface impl can't handle our tile size queries.
    if (!isa<linalg::LinalgOp>(target.getOperation()))
      return WalkResult::advance();

    auto config = getLoweringConfig(target.getOperation());
    if (!config)
      return WalkResult::advance();

    auto tiles = getLoweringConfigTileSizes(config, levelKey);
    if (tiles.empty() || !llvm::any_of(tiles, [](int64_t t) { return t > 0; }))
      return WalkResult::advance();

    // Skip ops already tiled at Subgroup level: they live inside a
    // #gpu.warp-mapped scf.forall from a prior run of this pass.
    // Without this guard a second invocation (e.g. after canonicalization)
    // would produce double-nested warp foralls.
    if (tilingLevel == NovaTilingLevel::Subgroup) {
      bool alreadyWarpTiled = false;
      for (Operation *parent = target->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (auto forall = dyn_cast<scf::ForallOp>(parent)) {
          auto mapping = forall.getMappingAttr();
          if (mapping && !mapping.empty() &&
              isa<gpu::GPUWarpMappingAttr>(mapping.getValue().front())) {
            alreadyWarpTiled = true;
            break;
          }
        }
      }
      if (alreadyWarpTiled)
        return WalkResult::advance();
    }

    targets.push_back(target);
    return WalkResult::advance();
  });

  return targets;
}

//===----------------------------------------------------------------------===//
// getTileSizes — read directly from config using level key
//===----------------------------------------------------------------------===//

static SmallVector<OpFoldResult>
getTileSizes(RewriterBase &rewriter, TilingInterface tilingOp,
             NovaTilingLevel tilingLevel) {
  Operation *op = tilingOp.getOperation();

  // Get numLoops safely via linalg interface.
  int64_t numLoops = 0;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    numLoops = linalgOp.getNumLoops();
  else
    numLoops = tilingOp.getLoopIteratorTypes().size();
  auto zero = rewriter.getIndexAttr(0);

  auto config = getLoweringConfig(op);
  if (!config)
    return SmallVector<OpFoldResult>(numLoops, zero);

  StringRef levelKey = getLevelKey(tilingLevel);
  SmallVector<int64_t> tiles = getLoweringConfigTileSizes(config, levelKey);

  // Pad with zeros to numLoops.
  while ((int64_t)tiles.size() < numLoops)
    tiles.push_back(0);
  if ((int64_t)tiles.size() > numLoops)
    return SmallVector<OpFoldResult>(numLoops, zero);

  // Thread level: when the op is inside a warp-mapped forall (subgroup-tiled),
  // zero out contraction dims for MMA ops.  The vectorization + vector unrolling
  // pipeline handles per-warp subdivision to native MMA tile sizes.  Creating
  // a thread forall for contraction dims inside a warp forall would produce
  // incorrect GPU mapping (one thread per 16×8 tile instead of 32-thread
  // warp-cooperative mma.sync).
  //
  // Non-contraction dims and non-MMA ops keep their thread tiles unchanged.
  if (tilingLevel == NovaTilingLevel::Thread) {
    // Check if this op is nested inside a warp-mapped scf.forall.
    bool insideWarpForall = false;
    if (auto forall = op->getParentOfType<scf::ForallOp>()) {
      if (auto mapping = forall.getMappingAttr()) {
        for (auto attr : mapping.getValue()) {
          if (isa<gpu::GPUWarpMappingAttr>(attr)) {
            insideWarpForall = true;
            break;
          }
        }
      }
    }
    if (insideWarpForall) {
      int32_t mmaKind = getMmaKindRaw(config);
      if (mmaKind != 0) {
        // MMA contraction inside warp forall: zero out all tiles so no
        // thread forall is created. The matmul stays at warp granularity.
        auto subgroupTiles = getLoweringConfigTileSizes(config, kSubgroupKey);
        for (int i = 0; i < (int)tiles.size() &&
                        i < (int)subgroupTiles.size(); ++i) {
          if (subgroupTiles[i] > 0)
            tiles[i] = 0;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-tiling] zeroed MMA contraction thread tiles "
                      "(inside warp forall): [";
                   for (int64_t t : tiles) llvm::dbgs() << t << " ";
                   llvm::dbgs() << "]\n");
      }
    }
  }

  // Derived-thread config: recompute thread tiles from the op's actual
  // (K-tiled) loop ranges and the stored target_threads count.  This ensures
  // the copy forall trip count matches the matmul forall trip count, enabling
  // FuseForalls in Step 6.  Mirrors IREE's DerivedThreadConfigAttr.
  if (tilingLevel == NovaTilingLevel::Thread && isDerivedThreadConfig(config)) {
    int64_t targetThreads = getTargetThreadCount(config);
    int64_t blockDim = getBlockDim(config);
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
      unsigned elemBits =
          getElementTypeOrSelf(linalgOp->getResultTypes()[0])
              .getIntOrFloatBitWidth();
      tiles = deriveThreadTileSizes(loopRanges, targetThreads, elemBits,
                                    blockDim);
      // Pad/truncate to numLoops.
      while ((int64_t)tiles.size() < numLoops)
        tiles.push_back(0);
      tiles.resize(numLoops);

      LLVM_DEBUG(llvm::dbgs() << "[nova-tiling] derived thread tiles for "
                               << op->getName() << ": [";
                 for (int64_t t : tiles) llvm::dbgs() << t << " ";
                 llvm::dbgs() << "] target=" << targetThreads << "\n");
    }
  }

  // For non-derived-thread ops at Thread level, the correct per-subgroup
  // tile size is wg_tile[d] / wg_subgroup[d], not thread[d].
  // thread[d] is per-thread elements; the forall created here maps one
  // iteration to one subgroup, so its step must be the subgroup tile.
  if (tilingLevel == NovaTilingLevel::Thread && !isDerivedThreadConfig(config)) {
    auto wgTiles    = getLoweringConfigTileSizes(config, kWorkgroupKey);
    auto wgSubgroup = getLoweringConfigTileSizes(config, kWgSubgroupKey);
    if (!wgTiles.empty() && !wgSubgroup.empty()) {
      for (int64_t i = 0; i < (int64_t)tiles.size(); ++i) {
        if (tiles[i] == 0) continue;  // leave reduction/unused dims at 0
        int64_t sg = (i < (int64_t)wgSubgroup.size()) ? wgSubgroup[i] : 1;
        int64_t wg = (i < (int64_t)wgTiles.size())    ? wgTiles[i]    : 0;
        // sg == 0 means no subgroup tiling on this dim, leave tile as-is.
        tiles[i] = (sg > 0 && wg > 0) ? wg / sg : 0;
      }
    }
  }

  if (tilingLevel == NovaTilingLevel::Thread) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      SmallVector<int64_t> actualDims = linalgOp.getStaticLoopRanges();
      clampThreadTilesToMaxThreads(actualDims, tiles);
    }
  }

  // ── Subgroup level: validate tile sizes and clamp warp count ──────────────
  // Three invariants enforced here:
  //   1. subgroupTile[d] must not exceed workgroupTile[d].
  //   2. workgroupTile[d] must be divisible by subgroupTile[d] (no partial
  //      warps — every warp must own the same shape so layouts are uniform).
  //      If not, walk down from the config value to the largest exact divisor.
  //   3. Total warp count (product of forall trip counts) must not exceed
  //      kMaxThreadsPerBlock / kWarpSize.  If it does, double tile sizes
  //      (fewer trips = fewer warps) until the limit is satisfied.
  if (tilingLevel == NovaTilingLevel::Subgroup) {
    auto wgTiles = getLoweringConfigTileSizes(config, kWorkgroupKey);

    // Guards 1 and 2: per-dimension bounds and divisibility.
    for (int64_t i = 0; i < (int64_t)tiles.size(); ++i) {
      if (tiles[i] <= 0) continue;
      int64_t wg = (i < (int64_t)wgTiles.size()) ? wgTiles[i] : 0;
      if (wg <= 0) continue;

      // Guard 1: tile must not exceed workgroup tile.
      if (tiles[i] > wg)
        tiles[i] = wg;

      // Guard 2: tile must divide workgroup tile evenly.
      // Walk down from tiles[i] until wg % tiles[i] == 0.
      while (tiles[i] > 1 && wg % tiles[i] != 0)
        --tiles[i];
    }

    // Guard 3: total warp count <= kMaxThreadsPerBlock / kWarpSize.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      SmallVector<int64_t> actualDims = linalgOp.getStaticLoopRanges();
      constexpr int64_t kWarpSize = 32;
      const int64_t maxWarps = kMaxThreadsPerBlock / kWarpSize;

      auto computeTotalWarps = [&]() -> int64_t {
        int64_t total = 1;
        for (int i = 0; i < (int)tiles.size(); ++i) {
          if (tiles[i] <= 0) continue;
          if (i >= (int)actualDims.size() || actualDims[i] <= 0 ||
              ShapedType::isDynamic(actualDims[i])) continue;
          total *= (actualDims[i] + tiles[i] - 1) / tiles[i];
        }
        return total;
      };

      while (computeTotalWarps() > maxWarps) {
        // Double the tile size on the dimension with the most warp trips.
        int worstDim = -1;
        int64_t worstTrips = 0;
        for (int i = 0; i < (int)tiles.size(); ++i) {
          if (tiles[i] <= 0) continue;
          if (i >= (int)actualDims.size() || actualDims[i] <= 0 ||
              ShapedType::isDynamic(actualDims[i])) continue;
          int64_t trips = (actualDims[i] + tiles[i] - 1) / tiles[i];
          if (trips > worstTrips) { worstTrips = trips; worstDim = i; }
        }
        if (worstDim < 0 || worstTrips <= 1) break;
        tiles[worstDim] *= 2;
        // Never exceed the workgroup tile.
        if (worstDim < (int)wgTiles.size() && wgTiles[worstDim] > 0)
          tiles[worstDim] = std::min(tiles[worstDim], wgTiles[worstDim]);
      }
    }
  }

  SmallVector<OpFoldResult> tileSizes;
  for (int64_t t : tiles)
    tileSizes.push_back(rewriter.getIndexAttr(t));
  return tileSizes;
}



/// Fixes linalg.index uses inside tiled linalg.generic bodies for reduction
/// dims tiled with PartialReductionOuterParallel at the Thread level.
///
/// Root cause: when the thread tiling pass tiles a reduction dim, it creates a
/// scf.forall whose IV gives the tile-start offset. Formal operands are
/// correctly sliced via tensor.extract_slice (offset embedded in the slice),
/// but any tensor.extract on non-operand tensors inside the body still uses
/// linalg.index i — which, after tiling, gives the tile-local index (0..T-1)
/// rather than the global index (tile_start + 0..T-1).
///
/// Fix: for each tiled linalg.generic inside the forall, find every
/// linalg.index i where dim i was tiled as a reduction, and replace its result
/// with (forall_iv_for_dim_i + linalg.index i) everywhere it is used.
static void fixupReductionTiledLinalgIndex(IRRewriter &rewriter,
                                           linalg::LinalgOp originalOp,
                                           scf::SCFTileAndFuseResult &result,
                                           ArrayRef<OpFoldResult> tileSizes) {
  // Find the enclosing scf.forall created by the tiling pass.
  scf::ForallOp forallOp;
  for (LoopLikeOpInterface loop : result.loops) {
    if (auto fop = dyn_cast<scf::ForallOp>(loop.getOperation())) {
      forallOp = fop;
      break;
    }
  }
  if (!forallOp)
    return;

  // Map each tiled reduction dimension to its forall induction variable.
  // The forall IVs are created for non-zero tile dims in left-to-right order.
  auto iterTypes = originalOp.getIteratorTypesArray();
  SmallVector<std::pair<int, Value>> tiledReductionDimIVs;
  {
    SmallVector<Value> ivs = forallOp.getInductionVars();
    int ivIdx = 0;
    for (int i = 0;
         i < (int)tileSizes.size() && ivIdx < (int)ivs.size(); ++i) {
      if (!isZeroInteger(tileSizes[i])) {
        if (i < (int)iterTypes.size() &&
            linalg::isReductionIterator(iterTypes[i]))
          tiledReductionDimIVs.push_back({i, ivs[ivIdx]});
        ++ivIdx;
      }
    }
  }
  if (tiledReductionDimIVs.empty())
    return;

  // For each tiled linalg op, offset every linalg.index i (for a tiled
  // reduction dim i) by the corresponding forall IV so callers of linalg.index
  // that access non-operand tensors get the correct global position.
  for (Operation *tiledOp : result.tiledAndFusedOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(tiledOp);
    if (!linalgOp || linalgOp->getNumRegions() == 0)
      continue;

    SmallVector<std::pair<linalg::IndexOp, Value>> toFix;
    linalgOp->getRegion(0).walk([&](linalg::IndexOp indexOp) {
      for (auto [tiledDim, iv] : tiledReductionDimIVs) {
        if ((int)indexOp.getDim() == tiledDim) {
          toFix.push_back({indexOp, iv});
          break;
        }
      }
    });

    for (auto [indexOp, iv] : toFix) {
      rewriter.setInsertionPointAfter(indexOp);
      Value localIdx = indexOp.getResult();
      // globalIdx = tile start (forall IV) + tile-local offset (linalg.index).
      Value globalIdx =
          rewriter.create<arith::AddIOp>(indexOp.getLoc(), iv, localIdx);
      // Replace all uses of localIdx with globalIdx except in the AddIOp
      // itself, which reads localIdx as one of its operands.
      localIdx.replaceUsesWithIf(globalIdx, [&](OpOperand &use) {
        return use.getOwner() != globalIdx.getDefiningOp();
      });
    }
  }
}

static LogicalResult
applyTileAndFuseToEachRoot(func::FuncOp funcOp, IRRewriter &rewriter,
                           SmallVector<TilingInterface> &targetOps,
                           NovaTilingLevel tilingLevel) {
  MLIRContext *ctx = funcOp.getContext();

  // Build a set of payload ops for fusion control (avoid fusing independently-
  // tiled ops into each other). Mirrors IREE's payloadOps set.
  llvm::SmallDenseSet<TilingInterface> payloadOps(targetOps.begin(),
                                                   targetOps.end());

  for (TilingInterface tilingOp : targetOps) {
    // Skip if already erased by a previous iteration's replacement.
    // After replaceAllUsesWith + eraseOp, the op is removed from its block.
    if (!tilingOp->getBlock())
      continue;

    // Verify it's still a linalg op with config (memory could be reused).
    if (!isa<linalg::LinalgOp>(tilingOp.getOperation()))
      continue;
    if (!getLoweringConfig(tilingOp.getOperation()))
      continue;

    DominanceInfo dominanceInfo(tilingOp);

    // Collect the transitive set of ops that will be tiled+fused with this
    // root. Used for yield replacement computation.
    llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
    collectTiledAndFusedOps(tilingOp.getOperation(), tiledAndFusedOps);

    // Compute yieldReplacementsFor: ops whose results have users dominated
    // by the root op need yield replacement to maintain correctness.
    // Mirrors IREE TileAndFuseUtils.cpp lines 287-296.
    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto *op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingOp.getOperation(),
                                                   user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    // Get tile sizes from config.
    SmallVector<OpFoldResult> tileSizes =
        getTileSizes(rewriter, tilingOp, tilingLevel);

    // If all tile sizes are 0, skip.
    bool anyNonZero = llvm::any_of(tileSizes, [](OpFoldResult ofr) {
      auto cst = getConstantIntValue(ofr);
      return cst && *cst != 0;
    });
    if (!anyNonZero)
      continue;

    // Build tiling options.
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    if (tilingLevel == NovaTilingLevel::Reduction) {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    } else {
      // Thread / Subgroup: parallel scf.forall with GPU mapping.
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // Partial reduction (parallelising K across warps/threads) is only
      // meaningful at Thread level. At Subgroup level the K dimension is
      // already sequential (scf.for from the Reduction pass) and
      // subgroupTiles[k]=0, so this path must never fire there — it would
      // produce warp-level partial reductions with no warp shuffle to combine.
      bool hasReductionTile = false;
      if (tilingLevel == NovaTilingLevel::Thread) {
        if (auto linalgOp =
                dyn_cast<linalg::LinalgOp>(tilingOp.getOperation())) {
          auto iterTypes = linalgOp.getIteratorTypesArray();
          for (int i = 0;
               i < (int)tileSizes.size() && i < (int)iterTypes.size(); ++i) {
            if (!isZeroInteger(tileSizes[i]) &&
                linalg::isReductionIterator(iterTypes[i])) {
              hasReductionTile = true;
              break;
            }
          }
        }
      }
      if (hasReductionTile) {
        tilingOptions.setReductionTilingStrategy(
            ReductionTilingStrategy::PartialReductionOuterParallel);
      }

      SmallVector<Attribute> mapping;
      int idx = 0;
      for (auto size : tileSizes) {
        if (isZeroInteger(size))
          continue;
        unsigned mappingId =
            static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
        // Mapping is determined purely by tiling level — no mmaKind check.
        // Subgroup level → always #gpu.warp (MMA ops, warp-granularity tiles).
        // Thread level   → always #gpu.thread (copy/fill/elementwise ops).
        // MMA ops have threadTiles=0 after NovaKernelConfig Change 1 so the
        // Thread pass never reaches this code for them; the guard is gone.
        if (tilingLevel == NovaTilingLevel::Subgroup) {
          mapping.push_back(gpu::GPUWarpMappingAttr::get(
              ctx, static_cast<gpu::MappingId>(mappingId)));
        } else {
          mapping.push_back(gpu::GPUThreadMappingAttr::get(
              ctx, static_cast<gpu::MappingId>(mappingId)));
        }
      }
      // Reverse so innermost dim gets LinearDim0 (fastest-moving).
      tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));
    }

    // Build fusion control function matching IREE's logic.
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      Operation *owner = originalProducer.getOwner();

      // Reduction level: do not fuse pad ops here; handled by cleanup pattern
      // (ExtractSliceOfPadTensorSwap) after tiling.
      // Subgroup + Thread levels: fuse pad ops inline so they land inside the
      // warp/thread forall body and are not left dangling outside it.
      if (tilingLevel == NovaTilingLevel::Reduction) {
        if (isa<tensor::PadOp>(owner))
          return std::nullopt;
      }

      // Subgroup level: do NOT fuse ops with derived_thread config into the
      // warp forall.  These are workgroup-level cooperative copies that must
      // remain at block scope so all threads cooperatively load the full tile.
      if (tilingLevel == NovaTilingLevel::Subgroup) {
        if (auto ownerConfig = getLoweringConfig(owner)) {
          if (isDerivedThreadConfig(ownerConfig))
            return std::nullopt;
        }
      }

      // Yield replacement: needed for ops with post-root users, but NOT
      // at reduction level (would yield large tensors).
      bool yieldProducerReplacement = false;
      if (tilingLevel != NovaTilingLevel::Reduction) {
        yieldProducerReplacement = yieldReplacementsFor.contains(owner);
      }

      // Core fusion decision: fuse if the producer is NOT independently
      // tiled at this level (i.e., not in the payloadOps set).
      bool shouldFuse = false;
      if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
        shouldFuse = !payloadOps.contains(tilingOwner);
      }

      // Do not fuse destination operands at reduction level.
      if (isDestinationOperand &&
          tilingLevel == NovaTilingLevel::Reduction) {
        shouldFuse = false;
      }

      // At Subgroup level, do NOT fuse ops marked nova.promote_to_workgroup
      // inside the warp forall. These are cooperative global→smem copies that
      // must remain at block scope so all threads cooperate on a single
      // workgroup-sized buffer. If fused inside the warp forall, each warp
      // iteration would write to the same per-warp-sized smem region and
      // overwrite each other's data (race condition → CUDA_ERROR_ILLEGAL_ADDRESS).
      if (tilingLevel == NovaTilingLevel::Subgroup &&
          owner->hasAttr("nova.promote_to_workgroup")) {
        shouldFuse = false;
      }

      if (shouldFuse) {
        return scf::SCFTileAndFuseOptions::ControlFnResult{
            yieldProducerReplacement};
      }
      return std::nullopt;
    };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    // Build cleanup patterns (passed to tileConsumerAndFuseProducersUsingSCF).
    RewritePatternSet cleanupPatterns(ctx);

    // Fuse pad without zero-slice guard for Reduction/Thread/Subgroup levels.
    if (tilingLevel == NovaTilingLevel::Reduction ||
        tilingLevel == NovaTilingLevel::Thread    ||
        tilingLevel == NovaTilingLevel::Subgroup) {
      cleanupPatterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
    ctx, [](tensor::ExtractSliceOp sliceOp) -> std::optional<bool> {
      auto padOp = sliceOp.getSource().getDefiningOp<tensor::PadOp>();
      if (!padOp) return false;
      for (Operation *user : padOp->getUsers())
        if (user->hasAttr("nova.promote_to_workgroup"))
          return std::nullopt; // skip this pattern entirely
      return false;
    });
    }

    // Skip cleanup for Subgroup level — IREE skips because lane tiling
    // follows and fusion must happen there.
    if (tilingLevel != NovaTilingLevel::Subgroup) {
      tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
          cleanupPatterns);
    }

    tileAndFuseOptions.cleanupPatterns =
        FrozenRewritePatternSet(std::move(cleanupPatterns));

    // Tile and fuse.
    rewriter.setInsertionPoint(tilingOp);
    FailureOr<scf::SCFTileAndFuseResult> result =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingOp,
                                                  tileAndFuseOptions);
    if (failed(result)) {
      tilingOp.emitWarning()
          << "nova-gpu-apply-tiling-level: tiling failed at level "
          << static_cast<int>(tilingLevel) << ", skipping\n";
      continue;
    }

    // Fix tile-local linalg.index in tensor.extract for reduction dims tiled
    // at the Thread level via PartialReductionOuterParallel.
    //
    // When tiling a reduction dim across threads, formal operands are correctly
    // sliced (the tile-start offset is baked into tensor.extract_slice), but
    // tensor.extract on non-operand tensors inside the linalg body still uses
    // linalg.index i with a tile-local value (0..tile_size-1). Those callers
    // need the global index (forall_iv + local), so we add the forall IV here.
    if (tilingLevel == NovaTilingLevel::Thread) {
      fixupReductionTiledLinalgIndex(
          rewriter, cast<linalg::LinalgOp>(tilingOp.getOperation()), *result,
          tileSizes);
    }

    // Replace original uses with tiled results.
    // Use replaceAllUsesWith (not dominance-based replaceUsesWithIf) to
    // ensure the original op becomes use_empty and can be erased. This is
    // safe because the tiled loop result dominates all original users.
    for (auto [origValue, replacement] : result->replacements)
      rewriter.replaceAllUsesWith(origValue, replacement);
  }
  return success();
}



static LogicalResult applyCleanupPatterns(func::FuncOp funcOp,
                                          NovaTilingLevel tilingLevel) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  // Fold empty tensors and merge consecutive insert/extract slices.
  tensor::populateFoldTensorEmptyPatterns(patterns);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitError("nova-gpu-apply-tiling-level: cleanup patterns failed\n");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass implementations — one per level
//===----------------------------------------------------------------------===//

template <typename DerivedPass>
struct NovaGPUApplyTilingLevelBase
    : public PassWrapper<DerivedPass, OperationPass<func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, affine::AffineDialect,
                    gpu::GPUDialect>();
  }

  void applyTilingLevel(NovaTilingLevel tilingLevel) {
    func::FuncOp funcOp = this->getOperation();
    IRRewriter rewriter(funcOp.getContext());

    auto targetOps = getTiledOps(funcOp, tilingLevel);
    if (targetOps.empty())
      return;

    if (failed(applyTileAndFuseToEachRoot(funcOp, rewriter, targetOps,
                                           tilingLevel))) {
      this->signalPassFailure();
      return;
    }

    if (failed(applyCleanupPatterns(funcOp, tilingLevel)))
      this->signalPassFailure();
  }
};

//===--- Reduction pass ---================================================//

struct NovaGPUApplyTilingLevelReductionPass final
    : NovaGPUApplyTilingLevelBase<NovaGPUApplyTilingLevelReductionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUApplyTilingLevelReductionPass)

  NovaGPUApplyTilingLevelReductionPass() = default;
  NovaGPUApplyTilingLevelReductionPass(
      const NovaGPUApplyTilingLevelReductionPass &pass)
      : NovaGPUApplyTilingLevelBase(pass) {}

  void runOnOperation() override {
    applyTilingLevel(NovaTilingLevel::Reduction);
  }

  StringRef getArgument() const override {
    return "nova-gpu-apply-tiling-level-reduction";
  }
  StringRef getDescription() const override {
    return "Tiles reduction (K) dimension into inner scf.for loop. "
           "Config-driven. Mirrors IREE GPUApplyTilingLevel(Reduction).";
  }
};

//===--- Thread pass ---===================================================//

struct NovaGPUApplyTilingLevelThreadPass final
    : NovaGPUApplyTilingLevelBase<NovaGPUApplyTilingLevelThreadPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUApplyTilingLevelThreadPass)

  NovaGPUApplyTilingLevelThreadPass() = default;
  NovaGPUApplyTilingLevelThreadPass(
      const NovaGPUApplyTilingLevelThreadPass &pass)
      : NovaGPUApplyTilingLevelBase(pass) {}

  void runOnOperation() override { applyTilingLevel(NovaTilingLevel::Thread); }

  StringRef getArgument() const override {
    return "nova-gpu-apply-tiling-level-thread";
  }
  StringRef getDescription() const override {
    return "Tiles M/N parallel dims to per-thread register tiles. "
           "Config-driven. Mirrors IREE GPUApplyTilingLevel(Thread).";
  }
};

//===--- Subgroup pass ---=================================================//

struct NovaGPUApplyTilingLevelSubgroupPass final
    : NovaGPUApplyTilingLevelBase<NovaGPUApplyTilingLevelSubgroupPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUApplyTilingLevelSubgroupPass)

  NovaGPUApplyTilingLevelSubgroupPass() = default;
  NovaGPUApplyTilingLevelSubgroupPass(
      const NovaGPUApplyTilingLevelSubgroupPass &pass)
      : NovaGPUApplyTilingLevelBase(pass) {}

  void runOnOperation() override { applyTilingLevel(NovaTilingLevel::Subgroup); }

  StringRef getArgument() const override {
    return "nova-gpu-apply-tiling-level-subgroup";
  }
  StringRef getDescription() const override {
    return "Tiles M/N parallel dims to per-subgroup (warp) tiles. "
           "Config-driven. Mirrors IREE GPUApplyTilingLevel(Subgroup).";
  }
};

//===----------------------------------------------------------------------===//
// Public factory functions & registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUApplyTilingLevelReductionPass() {
  return std::make_unique<NovaGPUApplyTilingLevelReductionPass>();
}
std::unique_ptr<Pass> createNovaGPUApplyTilingLevelThreadPass() {
  return std::make_unique<NovaGPUApplyTilingLevelThreadPass>();
}
std::unique_ptr<Pass> createNovaGPUApplyTilingLevelSubgroupPass() {
  return std::make_unique<NovaGPUApplyTilingLevelSubgroupPass>();
}

void registerNovaGPUApplyTilingLevelReductionPass() {
  PassRegistration<NovaGPUApplyTilingLevelReductionPass>();
}
void registerNovaGPUApplyTilingLevelThreadPass() {
  PassRegistration<NovaGPUApplyTilingLevelThreadPass>();
}
void registerNovaGPUApplyTilingLevelSubgroupPass() {
  PassRegistration<NovaGPUApplyTilingLevelSubgroupPass>();
}

} // namespace mlir::nova
