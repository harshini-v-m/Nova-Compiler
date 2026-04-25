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

enum class NovaTilingLevel {
  Reduction,
  Thread,
  Subgroup,
};

static constexpr int64_t kMaxThreadsPerBlock = 1024;

static StringRef getLevelKey(NovaTilingLevel level) {
  switch (level) {
  case NovaTilingLevel::Reduction: return kReductionKey;
  case NovaTilingLevel::Thread:    return kThreadKey;
  case NovaTilingLevel::Subgroup:  return kSubgroupKey;
  }
  llvm_unreachable("unknown tiling level");
}

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
      if (trips > worstTrips) { worstTrips = trips; worstDim = i; }
    }
    if (worstDim < 0 || worstTrips <= 1) break;
    threadTiles[worstDim] *= 2;
  }
}

//===----------------------------------------------------------------------===//
// getTiledOps — includes reference's warp-tiled guard for Subgroup level
//===----------------------------------------------------------------------===//

static SmallVector<TilingInterface>
getTiledOps(func::FuncOp funcOp, NovaTilingLevel tilingLevel) {
  SmallVector<TilingInterface> targets;
  StringRef levelKey = getLevelKey(tilingLevel);

  funcOp.walk([&](TilingInterface target) {
    if (!isa<linalg::LinalgOp>(target.getOperation()))
      return WalkResult::advance();

    auto config = getLoweringConfig(target.getOperation());
    if (!config)
      return WalkResult::advance();

    auto tiles = getLoweringConfigTileSizes(config, levelKey);
    if (tiles.empty() || !llvm::any_of(tiles, [](int64_t t) { return t > 0; }))
      return WalkResult::advance();

    // FROM REFERENCE: skip ops already inside a warp-mapped forall at
    // Subgroup level to prevent double-nested warp foralls on re-runs.
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
// getTileSizes — merged: yours + reference's warp-forall MMA zeroing,
//                wg/wg_subgroup computation, and blockDim derived thread
//===----------------------------------------------------------------------===//

static SmallVector<OpFoldResult>
getTileSizes(RewriterBase &rewriter, TilingInterface tilingOp,
             NovaTilingLevel tilingLevel) {
  Operation *op = tilingOp.getOperation();

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

  while ((int64_t)tiles.size() < numLoops)
    tiles.push_back(0);
  if ((int64_t)tiles.size() > numLoops)
    return SmallVector<OpFoldResult>(numLoops, zero);

  // FROM REFERENCE: at Thread level inside a warp forall, zero out MMA
  // contraction dims so no thread forall is created for them. The
  // vectorization + vector unrolling pipeline handles per-warp subdivision.
  if (tilingLevel == NovaTilingLevel::Thread) {
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

  // Derived-thread: recompute tiles from actual loop ranges + target_threads.
  // MERGED: reference passes blockDim to deriveThreadTileSizes, yours doesn't.
  // Using reference signature as it's more complete.
  if (tilingLevel == NovaTilingLevel::Thread && isDerivedThreadConfig(config)) {
    int64_t targetThreads = getTargetThreadCount(config);
    int64_t blockDim = getBlockDim(config);  // FROM REFERENCE
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
      unsigned elemBits =
          getElementTypeOrSelf(linalgOp->getResultTypes()[0])
              .getIntOrFloatBitWidth();
      // REFERENCE passes blockDim as extra arg
      tiles = deriveThreadTileSizes(loopRanges, targetThreads, elemBits,
                                    blockDim);
      while ((int64_t)tiles.size() < numLoops)
        tiles.push_back(0);
      tiles.resize(numLoops);

      LLVM_DEBUG(llvm::dbgs() << "[nova-tiling] derived thread tiles for "
                               << op->getName() << ": [";
                 for (int64_t t : tiles) llvm::dbgs() << t << " ";
                 llvm::dbgs() << "] target=" << targetThreads << "\n");
    }
  }

  // FROM REFERENCE: non-derived-thread at Thread level → use wg_tile /
  // wg_subgroup as the per-subgroup forall step instead of raw thread[d].
  if (tilingLevel == NovaTilingLevel::Thread && !isDerivedThreadConfig(config)) {
    auto wgTiles    = getLoweringConfigTileSizes(config, kWorkgroupKey);
    auto wgSubgroup = getLoweringConfigTileSizes(config, kWgSubgroupKey);
    if (!wgTiles.empty() && !wgSubgroup.empty()) {
      for (int64_t i = 0; i < (int64_t)tiles.size(); ++i) {
        if (tiles[i] == 0) continue;
        int64_t sg = (i < (int64_t)wgSubgroup.size()) ? wgSubgroup[i] : 1;
        int64_t wg = (i < (int64_t)wgTiles.size())    ? wgTiles[i]    : 0;
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

  // Subgroup: validate tile sizes — guard divisibility and warp count.
  if (tilingLevel == NovaTilingLevel::Subgroup) {
    auto wgTiles = getLoweringConfigTileSizes(config, kWorkgroupKey);

    for (int64_t i = 0; i < (int64_t)tiles.size(); ++i) {
      if (tiles[i] <= 0) continue;
      int64_t wg = (i < (int64_t)wgTiles.size()) ? wgTiles[i] : 0;
      if (wg <= 0) continue;
      if (tiles[i] > wg) tiles[i] = wg;
      while (tiles[i] > 1 && wg % tiles[i] != 0)
        --tiles[i];
    }

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

//===----------------------------------------------------------------------===//
// fixupReductionTiledLinalgIndex — FROM REFERENCE, missing in yours entirely
//===----------------------------------------------------------------------===//

static void fixupReductionTiledLinalgIndex(IRRewriter &rewriter,
                                           linalg::LinalgOp originalOp,
                                           scf::SCFTileAndFuseResult &result,
                                           ArrayRef<OpFoldResult> tileSizes) {
  scf::ForallOp forallOp;
  for (LoopLikeOpInterface loop : result.loops) {
    if (auto fop = dyn_cast<scf::ForallOp>(loop.getOperation())) {
      forallOp = fop;
      break;
    }
  }
  if (!forallOp)
    return;

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
      Value globalIdx =
          rewriter.create<arith::AddIOp>(indexOp.getLoc(), iv, localIdx);
      localIdx.replaceUsesWithIf(globalIdx, [&](OpOperand &use) {
        return use.getOwner() != globalIdx.getDefiningOp();
      });
    }
  }
}

//===----------------------------------------------------------------------===//
// applyTileAndFuseToEachRoot
//===----------------------------------------------------------------------===//

static LogicalResult
applyTileAndFuseToEachRoot(func::FuncOp funcOp, IRRewriter &rewriter,
                           SmallVector<TilingInterface> &targetOps,
                           NovaTilingLevel tilingLevel) {
  MLIRContext *ctx = funcOp.getContext();

  llvm::SmallDenseSet<TilingInterface> payloadOps(targetOps.begin(),
                                                   targetOps.end());

  for (TilingInterface tilingOp : targetOps) {
    if (!tilingOp->getBlock())
      continue;
    if (!isa<linalg::LinalgOp>(tilingOp.getOperation()))
      continue;
    if (!getLoweringConfig(tilingOp.getOperation()))
      continue;

    DominanceInfo dominanceInfo(tilingOp);

    llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
    collectTiledAndFusedOps(tilingOp.getOperation(), tiledAndFusedOps);

    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto *op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingOp.getOperation(),
                                                   user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    SmallVector<OpFoldResult> tileSizes =
        getTileSizes(rewriter, tilingOp, tilingLevel);

    bool anyNonZero = llvm::any_of(tileSizes, [](OpFoldResult ofr) {
      auto cst = getConstantIntValue(ofr);
      return cst && *cst != 0;
    });
    if (!anyNonZero)
      continue;

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    if (tilingLevel == NovaTilingLevel::Reduction) {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    } else {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // Partial reduction only at Thread level (reference guards this).
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
        if (tilingLevel == NovaTilingLevel::Subgroup) {
          mapping.push_back(gpu::GPUWarpMappingAttr::get(
              ctx, static_cast<gpu::MappingId>(mappingId)));
        } else {
          mapping.push_back(gpu::GPUThreadMappingAttr::get(
              ctx, static_cast<gpu::MappingId>(mappingId)));
        }
      }
      tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      Operation *owner = originalProducer.getOwner();

      // REFERENCE: only block pad fusion at Reduction level.
      // Thread and Subgroup levels fuse pads inline.
      if (tilingLevel == NovaTilingLevel::Reduction) {
        if (isa<tensor::PadOp>(owner))
          return std::nullopt;
      }

      // REFERENCE: at Subgroup level, don't fuse derived_thread ops
      // (cooperative copies must stay at block scope).
      if (tilingLevel == NovaTilingLevel::Subgroup) {
        if (auto ownerConfig = getLoweringConfig(owner)) {
          if (isDerivedThreadConfig(ownerConfig))
            return std::nullopt;
        }
      }

      bool yieldProducerReplacement = false;
      if (tilingLevel != NovaTilingLevel::Reduction) {
        yieldProducerReplacement = yieldReplacementsFor.contains(owner);
      }

      bool shouldFuse = false;
      if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
        shouldFuse = !payloadOps.contains(tilingOwner);
      }

      if (isDestinationOperand &&
          tilingLevel == NovaTilingLevel::Reduction) {
        shouldFuse = false;
      }

      // REFERENCE: don't fuse nova.promote_to_workgroup ops inside warp
      // forall — would cause races on shared memory.
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

    RewritePatternSet cleanupPatterns(ctx);

    // REFERENCE: fuse pads at all levels but guard promote_to_workgroup.
    if (tilingLevel == NovaTilingLevel::Reduction ||
        tilingLevel == NovaTilingLevel::Thread    ||
        tilingLevel == NovaTilingLevel::Subgroup) {
      cleanupPatterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
          ctx, [](tensor::ExtractSliceOp sliceOp) -> std::optional<bool> {
            auto padOp = sliceOp.getSource().getDefiningOp<tensor::PadOp>();
            if (!padOp) return false;
            for (Operation *user : padOp->getUsers())
              if (user->hasAttr("nova.promote_to_workgroup"))
                return std::nullopt;
            return false;
          });
    }

    if (tilingLevel != NovaTilingLevel::Subgroup) {
      tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
          cleanupPatterns);
    }

    tileAndFuseOptions.cleanupPatterns =
        FrozenRewritePatternSet(std::move(cleanupPatterns));

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

    // REFERENCE: fixup linalg.index for reduction dims tiled at Thread level.
    if (tilingLevel == NovaTilingLevel::Thread) {
      fixupReductionTiledLinalgIndex(
          rewriter, cast<linalg::LinalgOp>(tilingOp.getOperation()), *result,
          tileSizes);
    }

    for (auto [origValue, replacement] : result->replacements)
      rewriter.replaceAllUsesWith(origValue, replacement);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// applyCleanupPatterns — REFERENCE: always runs including Subgroup level
//===----------------------------------------------------------------------===//

static LogicalResult applyCleanupPatterns(func::FuncOp funcOp,
                                          NovaTilingLevel tilingLevel) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

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
// Pass base — REFERENCE: always runs cleanup (no Subgroup skip)
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

    // REFERENCE: always run cleanup, no Subgroup skip.
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