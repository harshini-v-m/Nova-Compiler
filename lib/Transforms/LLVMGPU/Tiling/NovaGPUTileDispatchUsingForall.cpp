#include "Passes.h"
#include "NovaGPUTileAndFuseUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaKernelConfig.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-tile-dispatch"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Returns true if the operation implements TilingInterface.
static bool isComputeOp(Operation *op) {
  return isa<TilingInterface>(op);
}

/// Returns true if the op is a contraction-like op (e.g. matmul).
static bool isContractionOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && linalg::isaContractionOpInterface(linalgOp);
}

/// Returns true if the op is inside a scf.forall with GPU block mapping.
static bool isInsideWorkgroupForall(Operation *op) {
  auto parent = op->getParentOfType<scf::ForallOp>();
  while (parent) {
    auto mapping = parent.getMappingAttr();
    if (mapping && llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUBlockMappingAttr>(attr);
        })) {
      return true;
    }
    parent = parent->getParentOfType<scf::ForallOp>();
  }
  return false;
}

/// Collects all compute operations in the function.
static SmallVector<Operation *> getComputeOps(func::FuncOp funcOp) {
  SmallVector<Operation *> computeOps;
  funcOp.walk([&](Operation *op) {
    if (isComputeOp(op))
      computeOps.push_back(op);
  });
  return computeOps;
}

/// Returns true if it is allowed to leave the op outside distribution loops.
static bool isAllowedToFailOnConsumerFusion(Operation *op) {
  return isa<linalg::PackOp>(op);
}

/// Returns true if all compute ops are within scf.forall distribution loops.
static bool verifyComputeOpsAfterDistribution(func::FuncOp funcOp) {
  WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::ForallOp>(op) || !isComputeOp(op))
      return WalkResult::skip();
    if (!isAllowedToFailOnConsumerFusion(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !res.wasInterrupted();
}

/// Returns true if any value produced by `producer` is used as an init value
/// for the DPS `user`.
static bool isUsedAsInit(Operation *producer, Operation *user) {
  auto dpsIface = dyn_cast<DestinationStyleOpInterface>(user);
  if (!dpsIface)
    return false;
  ValueRange results = producer->getResults();
  return llvm::any_of(dpsIface.getDpsInits(), [&](Value operand) {
    return llvm::is_contained(results, operand);
  });
}

/// Creates GPU block mapping attributes for non-zero tile dimensions.
/// For ≤3 active dims: uses DimX/DimY/DimZ (innermost first).
/// For >3 active dims: uses LinearDim0..N (innermost first, reversed).
static SmallVector<Attribute> getMapping(MLIRContext *context,
                                          ArrayRef<OpFoldResult> tileSizes) {
  int numActiveDims = 0;
  for (auto tileSize : tileSizes) {
    std::optional<int64_t> cst = getConstantIntValue(tileSize);
    if (!cst || *cst != 0)
      numActiveDims++;
  }

  SmallVector<Attribute> mapping;

  if (numActiveDims > 3) {
    unsigned idx = 0;
    for (auto tileSize : tileSizes) {
      std::optional<int64_t> cst = getConstantIntValue(tileSize);
      if (cst && *cst == 0)
        continue;
      unsigned mappingId =
          static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
      mapping.push_back(gpu::GPUBlockMappingAttr::get(
          context, static_cast<gpu::MappingId>(mappingId)));
    }
    return llvm::to_vector(llvm::reverse(mapping));
  }

  // ≤3 active dimensions: use 3D block mapping (x, y, z).
  int dim = 0;
  for (auto tileSize : llvm::reverse(tileSizes)) {
    std::optional<int64_t> cst = getConstantIntValue(tileSize);
    if (cst && *cst == 0)
      continue;
    switch (dim) {
    case 0:
      mapping.push_back(gpu::GPUBlockMappingAttr::get(
          context, gpu::MappingId::DimX));
      break;
    case 1:
      mapping.push_back(gpu::GPUBlockMappingAttr::get(
          context, gpu::MappingId::DimY));
      break;
    case 2:
      mapping.push_back(gpu::GPUBlockMappingAttr::get(
          context, gpu::MappingId::DimZ));
      break;
    default:
      break;
    }
    dim++;
  }
  return llvm::to_vector(llvm::reverse(mapping));
}

//===----------------------------------------------------------------------===//
// Tiling info extraction
//===----------------------------------------------------------------------===//

struct TilingInfo {
  Operation *tilableOp;
  SmallVector<OpFoldResult> tileSizes;
};

/// Reads workgroup tile sizes from the lowering_config attribute.
/// Zeros out reduction dims (workgroup tiling is parallel-only).
/// Applies full-tile optimization (zero tile when loopSize == tileSize).
static FailureOr<TilingInfo> getTiledAndDistributionInfo(
    RewriterBase &rewriter, Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return failure();

  int numLoops = linalgOp.getNumLoops();
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));

  // Read workgroup tile sizes from lowering_config.
  auto config = getLoweringConfig(op);
  if (!config)
    return failure(); // No config → can't tile.

  SmallVector<int64_t> wgTiles =
      getLoweringConfigTileSizes(config, kWorkgroupKey);
  if (wgTiles.size() != static_cast<size_t>(numLoops))
    return failure();

  for (int i = 0; i < numLoops; ++i)
    tileSizes[i] = rewriter.getIndexAttr(wgTiles[i]);

  // Zero out non-parallel (reduction) dims at workgroup level.
  for (int i = 0; i < numLoops; ++i) {
    if (!linalg::isParallelIterator(linalgOp.getIteratorTypesArray()[i]))
      tileSizes[i] = rewriter.getIndexAttr(0);
  }

  // Full-tile optimization: zero tile size when staticLoopSize == tileSize.
  // Prevents single-trip scf.forall loops that block cleanup patterns.
  // Keep at least one non-zero tile so the forall loop is created.
  // Exception: ops with reduction iterators must stay in a forall for GPU.
  bool hasReductionIter = llvm::any_of(linalgOp.getIteratorTypesArray(),
                                        linalg::isReductionIterator);
  if (!hasReductionIter) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    auto tilingIface = cast<TilingInterface>(op);
    SmallVector<Range> bounds = tilingIface.getIterationDomain(rewriter);

    int numNonZero = 0;
    for (auto &ts : tileSizes) {
      if (auto cst = getConstantIntValue(ts))
        if (*cst != 0)
          ++numNonZero;
    }

    for (int i = (int)tileSizes.size() - 1; i >= 0; --i) {
      if (numNonZero <= 1)
        break;
      auto tsCst = getConstantIntValue(tileSizes[i]);
      if (!tsCst || *tsCst == 0)
        continue;
      if (i >= (int)bounds.size())
        continue;
      auto boundCst = getConstantIntValue(bounds[i].size);
      if (boundCst && *boundCst == *tsCst) {
        tileSizes[i] = rewriter.getIndexAttr(0);
        --numNonZero;
      }
    }
  }

  return TilingInfo{op, tileSizes, };
}

//===----------------------------------------------------------------------===//
// Main Pass
//===----------------------------------------------------------------------===//

struct NovaTileAndDistributePass
    : public PassWrapper<NovaTileAndDistributePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaTileAndDistributePass)

  NovaTileAndDistributePass() = default;
  NovaTileAndDistributePass(const NovaTileAndDistributePass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());

    // =====================================================================
    // Two-phase tiling strategy (simplified from previous 3-pass):
    //
    // Phase 1: Tile all ops that have a lowering_config attribute.
    //          Process in reverse order (innermost compute first) so that
    //          producer fusion pulls upstream ops into the tile loop.
    //          Consumer fusion pulls downstream epilogues (bias, relu).
    //
    // Phase 2: Wrap any remaining un-distributed compute ops in a
    //          single-block scf.forall (for full-reduction chains that
    //          have all-zero workgroup tiles).
    // =====================================================================

    llvm::SmallPtrSet<Operation *, 16> handledOps;

    // --- Helper: tile a root op with producer + optional consumer fusion ----
    // When doConsumerFusion=false, only producer fusion happens. This is used
    // for contractions: we tile ALL contractions first (without consumer
    // fusion), then do consumer fusion in a separate pass. This prevents
    // tileAndFuseConsumerOfSlices from dragging sibling contractions into
    // the forall (the MLIR utility moves intervening ops between the forall
    // and the consumer into the forall body).
    auto tileRoot = [&](Operation *rootOp, bool &didTile,
                        bool doConsumerFusion = true) -> LogicalResult {
      didTile = false;
      if (handledOps.count(rootOp))
        return success();

      auto infoOr = getTiledAndDistributionInfo(rewriter, rootOp);
      if (failed(infoOr))
        return success(); // Skip non-tilable ops gracefully.
      TilingInfo info = *infoOr;

      // Check if all tile sizes are zero (full reduction, no parallel dims).
      // These are handled by Phase 2.
      bool allZero = true;
      for (auto &ts : info.tileSizes) {
        if (auto cst = getConstantIntValue(ts)) {
          if (*cst != 0) {
            allZero = false;
            break;
          }
        } else {
          allZero = false;
          break;
        }
      }
      if (allZero)
        return success(); // Defer to Phase 2.

      auto tilingInterface = cast<TilingInterface>(rootOp);

      // Collect fusion cluster.
      llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
      collectTiledAndFusedOps(rootOp, tiledAndFusedOps);

      LLVM_DEBUG(llvm::dbgs() << "[nova-tile-dispatch] Root: "
                             << rootOp->getName()
                             << " cluster=" << tiledAndFusedOps.size() << "\n");

      DominanceInfo dominanceInfo(rootOp);
      llvm::DenseSet<Operation *> yieldReplacementsFor;
      for (auto op : tiledAndFusedOps) {
        if (llvm::any_of(op->getUsers(), [&](Operation *user) {
              if (isUsedAsInit(op, user))
                return false;
              return dominanceInfo.properlyDominates(rootOp, user) ||
                     !tiledAndFusedOps.contains(user);
            })) {
          yieldReplacementsFor.insert(op);
        }
      }

      // Configure tiling options.
      scf::SCFTilingOptions tilingOptions;
      tilingOptions.setTileSizes(info.tileSizes);
      auto mapping = getMapping(&getContext(), info.tileSizes);
      tilingOptions.setMapping(mapping);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // Fusion control: fuse producers except pad ops and contractions.
      scf::SCFTileAndFuseOptions tileAndFuseOptions;
      tileAndFuseOptions.setTilingOptions(tilingOptions);

      tileAndFuseOptions.setFusionControlFn(
          [&](tensor::ExtractSliceOp sliceOp, OpResult producer,
              bool isDest)
              -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
            Operation *producerOp = producer.getOwner();
            // Don't fuse pad ops into the workgroup loop.
            if (isa<tensor::PadOp>(producerOp)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[nova-tile-dispatch] Fusion BLOCKED (pad): "
                         << producerOp->getName() << "\n");
              return std::nullopt;
            }
            // Contractions stay as independent roots.
            if (isContractionOp(producerOp)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[nova-tile-dispatch] Fusion BLOCKED (contraction): "
                         << producerOp->getName() << "\n");
              return std::nullopt;
            }
            // Producers that have their own lowering_config are scheduled to be
            // tiled independently in Phase 1b.  Fusing them here would
            // recompute the same tensor once per root forall (e.g., the
            // embedding gather op feeding LN-mean, LN-var, LN-normalize, and
            // the residual-add — 7× redundant scatter-reads from global memory).
            // Let Phase 1b tile these producers into their own foralls; roots
            // will read from the materialized result via extract_slice.
            if (getLoweringConfig(producerOp)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[nova-tile-dispatch] Fusion BLOCKED (has own "
                            "lowering_config): "
                         << producerOp->getName() << "\n");
              return std::nullopt;
            }
            // Don't fuse producers that have reduction iterator types.
            // Reductions (e.g. LayerNorm mean/variance, softmax sum) need
            // to see ALL elements along the reduction dimension. When fused
            // as a producer into a consumer's tiled forall, each workgroup
            // computes a PARTIAL reduction over its tile — producing wrong
            // results and racing on the reduction output buffer.
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(producerOp)) {
              for (auto iterType : linalgOp.getIteratorTypesArray()) {
                if (iterType != utils::IteratorType::parallel) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "[nova-tile-dispatch] Fusion BLOCKED "
                             << "(has reduction): "
                             << producerOp->getName() << "\n");
                  return std::nullopt;
                }
              }
            }
            // Don't fuse a producer that has outside users (would create a
            // shared_out in the forall) if its slice doesn't depend on ALL of
            // the forall's induction variables. Such a producer is independent
            // of some tiled dimension — every tile of that dimension would
            // recompute the same result AND write it to the same shared_out
            // slot (benign race, but O(N-tiles) redundant work).
            //
            // Example: layernorm output (8×1024×384) fused into a
            // (batch=1, tokens=128, N=64) matmul forall. The extract_slice
            // uses IVs arg51 (batch), arg52 (tokens) but NOT arg53 (N), so
            // all 24 N-tiles compute identical layernorms. Block fusion so the
            // layernorm gets its own (batch, tokens) kernel (Phase 1a-3).
            if (yieldReplacementsFor.contains(producerOp)) {
              auto parentForall =
                  sliceOp->getParentOfType<scf::ForallOp>();
              if (parentForall) {
                auto ivs = parentForall.getInductionVars();
                if (ivs.size() > 1) {
                  // BFS backward from each offset/size value to see if all
                  // forall IVs are reachable. This handles multi-level chains
                  // like affine.apply(affine.apply(iv, stride), offset) or
                  // arith.addi(arith.muli(iv, c), base) that the old one-level
                  // expansion would miss, causing unnecessary fusion blocks.
                  SmallPtrSet<Value, 32> visited;
                  SmallVector<Value> worklist;
                  auto enqueue = [&](Value v) {
                    if (visited.insert(v).second)
                      worklist.push_back(v);
                  };
                  for (auto ofr : sliceOp.getMixedOffsets())
                    if (auto val = dyn_cast<Value>(ofr))
                      enqueue(val);
                  for (auto ofr : sliceOp.getMixedSizes())
                    if (auto val = dyn_cast<Value>(ofr))
                      enqueue(val);
                  while (!worklist.empty()) {
                    Value v = worklist.pop_back_val();
                    // Stop at block arguments (IVs are block args of forall).
                    if (isa<BlockArgument>(v))
                      continue;
                    if (auto *defOp = v.getDefiningOp()) {
                      // Don't cross forall boundaries or region ops.
                      if (defOp->getParentRegion() !=
                          sliceOp->getParentRegion())
                        continue;
                      for (Value operand : defOp->getOperands())
                        enqueue(operand);
                    }
                  }
                  bool allIVsUsed = llvm::all_of(
                      ivs, [&](Value iv) {
                        return visited.contains(iv);
                      });
                  if (!allIVsUsed) {
                    LLVM_DEBUG(llvm::dbgs()
                               << "[nova-tile-dispatch] Fusion BLOCKED "
                               << "(outside users + slice omits forall IV): "
                               << producerOp->getName() << "\n");
                    return std::nullopt;
                  }
                }
              }
            }

            LLVM_DEBUG(llvm::dbgs()
                       << "[nova-tile-dispatch] Fusion ALLOWED: "
                       << producerOp->getName() << "\n");
            bool yieldProducerReplacement =
                yieldReplacementsFor.contains(producerOp);
            return scf::SCFTileAndFuseOptions::ControlFnResult{
                yieldProducerReplacement};
          });

      // Cleanup patterns.
      RewritePatternSet cleanupPatterns(&getContext());
      tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns,
                                                         &getContext());
      tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns,
                                                &getContext());
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
          cleanupPatterns);
      cleanupPatterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
          &getContext(), [](tensor::ExtractSliceOp) { return false; });
      tileAndFuseOptions.cleanupPatterns =
          FrozenRewritePatternSet(std::move(cleanupPatterns));

      // Execute tile & fuse.
      FailureOr<scf::SCFTileAndFuseResult> result;
      if (rootOp->getNumResults() > 0) {
        result = scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, tilingInterface, tileAndFuseOptions);
      } else {
        auto tileResult =
            scf::tileUsingSCF(rewriter, tilingInterface, tilingOptions);
        if (succeeded(tileResult))
          rewriter.eraseOp(rootOp);
        didTile = true;
        return success();
      }

      if (succeeded(result)) {
        didTile = true;
        handledOps.insert(rootOp);
        for (auto op : result->tiledAndFusedOps) {
          handledOps.insert(op);
          // Strip lowering_config from fused non-root ops (defense-in-depth).
          // Only roots should retain configs for thread tiling.
          // Check by op type: all-parallel ops are non-roots that fused in.
          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            bool isAllParallel = llvm::all_of(
                linalgOp.getIteratorTypesArray(),
                [](auto t) { return t == utils::IteratorType::parallel; });
            if (isAllParallel && getLoweringConfig(op))
              removeLoweringConfig(op);
          }
        }
      } else {
        return failure();
      }

      // Replace results (with dominance check).
      for (auto [origValue, replacement] : result->replacements) {
        Value replacementCopy = replacement;
        rewriter.replaceUsesWithIf(
            origValue, replacement, [&](OpOperand &use) {
              Operation *user = use.getOwner();
              return !isa<tensor::DimOp>(user) &&
                     dominanceInfo.dominates(replacementCopy, user);
            });
      }

      // Consumer fusion (epilogue: bias, relu, etc.).
      // Skipped for contractions in Phase 1a — deferred to Phase 1a-2 after
      // all contractions are tiled (prevents dragging sibling contractions).
      if (doConsumerFusion) {
        SmallVector<LoopLikeOpInterface> loops = result->loops;
        if (!result->tiledAndFusedOps.empty() && !loops.empty()) {
          FailureOr<std::queue<Operation *>> newFusionOpportunities =
              fuseConsumersIntoForall(
                  rewriter, result->tiledAndFusedOps.getArrayRef(), loops,
                  [&](Operation *op) {
                    if (isContractionOp(op))
                      return false;
                    return tiledAndFusedOps.contains(op);
                  });
          if (succeeded(newFusionOpportunities)) {
            fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                                 tileAndFuseOptions, loops);
          }
          // Strip configs from fused all-parallel consumer ops (they are now
          // inside the root's forall and should not get independent thread
          // foralls). Root ops (contractions/reductions) keep their configs.
          for (auto op : result->tiledAndFusedOps) {
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
              bool isAllParallel = llvm::all_of(
                  linalgOp.getIteratorTypesArray(),
                  [](auto t) { return t == utils::IteratorType::parallel; });
              if (isAllParallel && getLoweringConfig(op))
                removeLoweringConfig(op);
            }
          }
        }
      }
      return success();
    };

    // --- Phase 1a: Tile contraction ops (matmul) first ----
    // For each contraction, check if consumer fusion is safe: no untiled
    // contractions sit between the forall and its consumers in IR order.
    // If safe, fuse consumers immediately (e.g. bias+relu epilogue).
    // If unsafe (sibling contraction intervenes), defer to Phase 1a-2.
    {
      bool foundOne = true;
      while (foundOne) {
        foundOne = false;
        SmallVector<Operation *> computeOps = getComputeOps(funcOp);
        for (Operation *rootOp : computeOps) {
          if (isInsideWorkgroupForall(rootOp))
            continue;
          if (handledOps.count(rootOp))
            continue;
          if (!getLoweringConfig(rootOp))
            continue;
          if (!isContractionOp(rootOp))
            continue;

          // Check if consumer fusion is safe: walk ops in the parent block
          // after rootOp. If we reach a consumer of rootOp's results before
          // hitting an untiled contraction, it's safe to fuse consumers.
          // If an untiled contraction appears first, defer consumer fusion.
          bool safeForConsumerFusion = true;
          Block *parentBlock = rootOp->getBlock();
          if (parentBlock) {
            // Collect rootOp's result users (potential consumers).
            llvm::SmallPtrSet<Operation *, 8> consumers;
            for (Value result : rootOp->getResults()) {
              for (Operation *user : result.getUsers()) {
                if (user != rootOp)
                  consumers.insert(user);
              }
            }
            // Walk forward from rootOp in the block.
            bool foundConsumerFirst = false;
            bool foundContractionFirst = false;
            for (auto it = std::next(rootOp->getIterator()),
                      end = parentBlock->end();
                 it != end; ++it) {
              Operation *op = &*it;
              if (consumers.contains(op)) {
                foundConsumerFirst = true;
                break;
              }
              if (isContractionOp(op) && !isInsideWorkgroupForall(op) &&
                  !handledOps.count(op)) {
                foundContractionFirst = true;
                break;
              }
            }
            // Unsafe only when an untiled contraction appears before any
            // consumer — the MLIR consumer fusion utility would drag it in.
            if (foundContractionFirst && !foundConsumerFirst)
              safeForConsumerFusion = false;
          }

          LLVM_DEBUG(llvm::dbgs()
                     << "[nova-tile-dispatch] Phase 1a: tiling contraction "
                     << rootOp->getName()
                     << " consumerFusion="
                     << (safeForConsumerFusion ? "immediate" : "deferred")
                     << "\n");

          bool didTile = false;
          if (failed(tileRoot(rootOp, didTile,
                              /*doConsumerFusion=*/safeForConsumerFusion))) {
            signalPassFailure();
            return;
          }
          if (didTile) {
            foundOne = true;
            break; // Re-collect.
          }
          handledOps.insert(rootOp);
        }
      }
    }

    // --- Phase 1a-2: Deferred consumer fusion for contraction foralls ----
    // For contractions where consumer fusion was deferred (unsafe due to
    // sibling contractions), now that all contractions are in their own
    // foralls, consumer fusion can safely fuse epilogues.
    {
      SmallVector<scf::ForallOp> contractionForalls;
      funcOp.walk([&](scf::ForallOp forall) {
        auto mapping = forall.getMappingAttr();
        if (!mapping || !llvm::any_of(mapping.getValue(), [](Attribute attr) {
              return isa<gpu::GPUBlockMappingAttr>(attr);
            }))
          return;
        bool hasContraction = false;
        forall.walk([&](linalg::LinalgOp linalgOp) {
          if (linalg::isaContractionOpInterface(linalgOp))
            hasContraction = true;
        });
        if (hasContraction)
          contractionForalls.push_back(forall);
      });

      for (auto forall : contractionForalls) {
        // Check if this forall already has non-contraction consumers fused.
        // If so, consumer fusion was already done in Phase 1a (safe path).
        bool hasEpilogue = false;
        forall.walk([&](Operation *op) {
          if (isa<TilingInterface>(op) && !isContractionOp(op))
            hasEpilogue = true;
        });
        if (hasEpilogue)
          continue; // Already fused in Phase 1a.

        SmallVector<Operation *> tiledOps;
        forall.walk([&](Operation *op) {
          if (isa<TilingInterface>(op))
            tiledOps.push_back(op);
        });
        if (tiledOps.empty())
          continue;

        SmallVector<LoopLikeOpInterface> loops = {
            cast<LoopLikeOpInterface>(forall.getOperation())};

        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-tile-dispatch] Phase 1a-2: deferred consumer "
                   << "fusion for contraction forall\n");

        FailureOr<std::queue<Operation *>> newFusionOpportunities =
            fuseConsumersIntoForall(
                rewriter, tiledOps, loops,
                [&](Operation *op) {
                  if (isContractionOp(op))
                    return false;
                  return isa<TilingInterface>(op);
                });

        if (succeeded(newFusionOpportunities) &&
            !newFusionOpportunities->empty()) {
          scf::SCFTileAndFuseOptions fakeOptions;
          fakeOptions.setFusionControlFn(
              [](tensor::ExtractSliceOp, OpResult, bool)
                  -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
                return scf::SCFTileAndFuseOptions::ControlFnResult{false};
              });
          fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                               fakeOptions, loops);
        }
        // Strip configs from all-parallel ops inside the forall
        // (they are now fused and should not get independent thread foralls).
        // Root ops (contractions/reductions) keep their configs.
        forall.walk([&](linalg::LinalgOp linalgOp) {
          bool isAllParallel = llvm::all_of(
              linalgOp.getIteratorTypesArray(),
              [](auto t) { return t == utils::IteratorType::parallel; });
          if (isAllParallel && getLoweringConfig(linalgOp.getOperation()))
            removeLoweringConfig(linalgOp.getOperation());
        });
      }
    }

    // --- Phase 1a-3: Rescue prologues that failed producer fusion -----------
    // Prologues (LN normalize, weight broadcast, etc.) were correctly
    // identified in SelectLoweringStrategy and skipped (no config), expecting
    // to be pulled into contraction foralls via producer fusion in Phase 1a.
    // If producer fusion failed (MLIR tileAndFuseProducerOfSlice returned
    // std::nullopt — e.g., non-contiguous slice, dynamic shape, or the
    // producer's result isn't a direct extract_slice source), the prologue
    // is still outside any forall with no config → Phase 2 serializes it.
    //
    // Fix: stamp a default config on these stranded ops so Phase 1b tiles
    // them independently (multi-block). Suboptimal vs. fusion (extra global
    // memory round-trip), but orders of magnitude better than single-block.
    {
      // Use sm_86 target for default config computation. The target is only
      // used for padding heuristics; tile sizes use hardcoded constants.
      NVIDIATargetInfo rescueTarget = getNVIDIATargetInfo("sm_86");
      funcOp.walk([&](linalg::LinalgOp op) {
        Operation *rawOp = op.getOperation();
        // Only rescue ops that are outside foralls and have no config.
        if (isInsideWorkgroupForall(rawOp))
          return;
        if (getLoweringConfig(rawOp))
          return;
        // FillOps always fuse as DPS inits — leave them for Phase 2.
        if (isa<linalg::FillOp>(rawOp))
          return;
        // Only rescue ops with tensor results (Phase 2 requirement).
        if (rawOp->getNumResults() == 0)
          return;
        auto resultType =
            dyn_cast<RankedTensorType>(rawOp->getResult(0).getType());
        if (!resultType)
          return;
        // Only rescue large ops (small ones are fine single-block).
        int64_t numElements = 1;
        for (int64_t dim : resultType.getShape()) {
          if (!ShapedType::isDynamic(dim))
            numElements *= dim;
        }
        if (numElements <= 1024)
          return;

        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-tile-dispatch] Phase 1a-3: rescuing unfused "
                   << rawOp->getName() << " (" << numElements
                   << " elements) with default config\n");
        (void)setDefaultConfig(op, rescueTarget);
      });
    }

    // --- Phase 1b: Tile remaining ops with lowering_config in reverse order -
    // Non-contraction roots (standalone reductions, elementwise that aren't
    // epilogues) get tiled after all contractions are distributed.
    // Consumer fusion is enabled since all contractions are already in foralls.
    {
      bool foundOne = true;
      while (foundOne) {
        foundOne = false;
        SmallVector<Operation *> computeOps = getComputeOps(funcOp);
        for (Operation *rootOp : llvm::reverse(computeOps)) {
          if (isInsideWorkgroupForall(rootOp))
            continue;
          if (handledOps.count(rootOp))
            continue;
          if (!getLoweringConfig(rootOp))
            continue;
          bool didTile = false;
          if (failed(tileRoot(rootOp, didTile,
                              /*doConsumerFusion=*/true))) {
            signalPassFailure();
            return;
          }
          if (didTile) {
            foundOne = true;
            break; // Re-collect.
          }
          handledOps.insert(rootOp);
        }
      }
    }

    // --- Phase 2: Wrap remaining un-distributed ops in single-block forall
    // For full-reduction ops (all-zero workgroup tiles) and any compute ops
    // that didn't get tiled in Phase 1. Each untiled op gets individually
    // wrapped in a scf.forall(0 to 1) with GPU block mapping.
    {
      SmallVector<Operation *> computeOps = getComputeOps(funcOp);
      for (Operation *op : computeOps) {
        if (isInsideWorkgroupForall(op))
          continue;
        if (op->getParentOp() != funcOp.getOperation())
          continue;
        if (op->getNumResults() == 0)
          continue;

        // Collect ranked-tensor result types. For multi-output ops (e.g. a
        // horizontally-fused linalg.generic with N outputs) we must wire ALL
        // results through the forall — one shared_out and one
        // parallel_insert_slice per result. Skip the op if any result is not
        // a ranked tensor (no sensible forall wrapping possible).
        SmallVector<RankedTensorType> resultTypes;
        for (Value res : op->getResults()) {
          auto rt = dyn_cast<RankedTensorType>(res.getType());
          if (!rt) {
            resultTypes.clear();
            break;
          }
          resultTypes.push_back(rt);
        }
        if (resultTypes.empty())
          continue;

        RankedTensorType resultType = resultTypes[0];

        // Warn if a large op is being serialized to a single workgroup.
        // This usually indicates a bug in SelectLoweringStrategy (op should
        // have received a lowering_config but didn't).
        int64_t numElements = 1;
        for (int64_t dim : resultType.getShape()) {
          if (!ShapedType::isDynamic(dim))
            numElements *= dim;
        }
        if (numElements > 1024) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[nova-tile-dispatch] WARNING: wrapping large op ("
                     << numElements << " elements) in single-block forall: "
                     << *op << "\n");
          // Only warn for ops that genuinely lack a config. Ops with configs
          // (all-zero workgroup tiles, reductions) are intentionally single-
          // block. Fill ops always fuse as DPS inits — single-block is expected.
          if (!getLoweringConfig(op) && !isa<linalg::FillOp>(op)) {
            op->emitWarning("large op (")
                << numElements
                << " elements) serialized to 1 workgroup — missing "
                   "lowering_config?";
          }
        }

        Location loc = op->getLoc();
        rewriter.setInsertionPoint(op);

        // Create one tensor.empty per result as shared_out for the forall.
        SmallVector<Value> emptyTensors;
        for (RankedTensorType rt : resultTypes) {
          SmallVector<OpFoldResult> emptySizes;
          for (int64_t dim = 0; dim < rt.getRank(); ++dim)
            emptySizes.push_back(rewriter.getIndexAttr(rt.getDimSize(dim)));
          emptyTensors.push_back(tensor::EmptyOp::create(
              rewriter, loc, emptySizes, rt.getElementType()));
        }

        SmallVector<OpFoldResult> lbs = {rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> ubs = {rewriter.getIndexAttr(1)};
        SmallVector<OpFoldResult> steps = {rewriter.getIndexAttr(1)};
        SmallVector<Attribute> blockMapping = {gpu::GPUBlockMappingAttr::get(
            &getContext(), gpu::MappingId::DimX)};

        auto forallOp = scf::ForallOp::create(
            rewriter, loc, lbs, ubs, steps, ValueRange{emptyTensors},
            ArrayAttr::get(&getContext(), blockMapping));

        Block *body = forallOp.getBody();
        op->moveBefore(body, body->without_terminator().end());

        // Create one parallel_insert_slice per result in the terminator.
        rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
        for (unsigned i = 0; i < resultTypes.size(); ++i) {
          RankedTensorType rt = resultTypes[i];
          int64_t rank = rt.getRank();
          SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
          SmallVector<OpFoldResult> sizes;
          for (int64_t dim = 0; dim < rank; ++dim)
            sizes.push_back(rewriter.getIndexAttr(rt.getDimSize(dim)));
          SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
          tensor::ParallelInsertSliceOp::create(
              rewriter, loc, op->getResult(i),
              forallOp.getRegionIterArgs()[i],
              offsets, sizes, strides);
        }

        // Replace uses of each result outside the forall.
        for (unsigned i = 0; i < resultTypes.size(); ++i) {
          Value resultVal = op->getResult(i);
          rewriter.replaceUsesWithIf(
              resultVal, forallOp.getResult(i), [&](OpOperand &use) {
                return !forallOp->isProperAncestor(use.getOwner());
              });
        }
      }
    }

    // --- Cleanup patterns after tiling ---
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
          context, [](tensor::ExtractSliceOp) { return false; });
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
      tensor::DimOp::getCanonicalizationPatterns(patterns, context);
      // NOTE: scf::ForallOp canonicalization is intentionally OMITTED.
      // The upstream pattern inlines single-trip foralls, which would pull
      // reduction ops out of their block-mapped forall.

      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitOpError("tiling cleanup failed");
        return signalPassFailure();
      }
    }

    // Final verification.
    if (!verifyComputeOpsAfterDistribution(funcOp)) {
      funcOp.emitOpError(
          "failed to distribute all compute ops to workgroups");
      signalPassFailure();
      return;
    }
  }

  StringRef getArgument() const override {
    return "nova-tile-and-distribute";
  }
  StringRef getDescription() const override {
    return "Tiles compute operations to workgroups using scf.forall";
  }
};

std::unique_ptr<Pass> createNovaTileAndDistributeToWorkgroupsPass() {
  return std::make_unique<NovaTileAndDistributePass>();
}

void registerNovaTileAndDistributePass() {
  PassRegistration<NovaTileAndDistributePass>();
}

} // namespace mlir::nova
