#include "Passes.h"
#include "NovaGPUTileAndFuseUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

using namespace mlir;

namespace mlir::nova {

// --- Utility Functions ---

/// Returns true if the operation is a compute operation (implements TilingInterface).
static bool isComputeOp(Operation *op) {
  return isa<TilingInterface>(op);
}

/// Collects all compute operations in the function.
static SmallVector<Operation *> getComputeOps(func::FuncOp funcOp) {
  SmallVector<Operation *> computeOps;
  funcOp.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      computeOps.push_back(op);
    }
  });
  return computeOps;
}

/// Returns true if it is allowed to leave the `op` outside distribution loops.
/// E.g., linalg.pack op can only be fused as a consumer in perfect tiling scenario.
static bool isAllowedToFailOnConsumerFusion(Operation *op) {
  return isa<linalg::PackOp>(op);
}

/// Returns true if all the compute ops are within scf.forall distribution
/// loops, except the ops that are allowed to stay outside.
static bool verifyComputeOpsAfterDistribution(func::FuncOp funcOp) {
  WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::ForallOp>(op) || !isComputeOp(op)) {
      return WalkResult::skip();
    }
    if (!isAllowedToFailOnConsumerFusion(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !res.wasInterrupted();
}

/// Returns true if any value produced by `producer` is used as an init value
/// for the DPS `user`. Returns false if the user is not in DPS.
static bool isUsedAsInit(Operation *producer, Operation *user) {
  auto dpsIface = dyn_cast<DestinationStyleOpInterface>(user);
  if (!dpsIface) {
    return false;
  }
  ValueRange results = producer->getResults();
  return llvm::any_of(dpsIface.getDpsInits(), [&](Value operand) {
    return llvm::is_contained(results, operand);
  });
}

static SmallVector<Attribute> getMapping(MLIRContext *context, ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<Attribute> mapping;
  // Iterate in reverse: Inner loop -> Block X, Next -> Block Y, Next -> Block Z
  int dim = 0;
  for (auto tileSize : llvm::reverse(tileSizes)) {
      std::optional<int64_t> cst = getConstantIntValue(tileSize);
      if (cst && *cst == 0) continue; // Skip size 0 tiles
      
      switch (dim) {
          case 0: mapping.push_back(gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimX)); break;
          case 1: mapping.push_back(gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimY)); break;
          case 2: mapping.push_back(gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimZ)); break;
          default: break; 
      }
      dim++;
  }
  // Reverse back to match loop order
  return llvm::to_vector(llvm::reverse(mapping));
}

/// Checks whether we have static dimension for all the loop bounds and steps.
static bool areAllStaticLoopBounds(scf::ForallOp forallOp) {
  for (auto [lb, ub, step] : llvm::zip_equal(forallOp.getMixedLowerBound(),
                                             forallOp.getMixedUpperBound(),
                                             forallOp.getMixedStep())) {
    if (!getConstantIntValue(lb) || !getConstantIntValue(ub) || !getConstantIntValue(step)) {
      return false;
    }
  }
  return true;
}

struct TilingInfo {
  Operation *tilableOp;
  SmallVector<OpFoldResult> tileSizes;
  SmallVector<int64_t> interchange;  // TODO: Implement loop interchange support (see missing_functionality_analysis.md #7)
};

static FailureOr<TilingInfo> getTiledAndDistributionInfo(RewriterBase &rewriter,
                                                          Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return failure();

  int numLoops = linalgOp.getNumLoops();
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));

  // Heuristic tile sizes (TODO: replace with LoweringConfigAttr when available).
  // Tile the two innermost parallel dims to 128 (M->Y, N->X), batch dims to 1.
  // Reduction dims (K) stay at 0 — not tiled at workgroup level.
  if (numLoops >= 2) {
    int parallelDimCount = 0;
    for (int i = numLoops - 1; i >= 0; --i) {
      if (!linalg::isParallelIterator(linalgOp.getIteratorTypesArray()[i]))
        continue;
      if (parallelDimCount == 0)
        tileSizes[i] = rewriter.getIndexAttr(128); // N -> BlockX
      else if (parallelDimCount == 1)
        tileSizes[i] = rewriter.getIndexAttr(128); // M -> BlockY
      else
        tileSizes[i] = rewriter.getIndexAttr(1);   // Batch -> BlockZ
      ++parallelDimCount;
    }
  }

  // PartitionableLoops equivalent: zero out non-parallel (reduction) dims.
  // IREE uses PartitionableLoopsInterface for this, which is IREE-specific.
  // We achieve the same by inspecting iterator types directly.
  for (int i = 0; i < numLoops; ++i) {
    if (!linalg::isParallelIterator(linalgOp.getIteratorTypesArray()[i]))
      tileSizes[i] = rewriter.getIndexAttr(0);
  }

  // Full-tile optimization: zero tile size when staticLoopSize == tileSize.
  // This prevents single-trip scf.forall loops, which can block cleanup patterns.
  // Mirrors IREE's getTiledAndDistributionInfo (TileDispatchUsingForall.cpp:111-138).
  // Keep at least one non-zero tile size so the forall loop is still created.
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    auto tilingIface = cast<TilingInterface>(op);
    SmallVector<Range> bounds = tilingIface.getIterationDomain(rewriter);

    // Count current non-zero tile sizes.
    int numNonZero = 0;
    for (auto &ts : tileSizes) {
      if (auto cst = getConstantIntValue(ts))
        if (*cst != 0) ++numNonZero;
    }

    // Zero out full-tile dims from innermost outward, keeping at least 1.
    for (int i = (int)tileSizes.size() - 1; i >= 0; --i) {
      if (numNonZero <= 1) break;
      auto tsCst = getConstantIntValue(tileSizes[i]);
      if (!tsCst || *tsCst == 0) continue;
      if (i >= (int)bounds.size()) continue;
      auto boundCst = getConstantIntValue(bounds[i].size);
      if (boundCst && *boundCst == *tsCst) {
        tileSizes[i] = rewriter.getIndexAttr(0);
        --numNonZero;
      }
    }
  }

  return TilingInfo{op, tileSizes, {}};
}

// --- Main Pass ---

struct NovaTileAndDistributePass : public PassWrapper<NovaTileAndDistributePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaTileAndDistributePass)

  NovaTileAndDistributePass() = default;
  NovaTileAndDistributePass(const NovaTileAndDistributePass &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, linalg::LinalgDialect, 
                    tensor::TensorDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());

    // 1. Find Root Op.
    // Matches IREE's heuristic: select the last compute op that has a workgroup
    // tiling level. Since Nova lacks LoweringConfigAttr, we select the last
    // compute op (IREE selects last op with a lowering config).
    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    Operation *rootOp = computeOps.empty() ? nullptr : computeOps.back();
    if (!rootOp) return;

    // 2. Info
    auto infoOr = getTiledAndDistributionInfo(rewriter, rootOp);
    if (failed(infoOr)) return;
    TilingInfo info = *infoOr;
    auto tilingInterface = cast<TilingInterface>(rootOp);

    // 2b. Collect Fusion Cluster
    llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
    collectTiledAndFusedOps(rootOp, tiledAndFusedOps);
    
    DominanceInfo dominanceInfo(rootOp);
    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
        // Require replacement for values that are used after the main tilable op or
        // by ops that will definitely not be fused. Note that if a value is used as
        // an init of a DPS op, the user currently cannot be fused. Having a
        // replacement for it would attempt fusion and fail, so avoid such cases.
        if (llvm::any_of(op->getUsers(), [&](Operation *user) {
              if (isUsedAsInit(op, user)) {
                return false;
              }
              return dominanceInfo.properlyDominates(rootOp, user) ||
                     !tiledAndFusedOps.contains(user);
            })) {
          yieldReplacementsFor.insert(op);
        }
    }

    // 3. Configure Options
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(info.tileSizes);
    
    auto mapping = getMapping(&getContext(), info.tileSizes);
    tilingOptions.setMapping(mapping);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    
    // TODO: Implement WorkgroupReorderingStrategy (see missing_functionality_analysis.md #8)
    // Support custom loop generation for advanced workgroup iteration orders (e.g., Z-order curves).
    // if (workgroupReorderingStrategy) {
    //   tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::CustomOp);
    //   tilingOptions.setCustomLoopGenerationFns(loopHeaderFn, terminatorFn);
    // }

    // 4. Fusion Options
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);
    
    // Control Fn - Skip Pad fusion, use yieldReplacementsFor
    tileAndFuseOptions.setFusionControlFn([&](tensor::ExtractSliceOp sliceOp, OpResult producer, bool isDest) 
                                           -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        Operation* producerOp = producer.getOwner();
        if (isa<tensor::PadOp>(producerOp)) return std::nullopt; 
        bool yieldProducerReplacement = yieldReplacementsFor.contains(producerOp);
        return scf::SCFTileAndFuseOptions::ControlFnResult{yieldProducerReplacement}; 
    });

    // 5. Cleanup Patterns
    RewritePatternSet cleanupPatterns(&getContext());
    tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, &getContext());
    tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns, &getContext());
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(cleanupPatterns);
    
    // Add ExtractSliceOfPadTensorSwapPattern without zero guard
    cleanupPatterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
          &getContext(), [](tensor::ExtractSliceOp) { return false; });
    
    // TODO: Add additional cleanup patterns (see missing_functionality_analysis.md #9)
    // populateSwapExtractWithExpandPattern(cleanupPatterns);
    // populateFoldExtractSliceOfBroadcastPattern(cleanupPatterns);

    tileAndFuseOptions.cleanupPatterns = FrozenRewritePatternSet(std::move(cleanupPatterns));

    // 6. Execute Tile & Fuse (Producer Fusion)
    FailureOr<scf::SCFTileAndFuseResult> result;
    if (rootOp->getNumResults() > 0) {
        result = scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterface, tileAndFuseOptions);
    } else {
        auto tileResult = scf::tileUsingSCF(rewriter, tilingInterface, tilingOptions);
        if (succeeded(tileResult)) {
            rewriter.eraseOp(rootOp);
        }
        return; 
    }

    if (failed(result)) {
        signalPassFailure();
        return;
    }

    // Replace results (with dominance check)
    for (auto [origValue, replacement] : result->replacements) {
        Value replacementCopy = replacement;
        rewriter.replaceUsesWithIf(origValue, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return !isa<tensor::DimOp>(user) &&
                   dominanceInfo.dominates(replacementCopy, user);
        });
    }

    // 7. Execute Consumer Fusion
    SmallVector<LoopLikeOpInterface> loops = result->loops;
    if (!result->tiledAndFusedOps.empty() && !loops.empty()) {
         FailureOr<std::queue<Operation *>> newFusionOpportunities =
            fuseConsumersIntoForall(
                rewriter, result->tiledAndFusedOps.getArrayRef(),
                loops, [&](Operation *op) {
                  return tiledAndFusedOps.contains(op);
                });

         if (succeeded(newFusionOpportunities)) {
            fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                                tileAndFuseOptions, loops);
        } else {
            // Verify that consumer fusion didn't leave compute ops outside
            if (!verifyComputeOpsAfterDistribution(funcOp)) {
                funcOp.emitOpError("failed to fuse all consumers into scf.forall");
                signalPassFailure();
                return;
            }
        }
    }
    
    // TODO: Implement transpose workgroup support (see missing_functionality_analysis.md #10)
    // Swap X and Y mapping attributes when transposeWorkgroup option is enabled.
    // if (transposeWorkgroup && areAllStaticLoopBounds(forallOp) && mappingSize >= 2) {
    //   std::swap(mappingAttrs[mappingSize - 1], mappingAttrs[mappingSize - 2]);
    // }

    // Cleanup after tiling and consumer fusion.
    // Mirrors IREE's TileDispatchUsingForall cleanup (TileDispatchUsingForall.cpp:306-431).
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      // Swap extract_slice(pad(x)) -> pad(extract_slice(x)).
      // Pushes pads inward so FoldFillIntoPad can eliminate zero-fill pads.
      // No zero-slice guard: scf.forall loop bounds already prevent empty tiles.
      patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
          context, [](tensor::ExtractSliceOp) { return false; });

      // Standard tiling canonicalization: fold affine.min/max, remove unit loops.
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);

      // Fold tensor.empty ops that are no longer needed after padding.
      tensor::populateFoldTensorEmptyPatterns(patterns);

      // Merge consecutive extract/insert slice ops to simplify later patterns.
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

      // Canonicalize extract_slice and forall ops.
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
      tensor::DimOp::getCanonicalizationPatterns(patterns, context);
      scf::ForallOp::getCanonicalizationPatterns(patterns, context);

      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitOpError("tiling cleanup failed");
        return signalPassFailure();
      }
    }

  }

  StringRef getArgument() const override { return "nova-tile-and-distribute"; }
  StringRef getDescription() const override { return "Tiles compute operations to workgroups using scf.forall"; }
};

std::unique_ptr<Pass> createNovaTileAndDistributeToWorkgroupsPass() {
  return std::make_unique<NovaTileAndDistributePass>();
}

void registerNovaTileAndDistributePass() {
    PassRegistration<NovaTileAndDistributePass>();
}

} // namespace mlir::nova
