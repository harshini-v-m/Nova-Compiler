#include "Passes.h"
#include "NovaGPUTileAndFuseUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
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

using namespace mlir;

namespace mlir::nova {

// --- Utility Functions ---

/// Returns true if the operation is a compute operation (implements TilingInterface).
static bool isComputeOp(Operation *op) {
  return isa<TilingInterface>(op);
}

/// Returns true if the op is a contraction-like op (e.g. matmul).
static bool isContractionOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && linalg::isaContractionOpInterface(linalgOp);
}

/// Returns true if the op is already inside any scf.forall distribution loop.
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
  // Count non-zero tile dimensions.
  int numActiveDims = 0;
  for (auto tileSize : tileSizes) {
    std::optional<int64_t> cst = getConstantIntValue(tileSize);
    if (!cst || *cst != 0)
      numActiveDims++;
  }

  SmallVector<Attribute> mapping;

  if (numActiveDims > 3) {
    // >3 active dimensions: use linear block mapping (linearizes all dims).
    // IREE uses this approach for high-dimensional ops.
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
    // IREE reverses so innermost dim gets LinearDim0.
    return llvm::to_vector(llvm::reverse(mapping));
  }

  // ≤3 active dimensions: use 3D block mapping (x, y, z).
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

  // Try to read workgroup tile sizes from the lowering_config attribute
  // stamped by NovaGPUSelectLoweringStrategy. This avoids hardcoded tile sizes.
  bool usedConfig = false;
  if (auto config = getLoweringConfig(op)) {
    SmallVector<int64_t> wgTiles =
        getLoweringConfigTileSizes(config, kWorkgroupKey);
    if (wgTiles.size() == static_cast<size_t>(numLoops)) {
      for (int i = 0; i < numLoops; ++i)
        tileSizes[i] = rewriter.getIndexAttr(wgTiles[i]);
      usedConfig = true;
    }
  }

  if (!usedConfig) {
    // Heuristic fallback: tile parallel dims to 128, batch to 1, reduction to 0.
    int parallelDimCount = 0;
    for (int i = numLoops - 1; i >= 0; --i) {
      if (!linalg::isParallelIterator(linalgOp.getIteratorTypesArray()[i]))
        continue;
      if (parallelDimCount < 2)
        tileSizes[i] = rewriter.getIndexAttr(128);
      else
        tileSizes[i] = rewriter.getIndexAttr(1);   // Batch -> BlockZ
      ++parallelDimCount;
    }
  }

  // Zero out non-parallel (reduction) dims at workgroup level.
  // Workgroup distribution is only for parallel dimensions.
  for (int i = 0; i < numLoops; ++i) {
    if (!linalg::isParallelIterator(linalgOp.getIteratorTypesArray()[i]))
      tileSizes[i] = rewriter.getIndexAttr(0);
  }

  // Full-tile optimization: zero tile size when staticLoopSize == tileSize.
  // This prevents single-trip scf.forall loops, which can block cleanup patterns.
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

    // =========================================================================
    // Two-pass tiling strategy (mirrors IREE's dispatch formation):
    //
    // Pass 1: Tile contraction ops (matmuls) in FORWARD order. After tiling
    //         each matmul, fuse downstream elementwise consumers (bias, relu)
    //         as epilogues. This gives: matmul+bias+relu per kernel.
    //
    // Pass 2: Tile any remaining unfused compute ops (standalone elementwise
    //         ops that are not consumers of any matmul).
    //
    // Previously we iterated in REVERSE, which caused elementwise ops between
    // two matmuls (relu1 between matmul1 and matmul2) to be fused as
    // PRODUCERS of the later matmul — causing redundant recomputation.
    // =========================================================================

    llvm::SmallPtrSet<Operation *, 16> handledOps;

    // --- Helper lambda: tile a root op, fuse producers + consumers -----------
    // Returns failure() on error. Sets didTile=true if tiling actually happened,
    // false if the op was skipped (not a LinalgOp, etc.).
    auto tileRoot = [&](Operation *rootOp, bool &didTile) -> LogicalResult {
      didTile = false;
      if (handledOps.count(rootOp))
        return success();
      // 2. Info
      auto infoOr = getTiledAndDistributionInfo(rewriter, rootOp);
      if (failed(infoOr))
        return success(); // skip non-tilable ops gracefully
      TilingInfo info = *infoOr;
      auto tilingInterface = cast<TilingInterface>(rootOp);

      // 2b. Collect Fusion Cluster
      llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
      collectTiledAndFusedOps(rootOp, tiledAndFusedOps);

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

      // 3. Configure Options
      scf::SCFTilingOptions tilingOptions;
      tilingOptions.setTileSizes(info.tileSizes);
      auto mapping = getMapping(&getContext(), info.tileSizes);
      tilingOptions.setMapping(mapping);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // 4. Fusion Options
      scf::SCFTileAndFuseOptions tileAndFuseOptions;
      tileAndFuseOptions.setTilingOptions(tilingOptions);

      tileAndFuseOptions.setFusionControlFn(
          [&](tensor::ExtractSliceOp sliceOp, OpResult producer,
              bool isDest)
              -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
            Operation *producerOp = producer.getOwner();
            if (isa<tensor::PadOp>(producerOp))
              return std::nullopt;
            // Block contraction fusion — GEMMs stay as independent roots.
            if (isContractionOp(producerOp))
              return std::nullopt;
            bool yieldProducerReplacement =
                yieldReplacementsFor.contains(producerOp);
            return scf::SCFTileAndFuseOptions::ControlFnResult{
                yieldProducerReplacement};
          });

      // 5. Cleanup Patterns
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

      // 6. Execute Tile & Fuse (Producer Fusion)
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
        }
      } else {
        return failure();
      }

      // Replace results (with dominance check)
      for (auto [origValue, replacement] : result->replacements) {
        Value replacementCopy = replacement;
        rewriter.replaceUsesWithIf(
            origValue, replacement, [&](OpOperand &use) {
              Operation *user = use.getOwner();
              return !isa<tensor::DimOp>(user) &&
                     dominanceInfo.dominates(replacementCopy, user);
            });
      }

      // 7. Execute Consumer Fusion (epilogue: bias, relu, etc.)
      SmallVector<LoopLikeOpInterface> loops = result->loops;
      if (!result->tiledAndFusedOps.empty() && !loops.empty()) {
        FailureOr<std::queue<Operation *>> newFusionOpportunities =
            fuseConsumersIntoForall(
                rewriter, result->tiledAndFusedOps.getArrayRef(), loops,
                [&](Operation *op) {
                  // Only fuse non-contraction consumers. Contraction ops
                  // (matmuls) must remain independent roots — fusing them
                  // as consumers would collapse all layers into one forall.
                  if (isContractionOp(op))
                    return false;
                  // Do NOT fuse full-reduction consumers (all reduction
                  // iterators, no parallel dims). They need the FULL
                  // producer output, not one tile per block. Fusing them
                  // causes all blocks to race on the same scalar accumulator.
                  if (auto lg = dyn_cast<linalg::LinalgOp>(op)) {
                    auto iters = lg.getIteratorTypesArray();
                    if (!iters.empty() &&
                        !llvm::any_of(iters, linalg::isParallelIterator) &&
                        llvm::any_of(iters, linalg::isReductionIterator))
                      return false;
                  }
                  return tiledAndFusedOps.contains(op);
                });
        if (succeeded(newFusionOpportunities)) {
          fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                               tileAndFuseOptions, loops);
        }
      }
      return success();
    };

    // --- Pass 1: Tile contraction ops (matmuls) in forward order -------------
    // Consumer fusion will pull bias+relu into each matmul's forall.
    // Re-collect after each matmul tile because consumer fusion may delete
    // downstream elementwise ops that were in the snapshot list.
    // Safe pattern: collect fresh list at top of each while iteration, then
    // process exactly ONE untiled matmul per outer loop.
    {
      bool foundOne = true;
      while (foundOne) {
        foundOne = false;
        SmallVector<Operation *> computeOps = getComputeOps(funcOp);
        for (Operation *rootOp : computeOps) {
          if (!isContractionOp(rootOp))
            continue;
          if (isInsideWorkgroupForall(rootOp))
            continue;
          if (handledOps.count(rootOp))
            continue;
          bool didTile = false;
          if (failed(tileRoot(rootOp, didTile))) {
            signalPassFailure();
            return;
          }
          if (didTile) {
            foundOne = true; // Processed one; re-collect at top of while.
            break;
          }
          // Even if it didn't tile (skipped), mark it handled to avoid infinite loop
          handledOps.insert(rootOp);
        }
      }
    }

    // --- Pass 2: Tile remaining unfused elementwise ops (REVERSE order) ------
    // Mirrors IREE's tileConsumerAndFuseProducersUsingSCF contract:
    //   - Tile the LAST (sink) op in a chain as the "consumer/root".
    //   - Let producer fusion (inside tileConsumerAndFuseProducersUsingSCF)
    //     pull all chained producers up into the same scf.forall.
    //
    // This avoids the consumer fusion path entirely for elementwise chains,
    // eliminating the stale-tiledAndFusedOps iterator-invalidation bug.
    {
      SmallVector<Operation *> remainingOps = getComputeOps(funcOp);
      // Process in REVERSE so the last (sink) op is tiled first.
      for (Operation *rootOp : llvm::reverse(remainingOps)) {
        if (isInsideWorkgroupForall(rootOp))
          continue;
        // Only tile each "sink": skip if this op's result feeds another
        // untiled compute op (it will be fused as a producer of that op).
        bool hasTilableConsumer = llvm::any_of(rootOp->getUsers(), [](Operation *user) {
          return isa<TilingInterface>(user);
        });
        // If this op has a tilable consumer that is still outside a forall,
        // defer it — it will be fused as a producer when that consumer is tiled.
        if (hasTilableConsumer) {
          bool consumerPending = llvm::any_of(rootOp->getUsers(), [](Operation *user) {
            return isa<TilingInterface>(user) && !user->getParentOfType<scf::ForallOp>();
          });
          if (consumerPending)
            continue;
        }
        bool didTile = false;
        if (failed(tileRoot(rootOp, didTile))) {
          signalPassFailure();
          return;
        }
      }
    }
    
    // --- Pass 3: Wrap remaining un-distributed ops in single-block forall ---
    // Full reductions (all reduction iterators, no parallel dims) and their
    // consumer chains have all-zero workgroup tiles and are not tiled by
    // Pass 1 or Pass 2. We find the "sink" op (last in chain) and wrap the
    // entire producer/consumer chain in a 1-trip scf.forall with block mapping
    // so they become single-block GPU kernels.
    {
      llvm::SmallPtrSet<Operation *, 16> alreadyWrapped;

      // Find all un-distributed ops and wrap each connected chain.
      // Trace back from func.return operands AND from any
      // bufferization.materialize_in_destination sources. The latter handles
      // void-return functions (e.g. SCE) where the final computed tensor is
      // written into a memref argument rather than returned directly, so
      // returnOp.getOperands() would be empty and the chain would be missed.
      SmallVector<Value> rootValues;
      auto returnOp = cast<func::ReturnOp>(funcOp.getBody().back().getTerminator());
      for (Value retVal : returnOp.getOperands())
        rootValues.push_back(retVal);
      funcOp.walk([&](bufferization::MaterializeInDestinationOp matOp) {
        rootValues.push_back(matOp.getSource());
      });

      for (Value retVal : rootValues) {
        // Trace back through the chain to find all un-distributed ops.
        SmallVector<Operation *> opsToMove;
        SmallVector<Operation *> worklist;
        llvm::SmallPtrSet<Operation *, 16> visited;

        if (auto defOp = retVal.getDefiningOp()) {
          if (!isInsideWorkgroupForall(defOp) &&
              defOp->getParentOp() == funcOp.getOperation())
            worklist.push_back(defOp);
        }

        while (!worklist.empty()) {
          Operation *curr = worklist.pop_back_val();
          if (!visited.insert(curr).second)
            continue;
          if (isInsideWorkgroupForall(curr))
            continue;
          if (alreadyWrapped.count(curr))
            continue;
          // Skip function arguments (no defining op).
          if (curr->getParentOp() != funcOp.getOperation())
            continue;
          // Skip scf.forall ops that already have GPU block mapping —
          // they are already distributed kernels and must not be moved
          // into the single-block wrapper.
          if (auto forallOp = dyn_cast<scf::ForallOp>(curr)) {
            auto mapping = forallOp.getMappingAttr();
            if (mapping && llvm::any_of(mapping.getValue(), [](Attribute attr) {
                  return isa<gpu::GPUBlockMappingAttr>(attr);
                }))
              continue;
          }

          opsToMove.push_back(curr);
          for (Value operand : curr->getOperands()) {
            if (auto defOp = operand.getDefiningOp()) {
              if (defOp->getParentOp() == funcOp.getOperation() &&
                  !isInsideWorkgroupForall(defOp))
                worklist.push_back(defOp);
            }
          }
        }

        if (opsToMove.empty())
          continue;

        // Filter out ops whose results have uses outside the chain.
        // These must stay at function level; the wrapper forall will
        // implicitly capture their values. Without this filter, moving
        // e.g. bufferization.to_tensor into the wrapper would hide it
        // from other kernels that also need the same input tensor.
        //
        // Exempt: retVal's defining op — its external use will be
        // replaced by the forall result via replaceUsesWithIf.
        // Iterative: removing one op may expose another's user as
        // external, so repeat until stable.
        {
          Operation *retDefOp = retVal.getDefiningOp();
          llvm::SmallPtrSet<Operation *, 16> moveSet(opsToMove.begin(),
                                                      opsToMove.end());
          bool changed = true;
          while (changed) {
            changed = false;
            SmallVector<Operation *> filtered;
            for (Operation *op : opsToMove) {
              // Never filter the op that defines retVal — its external
              // use is redirected to the forall result.
              if (op == retDefOp) {
                filtered.push_back(op);
                continue;
              }
              bool hasExternalUser = false;
              for (Value res : op->getResults()) {
                for (Operation *user : res.getUsers()) {
                  if (!moveSet.contains(user)) {
                    hasExternalUser = true;
                    break;
                  }
                }
                if (hasExternalUser) break;
              }
              if (hasExternalUser) {
                moveSet.erase(op);
                changed = true;
              } else {
                filtered.push_back(op);
              }
            }
            opsToMove = std::move(filtered);
          }
        }

        if (opsToMove.empty())
          continue;

        // Check if this chain contains a full-reduction op (all reduction, no
        // parallel dims). Only wrap chains that actually need GPU distribution.
        bool hasFullReduction = false;
        for (Operation *moveOp : opsToMove) {
          auto lg = dyn_cast<linalg::LinalgOp>(moveOp);
          if (!lg) continue;
          auto iters = lg.getIteratorTypesArray();
          if (!iters.empty() &&
              !llvm::any_of(iters, linalg::isParallelIterator) &&
              llvm::any_of(iters, linalg::isReductionIterator)) {
            hasFullReduction = true;
            break;
          }
        }
        if (!hasFullReduction)
          continue;

        // Sort ops in topological order.
        llvm::stable_sort(opsToMove, [&](Operation *a, Operation *b) {
          return a->isBeforeInBlock(b);
        });

        // Find the last DPS op in the chain to use its result as the
        // forall's output. If the return value comes from a non-DPS op
        // (like tensor.expand_shape), use the output type directly.
        auto retType = dyn_cast<RankedTensorType>(retVal.getType());
        if (!retType)
          continue;

        // Use the return value's producer's output as the shared_out.
        // Create a tensor.empty as the shared_out for the forall.
        // Insert the wrapper AFTER all external operands are defined.
        // This handles the case where the reduction consumes the result of
        // an already-distributed forall (e.g. matmul) that wasn't moved.
        Operation *insertAfter = nullptr;
        llvm::SmallPtrSet<Operation *, 16> opsToMoveSet(opsToMove.begin(),
                                                         opsToMove.end());
        for (Operation *moveOp : opsToMove) {
          for (Value operand : moveOp->getOperands()) {
            if (auto defOp = operand.getDefiningOp()) {
              if (!opsToMoveSet.contains(defOp) &&
                  defOp->getParentOp() == funcOp.getOperation()) {
                if (!insertAfter || defOp->isBeforeInBlock(insertAfter) == false)
                  insertAfter = defOp;
              }
            }
          }
        }
        if (insertAfter)
          rewriter.setInsertionPointAfter(insertAfter);
        else
          rewriter.setInsertionPoint(opsToMove.front());
        Location loc = opsToMove.front()->getLoc();

        SmallVector<OpFoldResult> emptySizes;
        for (int64_t dim = 0; dim < retType.getRank(); ++dim)
          emptySizes.push_back(rewriter.getIndexAttr(retType.getDimSize(dim)));
        Value emptyTensor = tensor::EmptyOp::create(
            rewriter, loc, emptySizes, retType.getElementType());

        SmallVector<OpFoldResult> lbs = {rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> ubs = {rewriter.getIndexAttr(1)};
        SmallVector<OpFoldResult> steps = {rewriter.getIndexAttr(1)};
        SmallVector<Attribute> blockMapping = {gpu::GPUBlockMappingAttr::get(
            &getContext(), gpu::MappingId::DimX)};

        auto forallOp = scf::ForallOp::create(
            rewriter, loc, lbs, ubs, steps, ValueRange{emptyTensor},
            ArrayAttr::get(&getContext(), blockMapping));

        Block *body = forallOp.getBody();

        // Move all ops into the forall body (in topological order).
        for (Operation *moveOp : opsToMove)
          moveOp->moveBefore(body, body->without_terminator().end());

        // Create parallel_insert_slice in the terminator.
        rewriter.setInsertionPointToEnd(
            forallOp.getTerminator().getBody());
        int64_t rank = retType.getRank();
        SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim = 0; dim < rank; ++dim)
          sizes.push_back(rewriter.getIndexAttr(retType.getDimSize(dim)));
        SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

        tensor::ParallelInsertSliceOp::create(
            rewriter, loc, retVal,
            forallOp.getRegionIterArgs()[0],
            offsets, sizes, strides);

        // Replace the return value with the forall result, but only for
        // uses OUTSIDE the forall (the parallel_insert_slice inside must
        // keep referencing the original value).
        rewriter.replaceUsesWithIf(
            retVal, forallOp.getResult(0), [&](OpOperand &use) {
              return !forallOp->isProperAncestor(use.getOwner());
            });

        for (Operation *moveOp : opsToMove)
          alreadyWrapped.insert(moveOp);
      }
    }

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

    // Final verification: Ensure all compute ops are now inside workgroup loops.
    if (!verifyComputeOpsAfterDistribution(funcOp)) {
        funcOp.emitOpError("failed to distribute all compute ops to workgroups");
        signalPassFailure();
        return;
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