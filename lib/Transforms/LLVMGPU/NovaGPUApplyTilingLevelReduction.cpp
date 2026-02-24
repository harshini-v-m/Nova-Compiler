// Nova GPU Apply Tiling Level Pass
//
// A unified pass that tiles a specific level (Reduction, Thread, or Subgroup)
// for contraction ops (linalg.matmul / linalg.batch_matmul) found inside
// workgroup scf.forall loops.
//
// Architecture mirrors IREE's GPUApplyTilingLevel.cpp exactly:
//   - An enum `NovaTilingLevel` selects which loop level to tile.
//   - `getTiledOps()` walks the function and collects all TilingInterface ops
//     that should be tiled at the requested level (IREE reads from
//     LoweringConfigAttr; Nova uses heuristic sizes until config is added).
//   - `applyTileAndFuseToEachRoot()` tiles + fuses producers for each root.
//   - Cleanup patterns mirror IREE's post-tiling cleanup block.
//
// Tiling levels:
//   Reduction  — K-dimension → scf.for (sequential)
//                Tile size  = kReductionTile (32)
//   Thread     — M/N-dimensions into threads → scf.forall w/ thread mapping
//                Tile size  = kThreadTile (4 per dim)
//   Subgroup   — M/N-dimensions into warp subgroups → scf.forall w/ warp mapping
//                Tile size  = kSubgroupTile (16 per dim)
//
// TODO: Replace all heuristic tile sizes with reads from NovaLoweringConfigAttr
//       once that attribute is implemented and stamped on ops by the config
//       selection pass.
//
// After Reduction-level tiling (replaces old NovaGPUApplyTilingLevelReduction):
//   scf.for %k = 0 to K step 32 {
//     A_k = extract_slice %promoted_A [0, %k][M, 32]
//     B_k = extract_slice %promoted_B [%k, 0][32, N]
//     C   = linalg.matmul ins(A_k, B_k) outs(C_acc)
//   }

#include "Passes.h"
#include "NovaGPUTileAndFuseUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Tiling Level Enum
// Mirrors IREE::GPU::TilingLevel.
//===----------------------------------------------------------------------===//

enum class NovaTilingLevel {
  Reduction, // K dimension of matmul → scf.for (sequential)
  Thread,    // M/N per-thread tile  → scf.forall + thread mapping
  Subgroup,  // M/N per-warp tile    → scf.forall + warp mapping
};

//===----------------------------------------------------------------------===//
// Heuristic tile sizes (replace with LoweringConfigAttr reads in the future)
//===----------------------------------------------------------------------===//

// K-tile: chosen so A-tile (M×K f32) and B-tile (K×N f32) fit in shared mem.
static constexpr int64_t kReductionTile = 32;

// Thread-level tile per dimension (register tile per thread).
static constexpr int64_t kThreadTile = 4;

// Subgroup-level tile per dimension (warp tile).
static constexpr int64_t kSubgroupTile = 16;

//===----------------------------------------------------------------------===//
// Helper: is this a workgroup scf.forall?
//===----------------------------------------------------------------------===//

static bool isWorkgroupForall(scf::ForallOp forallOp) {
  auto mappingAttr = forallOp.getMappingAttr();
  if (!mappingAttr)
    return false;
  return llvm::any_of(mappingAttr.getValue(), [](Attribute attr) {
    return isa<gpu::GPUBlockMappingAttr>(attr);
  });
}

//===----------------------------------------------------------------------===//
// getTiledOps — mirrors IREE's getTiledOps(funcOp, tilingLevel)
//
// IREE: reads LoweringConfigAttr on op → checks hasTilingLevel(opaqueLevel)
// Nova: walks for LinalgOps that are contractions (all levels share same ops).
//       Thread/Subgroup also tile any linalg.copy produced by shared-memory
//       promotion that the K-loop left visible.
//===----------------------------------------------------------------------===//

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(func::FuncOp funcOp, NovaTilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;

  funcOp.walk([&](Operation *op) {
    // Only ops implementing TilingInterface matter.
    auto tilingOp = dyn_cast<TilingInterface>(op);
    if (!tilingOp)
      return WalkResult::advance();

    // For Reduction: collect contraction ops inside workgroup foralls.
    if (tilingLevel == NovaTilingLevel::Reduction) {
      if (!isa<linalg::LinalgOp>(op))
        return WalkResult::advance();
      auto linalgOp = cast<linalg::LinalgOp>(op);
      if (!linalg::isaContractionOpInterface(linalgOp))
        return WalkResult::advance();
      // Only inside a workgroup forall.
      auto parentForall = op->getParentOfType<scf::ForallOp>();
      if (!parentForall || !isWorkgroupForall(parentForall))
        return WalkResult::advance();
      targets.insert(tilingOp);
      return WalkResult::advance();
    }

    // For Thread / Subgroup: collect contraction ops that are now inside the
    // K-loop (scf.for) which itself is inside the workgroup forall.
    // We tile whatever contraction ops remain after Reduction tiling.
    if (tilingLevel == NovaTilingLevel::Thread ||
        tilingLevel == NovaTilingLevel::Subgroup) {
      if (!isa<linalg::LinalgOp>(op))
        return WalkResult::advance();
      auto linalgOp = cast<linalg::LinalgOp>(op);
      if (!linalg::isaContractionOpInterface(linalgOp))
        return WalkResult::advance();
      // Must be inside a workgroup forall (possibly nested inside scf.for).
      auto parentForall = op->getParentOfType<scf::ForallOp>();
      if (!parentForall || !isWorkgroupForall(parentForall))
        return WalkResult::advance();
      targets.insert(tilingOp);
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  return targets;
}

//===----------------------------------------------------------------------===//
// getTileSizes — build tile size vector for a given linalg op + level
//===----------------------------------------------------------------------===//

static SmallVector<OpFoldResult>
getTileSizes(RewriterBase &rewriter, linalg::LinalgOp op,
             NovaTilingLevel tilingLevel) {
  int numLoops = op.getNumLoops();
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));

  if (tilingLevel == NovaTilingLevel::Reduction) {
    // Tile only reduction (K) dims.
    // Prefer reading the reduction tile size from the op's lowering_config
    // attribute (set by NovaGPUSelectLoweringStrategy).  Fall back to the
    // static kReductionTile constant when no config is present.
    int64_t kStep = kReductionTile; // default fallback
    if (auto config = getLoweringConfig(op.getOperation())) {
      SmallVector<int64_t> redTiles =
          getLoweringConfigTileSizes(config, kReductionKey);
      // Find the last non-zero entry in the reduction tile array (K dim).
      for (int64_t t : llvm::reverse(redTiles)) {
        if (t > 0) { kStep = t; break; }
      }
    }
    for (int i = 0; i < numLoops; ++i) {
      if (linalg::isReductionIterator(op.getIteratorTypesArray()[i]))
        tileSizes[i] = rewriter.getIndexAttr(kStep);
    }
    return tileSizes;
  }

  if (tilingLevel == NovaTilingLevel::Thread) {
    // Tile only parallel (M/N) dims.
    // Use lowering_config `thread` tile size if available, otherwise fallback.
    if (auto config = getLoweringConfig(op.getOperation())) {
      SmallVector<int64_t> threadTiles =
          getLoweringConfigTileSizes(config, kThreadKey);
      if (threadTiles.size() == static_cast<size_t>(numLoops)) {
        for (int i = 0; i < numLoops; ++i) {
          tileSizes[i] = rewriter.getIndexAttr(threadTiles[i]);
        }
        return tileSizes;
      }
    }
    for (int i = 0; i < numLoops; ++i) {
      if (linalg::isParallelIterator(op.getIteratorTypesArray()[i]))
        tileSizes[i] = rewriter.getIndexAttr(kThreadTile);
    }
    return tileSizes;
  }

  if (tilingLevel == NovaTilingLevel::Subgroup) {
    // Tile only parallel (M/N) dims.
    // Use lowering_config `subgroup` tile size if available, otherwise fallback.
    if (auto config = getLoweringConfig(op.getOperation())) {
      SmallVector<int64_t> subgroupTiles =
          getLoweringConfigTileSizes(config, kSubgroupKey);
      if (subgroupTiles.size() == static_cast<size_t>(numLoops)) {
        for (int i = 0; i < numLoops; ++i) {
          tileSizes[i] = rewriter.getIndexAttr(subgroupTiles[i]);
        }
        return tileSizes;
      }
    }
    for (int i = 0; i < numLoops; ++i) {
      if (linalg::isParallelIterator(op.getIteratorTypesArray()[i]))
        tileSizes[i] = rewriter.getIndexAttr(kSubgroupTile);
    }
    return tileSizes;
  }

  return tileSizes; // all zeros — should not reach here
}

//===----------------------------------------------------------------------===//
// buildFusionControlFn — which producers to fuse into the tiled loop
//
// Mirrors IREE's SCFTileAndFuseOptions fusion control in TileAndFuseUtils.cpp.
//===----------------------------------------------------------------------===//

static scf::SCFTileAndFuseOptions::ControlFnTy
buildFusionControlFn(NovaTilingLevel tilingLevel) {
  return [tilingLevel](tensor::ExtractSliceOp candidateSliceOp,
                       OpResult originalProducer,
                       bool isDestinationOperand)
             -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *owner = originalProducer.getOwner();

    // Never fuse accumulators (destination operands).
    if (isDestinationOperand)
      return std::nullopt;

    if (tilingLevel == NovaTilingLevel::Reduction) {
      // Reduction loop: fuse linalg.copy (shared-memory loads) but NOT pad ops.
      // Pad ops with boundary guards must stay outside the K-loop to avoid
      // generating incorrect zero-slice guards inside each K-iteration.
      if (isa<tensor::PadOp>(owner))
        return std::nullopt;
      // Fuse everything else (linalg.copy, alloc_tensor, extract_slice, etc.)
      return scf::SCFTileAndFuseOptions::ControlFnResult{
          /*yieldProducerReplacement=*/false};
    }

    // Thread / Subgroup: fuse all non-destination producers.
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        /*yieldProducerReplacement=*/false};
  };
}

//===----------------------------------------------------------------------===//
// applyTilingLevelToOps — the main tiling loop
// Mirrors IREE's applyTileAndFuseToEachRoot() in TileAndFuseUtils.cpp.
//===----------------------------------------------------------------------===//

static LogicalResult
applyTilingLevelToOps(func::FuncOp funcOp, IRRewriter &rewriter,
                      llvm::SmallDenseSet<TilingInterface> &targetOps,
                      NovaTilingLevel tilingLevel) {
  MLIRContext *ctx = funcOp.getContext();

  for (TilingInterface tilingOp : targetOps) {
    // Skip if it was already erased by a previous iteration.
    if (!tilingOp || tilingOp->getParentOp() == nullptr)
      continue;

    auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation());
    if (!linalgOp)
      continue;

    SmallVector<OpFoldResult> tileSizes =
        getTileSizes(rewriter, linalgOp, tilingLevel);

    // If all tile sizes are 0, skip this op.
    bool anyNonZero = llvm::any_of(tileSizes, [](OpFoldResult ofr) {
      auto cst = getConstantIntValue(ofr);
      return cst && *cst != 0;
    });
    if (!anyNonZero)
      continue;

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    if (tilingLevel == NovaTilingLevel::Reduction) {
      // K-dimension: sequential scf.for (mirrors IREE Reduction level).
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    } else {
      // Thread / Subgroup: parallel scf.forall with GPU mapping attributes.
      // Mirrors IREE TileAndFuseUtils.cpp lines 321-343:
      //   Thread  → GPUThreadMappingAttr (LinearDim0, LinearDim1, ...)
      //   Subgroup → GPUWarpMappingAttr  (LinearDim0, LinearDim1, ...)
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // Build one mapping attribute per non-zero tile dimension.
      SmallVector<Attribute> mapping;
      unsigned idx = 0;
      for (auto size : tileSizes) {
        auto cst = getConstantIntValue(size);
        if (!cst || *cst == 0)
          continue;
        unsigned mappingId =
            static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
        if (tilingLevel == NovaTilingLevel::Thread) {
          mapping.push_back(gpu::GPUThreadMappingAttr::get(
              ctx, static_cast<gpu::MappingId>(mappingId)));
        } else {
          // Subgroup → warp-level mapping.
          mapping.push_back(gpu::GPUWarpMappingAttr::get(
              ctx, static_cast<gpu::MappingId>(mappingId)));
        }
      }
      // IREE reverses the mapping so that the innermost dimension gets
      // LinearDim0 (the fastest-moving warp/thread dimension).
      tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);
    tileAndFuseOptions.setFusionControlFn(buildFusionControlFn(tilingLevel));

    rewriter.setInsertionPoint(tilingOp);
    FailureOr<scf::SCFTileAndFuseResult> result =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, tilingOp, tileAndFuseOptions);

    if (failed(result)) {
      tilingOp.emitWarning()
          << "nova-gpu-apply-tiling-level: tiling failed at level "
          << static_cast<int>(tilingLevel) << ", skipping\n";
      continue;
    }

    // Replace original uses with tiled results.
    for (auto [origValue, replacement] : result->replacements)
      rewriter.replaceAllUsesWith(origValue, replacement);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// applyCleanupPatterns — post-tiling cleanup
// Mirrors IREE GPUApplyTilingLevel.cpp lines 107–122.
//===----------------------------------------------------------------------===//

static LogicalResult applyCleanupPatterns(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  // Fold empty tensors and merge consecutive insert/extract slices.
  // These both simplify the loop body and enable later hoisting.
  tensor::populateFoldTensorEmptyPatterns(patterns);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

  // Swap extract_slice(pad(x)) → pad(extract_slice(x)).
  // Essential when pad ops are kept outside the K-loop: without this the loop
  // would extract a slice from a large padded tensor on every iteration;
  // with this it pads only the small slice needed for the current iteration.
  patterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
      ctx,
      [](tensor::ExtractSliceOp) -> bool {
        return false; // no zero-slice guard needed
      });

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
// Pass implementations — one per level (mirrors IREE's single pass w/ option)
//===----------------------------------------------------------------------===//

// --- Helper base (shared runOnOperation logic) ---
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

    if (failed(applyTilingLevelToOps(funcOp, rewriter, targetOps,
                                     tilingLevel))) {
      this->signalPassFailure();
      return;
    }

    if (failed(applyCleanupPatterns(funcOp)))
      this->signalPassFailure();
  }
};

//===--- Reduction pass ---=================================================//

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
    return "Tiles reduction (K) dimension of matmul into inner scf.for loop. "
           "Fuses A/B producer copies. Mirrors IREE GPUApplyTilingLevel(Reduction).";
  }
};

//===--- Thread pass ---====================================================//

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
           "Mirrors IREE GPUApplyTilingLevel(Thread).";
  }
};

//===--- Subgroup pass ---==================================================//

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
           "Mirrors IREE GPUApplyTilingLevel(Subgroup).";
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
