// NovaGPUGreedilyDistributeToThreadsPass
//
// Fallback pass that distributes orphaned parallel ops to threads.
// Runs AFTER FuseAndHoist — any TilingInterface op that is NOT inside a
// thread-mapped scf.forall gets tiled into one using derived thread tiles.
//
// This handles ops that:
//  - Were correctly classified as prologues/epilogues (no lowering_config)
//  - Could not fuse into any root's thread forall (tensor.reshape barriers,
//    reduction producer blocking, no root consumer)
//  - Got workgroup-only distribution from Phase 2 of TileAndDistribute
//
// Without this pass, these ops launch kernels with threads(1,1,1).
//
// Mirrors IREE's GPUGreedilyDistributeToThreads.cpp.

#include "Passes.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "nova-gpu-greedy-distribute-to-threads"

using namespace mlir;

namespace {

/// Default target thread count for orphaned ops.
constexpr int64_t kDefaultTargetThreads = 256;

/// Returns true if `forall` has thread or warp GPU mapping attributes.
bool isThreadOrWarpMapped(scf::ForallOp forall) {
  auto mapping = forall.getMappingAttr();
  if (!mapping)
    return false;
  for (Attribute attr : mapping.getValue()) {
    if (isa<gpu::GPUThreadMappingAttr>(attr) ||
        isa<gpu::GPUWarpMappingAttr>(attr))
      return true;
  }
  return false;
}

/// Tile `tilingInterfaceOp` into a thread-mapped scf.forall using derived
/// thread tile sizes.  Greedily fuses all producers.  If tiling fails,
/// returns silently (best-effort — later verification catches failures).
void tileToThreads(RewriterBase &rewriter,
                   TilingInterface tilingInterfaceOp) {
  rewriter.setInsertionPoint(tilingInterfaceOp);
  MLIRContext *ctx = rewriter.getContext();

  auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingInterfaceOp.getOperation());
  if (!linalgOp)
    return;

  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  auto iterTypes = linalgOp.getIteratorTypesArray();
  int64_t numLoops = loopRanges.size();

  // Only tile parallel dimensions.
  SmallVector<int64_t> parallelRanges;
  SmallVector<int> parallelDims;
  for (int i = 0; i < (int)numLoops; ++i) {
    if (iterTypes[i] == utils::IteratorType::parallel &&
        loopRanges[i] != ShapedType::kDynamic && loopRanges[i] > 0) {
      parallelRanges.push_back(loopRanges[i]);
      parallelDims.push_back(i);
    }
  }

  if (parallelDims.empty())
    return;

  // Get element bit width for vectorization hint.
  unsigned elemBits = 32;
  if (!linalgOp->getResultTypes().empty()) {
    if (auto shaped = dyn_cast<ShapedType>(linalgOp->getResultTypes()[0]))
      elemBits = shaped.getElementType().getIntOrFloatBitWidth();
    else if (linalgOp->getResultTypes()[0].isIntOrFloat())
      elemBits = linalgOp->getResultTypes()[0].getIntOrFloatBitWidth();
  }

  // Derive thread tiles for the parallel dims.
  // Try progressively smaller thread counts until we find one that gives
  // a valid tiling (total threads ≤ 1024 and exact divisibility).
  // deriveThreadTileSizes returns all-ones when flatTrips % target != 0,
  // which would create threadCount = flatTrips (potentially millions).
  SmallVector<int64_t> parallelTiles;
  static constexpr int64_t kMaxThreads = 1024;
  bool foundValidTiling = false;

  int64_t flatTrips = 1;
  for (int64_t r : parallelRanges)
    flatTrips *= r;

  for (int64_t target = kDefaultTargetThreads; target >= 32;
       target /= 2) {
    parallelTiles =
        nova::deriveThreadTileSizes(parallelRanges, target, elemBits);

    // Compute actual thread count from derived tiles.
    int64_t actualThreads = 1;
    for (int i = 0; i < (int)parallelRanges.size(); ++i)
      actualThreads *= (parallelRanges[i] / parallelTiles[i]);

    if (actualThreads <= kMaxThreads && actualThreads > 1) {
      foundValidTiling = true;
      break;
    }
  }

  if (!foundValidTiling)
    return;

  // Build full tile sizes (0 for reduction dims, derived for parallel dims).
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));
  for (int i = 0; i < (int)parallelDims.size(); ++i)
    tileSizes[parallelDims[i]] = rewriter.getIndexAttr(parallelTiles[i]);

  // Check if any tile is non-zero.
  bool anyNonZero = llvm::any_of(tileSizes, [](OpFoldResult ofr) {
    auto cst = getConstantIntValue(ofr);
    return cst && *cst != 0;
  });
  if (!anyNonZero)
    return;

  // Build tiling options.
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  // Build GPU thread mapping (descending: innermost dim gets LinearDim0).
  SmallVector<Attribute> mapping;
  int idx = 0;
  for (auto size : tileSizes) {
    if (!isZeroInteger(size)) {
      unsigned mappingId =
          static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
      mapping.push_back(gpu::GPUThreadMappingAttr::get(
          ctx, static_cast<gpu::MappingId>(mappingId)));
    }
  }
  tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));

  // Fusion control: always fuse producers, never yield replacements.
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        /*yieldProducerReplacement=*/false};
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  // Tile and fuse.
  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults))
    return;

  // Replace the tiling root.
  for (OpResult res : tilingInterfaceOp->getResults()) {
    if (auto replacement = tiledResults->replacements.lookup(res))
      rewriter.replaceAllUsesWith(res, replacement);
  }

  if (tilingInterfaceOp->use_empty())
    rewriter.eraseOp(tilingInterfaceOp);
}

/// Recursively process the given region, tiling all tilable operations not
/// already inside a thread-mapped scf.forall.
void processRegion(RewriterBase &rewriter, Region *region) {
  for (Block &block : llvm::reverse(region->getBlocks())) {
    SmallVector<Operation *> targetOps =
        llvm::map_to_vector(llvm::reverse(block.getOperations()),
                            [](Operation &op) { return &op; });

    for (Operation *op : targetOps) {
      if (op->use_empty())
        continue;

      // Skip thread/warp-mapped foralls — already distributed.
      if (auto forall = dyn_cast<scf::ForallOp>(op)) {
        if (isThreadOrWarpMapped(forall))
          continue;
      }

      // Tile linalg ops to threads.
      if (auto tilableOp = dyn_cast<TilingInterface>(op)) {
        if (isa<linalg::LinalgOp>(op)) {
          tileToThreads(rewriter, tilableOp);
          continue;
        }
      }

      // Recurse into nested regions (e.g., block-mapped foralls, scf.for).
      for (auto &nestedRegion : op->getRegions())
        processRegion(rewriter, &nestedRegion);
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NovaGPUGreedilyDistributeToThreadsPass
    : public PassWrapper<NovaGPUGreedilyDistributeToThreadsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUGreedilyDistributeToThreadsPass);

  StringRef getArgument() const override {
    return "nova-gpu-greedy-distribute-to-threads";
  }
  StringRef getDescription() const override {
    return "Greedily distribute orphaned parallel ops to GPU threads";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, affine::AffineDialect,
                    gpu::GPUDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());
    for (auto &region : funcOp->getRegions())
      processRegion(rewriter, &region);
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API (in mlir::nova namespace)
//===----------------------------------------------------------------------===//

namespace mlir::nova {

std::unique_ptr<Pass> createNovaGPUGreedilyDistributeToThreadsPass() {
  return std::make_unique<NovaGPUGreedilyDistributeToThreadsPass>();
}

void registerNovaGPUGreedilyDistributeToThreadsPass() {
  PassRegistration<NovaGPUGreedilyDistributeToThreadsPass>();
}

} // namespace mlir::nova
