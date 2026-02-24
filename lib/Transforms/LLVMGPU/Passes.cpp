#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir;

namespace mlir::nova {

// Pipeline matching IREE's addGPUTileAndFusePassPipeline order.
//
// IREE reference (LLVMGPU/Passes.cpp addGPUTileAndFusePassPipeline):
//
//   Step 0  : SelectLoweringStrategy   → stamp #lowering_config on every matmul
//   Step 1  : TileAndDistribute        → scf.forall {block} per workgroup
//           : ConfigTrackingCanonalize → propagate config to new tiled ops
//   Step 2  : PadOperands              → pad A/B/C to static tile sizes
//           : ConfigTrackingCanonalize
//   Step 3  : PromoteMatmulOperands    → global→shared two-stage copy + barrier
//           : ConfigTrackingCanonalize
//   Step 4  : TilingLevel::Reduction   → scf.for over K dimension
//           : ConfigTrackingCanonalize  ← CRITICAL: new matmul inside for loop
//                                         must inherit config for Thread tiling
//   Step 5  : TilingLevel::Thread      → per-thread M/N register tiles
//           : ConfigTrackingCanonalize
//   Step 6  : TilingLevel::Subgroup    → per-subgroup (warp) M/N tiles
//           : ConfigTrackingCanonalize
//   Step 7  : Bufferize                → tensor → memref
//
// NOTE: After each tiling step, ops are replaced.  Plain canonicalize would
// drop the `lowering_config` attribute.  ConfigTrackingCanonalize propagates
// it to the replacement ops so every subsequent tiling level can read it.

void addNovaGPUOptimizedPipeline(OpPassManager &pm,
                                  StringRef cudaArch) {
  // -------------------------------------------------------------------------
  // Step 0: Select lowering strategy
  // Computes tile sizes / MMA intrinsic / promoted operands for every linalg
  // matmul and attaches #nova.lowering_config dict attribute.
  // Mirrors IREE's LLVMGPUSelectLoweringStrategy.
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUSelectLoweringStrategyPass(cudaArch));

  // -------------------------------------------------------------------------
  // Step 1: Tile and distribute to workgroups using scf.forall {block_id}
  // Fuses producers/consumers into the forall loop.
  // Mirrors IREE's tileAndDistributeToWorkgroup(useForall=true).
  // -------------------------------------------------------------------------
  pm.addPass(createNovaTileAndDistributeToWorkgroupsPass());
  // Config-tracking canonicalize: after tile-and-distribute, the original root
  // op is replaced by a new op inside the forall — propagate its config.
  pm.addPass(createNovaConfigTrackingCanonicalizerPass());
  pm.addPass(createCSEPass());

  // -------------------------------------------------------------------------
  // Step 2: Pad operands to static tile sizes
  // Enables aligned shared memory copies and better vectorization.
  // Mirrors IREE's createGPUPadOperandsPass().
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUPadOperandsPass());
  pm.addPass(createNovaConfigTrackingCanonicalizerPass());
  pm.addPass(createCSEPass());

  // -------------------------------------------------------------------------
  // Step 3: Promote matmul A/B operands to shared memory
  // Inserts two-stage copy (global→shared) + nova.fusion_barrier.
  // Mirrors IREE's createGPUPromoteMatmulOperandsPass().
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUPromoteMatmulOperandsPass());
  pm.addPass(createNovaConfigTrackingCanonicalizerPass());
  pm.addPass(createCSEPass());

  // -------------------------------------------------------------------------
  // Step 4: Tile reduction (K) dimension → sequential scf.for
  // Fuses A/B linalg.copy producers into the K-loop.
  // Mirrors IREE's GPUApplyTilingLevel(Reduction).
  // NOTE: After this pass, the linalg.matmul moves inside scf.for.
  //       The ConfigTrackingCanonalize below propagates the config into the
  //       new inner matmul so Thread tiling can read tile sizes from it.
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUApplyTilingLevelReductionPass());
  // CRITICAL: this canonicalize MUST be config-tracking — the matmul inside
  // the K-loop is a NEW op without the config attr until we propagate it.
  pm.addPass(createNovaConfigTrackingCanonicalizerPass());
  pm.addPass(createCSEPass());

  // -------------------------------------------------------------------------
  // Step 5: Tile thread-level M/N dimensions → per-thread register tiles
  // Creates a scf.for (or scf.forall) loop over the per-thread M/N tile.
  // Mirrors IREE's GPUApplyTilingLevel(Thread).
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUApplyTilingLevelThreadPass());
  pm.addPass(createNovaConfigTrackingCanonicalizerPass());
  pm.addPass(createCSEPass());

  // -------------------------------------------------------------------------
  // Step 6: Tile subgroup (warp) M/N dimensions → per-warp tiles
  // Mirrors IREE's GPUApplyTilingLevel(Subgroup).
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUApplyTilingLevelSubgroupPass());
  pm.addPass(createNovaConfigTrackingCanonicalizerPass());
  pm.addPass(createCSEPass());

  // -------------------------------------------------------------------------
  // Step 7: Fuse and hoist parallel loops
  // After thread/subgroup tiling we have multiple producer scf.forall {thread}
  // and consumer scf.forall {thread} ops. This pass greedily merges them into
  // a single flat forall and hoists foralls out of the K-loop when safe.
  // CRITICAL: must run BEFORE bufferization (uses tensor-level IR).
  // Mirrors IREE's createGPUFuseAndHoistParallelLoopsPass().
  // -------------------------------------------------------------------------
  pm.addPass(createNovaGPUFuseAndHoistParallelLoopsPass());

  // -------------------------------------------------------------------------
  // Step 7 (was TODO): GPU-aware bufferization
  // Erases nova.fusion_barrier, infers memory spaces, bufferizes with GPU
  // allocation function. Mirrors IREE's addGPUBufferizePasses().
  // -------------------------------------------------------------------------
  addNovaGPUBufferizePasses(pm);

  // TODO: DistributeForall (IREE: GPUDistributeForallPass)
  // Lowers scf.forall {gpu.thread} → gpu.thread_id indexing.
}

// --- Pass Registration ---

void registerNovaLLVMGPUPasses() {
    registerNovaConfigTrackingCanonicalizerPass();
    registerNovaGPUSelectLoweringStrategyPass();
    registerNovaTileAndDistributePass();
    registerNovaGPUPadOperandsPass();
    registerNovaGPUPromoteMatmulOperandsPass();
    registerNovaGPUApplyTilingLevelReductionPass();
    registerNovaGPUApplyTilingLevelThreadPass();
    registerNovaGPUApplyTilingLevelSubgroupPass();
    registerNovaGPUFuseAndHoistParallelLoopsPass();
    registerNovaGPUEraseFusionBarriersPass();
    registerNovaGPUInferMemorySpacePass();
    registerNovaGPUComprehensiveBufferizePass();
    registerNovaEliminateEmptyTensorsPass();
}

// ---------------------------------------------------------------------------
// addNovaGPUBufferizePasses
// Mirrors IREE's addGPUBufferizePasses() from LLVMGPU/Passes.cpp.
//
// Three steps in order:
//  1. NovaGPUInferMemorySpacePass: tag every unmarked alloc_tensor as
//     workgroup or private based on usage pattern.
//  2. createEmptyTensorToAllocTensorPass: convert tensor.empty → alloc_tensor.
//  3. NovaGPUComprehensiveBufferizePass: erase nova.fusion_barrier ops
//     (GAP 2) then run OneShotBufferize with GPU-aware alloc/copy fns (GAP 3).
// ---------------------------------------------------------------------------
void addNovaGPUBufferizePasses(OpPassManager &pm) {
  // Pre-bufferize passes.
  pm.addPass(createNovaEliminateEmptyTensorsPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(createNovaGPUInferMemorySpacePass());

  // GPU-aware comprehensive bufferize: erases nova.fusion_barrier (GAP 2)
  // and runs OneShotBufferize with GPU alloc/copy fns (GAP 3).
  pm.addPass(createNovaGPUComprehensiveBufferizePass());

  // Post-bufferization cleanup.
  pm.addPass(memref::createResolveShapedTypeResultDimsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

} // namespace mlir::nova

