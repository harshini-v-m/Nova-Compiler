//===- Passes.cpp - Nova GPU Pass Pipeline --------------------------------===//
//
// Defines the Nova GPU optimized compilation pipeline and the
// addNovaGPUBufferizePasses helper.
//
// The main entry point is addNovaGPUOptimizedPipeline(), which mirrors IREE's
// addGPUTileAndFusePassPipeline in LLVMGPU/Passes.cpp.  It drives the full
// tensor → PTX lowering sequence: Nova dialect lowering, TOSA conversion,
// tiling and fusion, GPU-aware bufferization, forall-to-gpu.launch conversion,
// NVVM lowering, and final LLVM IR generation.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Transforms/Passes.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/GenerateDynamicWrapper.h"
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
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
// Nova frontend translation passes
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
// TOSA conversion passes
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
// LLVM lowering
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "Compiler/Transforms/FixGpuLaunch.h"

#include "Compiler/Transforms/LLVMGPU/LoopSplit.h"
#include "Compiler/Transforms/LLVMGPU/NovaScfLoopUnroll.h"
#include "Compiler/Transforms/LLVMGPU/NovaScfLoopVectorize.h"

// Vectorization + tensor-core pipeline includes (Batch 2+)
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
// NVGPU tensor-core hardware mapping (Batch 4)
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;

namespace mlir::nova
{

  /*
  --------------------------------------------------
  CORE LOGIC — addNovaGPUOptimizedPipeline

  Full Nova GPU compilation pipeline, mirroring IREE's
  addGPUTileAndFusePassPipeline (LLVMGPU/Passes.cpp).

  Pipeline order (step numbers match IREE reference):

    Step -1   : Fuse elementwise ops before tiling
                Chains like exp→log must become a single linalg.generic
                BEFORE the tiling pipeline stamps per-op configs. Without
                this, each op gets its own thread-mapped forall and
                FuseAndHoist may not converge.

    Step -0.5 : Fold unit-extent dims before tiling
                Softmax keepdims reductions produce rank-mismatched ops
                that block fusion.  Folding here collapses them.

    Step 0    : SelectLoweringStrategy
                Stamps #nova.lowering_config on every matmul/reduction op.

    Step 1    : TileAndDistribute → scf.forall {block} per workgroup
    Step 2    : PadOperands → pad A/B/C to static tile sizes
    Step 3    : PromoteMatmulOperands → global→shared copy + barrier [BEFORE K-TILE]
    Step 4    : ApplyTilingLevelReduction → K-dim → scf.for [AFTER PROMOTE]
    Step 5    : ApplyTilingLevelThread → per-thread M/N register tiles
    Step 6    : FuseAndHoistParallelLoops
    Step 7    : NormalizeLoopBounds
    Step 7.75 : Post-tiling Linalg cleanup (GeneralizeNamedOps, ElementwiseFusion)
    Step 8    : GPU-aware bufferization (tensor → memref)
    Step 8.5  : Normalize again — remove degenerate (1,1) foralls
    Step 9    : scf.forall → gpu.launch (dynamic block dims)
    Step 10   : linalg → scf loops
    Step 11   : Insert gpu.barrier at workgroup memory write→read transitions
    Step 12   : Outline gpu.launch bodies → gpu.module kernels
    Step 12.5 : Convert workgroup memref.alloc → memref.global inside gpu.module
    Step 12.75: Convert host-side cross-kernel allocs to gpu.alloc
    Step 13   : Full CUDA/NVVM LLVM lowering

  IMPORTANT: After each tiling step, ops are replaced. Plain canonicalize
  drops the `lowering_config` attribute. ConfigTrackingCanonicalize propagates
  it to the replacement ops so every subsequent tiling level can read it.
  --------------------------------------------------
  */

  // CORE LOGIC
  void addNovaGPUOptimizedPipeline(OpPassManager &pm,
                                   StringRef cudaArch)
  {
    // Default to sm_86 (RTX 3060 / Ampere) when no architecture is specified.
    StringRef arch = cudaArch.empty() ? "sm_86" : cudaArch;

    // ---- Nova dialect → Arith / Tosa / Linalg lowering ----
    // Strip #nova.device encoding from all tensor types before conversion.
    // The conversion passes (NovaToArith/Tosa/Linalg) use applyPartialConversion
    // with no TypeConverter, so they cannot materialize device-encoded types to
    // plain tensor types. RemDevAttrPass rewrites all result types in-place,
    // making every tensor<..., #nova.device<"N">> into a plain tensor<...>.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createRemDevAttrPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createNovaToLinalgGenericLoweringPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createNovaToLinalgNamedPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToTensorPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // -------------------------------------------------------------------------
    // Step -1: Fuse elementwise ops before tiling
    //
    // Chains like exp→exp2→log→log2→log10 must become a single linalg.generic
    // BEFORE the tiling pipeline stamps per-op configs and creates per-op
    // scf.forall loops. Without this, each elementwise op gets its own
    // thread-mapped forall and the FuseAndHoist pass may not converge
    // (non-deterministic pattern ordering causes intermittent stalls).
    // IREE performs the same fusion before its tile-and-fuse pipeline.
    // -------------------------------------------------------------------------
    // pm.addPass(createLinalgElementwiseOpFusionPass());
    // pm.addNestedPass<mlir::func::FuncOp>(createNovaFuseReductionIntoProducerPass());
    pm.addNestedPass<mlir::func::FuncOp>(createFuseMatmulBiasPass());
    pm.addNestedPass<mlir::func::FuncOp>(createNovaElementwiseOpFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(createNovaCheckInsParallelFuse());
    pm.addNestedPass<mlir::func::FuncOp>(createNovaLinalgHorizontalFusionPass());
     pm.addNestedPass<mlir::func::FuncOp>(createNovaMultiConsumerFusion());
    // pm.addNestedPass<mlir::func::FuncOp>(createNovaLinalgVerticalFusionPass());

    

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step -0.5: Fold unit-extent dims before tiling
    //
    // Softmax lowering produces keepdims reductions (e.g. 4x1x8 → 4x1) that
    // insert tensor.expand_shape between 2D elementwise and 3D reduction ops.
    // This rank mismatch prevents the tiling pass from fusing all ops into a
    // single GPU kernel. Folding unit dims here collapses the reductions to
    // 2D (matching the elementwise ops), enabling uniform tiling and fusion.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 0: Select lowering strategy
    // Stamps #nova.lowering_config on each linalg contraction/reduction op.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUSelectLoweringStrategyPass(arch));

    // -------------------------------------------------------------------------
    // Step 1: Tile and distribute to workgroups
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaTileAndDistributeToWorkgroupsPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 2: Pad operands to static multiples of tile sizes
    //
    // Reads the "padding" list from each op's lowering_config (set by
    // SelectLoweringStrategy). Pads parallel dims to workgroup tile and
    // reduction dims to reduction tile. This prevents dynamic alloca in
    // GPU kernels and ensures correct K-dimension coverage.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUPadOperandsPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 3: Promote operands to shared memory   [BEFORE K-TILING]
    //
    // IREE order: Pad → Promote → Reduction tiling.
    // Promotion inserts alloc_tensor(workgroup) + copy + fusion_barrier on
    // the FULL workgroup slice. Then reduction tiling (Step 4) creates the
    // scf.for K-loop, which tiles the copies along K — each K-iteration
    // copies only [wgM × kStep] into shared memory. BufferLoopHoisting
    // (post-bufferize) then hoists the alloc out of the K-loop so it is
    // reused across iterations.
    //
    // If promotion were done AFTER K-tiling, the operand shapes would
    // already be K-sliced, making the shared memory allocation pattern
    // harder to set up correctly.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUPromoteMatmulOperandsPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 4: Tile reduction (K) dimension   [AFTER PROMOTION]
    //
    // After K-tiling, the matmul operates on [wgM × kStep] and [kStep × wgN]
    // slices. The promoted copies (from Step 3) are also tiled along K,
    // so each K-iteration loads only kStep-sized data into shared memory.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelReductionPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 5: Subgroup (warp) tiling — BEFORE thread tiling
    //
    // Splits the workgroup tile across warps via scf.forall + GPUWarpMappingAttr.
    // For MMA configs: each warp gets a subgroupTile-sized slice (e.g. 32×16).
    // For SIMT configs: subgroupTiles are all 0, pass is a no-op.
    //
    // Must run BEFORE thread tiling so thread foralls are nested inside warp
    // foralls. The zero-tile logic in ApplyTilingLevelThread detects
    // isInsideWarpForall && mmaKind!=0 and zeros out thread tiles, deferring
    // intra-warp distribution to ConvertVectorToGPU(nvgpu).
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUApplyTilingLevelSubgroupPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 5b: Thread tiling — per-thread register tiles within each warp
    //
    // For MMA ops (threadTiles=[0,0]): no thread forall created — matmul stays
    // at subgroup (warp) granularity for vectorization + ConvertVectorToGPU.
    // For SIMT ops: normal per-thread tiling applies.
    // For copies: derived thread tiles recomputed from targetThreads.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUApplyTilingLevelThreadPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 6: Normalize forall loop bounds (lb=0, step=1)
    // Must run BEFORE FuseAndHoist — FuseForalls requires isNormalized()
    // (step==1) to compute flat trip counts for fusion matching.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 6.5: Fuse and hoist parallel loops
    // Now that foralls are normalized (lb=0, step=1), FuseForalls can compare
    // flat trip counts and merge copy foralls into matmul foralls.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUFuseAndHoistParallelLoopsPass());

    // -------------------------------------------------------------------------
    // Step 6.6: Greedily distribute orphaned ops to threads
    // After FuseAndHoist, any parallel op NOT inside a thread-mapped forall
    // is distributed here as a fallback.  Without this, orphaned ops launch
    // kernels with threads(1,1,1).
    // Mirrors IREE's GPUGreedilyDistributeToThreads pass.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUGreedilyDistributeToThreadsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 6.75: Lower barrier_region → two value_barrier ops
    // FuseAndHoist generates nova.barrier_region for dimension-mismatched
    // forall fusion. Lower these before bufferization.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPULowerBarrierRegionPass());

    // -------------------------------------------------------------------------
    // Stage 14: Post-tiling Linalg cleanup  (Step 7.75)
    //
    // For sm_80+: skip GeneralizeNamedOps here so linalg.matmul/batch_matmul
    // remains a named op. GenericVectorization (Stage 18) will directly emit
    // vector.contract from the named op. linalg.generic with a matmul body
    // would produce vector.multi_reduction instead, which loses the
    // vector.contract → nvgpu.mma.sync tensor-core path.
    //
    // For pre-sm_80 targets: generalize first so everything goes through the
    // generic vectorization and outer-product lowering path.
    //
    // ElementwiseFusion always runs to fuse any epilogue elementwise ops not
    // consumed during workgroup tiling.
    // -------------------------------------------------------------------------
    if (arch.starts_with("sm_") && arch.substr(3).compare("80") >= 0) {
      // sm_80+: keep linalg.matmul as named op for vector.contract path.
      pm.addPass(createLinalgElementwiseOpFusionPass());
    } else {
      pm.addPass(createLinalgGeneralizeNamedOpsPass());
      pm.addPass(createLinalgElementwiseOpFusionPass());
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Stage 15: Cast Types to Fit MMA  [BEFORE Vectorization — CRITICAL]
    //
    // Normalizes strided-memref indices for WMMA load/store ops before NVVM
    // legalization. For the nvgpu.mma.sync (Ampere sm_80+) path this is a
    // no-op (nvgpu.mma.sync has no SubgroupMma* ops). For the WMMA path
    // (sm_70–sm_75) it folds subview/reinterpret_cast offsets into indices.
    //
    // Must run BEFORE GenericVectorization so linalg ops have correct types
    // when vectorized to vector.contract.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUCastTypeToFitMMAPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Stage 18: Vectorization  [CORE TENSOR-LEVEL VECTORIZATION]
    //
    //   GenericVectorization:
    //     - linalg.matmul / batch_matmul → vector.contract (named ops, sm_80+)
    //     - linalg.generic (matmul body) → vector.contract
    //     - other linalg.generic → vector.multi_reduction → InnerReduction
    //       → vector.contract (dot-product form, single rounding via llvm.fma)
    //     Propagates lowering_config (mma_kind) to the new vector.contract.
    //
    //   OptimizeVectorTransfer:
    //     Folds redundant transfer_read/write pairs and sinks/hoists vector ops
    //     to improve memory access patterns before later passes.
    //
    //   HoistVectorExtractInsertSlice:
    //     Hoists extract/insert out of loops to expose contiguous vector
    //     accesses for future UnrollToIntrinsics (Batch 3).
    //
    //   DropVectorUnitDims:
    //     Removes unit dimensions from vector types
    //     (e.g. vector<1x16xf32> → vector<16xf32>) to simplify MMA matching.
    //
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUGenericVectorizationPass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUHoistVectorExtractInsertSlicePass());
    // pm.addNestedPass<func::FuncOp>(createNovaGPUHoistVectorExtractInsertSlicePass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUDropVectorUnitDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Stage 19: Unroll vector.contract to MMA Intrinsic Shapes  [Batch 3]
    //
    // Reads mma_kind from lowering_config and unrolls vector.contract ops to
    // match the exact MMA instruction tile shape:
    //   MMA_SYNC_TF32_16x8x8  → 16×8×8 vector.contract fragments
    //   MMA_SYNC_F16_16x8x16  → 16×8×16 vector.contract fragments
    //   WMMA_F32_16x16x16     → 16×16×16 fragments
    //
    // Each fragment corresponds to one future nvgpu.mma.sync or
    // gpu.subgroup_mma_compute call. The extract_strided_slice ops produced
    // here are folded by FoldExtractStridedSliceFromTransferRead inside
    // GpuHardwareMappingPass post-outlining (Batch 4).
    //
    // MUST run AFTER GenericVectorization (vector.contract exists) and AFTER
    // CastTypeToFitMMA (types are correct).
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUUnrollToIntrinsicsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Stage 20: Generalize Remaining Named Ops  (sm_80+ only, post-vectorize)
    //
    // For sm_80+: named contraction ops are already gone (vectorized in
    // Stage 18). Generalize any remaining named non-contraction linalg ops
    // so bufferization handles them uniformly.
    // For pre-sm_80: already generalized in Stage 14.
    // -------------------------------------------------------------------------
    if (arch.starts_with("sm_") && arch.substr(3).compare("80") >= 0) {
      pm.addPass(createLinalgGeneralizeNamedOpsPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }

    // -------------------------------------------------------------------------
    // Stage 21: Host-Side Vector Transfer Lowering  [BEFORE Bufferization]
    //
    // Lower rank>1 vector.transfer_read/write to loops of rank-1 transfers.
    // This prevents host-side arith.divui (grid/block tiling math) from
    // triggering the hardware-native vector-to-gpu legalizer.
    //
    // IMPORTANT: targetRank=1 (NOT 0). targetRank=0 would scalarize ALL
    // vector.transfer_read/write ops including those feeding MMA-configured
    // vector.contract ops inside warp foralls. Scalarizing MMA transfers
    // destroys the 2D vector shapes that ConvertVectorToGPU needs to match
    // for nvgpu.mma.sync generation. Rank-1 transfers are handled later by
    // ConvertVectorToLLVMPass inside the GPU module (Stage 38d2).
    // -------------------------------------------------------------------------
    {
      auto &funcPm = pm.nest<func::FuncOp>();
      VectorTransferToSCFOptions opts;
      opts.setTargetRank(1);
      funcPm.addPass(createConvertVectorToSCFPass(opts));
      funcPm.addPass(createCanonicalizerPass());
      funcPm.addPass(createCSEPass());
    }

    // -------------------------------------------------------------------------
    // Step 8: GPU-aware bufferization (tensor → memref)
    //
    // Uses NovaGPUComprehensiveBufferizePass which has a GPU-aware copy
    // function (gpuCopyFn) that emits linalg.copy instead of memref.copy
    // inside scf.forall bodies. This is critical: memref.copy would later
    // become gpu.memcpy (via ConvertMemRefToGpu) which convert-gpu-to-nvvm
    // cannot lower when operands have strided layouts. linalg.copy is
    // expanded to scf.for + memref.load/store by createConvertLinalgToLoopsPass.
    // -------------------------------------------------------------------------
    addNovaGPUBufferizePasses(pm);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertBufferizationToMemRefPass());

    // -------------------------------------------------------------------------
    // Stage 23: Vectorize Shared Memory Copies  [AFTER bufferization]  (Batch 4)
    //
    // Vectorizes linalg.copy ops on shared memory buffers to produce wide
    // vector loads targeting 128-bit LDS.128 instructions on sm_80+.
    // Must run AFTER bufferization (memrefs must exist).
    // These copies are NOT unrolled to MMA shapes — they target coalesced
    // load bandwidth, not compute.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUVectorizeMemrefCopyPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 8.5: Eliminate degenerate single-iteration foralls
    //
    // After bufferization, some scf.forall ops with bounds (1, 1) may remain
    // (e.g. from padding small tensors to tile size). These block-mapped
    // degenerate foralls inside thread-mapped contexts cause the transform
    // interpreter to fail. Re-running normalize-loop-bounds eliminates them.
    // -------------------------------------------------------------------------
       pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());


    pm.addNestedPass<func::FuncOp>(createNovaGPUFillCopyForwardingPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());

   pm.addNestedPass<func::FuncOp>(createNovaGPUCoalesceWorkgroupBuffersPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());
   
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 9: scf.forall → gpu.launch (C++ pass with dynamic block dims)
    //
    // Replaces the transform script (gpu_forall_to_launch.mlir) which
    // hardcoded block_dims = [32, 32, 1]. This pass computes block dims
    // from the actual thread-mapped forall iteration bounds.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUMapForallToGPUPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 10: Lower remaining linalg → scf loops
    // -------------------------------------------------------------------------
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());


    // start of my Pass
    

    pm.addNestedPass<func::FuncOp>(
        mlir::nova::createSCFScalarizeAccumulatorPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Warp butterfly shuffle: replace double-atomic reduction patterns
    // (which over-count by N×) with register-based shuffle + thread-0 write.
    pm.addNestedPass<func::FuncOp>(
        mlir::nova::createNovaWarpShuffleReductionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(mlir::nova::createNovaScfLoopSplitPass());
    pm.addPass(createCanonicalizerPass());

    // pm.addNestedPass<func::FuncOp>(
    //     mlir::nova::createNovaScfLoopVectorizePass(16));
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(createCSEPass());

    // pm.addNestedPass<func::FuncOp>(mlir::nova::createNovaScfLoopUnrollPass(4));
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(createCSEPass());
     
     pm.addNestedPass<mlir::func::FuncOp>(createNovaRepositionStorePass());

    // // end of my pass

    // -------------------------------------------------------------------------
    // Step 11: Insert gpu.barrier at workgroup memory write→read transitions
    //
    // Must run AFTER linalg-to-loops (so scalar store/load ops are visible)
    // and AFTER forall→gpu.launch and bufferization.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUInsertWorkgroupBarriersPass());

    // -------------------------------------------------------------------------
    // Step 12: Outline gpu.launch bodies into gpu.module kernels
    //
    // Must happen BEFORE the shared-mem conversion so that memref.global +
    // memref.get_global for workgroup memory are created INSIDE the gpu.module
    // (matching IREE's ConvertSharedMemAllocOp flow), not in the outer host
    // module where finalize-memref-to-llvm cannot lower
    // #gpu.address_space<workgroup> types.
    // -------------------------------------------------------------------------
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(mlir::nova::createRenameGpuKernelsPass());
    pm.addPass(createCanonicalizerPass());

    // -------------------------------------------------------------------------
    // Stage 35: Device-Side Hardware Mapping  (inside gpu.GPUModuleOp)  [Batch 4]
    //
    // ALL hardware-native mapping runs inside the gpu.module to prevent
    // host-side preparational math from triggering MMA legalizers.
    //
    //   35a. GpuHardwareMappingPass (per gpu::GPUFuncOp):
    //        Step 1 — Convert non-indexing arith.divui/remui to affine.apply
    //                 (naked divisions crash the legalizer).
    //        Step 2 — FoldExtractStridedSlice(transfer_read) → narrowed
    //                 transfer_read. Required because PrepareVectorToMMAPatterns
    //                 refuses to match vector.contract whose LHS comes from
    //                 extract_strided_slice. UnrollToIntrinsics (Stage 19)
    //                 produces these slices; this fold recovers direct form.
    //        Step 3 — populatePrepareVectorToMMAPatterns(useNvGpu=true):
    //                 Transforms transfer_read/write → nvgpu fragment load/store
    //                 for m16n8k8 (TF32) and m16n8k16 (F16) MMA shapes.
    //
    //   35b. createConvertVectorToGPUPass(useNvGpu=true):
    //        vector.contract → nvgpu.mma.sync.
    //        Must run AFTER 35a (prepare patterns applied).
    //
    //   35c. FixMmaSyncTF32Pass: sets tf32_enabled on nvgpu.mma.sync for f32
    //        inputs that VectorToGPU left without the attribute, preventing
    //        ConvertNVGPUToNVVM from failing (Stage 38g).
    // -------------------------------------------------------------------------
    {
      auto &gpuHwPm = pm.nest<gpu::GPUModuleOp>();

      // 35a. GpuHardwareMappingPass.
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct GpuHardwareMappingPass
            : public PassWrapper<GpuHardwareMappingPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuHardwareMappingPass)

          void runOnOperation() override {
            auto funcOp = getOperation();

            // Step 1: Shield non-indexing divui/remui from the legalizer.
            funcOp.walk([&](Operation *op) {
              auto isIndexingMath = [](Value v) -> bool {
                for (auto &use : v.getUses()) {
                  if (auto r = dyn_cast<vector::TransferReadOp>(use.getOwner()))
                    for (auto idx : r.getIndices())
                      if (idx == v) return true;
                  if (auto w = dyn_cast<vector::TransferWriteOp>(use.getOwner()))
                    for (auto idx : w.getIndices())
                      if (idx == v) return true;
                }
                return false;
              };
              if (auto divOp = dyn_cast<arith::DivUIOp>(op)) {
                if (!isIndexingMath(divOp.getResult())) {
                  OpBuilder b(divOp);
                  auto map = AffineMap::get(
                      2, 0,
                      b.getAffineDimExpr(0).floorDiv(b.getAffineDimExpr(1)),
                      b.getContext());
                  auto applyOp = b.create<affine::AffineApplyOp>(
                      divOp.getLoc(), map,
                      ValueRange{divOp.getLhs(), divOp.getRhs()});
                  divOp.replaceAllUsesWith(applyOp.getResult());
                  divOp.erase();
                }
              } else if (auto remOp = dyn_cast<arith::RemUIOp>(op)) {
                if (!isIndexingMath(remOp.getResult())) {
                  OpBuilder b(remOp);
                  auto map = AffineMap::get(
                      2, 0,
                      b.getAffineDimExpr(0) % b.getAffineDimExpr(1),
                      b.getContext());
                  auto applyOp = b.create<affine::AffineApplyOp>(
                      remOp.getLoc(), map,
                      ValueRange{remOp.getLhs(), remOp.getRhs()});
                  remOp.replaceAllUsesWith(applyOp.getResult());
                  remOp.erase();
                }
              }
            });

            // Step 2: Fold extract_strided_slice(transfer_read) → narrowed
            //         transfer_read so LHS reaches MMA prepare patterns.
            {
              struct FoldExtractStridedSliceFromTransferRead
                  : public OpRewritePattern<vector::ExtractStridedSliceOp> {
                using OpRewritePattern::OpRewritePattern;
                LogicalResult matchAndRewrite(
                    vector::ExtractStridedSliceOp extractOp,
                    PatternRewriter &rewriter) const override {
                  auto extractType = cast<VectorType>(extractOp.getType());
                  if (extractType.getRank() != 2) return failure();
                  auto readOp = extractOp.getVector()
                                    .getDefiningOp<vector::TransferReadOp>();
                  if (!readOp) return failure();
                  if (!isa<MemRefType>(readOp.getBase().getType()))
                    return failure();
                  if (readOp.getMask() || readOp.hasOutOfBoundsDim())
                    return failure();
                  auto strides =
                      extractOp.getStrides().getAsValueRange<IntegerAttr>();
                  if (!llvm::all_of(strides,
                                    [](const APInt &v) { return v.isOne(); }))
                    return failure();
                  SmallVector<int64_t> offsets;
                  for (auto attr :
                       extractOp.getOffsets().getAsValueRange<IntegerAttr>())
                    offsets.push_back(attr.getSExtValue());
                  if ((int64_t)offsets.size() !=
                      (int64_t)readOp.getIndices().size())
                    return failure();
                  Location loc = extractOp.getLoc();
                  SmallVector<Value> newIndices =
                      llvm::to_vector(readOp.getIndices());
                  for (size_t i = 0; i < offsets.size(); ++i) {
                    if (offsets[i] != 0) {
                      Value off = rewriter.create<arith::ConstantIndexOp>(
                          loc, offsets[i]);
                      newIndices[i] =
                          rewriter.create<arith::AddIOp>(loc, newIndices[i], off);
                    }
                  }
                  SmallVector<bool> inBounds(2, true);
                  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
                      extractOp, extractType, readOp.getBase(), newIndices,
                      readOp.getPermutationMapAttr(), readOp.getPadding(),
                      /*mask=*/Value{},
                      rewriter.getBoolArrayAttr(inBounds));
                  return success();
                }
              };
              RewritePatternSet foldPatterns(&getContext());
              foldPatterns.add<FoldExtractStridedSliceFromTransferRead>(
                  &getContext());
              (void)applyPatternsAndFoldGreedily(funcOp,
                                                 std::move(foldPatterns));
            }

            // Step 2b: Decompose insert_strided_slice + transfer_write chains.
            //
            // UnrollToIntrinsics produces:
            //   %v0 = contract(...) : vector<16x8>  // MMA tile [0,0]
            //   %v1 = contract(...) : vector<16x8>  // MMA tile [0,8]
            //   %a  = insert_strided_slice %v0, %init {offsets=[0,0]}
            //   %b  = insert_strided_slice %v1, %a   {offsets=[0,8]}
            //   ...
            //   transfer_write %final, memref[base0, base1] : vector<32x16>
            //
            // ConvertVectorToGPU cannot handle insert_strided_slice in the
            // forward slice from contracts. We decompose into individual writes:
            //   transfer_write %v0, memref[base0+0,  base1+0]  : vector<16x8>
            //   transfer_write %v1, memref[base0+0,  base1+8]  : vector<16x8>
            //   ...
            {
              struct FoldInsertStridedSliceIntoTransferWrite
                  : public OpRewritePattern<vector::TransferWriteOp> {
                using OpRewritePattern::OpRewritePattern;
                LogicalResult matchAndRewrite(
                    vector::TransferWriteOp writeOp,
                    PatternRewriter &rewriter) const override {
                  // Only handle 2D writes to memref.
                  auto writeType = writeOp.getVectorType();
                  if (writeType.getRank() != 2) return failure();
                  if (!isa<MemRefType>(writeOp.getBase().getType()))
                    return failure();
                  if (writeOp.getMask() || writeOp.hasOutOfBoundsDim())
                    return failure();
                  // Must have identity permutation map.
                  if (!writeOp.getPermutationMap().isIdentity())
                    return failure();

                  // Trace back through the chain of insert_strided_slice ops
                  // to collect all individual tile values and their offsets.
                  struct TileInfo {
                    Value value;       // the vector<16x8> being inserted
                    SmallVector<int64_t, 2> offsets;
                  };
                  SmallVector<TileInfo> tiles;
                  Value current = writeOp.getVector();
                  bool foundChain = false;
                  while (auto insertOp =
                             current.getDefiningOp<vector::InsertStridedSliceOp>()) {
                    foundChain = true;
                    auto strides =
                        insertOp.getStrides().getAsValueRange<IntegerAttr>();
                    if (!llvm::all_of(strides,
                                      [](const APInt &v) { return v.isOne(); }))
                      return failure();
                    SmallVector<int64_t, 2> offsets;
                    for (auto attr :
                         insertOp.getOffsets().getAsValueRange<IntegerAttr>())
                      offsets.push_back(attr.getSExtValue());
                    if (offsets.size() != 2) return failure();
                    tiles.push_back({insertOp.getValueToStore(), offsets});
                    current = insertOp.getDest();
                  }
                  if (!foundChain) return failure();

                  // Emit individual transfer_write ops for each tile.
                  Location loc = writeOp.getLoc();
                  SmallVector<Value> baseIndices =
                      llvm::to_vector(writeOp.getIndices());
                  for (auto &tile : tiles) {
                    SmallVector<Value> newIndices =
                        llvm::to_vector(baseIndices);
                    for (size_t i = 0; i < tile.offsets.size(); ++i) {
                      if (tile.offsets[i] != 0) {
                        Value off = rewriter.create<arith::ConstantIndexOp>(
                            loc, tile.offsets[i]);
                        newIndices[i] = rewriter.create<arith::AddIOp>(
                            loc, newIndices[i], off);
                      }
                    }
                    auto tileType = cast<VectorType>(tile.value.getType());
                    SmallVector<bool> inBounds(tileType.getRank(), true);
                    rewriter.create<vector::TransferWriteOp>(
                        loc, tile.value, writeOp.getBase(), newIndices,
                        inBounds);
                  }
                  rewriter.eraseOp(writeOp);
                  return success();
                }
              };
              RewritePatternSet writeFoldPatterns(&getContext());
              writeFoldPatterns.add<FoldInsertStridedSliceIntoTransferWrite>(
                  &getContext());
              (void)applyPatternsAndFoldGreedily(funcOp,
                                                  std::move(writeFoldPatterns));
            }

            // Step 3: Prepare transfers → nvgpu fragment form.
            {
              RewritePatternSet patterns(&getContext());
              populatePrepareVectorToMMAPatterns(patterns, /*useNvGpu=*/true);
              if (failed(applyPatternsAndFoldGreedily(funcOp,
                                                      std::move(patterns))))
                return signalPassFailure();
            }
          }
          StringRef getArgument() const override {
            return "nova-gpu-hardware-mapping";
          }
        };
        return std::make_unique<GpuHardwareMappingPass>();
      }());

      // DEBUG: Dump full IR before ConvertVectorToGPU to diagnose unmatched contracts.
      // gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
      //   struct DumpBeforeVectorToGPUPass
      //       : public PassWrapper<DumpBeforeVectorToGPUPass,
      //                            OperationPass<gpu::GPUFuncOp>> {
      //     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpBeforeVectorToGPUPass)
      //     void runOnOperation() override {
      //       auto funcOp = getOperation();
      //       bool hasContracts = false;
      //       funcOp.walk([&](vector::ContractionOp) { hasContracts = true; });
      //       if (hasContracts) {
      //         llvm::errs() << "\n===== IR BEFORE ConvertVectorToGPU =====\n";
      //         funcOp.print(llvm::errs(), OpPrintingFlags().useLocalScope());
      //         llvm::errs() << "\n===== END IR DUMP =====\n\n";
      //       }
      //     }
      //     StringRef getArgument() const override {
      //       return "nova-dump-before-vector-to-gpu";
      //     }
      //   };
      //   return std::make_unique<DumpBeforeVectorToGPUPass>();
      // }());

      // 35b. vector.contract → nvgpu.mma.sync.
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>(
          createConvertVectorToGPUPass(/*useNvGpu=*/true));

      // 35b2. Clean up dead ops left by ConvertVectorToGPU.
      //
      // ConvertVectorToGPU replaces the dataflow (transfer_read → contract →
      // transfer_write) with (ldmatrix/scalar_load → mma.sync → vector.store)
      // but does NOT erase the original transfer_read + contract ops. They
      // become dead code. Without DCE, the subsequent LLVM lowering pass
      // attempts to lower these dead full-tile contracts via scalar loops,
      // causing compilation stalls on large matrices.
      gpuHwPm.addPass(createCanonicalizerPass());
      gpuHwPm.addPass(createCSEPass());

      // DEBUG: Dump IR AFTER ConvertVectorToGPU
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct DumpAfterVectorToGPUPass
            : public PassWrapper<DumpAfterVectorToGPUPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpAfterVectorToGPUPass)
          void runOnOperation() override {
            bool hasContracts = false;
            getOperation().walk([&](vector::ContractionOp) { hasContracts = true; });
            if (hasContracts) {
              llvm::errs() << "\n===== IR AFTER ConvertVectorToGPU (unconverted contracts remain) =====\n";
              getOperation().print(llvm::errs(), OpPrintingFlags().useLocalScope());
              llvm::errs() << "\n===== END POST-CONVERT DUMP =====\n\n";
            }
          }
          StringRef getArgument() const override {
            return "nova-dump-after-vector-to-gpu";
          }
        };
        return std::make_unique<DumpAfterVectorToGPUPass>();
      }());

      // 35c. Fix missing tf32_enabled on f32 nvgpu.mma.sync ops.
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct FixMmaSyncTF32Pass
            : public PassWrapper<FixMmaSyncTF32Pass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixMmaSyncTF32Pass)
          void runOnOperation() override {
            getOperation().walk([](nvgpu::MmaSyncOp mmaSyncOp) {
              auto aType =
                  dyn_cast<VectorType>(mmaSyncOp.getMatrixA().getType());
              if (aType && aType.getElementType().isF32() &&
                  !mmaSyncOp.getTf32Enabled())
                mmaSyncOp.setTf32Enabled(true);
            });
          }
          StringRef getArgument() const override {
            return "nova-fix-mma-sync-tf32";
          }
        };
        return std::make_unique<FixMmaSyncTF32Pass>();
      }());

      // Diagnostic: verify MMA conversion produced nvgpu.mma.sync ops.
      // Logs a warning when vector.contract ops survive ConvertVectorToGPU
      // unconverted — indicates a matching failure in PrepareVectorToMMA or
      // ConvertVectorToGPU that needs debugging.
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() {
        struct VerifyMMAConversionPass
            : public PassWrapper<VerifyMMAConversionPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyMMAConversionPass)
          void runOnOperation() override {
            int mmaCount = 0, contractCount = 0;
            bool hasMmaConfig = false;
            getOperation().walk([&](Operation *op) {
              if (isa<nvgpu::MmaSyncOp>(op)) mmaCount++;
              if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
                contractCount++;
                // Check if any ancestor carries mma_kind > 0 — that means
                // this contract was SUPPOSED to be MMA but failed to convert.
                Operation *cur = contractOp.getOperation();
                while (cur) {
                  if (auto cfg = getLoweringConfig(cur)) {
                    if (getMmaKindRaw(cfg) != 0) {
                      hasMmaConfig = true;
                      break;
                    }
                  }
                  cur = cur->getParentOp();
                }
                // Only dump details for contracts that were meant for MMA.
                if (hasMmaConfig && contractCount <= 3) {
                  llvm::errs() << "[nova-mma-verify] unconverted contract #"
                               << contractCount << ":\n";
                  llvm::errs() << "  LHS type: " << contractOp.getLhs().getType() << "\n";
                  llvm::errs() << "  RHS type: " << contractOp.getRhs().getType() << "\n";
                  llvm::errs() << "  ACC type: " << contractOp.getAcc().getType() << "\n";
                  llvm::errs() << "  indexing_maps: ";
                  for (auto map : contractOp.getIndexingMapsArray())
                    llvm::errs() << map << " ";
                  llvm::errs() << "\n";
                  if (auto defOp = contractOp.getLhs().getDefiningOp())
                    llvm::errs() << "  LHS def: " << defOp->getName() << "\n";
                }
              }
            });
            if (contractCount > 0 && mmaCount == 0 && hasMmaConfig)
              llvm::errs() << "[nova-mma-verify] WARNING: " << contractCount
                           << " vector.contract ops remain unconverted in "
                           << getOperation().getName()
                           << " — ConvertVectorToGPU did not match.\n";
            else if (contractCount > 0 && mmaCount == 0 && !hasMmaConfig)
              llvm::errs() << "[nova-mma-verify] SIMT: " << contractCount
                           << " vector.contract op(s) in "
                           << getOperation().getName()
                           << " using SIMT (CUDA core FMA) path.\n";
            else if (contractCount > 0 && mmaCount > 0)
              llvm::errs() << "[nova-mma-verify] PARTIAL: " << mmaCount
                           << " mma.sync + " << contractCount
                           << " unconverted contracts in "
                           << getOperation().getName() << "\n";
            else if (mmaCount > 0)
              llvm::errs() << "[nova-mma-verify] OK: " << mmaCount
                           << " nvgpu.mma.sync ops in "
                           << getOperation().getName() << "\n";
          }
          StringRef getArgument() const override {
            return "nova-verify-mma-conversion";
          }
        };
        return std::make_unique<VerifyMMAConversionPass>();
      }());

      gpuHwPm.addPass(createCanonicalizerPass());
      gpuHwPm.addPass(createCSEPass());

      // 35d. Scalarize MMA fragment vector stores.
      //
      // ConvertVectorToGPU generates vector.store of vector<2xf32> for MMA
      // result fragments (VectorToGPU.cpp:935). The LLVM 21 NVPTX backend
      // miscompiles these by packing two f32 into i64 and emitting:
      //   mov.b64 %rd, {%r, %r}; st.global.b32 [addr], %rd  (INVALID PTX)
      // Scalarizing to individual f32 memref.store ops avoids the backend bug.
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct ScalarizeMMAFragmentStoresPass
            : public PassWrapper<ScalarizeMMAFragmentStoresPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
              ScalarizeMMAFragmentStoresPass)
          void runOnOperation() override {
            SmallVector<vector::StoreOp> toScalarize;
            getOperation().walk([&](vector::StoreOp storeOp) {
              auto vecType = storeOp.getVectorType();
              if (vecType.getRank() == 1 && vecType.getDimSize(0) == 2 &&
                  vecType.getElementType().isF32())
                toScalarize.push_back(storeOp);
            });
            if (toScalarize.empty())
              return;

            OpBuilder builder(&getContext());
            for (auto storeOp : toScalarize) {
              builder.setInsertionPoint(storeOp);
              Location loc = storeOp.getLoc();
              Value vec = storeOp.getValueToStore();
              Value base = storeOp.getBase();
              SmallVector<Value> indices(storeOp.getIndices());

              for (int64_t i = 0; i < 2; ++i) {
                Value elem = builder.create<vector::ExtractOp>(
                    loc, vec, ArrayRef<int64_t>{i});
                SmallVector<Value> storeIndices(indices);
                if (!storeIndices.empty()) {
                  Value offset =
                      builder.create<arith::ConstantIndexOp>(loc, i);
                  storeIndices.back() = builder.create<arith::AddIOp>(
                      loc, storeIndices.back(), offset);
                }
                builder.create<memref::StoreOp>(loc, elem, base,
                                                storeIndices);
              }
              storeOp.erase();
            }
          }
          StringRef getArgument() const override {
            return "nova-scalarize-mma-fragment-stores";
          }
        };
        return std::make_unique<ScalarizeMMAFragmentStoresPass>();
      }());
      gpuHwPm.addPass(createCanonicalizerPass());
    }

    // -------------------------------------------------------------------------
    // Step 12.5: Convert workgroup memref.alloc → memref.global + get_global
    //
    // Runs INSIDE the gpu.module so globals are scoped to the kernel and
    // never appear in the outer host module.
    // Ported from IREE's ConvertSharedMemAllocOp + DropSharedMemoryDeallocOp.
    // -------------------------------------------------------------------------
    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      gpuPm.addPass(createNovaConvertSharedMemAllocsPass());
      gpuPm.addPass(createCanonicalizerPass());
    }

    // -------------------------------------------------------------------------
    // Step 12.75: Convert host-side cross-kernel allocations to GPU allocations
    //
    // Handles intermediate buffers (e.g. `%alloc = memref.alloc()`) on the
    // host side that are passed into GPU kernels, converting them from the
    // default memory space to `gpu.alloc` (which lowers to `cudaMalloc`).
    // -------------------------------------------------------------------------
    pm.addPass(nova::createConvertMemRefToGpuPass());
    pm.addPass(createCanonicalizerPass());

    // -------------------------------------------------------------------------
    // Step 13: Full CUDA/NVVM LLVM lowering
    //
    // Lowers gpu.module → PTX binary, then lowers host code → LLVM IR.
    // Mirrors IREE's addLowerToLLVMGPUPasses (LLVMGPU/Passes.cpp).
    // -------------------------------------------------------------------------

    // PERFORMANCE CRITICAL — 13.0: Attach NVVM target descriptor
    // These flags directly control PTX code quality:
    //   optLevel=3   → full LLVM O3 optimisations inside PTXAS
    //   fastFlag     → allow fast-math reassociation in PTXAS
    //   ftzFlag      → flush denormals to zero (halves operand precision cost)
    //   features=+ptx86 → PTX 8.6 (CUDA 13.0 / sm_86+).
    //   PTX 8.1+ is required for kernel params > 4352 bytes (MMA kernels
    //   pass large by-value accumulator vectors via param space).
    //   PTX 7.6 was limited to 4352 bytes → CUDA_ERROR_INVALID_PTX.
    //   LLVM 21 supports: ptx80, ptx86, ptx87, ptx88 (no ptx81-85).
    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip = arch.str();
    nvvmTargetOptions.features = "+ptx86";
    nvvmTargetOptions.optLevel = 3;
    nvvmTargetOptions.fastFlag = true;
    nvvmTargetOptions.ftzFlag = true;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

    // 13.1 — Outline GPU kernels and insert async tokens (required by
    //         GpuModuleToBinaryPass).
    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

    // 13.2 — Lower gpu.module contents to NVVM / LLVM.
    //
    // Pass ordering is critical:
    //   1. expand-strided-metadata : decompose memref.subview while still
    //      memref types (before gpu-to-nvvm changes pointer types).
    //   2. lower-affine            : affine.apply → arith ops.
    //   3. gpu.address_space lower : #gpu.address_space<private/workgroup/global>
    //      → NVVM integer address spaces (5/3/1) before finalizeMemRefToLLVM
    //      (which requires integer address spaces for LLVM type conversion).
    //   4. scf-to-cf               : lower SCF control flow.
    //   5. gpu-to-nvvm             : gpu.func signature + gpu ops → NVVM.
    //   6. remaining dialect conversions (index, arith, math).
    //   7. reconcile-unrealized-casts: erase conversion cast chains.
    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      // 38a. Strided metadata + alias folding (before NVVM pointer types).
      gpuPm.addPass(memref::createExpandStridedMetadataPass());
      gpuPm.addPass(memref::createFoldMemRefAliasOpsPass());
      // 38b. Residual MMA type fixes post-outlining (Batch 4).
      gpuPm.addPass(createNovaGPUCastTypeToFitMMAPass());
      // 38c. Affine math cleanup (index domain).
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createLowerAffinePass());
      gpuPm.addPass(createNovaGPULowerMemorySpacePass());
    gpuPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct VectorToSCFOnGPUFuncPass
            : public PassWrapper<VectorToSCFOnGPUFuncPass,
                                 OperationPass<gpu::GPUFuncOp>> {
            MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToSCFOnGPUFuncPass)
            void runOnOperation() override {
                RewritePatternSet patterns(&getContext());
                VectorTransferToSCFOptions opts;
                // Keep default targetRank=1: only lower rank>1 transfers to loops
                // of rank-1 transfers. Rank-1 transfers (e.g. vector<1xf32> on
                // SIMT matvec path) are handled by ConvertVectorToLLVMPass (38d2).
                // targetRank=0 would scalarize MMA tile loads, breaking the tensor
                // core path for large matrices.
                populateVectorToSCFConversionPatterns(patterns, opts);
                if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
                    signalPassFailure();
            }
            StringRef getArgument() const override {
                return "nova-vector-to-scf-gpu-func";
            }
        };
        return std::make_unique<VectorToSCFOnGPUFuncPass>();
    }());

    // 38d2. ConvertVectorToLLVM on GPU kernels — BEFORE ConvertGpuOpsToNVVMOps.
    //
    // CRITICAL: The funcPm.nest<gpu::GPUFuncOp>() block below (38j-n) is
    // effectively dead code — ConvertGpuOpsToNVVMOps (38h) converts gpu.func →
    // llvm.func, so no gpu::GPUFuncOp ops remain when that block runs.
    // vector.contract (e.g. vector<4xf32>,vector<4x1xf32>→vector<1xf32> on
    // matvec SIMT path) must be lowered to LLVM scalar FMAs HERE, while
    // memrefs are still in their original form.
      gpuPm.addPass(createSCFToControlFlowPass());                           // SCF first
      gpuPm.addPass(createConvertNVGPUToNVVMPass());                         // nvgpu → nvvm FIRST
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createConvertVectorToLLVMPass());  // vector → LLVM AFTER nvgpu lowered
      // 38h. gpu.* → NVVM intrinsics.
      // NOTE: ConvertGpuOpsToNVVMOps MUST run before ConvertVectorToLLVMPass.
      //   ConvertGpuOpsToNVVMOps converts gpu.func → llvm.func, rewriting all
      //   `index`-typed function arguments to i64 via its own TypeConverter.
      //   If ConvertVectorToLLVMPass ran first (while the function is still
      //   gpu.func with index args), the VectorToLLVM converter would insert
      //   lazy `builtin.unrealized_conversion_cast index → i64` ops to bridge
      //   the type gap. Those casts are then NOT eliminated by the subsequent
      //   ConvertGpuOpsToNVVMOps pass (which uses a separate type converter
      //   chain), leaving unreconciled casts that crash LLVM translation.
      ConvertGpuOpsToNVVMOpsOptions nvvmOpts;
      gpuPm.addPass(createConvertGpuOpsToNVVMOps(nvvmOpts));
      // 38h-post. Lower any affine.apply ops that survived into llvm.func bodies.
      //   The 38c createLowerAffinePass() is nested in gpu::GPUFuncOp; after
      //   ConvertGpuOpsToNVVMOps the kernel body becomes an llvm.func, so any
      //   residual affine.apply ops are no longer reachable by the GPUFuncOp-
      //   scoped pass. Run LowerAffine at module level here to catch them all.
      gpuPm.addPass(createLowerAffinePass());
      // 38i-n. LLVM type lowering.
      gpuPm.addPass(createConvertIndexToLLVMPass()); // eliminates index↔i64 casts
      gpuPm.addPass(createArithToLLVMConversionPass());
      gpuPm.addPass(createConvertMathToLLVMPass());
      gpuPm.addPass(createConvertVectorToLLVMPass()); // all index→i64 already done
      gpuPm.addPass(createReconcileUnrealizedCastsPass());
      gpuPm.addPass(createReconcileUnrealizedCastsPass()); // catch chains
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // 13.3 — Compile the gpu.module blobs to a PTX ISA binary embedded in IR.
    GpuModuleToBinaryPassOptions binaryOptions;
    binaryOptions.toolkitPath = "/usr/local/cuda-13.0";
    binaryOptions.compilationTarget = "isa";
    pm.addPass(createGpuModuleToBinaryPass(binaryOptions));

    // 13.4 — Convert any remaining #gpu.address_space<private> on host-side
    //         memrefs to generic address space 0 BEFORE gpu-to-llvm, so that
    //         gpu.launch_func args don't carry symbolic address spaces that
    //         the LLVM lowering cannot convert.
    pm.addPass(createNovaGPULowerMemorySpacePass());

    // 13.4b — Lower gpu.* host ops (gpu.alloc, gpu.launch_func, etc.) to LLVM
    //          runtime calls (mgpuMemAlloc, mgpuLaunchKernel, etc.).
    GpuToLLVMConversionPassOptions hostOpts;
    pm.addPass(createGpuToLLVMConversionPass(hostOpts));
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // 13.5 — Lower remaining host dialects to LLVM.
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createConvertVectorToLLVMPass()); 
    pm.addPass(mlir::nova::createGenerateDynamicWrapperPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // ---------------------------------------------------------------------------
  // Pass Registration
  // ---------------------------------------------------------------------------

  void registerNovaLLVMGPUPasses()
  {
    registerNovaConfigTrackingCanonicalizerPass();
    registerNovaGPUSelectLoweringStrategyPass();
    registerNovaTileAndDistributePass();
    registerNovaGPUPadOperandsPass();
    registerNovaGPUPromoteMatmulOperandsPass();
    registerNovaGPUApplyTilingLevelReductionPass();
    registerNovaGPUApplyTilingLevelThreadPass();
    registerNovaGPUApplyTilingLevelSubgroupPass();
    registerNovaGPUFuseAndHoistParallelLoopsPass();
    registerNovaGPUGreedilyDistributeToThreadsPass();
    registerNovaGPULowerBarrierRegionPass();
    registerNovaGPUEraseFusionBarriersPass();
    registerNovaGPUInferMemorySpacePass();
    registerNovaGPUComprehensiveBufferizePass();
    registerNovaGPUInsertWorkgroupBarriersPass();
    registerNovaNormalizeLoopBoundsPass();
    registerNovaEliminateEmptyTensorsPass();
    registerNovaConvertSharedMemAllocsPass();
    registerNovaGPUMapForallToGPUPass();
    registerNovaGPULowerMemorySpacePass();
    registerNovaWarpShuffleReductionPass();
    registerNovaGPUFillCopyForwardingPass();
    registerNovaGPUCoalesceWorkgroupBuffersPass();

    // Register the full optimized pipeline as a named pipeline so it can be
    // invoked from mlir-opt with --nova-gpu-optimized-pipeline.


    // nova-scf-loop-unroll: wraps mlir::scf::loopUnrollByFactor.
    // Usage from nova-opt: --nova-scf-loop-unroll
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return mlir::nova::createNovaScfLoopUnrollPass();
    });

    // scf-for-loop-specialization: already registered by registerAllPasses()
    // in nova-opt.cpp, but registered here explicitly so it appears in the
    // Nova pass list when --help is run against the pipeline.
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return mlir::createForLoopSpecializationPass();
    });

    // scf-for-loop-peeling: peels the last partial iteration off each scf.for.
    // Usage from nova-opt: --scf-for-loop-peeling
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return mlir::createForLoopPeelingPass();
    });


    PassPipelineRegistration<>(
        "nova-gpu-optimized-pipeline",
        "Nova GPU Optimized Pipeline (tile+fuse → normalize → bufferize → "
        "gpu.launch → linalg-to-loops)",
        [](OpPassManager &pm)
        { addNovaGPUOptimizedPipeline(pm, ""); });
  }

  /*
  --------------------------------------------------
  CORE LOGIC — addNovaGPUBufferizePasses

  GPU-aware bufferization helper, mirroring IREE's addGPUBufferizePasses().

  Three ordered steps:
    1. NovaEliminateEmptyTensorsPass    — eliminate tensor.empty ops by
       finding existing destination tensors (reduces allocations).
    2. createEmptyTensorToAllocTensorPass — convert remaining tensor.empty
       → bufferization.alloc_tensor for OneShotBufferize.
    3. NovaGPUInferMemorySpacePass      — tag each alloc_tensor as
       workgroup (shared SRAM) or private (register/stack) based on
       usage pattern. Must run before bufferization so the allocation
       function receives the correct MemRefType.
    4. NovaGPUComprehensiveBufferizePass — erase nova.fusion_barrier ops
       then run OneShotBufferize with GPU-aware alloc/copy functions.
    5. Post-bufferization cleanup: resolve shaped-type result dims, then
       canonicalize + CSE.
    6. BufferLoopHoistingPass — hoist memref.alloc out of loops where
       possible (moves shared memory allocations out of the K-reduction
       loop so they are reused across iterations).
    7. BufferDeallocationPipeline — insert memref.dealloc for all allocs.
  --------------------------------------------------
  */

  // CORE LOGIC
  void addNovaGPUBufferizePasses(OpPassManager &pm)
  {
    // Pre-bufferization passes (per-function).
    pm.addNestedPass<func::FuncOp>(createNovaEliminateEmptyTensorsPass());
    pm.addNestedPass<func::FuncOp>(
        bufferization::createEmptyTensorToAllocTensorPass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUInferMemorySpacePass());

    // GPU-aware comprehensive bufferize (module-level).
    // Erases nova.fusion_barrier, converts function boundaries with identity
    // layout map, and runs OneShotBufferize with GPU alloc/copy functions.
    pm.addPass(createNovaGPUComprehensiveBufferizePass());

    // Post-bufferization cleanup (per-function).
    pm.addNestedPass<func::FuncOp>(
        memref::createResolveShapedTypeResultDimsPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());

    // NOTE: BufferLoopHoistingPass is DISABLED.
    //
    // This pass hoists memref.alloc out of loops to reuse buffers across
    // iterations. However, it treats scf.forall (parallel) the same as
    // scf.for (sequential). Hoisting an alloc out of a parallel forall
    // collapses N per-thread buffers into one shared buffer, creating a
    // race condition between threads. The per-thread K-accumulator
    // (memref<1x1x16xf32>) gets hoisted from the thread forall to
    // function scope, causing all 256 threads to share one 64-byte buffer.
    //
    // TODO: Implement a custom hoisting pass that only hoists from
    // sequential loops (scf.for) and not from parallel loops (scf.forall).
    // pm.addNestedPass<func::FuncOp>(
    //     bufferization::createBufferLoopHoistingPass());

    // Insert memref.dealloc for all memref.alloc ops.
    bufferization::BufferDeallocationPipelineOptions deallocOpts;
    bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);
  }

} // namespace mlir::nova
