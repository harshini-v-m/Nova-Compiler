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
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
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
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "Compiler/Transforms/FixGpuLaunch.h"

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

  // ---------------------------------------------------------------------------
  // Vectorization Pipeline Builders
  // ---------------------------------------------------------------------------

  /// Path A: CUDA Core (Generic Vectorization)
  void addNovaGPUVectorizationPasses(OpPassManager &pm) {
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUGenericVectorizationPass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUSubsetHoistingPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  /// Path B: Tensor Core (MMA Vectorization)
  void addNovaGPUTensorCoreVectorizationPasses(OpPassManager &pm) {
    // 1. Pack subgroup-level tiles into hardware MMA shapes (e.g. 16x16x16)
    // pm.addNestedPass<func::FuncOp>(createNovaGPUPackToIntrinsicsPass());
    
    // 2. Vectorize using generic vectorization
    pm.addNestedPass<func::FuncOp>(createNovaGPUGenericVectorizationPass());
    
    // 3. Distribute MMA vectors across threads (SIMD-to-SIMT)
    // pm.addNestedPass<func::FuncOp>(createNovaGPUVectorDistributePass());
    
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // CORE LOGIC
  void addNovaGPUOptimizedPipeline(OpPassManager &pm,
                                   StringRef cudaArch)
  {
    // Default to sm_86 (RTX 3060 / Ampere) when no architecture is specified.
    StringRef arch = cudaArch.empty() ? "sm_86" : cudaArch;

    // ---- Nova dialect → Arith / Tosa / Linalg lowering ----
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::nova::createNovaToArithLoweringPass());
    pm.addPass(mlir::nova::createNovaToTosaLoweringPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createNovaElementwiseToLinalgPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createNovaToLinalgPass());

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
    pm.addPass(createLinalgElementwiseOpFusionPass());
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
    // Step 5: Tile thread-level M/N dimensions
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUApplyTilingLevelThreadPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Step 6 (Subgroup tiling) is disabled — warp-level tiling is not yet
    // tuned for all shapes.  Re-enable when subgroup config is stable.
    //   pm.addNestedPass<func::FuncOp>(createNovaGPUApplyTilingLevelSubgroupPass());
    //   pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    //   pm.addPass(createCSEPass());

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
    // Step 7.75: Post-tiling Linalg cleanup — runs AFTER all tiling/fusion
    // walks are complete and BEFORE bufferization (still tensor form).
    //
    //  GeneralizeNamedOps:    linalg.matmul → linalg.generic for uniform
    //                         bufferization and vectorization treatment.
    //  ElementwiseOpFusion:   fuses any elementwise ops not consumed as
    //                         epilogues during workgroup tiling.
    //
    // IMPORTANT: A second FoldUnitExtentDims was here but was removed.
    // It breaks batch matmul (3D+) by folding unit dims inside scf.if
    // branches independently, creating type mismatches between the
    // static-padded and dynamic-sliced branches.  The pre-tiling
    // FoldUnitExtentDims (Step -0.5) handles the critical softmax case.
    // -------------------------------------------------------------------------
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(createLinalgElementwiseOpFusionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 7.8: Vectorization (Dual-Path)
    //
    // For matmuls on f16/i8, uses Tensor Core (MMA) path.
    // Otherwise falls back to Generic (CUDA Core) vectorization.
    // -------------------------------------------------------------------------
    if (arch.starts_with("sm_") && arch.substr(3).compare("80") >= 0) {
      addNovaGPUTensorCoreVectorizationPasses(pm);
    } else {
      addNovaGPUVectorizationPasses(pm);
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
    // Step 8.1: Vectorize Memref Copies
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

    // -------------------------------------------------------------------------
    // Step 11: Insert gpu.barrier at workgroup memory write→read transitions
    //
    // Must run AFTER linalg-to-loops (so scalar store/load ops are visible)
    // and AFTER forall→gpu.launch and bufferization.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUInsertWorkgroupBarriersPass());

    // -------------------------------------------------------------------------
    // Step 12: Host↔Device Memory Conversion and Vector Lowering
    //
    // Run vector lowerings and finalize-memref-to-llvm on the host side
    // so they can handle the logic inside @gpu.launch regions before
    // they are outlined into @gpu.module kernels. This avoids
    // PassManager restriction errors on @gpu.func.
    // -------------------------------------------------------------------------
    mlir::VectorTransferToSCFOptions xferOpts;
    xferOpts.setTargetRank(0); // Scalarize all transfers for GPU
    xferOpts.enableLowerTensors();
    
    pm.addNestedPass<func::FuncOp>(mlir::createConvertVectorToSCFPass(xferOpts));
    pm.addNestedPass<func::FuncOp>(mlir::vector::createLowerVectorMultiReductionPass());
    
    pm.addPass(nova::createConvertMemRefToGpuPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 12.5: Outline gpu.launch bodies into gpu.module kernels
    // -------------------------------------------------------------------------
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(mlir::nova::createRenameGpuKernelsPass());
    pm.addPass(createCanonicalizerPass());

    // -------------------------------------------------------------------------
    // Step 12.75: Convert workgroup memref.alloc → memref.global
    // -------------------------------------------------------------------------
    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      gpuPm.addPass(createNovaConvertSharedMemAllocsPass());
      gpuPm.addPass(createCanonicalizerPass());
    }

    // -------------------------------------------------------------------------
    // Step 13: Full CUDA/NVVM LLVM lowering
    // -------------------------------------------------------------------------
    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip = arch.str();
    nvvmTargetOptions.features = "+ptx76";
    nvvmTargetOptions.optLevel = 3;
    nvvmTargetOptions.fastFlag = true;
    nvvmTargetOptions.ftzFlag = true;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      gpuPm.addPass(memref::createExpandStridedMetadataPass());
      
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createLowerAffinePass());
      gpuPm.addPass(createNovaGPULowerMemorySpacePass());
      
      // Full LLVM lowering inside the GPU module
      gpuPm.addPass(createSCFToControlFlowPass());
      gpuPm.addPass(createConvertIndexToLLVMPass());
      gpuPm.addPass(createArithToLLVMConversionPass());
      gpuPm.addPass(memref::createExpandStridedMetadataPass());
      
      gpuPm.addPass(mlir::createConvertVectorToLLVMPass());
      
      ConvertGpuOpsToNVVMOpsOptions nvvmOpts;
      gpuPm.addPass(createConvertGpuOpsToNVVMOps(nvvmOpts));
      gpuPm.addPass(createConvertMathToLLVMPass());
      gpuPm.addPass(createUBToLLVMConversionPass());
      gpuPm.addPass(createReconcileUnrealizedCastsPass());
      
      // Promote __global_memory__ globals from device global (AS 0) to
      // shared memory (AS 3). Must run after full LLVM lowering.
      gpuPm.addPass(createNovaGPUPromoteGlobalsToSharedPass());
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
    {
      auto &funcPm = pm.nest<func::FuncOp>();
      funcPm.addPass(createSCFToControlFlowPass());
      funcPm.addPass(createArithToLLVMConversionPass());
      funcPm.addPass(memref::createExpandStridedMetadataPass());
    }
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
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
    registerNovaGPUFillCopyForwardingPass();
    registerNovaGPUCoalesceWorkgroupBuffersPass();

    // Vectorization
    registerNovaGPUGenericVectorizationPass();
    registerNovaGPUSubsetHoistingPass();
    registerNovaGPUVectorizeMemrefCopyPass();
    registerNovaGPUUnrollToIntrinsicsPass();

    // Register the full optimized pipeline as a named pipeline so it can be
    // invoked from mlir-opt with --nova-gpu-optimized-pipeline.
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
