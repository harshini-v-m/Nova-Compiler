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
  Step 3    : ApplyTilingLevelReduction → K-dim → scf.for [MOVED BEFORE PROMOTE]
  Step 4    : PromoteMatmulOperands → global→shared copy + barrier
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
    // Step 2: Pad operands
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUPadOperandsPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 3: Tile reduction (K) dimension   [MOVED BEFORE PROMOTION]
    //
    // After K-tiling, the matmul operates on [wgM × kStep] and [kStep × wgN]
    // slices. Promotion in Step 4 then allocates only K-tile-sized shared mem
    // instead of the full global operand size.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelReductionPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 4: Promote matmul A/B operands to shared memory  [MOVED AFTER K-TILE]
    //
    // Now inside the K-loop, so shared memory holds only one K-tile at a time.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUPromoteMatmulOperandsPass());
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
    // Step 6: Fuse and hoist parallel loops
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUFuseAndHoistParallelLoopsPass());

    // -------------------------------------------------------------------------
    // Step 7: Normalize forall loop bounds (lb=0, step=1)
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

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
    // Step 8: GPU-aware bufferization (tensor → memref)
    // -------------------------------------------------------------------------
    addNovaGPUBufferizePasses(pm);

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
    //   features=+ptx76 → PTX 7.6 instruction set for Ampere (sm_86)
    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple    = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip      = arch.str();
    nvvmTargetOptions.features  = "+ptx76";
    nvvmTargetOptions.optLevel  = 3;
    nvvmTargetOptions.fastFlag  = true;
    nvvmTargetOptions.ftzFlag   = true;
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
      gpuPm.addPass(memref::createExpandStridedMetadataPass());
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createLowerAffinePass());
      gpuPm.addPass(createNovaGPULowerMemorySpacePass());
      gpuPm.addPass(createSCFToControlFlowPass());
      ConvertGpuOpsToNVVMOpsOptions nvvmOpts;
      gpuPm.addPass(createConvertGpuOpsToNVVMOps(nvvmOpts));
      gpuPm.addPass(createConvertIndexToLLVMPass());
      gpuPm.addPass(createArithToLLVMConversionPass());
      gpuPm.addPass(createConvertMathToLLVMPass());
      gpuPm.addPass(createReconcileUnrealizedCastsPass());
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // 13.3 — Compile the gpu.module blobs to a PTX ISA binary embedded in IR.
    GpuModuleToBinaryPassOptions binaryOptions;
    binaryOptions.toolkitPath        = "/usr/local/cuda-13.0";
    binaryOptions.compilationTarget  = "isa";
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
    registerNovaGPUEraseFusionBarriersPass();
    registerNovaGPUInferMemorySpacePass();
    registerNovaGPUComprehensiveBufferizePass();
    registerNovaGPUInsertWorkgroupBarriersPass();
    registerNovaNormalizeLoopBoundsPass();
    registerNovaEliminateEmptyTensorsPass();
    registerNovaConvertSharedMemAllocsPass();
    registerNovaGPUMapForallToGPUPass();
    registerNovaGPULowerMemorySpacePass();

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

    // Hoist memref.alloc ops out of loops where possible.
    // This moves shared memory allocations out of the K-reduction loop so
    // they are reused across iterations instead of being re-allocated each
    // step.
    pm.addNestedPass<func::FuncOp>(
        bufferization::createBufferLoopHoistingPass());

    // Insert memref.dealloc for all memref.alloc ops.
    bufferization::BufferDeallocationPipelineOptions deallocOpts;
    bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);
  }

} // namespace mlir::nova
