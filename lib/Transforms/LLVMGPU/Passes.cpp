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
// Transform dialect includes no longer needed (replaced by NovaGPUMapForallToGPU C++ pass)
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

using namespace mlir;

namespace mlir::nova
{

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
                                   StringRef cudaArch)
  {
    // Default to sm_86 (RTX 3060 / Ampere) when no architecture is specified.
    StringRef arch = cudaArch.empty() ? "sm_86" : cudaArch;
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
    // Chains like exp→exp2→log→log2→log10 must become a single linalg.generic
    // BEFORE the tiling pipeline stamps per-op configs and creates per-op
    // scf.forall loops.  Without this, each elementwise op gets its own
    // thread-mapped forall and the FuseAndHoist pass may not converge
    // (non-deterministic pattern ordering causes intermittent stalls).
    // IREE performs the same fusion before its tile-and-fuse pipeline.
    // -------------------------------------------------------------------------
    pm.addPass(createLinalgElementwiseOpFusionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step -0.5: Fold unit-extent dims before tiling
    // Softmax lowering produces keepdims reductions (e.g. 4x1x8 → 4x1) that
    // insert tensor.expand_shape between 2D elementwise and 3D reduction ops.
    // This rank mismatch prevents the tiling pass from fusing all ops into a
    // single GPU kernel.  Folding unit dims here collapses the reductions to
    // 2D (matching the elementwise ops), enabling uniform tiling and fusion.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 0: Select lowering strategy
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
    // After K-tiling, the matmul operates on [wgM × kStep] and [kStep × wgN]
    // slices. Promotion in Step 4 then allocates only K-tile-sized shared mem.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelReductionPass());
    pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 4: Promote matmul A/B operands to shared memory  [MOVED AFTER K-TILE]
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

    // -------------------------------------------------------------------------
    // Step 6: Tile subgroup (warp) M/N dimensions
    // -------------------------------------------------------------------------
    //   pm.addNestedPass<func::FuncOp>(createNovaGPUApplyTilingLevelSubgroupPass());
    //   pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
    //   pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 7: Fuse and hoist parallel loops
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUFuseAndHoistParallelLoopsPass());

    // -------------------------------------------------------------------------
    // Step 7.5: Normalize forall loop bounds
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 7.75: Post-tiling Linalg cleanup (tensor level)
    // Runs AFTER all tiling/fusion walks are complete (no more isContractionOp
    // or ConfigTracking walks) and BEFORE bufferization (still tensor form).
    //
    // * GeneralizeNamedOps: linalg.matmul → linalg.generic for uniform
    //   bufferization and vectorization treatment.
    // * FoldUnitExtentDims: collapses unit dims left behind by thread tiling
    //   (e.g. [1×128] → [128]) to reduce buffer sizes.
    // * ElementwiseOpFusion: fuses any elementwise ops that were NOT consumed
    //   as epilogues during Step 1 workgroup tiling (rare, but possible when
    //   a reshape separates producer and consumer).
    // -------------------------------------------------------------------------
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    // NOTE: A second FoldUnitExtentDims was here to collapse unit dims from
    // thread tiling (e.g. [1x128] → [128]). Removed because it breaks batch
    // matmul (3D+): the pass folds unit dims inside scf.if branches
    // independently, creating type mismatches between static-padded and
    // dynamic-sliced branches. The first FoldUnitExtentDims (before tiling)
    // handles the critical case (softmax keepdims fusion).
    pm.addPass(createLinalgElementwiseOpFusionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 8: GPU-aware bufferization
    // -------------------------------------------------------------------------
    addNovaGPUBufferizePasses(pm);

    // -------------------------------------------------------------------------
    // Step 8.5: Eliminate degenerate single-iteration foralls
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
    // Replaces the transform script (gpu_forall_to_launch.mlir) which
    // hardcoded block_dims = [32, 32, 1]. This pass computes block dims
    // from the actual thread-mapped forall iteration bounds.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUMapForallToGPUPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 10: Lower remaining linalg → scf loops, affine → arith
    // -------------------------------------------------------------------------
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // -------------------------------------------------------------------------
    // Step 11: Insert gpu.barrier at workgroup memory write→read transitions
    // Must run AFTER linalg-to-loops (so scalar store/load ops are visible)
    // and AFTER forall→gpu.launch and bufferization.
    // -------------------------------------------------------------------------
    pm.addNestedPass<func::FuncOp>(createNovaGPUInsertWorkgroupBarriersPass());

    // -------------------------------------------------------------------------
    // Step 12: Outline gpu.launch bodies into gpu.module kernels.
    // This must happen BEFORE the shared-mem conversion so that the
    // memref.global + memref.get_global for workgroup memory are created
    // INSIDE the gpu.module (matching IREE's ConvertSharedMemAllocOp flow)
    // rather than in the outer host module where finalize-memref-to-llvm
    // cannot lower #gpu.address_space<workgroup> types.
    // -------------------------------------------------------------------------
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(createCanonicalizerPass());

    // -------------------------------------------------------------------------
    // Step 12.5: Inside the gpu.module, convert workgroup memref.alloc ops
    // to memref.global + memref.get_global and drop workgroup memref.dealloc.
    // Ported from IREE's ConvertSharedMemAllocOp + DropSharedMemoryDeallocOp
    // (ConvertToLLVM.cpp:152-203 / GPUPatterns.cpp:211-223).
    // Running inside gpu.module ensures the globals are scoped to the kernel
    // and never appear in the outer host module.
    // -------------------------------------------------------------------------
    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      gpuPm.addPass(createNovaConvertSharedMemAllocsPass());
      gpuPm.addPass(createCanonicalizerPass());
    }

    // -------------------------------------------------------------------------
    // Step 12.75: Convert host-side cross-kernel allocations to GPU allocations.
    // This handles intermediate buffers (e.g. `%alloc = memref.alloc()`) on
    // the host side that are passed into GPU kernels, converting them from
    // default memory space `memref.alloc` to `gpu.alloc` (which lowers to
    // `cudaMalloc`).
    // -------------------------------------------------------------------------
    pm.addPass(nova::createConvertMemRefToGpuPass());
    pm.addPass(createCanonicalizerPass());

    // -------------------------------------------------------------------------
    // Step 13: Full CUDA/NVVM LLVM lowering
    // Lowers gpu.module → PTX binary, then lowers host code → LLVM IR.
    // Mirrors IREE's addLowerToLLVMGPUPasses (LLVMGPU/Passes.cpp).
    // -------------------------------------------------------------------------

    // 13.0 — Attach an NVVM target descriptor to each gpu.module so that
    // createGpuModuleToBinaryPass knows how to compile it to PTX.
    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip = arch.str();
    nvvmTargetOptions.features = "+ptx76";  // PTX 7.6 for sm_86 (Ampere)
    nvvmTargetOptions.optLevel = 3;
    nvvmTargetOptions.fastFlag = true;
    nvvmTargetOptions.ftzFlag = true;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

    // 13.1 — Async gpu region: wrap gpu.launch_func in async token chains
    //         (required by GpuModuleToBinaryPass).
    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

    // 13.2 — Lower the contents of each gpu.module to NVVM / LLVM.
    //
    // Pass ordering is critical here:
    //   1. expand-strided-metadata: decompose memref.subview →
    //      memref.extract_strided_metadata + affine.apply + reinterpret_cast.
    //      Must run while operands are still memref types (before gpu-to-nvvm).
    //   2. lower-affine: convert affine.apply → arith ops.
    //   3. finalize-memref-to-llvm: convert all memref ops to LLVM structs.
    //   4. scf-to-cf: lower scf control flow.
    //   5. gpu-to-nvvm: convert gpu.func signature + gpu ops to NVVM.
    //   6. remaining dialect conversions (index, arith, math).
    //   7. reconcile-unrealized-casts: erase conversion cast chains.
    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      // Phase 1: Decompose and lower memrefs while types are still memrefs.
      gpuPm.addPass(memref::createExpandStridedMetadataPass());
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createLowerAffinePass());
      // Convert #gpu.address_space<private/workgroup/global> to NVVM integer
      // address spaces (5/3/1) before finalizeMemRefToLLVM, which requires
      // integer address spaces for LLVM type conversion.
      gpuPm.addPass(createNovaGPULowerMemorySpacePass());
      // gpuPm.addPass(createFinalizeMemRefToLLVMConversionPass());
      // Phase 2: Lower control flow and GPU ops.
      gpuPm.addPass(createSCFToControlFlowPass());
      ConvertGpuOpsToNVVMOpsOptions nvvmOpts;
      gpuPm.addPass(createConvertGpuOpsToNVVMOps(nvvmOpts));
      // Phase 3: Lower remaining dialects to LLVM.
      gpuPm.addPass(createConvertIndexToLLVMPass());
      gpuPm.addPass(createArithToLLVMConversionPass());
      gpuPm.addPass(createConvertMathToLLVMPass());
      gpuPm.addPass(createReconcileUnrealizedCastsPass());
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // 13.3 — Compile the gpu.module blobs to a PTX ISA binary embedded in the IR.
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
    pm.addPass(mlir::nova::createGenerateDynamicWrapperPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // --- Pass Registration ---

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

    // Register the full optimized pipeline as a named pipeline.
    PassPipelineRegistration<>(
        "nova-gpu-optimized-pipeline",
        "Nova GPU Optimized Pipeline (tile+fuse → normalize → bufferize → "
        "gpu.launch → linalg-to-loops)",
        [](OpPassManager &pm)
        { addNovaGPUOptimizedPipeline(pm, ""); });
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
  void addNovaGPUBufferizePasses(OpPassManager &pm)
  {
    // Pre-bufferize passes (per-function).
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
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

    // Hoist memref.alloc ops out of loops where possible. This moves shared
    // memory allocations out of the K-reduction loop so they are reused
    // across iterations instead of being allocated/deallocated each step.
    pm.addNestedPass<func::FuncOp>(
        bufferization::createBufferLoopHoistingPass());

    // Insert memref.dealloc for all memref.alloc ops.
    bufferization::BufferDeallocationPipelineOptions deallocOpts;
    bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);
  }

} // namespace mlir::nova
