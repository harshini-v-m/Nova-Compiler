add_mlir_library(NovaLLVMGPUTransforms
  # Top-level pipeline
  Passes.cpp

  # ── Config ────────────────────────────────────────────────────────────────
  # NVIDIA target description table
  Config/NVIDIATargetUtils.cpp
  # LoweringConfig read/write utilities
  Config/NovaGPULoweringConfigUtils.cpp
  # Kernel config heuristics
  Config/NovaKernelConfig.cpp
  # Strategy selection pass
  Config/NovaGPUSelectLoweringStrategy.cpp
  # Config-propagating canonicalizer
  Config/NovaConfigTrackingCanonicalizer.cpp
  # Configure MMA tensor layouts
  Config/NovaGPUConfigureTensorLayouts.cpp

  # ── Tiling ────────────────────────────────────────────────────────────────
  # Tile-and-fuse shared utilities
  Tiling/NovaGPUTileAndFuseUtils.cpp
  # Workgroup tiling via scf.forall
  Tiling/NovaGPUTileDispatchUsingForall.cpp
  # Reduction (K-dim) tiling inside forall
  Tiling/NovaGPUApplyTilingLevelReduction.cpp
  # Pad linalg operands to tile-size multiples
  Tiling/NovaGPUPadOperands.cpp
  # Promote matmul A/B operands to shared memory
  Tiling/NovaGPUPromoteMatmulOperands.cpp

  # ── Fusion ────────────────────────────────────────────────────────────────
  # Fuse and hoist parallel scf.forall loops
  Fusion/NovaGPUFuseAndHoistParallelLoops.cpp
  # Distribute orphaned parallel ops to GPU threads
  Fusion/NovaGPUGreedilyDistributeToThreads.cpp
  # Rematerialization / reduction-into-producer fusion
  Fusion/NovaFuseReductionIntoProducer.cpp
  # Element-wise fusion (prevents unwanted matmul fusion)
  Fusion/NovaElementwiseOpFusionPass.cpp
  # Parallel fusion for ops with identical inputs
  Fusion/NovaCheckInsParallelFuse.cpp
  # Horizontal linalg fusion for reduction cases
  Fusion/NovaLinalgFusion.cpp
  # Vertical fusion: reduction producer + parallel consumer
  Fusion/NovaVerticalFusion.cpp

  Fusion/NovaMultiConsumerFusion.cpp

  # ── Memory ────────────────────────────────────────────────────────────────
  # GPU-aware comprehensive bufferization
  Memory/NovaGPUBufferize.cpp
  # Infer shared vs. private memory spaces
  Memory/NovaGPUInferMemorySpace.cpp
  # Convert workgroup alloc → memref.global (shared mem)
  Memory/NovaConvertSharedMemAllocs.cpp
  # Coalesce workgroup allocs with non-overlapping live ranges
  Memory/NovaGPUCoalesceWorkgroupBuffers.cpp
  # Forward fill/copy operations
  Memory/NovaGPUFillCopyForwarding.cpp
  # Eliminate tensor.empty ops pre-bufferization
  Memory/NovaEliminateEmptyTensors.cpp
  Memory/NovaRepositionStore.cpp
  Memory/NovaGPUApplySwizzle.cpp
  Memory/NovaGPUReduceBankConflicts.cpp
  # ── Loops ─────────────────────────────────────────────────────────────────
  # Unroll scf loops
  Loops/NovaScfLoopUnroll.cpp
  # Vectorize scf loops
  Loops/NovaScfLoopVectorize.cpp
  # Loop splitting transformation
  Loops/LoopSplit.cpp
  # Scalarize accumulator across SCF loops
  Loops/SCFScalarizeAccumulator.cpp
  # Normalize scf.forall bounds to lb=0, step=1
  Loops/NovaNormalizeLoopBounds.cpp
  # Warp-level shuffle reduction
  Loops/NovaWarpShuffleReduction.cpp
  # Stride-32 parallelization for single-thread reductions
  Loops/NovaStrideReduction.cpp

  # ── Lowering ──────────────────────────────────────────────────────────────
  # Convert scf.forall → gpu.launch with dynamic block dims
  Lowering/NovaGPUMapForallToGPU.cpp
  # Lower nova.barrier_region → two nova.value_barrier ops
  Lowering/NovaGPULowerBarrierRegion.cpp
  # Erase nova.fusion_barrier ops
  Lowering/NovaGPUEraseFusionBarriers.cpp

  # ── Vectorization ──────────────────────────────────────────────────────────
  # Normalize MMA load/store memref indices for NVVM legalization
  Vectorization/NovaGPUCastTypeToFitMMA.cpp
  # OptimizeVectorTransfer, DropVectorUnitDims, HoistVectorExtractInsertSlice
  Vectorization/NovaGPUVectorCleanup.cpp
  # linalg → vector.contract / vector.transfer (SIMD vectorization)
  Vectorization/NovaGPUGenericVectorization.cpp
  
  Vectorization/NovaGPUHoistExtractInsert.cpp
  # Vector size inference for dynamic shapes (ValueBounds UB analysis)
  Vectorization/NovaVectorSizeUtils.cpp
  # VectorizeMemrefCopy, UnrollToIntrinsics, LowerPackOps
  Vectorization/NovaGPUVectorization.cpp
  
  Vectorization/NovaGPUVectorAlloc.cpp
  Vectorization/NovaGPUCombineValueSemanticBarriers.cpp
  Vectorization/NovaGPUVectorDistribution.cpp

  DEPENDS
  MLIRNovaOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRVectorDialect
  MLIRGPUDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRSCFTransforms
  MLIRVectorTransforms
  MLIRLinalgTransforms
  MLIRGPUTransforms
  MLIRMemRefTransforms
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRTosaDialect
  MLIROpenMPDialect
  MLIRAnalysis
  MLIRValueBoundsOpInterface
  MLIRTensorTransforms
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRLinalgTransformOps
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  MLIRVectorToSCF
  MLIRVectorToGPU
  MLIRNVGPUDialect
  MLIRNVVMDialect
  MLIRNVGPUToNVVM
  
  # IREE Dependencies (Using existing Nova linkage/vendored paths if available, otherwise adapting)
  # For now, assuming we implement simplified versions or link against IREE libraries if setup.
  # Given previous context, we likely reimplement simplified passes.
)

target_include_directories(NovaLLVMGPUTransforms
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_compile_definitions(NOVA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
