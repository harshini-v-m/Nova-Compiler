add_mlir_library(NovaLLVMGPUTransforms
  Passes.cpp
  NovaGPUTileAndFuseUtils.cpp
  NovaGPUTileDispatchUsingForall.cpp
  NovaGPUPadOperands.cpp
  NovaGPUPromoteMatmulOperands.cpp
  SCFScalarizeAccumulator.cpp
  NovaGPUApplyTilingLevelReduction.cpp
  # GAP 2: Erase fusion barriers before bufferization
  NovaGPUEraseFusionBarriers.cpp
  # GAP 3: Infer GPU memory spaces before bufferization
  NovaGPUInferMemorySpace.cpp
  # GAP 3: Comprehensive GPU-aware bufferize pass (mirrors IREEComprehensiveBufferizePass)
  NovaGPUBufferize.cpp
  # Pre-bufferize: eliminate tensor.empty ops (mirrors IREE's EliminateEmptyTensorsPass)
  NovaEliminateEmptyTensors.cpp
  # Config propagation (mirrors IREE's ConfigTrackingCanonicalizerPass)
  NovaConfigTrackingCanonicalizer.cpp
  # Phase 2: LoweringConfig read/write utilities
  NovaGPULoweringConfigUtils.cpp
  # Phase 3: NVIDIA target description table
  NVIDIATargetUtils.cpp
  # Phase 4: Kernel config heuristic
  NovaKernelConfig.cpp
  # Phase 5: Strategy pass
  NovaGPUSelectLoweringStrategy.cpp
  # Phase 6: Fuse and hoist parallel loops (mirrors IREE GPUFuseAndHoistParallelLoops)
  NovaGPUFuseAndHoistParallelLoops.cpp
  # Normalize scf.forall bounds to lb=0, step=1 (mirrors IREE NormalizeLoopBounds)
  NovaNormalizeLoopBounds.cpp
  # Convert workgroup memref.alloc → memref.global + drop deallocs (mirrors IREE)
  NovaConvertSharedMemAllocs.cpp
  # Convert scf.forall → gpu.launch with dynamic block dims (replaces transform script)
  NovaGPUMapForallToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

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
  MLIRTensorTransforms
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRLinalgTransformOps
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  
  # IREE Dependencies (Using existing Nova linkage/vendored paths if available, otherwise adapting)
  # For now, assuming we implement simplified versions or link against IREE libraries if setup.
  # Given previous context, we likely reimplement simplified passes.
)

add_compile_definitions(NOVA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
