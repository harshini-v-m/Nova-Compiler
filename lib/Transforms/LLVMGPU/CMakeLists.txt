add_mlir_library(NovaLLVMGPUTransforms
  Passes.cpp
  NovaGPUTileAndFuseUtils.cpp
  NovaGPUTileDispatchUsingForall.cpp
  NovaGPUPadOperands.cpp
  NovaGPUPromoteMatmulOperands.cpp
  NovaGPUApplyTilingLevelReduction.cpp
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
  
  # IREE Dependencies (Using existing Nova linkage/vendored paths if available, otherwise adapting)
  # For now, assuming we implement simplified versions or link against IREE libraries if setup.
  # Given previous context, we likely reimplement simplified passes.
)
