add_mlir_library(NovaCompilerAPI
  NovaCompilerAPI.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  
  # Core Dialects and Transforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRAsyncDialect
  MLIRAsyncTransforms
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRTosaDialect
  MLIRTosaTransforms
  
  # Bufferization
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRBufferizationTransformOps
  
  # GPU and NVVM related
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRNVVMDialect
  MLIRNVVMToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRNVGPUToNVVM
  MLIRNVGPUTransforms

  # EmitC and other utilities
  MLIREmitCDialect
  MLIREmitCTransforms
  MLIRUBDialect
  MLIRMathDialect
  MLIRMathTransforms
  MLIRControlFlowDialect
  MLIRControlFlowTransforms
  
  MLIRNova
  NovaPipelines
)
