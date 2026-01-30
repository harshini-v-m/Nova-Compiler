add_mlir_library(NovaPipelines
  Pipeline.cpp
  Gpupipeline.cpp 
     

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Compiler

  DEPENDS
  MLIRNovaOpsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRAffineTransformPasses 
  AffineTransforms 
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRAffineAnalysis
  MLIRLinalgTransforms
  MLIRLinalgDialect
  MLIRNova
   

  
  MLIRNovaToArithTransforms
  MLIRNovaToLinalgTransforms
  MLIRNovaToTosaTransforms
  MLIRNovaToGpu

  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRBufferizationToMemRef
  MLIRSCFToGPU
  MLIRGPUTransforms
  MLIRVectorToLLVM
  MLIRVectorToLLVMPass
  MLIRGPUPipelines
  MLIRNVVMTarget
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRGPUToNVVMTransforms
  MLIRNVVMToLLVMIRTranslation
  MLIRTosaToLinalg
  MLIRTosaToArith
  MLIRTosaToTensor
  MLIRTosaToSCF
  MLIRMathTransforms
  MLIRUBToLLVM
  CompilerTransforms
)

target_include_directories(NovaPipelines PUBLIC 
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/Compiler/Pipeline>
)