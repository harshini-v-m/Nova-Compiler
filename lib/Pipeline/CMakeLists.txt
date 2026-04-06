add_mlir_library(NovaPipelines
  Pipeline.cpp
  Gpupipeline.cpp 
     

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
   

  
  MLIRNovaToLinalgTransforms
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
  MLIRVectorToGPU
  MLIRVectorToSCF
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
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRLinalgTransformOps

)

target_include_directories(NovaPipelines PUBLIC 
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/Compiler/Pipeline>
)

add_compile_definitions(NOVA_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")