add_llvm_executable(nova-api-test nova-api-test.cpp)

llvm_update_compile_flags(nova-api-test)
target_link_libraries(nova-api-test PRIVATE
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms

  # Your custom libraries
  MLIRNova
  MLIRAffineTransformPasses
  
  MLIRNovaToLinalgTransforms
  
  # Dependencies ordered by usage: NovaPipelines depends on CompilerTransforms
  NovaPipelines
  CompilerTransforms
  NovaCompilerAPI
  MLIRNovaVectorExt

  # Core dialects that we know exist and are needed
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRAffineAnalysis
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRLinalgUtils
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathTransforms
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRVectorDialect
  MLIRVectorTransforms

  # LLVM IR translation support
  MLIRToLLVMIRTranslationRegistration
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRNVVMTarget
  
  # OpenMP and Complex conversion support
  MLIROpenMPToLLVM
  MLIRComplexToLLVM
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  Target
  Native
  NVPTX
)

install(TARGETS nova-api-test RUNTIME DESTINATION bin)
