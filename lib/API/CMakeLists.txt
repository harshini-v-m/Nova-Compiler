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
  
  MLIRNova
  NovaPipelines
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRComplexToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRUBToLLVM
  MLIRVectorToLLVM
  MLIROpenMPToLLVM
  MLIRGPUToGPURuntimeTransforms
  MLIRNVVMToLLVM
  MLIRNVVMTarget
)
