add_mlir_library(CompilerTransforms
  FuseMatmulBias.cpp
  ConvertMemRefToGpu.cpp
  GenerateDynamicWrapper.cpp
  VectorOpt.cpp
  RemDevAttr.cpp
  RenameGpuKernels.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRTransformUtils
  MLIRNova
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRGPUDialect
)
add_subdirectory(Affine)
add_subdirectory(LLVMGPU)