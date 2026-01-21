# Add CleanupPass to the list of sources
add_mlir_library(CompilerTransforms
  AddGpuMemoryCopies.cpp
  FuseMatmulBias.cpp
  ConvertMemRefToGpu.cpp
  FixGpuLaunch.cpp
  GenerateDynamicWrapper.cpp

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