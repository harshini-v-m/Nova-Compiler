# Add CleanupPass to the list of sources
add_mlir_library(CompilerTransforms
  CleanupPass.cpp
  AddGpuMemoryCopies.cpp
  FuseMatmulBias.cpp
  ConvertMemRefToGpu.cpp
  FastmathFlag.cpp
  FixGpuLaunch.cpp
  ParallelizeOuterLoops.cpp
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