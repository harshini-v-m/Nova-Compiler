add_mlir_library(CompilerTransforms
  AddGpuMemoryCopies.cpp
  AffineScalarizeAccumulator.cpp
  FuseMatmulBias.cpp
  ConvertMemRefToGpu.cpp
  FixGpuLaunch.cpp
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