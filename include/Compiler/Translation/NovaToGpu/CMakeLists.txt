add_mlir_library(MLIRNovaToGpu
  NovaToGpu.cpp
  matmul.cpp
  FusionKernelEmitter.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRGPUDialect
  MLIRNVGPUDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRNova
)

target_include_directories(MLIRNovaToGpu PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Compiler/Translation/NovaToGpu
)

install(TARGETS MLIRNovaToGpu
  EXPORT mlir-compilerTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
