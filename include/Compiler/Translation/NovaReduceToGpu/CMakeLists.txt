add_mlir_library(MLIRNovaReduceToGpu
  NovaReduceToGpu.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRGPUDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRNova
)

target_include_directories(MLIRNovaReduceToGpu PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Compiler/Translation/NovaReduceToGpu
)

install(TARGETS MLIRNovaReduceToGpu
  EXPORT mlir-compilerTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
