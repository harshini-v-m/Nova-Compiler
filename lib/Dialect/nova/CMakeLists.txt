# set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_mlir_dialect_library(MLIRNova
  NovaDialect.cpp
  NovaOps.cpp
  NovaCanonicalizations.cpp
  Broadcast.cpp
  NovaVectorLayoutAttr.cpp

  DEPENDS
  MLIRNovaOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRArithDialect
)

target_include_directories(MLIRNova PRIVATE
  # ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/include
)
