add_mlir_dialect_library(MLIRNovaVectorExt
  NovaVectorExtAttrs.cpp
  NovaVectorExtDialect.cpp
  NovaVectorExtOps.cpp
  NovaVectorExtFoldUnitExtentDims.cpp
  NovaVectorizeVectorExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Compiler/Dialect/NovaVectorExt

  DEPENDS
  MLIRNovaVectorExtInterfacesIncGen
  MLIRNovaVectorExtDialectIncGen
  MLIRNovaVectorExtAttrsIncGen
  MLIRNovaVectorExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRVectorDialect
  MLIRLinalgDialect
  MLIRDialectUtils
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRPass
)

target_include_directories(MLIRNovaVectorExt PRIVATE
  ${PROJECT_BINARY_DIR}/include
)
