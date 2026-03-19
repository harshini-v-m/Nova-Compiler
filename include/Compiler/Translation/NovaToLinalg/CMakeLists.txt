# CMakeLists.txt for Nova to linalg Lowering Pass

add_mlir_library(MLIRNovaToLinalgTransforms
  NovaToLinalg.cpp
  TNovaToLinalg.cpp
  NovaLinalgFusion.cpp
  NovaToLinalg.h

  DEPENDS
  MLIRLinalgDialect
  MLIRSCFDialect

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRNova
)

target_include_directories(MLIRNovaToLinalgTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Compiler/Translation/NovaToLinalg
)