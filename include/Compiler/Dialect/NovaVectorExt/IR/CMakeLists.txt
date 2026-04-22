# Generate interface declarations (AttrInterface)
set(LLVM_TARGET_DEFINITIONS NovaVectorExtInterfaces.td)
mlir_tablegen(NovaVectorExtAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(NovaVectorExtAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRNovaVectorExtInterfacesIncGen)
add_dependencies(mlir-headers MLIRNovaVectorExtInterfacesIncGen)

# Generate dialect declarations
set(LLVM_TARGET_DEFINITIONS NovaVectorExtBase.td)
mlir_tablegen(NovaVectorExtDialect.h.inc -gen-dialect-decls -dialect=nova_vector_ext)
mlir_tablegen(NovaVectorExtDialect.cpp.inc -gen-dialect-defs -dialect=nova_vector_ext)
add_public_tablegen_target(MLIRNovaVectorExtDialectIncGen)
add_dependencies(mlir-headers MLIRNovaVectorExtDialectIncGen)

# Generate attribute declarations
set(LLVM_TARGET_DEFINITIONS NovaVectorExtAttrs.td)
mlir_tablegen(NovaVectorExtAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=nova_vector_ext)
mlir_tablegen(NovaVectorExtAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=nova_vector_ext)
add_public_tablegen_target(MLIRNovaVectorExtAttrsIncGen)
add_dependencies(mlir-headers MLIRNovaVectorExtAttrsIncGen)

# Generate op declarations
set(LLVM_TARGET_DEFINITIONS NovaVectorExtOps.td)
mlir_tablegen(NovaVectorExtOps.h.inc -gen-op-decls)
mlir_tablegen(NovaVectorExtOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRNovaVectorExtOpsIncGen)
add_dependencies(mlir-headers MLIRNovaVectorExtOpsIncGen)
