#ifndef NOVA_DIALECT_VECTOREXT_BASE
#define NOVA_DIALECT_VECTOREXT_BASE

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def NovaVectorExt_Dialect : Dialect {
  let name = "nova_vector_ext";
  let cppNamespace = "::mlir::nova::vec_ext";
  let summary = [{
    Nova Vector Extensions.
  }];
  let description = [{
    A dialect for experimenting with vector operations targeting GPU tensor
    cores via the Nova compiler pipeline.
  }];
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
  let dependentDialects = [
    "affine::AffineDialect",
    "vector::VectorDialect"
  ];
}

//===---------------------------------------------------------------------===//
// Vector layout attribute helpers
//===---------------------------------------------------------------------===//

class NovaVectorExt_Attr<string name, list<Trait> traits = []>
  : AttrDef<NovaVectorExt_Dialect, name, traits>;

#endif // NOVA_DIALECT_VECTOREXT_BASE
