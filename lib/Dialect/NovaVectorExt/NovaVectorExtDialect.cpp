#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nova::vec_ext;

// Dialect tablegen definitions
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.cpp.inc"

// Interface tablegen definitions
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtAttrInterfaces.cpp.inc"

namespace mlir::nova::vec_ext {

struct NovaVectorExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<NestedLayoutAttr>(attr)) {
      os << "nested";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void NovaVectorExtDialect::initialize() {
  addInterfaces<NovaVectorExtDialectOpAsmInterface>();
  registerAttributes();

#define GET_OP_LIST
  addOperations<
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtOps.cpp.inc"
      >();
}

} // namespace mlir::nova::vec_ext
