#ifndef NOVA_DIALECT_VECTOREXT_DIALECT_H
#define NOVA_DIALECT_VECTOREXT_DIALECT_H

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

// clang-format off
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtAttrs.h.inc"

#define GET_OP_CLASSES
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtOps.h.inc"
// clang-format on

#endif // NOVA_DIALECT_VECTOREXT_DIALECT_H
