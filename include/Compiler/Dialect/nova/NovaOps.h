#ifndef NOVA_OPS_H
#define NOVA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Compiler/Dialect/nova/NovaDialect.h"

#define GET_OP_CLASSES

#include "Compiler/Dialect/nova/NovaOpsEnums.h.inc"


// Ensure custom op traits are visible to TableGen-generated headers.
#include "Compiler/Dialect/nova/Traits.h"
#include "Compiler/Dialect/nova/NovaOps.h.inc"


#endif // NOVA_OPS_H
