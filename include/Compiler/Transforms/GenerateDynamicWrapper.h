//===- GenerateDynamicWrapper.h - Dynamic args wrapper pass -----*- C++ -*-===//
//
// Nova Compiler - Transforms
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_GENERATE_DYNAMIC_WRAPPER_H
#define NOVA_TRANSFORMS_GENERATE_DYNAMIC_WRAPPER_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

/// Creates a pass that generates a wrapper function for dynamic argument
/// dispatch. The wrapper takes (result_ptr, args_array_ptr) and unpacks
/// the array to call the original function with individual arguments.
std::unique_ptr<Pass> createGenerateDynamicWrapperPass();

} // namespace nova
} // namespace mlir

#endif // NOVA_TRANSFORMS_GENERATE_DYNAMIC_WRAPPER_H
