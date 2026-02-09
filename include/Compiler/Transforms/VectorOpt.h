#ifndef COMPILER_TRANSFORMS_VECTOROPT_H
#define COMPILER_TRANSFORMS_VECTOROPT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createVectorOptPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_VECTOROPT_H
