#ifndef COMPILER_TRANSFORMS_FIXGPULAUNCH_H
#define COMPILER_TRANSFORMS_FIXGPULAUNCH_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createGpuRuntimeLoweringPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_FIXGPULAUNCH_H
