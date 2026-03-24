#ifndef COMPILER_TRANSFORMS_ADD_GPU_MEMORY_COPIES_H
#define COMPILER_TRANSFORMS_ADD_GPU_MEMORY_COPIES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createAddGpuMemoryCopiesPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_ADD_GPU_MEMORY_COPIES_H
