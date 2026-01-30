#ifndef COMPILER_TRANSLATION_NOVAREDUCETOGPU_H
#define COMPILER_TRANSLATION_NOVAREDUCETOGPU_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createNovaReduceToGpuPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSLATION_NOVAREDUCETOGPU_H
