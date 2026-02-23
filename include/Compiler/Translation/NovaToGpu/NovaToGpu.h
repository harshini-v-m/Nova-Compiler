#ifndef COMPILER_TRANSLATION_NOVATOGPU_H
#define COMPILER_TRANSLATION_NOVATOGPU_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createNovaToGpuPass();
void populateMatmulPatterns(RewritePatternSet &patterns);

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSLATION_NOVATOGPU_H
