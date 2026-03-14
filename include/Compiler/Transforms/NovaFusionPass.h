#ifndef COMPILER_TRANSFORMS_NOVA_FUSION_PASS_H_
#define COMPILER_TRANSFORMS_NOVA_FUSION_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

namespace nova {

/// Populate all Nova fusion patterns (SCE fwd+bwd, etc.)
void populateNovaFusionPatterns(RewritePatternSet &patterns);

/// Create the NovaFusionPass
std::unique_ptr<Pass> createNovaFusionPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_NOVA_FUSION_PASS_H_
