//===- NovaConfigTrackingCanonicalizer.cpp - Config-preserving canonicalize ===//
//
// A canonicalize pass that propagates `lowering_config` attributes to newly
// created ops when the original op is replaced by a rewrite pattern.
//
// Why this is needed:
//   After K-tiling, the original linalg.matmul is replaced by a new one inside
//   the scf.for loop. A plain canonicalize pass would drop the `lowering_config`
//   attribute, so Thread and Subgroup tiling passes see no config on the new op
//   and fall back to hardcoded heuristic values — producing wrong tile sizes.
//
//   The ConfigTrackingListener detects each replacement, copies the
//   `lowering_config` dict from the old op to the new op (if they share the
//   same op name), and all subsequent tiling levels can read the config.
//
// Mirrors IREE's ConfigTrackingCanonicalizer.cpp exactly in mechanism,
// using Nova's `kLoweringConfigAttrName` instead of IREE's interface.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// ConfigTrackingListener
//
// CORE LOGIC — hooked into the greedy rewriter. When an op is replaced,
// if the original op carried a `lowering_config` attribute AND the replacement
// op is the same kind (e.g., both linalg.matmul), we copy the config forward.
//
// This ensures that tiling configs survive through canonicalization-induced
// replacements (e.g., fold/propagation rewrites that clone the op).
//===----------------------------------------------------------------------===//

class ConfigTrackingListener : public RewriterBase::Listener {
public:
  void notifyOperationReplaced(Operation *op, ValueRange replacements) override {
    // Nothing to propagate if the replacement set is empty.
    if (replacements.empty())
      return;

    // If the original op had no lowering_config, nothing to propagate.
    DictionaryAttr cfg = getLoweringConfig(op);
    if (!cfg)
      return;

    // IMPORTANT: Walk through tensor.cast-like ops to reach the actual
    // defining op. Canonicalize sometimes wraps new ops in a cast, so we
    // peel those off to find the true replacement before doing the name check.
    auto skipCasts = [](Value v) -> Operation * {
      Operation *defOp = v.getDefiningOp();
      if (!defOp) return nullptr;
      while (auto cast = dyn_cast<tensor::CastOp>(defOp))
        defOp = cast.getSource().getDefiningOp();
      return defOp;
    };

    Operation *newOp = skipCasts(replacements.front());
    // Only propagate if the replacement is the same op kind (conservative).
    // Propagating configs across op-kind boundaries would incorrectly stamp
    // tiling configs on unrelated ops.
    if (!newOp || newOp->getName() != op->getName())
      return;

    // Verify all replacement values come from the same op.
    for (Value v : replacements.drop_front()) {
      if (skipCasts(v) != newOp)
        return;
    }

    // Don't overwrite an existing config — preserves configs stamped by the
    // strategy pass over any that would come from a parent op.
    if (getLoweringConfig(newOp))
      return;

    setLoweringConfig(newOp, cfg);
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NovaConfigTrackingCanonicalizerPass
    : public PassWrapper<NovaConfigTrackingCanonicalizerPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaConfigTrackingCanonicalizerPass)

  NovaConfigTrackingCanonicalizerPass() = default;
  NovaConfigTrackingCanonicalizerPass(
      const NovaConfigTrackingCanonicalizerPass &) = default;

  // Collect all canonicalization patterns once per pass instantiation so
  // they are shared across repeated invocations (e.g., in a pipeline loop).
  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet owning(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owning);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(owning, ctx);
    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owning));
    return success();
  }

  void runOnOperation() override {
    ConfigTrackingListener listener;
    GreedyRewriteConfig cfg;
    cfg.setListener(&listener);
    // Non-convergence is not treated as a failure (same policy as the upstream
    // canonicalize pass — the IR is still valid after the iteration limit).
    (void)applyPatternsGreedily(getOperation(), *patterns, cfg);
  }

  StringRef getArgument() const override {
    return "nova-config-tracking-canonicalize";
  }
  StringRef getDescription() const override {
    return "Canonicalize while propagating lowering_config to replacement ops";
  }

  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaConfigTrackingCanonicalizerPass() {
  return std::make_unique<NovaConfigTrackingCanonicalizerPass>();
}

void registerNovaConfigTrackingCanonicalizerPass() {
  PassRegistration<NovaConfigTrackingCanonicalizerPass>();
}

} // namespace mlir::nova
