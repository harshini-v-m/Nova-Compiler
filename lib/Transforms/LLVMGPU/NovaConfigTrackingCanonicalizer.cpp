//===- NovaConfigTrackingCanonicalizer.cpp - Config-aware canonicalize ----===//
//
// A canonicalize pass that propagates `lowering_config` attributes to newly
// created ops when the original op is replaced by a rewrite pattern.
//
// Without this: after K-tiling, the original linalg.matmul is replaced by a
// new one inside the scf.for loop. A plain canonicalize pass drops the
// `lowering_config` attribute, so Thread and Subgroup tiling passes see no
// config on the new op and fall back to hardcoded values.
//
// With this: the `ConfigTrackingListener` detects the replacement, copies the
// `lowering_config` dict from the old op to the new op, and all subsequent
// tiling levels can read the config.
//
// Mirrors IREE's `ConfigTrackingCanonicalizer.cpp` exactly in mechanism,
// using Nova's `kLoweringConfigAttrName` instead of IREE's interface.
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
// Hooked into the greedy rewriter. When an op is replaced, if the original op
// carried a `lowering_config` attribution and the replacement op is the same
// kind (e.g., both linalg.matmul), we copy the config forward.
//===----------------------------------------------------------------------===//

class ConfigTrackingListener : public RewriterBase::Listener {
public:
  void notifyOperationReplaced(Operation *op, ValueRange replacements) override {
    // No replacements → nothing to propagate.
    if (replacements.empty())
      return;

    // If the original op had no lowering_config, nothing to propagate.
    DictionaryAttr cfg = getLoweringConfig(op);
    if (!cfg)
      return;

    // Walk through cast-like ops to reach the defining op.
    auto skipCasts = [](Value v) -> Operation * {
      Operation *defOp = v.getDefiningOp();
      if (!defOp) return nullptr;
      while (auto cast = dyn_cast<tensor::CastOp>(defOp))
        defOp = cast.getSource().getDefiningOp();
      return defOp;
    };

    Operation *newOp = skipCasts(replacements.front());
    if (!newOp || newOp->getName() != op->getName())
      return;

    // All replacements must come from the same op (conservative check).
    for (Value v : replacements.drop_front()) {
      if (skipCasts(v) != newOp)
        return;
    }

    // Don't overwrite an existing config.
    if (getLoweringConfig(newOp))
      return;

    setLoweringConfig(newOp, cfg);
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaConfigTrackingCanonicalizerPass
    : public PassWrapper<NovaConfigTrackingCanonicalizerPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaConfigTrackingCanonicalizerPass)

  NovaConfigTrackingCanonicalizerPass() = default;
  NovaConfigTrackingCanonicalizerPass(
      const NovaConfigTrackingCanonicalizerPass &) = default;

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
    // Non-convergence is not a failure (same as upstream canonicalize).
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
