#include "Compiler/Transforms/NovaFusionPass.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nova;

namespace {

//===----------------------------------------------------------------------===//
// Pattern: Fuse nova.sce + nova.sce_backward -> nova.sce_fwd_bwd
//===----------------------------------------------------------------------===//
struct FuseSceFwdBwdPattern : public OpRewritePattern<nova::SceOp> {
  using OpRewritePattern<nova::SceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::SceOp sceOp,
                                PatternRewriter &rewriter) const override {
    Value logits = sceOp.getLogits();
    Value targets = sceOp.getTargets();

    // Scan users of logits to find a matching nova.sce_backward
    nova::SceBackwardOp matchedBwd = nullptr;
    for (Operation *user : logits.getUsers()) {
      if (user == sceOp.getOperation())
        continue;
      auto bwdOp = dyn_cast<nova::SceBackwardOp>(user);
      if (!bwdOp)
        continue;
      // Check same operands
      if (bwdOp.getLogits() == logits && bwdOp.getTargets() == targets) {
        matchedBwd = bwdOp;
        break;
      }
    }

    if (!matchedBwd)
      return failure();

    // Create the fused op
    Location loc = sceOp.getLoc();
    int64_t dim = matchedBwd.getDim();

    auto fusedOp = rewriter.create<nova::SceFwdBwdOp>(
        loc, logits, targets,
        rewriter.getI64IntegerAttr(dim));

    // Replace both original ops
    rewriter.replaceOp(sceOp, fusedOp.getLoss());
    rewriter.replaceOp(matchedBwd, fusedOp.getGradLogits());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// The pass itself
//===----------------------------------------------------------------------===//
struct NovaFusionPass
    : public PassWrapper<NovaFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaFusionPass)

  StringRef getArgument() const final { return "nova-fusion"; }

  StringRef getDescription() const final {
    return "Fuse Nova dialect operation pairs that share computation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nova::NovaDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateNovaFusionPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace nova {

void populateNovaFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseSceFwdBwdPattern>(patterns.getContext());
  // Future fusion patterns go here:
  // patterns.add<FuseOtherPattern>(patterns.getContext());
}

std::unique_ptr<Pass> createNovaFusionPass() {
  return std::make_unique<NovaFusionPass>();
}

} // namespace nova
} // namespace mlir
