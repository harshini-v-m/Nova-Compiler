//===- NovaGPULowerBarrierRegion.cpp - Lower barrier_region to value_barrier ===//
//
// Lowers nova.barrier_region ops into two nova.value_barrier ops (one for
// writes, one for reads) with the body inlined between them.
//
// This pass runs after FuseAndHoistParallelLoops (which generates
// barrier_region ops during dimension-mismatched forall fusion) and before
// bufferization (which handles value_barrier → gpu.barrier).
//
// Mirrors IREE's LowerBarrierRegion pattern (Transforms.cpp:1957-1987).
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Pattern: LowerBarrierRegion
//===----------------------------------------------------------------------===//

struct LowerBarrierRegion final
    : public OpRewritePattern<nova::BarrierRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::BarrierRegionOp barrierRegionOp,
                                PatternRewriter &rewriter) const override {
    Location loc = barrierRegionOp.getLoc();

    // Step 1. Synchronize the workers on the shared dest (write barrier).
    auto writeBarrier = rewriter.create<nova::ValueBarrierOp>(
        loc, barrierRegionOp.getInputs());

    // Step 2. Inline the barrier region body, replacing block args with
    // the write barrier results.
    auto terminator = barrierRegionOp.getBody()->getTerminator();
    rewriter.inlineBlockBefore(barrierRegionOp.getBody(), barrierRegionOp,
                               writeBarrier.getResults());
    rewriter.setInsertionPoint(terminator);

    // Step 3. Synchronize the result values (read barrier).
    auto readBarrier = rewriter.create<nova::ValueBarrierOp>(
        loc, terminator->getOperands());

    rewriter.replaceAllUsesWith(barrierRegionOp.getResults(),
                                readBarrier.getResults());
    rewriter.eraseOp(terminator);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPULowerBarrierRegionPass
    : public PassWrapper<NovaGPULowerBarrierRegionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPULowerBarrierRegionPass)

  NovaGPULowerBarrierRegionPass() = default;
  NovaGPULowerBarrierRegionPass(const NovaGPULowerBarrierRegionPass &) =
      default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nova::NovaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LowerBarrierRegion>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("failed to lower barrier_region ops");
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-lower-barrier-region";
  }
  StringRef getDescription() const override {
    return "Lowers nova.barrier_region to two nova.value_barrier ops";
  }
};

std::unique_ptr<Pass> createNovaGPULowerBarrierRegionPass() {
  return std::make_unique<NovaGPULowerBarrierRegionPass>();
}

void registerNovaGPULowerBarrierRegionPass() {
  PassRegistration<NovaGPULowerBarrierRegionPass>();
}

} // namespace mlir::nova
