// Nova GPU Erase Fusion Barriers Pass
//
// Erases all nova.fusion_barrier ops by replacing them with their source
// operand. This is the Nova equivalent of what IREE does at the start of
// IREEComprehensiveBufferizePass::runOnOperation():
//
//   funcOp.walk([&](IREE::Codegen::FusionBarrierOp barrier) {
//     rewriter.replaceOp(barrier, barrier.getSource());
//   });
//
// nova.fusion_barrier is inserted by NovaGPUPromoteMatmulOperandsPass to
// prevent the global→shared linalg.copy (Stage 1) from being fused into the
// per-thread linalg.copy (Stage 2) during tiling. Once tiling and all fusion
// decisions have been made, the barrier is no longer needed and must be removed
// before bufferization (it has no bufferized form).
//
// This pass must run as the first step of addNovaGPUBufferizePasses.

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::nova {

struct NovaGPUEraseFusionBarriersPass
    : public PassWrapper<NovaGPUEraseFusionBarriersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUEraseFusionBarriersPass)

  NovaGPUEraseFusionBarriersPass() = default;
  NovaGPUEraseFusionBarriersPass(const NovaGPUEraseFusionBarriersPass &) =
      default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nova::NovaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Collect barriers first to avoid iterator invalidation.
    SmallVector<FusionBarrierOp> barriers;
    funcOp.walk([&](FusionBarrierOp barrier) { barriers.push_back(barrier); });

    for (FusionBarrierOp barrier : barriers)
      rewriter.replaceOp(barrier, barrier.getSource());
  }

  StringRef getArgument() const override {
    return "nova-gpu-erase-fusion-barriers";
  }
  StringRef getDescription() const override {
    return "Erases nova.fusion_barrier ops before bufferization";
  }
};

std::unique_ptr<Pass> createNovaGPUEraseFusionBarriersPass() {
  return std::make_unique<NovaGPUEraseFusionBarriersPass>();
}

void registerNovaGPUEraseFusionBarriersPass() {
  PassRegistration<NovaGPUEraseFusionBarriersPass>();
}

} // namespace mlir::nova
