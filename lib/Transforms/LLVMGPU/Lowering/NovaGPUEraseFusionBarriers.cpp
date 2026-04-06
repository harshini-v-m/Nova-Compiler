//===- NovaGPUEraseFusionBarriers.cpp - Erase fusion barriers pre-bufferize ===//
//
// Erases all nova.fusion_barrier ops by replacing each one with its source
// operand. This is the Nova equivalent of what IREE does at the start of
// IREEComprehensiveBufferizePass::runOnOperation():
//
//   funcOp.walk([&](IREE::Codegen::FusionBarrierOp barrier) {
//     rewriter.replaceOp(barrier, barrier.getSource());
//   });
//
// Why nova.fusion_barrier exists:
//   NovaGPUPromoteMatmulOperandsPass inserts a nova.fusion_barrier between
//   Stage 1 (global→shared linalg.copy) and Stage 2 (per-thread linalg.copy).
//   The barrier is an opaque identity — the fusion analysis cannot see through
//   it, so Stage 1 and Stage 2 always end up in separate loops. This prevents
//   the global→shared cooperative copy from being incorrectly fused into the
//   per-thread loop that runs only for each thread's tile.
//
// IMPORTANT: This pass must run as the first step of addNovaGPUBufferizePasses.
//   nova.fusion_barrier has no bufferized form (it is a tensor-level fence).
//   If it is not erased before OneShotBufferize, the bufferizer will error on
//   the unknown op.
//
//===----------------------------------------------------------------------===//

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

    // IMPORTANT: Collect all barriers into a snapshot list before erasing.
    // Walking and erasing at the same time invalidates the walk iterator.
    SmallVector<FusionBarrierOp> barriers;
    funcOp.walk([&](FusionBarrierOp barrier) { barriers.push_back(barrier); });

    // Replace each barrier with its source operand — a pure identity erasure.
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
