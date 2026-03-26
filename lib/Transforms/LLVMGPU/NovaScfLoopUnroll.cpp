//===- NovaScfLoopUnroll.cpp - SCF loop unroll pass -----------------------===//
//
// Wraps mlir::loopUnrollByFactor as a proper MLIR pass so it can be inserted
// into the Nova GPU pipeline and invoked standalone via nova-opt with
// --nova-scf-loop-unroll.
//
// Pipeline position: after scf-for-loop-specialization (which peels remainder
// loops so trip-counts are multiples of the factor), and after LICM (so
// loop-invariant ops are hoisted before they get replicated by unrolling).
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaScfLoopUnroll.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct NovaScfLoopUnrollPass
    : public PassWrapper<NovaScfLoopUnrollPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaScfLoopUnrollPass)

  StringRef getArgument()    const final { return "nova-scf-loop-unroll"; }
  StringRef getDescription() const final {
    return "Unroll scf.for loops by a fixed factor (loopUnrollByFactor)";
  }

  explicit NovaScfLoopUnrollPass(uint64_t factor = 4) : unrollFactor(factor) {}

  void runOnOperation() override {
    if (unrollFactor <= 1)
      return; // factor=0 asserts in MLIR; factor=1 is a no-op

    SmallVector<scf::ForOp, 8> loops;
    // Post-order walk: inner loops are collected before their parents,
    // so we unroll inner loops first and never hold a stale pointer to
    // an op erased by unrolling an enclosing loop.
    getOperation()->walk([&](scf::ForOp forOp) {
      loops.push_back(forOp);
    });
    for (auto loop : loops) {
      // Guard against loops already erased by a prior outer-loop unroll
      // (rare, but possible if unrollFactor == 1 slips through or the loop
      // was DCE'd between iterations).
      if (!loop || loop->getParentOp() == nullptr)
        continue;
      (void)mlir::loopUnrollByFactor(loop, unrollFactor);
    }
  }

  uint64_t unrollFactor;
};

} // namespace

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createNovaScfLoopUnrollPass(uint64_t unrollFactor) {
  return std::make_unique<NovaScfLoopUnrollPass>(unrollFactor);
}

} // namespace nova
} // namespace mlir
