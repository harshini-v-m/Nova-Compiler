//===- NovaGPUHoistAccumulator.cpp - Hoist C accumulator out of K-loop -===//
//
// Moves the matmul C-accumulator's vector.transfer_read/transfer_write pair
// out of the K-loop, replacing the in-loop read with an scf.for iter_arg and
// the in-loop write with a yield update.
//
// ── Why this pass exists ──────────────────────────────────────────────────
//
// After NovaGPUComprehensiveBufferize, the K-loop's accumulator iter_arg
// becomes a memref subview aliased into the *global* output buffer.  Inside
// the warp scf.forall body each iteration ends up doing
//
//     vector.transfer_read  %C_subview ...   ← ld.global per mma
//     compute / mma.sync
//     vector.transfer_write %C_subview ...   ← st.global per mma
//
// Per K-step that's 2 ld.global + 2 st.global per inner mma fragment, on top
// of the cp.async-staged A/B operands.  The accumulator never reaches
// registers and the kernel is dominated by HBM traffic.
//
// ── Why post-MapForallToGPU ───────────────────────────────────────────────
//
// Pre-bufferize: the read/write live inside an scf.forall (warp grid) inside
//   the scf.for K-loop.  Upstream `linalg::hoistRedundantVectorTransfers`
//   only handles transfers directly inside scf.for and cannot pierce the
//   nested forall.
// Post-bufferize, pre-MapForall: the structure is the same with memrefs.
//   Same hoist-piercing problem.
// Post-MapForallToGPU: the warp scf.forall has been replaced by gpu.thread_id
//   based index math at the K-loop's body level.  The transfer_read/write
//   are now direct children of the scf.for (or under flat scf.ifs) hitting a
//   loop-invariant memref subview at loop-invariant indices.  This is the
//   exact pattern `linalg::hoistRedundantVectorTransfers` was written for.
//
// ── Pipeline position ─────────────────────────────────────────────────────
//
//   AFTER  NovaGPUMapForallToGPUPass     — warp forall is flattened
//   BEFORE NovaGPUCreateAsyncCopiesPass  — async copy lowering assumes the
//                                          K-loop body shape it sees
//   BEFORE NovaGPUPipeliningPass         — software pipelining is cleaner
//                                          when the accumulator is already
//                                          a loop-carried SSA value
//
// ── What this pass does NOT do ────────────────────────────────────────────
//
// We delegate the actual transformation to the upstream helper
// `linalg::hoistRedundantVectorTransfers`.  This pass is a thin scheduler
// that walks every func.FuncOp and runs the helper on each scf.for found
// inside it (including those buried in gpu.launch bodies after MapForall).
// If the helper's preconditions don't hold for a given loop, it is a no-op
// for that loop — safe by construction.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-hoist-accumulator"

using namespace mlir;

namespace mlir::nova {

namespace {

struct NovaGPUHoistAccumulatorPass
    : public PassWrapper<NovaGPUHoistAccumulatorPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUHoistAccumulatorPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect,
                    memref::MemRefDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // hoistRedundantVectorTransfers operates on a function-like region by
    // walking its scf.for ops.  It tolerates being called multiple times,
    // and walks into gpu.launch bodies (which are regular regions after
    // MapForallToGPU lowered the block-mapped scf.forall).  One call
    // covers every nested loop — no need to enumerate scf.fors ourselves.
    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] hoistRedundantVectorTransfers done on @"
               << funcOp.getName() << "\n");
  }

  StringRef getArgument() const override {
    return "nova-gpu-hoist-accumulator";
  }
  StringRef getDescription() const override {
    return "Hoist matmul C-accumulator vector.transfer_read/transfer_write "
           "out of the K-loop after MapForallToGPU has flattened the warp "
           "scf.forall.";
  }
};

} // namespace

std::unique_ptr<Pass> createNovaGPUHoistAccumulatorPass() {
  return std::make_unique<NovaGPUHoistAccumulatorPass>();
}

void registerNovaGPUHoistAccumulatorPass() {
  PassRegistration<NovaGPUHoistAccumulatorPass>();
}

} // namespace mlir::nova
