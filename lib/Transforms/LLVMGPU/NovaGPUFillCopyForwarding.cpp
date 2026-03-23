//===- NovaGPUFillCopyForwarding.cpp - Fold fill+copy into direct fill ----===//
//
// Eliminates redundant intermediate allocations produced by OneShotBufferize
// when it conservatively lowers tensor-level constant initialization.
//
// Pattern matched (any address space, any MemRefType):
//
//   %tmp  = memref.alloc() : memref<...>
//   linalg.fill ins(%cst : T) outs(%tmp : memref<...>)
//   linalg.copy ins(%tmp : memref<...>) outs(%dst : memref<...>)
//   [memref.dealloc %tmp]           -- optional (if dealloc pipeline ran)
//
// Transformed to:
//
//   linalg.fill ins(%cst : T) outs(%dst : memref<...>)
//   -- %tmp, its fill, copy, and dealloc are all erased --
//
// Semantics preservation:
//   The new fill is placed at the COPY's original position, not the fill's.
//   This ensures any reads of %dst between the original fill and copy still
//   observe the pre-transformation %dst content — exactly as in the original.
//
// Why this pattern appears:
//   OneShotBufferize creates fresh allocations for bufferization.alloc_tensor
//   ops that feed through a tensor.insert_slice into a shared-out.  When the
//   source tensor is zero-initialized (e.g., for reduction accumulators), the
//   bufferizer emits:
//     alloc %tmp → fill(%tmp, 0) → copy(%tmp → %dst)
//   instead of directly filling %dst.  This pass removes the intermediate.
//
// Preconditions for the transformation (all must hold):
//   1. %tmp is a memref.alloc result — not a function argument or subview.
//   2. %tmp's only users are: exactly one linalg.fill, exactly one linalg.copy
//      whose source is the alloc directly (not via a subview), and optionally
//      one memref.dealloc.  No other users are permitted.
//   3. The copy source type matches %tmp's type exactly (same shape, element
//      type, address space) — ensures the fill covers the same region as copy.
//
// Generality:
//   - Works for any MemRefType and any address space (workgroup, private, global).
//   - Works for any fill constant (not just zero).
//   - Handles the case where the dealloc was inserted by the buffer dealloc
//     pipeline (Step 8 of addNovaGPUBufferizePasses).
//
// IREE equivalent: mlir::linalg::populateFoldFillIntoCopyPatterns(), extended
//   here to automatically clean up the alloc+dealloc around the eliminated copy.
//
// Pipeline position: Step 8.1 — after addNovaGPUBufferizePasses (Step 8),
//   before NovaGPUReduceBankConflictsPass (Step 8.25).  Running before bank
//   conflict padding keeps type matching simple (un-padded types).
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-fill-copy-forwarding"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Returns true if \p op is a view-like memref op that aliases its source
/// (SubView, Cast, ReinterpretCast, ExpandShape, CollapseShape, Transpose).
/// These ops do not consume the memref value — they produce an alias.
static bool isViewLikeMemrefOp(Operation *op) {
  return isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp,
             memref::ExpandShapeOp, memref::CollapseShapeOp,
             memref::TransposeOp>(op);
}

/// Partition the users of \p allocVal (following view-like aliases
/// transitively) into three buckets:
///   - \p fillOp  : the single linalg.fill using the alloc as its output
///   - \p copyOps : linalg.copy ops using the alloc (directly) as source
///   - \p deallocOps : memref.dealloc ops
///   - returns false if any unexpected user is found
///
/// Only *direct* uses of \p allocVal (not via subviews) are accepted for the
/// fill and copy roles; subviews feeding into anything other than the three
/// accepted categories cause the function to return false.
static bool classifyAllocUsers(Value allocVal,
                               linalg::FillOp &fillOp,
                               SmallVectorImpl<linalg::CopyOp> &copyOps,
                               SmallVectorImpl<memref::DeallocOp> &deallocOps) {
  // BFS over the use-def chain following view-like ops.
  SmallVector<Value> worklist = {allocVal};
  SmallPtrSet<Value, 8> visitedVals;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visitedVals.insert(v).second)
      continue;
    bool isDirectAlloc = (v == allocVal);

    for (Operation *user : v.getUsers()) {
      if (isViewLikeMemrefOp(user)) {
        // Recurse into the aliased result.
        for (Value res : user->getResults())
          worklist.push_back(res);
        continue;
      }
      if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        // The fill must write directly into the alloc (not via a subview).
        if (!isDirectAlloc)
          return false;
        if (fill->getOperand(1) != allocVal)
          return false;
        if (fillOp)
          return false; // Multiple fills — too complex to handle.
        fillOp = fill;
        continue;
      }
      if (auto copy = dyn_cast<linalg::CopyOp>(user)) {
        // Only accept direct source uses (not subview sources).
        if (!isDirectAlloc)
          return false;
        if (copy->getOperand(0) != allocVal)
          return false;
        copyOps.push_back(copy);
        continue;
      }
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
        deallocOps.push_back(dealloc);
        continue;
      }
      // Any other user (e.g. linalg.generic reading the alloc, a gpu.memcpy,
      // another alloc-level op) disqualifies the pattern.
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Core forwarding logic (used both from the pattern and the manual walk)
//===----------------------------------------------------------------------===//

/// Attempt to forward a single fill+copy chain rooted at \p copyOp.
/// Returns success() if the transformation fired.
static LogicalResult tryForwardFillCopy(linalg::CopyOp copyOp,
                                        IRRewriter &rewriter) {
  // The copy source must be a direct memref.alloc result.
  Value copySrc = copyOp->getOperand(0); // linalg.copy ins()[0]
  auto allocOp = copySrc.getDefiningOp<memref::AllocOp>();
  if (!allocOp)
    return failure();

  // Classify all users of the alloc.
  linalg::FillOp fillOp;
  SmallVector<linalg::CopyOp> copyOps;
  SmallVector<memref::DeallocOp> deallocOps;
  if (!classifyAllocUsers(allocOp.getResult(), fillOp, copyOps, deallocOps))
    return failure();

  // Preconditions:
  //   - Exactly one fill, writing the whole alloc.
  //   - Exactly one copy (the one we matched), reading the whole alloc.
  //   - At most one dealloc.
  if (!fillOp)
    return failure();
  if (copyOps.size() != 1 || copyOps[0] != copyOp)
    return failure();
  if (deallocOps.size() > 1)
    return failure();

  // The copy source type must match the alloc type exactly so the fill
  // covers exactly the same elements as the copy would have transferred.
  if (allocOp.getType() != cast<MemRefType>(copySrc.getType()))
    return failure();

  Value fillScalar = fillOp->getOperand(0); // linalg.fill ins()[0] = scalar
  Value copyDst    = copyOp->getOperand(1); // linalg.copy outs()[0] = dest

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-fill-copy-forwarding] forwarding:\n"
                 << "  fill  : " << fillOp << "\n"
                 << "  copy  : " << copyOp << "\n"
                 << "  alloc : " << allocOp << "\n"
                 << "  → new fill into: " << copyDst << "\n";
  });

  // Insert the replacement fill at the COPY's position so that any reads of
  // copyDst between the original fill and copy still see the old content.
  rewriter.setInsertionPoint(copyOp);
  rewriter.create<linalg::FillOp>(copyOp.getLoc(),
                                   ValueRange{fillScalar},
                                   ValueRange{copyDst});

  // Erase in reverse use-def order so each erasure leaves no dangling uses.
  //   1. The copy (was using allocOp as input).
  rewriter.eraseOp(copyOp);
  //   2. The fill (was using allocOp as output).
  rewriter.eraseOp(fillOp);
  //   3. The dealloc, if present (was using allocOp).
  if (!deallocOps.empty())
    rewriter.eraseOp(deallocOps[0]);
  //   4. The alloc itself (no more users).
  rewriter.eraseOp(allocOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NovaGPUFillCopyForwardingPass
    : public PassWrapper<NovaGPUFillCopyForwardingPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUFillCopyForwardingPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Snapshot all linalg.copy ops before we start modifying the IR.
    // Collecting upfront avoids iterator invalidation when ops are erased.
    SmallVector<linalg::CopyOp> copyOps;
    funcOp.walk([&](linalg::CopyOp op) { copyOps.push_back(op); });

    unsigned numForwarded = 0;
    for (linalg::CopyOp copyOp : copyOps) {
      // Skip ops that were erased by a prior iteration (e.g. if a copy was
      // reached both as a root and as a side effect of forwarding another).
      if (!copyOp->getBlock())
        continue;
      if (succeeded(tryForwardFillCopy(copyOp, rewriter)))
        ++numForwarded;
    }

    LLVM_DEBUG(llvm::dbgs() << "[nova-fill-copy-forwarding] forwarded "
                            << numForwarded << " fill+copy chains\n");
  }

  StringRef getArgument() const override {
    return "nova-gpu-fill-copy-forwarding";
  }
  StringRef getDescription() const override {
    return "Eliminates redundant intermediate allocations by forwarding "
           "linalg.fill through linalg.copy (fill+copy → direct fill). "
           "Generic: fires for any address space and any MemRefType.";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUFillCopyForwardingPass() {
  return std::make_unique<NovaGPUFillCopyForwardingPass>();
}

void registerNovaGPUFillCopyForwardingPass() {
  PassRegistration<NovaGPUFillCopyForwardingPass>();
}

} // namespace mlir::nova
