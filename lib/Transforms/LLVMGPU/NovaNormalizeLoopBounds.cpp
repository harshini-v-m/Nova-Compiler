//===- NovaNormalizeLoopBounds.cpp - Normalize scf.forall bounds ----------===//
//
// Transforms scf.forall loops to have lb=0, step=1 ("normalized" form).
// Inserts affine.apply ops to compute denormalized induction variable values.
//
// Before:
//   scf.forall (%i) = (%lb) to (%ub) step (%s) { use(%i) }
//
// After:
//   scf.forall (%i) = (0) to (ceildiv(%ub - %lb, %s)) step (1) {
//     %denorm = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%i)[%s, %lb]
//     use(%denorm)
//   }
//
// Mirrors IREE's NormalizeLoopBounds.cpp from Codegen/Common/.
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "nova-normalize-loop-bounds"

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Eliminate degenerate single-iteration scf.forall ops.
//
// A scf.forall with all upper bounds == 1 executes its body exactly once.
// After bufferization (memref mode), these have no results and can be
// inlined: replace induction variables with constant 0, move body ops
// before the forall, and erase it.
//===----------------------------------------------------------------------===//
static LogicalResult eliminateDegenerateForall(IRRewriter &rewriter,
                                                scf::ForallOp forallOp) {
  // Only handle foralls with no results (post-bufferization memref form).
  if (forallOp.getNumResults() != 0)
    return failure();

  // Only eliminate degenerate foralls that are nested inside another forall.
  // Top-level block-mapped foralls must be preserved as GPU launch boundaries,
  // even when they have a single iteration (e.g. small tensors with 1 block).
  if (!forallOp->getParentOfType<scf::ForallOp>())
    return failure();

  // Check that all upper bounds are statically 1.
  for (OpFoldResult ub : forallOp.getMixedUpperBound()) {
    std::optional<int64_t> ubVal = getConstantIntValue(ub);
    if (!ubVal || *ubVal != 1)
      return failure();
  }

  // Replace all induction variable uses with constant 0.
  Location loc = forallOp.getLoc();
  rewriter.setInsertionPoint(forallOp);

  // Only create the constant if any IV is actually used.
  Value zero;
  for (Value iv : forallOp.getInductionVars()) {
    if (!iv.use_empty()) {
      if (!zero) {
        zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));
      }
      rewriter.replaceAllUsesWith(iv, zero);
    }
  }

  // Move all body ops (except the terminator) before the forall.
  Block *body = forallOp.getBody();
  Operation *terminator = body->getTerminator();
  for (auto &op : llvm::make_early_inc_range(*body)) {
    if (&op == terminator)
      continue;
    op.moveBefore(forallOp);
  }

  rewriter.eraseOp(forallOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Helper: compute ceildiv(ub - lb, step) as the normalized upper bound.
//===----------------------------------------------------------------------===//
static OpFoldResult emitNormalizedUpperBound(RewriterBase &rewriter,
                                             Location loc, OpFoldResult lb,
                                             OpFoldResult ub,
                                             OpFoldResult step) {
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  return affine::makeComposedFoldedAffineApply(
      rewriter, loc, (d0 - d1).ceilDiv(d2), {ub, lb, step});
}

//===----------------------------------------------------------------------===//
// Helper structure for the new loop bounds.
//===----------------------------------------------------------------------===//
namespace {
struct LoopRanges {
  SmallVector<OpFoldResult> lowerBounds;
  SmallVector<OpFoldResult> upperBounds;
  SmallVector<OpFoldResult> steps;
};
} // namespace

//===----------------------------------------------------------------------===//
// Core: emit normalized bounds and replace IV uses with denormalized values.
//===----------------------------------------------------------------------===//
static FailureOr<LoopRanges>
emitNormalizedLoopBounds(RewriterBase &rewriter, Location loc, Block *body,
                         ValueRange ivs, ArrayRef<OpFoldResult> lbs,
                         ArrayRef<OpFoldResult> ubs,
                         ArrayRef<OpFoldResult> steps) {
  Attribute zero = rewriter.getIndexAttr(0);
  Attribute one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> newLbs;
  SmallVector<OpFoldResult> newUbs;
  SmallVector<OpFoldResult> newSteps;
  for (auto [iv, lb, ub, step] : llvm::zip_equal(ivs, lbs, ubs, steps)) {
    std::optional<int64_t> stepInt = getConstantIntValue(step);
    // Bail out on negative steps.
    if (!stepInt || stepInt.value() <= 0) {
      return failure();
    }

    // The lower bound and step of a normalized loop is always zero/one.
    newLbs.push_back(zero);
    newSteps.push_back(one);

    // Compute the normalized upper bound.
    OpFoldResult newUb = emitNormalizedUpperBound(rewriter, loc, lb, ub, step);
    newUbs.push_back(newUb);

    // Compute and replace the denormalized loop iterator argument in the loop
    // body with an insertion guard.
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(body);
      AffineExpr idx, stepExpr, lbExpr;
      bindDims(rewriter.getContext(), idx, stepExpr, lbExpr);
      affine::AffineApplyOp denormalizedIV = affine::makeComposedAffineApply(
          rewriter, loc, idx * stepExpr + lbExpr, {iv, step, lb});
      SmallPtrSet<Operation *, 2> preserve = {iv.getDefiningOp(),
                                              denormalizedIV};
      rewriter.replaceAllUsesExcept(iv, denormalizedIV.getResult(), preserve);
    }
  }
  return LoopRanges{newLbs, newUbs, newSteps};
}

//===----------------------------------------------------------------------===//
// normalizeLoopBounds for scf::ForallOp
//===----------------------------------------------------------------------===//
static LogicalResult normalizeLoopBounds(RewriterBase &rewriter,
                                         scf::ForallOp forallOp) {
  OpBuilder::InsertionGuard g(rewriter);
  if (forallOp.isNormalized()) {
    return success();
  }

  rewriter.setInsertionPoint(forallOp);
  FailureOr<LoopRanges> newLoopParams = emitNormalizedLoopBounds(
      rewriter, forallOp.getLoc(), forallOp.getBody(),
      forallOp.getInductionVars(), forallOp.getMixedLowerBound(),
      forallOp.getMixedUpperBound(), forallOp.getMixedStep());
  if (failed(newLoopParams)) {
    return failure();
  }

  rewriter.setInsertionPointAfter(forallOp);
  auto newLoop = scf::ForallOp::create(
      rewriter, rewriter.getUnknownLoc(), newLoopParams->lowerBounds,
      newLoopParams->upperBounds, newLoopParams->steps, forallOp.getOutputs(),
      forallOp.getMapping());
  rewriter.eraseOp(newLoop.getTerminator());
  rewriter.mergeBlocks(forallOp.getBody(), newLoop.getBody(),
                       newLoop.getBody()->getArguments());
  rewriter.replaceOp(forallOp, newLoop);

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//
struct NovaNormalizeLoopBoundsPass
    : public PassWrapper<NovaNormalizeLoopBoundsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaNormalizeLoopBoundsPass)

  NovaNormalizeLoopBoundsPass() = default;
  NovaNormalizeLoopBoundsPass(const NovaNormalizeLoopBoundsPass &) = default;

  StringRef getArgument() const override {
    return "nova-normalize-loop-bounds";
  }
  StringRef getDescription() const override {
    return "Normalize scf.forall loop bounds to lb=0, step=1";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);

    // First pass: eliminate degenerate single-iteration foralls (inner→outer).
    SmallVector<scf::ForallOp> forallOps;
    funcOp.walk([&](scf::ForallOp op) { forallOps.push_back(op); });
    for (auto forallOp : llvm::reverse(forallOps))
      (void)eliminateDegenerateForall(rewriter, forallOp);

    // Second pass: normalize remaining forall loop bounds.
    funcOp.walk([&](scf::ForallOp forallOp) {
      (void)normalizeLoopBounds(rewriter, forallOp);
    });
  }
};

std::unique_ptr<Pass> createNovaNormalizeLoopBoundsPass() {
  return std::make_unique<NovaNormalizeLoopBoundsPass>();
}

void registerNovaNormalizeLoopBoundsPass() {
  PassRegistration<NovaNormalizeLoopBoundsPass>();
}

} // namespace mlir::nova
