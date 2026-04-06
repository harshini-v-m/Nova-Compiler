//===- LoopSplit.cpp - nova-scf-loop-split pass ---------------------------===//
//
// Splits scf.for loops at a condition boundary found in the loop body.
//
// Matched pattern (IV on either side of comparison; step must be 1):
//
//   for i = lb to ub step 1, init(inits...) iter_arg(args...) {
//       [pre_ops]
//       %r... = scf.if arith.cmpi(pred, i, sp) {   // sp is loop-invariant
//           [then_ops]   scf.yield %t...
//       } else {
//           [else_ops]   scf.yield %e...
//       }
//       [post_ops referencing %r...]
//       scf.yield %yield...
//   }
//
// Emitted:
//
//   %split_ub = arith.maxsi(arith.minsi(sp_adj, ub), lb)
//
//   %mid... = scf.for i = lb to %split_ub step 1, init(inits...) {
//       [pre_ops]
//       [branch_A inlined]     // condition always holds in this range
//       [post_ops using branch_A results]
//       scf.yield ...
//   }
//
//   %out... = scf.for i = %split_ub to ub step 1, init(%mid...) {
//       [pre_ops]
//       [branch_B inlined]     // condition never holds in this range
//       [post_ops using branch_B results]
//       scf.yield ...
//   }
//
//   // original loop results replaced by %out...
//
// Predicate → split point (sp_adj) and branch assignment:
//   slt(iv, sp)  → sp_adj = sp,   loop1 = then, loop2 = else
//   sle(iv, sp)  → sp_adj = sp+1, loop1 = then, loop2 = else
//   sgt(iv, sp)  → sp_adj = sp+1, loop1 = else, loop2 = then
//   sge(iv, sp)  → sp_adj = sp,   loop1 = else, loop2 = then
//   (IV on right: predicate is flipped to the above forms)
//
// Legality:
//   - Loop step must be the constant 1.
//   - The scf.if must be a direct child of the loop body (not inside a nested
//     scf.for), and its condition must compare the IV against an outside value.
//   - If the scf.if has no else block it must also have no results.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/LoopSplit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-scf-loop-split"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

struct SplitInfo {
  scf::IfOp            ifOp;
  Value                splitVal;   // loop-invariant value from the comparison
  arith::CmpIPredicate normalPred; // predicate normalized to: iv <pred> splitVal
};

/// Return the constant integer value of \p v, or nullopt.
static std::optional<int64_t> getConstIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  return std::nullopt;
}

/// Scan the direct body ops of \p loop for a splittable scf.if.
/// Only looks at immediate children — nested loops are skipped entirely.
static std::optional<SplitInfo> findSplitInfo(scf::ForOp loop) {
  // Step must be the constant 1 so that split_ub is always iteration-aligned.
  auto step = getConstIndex(loop.getStep());
  if (!step || *step != 1)
    return std::nullopt;

  Value iv = loop.getInductionVar();

  for (Operation &op : loop.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (!ifOp)
      continue;

    // Condition must be a signed integer comparison.
    auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      continue;

    auto pred = cmp.getPredicate();
    if (pred != arith::CmpIPredicate::slt &&
        pred != arith::CmpIPredicate::sle &&
        pred != arith::CmpIPredicate::sgt &&
        pred != arith::CmpIPredicate::sge)
      continue;

    // One operand must be the IV; the other must be defined outside the loop.
    Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
    Value splitVal;
    arith::CmpIPredicate normalPred;

    if (lhs == iv && loop.isDefinedOutsideOfLoop(rhs)) {
      splitVal  = rhs;
      normalPred = pred;
    } else if (rhs == iv && loop.isDefinedOutsideOfLoop(lhs)) {
      splitVal = lhs;
      // Swap sides: slt(sp,iv) = sgt(iv,sp), etc.
      switch (pred) {
      case arith::CmpIPredicate::slt: normalPred = arith::CmpIPredicate::sgt; break;
      case arith::CmpIPredicate::sle: normalPred = arith::CmpIPredicate::sge; break;
      case arith::CmpIPredicate::sgt: normalPred = arith::CmpIPredicate::slt; break;
      case arith::CmpIPredicate::sge: normalPred = arith::CmpIPredicate::sle; break;
      default: continue;
      }
    } else {
      continue;
    }

    // An else-less scf.if is only legal to split if it has no results
    // (loop2 would have no else-branch ops and nothing to map results to).
    if (!ifOp.elseBlock() && ifOp.getNumResults() != 0)
      continue;

    return SplitInfo{ifOp, splitVal, normalPred};
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Body builder
//===----------------------------------------------------------------------===//

/// Clone the body of \p orig into the current insertion point of \p b,
/// inlining the chosen branch (\p useThen) of \p ifOp in place of the
/// scf.if, and emitting the remapped scf.yield at the end.
///
/// \p newIV and \p newIterArgs are the fresh induction variable and
/// iter-arg block arguments of the destination loop.
static void buildSplitBody(OpBuilder &b, Location loc, scf::ForOp orig,
                            scf::IfOp ifOp, bool useThen, Value newIV,
                            ValueRange newIterArgs) {
  IRMapping mp;
  mp.map(orig.getInductionVar(), newIV);
  for (auto [a, v] : llvm::zip(orig.getRegionIterArgs(), newIterArgs))
    mp.map(a, v);

  for (Operation &op : orig.getBody()->without_terminator()) {
    if (&op == ifOp.getOperation()) {
      // Inline the chosen branch, mapping ifOp results → branch yield values.
      Block *branch = useThen ? ifOp.thenBlock() : ifOp.elseBlock();
      if (branch) {
        for (Operation &bop : *branch) {
          if (bop.hasTrait<OpTrait::IsTerminator>()) {
            // scf.yield of the branch: its operands are the ifOp results.
            for (auto [res, yv] :
                 llvm::zip(ifOp.getResults(), bop.getOperands()))
              mp.map(res, mp.lookupOrDefault(yv));
            break;
          }
          b.clone(bop, mp); // updates mp with new result mappings
        }
      }
      // No branch (else-less if, useThen=false) → ifOp has no results →
      // nothing to emit and nothing to map.
      continue;
    }
    b.clone(op, mp);
  }

  // Emit the remapped yield.
  SmallVector<Value> yields;
  for (Value v : orig.getBody()->getTerminator()->getOperands())
    yields.push_back(mp.lookupOrDefault(v));
  b.create<scf::YieldOp>(loc, yields);
}

//===----------------------------------------------------------------------===//
// Rewrite pattern
//===----------------------------------------------------------------------===//

struct LoopSplitPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    auto info = findSplitInfo(loop);
    if (!info)
      return failure();

    Location loc = loop.getLoc();

    // ── Compute split_ub ────────────────────────────────────────────────────
    //
    // normalPred tells us: iv <normalPred> splitVal is true in loop1.
    //
    //   slt(iv, sp)  → split at sp:   loop1=[lb,sp),   loop2=[sp,ub)
    //   sle(iv, sp)  → split at sp+1: loop1=[lb,sp+1), loop2=[sp+1,ub)
    //   sgt(iv, sp)  → split at sp+1: loop1=[lb,sp+1), loop2=[sp+1,ub)
    //   sge(iv, sp)  → split at sp:   loop1=[lb,sp),   loop2=[sp,ub)
    //
    // thenFirst=true  → loop1 uses then-branch, loop2 uses else-branch.
    // thenFirst=false → loop1 uses else-branch, loop2 uses then-branch.

    Value rawSplit = info->splitVal;
    bool  thenFirst;

    switch (info->normalPred) {
    case arith::CmpIPredicate::slt:
      thenFirst = true;
      break;
    case arith::CmpIPredicate::sle: {
      thenFirst = true;
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rawSplit  = rewriter.create<arith::AddIOp>(loc, info->splitVal, one);
      break;
    }
    case arith::CmpIPredicate::sgt: {
      thenFirst = false;
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rawSplit  = rewriter.create<arith::AddIOp>(loc, info->splitVal, one);
      break;
    }
    case arith::CmpIPredicate::sge:
      thenFirst = false;
      break;
    default:
      return failure(); // unreachable; filtered by findSplitInfo
    }

    // Clamp: split_ub ∈ [lb, ub] so out-of-range sp produces one empty loop.
    Value splitUb = rewriter.create<arith::MinSIOp>(loc, rawSplit,
                                                    loop.getUpperBound());
    splitUb = rewriter.create<arith::MaxSIOp>(loc, splitUb,
                                              loop.getLowerBound());

    LLVM_DEBUG(llvm::dbgs() << "[nova-scf-loop-split] splitting loop at "
                             << (thenFirst ? "then" : "else") << "-first boundary\n");

    // ── Loop 1: [lb, splitUb) ───────────────────────────────────────────────
    auto loop1 = rewriter.create<scf::ForOp>(
        loc,
        loop.getLowerBound(), splitUb, loop.getStep(),
        loop.getInits(),
        [&](OpBuilder &b, Location l, Value iv1, ValueRange args1) {
          buildSplitBody(b, l, loop, info->ifOp, thenFirst, iv1, args1);
        });

    loop1->setAttr("nova.loop.split", rewriter.getUnitAttr());

    // ── Loop 2: [splitUb, ub) ───────────────────────────────────────────────
    // Initialized from loop1's results so iter_args flow correctly.
    auto loop2 = rewriter.create<scf::ForOp>(
        loc,
        splitUb, loop.getUpperBound(), loop.getStep(),
        loop1.getResults(),
        [&](OpBuilder &b, Location l, Value iv2, ValueRange args2) {
          buildSplitBody(b, l, loop, info->ifOp, !thenFirst, iv2, args2);
        });

    loop2->setAttr("nova.loop.split", rewriter.getUnitAttr());

    rewriter.replaceOp(loop, loop2.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaScfLoopSplitPass
    : public PassWrapper<NovaScfLoopSplitPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaScfLoopSplitPass)

  StringRef getArgument()    const final { return "nova-scf-loop-split"; }
  StringRef getDescription() const final {
    return "Split scf.for loops at scf.if condition boundaries (step-1 loops)";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LoopSplitPattern>(ctx);

    // Top-down: process outer loops before inner — after an outer split, the
    // two resulting loops may each contain inner loops that are candidates too.
    GreedyRewriteConfig cfg;
    cfg.setUseTopDownTraversal(true);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), cfg)))
      getOperation()->emitRemark(
          "nova-scf-loop-split: rewriter did not converge; "
          "some split opportunities may be missed");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createNovaScfLoopSplitPass() {
  return std::make_unique<NovaScfLoopSplitPass>();
}

} // namespace nova
} // namespace mlir
