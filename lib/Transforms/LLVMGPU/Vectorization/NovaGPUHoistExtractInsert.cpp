//===- NovaGPUHoistVectorExtractInsertSlice.cpp ---------------------------===//
//
// Hoists redundant vector/tensor extract+insert slice ops out of loops and
// folds away no-op identity slices.
//
// Handles BOTH vector and tensor slice ops:
//   - vector.extract_strided_slice / vector.insert_strided_slice
//   - tensor.extract_slice / tensor.insert_slice
//
// Transformation order (each step feeds the next):
//
//   Step 1 — Move tensor.extract_slice / vector.extract_strided_slice to
//             the earliest valid position in their block.
//             Exposes them to loop-invariant code motion in Steps 2-3.
//
//   Step 2 — linalg::hoistRedundantVectorTransfers
//             Hoists vector.transfer_read/write pairs that are loop-invariant.
//
//   Step 3 — moveLoopInvariantCode (scf.for)
//             Standard MLIR LICM: moves any op whose operands are all defined
//             outside the loop to just before the loop.
//
//   Step 4 — hoistLoopInvariantSubsets (MLIR built-in)
//             Hoists extract_slice/insert_slice pairs where both ops use the
//             same loop iter_arg as their tensor.
//
//   Step 5 — hoistSubsetWithLoopInvariantTensor (custom, mirrors IREE)
//             Extends Step 4: hoists when the insertion destination is a
//             loop-invariant tensor (not necessarily the iter_arg itself).
//
//   Step 6 — Cleanup patterns (greedy):
//             • CastLikeExtractSliceFolder   — no-op tensor.extract_slice → source
//             • CastLikeInsertSliceFolder    — no-op tensor.insert_slice  → source
//             • VectorExtractStridedFolder   — no-op vector.extract_strided_slice → source
//             • VectorInsertStridedFolder    — no-op vector.insert_strided_slice  → source
//             • scf::ForOp canonicalization  — remove dead iter_args
//             • TransferWriteOp canon        — simplify vector writes
//             • VectorTransferTensorSlice    — transfer + slice combos
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "nova-gpu-hoist-slice"

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// §1  Helper — earliest insertion point inside a block
//===----------------------------------------------------------------------===//

/// Find the earliest position in `block` where `op` can legally be placed.
/// That is: just after the last operand of `op` that is defined in `block`.
/// Block arguments (iter_args, func args) are always in scope — they are skipped.
///
/// Example:
///   %a = some_op          ← op A
///   %b = other_op %a      ← op B  (defines %b)
///   %c = extract_slice %b ← wants to move up; earliest point = after op B
static Operation *getEarliestInsertionPoint(Block *block, Operation *op) {
  // Start at the very first op in the block as the initial candidate.
  Operation *point = &block->front();
  DominanceInfo dom(point);

  for (Value operand : op->getOperands()) {
    // Block arguments are always available at the top — nothing to check.
    if (isa<BlockArgument>(operand))
      continue;
    Operation *defOp = operand.getDefiningOp();
    // If defOp does not dominate the current candidate, move candidate down.
    if (!dom.dominates(defOp, point))
      point = defOp;
  }
  return point; // op must be placed after this point.
}

//===----------------------------------------------------------------------===//
// §2  Loop-invariant subset hoisting (mirrors IREE's implementation)
//     Works for BOTH tensor and vector slice ops via SubsetOpInterface.
//===----------------------------------------------------------------------===//

/// Returns true if `insertion` (an insert_slice-like op) is safe to hoist
/// out of `loopLike`. All operands except the source being inserted must be
/// loop-invariant (defined outside the loop).
static bool canBeHoisted(LoopLikeOpInterface loopLike,
                         SubsetInsertionOpInterface insertion) {
  // Never hoist terminators — they must stay at the end of their block.
  if (insertion->hasTrait<OpTrait::IsTerminator>())
    return false;

  auto walkFn = [&](Operation *child) -> WalkResult {
    for (OpOperand &operand : child->getOpOperands()) {
      // Operands defined inside a nested region of `insertion` are fine —
      // they won't dangle when we move `insertion` outside the loop.
      if (insertion->isAncestor(
              operand.get().getParentRegion()->getParentOp()))
        continue;
      // The source operand is intentionally loop-variant (it's the result
      // of each iteration's computation). Allow it.
      if (&operand == &insertion.getSourceOperand())
        continue;
      // Any other loop-defined operand blocks hoisting.
      if (!loopLike.isDefinedOutsideOfLoop(operand.get()))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !insertion->walk(walkFn).wasInterrupted();
}

/// Try to hoist the extract+insert slice pair for one iter_arg (`idx`) out of
/// the loop. Returns the (possibly rebuilt) loop op.
///
/// This handles the case where the insert_slice destination is a loop-invariant
/// tensor — not necessarily the iter_arg itself. Example:
///
///   Before:
///     for %i iter_args(%t = %init) {
///       %a = extract_slice %t[0,0][8,8]
///       %c = compute %a
///       %d = insert_slice %c into %other_tensor[0,0][8,8]   ← invariant dest
///       yield %d
///     }
///
///   After:
///     %a = extract_slice %init[0,0][8,8]          ← before loop
///     %loop = for %i iter_args(%t = %a) {
///       %c = compute %t
///       yield %c
///     }
///     %out = insert_slice %loop into %other_tensor[0,0][8,8]  ← after loop
static LoopLikeOpInterface
hoistSubsetAtIterArg(RewriterBase &rewriter,
                     LoopLikeOpInterface loopLike,
                     int64_t idx) {
  // The value yielded at position `idx` must come from an insert_slice-like op.
  auto insertion = loopLike.getYieldedValues()[idx]
                       .getDefiningOp<SubsetInsertionOpInterface>();
  if (!insertion || !canBeHoisted(loopLike, insertion))
    return loopLike;

  // Keep retrying after each successful hoist (there may be multiple
  // extractions to hoist). Cap at 10 to guard against degenerate cases.
  bool changed = true;
  int iters = 0;
  while (changed && iters++ < 10) {
    changed = false;

    // Scan users of the iter_arg for a matching extract_slice-like op.
    for (Operation *user : loopLike.getRegionIterArgs()[idx].getUsers()) {
      auto extraction = dyn_cast<SubsetExtractionOpInterface>(user);
      if (!extraction)
        continue;

      // The extraction must read from the exact same subset (slice offsets
      // and sizes) as the insertion writes to.
      // The lambda always returns true because we only care about the slice
      // position, not which tensor the slices come from.
      if (!extraction.operatesOnEquivalentSubset(
              insertion, [](Value, Value) { return true; }))
        continue;

      // Provide the new yield value: the raw per-iteration tile (the source
      // of the insert_slice), not the full inserted tensor.
      NewYieldValuesFn yieldFn =
          [&](OpBuilder &, Location,
              ArrayRef<BlockArgument>) -> SmallVector<Value> {
        return {insertion.getSourceOperand().get()};
      };

      // Rebuild the loop with one extra iter_arg for the per-iteration tile.
      // replaceInitOperandUsesInLoop=true: inside the loop, uses of the
      // extraction result are replaced by the new iter_arg automatically.
      FailureOr<LoopLikeOpInterface> newLoop =
          loopLike.replaceWithAdditionalYields(
              rewriter, extraction.getResult(),
              /*replaceInitOperandUsesInLoop=*/true, yieldFn);
      if (failed(newLoop))
        return loopLike;
      loopLike = *newLoop;

      BlockArgument iterArg    = loopLike.getRegionIterArgs()[idx];
      OpResult      loopResult = loopLike.getTiedLoopResult(iterArg);
      OpResult      newResult  = loopLike.getLoopResults()->back();

      // Move the extraction BEFORE the loop and point it at the loop's
      // initial value (the full tensor before any iterations).
      rewriter.moveOpBefore(extraction, loopLike);
      extraction.getSourceOperand().set(
          loopLike.getTiedLoopInit(iterArg)->get());

      // Clone the insertion AFTER the loop to write the final tile back into
      // the destination tensor.
      rewriter.setInsertionPointAfter(loopLike);
      auto newInsertion = cast<SubsetInsertionOpInterface>(
          rewriter.clone(*insertion.getOperation()));

      // Wire: old loop result → new insertion's output.
      rewriter.replaceAllUsesWith(loopResult,
                                  newInsertion.getUpdatedDestination());
      // Wire: new insertion's source ← final tile from the new loop result.
      newInsertion.getSourceOperand().set(newResult);

      changed = true;
      break; // IR changed — restart user scan.
    }
  }
  return loopLike;
}

/// Top-level driver: try hoisting for every iter_arg of the loop.
static void hoistSubsetWithLoopInvariantTensor(RewriterBase &rewriter,
                                               LoopLikeOpInterface loopLike) {
  for (int64_t i = 0;
       i < static_cast<int64_t>(loopLike.getRegionIterArgs().size()); ++i)
    loopLike = hoistSubsetAtIterArg(rewriter, loopLike, i);
}

//===----------------------------------------------------------------------===//
// §3  Cleanup patterns — identity slice folding (tensor + vector)
//===----------------------------------------------------------------------===//

/// Fold a tensor.extract_slice that is a no-op (takes the full tensor,
/// source and result types are identical) into its source.
///
/// Example:
///   %b = tensor.extract_slice %a[0,0][8,8][1,1]
///            : tensor<8x8xf32> to tensor<8x8xf32>
///   → everywhere %b is used, replace with %a.
struct CastLikeExtractSliceFolder final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!tensor::isCastLikeExtractSliceOp(op) ||
        op.getSourceType() != op.getResultType())
      return failure();
    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};

/// Fold a tensor.insert_slice that is a no-op (inserts the full tensor back,
/// source and result types are identical) into its source.
struct CastLikeInsertSliceFolder final
    : OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!tensor::isCastLikeInsertSliceOp(op) ||
        op.getSourceType() != op.getResultType())
      return failure();
    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};

/// Fold a vector.extract_strided_slice that extracts the whole vector
/// (offsets all zero, sizes == source sizes, strides all one).
///
/// Example:
///   %b = vector.extract_strided_slice %a
///            {offsets=[0,0], sizes=[8,8], strides=[1,1]}
///            : vector<8x8xf32> to vector<8x8xf32>
///   → replaced by %a.
struct VectorExtractStridedFolder final
    : OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern<vector::ExtractStridedSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Result type must match source type exactly.
    if (op.getType() != op.getVector().getType())
      return failure();

    // All offsets must be zero — otherwise it's a real slice.
    auto offsets = llvm::to_vector(
        op.getOffsets().getAsRange<IntegerAttr>());
    if (llvm::any_of(offsets, [](IntegerAttr a) {
          return a.getInt() != 0;
        }))
      return failure();

    // All strides must be one — strided access is not a no-op.
    auto strides = llvm::to_vector(
        op.getStrides().getAsRange<IntegerAttr>());
    if (llvm::any_of(strides, [](IntegerAttr a) {
          return a.getInt() != 1;
        }))
      return failure();

    rewriter.replaceOp(op, op.getVector());
    return success();
  }
};

/// Fold a vector.insert_strided_slice that inserts the whole vector back
/// at offset zero with stride one (i.e. a no-op identity insert).
///
/// Example:
///   %b = vector.insert_strided_slice %small, %dest
///            {offsets=[0,0], strides=[1,1]}
///            : vector<8x8xf32> into vector<8x8xf32>
///   → replaced by %small (source == dest size, full overwrite).
struct VectorInsertStridedFolder final
    : OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern<vector::InsertStridedSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Source and destination vector types must be identical.
    if (op.getValueToStore().getType() != op.getDest().getType())
      return failure();

    auto offsets = llvm::to_vector(
        op.getOffsets().getAsRange<IntegerAttr>());
    if (llvm::any_of(offsets, [](IntegerAttr a) {
          return a.getInt() != 0;
        }))
      return failure();

    auto strides = llvm::to_vector(
        op.getStrides().getAsRange<IntegerAttr>());
    if (llvm::any_of(strides, [](IntegerAttr a) {
          return a.getInt() != 1;
        }))
      return failure();

    rewriter.replaceOp(op, op.getValueToStore());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// §4  The pass
//===----------------------------------------------------------------------===//

struct NovaGPUHoistVectorExtractInsertSlicePass
    : public PassWrapper<NovaGPUHoistVectorExtractInsertSlicePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUHoistVectorExtractInsertSlicePass)

  StringRef getArgument() const override { return "nova-gpu-hoist-slice"; }
  StringRef getDescription() const override {
    return "Hoist vector/tensor extract+insert slice pairs out of loops "
           "and fold away identity (no-op) slices for both tensor and "
           "vector dialects.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect,
                    tensor::TensorDialect, bufferization::BufferizationDialect,linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    // ------------------------------------------------------------------
    // Step 1: Move tensor.extract_slice and vector.extract_strided_slice
    // to the earliest valid position in their block.
    //
    // Why: hoisting analysis (Steps 3-5) works best when the candidate
    // op is as high as possible in its block. If the extract is buried
    // below unrelated ops, LICM cannot see it as a candidate.
    // ------------------------------------------------------------------
    funcOp.walk([&](Operation *op) {
      // Handle both tensor and vector extract ops.
      if (!isa<tensor::ExtractSliceOp,
               vector::ExtractStridedSliceOp>(op))
        return;
      Block *block = op->getBlock();
      Operation *earliest = getEarliestInsertionPoint(block, op);
      op->moveAfter(earliest);
    });

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-hoist-slice] after moving extracts early\n");

    // ------------------------------------------------------------------
    // Step 2: Hoist redundant vector.transfer_read/write pairs.
    // Finds pairs where the read and write hit the same address and the
    // data doesn't change across loop iterations, then moves both outside.
    // ------------------------------------------------------------------
    linalg::hoistRedundantVectorTransfers(funcOp);
    SmallVector<std::pair<vector::TransferReadOp, scf::ForallOp>> sinkWork;
funcOp.walk([&](vector::TransferReadOp readOp) {
  auto vType = readOp.getVectorType();
  int64_t numElements = vType.getNumElements();
  if (numElements <= 32)
    return; // Small reads are fine to hoist

  // Check if the next op is a warp-mapped forall
  Operation *nextOp = readOp->getNextNode();
  if (!nextOp)
    return;
  auto forallOp = dyn_cast<scf::ForallOp>(nextOp);
  if (!forallOp)
    return;

  // Check if it's warp-mapped
  auto mapping = forallOp.getMappingAttr();
  if (!mapping || mapping.getValue().empty())
    return;
  bool isWarpMapped = llvm::all_of(mapping.getValue(), [](Attribute a) {
    return isa<gpu::GPUWarpMappingAttr>(a);
  });
  if (!isWarpMapped)
    return;

  // Check all uses are inside the forall
  bool allUsesInside = llvm::all_of(readOp->getUsers(), [&](Operation *user) {
    return forallOp->isProperAncestor(user);
  });
  if (!allUsesInside)
    return;

  sinkWork.emplace_back(readOp, forallOp);
});

for (auto &[readOp, forallOp] : sinkWork) {
  readOp->moveBefore(&forallOp.getBody()->front());
}

LLVM_DEBUG(llvm::dbgs()
           << "[nova-hoist-slice] after sinking large reads back into warp foralls\n");

    // Re-sink tensor.empty / alloc_tensor ops that were incorrectly hoisted
    // out of scf.forall by the LICM preamble inside hoistRedundantVectorTransfers.
    //
    // The upstream LICM treats tensor.empty as Pure (no side effects) and
    // hoists it across ANY LoopLikeOpInterface boundary, including scf.forall
    // ops with GPU block/thread mapping.  Those tensors are destined to become
    // per-workgroup shared memory (block-level forall) or per-thread register
    // files (thread-level forall) during later GPU lowering.  Hoisting them to
    // function scope turns them into host-side allocations → segfault.
    //
    // Strategy: for each candidate op at function scope, walk every user and
    // collect its "forall ancestor chain" (outermost → innermost forall).
    // Then compute the Lowest Common Ancestor (LCA) across all users: the
    // deepest forall that encloses every user.  Sink the op to the front of
    // that LCA forall body.  This correctly handles:
    //
    //  • tensor.empty used as shared_outs of a THREAD forall
    //      → users' chains share the enclosing BLOCK forall as the LCA
    //      → sunk to BLOCK forall (becomes __shared__ per workgroup) ✓
    //
    //  • tensor.empty consumed directly inside a THREAD forall (register)
    //      → all users inside the same THREAD forall
    //      → sunk to THREAD forall (stays per-thread) ✓
    //
    //  • tensor.empty with users spanning BLOCK and THREAD forall levels
    //      → LCA is the BLOCK forall (outermost common ancestor)
    //      → sunk to BLOCK forall ✓
    //
    //  • tensor.empty with a user outside any forall at all
    //      → LCA is null (no common forall ancestor) → not sunk (safe) ✓
    //
    // NOTE: Step 3 (LICM on scf.for) runs after this and may hoist the
    // re-sunk ops out of scf.for loops, which is intentional and correct —
    // only scf.forall (GPU tiling boundaries) must be respected here.

    // Helper: build the chain [outermost forall, ..., innermost forall]
    // for the given op's parent region.
    auto getForallChain = [](Operation *op) -> SmallVector<scf::ForallOp> {
      SmallVector<scf::ForallOp> chain;
      Operation *cur = op->getParentOp();
      while (cur) {
        if (auto fa = dyn_cast<scf::ForallOp>(cur))
          chain.push_back(fa);
        cur = cur->getParentOp();
      }
      std::reverse(chain.begin(), chain.end()); // outermost first
      return chain;
    };

    funcOp.walk([&](Operation *op) {
      if (!isa<tensor::EmptyOp, bufferization::AllocTensorOp>(op))
        return;
      // Already inside a forall — hoistRedundantVectorTransfers didn't touch it.
      if (op->getParentOfType<scf::ForallOp>())
        return;

      // Collect the forall ancestor chain for every user.
      SmallVector<SmallVector<scf::ForallOp>> userChains;
      for (Operation *user : op->getUsers())
        userChains.push_back(getForallChain(user));

      if (userChains.empty())
        return;

      // Find the LCA: walk the common prefix of all chains.
      scf::ForallOp lca = nullptr;
      size_t depth = userChains[0].size();
      for (size_t i = 0; i < depth; ++i) {
        scf::ForallOp candidate = userChains[0][i];
        bool allMatch = llvm::all_of(userChains, [&](const auto &chain) {
          return i < chain.size() && chain[i] == candidate;
        });
        if (!allMatch)
          break;
        lca = candidate; // deepest common forall so far
      }

      // No common forall ancestor — at least one user is outside any forall;
      // leave the op where it is (function scope) to avoid breaking that user.
      if (!lca)
        return;

      op->moveBefore(&lca.getBody()->front());
    });
    // ------------------------------------------------------------------
    // Step 3: Standard MLIR loop-invariant code motion.
    // Any op whose ALL operands are defined outside the loop is moved
    // to just before the loop. Catches leftover invariant computations
    // that Steps 1-2 exposed.
    // ------------------------------------------------------------------
    funcOp.walk([](scf::ForOp forOp) {
      moveLoopInvariantCode(forOp);
    });

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-hoist-slice] after moveLoopInvariantCode\n");

    // ------------------------------------------------------------------
    // Step 3b: Hoist tensor.empty / alloc_tensor that are K-loop-invariant
    // but nested inside an inner scf.forall (e.g. warp or thread forall).
    //
    // The standard LICM in Step 3 only moves ops that are *direct* children
    // of the scf.for body and are loop-invariant.  In our pipeline, after
    // subgroup tiling the smem buffer allocations (tensor.empty for the A and
    // B promoted tiles) live inside the warp scf.forall, which itself is a
    // direct child of the K-loop scf.for.  LICM treats scf.forall as an
    // opaque region and never looks inside it for invariant ops.
    //
    // What we want:
    //   scf.for %k (K-loop)
    //     scf.forall (%warp) in (2,1)   ← warp forall
    //       %smem_A = tensor.empty() : tensor<32x64xf32>   ← allocate EVERY k
    //       %smem_B = tensor.empty() : tensor<64x16xf32>   ← allocate EVERY k
    //       ... copy A/B into smem ...
    //       ... mma ...
    //
    // After this step:
    //   scf.forall (%warp) in (2,1)        ← warp forall (unchanged)
    //     %smem_A = tensor.empty()          ← hoisted to before K-loop,
    //     %smem_B = tensor.empty()          ← inside the warp forall body
    //     scf.for %k (K-loop)
    //       ... copy A/B into smem ...
    //       ... mma ...
    //
    // Strategy: walk every scf.for.  For each tensor.empty / alloc_tensor
    // found anywhere inside it, check if the op is K-loop-invariant (none of
    // its operands use the for's IV or iter_args, transitively).  If so, find
    // the outermost scf.forall ancestor that is still *inside* the scf.for
    // (the "gateway" forall), and move the empty op to just before the for
    // loop, but at the same nesting depth as the gateway forall — i.e. inside
    // the gateway forall's enclosing block, just before the scf.for.
    // ------------------------------------------------------------------
    // Collect all (forOp, emptyOp) pairs first — no IR mutation during walk.
    // Key: the scf.for to hoist before. Value: the tensor.empty to move.
    SmallVector<std::pair<scf::ForOp, Operation *>> hoistWork;
    funcOp.walk([&](scf::ForOp forOp) {
      forOp.walk([&](Operation *op) {
        if (!isa<tensor::EmptyOp, bufferization::AllocTensorOp>(op))
          return;

        bool hoistable = true;
        for (Value operand : op->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            // Operand produced by an op inside forOp — not available outside.
            if (forOp->isProperAncestor(defOp)) {
              hoistable = false;
              break;
            }
          } else {
            // Operand is a block argument — check if its owner block lives
            // inside forOp (e.g. IV or iter_arg of a nested loop/forall).
            Block *parentBlock = cast<BlockArgument>(operand).getOwner();
            if (forOp->isProperAncestor(parentBlock->getParentOp())) {
              hoistable = false;
              break;
            }
          }
        }
        if (hoistable)
          hoistWork.emplace_back(forOp, op);
      });
    });

    // Now apply all moves outside any walk — safe to mutate IR.
    // Process innermost forOps first (walk order is post-order for the outer
    // walk, which gives innermost forOps first), so hoisting out of a nested
    // for before an outer one is handled naturally.
    for (auto &[forOp, op] : hoistWork) {
      // Skip if already moved out by a previous iteration (e.g. hoisted out
      // of an inner for by a prior entry, now no longer inside this forOp).
      if (!forOp->isProperAncestor(op))
        continue;
      // Operation::moveBefore works across blocks: it removes op from its
      // current block and inserts it before forOp in forOp's block.
      op->moveBefore(forOp);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-hoist-slice] after hoisting K-loop-invariant empties\n");

    // ------------------------------------------------------------------
    // Step 4: MLIR built-in subset hoisting.
    // Handles the standard case: extract_slice and insert_slice both use
    // the same loop iter_arg as their tensor. Works for both tensor and
    // vector slice ops via SubsetOpInterface.
    // ------------------------------------------------------------------
    funcOp.walk([&](scf::ForOp forOp) {
      hoistLoopInvariantSubsets(rewriter, forOp);
    });

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-hoist-slice] after hoistLoopInvariantSubsets\n");

    // ------------------------------------------------------------------
    // Step 5: Extended subset hoisting (mirrors IREE).
    // Handles the case where the insert_slice destination is a
    // loop-invariant tensor that is NOT the iter_arg. The MLIR built-in
    // in Step 4 does not cover this case.
    //
    // Works for both tensor.insert_slice and vector.insert_strided_slice
    // because both implement SubsetInsertionOpInterface.
    // ------------------------------------------------------------------
    funcOp.walk([&](scf::ForOp forOp) {
      hoistSubsetWithLoopInvariantTensor(rewriter, forOp);
    });

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-hoist-slice] after hoistSubsetWithLoopInvariantTensor\n");

    // ------------------------------------------------------------------
    // Step 6: Cleanup patterns — run greedily until fixed point.
    //
    //  Tensor identity folds:
    //    CastLikeExtractSliceFolder  — extract whole tensor → use source
    //    CastLikeInsertSliceFolder   — insert whole tensor  → use source
    //
    //  Vector identity folds (NEW — not in original pass):
    //    VectorExtractStridedFolder  — extract whole vector → use source
    //    VectorInsertStridedFolder   — insert whole vector  → use source
    //
    //  Standard canonicalization:
    //    scf::ForOp                  — remove dead iter_args
    //    TransferWriteOp             — simplify vector writes
    //    populateVectorTransferTensorSliceTransforms
    //                                — transfer + slice combos
    // ------------------------------------------------------------------
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Tensor no-op slice folding.
    patterns.add<CastLikeExtractSliceFolder>(ctx);
    patterns.add<CastLikeInsertSliceFolder>(ctx);

    // Vector no-op slice folding (handles vector dialect in addition to tensor).
    patterns.add<VectorExtractStridedFolder>(ctx);
    patterns.add<VectorInsertStridedFolder>(ctx);

    // Standard canonicalization.
    scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    vector::populateVectorInsertExtractStridedSliceTransforms(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "nova-gpu-hoist-slice: greedy pattern application failed");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[nova-hoist-slice] done\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUHoistVectorExtractInsertSlicePass() {
  return std::make_unique<NovaGPUHoistVectorExtractInsertSlicePass>();
}

void registerNovaGPUHoistVectorExtractInsertSlicePass() {
  PassRegistration<NovaGPUHoistVectorExtractInsertSlicePass>();
}

} // namespace mlir::nova