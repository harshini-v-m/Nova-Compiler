//===- NovaGPUCombineValueSemanticBarriers.cpp - Merge value barriers ------===//
//
// Combines multiple nova.value_barrier ops in the same block into a single
// one, reducing the number of gpu.barrier instructions emitted after
// bufferization.
//
// After GPUVectorAllocPass each {shared_memory_conversion} to_layout pair
// produces two value_barriers:
//
//   %wb_a = nova.value_barrier %vec_a   : vector<128x16xf32>   ← write A
//   ...alloc + write A...
//   %rb_a = nova.value_barrier %ten_a   : tensor<128x16xf32, workgroup>
//
//   %wb_b = nova.value_barrier %vec_b   : vector<16x64xf32>    ← write B
//   ...alloc + write B...
//   %rb_b = nova.value_barrier %ten_b   : tensor<16x64xf32, workgroup>
//
// After this pass:
//   %wb   = nova.value_barrier %vec_a, %vec_b   ← one combined write barrier
//   %rb   = nova.value_barrier %ten_a, %ten_b   ← one combined read  barrier
//
// The two write barriers can be combined (both vector-type); the two read
// barriers can be combined (both tensor-type). Write and read barriers are
// NOT merged together — that would eliminate the synchronization.
//
// Algorithm mirrors IREE's GPUCombineValueSemanticBarriersPass
// (iree/compiler/Codegen/Common/GPU/GPUCombineValueSemanticBarriers.cpp).
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// Slice-movement helpers (direct port from IREE)
//===----------------------------------------------------------------------===//

/// Move all ops in |slice| (topologically sorted) to just before |barrier|.
static void moveBackwardSliceBeforeBarrier(RewriterBase &rewriter,
                                           SetVector<Operation *> &slice,
                                           Operation *barrier) {
  slice = topologicalSort(slice);
  for (Operation *op : slice)
    rewriter.moveOpBefore(op, barrier);
}

/// Move all ops in |slice| (topologically sorted, reversed) to just after
/// |barrier|.
static void moveForwardSliceAfterBarrier(RewriterBase &rewriter,
                                         SetVector<Operation *> &slice,
                                         Operation *barrier) {
  slice = topologicalSort(slice);
  for (Operation *op : llvm::reverse(slice))
    rewriter.moveOpAfter(op, barrier);
}

/// Given two value_barrier ops in the same block, rearrange the IR so that
/// all producers of their inputs appear before the leading barrier and all
/// consumers of their outputs appear after the trailing barrier.
/// On entry barrierA/B may be in any order; on exit barrierA is before
/// barrierB. Returns failure if the two barriers form a dependency chain.
static LogicalResult enforceBarrierOrdering(RewriterBase &rewriter,
                                            Operation *&barrierA,
                                            Operation *&barrierB) {
  if (barrierB->isBeforeInBlock(barrierA))
    std::swap(barrierA, barrierB);

  assert(barrierA->getBlock() == barrierB->getBlock());
  Block *block = barrierA->getBlock();

  // ---- Backward slice: move producers before barrierA ----
  auto backwardFilter = [&](Operation *candidate) -> bool {
    return candidate->getBlock() == block &&
           candidate != block->getTerminator() &&
           !candidate->isBeforeInBlock(barrierA);
  };

  BackwardSliceOptions bOpts;
  bOpts.omitUsesFromAbove = false;
  bOpts.filter = backwardFilter;

  SetVector<Operation *> bwSliceA, bwSliceB;
  (void)getBackwardSlice(barrierA, &bwSliceA, bOpts);
  (void)getBackwardSlice(barrierB, &bwSliceB, bOpts);
  bwSliceA.insert(bwSliceB.begin(), bwSliceB.end());

  // If barrierA is in its own backward slice the barriers are chained.
  if (bwSliceA.contains(barrierA))
    return failure();

  moveBackwardSliceBeforeBarrier(rewriter, bwSliceA, barrierA);

  // ---- Forward slice: move consumers after barrierB ----
  auto forwardFilter = [&](Operation *candidate) -> bool {
    return candidate->getBlock() == block &&
           candidate != block->getTerminator() &&
           !barrierB->isBeforeInBlock(candidate);
  };

  ForwardSliceOptions fOpts;
  fOpts.filter = forwardFilter;

  SetVector<Operation *> fwSliceA, fwSliceB;
  getForwardSlice(barrierA, &fwSliceA, fOpts);
  getForwardSlice(barrierB, &fwSliceB, fOpts);
  fwSliceA.insert(fwSliceB.begin(), fwSliceB.end());

  if (fwSliceA.contains(barrierA))
    return failure();

  moveForwardSliceAfterBarrier(rewriter, fwSliceA, barrierB);

  return success();
}

//===----------------------------------------------------------------------===//
// Combine a pair of nova.value_barrier ops
//===----------------------------------------------------------------------===//

/// Returns true if all inputs of |op| are tensor-typed.
static bool hasTensorSemantics(nova::ValueBarrierOp op) {
  return llvm::all_of(op.getInputs(),
                      [](Value v) { return isa<RankedTensorType>(v.getType()); });
}

static FailureOr<nova::ValueBarrierOp>
combineValueBarrierPair(RewriterBase &rewriter, nova::ValueBarrierOp barrierA,
                        nova::ValueBarrierOp barrierB) {
  // Only combine if both are the same kind (both tensor or both vector).
  if (hasTensorSemantics(barrierA) != hasTensorSemantics(barrierB))
    return failure();

  Operation *opA = barrierA, *opB = barrierB;
  if (failed(enforceBarrierOrdering(rewriter, opA, opB)))
    return failure();
  barrierA = cast<nova::ValueBarrierOp>(opA);
  barrierB = cast<nova::ValueBarrierOp>(opB);

  // Insert the combined barrier after barrierB (always sink barriers).
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(barrierB);

  SmallVector<Value> operands(barrierA.getOperands());
  operands.append(barrierB.getOperands().begin(), barrierB.getOperands().end());

  auto combined = nova::ValueBarrierOp::create(rewriter, barrierB.getLoc(),
                                               operands);

  int nA = barrierA.getNumOperands();
  int nB = barrierB.getNumOperands();
  rewriter.replaceOp(barrierA, combined->getResults().slice(0, nA));
  rewriter.replaceOp(barrierB, combined->getResults().slice(nA, nB));

  return combined;
}

//===----------------------------------------------------------------------===//
// Driver: O(n²) pairwise combination over all barriers in a block
//===----------------------------------------------------------------------===//

static void combineBarriersInBlock(RewriterBase &rewriter, Block *block) {
  SmallVector<nova::ValueBarrierOp> barriers;
  for (Operation &op : block->getOperations()) {
    if (auto barrier = dyn_cast<nova::ValueBarrierOp>(op))
      barriers.push_back(barrier);
  }

  int n = barriers.size();
  for (int i = 0; i < n; ++i) {
    if (!barriers[i])
      continue;
    for (int j = i + 1; j < n; ++j) {
      if (!barriers[j])
        continue;
      auto combined = combineValueBarrierPair(rewriter, barriers[i], barriers[j]);
      if (succeeded(combined)) {
        barriers[i] = combined.value();
        barriers[j] = nullptr;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUCombineValueSemanticBarriersPass
    : public PassWrapper<NovaGPUCombineValueSemanticBarriersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUCombineValueSemanticBarriersPass)

  StringRef getArgument() const override {
    return "nova-gpu-combine-value-semantic-barriers";
  }
  StringRef getDescription() const override {
    return "Combine multiple nova.value_barrier ops in the same block into "
           "one, reducing the number of gpu.barrier instructions emitted after "
           "bufferization.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nova::NovaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect all blocks that contain at least one value_barrier.
    SmallVector<Block *> blocks;
    funcOp.walk([&](Block *block) {
      if (llvm::any_of(block->getOperations(), [](Operation &op) {
            return isa<nova::ValueBarrierOp>(op);
          }))
        blocks.push_back(block);
    });

    IRRewriter rewriter(&getContext());
    for (Block *block : blocks)
      combineBarriersInBlock(rewriter, block);
  }
};

} // namespace

std::unique_ptr<Pass> createNovaGPUCombineValueSemanticBarriersPass() {
  return std::make_unique<NovaGPUCombineValueSemanticBarriersPass>();
}

void registerNovaGPUCombineValueSemanticBarriersPass() {
  PassRegistration<NovaGPUCombineValueSemanticBarriersPass>();
}

} // namespace mlir::nova
