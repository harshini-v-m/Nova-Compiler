//===- NovaVectorLayoutAnalysis.cpp - Layout propagation analysis ---------===//
//
// Faithful port of IREE's VectorLayoutAnalysis.cpp.
// (iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp)
//
// Seeds layout constraints from nova_vector_ext.to_layout anchor ops, then
// runs a forward/backward fixpoint loop:
//
//   Forward  (value → users):
//     scf.for / scf.yield  → propagate to iter_arg, result
//     vector.yield (in mask) → propagate to mask result
//     elementwise ops       → all results get same layout as operand
//     vector.multi_reduction → project reduced dims away
//     vector.transpose       → permute layout
//     vector.contract (acc)  → result inherits acc layout
//     vector.contract (lhs/rhs) → if both known, infer result via
//                                  getRecombinedLayout; skip for MMA attrs
//     vector.gather          → result same layout
//     vector.transfer_write  → propagate to mask via inverse permutation
//     vector.shape_cast      → reshape layout
//
//   Backward (value → defining op):
//     BlockArg (scf.for)     → propagate to init and yielded value
//     elementwise            → all operands
//     nova_vector_ext.to_layout → operand
//     vector.multi_reduction → acc
//     vector.transpose       → source with inverse permutation
//     vector.broadcast       → source with projected layout
//     vector.contract        → acc only
//     vector.gather          → indices, mask, passthru
//     vector.transfer_read   → mask
//     vector.shape_cast      → source with reshaped layout
//
// Key difference from IREE:
//   - Uses nova::vec_ext namespace instead of IREE::VectorExt
//   - Uses nova_vector_ext.to_layout as anchor (same op, different namespace)
//   - Skips TransferGatherOp (IREE LinalgExt op not present in Nova)
//   - MMA marker attribute is "nova.gpu.mma" instead of "iree.gpu.mma"
//     (both are currently raw I32Attrs set by ConfigureTensorLayouts)
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaVectorLayoutAnalysis.h"

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "llvm/ADT/MapVector.h"
#include <queue>
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"

#define DEBUG_TYPE "nova-vector-layout-analysis"

using namespace mlir;
using namespace mlir::nova::vec_ext;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// LayoutInfo — layout map + propagation queues
//===----------------------------------------------------------------------===//

struct LayoutInfo {
  // ------------------------------------------------------------------
  // Core map: Value → layout
  // ------------------------------------------------------------------
  llvm::MapVector<Value, VectorLayoutInterface> layouts;

  // ------------------------------------------------------------------
  // Propagation queues
  // Forward  = value whose layout was just set; enqueue its users.
  // Backward = value whose layout was just set; enqueue its defOp.
  // ------------------------------------------------------------------
  std::queue<Value> forward;
  std::queue<Value> backward;

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  VectorLayoutInterface getLayout(Value val) const {
    auto it = layouts.find(val);
    return it != layouts.end() ? it->second : VectorLayoutInterface{};
  }
  bool hasLayout(Value val) const { return layouts.contains(val); }

  /// Set layout on `val` if not yet assigned; enqueue for propagation.
  /// Silently ignores non-shaped types (they have no layout).
  void setLayoutIfUnset(Value val, VectorLayoutInterface layout) {
    if (!isa<ShapedType>(val.getType()))
      return;
    if (hasLayout(val))
      return;
    layouts[val] = layout;
    forward.push(val);
    backward.push(val);
  }

  /// Set layout on the value of `operand`, creating a to_layout op to
  /// resolve conflicts when the value already has a different layout.
  /// Also clones constant-like ops so each layout gets its own def.
  /// Mirrors IREE's setLayoutOrClone().
  void setLayoutOrClone(OpOperand *operand, VectorLayoutInterface layout);

  // ------------------------------------------------------------------
  // Propagation
  // ------------------------------------------------------------------
  void propagateLayoutForward(Value val);
  void propagateLayoutBackward(Value val);
};

// ---------------------------------------------------------------------------
// setLayoutOrClone
// ---------------------------------------------------------------------------

void LayoutInfo::setLayoutOrClone(OpOperand *operand,
                                  VectorLayoutInterface layout) {
  if (!layout)
    return;
  Value val = operand->get();
  if (!isa<ShapedType>(val.getType()))
    return;

  OpBuilder b(operand->getOwner());

  // Always clone constant-like ops and duplicatable ops so each use-site can
  // carry its own layout without conflicting with other uses.
  if (Operation *defOp = val.getDefiningOp()) {
    bool isConstantLike = defOp->hasTrait<OpTrait::ConstantLike>();
    bool isDuplicatable =
        isa<vector::StepOp, vector::CreateMaskOp, vector::ConstantMaskOp>(
            defOp);
    if (isConstantLike || isDuplicatable) {
      b.setInsertionPoint(defOp);
      Operation *cloned = b.clone(*defOp);
      operand->set(cloned->getResult(0));
      layouts[cloned->getResult(0)] = layout;
      return;
    }
  }

  if (!hasLayout(val)) {
    layouts[val] = layout;
    forward.push(val);
    backward.push(val);
    return;
  }

  if (getLayout(val) != layout) {
    // Layout conflict: insert a to_layout op to convert to the required layout.
    b.setInsertionPoint(operand->getOwner());
    Value layouted = ToLayoutOp::create(b, val.getLoc(), val, layout);
    operand->set(layouted);
    layouts[layouted] = layout;
  }
}

// ---------------------------------------------------------------------------
// propagateLayoutForward — value → users
// ---------------------------------------------------------------------------

void LayoutInfo::propagateLayoutForward(Value val) {
  LLVM_DEBUG(llvm::dbgs() << "[nova-layout] forward: " << val << "\n");
  VectorLayoutInterface layout = getLayout(val);

  for (OpOperand &use : val.getUses()) {
    unsigned operandIdx = use.getOperandNumber();
    Operation *user = use.getOwner();

    // ------------------------------------------------------------------
    // scf.for: init operand → (iter_arg, result)
    // ------------------------------------------------------------------
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setLayoutIfUnset(arg, layout);
      setLayoutIfUnset(result, layout);
      continue;
    }

    // ------------------------------------------------------------------
    // scf.yield: yielded value → (iter_arg, result) of parent for/if
    // ------------------------------------------------------------------
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      Operation *parentOp = yieldOp->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        Value arg = forOp.getRegionIterArg(operandIdx);
        Value result = forOp->getResult(operandIdx);
        setLayoutIfUnset(arg, layout);
        setLayoutIfUnset(result, layout);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        // scf.if has no block arguments; results carry the layout.
        Value result = ifOp->getResult(operandIdx);
        setLayoutIfUnset(result, layout);
        continue;
      }
    }

    // ------------------------------------------------------------------
    // vector.yield inside vector.mask: yielded → mask result
    // ------------------------------------------------------------------
    if (auto yieldOp = dyn_cast<vector::YieldOp>(user)) {
      if (auto maskOp =
              dyn_cast<vector::MaskOp>(yieldOp->getParentOp())) {
        Value result = maskOp->getResult(operandIdx);
        setLayoutIfUnset(result, layout);
        continue;
      }
    }

    // ------------------------------------------------------------------
    // Elementwise: all results get the same layout as any operand
    // ------------------------------------------------------------------
    if (OpTrait::hasElementwiseMappableTraits(user)) {
      for (OpResult result : user->getOpResults())
        setLayoutIfUnset(result, layout);
      continue;
    }

    // ------------------------------------------------------------------
    // vector.multi_reduction
    //   source → project reduced dims → result layout
    //   acc    → result layout (same shape)
    // ------------------------------------------------------------------
    if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(user)) {
      if (multiReduce.getSource() == val) {
        // Propagate layout to mask if the reduction is inside a mask.
        if (auto maskOp =
                dyn_cast<vector::MaskOp>(multiReduce->getParentOp())) {
          setLayoutOrClone(&maskOp.getMaskMutable(), layout);
        }
        SmallVector<bool> reductionMask = multiReduce.getReductionMask();
        VectorLayoutInterface reduceLayout = layout.project(reductionMask);
        setLayoutIfUnset(multiReduce.getResult(), reduceLayout);
        continue;
      }
      if (multiReduce.getAcc() == val) {
        setLayoutIfUnset(multiReduce.getResult(), layout);
        continue;
      }
    }

    // ------------------------------------------------------------------
    // vector.transpose: result = layout.permute(perm)
    // ------------------------------------------------------------------
    if (auto transpose = dyn_cast<vector::TransposeOp>(user)) {
      if (transpose.getVector() == val) {
        setLayoutIfUnset(transpose.getResult(),
                         layout.permute(transpose.getPermutation()));
        continue;
      }
    }

    // ------------------------------------------------------------------
    // vector.contract
    //   acc → result inherits acc layout
    //   lhs/rhs → if both known, infer result via getRecombinedLayout
    //             skip if the contraction has an MMA attribute (fixed layout)
    // ------------------------------------------------------------------
    if (auto contract = dyn_cast<vector::ContractionOp>(user)) {
      if (contract.getAcc() == val) {
        setLayoutIfUnset(contract.getResult(), layout);
        continue;
      }
      if (contract.getLhs() == val || contract.getRhs() == val) {
        // MMA-attributed contractions have fixed layouts — don't infer.
        if (contract->hasAttr("nova.gpu.mma"))
          continue;

        // Propagate to mask if inside a vector.mask.
        if (auto maskOp =
                dyn_cast<vector::MaskOp>(contract->getParentOp())) {
          AffineMap map = contract.getMatchingIndexingMap(&use);
          if (map.isPermutation()) {
            setLayoutOrClone(&maskOp.getMaskMutable(),
                             layout.apply(inversePermutation(map)));
          }
        }

        // If the acc layout is already known, skip lhs/rhs recombination.
        // The acc-derived layout (from a to_layout anchor) takes priority
        // over the lhs/rhs recombined layout; using the recombined layout
        // here would cause a layout conflict that inserts stray to_layout ops.
        VectorLayoutInterface accLayout = getLayout(contract.getAcc());
        if (accLayout)
          continue;

        // If both lhs and rhs layouts are known, derive result layout.
        VectorLayoutInterface lhsLayout = getLayout(contract.getLhs());
        VectorLayoutInterface rhsLayout = getLayout(contract.getRhs());
        if (lhsLayout && rhsLayout) {
          AffineMap lhsMap = contract.getIndexingMapsArray()[0];
          AffineMap rhsMap = contract.getIndexingMapsArray()[1];
          AffineMap resMap = contract.getIndexingMapsArray()[2];
          VectorLayoutInterface resLayout = lhsLayout.getRecombinedLayout(
              {lhsLayout, rhsLayout}, {lhsMap, rhsMap}, resMap);

          // Validate thread count consistency before setting the result layout.
          // For MMA contractions, the recombined M×N thread count matches the
          // workgroup thread count (e.g., 16×16=256). For SIMT contractions,
          // the M-threads from LHS and N-threads from RHS are independently
          // chosen for their own operand dimensions; their product would exceed
          // the workgroup thread count (e.g., 8×128=1024 ≠ 256). Propagating
          // such an inconsistent layout causes DistributeTransferWrite to fail
          // (getNoOverlapCondition gets outer=0) and inserts a stray to_simd.
          // Skip propagation for the inconsistent (SIMT) case.
          bool consistent = true;
          if (auto resNested = dyn_cast<NestedLayoutAttr>(resLayout)) {
            if (auto lhsNested = dyn_cast<NestedLayoutAttr>(lhsLayout)) {
              int64_t resThreads = 1, lhsThreads = 1;
              for (int64_t t : resNested.getThreadTile()) resThreads *= t;
              for (int64_t t : lhsNested.getThreadTile()) lhsThreads *= t;
              if (resThreads != lhsThreads)
                consistent = false;
            }
          }
          if (consistent)
            setLayoutIfUnset(contract.getResult(), resLayout);
        }
        continue;
      }
    }

    // ------------------------------------------------------------------
    // vector.gather: result inherits source layout
    // ------------------------------------------------------------------
    if (auto gather = dyn_cast<vector::GatherOp>(user)) {
      setLayoutIfUnset(gather.getResult(), layout);
      continue;
    }

    // ------------------------------------------------------------------
    // vector.transfer_write: propagate to mask via inverse permutation
    // ------------------------------------------------------------------
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (!write.getMask())
        continue;
      OpOperand &mask = write.getMaskMutable()[0];
      AffineMap maskMap =
          inversePermutation(compressUnusedDims(write.getPermutationMap()));
      setLayoutOrClone(&mask, layout.apply(maskMap));
      continue;
    }

    // ------------------------------------------------------------------
    // vector.shape_cast: result = layout.reshape(resultShape)
    // ------------------------------------------------------------------
    if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(user)) {
      setLayoutIfUnset(
          shapeCast.getResult(),
          layout.reshape(shapeCast.getResultVectorType().getShape()));
      continue;
    }
  }
}

// ---------------------------------------------------------------------------
// propagateLayoutBackward — value → defining op
// ---------------------------------------------------------------------------

void LayoutInfo::propagateLayoutBackward(Value val) {
  LLVM_DEBUG(llvm::dbgs() << "[nova-layout] backward: " << val << "\n");
  VectorLayoutInterface layout = getLayout(val);

  // ------------------------------------------------------------------
  // Block argument of scf.for: propagate to init operand and yield value
  // ------------------------------------------------------------------
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Operation *parent = val.getParentBlock()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      OpOperand *yielded = forOp.getTiedLoopYieldedValue(blockArg);
      OpOperand *init    = forOp.getTiedLoopInit(blockArg);
      setLayoutOrClone(yielded, layout);
      setLayoutOrClone(init, layout);
    }
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return;

  // ------------------------------------------------------------------
  // Elementwise: all operands get the same layout as the result
  // ------------------------------------------------------------------
  if (OpTrait::hasElementwiseMappableTraits(defOp)) {
    for (OpOperand &operand : defOp->getOpOperands())
      setLayoutOrClone(&operand, layout);
    return;
  }

  // ------------------------------------------------------------------
  // nova_vector_ext.to_layout: operand gets the output layout
  // ------------------------------------------------------------------
  if (auto toLayout = dyn_cast<ToLayoutOp>(defOp)) {
    setLayoutOrClone(&toLayout.getInputMutable(), layout);
    return;
  }

  // ------------------------------------------------------------------
  // vector.multi_reduction: acc operand
  // ------------------------------------------------------------------
  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(defOp)) {
    setLayoutOrClone(&multiReduce.getAccMutable(), layout);
    return;
  }

  // ------------------------------------------------------------------
  // vector.transpose: source = layout.permute(inverse permutation)
  // ------------------------------------------------------------------
  if (auto transpose = dyn_cast<vector::TransposeOp>(defOp)) {
    setLayoutOrClone(
        &transpose.getVectorMutable(),
        layout.permute(invertPermutationVector(transpose.getPermutation())));
    return;
  }

  // ------------------------------------------------------------------
  // vector.broadcast: source = layout projected by broadcasted dims
  // Broadcasted dims are the leading dims added by broadcast.
  // ------------------------------------------------------------------
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(defOp)) {
    assert(broadcast.computeBroadcastedUnitDims().empty() &&
           "Stretching in broadcasting not supported yet.");
    if (!isa<VectorType>(broadcast.getSourceType()))
      return;
    int64_t numBroadcastedDims =
        broadcast.getResultVectorType().getRank() -
        cast<VectorType>(broadcast.getSourceType()).getRank();
    SmallVector<bool> reductionMask(layout.getRank(), false);
    // Leading dims are the broadcasted ones.
    std::fill(reductionMask.begin(),
              reductionMask.begin() + numBroadcastedDims, true);
    setLayoutOrClone(&broadcast.getSourceMutable(),
                     layout.project(reductionMask));
    return;
  }

  // ------------------------------------------------------------------
  // vector.contract: only propagate to acc; lhs/rhs NYI
  // ------------------------------------------------------------------
  if (auto contract = dyn_cast<vector::ContractionOp>(defOp)) {
    setLayoutOrClone(&contract.getAccMutable(), layout);
    return;
  }

  // ------------------------------------------------------------------
  // vector.gather: propagate to indices, mask, passthru
  // ------------------------------------------------------------------
  if (auto gather = dyn_cast<vector::GatherOp>(defOp)) {
    // getIndices() is a variadic static-index operand (not the index vector).
    // Distribute index_vec, mask, and pass_thru.
    setLayoutOrClone(&gather.getIndexVecMutable(), layout);
    setLayoutOrClone(&gather.getMaskMutable(), layout);
    setLayoutOrClone(&gather.getPassThruMutable(), layout);
    return;
  }

  // ------------------------------------------------------------------
  // vector.transfer_read: propagate to mask via inverse permutation
  // ------------------------------------------------------------------
  if (auto read = dyn_cast<vector::TransferReadOp>(defOp)) {
    if (!read.getMask())
      return;
    OpOperand &mask = read.getMaskMutable()[0];
    AffineMap maskMap =
        inversePermutation(compressUnusedDims(read.getPermutationMap()));
    setLayoutOrClone(&mask, layout.apply(maskMap));
    return;
  }

  // ------------------------------------------------------------------
  // vector.shape_cast: source = layout.reshape(sourceShape)
  // ------------------------------------------------------------------
  if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(defOp)) {
    setLayoutOrClone(
        &shapeCast.getSourceMutable(),
        layout.reshape(shapeCast.getSourceVectorType().getShape()));
    return;
  }
}

// ---------------------------------------------------------------------------
// propagateVectorLayoutInfo — public entry point
// ---------------------------------------------------------------------------

LogicalResult propagateVectorLayoutInfo(
    Operation *root,
    llvm::MapVector<Value, VectorLayoutInterface> &layouts) {

  LayoutInfo info;

  // Seed from all nova_vector_ext.to_layout ops in the IR.
  // These are the anchors placed by ConfigureTensorLayouts.
  root->walk([&](ToLayoutOp toLayout) {
    LLVM_DEBUG(llvm::dbgs() << "[nova-layout] anchor: " << toLayout << "\n");
    info.setLayoutIfUnset(toLayout.getResult(), toLayout.getLayout());
  });

  // Fixpoint loop: prefer forward propagation; only do backward when the
  // forward queue is empty. Mirrors IREE's priority ordering.
  while (!info.forward.empty() || !info.backward.empty()) {
    if (!info.forward.empty()) {
      Value val = info.forward.front();
      info.forward.pop();
      info.propagateLayoutForward(val);
    } else {
      Value val = info.backward.front();
      info.backward.pop();
      info.propagateLayoutBackward(val);
    }
  }

  layouts = std::move(info.layouts);
  return success();
}

} // namespace mlir::nova
