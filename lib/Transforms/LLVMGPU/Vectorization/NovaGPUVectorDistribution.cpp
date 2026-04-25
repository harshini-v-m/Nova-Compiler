//===- NovaGPUVectorDistribution.cpp - Vector distribution engine ---------===//
//
// Port of IREE's GPUVectorDistribution.cpp.
// (iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.cpp)
//
// See NovaGPUVectorDistribution.h for the full description.
//
// Key differences from IREE:
//   - Uses nova::vec_ext namespace instead of IREE::VectorExt
//   - Calls nova::propagateVectorLayoutInfo (NovaVectorLayoutAnalysis)
//   - ToSIMDOp / ToSIMTOp are nova_vector_ext.to_simd / to_simt
//   - NestedLayoutAttr is in nova::vec_ext
//   - Removed TransferGatherOp / MapScatter (not in Nova)
//
//===----------------------------------------------------------------------===//

#include "NovaGPUVectorDistribution.h"

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <deque>

#define DEBUG_TYPE "nova-gpu-vector-distribution"

using namespace mlir::nova::vec_ext;

namespace mlir::nova {

// Attribute names used to store per-op distribution signatures.
constexpr StringLiteral kLayoutStorageAttr = "__nova_layout_storage";
constexpr StringLiteral kRedistributeAttr  = "__nova_layout_redistribute";

//===----------------------------------------------------------------------===//
// Signature helpers (file-local)
//===----------------------------------------------------------------------===//

/// Attach a distribution signature to `op`.
/// The attribute stores [[operandLayouts...], [resultLayouts...]].
/// Vector operands/results carry their VectorLayoutInterface.
/// Non-vector operands/results carry a UnitAttr placeholder.
static LogicalResult
setOpSignature(Operation *op,
               const llvm::MapVector<Value, VectorLayoutInterface> &layouts,
               const VectorLayoutOptions &options) {
  SmallVector<Attribute> operands;
  SmallVector<Attribute> results;

  for (Value operand : op->getOperands()) {
    if (auto vecVal = dyn_cast<VectorValue>(operand)) {
      if (auto layout = layouts.lookup(vecVal)) {
        operands.push_back(layout);
        continue;
      }
      if (auto layout = options.getDefaultLayout(vecVal.getType())) {
        operands.push_back(layout);
        continue;
      }
      return failure(); // vector with no layout → skip this op
    }
    operands.push_back(UnitAttr::get(op->getContext()));
  }

  for (Value result : op->getResults()) {
    if (auto vecVal = dyn_cast<VectorValue>(result)) {
      if (auto layout = layouts.lookup(vecVal)) {
        results.push_back(layout);
        continue;
      }
      if (auto layout = options.getDefaultLayout(vecVal.getType())) {
        results.push_back(layout);
        continue;
      }
      return failure();
    }
    results.push_back(UnitAttr::get(op->getContext()));
  }

  Attribute sig[] = {ArrayAttr::get(op->getContext(), operands),
                     ArrayAttr::get(op->getContext(), results)};
  op->setAttr(kLayoutStorageAttr, ArrayAttr::get(op->getContext(), sig));
  return success();
}

static bool hasOpSignature(Operation *op) {
  return op->hasAttrOfType<ArrayAttr>(kLayoutStorageAttr);
}

static DistributionSignature getOpSignatureFromAttr(Operation *op) {
  ArrayAttr sigAttr = op->getAttrOfType<ArrayAttr>(kLayoutStorageAttr);
  assert(sigAttr && sigAttr.size() == 2 && "malformed signature attr");

  auto operandsAttr = cast<ArrayAttr>(sigAttr[0]);
  auto resultsAttr  = cast<ArrayAttr>(sigAttr[1]);
  assert(operandsAttr.size() == op->getNumOperands());
  assert(resultsAttr.size()  == op->getNumResults());

  DistributionSignature sig;

  auto add = [&](Value value, Attribute layout) {
    if (isa<UnitAttr>(layout)) {
      assert(!isa<VectorValue>(value));
      return;
    }
    assert(isa<VectorValue>(value));
    sig[cast<VectorValue>(value)] = cast<VectorLayoutInterface>(layout);
  };

  for (auto [v, l] : llvm::zip_equal(op->getOperands(), operandsAttr))
    add(v, l);
  for (auto [v, l] : llvm::zip_equal(op->getResults(), resultsAttr))
    add(v, l);

  return sig;
}

//===----------------------------------------------------------------------===//
// DistributionPattern method implementations
//===----------------------------------------------------------------------===//

VectorValue DistributionPattern::getDistributed(RewriterBase &rewriter,
                                                 VectorValue value,
                                                 VectorLayoutInterface layout) const {
  // Unwrap a to_simd — its source is already in distributed form.
  if (auto toSIMD = value.getDefiningOp<vec_ext::ToSIMDOp>())
    return cast<VectorValue>(toSIMD.getInput());

  // Insert to_simt to slice the SIMD vector to per-thread size.
  SmallVector<int64_t> distShape = layout.getDistributedShape();
  VectorType distType =
      VectorType::get(distShape, value.getType().getElementType());
  auto toSIMT =
      vec_ext::ToSIMTOp::create(rewriter, value.getLoc(), distType, value);
  return toSIMT.getResult();
}

SmallVector<Value>
DistributionPattern::getOpDistributedReplacements(RewriterBase &rewriter,
                                                   Operation *op,
                                                   ValueRange values) const {
  SmallVector<Value> replacements;
  for (auto [opResult, replacement] :
       llvm::zip_equal(op->getOpResults(), values)) {
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<VectorValue>(opResult);
      rewriter.setInsertionPointAfterValue(oldResult);
      Value toSIMD = vec_ext::ToSIMDOp::create(
          rewriter, oldResult.getLoc(), oldResult.getType(), replacement);
      replacement = toSIMD;
    }
    replacements.push_back(replacement);
  }
  return replacements;
}

void DistributionPattern::replaceOpWithDistributedValues(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  SmallVector<Value> replacements =
      getOpDistributedReplacements(rewriter, op, values);
  rewriter.replaceOp(op, replacements);
}

std::optional<DistributionSignature>
DistributionPattern::getOpSignature(Operation *op) const {
  if (!hasOpSignature(op))
    return std::nullopt;
  return getOpSignatureFromAttr(op);
}

void DistributionPattern::setSignatureForRedistribution(
    RewriterBase &rewriter, Operation *op,
    ArrayRef<VectorLayoutInterface> inputLayouts,
    ArrayRef<VectorLayoutInterface> outputLayouts) const {

  auto unitAttr = UnitAttr::get(rewriter.getContext());
  SmallVector<Attribute> inputAttrs(op->getNumOperands(), unitAttr);
  SmallVector<Attribute> outputAttrs(op->getNumResults(), unitAttr);

  auto isVec = [](Value v) { return isa<VectorType>(v.getType()); };
  assert(llvm::count_if(op->getOperands(), isVec) ==
         (int64_t)inputLayouts.size());
  assert(llvm::count_if(op->getResults(), isVec) ==
         (int64_t)outputLayouts.size());

  int vecIn = 0;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands()))
    if (isVec(operand))
      inputAttrs[idx] = inputLayouts[vecIn++];

  int vecOut = 0;
  for (auto [idx, result] : llvm::enumerate(op->getResults()))
    if (isVec(result))
      outputAttrs[idx] = outputLayouts[vecOut++];

  Attribute sig[] = {ArrayAttr::get(rewriter.getContext(), inputAttrs),
                     ArrayAttr::get(rewriter.getContext(), outputAttrs)};
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(kLayoutStorageAttr,
                ArrayAttr::get(rewriter.getContext(), sig));
    op->setAttr(kRedistributeAttr, unitAttr);
  });
}

LogicalResult
DistributionPattern::replaceParentMask(PatternRewriter &rewriter,
                                        vector::MaskOp maskOp) const {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(maskOp);

  auto sigMask = getOpSignature(maskOp);
  if (!sigMask)
    return rewriter.notifyMatchFailure(maskOp, "mask has no signature");

  SmallVector<Value> returns =
      maskOp.getBody()->getTerminator()->getOperands();
  for (auto [idx, ret] : llvm::enumerate(returns)) {
    if (auto vecRet = dyn_cast<VectorValue>(ret)) {
      auto maskRet = cast<VectorValue>(maskOp.getResult(idx));
      auto layout =
          dyn_cast<NestedLayoutAttr>(sigMask.value()[maskRet]);
      if (!layout)
        return rewriter.notifyMatchFailure(maskOp,
                                           "layout must be NestedLayoutAttr");
      ret = getDistributed(rewriter, vecRet, layout);
    }
  }
  rewriter.eraseOp(maskOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(maskOp.getBody(), maskOp);
  replaceOpWithDistributedValues(rewriter, maskOp, returns);
  return success();
}

//===----------------------------------------------------------------------===//
// VectorDistributionRewriter / VectorDistributionListener
//===----------------------------------------------------------------------===//

struct VectorDistributionRewriter : PatternRewriter {
  VectorDistributionRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

struct VectorDistributionListener : RewriterBase::Listener {
  bool hasOps() const { return !toBeDistributed.empty(); }
  void clear() { toBeDistributed.clear(); }
  const std::deque<Operation *> &ops() const { return toBeDistributed; }

  void notifyOperationModified(Operation *op) override {
    if (op->hasAttr(kRedistributeAttr) &&
        op->hasAttrOfType<ArrayAttr>(kLayoutStorageAttr)) {
      op->removeAttr(kRedistributeAttr);
      toBeDistributed.push_back(op);
    }
  }

private:
  std::deque<Operation *> toBeDistributed;
};

//===----------------------------------------------------------------------===//
// applyVectorDistribution — worklist-driven pattern application
//===----------------------------------------------------------------------===//

static void applyVectorDistribution(Operation *root,
                                    const FrozenRewritePatternSet &patterns) {
  VectorDistributionRewriter rewriter(root->getContext());
  VectorDistributionListener listener;
  rewriter.setListener(&listener);

  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Seed worklist: all ops with a signature except mask/yield
  // (masks are distributed when their body is distributed).
  std::deque<Operation *> worklist;
  root->walk([&](Operation *op) {
    if (hasOpSignature(op) && !isa<vector::MaskOp, vector::YieldOp>(op))
      worklist.push_back(op);
  });

  LLVM_DEBUG(llvm::dbgs() << "[nova-vdist] initial worklist size: "
                           << worklist.size() << "\n");

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (!op)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "[nova-vdist] distributing: ";
               op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
               llvm::dbgs() << "\n");

    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs() << "  -> no pattern matched\n");
      continue;
    }

    // Enqueue newly emitted ops marked for redistribution.
    if (listener.hasOps()) {
      worklist.insert(worklist.end(), listener.ops().begin(),
                      listener.ops().end());
      listener.clear();
    }
  }
}

//===----------------------------------------------------------------------===//
// distributeVectorOps — public entry point
//===----------------------------------------------------------------------===//

LogicalResult distributeVectorOps(Operation *root,
                                  RewritePatternSet &distributionPatterns,
                                  VectorLayoutOptions &options) {
  // 1. Layout analysis: propagate NestedLayoutAttr from to_layout anchors.
  LLVM_DEBUG(llvm::dbgs() << "[nova-vdist] running layout analysis\n");
  llvm::MapVector<Value, VectorLayoutInterface> layouts;
  if (failed(propagateVectorLayoutInfo(root, layouts))) {
    root->emitError("nova vector layout analysis failed");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[nova-vdist] layout analysis done, "
                           << layouts.size() << " values\n");

  // 2. Set distribution signatures on all ops whose vector values have layouts.
  root->walk([&](Operation *op) {
    (void)setOpSignature(op, layouts, options);
  });

  // 3. Apply distribution patterns via worklist.
  FrozenRewritePatternSet frozen(std::move(distributionPatterns));
  applyVectorDistribution(root, frozen);

  // 4. Canonicalize to_simd(to_simt(x)) → x and vice versa.
  {
    RewritePatternSet cleanup(root->getContext());
    vec_ext::ToSIMDOp::getCanonicalizationPatterns(cleanup, root->getContext());
    vec_ext::ToSIMTOp::getCanonicalizationPatterns(cleanup, root->getContext());
    if (failed(applyPatternsGreedily(root, std::move(cleanup))))
      return failure();
  }

  // 5. Remove signature attributes — no longer needed after distribution.
  root->walk([](Operation *op) {
    op->removeDiscardableAttr(kLayoutStorageAttr);
  });

  // 6. Verify: no stray to_simd/to_simt should remain with non-conversion users.
  if (options.verifyConversion()) {
    WalkResult hasStray = root->walk([](Operation *op) {
      if (isa<vec_ext::ToSIMDOp, vec_ext::ToSIMTOp>(op)) {
        for (Operation *user : op->getUsers()) {
          if (!isa<vec_ext::ToSIMDOp, vec_ext::ToSIMTOp>(user)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[nova-vdist] stray conversion: " << *op << "\n"
                       << "  user: " << *user << "\n");
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (hasStray.wasInterrupted())
      return failure();
  }

  return success();
}

} // namespace mlir::nova
