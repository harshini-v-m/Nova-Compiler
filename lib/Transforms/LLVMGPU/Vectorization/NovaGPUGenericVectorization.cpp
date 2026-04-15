//===- NovaGenericVectorization.cpp ---------------------------------------===//
//
// Vectorize linalg ops to vector.* for the Nova MMA pipeline.
//
// Two categories of ops:
//
//   A. MMA contraction ops (linalg.batch_matmul with nova.layout_* attrs):
//        Parks nova.layout_0/1/2 + lowering_config on the enclosing
//        scf.forall before linalg::vectorize erases the op, then transfers
//        them to the vector.contract after Phase 2 canonicalization.
//
//        Why park on the forall and not intercept at creation?
//        linalg::vectorize for batch_matmul emits an outer-product /
//        multi-reduce decomposition — NOT a vector.contract directly.
//        The vector.contract only appears after Phase 2's
//        populateVectorReductionToContractPatterns, which runs through its
//        own internal rewriter.  A RewriterBase::Listener on the outer
//        IRRewriter never fires for ops created by the pattern driver.
//        The enclosing scf.forall is stable across both phases and is the
//        only reliable anchor to correlate a contract back to its source.
//
//   B. Static-shape non-MMA ops (linalg.copy, tensor.pad, etc.):
//        linalg::vectorize(op, {})  →  vector.transfer_read/write
//        No masking needed — shapes are guaranteed static after padding.
//
//   C. Dynamic-shape ops (GELU epilogue on tensor<1x64x?xf32>):
//        inferNovaVectorSizes() → ValueBounds UB analysis → vectorSizes
//        linalg::vectorize(op, vectorSizes) → vector.mask { ... }
//        Mask eliminated in Phase 3 if statically all-true.
//
// Pipeline position:
//   NovaGPUConfigureTensorLayouts   → attaches nova.layout_* to matmul
//           ↓
//   NovaGenericVectorization        ← THIS PASS
//           ↓
//   NovaGPUUnrollToIntrinsics       → reads batch_counts from layout on contract
//           ↓
//   NovaGPUVectorDistribute         → reads sg/thread/elem from layout on contract
//
// Phase 1: Vectorize all LinalgOps + tensor.pad bottom-up.
//          MMA ops → park attrs on parent scf.forall, then plain vectorize.
//          Static ops → plain vectorize.
//          Dynamic ops → masked vectorize via ValueBounds.
// Phase 2: Canonicalize to vector.contract
//          (TransferPermMap + Sink + ReductionToContract).
// Phase 2.5: Transfer parked nova.layout_* from scf.forall → vector.contract.
// Phase 3: Eliminate always-true vector.mask ops.
// Phase 4: Canonicalize mask predicates.
// Phase 5: Lower vector.mask { transfer } to predicated form.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaVectorSizeUtils.h"
#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// §1  Constants
//===----------------------------------------------------------------------===//

static constexpr int64_t kMaxVectorSize = 4096LL * 4096LL;

static constexpr StringLiteral kLayout0        = "nova.layout_0";
static constexpr StringLiteral kLayout1        = "nova.layout_1";
static constexpr StringLiteral kLayout2        = "nova.layout_2";
static constexpr StringLiteral kLoweringConfig = "lowering_config";

// Temporary attributes parked on the parent scf.forall during Phase 1.
// Cleared during Phase 2.5 once transferred to vector.contract.
static constexpr StringLiteral kPendingLayout0 = "nova._pl0";
static constexpr StringLiteral kPendingLayout1 = "nova._pl1";
static constexpr StringLiteral kPendingLayout2 = "nova._pl2";
static constexpr StringLiteral kPendingConfig  = "nova._pc";

//===----------------------------------------------------------------------===//
// §2  Pure helpers
//===----------------------------------------------------------------------===//

static bool hasStaticShape(linalg::LinalgOp op) {
  for (OpOperand &operand : op->getOpOperands()) {
    auto shapedTy = dyn_cast<ShapedType>(operand.get().getType());
    if (shapedTy && !shapedTy.hasStaticShape())
      return false;
  }
  return true;
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp op) {
  int64_t maxFlat = 1;
  for (OpOperand &operand : op->getOpOperands()) {
    auto ty = dyn_cast<ShapedType>(operand.get().getType());
    if (!ty)
      continue;
    if (!ty.hasStaticShape())
      return failure();
    maxFlat = std::max(maxFlat, ty.getNumElements());
  }
  return success(maxFlat < kMaxVectorSize);
}

static bool isMmaContractionOp(Operation *op) {
  return op->hasAttr(kLayout0);
}

//===----------------------------------------------------------------------===//
// §3  MMA op vectorization — forall-parking layout transfer
//
// linalg::vectorize on batch_matmul emits an outer-product decomposition
// (vector.outerproduct / vector.multi_reduce), NOT a vector.contract.
// The contract only materialises in Phase 2 via
// populateVectorReductionToContractPatterns.
//
// Strategy:
//   1. Save nova.layout_* + lowering_config on the enclosing scf.forall
//      (stable through Phase 2) under temporary "nova._pl*" attr names.
//   2. Call linalg::vectorize normally.
//   3. After Phase 2, walk all vector.contract ops: find the nearest
//      ancestor scf.forall that carries pending attrs, attach them, and
//      remove the temporaries.
//===----------------------------------------------------------------------===//

static LogicalResult vectorizeMmaOp(IRRewriter &rewriter, Operation *op) {
  // Park layout attrs on the parent scf.forall before linalg::vectorize
  // erases op.  The forall is stable across Phase 2 while the linalg op
  // is not.
  auto parentForall = op->getParentOfType<scf::ForallOp>();
  if (!parentForall) {
    op->emitError("nova-generic-vectorization: MMA batch_matmul is not "
                  "enclosed in an scf.forall — cannot park layout attrs");
    return failure();
  }
  if (auto a = op->getAttr(kLayout0))        parentForall->setAttr(kPendingLayout0, a);
  if (auto a = op->getAttr(kLayout1))        parentForall->setAttr(kPendingLayout1, a);
  if (auto a = op->getAttr(kLayout2))        parentForall->setAttr(kPendingLayout2, a);
  if (auto a = op->getAttr(kLoweringConfig)) parentForall->setAttr(kPendingConfig,  a);

  FailureOr<linalg::VectorizationResult> result =
      linalg::vectorize(rewriter, op,
                        /*inputVectorSizes=*/{},
                        /*inputScalableVecDims=*/{},
                        /*vectorizeNDExtract=*/true);
  if (failed(result))
    return failure();

  rewriter.replaceOp(op, result->replacements);
  return success();
}

// Called after Phase 2.  Walks every vector.contract in funcOp; for any
// contract that is still missing nova.layout_*, climbs the parent chain
// looking for a scf.forall that carries pending attrs.  When found, attaches
// them to the contract and removes the temporaries from the forall.
static void transferParkedLayoutAttrs(func::FuncOp funcOp) {
  funcOp.walk([](vector::ContractionOp contractOp) {
    if (contractOp->hasAttr(kLayout0))
      return; // already annotated

    // Walk up through nested foralls until we find pending attrs.
    scf::ForallOp forall =
        contractOp->getParentOfType<scf::ForallOp>();
    while (forall) {
      if (!forall->hasAttr(kPendingLayout0)) {
        forall = forall->getParentOfType<scf::ForallOp>();
        continue;
      }
      if (auto a = forall->getAttr(kPendingLayout0))
        contractOp->setAttr(kLayout0, a);
      if (auto a = forall->getAttr(kPendingLayout1))
        contractOp->setAttr(kLayout1, a);
      if (auto a = forall->getAttr(kPendingLayout2))
        contractOp->setAttr(kLayout2, a);
      if (auto a = forall->getAttr(kPendingConfig))
        contractOp->setAttr(kLoweringConfig, a);
      forall->removeAttr(kPendingLayout0);
      forall->removeAttr(kPendingLayout1);
      forall->removeAttr(kPendingLayout2);
      forall->removeAttr(kPendingConfig);
      return;
    }
    // No pending attrs found — the contract did not originate from an MMA
    // op, or the forall structure is unexpected.  Not a hard error; the
    // absence of nova.layout_* will be caught by downstream passes.
  });
}

//===----------------------------------------------------------------------===//
// §4  The pass
//===----------------------------------------------------------------------===//

struct NovaGenericVectorizationPass
    : public PassWrapper<NovaGenericVectorizationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGenericVectorizationPass)

  StringRef getArgument() const override {
    return "nova-generic-vectorization";
  }
  StringRef getDescription() const override {
    return "Vectorize linalg ops to vector.contract / vector.transfer_*. "
           "MMA ops: park nova.layout_* on parent scf.forall, vectorize, "
           "then transfer to vector.contract after Phase 2 canonicalization. "
           "Static ops: direct vectorization. "
           "Dynamic ops: masked vectorization via ValueBounds inference.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override;
};

void NovaGenericVectorizationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = funcOp.getContext();
  IRRewriter rewriter(ctx);

  // -------------------------------------------------------------------------
  // Phase 1: Vectorize all LinalgOps and tensor.pad ops, bottom-up.
  //
  //   Case A — MMA contraction ops (have nova.layout_0 attr):
  //     Always static.  Park layout attrs on parent scf.forall, then call
  //     linalg::vectorize.  Attrs are transferred to vector.contract after
  //     Phase 2 (see Phase 2.5 below).
  //
  //   Case B — Static non-MMA linalg ops:
  //     Size-limit check, then plain linalg::vectorize with empty sizes.
  //
  //   Case C — Dynamic linalg ops (GELU boundary tiles):
  //     inferNovaVectorSizes → ValueBounds UB → masked vectorization.
  //
  //   Case D — tensor.pad with static result shape:
  //     Vectorize with sizes from the static result shape.
  // -------------------------------------------------------------------------
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op) || isa<tensor::PadOp>(op))
      candidates.push_back(op);
  });
  std::reverse(candidates.begin(), candidates.end()); // bottom-up

  for (Operation *op : candidates) {
    if (!op || op->getBlock() == nullptr)
      continue;

    rewriter.setInsertionPoint(op);

    // ── Case D: tensor.pad ────────────────────────────────────────────────
    if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      auto ty = padOp.getResultType();
      if (!ty.hasStaticShape())
        continue;
      SmallVector<int64_t> vectorSizes(ty.getShape());
      int64_t flat = std::accumulate(vectorSizes.begin(), vectorSizes.end(),
                                     int64_t(1), std::multiplies<int64_t>{});
      if (flat >= kMaxVectorSize)
        continue;
      SmallVector<bool> scalableDims(vectorSizes.size(), false);
      FailureOr<linalg::VectorizationResult> result =
          linalg::vectorize(rewriter, op, vectorSizes, scalableDims,
                            /*vectorizeNDExtract=*/true);
      if (succeeded(result))
        rewriter.replaceOp(op, result->replacements);
      continue;
    }

    auto linalgOp = cast<linalg::LinalgOp>(op);

    // ── Case A: MMA contraction op ────────────────────────────────────────
    if (isMmaContractionOp(op)) {
      if (!hasStaticShape(linalgOp)) {
        op->emitWarning() << "nova-generic-vectorization: MMA op has "
                             "dynamic shape — skipping";
        continue;
      }
      if (failed(isWithinVectorSizeLimit(linalgOp)))
        continue;
      if (failed(vectorizeMmaOp(rewriter, op))) {
        op->emitError() << "nova-generic-vectorization: failed to "
                           "vectorize MMA op";
        return signalPassFailure();
      }
      continue;
    }

    // ── Case B: Static non-MMA linalg op ─────────────────────────────────
    if (hasStaticShape(linalgOp)) {
    if (failed(isWithinVectorSizeLimit(linalgOp))) continue;

    // ── Global→shared promotion barrier ──────────────────────────────────
    // Two categories of linalg ops must NOT be vectorized here:
    //
    //  (1) linalg.copy {nova.promote_to_workgroup}
    //      This is the direct global→shared staging copy inserted by
    //      PromoteMatmulOperands.  It must stay as linalg.copy so that a
    //      future pass can pattern-match it and emit nvgpu.device_async_copy
    //      (cp.async PTX).  Vectorizing it into vector.transfer_read/write
    //      destroys the global→shared structural signature.
    //
    //  (2) Any linalg op whose result feeds a nova.promote_to_workgroup copy.
    //      For the weight (B-matrix) operand, PromoteMatmulOperands inserts a
    //      linalg.generic reshape (tensor<32x4> → tensor<1x32x4>) whose output
    //      is the `ins` of the promote_to_workgroup linalg.copy.  If we
    //      vectorize the generic here, the output lands in a tensor.empty()
    //      that OneShotBufferize aliases in-place with the workgroup buffer
    //      (because the copy's outs is the only consumer).  Both ins and outs
    //      of the linalg.copy then resolve to the same workgroup memref —
    //      turning the global→shared copy into a shared→shared self-copy and
    //      silently losing the actual global load.
    //
    // Shared→shared and shared→thread copies (plain linalg.copy without
    // nova.promote_to_workgroup, and not feeding one) are fine to vectorize —
    // they become vector.transfer_read/write which lower to LDS instructions.
    if (op->hasAttr("nova.promote_to_workgroup"))
      continue;
    bool feedsPromotionCopy = llvm::any_of(op->getUsers(), [](Operation *user) {
      return user->hasAttr("nova.promote_to_workgroup");
    });
    if (feedsPromotionCopy)
      continue;

    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op, {}, {},
                          /*vectorizeNDExtract=*/true);
    if (succeeded(result))
        rewriter.replaceOp(op, result->replacements);
    continue;
}
    // ── Case C: Dynamic linalg op — masked vectorization ─────────────────
    std::optional<nova::NovaVectorizationTileSizes> maybeSizes =
        nova::inferNovaVectorSizes(linalgOp);
    if (!maybeSizes)
      continue;
    int64_t flat =
        std::accumulate(maybeSizes->vectorSizes.begin(),
                        maybeSizes->vectorSizes.end(),
                        int64_t(1), std::multiplies<int64_t>{});
    if (flat >= kMaxVectorSize)
      continue;
    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op,
                          maybeSizes->vectorSizes,
                          maybeSizes->vectorScalableFlags,
                          /*vectorizeNDExtract=*/true);
    if (succeeded(result))
      rewriter.replaceOp(op, result->replacements);
  }

  // -------------------------------------------------------------------------
  // Phase 2: Canonicalize to vector.contract.
  // Run BEFORE layout transfer so contracts are in their final form when
  // we attach nova.layout_* in Phase 2.5.
  // -------------------------------------------------------------------------
  {
    RewritePatternSet contractPatterns(ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        contractPatterns);
    vector::populateSinkVectorOpsPatterns(contractPatterns);
    vector::populateVectorReductionToContractPatterns(contractPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(contractPatterns))))
      return signalPassFailure();
  }

  // -------------------------------------------------------------------------
  // Phase 2.5: Transfer parked nova.layout_* from scf.forall → vector.contract.
  // Phase 2 is what actually emits vector.contract (from the outer-product
  // decomposition linalg::vectorize produced).  Now that contracts exist,
  // pull the attrs we parked on the enclosing forall and attach them.
  // -------------------------------------------------------------------------
  transferParkedLayoutAttrs(funcOp);

  // -------------------------------------------------------------------------
  // Phase 3: Eliminate always-true vector.mask ops.
  // -------------------------------------------------------------------------
  vector::eliminateVectorMasks(rewriter, funcOp, /*vscaleRange=*/std::nullopt);

  // -------------------------------------------------------------------------
  // Phase 4: Canonicalize mask predicate expressions.
  // -------------------------------------------------------------------------
  {
    RewritePatternSet maskCanonPatterns(ctx);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        maskCanonPatterns);
    tensor::DimOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(maskCanonPatterns))))
      return signalPassFailure();
  }

  // -------------------------------------------------------------------------
  // Phase 5: Lower vector.mask { transfer } to predicated form.
  // -------------------------------------------------------------------------
  {
    RewritePatternSet maskLowerPatterns(ctx);
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        maskLowerPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(maskLowerPatterns))))
      return signalPassFailure();
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGenericVectorizationPass() {
  return std::make_unique<NovaGenericVectorizationPass>();
}

void registerNovaGenericVectorizationPass() {
  PassRegistration<NovaGenericVectorizationPass>();
}

} // namespace mlir::nova
