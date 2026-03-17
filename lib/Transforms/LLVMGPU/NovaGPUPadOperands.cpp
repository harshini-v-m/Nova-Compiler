//===- NovaGPUPadOperands.cpp - Pad linalg operands to static tile sizes --===//
//
// Pads linalg op operands (A, B, C for matmuls; inputs for reductions) to
// static multiples of the tile sizes read from the op's lowering_config
// attribute.  Ported from IREE's GPUPadOperands.cpp.
//
// Why this is needed:
//   - Non-aligned dimensions prevent static tile shapes, which cause
//     dynamic alloca ops inside the GPU kernel that NVPTX rejects.
//   - The K (reduction) dimension MUST be padded to the reduction tile step;
//     without this, the K-reduction loop silently truncates the last
//     (K % reductionTile) rows/columns, producing numerically wrong results.
//
// IMPORTANT: Padding values are always zero (safe for matmul / convolution
// because the output is not affected by zero-padding in the contraction dims).
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

// Pads all loop dimensions of a linalg op to multiples of the given sizes.
// Uses zero-padding for all operands (safe for matmul/convolution).
// CopyBackOp::None: the padded tensor is used inline; no explicit copy-back.
static LogicalResult padLinalgOpToStaticSizes(RewriterBase &rewriter,
                                              linalg::LinalgOp linalgOp,
                                              ArrayRef<int64_t> padding) {
  SmallVector<int64_t> paddingDims =
      llvm::to_vector(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  SmallVector<bool> nofoldFlags(linalgOp.getNumDpsInputs(), /*nofold=*/false);

  SmallVector<Attribute> paddingValueAttributes;
  for (auto &operand : linalgOp->getOpOperands()) {
    Type elemType = getElementTypeOrSelf(operand.get().getType());
    paddingValueAttributes.push_back(rewriter.getZeroAttr(elemType));
  }

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(paddingDims)
          .setPaddingValues(paddingValueAttributes)
          .setPadToMultipleOf(padding)
          .setNofoldFlags(nofoldFlags)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

  linalg::LinalgOp paddedOp;
  SmallVector<Value> newResults;
  SmallVector<tensor::PadOp> padOps;
  if (failed(linalg::rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                                        newResults, padOps))) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to pad linalg op operands");
  }
  rewriter.replaceOp(linalgOp, newResults);
  return success();
}

// Folds tensor.pad(extract_slice*(linalg.fill(cst)), cst) into
// linalg.fill(cst, tensor.empty(...)) when the padding constant matches the
// fill constant. Ported from IREE's populateFoldFillIntoPadPattern.
// Avoids materialising a large padded tensor just to fill it with a constant.
namespace {
struct FoldFillIntoPad : public OpRewritePattern<tensor::PadOp> {
  FoldFillIntoPad(MLIRContext *ctx) : OpRewritePattern<tensor::PadOp>(ctx) {}
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    // Walk through extract_slice chain to find the source fill op.
    Operation *currentOp = padOp.getSource().getDefiningOp();
    auto maybeExtractSlice =
        dyn_cast_if_present<tensor::ExtractSliceOp>(currentOp);
    while (currentOp && maybeExtractSlice) {
      currentOp = maybeExtractSlice.getSource().getDefiningOp();
      maybeExtractSlice =
          dyn_cast_if_present<tensor::ExtractSliceOp>(currentOp);
    }

    auto fillOp = dyn_cast_if_present<linalg::FillOp>(currentOp);
    if (!fillOp)
      return rewriter.notifyMatchFailure(padOp, "not from a linalg.fill");

    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue ||
        getAsOpFoldResult(padValue) !=
            getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get()))
      return rewriter.notifyMatchFailure(padOp, "constants don't match");

    Location loc = padOp.getLoc();
    RankedTensorType resultType = padOp.getResultType();
    auto emptyOp = tensor::EmptyOp::create(
        rewriter, loc, tensor::getMixedSizes(rewriter, loc, padOp),
        resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(padOp, padValue,
                                                emptyOp.getResult());
    return success();
  }
};
} // namespace

struct NovaGPUPadOperandsPass
    : public PassWrapper<NovaGPUPadOperandsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUPadOperandsPass)

  NovaGPUPadOperandsPass() = default;
  NovaGPUPadOperandsPass(const NovaGPUPadOperandsPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);

    // Walk all linalg ops and pad those with a "padding" list in their
    // lowering_config. Mirrors IREE's GPUPadOperands: all padding
    // intelligence is in the config (set by SelectLoweringStrategy),
    // the pass just reads and applies.
    bool padFailed = false;
    funcOp.walk([&](linalg::LinalgOp op) {
      DictionaryAttr config = getLoweringConfig(op);
      if (!config)
        return WalkResult::advance();

      std::optional<SmallVector<int64_t>> paddingSizes = getPaddingList(config);
      if (!paddingSizes)
        return WalkResult::advance();

      rewriter.setInsertionPoint(op);
      if (::mlir::failed(padLinalgOpToStaticSizes(rewriter, op, *paddingSizes)))
        padFailed = true;
      return WalkResult::advance();
    });
    if (padFailed)
      return signalPassFailure();

    // Fold fill+pad sequences: pad(fill(cst), cst) -> fill(cst, empty).
    MLIRContext *context = &getContext();
    RewritePatternSet cleanupPatterns(context);
    cleanupPatterns.add<FoldFillIntoPad>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns))))
      return signalPassFailure();
  }

  StringRef getArgument() const override { return "nova-gpu-pad-operands"; }
  StringRef getDescription() const override {
    return "Pads linalg operands to static multiples of tile sizes";
  }
};

std::unique_ptr<Pass> createNovaGPUPadOperandsPass() {
  return std::make_unique<NovaGPUPadOperandsPass>();
}

void registerNovaGPUPadOperandsPass() {
  PassRegistration<NovaGPUPadOperandsPass>();
}

} // namespace mlir::nova
