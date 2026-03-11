// Nova GPU Pad Operands Pass
// Ported from IREE's GPUPadOperands.cpp
//
// Key fix vs the original: the K (reduction) dimension is now padded to the
// next multiple of the reduction tile size read from the op's LoweringConfig.
// Without this, a non-aligned K causes the K-reduction loop to silently
// truncate the last (K % reductionTile) elements, producing wrong results,
// and also causes dynamic alloca in the GPU kernel, rejected by NVPTX.

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
  rewriter.replaceOp(linalgOp, newResults.front());
  return success();
}

// Returns padding sizes for contraction ops, reading tile sizes from the op's
// LoweringConfig attribute.
//
// - Parallel dims (M, N):   padded to the matching workgroup tile size.
// - Reduction dims (K):     padded to the matching reduction tile size.
//   → prevents K-truncation when K is not a multiple of the reduction step.
//   → makes all tile sizes static so PTX doesn't need dynamic alloca.
//
// Falls back to heuristic (M/N=128, K=1) when no config is present (e.g.
// when called on an op that wasn't annotated by the strategy pass).
static std::optional<SmallVector<int64_t>>
getPaddingSizes(linalg::LinalgOp linalgOp) {
  DictionaryAttr config = getLoweringConfig(linalgOp);
  if (!config) {
    if (!linalg::isaContractionOpInterface(linalgOp))
      return std::nullopt;
  }

  SmallVector<int64_t> wgTiles  = getLoweringConfigTileSizes(config, kWorkgroupKey);
  SmallVector<int64_t> redTiles = getLoweringConfigTileSizes(config, kReductionKey);

  int numLoops = linalgOp.getNumLoops();
  SmallVector<int64_t> padding(numLoops, 1);
  auto iterTypes = linalgOp.getIteratorTypesArray();

  // Walk loops in reverse so we visit N before M (innermost first).
  int parallelIdx  = 0; // 0=N, 1=M, ... (outermost last)
  int reductionIdx = 0; // 0=K (usually only one)

  for (int i = numLoops - 1; i >= 0; --i) {
    if (linalg::isParallelIterator(iterTypes[i])) {
      // Map reverse parallel index to forward wgTiles index.
      int wgIdx = (int)wgTiles.size() - 1 - parallelIdx;
      if (!wgTiles.empty() && wgIdx >= 0 && wgTiles[wgIdx] > 0)
        padding[i] = wgTiles[wgIdx];
      else
        padding[i] = 128; // heuristic fallback
      ++parallelIdx;
    } else {
      // Reduction dim: pad to the reduction tile step (e.g. 8 for K-step=8).
      int redIdx = (int)redTiles.size() - 1 - reductionIdx;
      if (!redTiles.empty() && redIdx >= 0 && redTiles[redIdx] > 0)
        padding[i] = redTiles[redIdx];
      // else leave at 1 (no-op pad for unrecognized reduction dims).
      ++reductionIdx;
    }
  }
  return padding;
}

// Folds tensor.pad(extract_slice*(linalg.fill(cst)), cst) into
// linalg.fill(cst, tensor.empty(...)) when the constants match.
// Ported from IREE's populateFoldFillIntoPadPattern (Transforms.cpp).
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

    // Walk all linalg ops and pad those with a padding config.
    // IREE reads sizes from LoweringConfig; Nova uses heuristic sizes.
    funcOp.walk([&](linalg::LinalgOp op) {
      std::optional<SmallVector<int64_t>> paddingSizes = getPaddingSizes(op);
      if (!paddingSizes)
        return WalkResult::advance();

      rewriter.setInsertionPoint(op);
      // Non-fatal if padding fails (op may already be statically sized).
      (void)padLinalgOpToStaticSizes(rewriter, op, *paddingSizes);
      return WalkResult::advance();
    });

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
