#include "Compiler/Transforms/FuseMatmulBias.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

// Returns true if `fillOp` fills with a scalar zero constant.
static bool isZeroFill(linalg::FillOp fillOp) {
  auto constOp = fillOp.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  if (auto fa = dyn_cast<FloatAttr>(constOp.getValue()))
    return fa.getValue().isZero();
  if (auto ia = dyn_cast<IntegerAttr>(constOp.getValue()))
    return ia.getValue().isZero();
  return false;
}

// Returns true if `op` is an elementwise add (all-parallel linalg.generic
// whose body yields an arith.addf / arith.addi result).
static bool isElementwiseAdd(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2)
    return false;
  if (op.getNumParallelLoops() != op.getNumLoops())
    return false;
  auto *termDef =
      op.getBody()->getTerminator()->getOperand(0).getDefiningOp();
  return termDef &&
         (isa<arith::AddFOp>(termDef) || isa<arith::AddIOp>(termDef));
}

//===----------------------------------------------------------------------===//
// FuseMatmulBiasIntoOuts
//
// Transforms the pattern:
//
//   %zero  = arith.constant 0.0
//   %fill  = linalg.fill ins(%zero) outs(%init)    -> tensor<...xf32>
//   %mm    = linalg.matmul ins(%A, %B) outs(%fill) -> tensor<MxNxf32>
//   %result= linalg.generic (elementwise add) ins(%mm, %bias) ...
//
// Into:
//
//   %bias_bcast = <broadcast %bias to MxN if needed>
//   %result     = linalg.matmul ins(%A, %B) outs(%bias_bcast) -> tensor<MxNxf32>
//
// The matmul accumulates into its outs tensor (C = A@B + C_init), so
// initialising outs with the bias instead of zeros fuses the add for free.
// No body cloning, no new generic op for the matmul computation.
//===----------------------------------------------------------------------===//

template <typename MatmulOpTy>
struct FuseMatmulBiasIntoOuts : public OpRewritePattern<MatmulOpTy> {
  using OpRewritePattern<MatmulOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOpTy matmulOp,
                                PatternRewriter &rewriter) const override {
    // 1. The matmul's outs must come from a linalg.fill(0).
    Value outsVal = matmulOp.getDpsInitOperand(0)->get();
    auto fillOp = outsVal.getDefiningOp<linalg::FillOp>();
    if (!fillOp || !isZeroFill(fillOp))
      return failure();

    // 2. The matmul must have exactly one use so replacing it is safe.
    if (!matmulOp->hasOneUse())
      return failure();

    // 3. That single use must be an elementwise add (the bias add).
    auto addOp =
        dyn_cast<linalg::GenericOp>(*matmulOp->user_begin());
    if (!addOp || !isElementwiseAdd(addOp))
      return failure();

    // 4. Identify which add input is the matmul result and which is the bias.
    Value mmResult = matmulOp.getResult(0);
    int biasIdx = -1;
    for (int i = 0; i < 2; ++i) {
      if (addOp.getDpsInputOperand(i)->get() != mmResult)
        biasIdx = i;
    }
    if (biasIdx < 0)
      return failure();

    Value bias = addOp.getDpsInputOperand(biasIdx)->get();

    auto matmulResultType = cast<RankedTensorType>(mmResult.getType());
    auto biasType = cast<RankedTensorType>(bias.getType());
    Location loc = matmulOp.getLoc();

    // 5. Build the bias-initialised outs tensor for the matmul.
    //
    //    Case A — bias shape already matches matmul output: use it directly.
    //    Case B — bias needs broadcasting (e.g. [N] bias for [M,N] matmul):
    //             create a single linalg.generic broadcast driven by the add's
    //             existing indexing map, writing into the fill's empty tensor.
    Value newOuts;
    if (biasType == matmulResultType) {
      newOuts = bias;
    } else {
      // Reuse the fill's output tensor (already the correct shape/type) as
      // the broadcast destination so we don't allocate an extra empty.
      Value broadcastDest = fillOp.getOutputs()[0];

      int rank = matmulResultType.getRank();
      AffineMap biasMap = addOp.getIndexingMapsArray()[biasIdx];
      AffineMap outMap  = rewriter.getMultiDimIdentityMap(rank);

      SmallVector<utils::IteratorType> iters(rank,
                                              utils::IteratorType::parallel);
      newOuts = rewriter
                    .create<linalg::GenericOp>(
                        loc, matmulResultType,
                        /*inputs=*/ValueRange{bias},
                        /*outputs=*/ValueRange{broadcastDest},
                        SmallVector<AffineMap>{biasMap, outMap}, iters,
                        [](OpBuilder &b, Location loc, ValueRange args) {
                          b.create<linalg::YieldOp>(loc, args[0]);
                        })
                    .getResult(0);
    }

    // 6. Swap the matmul's outs from zero-fill to bias-initialised tensor.
    //    The matmul op itself is unchanged — it still computes C = A@B + C_init,
    //    but now C_init carries the bias instead of zeros.
    rewriter.modifyOpInPlace(matmulOp, [&]() {
      matmulOp->setOperand(
          matmulOp.getDpsInitOperand(0)->getOperandNumber(), newOuts);
    });

    // 7. The add op is now redundant — its result equals the matmul result.
    rewriter.replaceOp(addOp, mmResult);
    // The linalg.fill becomes dead and is removed by DCE / canonicalization.
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct FuseMatmulBiasPass
    : public PassWrapper<FuseMatmulBiasPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulBiasPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulBiasIntoOuts<linalg::MatmulOp>,
                 FuseMatmulBiasIntoOuts<linalg::BatchMatmulOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "fuse-matmul-bias"; }
  StringRef getDescription() const final {
    return "Fuse bias-add into matmul/batch_matmul outs initializer";
  }
};

} // namespace

namespace mlir::nova {

std::unique_ptr<Pass> createFuseMatmulBiasPass() {
  return std::make_unique<FuseMatmulBiasPass>();
}

} // namespace mlir::nova
