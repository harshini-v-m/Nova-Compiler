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

// Returns true if `op` is a linalg.generic contraction: has at least one
// reduction dim, and its body is mul+add (the standard matmul body).
static bool isContractionGeneric(linalg::GenericOp op) {
  // Must have exactly 2 inputs, 1 output.
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
    return false;
  // Must have at least one reduction iterator.
  bool hasReduction = false;
  for (auto iter : op.getIteratorTypesArray())
    if (iter == utils::IteratorType::reduction) {
      hasReduction = true;
      break;
    }
  if (!hasReduction)
    return false;
  // Body must be: %mul = mulf %in0, %in1; %add = addf %mul, %out; yield %add
  Block *body = op.getBody();
  if (body->getNumArguments() != 3)
    return false;
  auto *terminator = body->getTerminator();
  if (terminator->getNumOperands() != 1)
    return false;
  auto *addOp = terminator->getOperand(0).getDefiningOp();
  if (!addOp || (!isa<arith::AddFOp>(addOp) && !isa<arith::AddIOp>(addOp)))
    return false;
  // One operand of the add must be the accumulator (block arg 2).
  Value acc = body->getArgument(2);
  bool accIsOperand = (addOp->getOperand(0) == acc ||
                       addOp->getOperand(1) == acc);
  if (!accIsOperand)
    return false;
  // The other operand must come from a mul.
  Value mulVal = (addOp->getOperand(0) == acc) ? addOp->getOperand(1)
                                                : addOp->getOperand(0);
  auto *mulOp = mulVal.getDefiningOp();
  return mulOp && (isa<arith::MulFOp>(mulOp) || isa<arith::MulIOp>(mulOp));
}

//===----------------------------------------------------------------------===//
// Core fusion logic — shared between named-op and generic-op patterns.
//
// Matches:
//   %fill   = linalg.fill(0) outs(%init)
//   %matmul = <matmul-like> ins(%A, %B) outs(%fill)   [single use]
//   %result = linalg.generic (elementwise add) ins(%matmul, %bias) outs(%dest)
//
// Rewrites to:
//   %bias_init = <broadcast %bias to matmul output shape if needed>
//   %matmul'   = <matmul-like> ins(%A, %B) outs(%bias_init)
//   (addOp replaced by %matmul')
//===----------------------------------------------------------------------===//

static LogicalResult fuseMatmulBias(Operation *matmulOp,
                                    Value outsVal,
                                    Value mmResult,
                                    PatternRewriter &rewriter) {
  // 1. outs must come from a zero fill.
  auto fillOp = outsVal.getDefiningOp<linalg::FillOp>();
  if (!fillOp || !isZeroFill(fillOp))
    return failure();

  // 2. Matmul must have exactly one use.
  if (!matmulOp->hasOneUse())
    return failure();

  // 3. That single use must be an elementwise add (bias add).
  auto addOp = dyn_cast<linalg::GenericOp>(*matmulOp->user_begin());
  if (!addOp || !isElementwiseAdd(addOp))
    return failure();

  // 4. Find which add input is bias (not the matmul result).
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
  Location loc = matmulOp->getLoc();

  // 5. Build bias-initialised outs for the matmul.
  Value newOuts;
  if (biasType == matmulResultType) {
    newOuts = bias;
  } else {
    Value broadcastDest = fillOp.getOutputs()[0];
    int rank = matmulResultType.getRank();
    AffineMap biasMap = addOp.getIndexingMapsArray()[biasIdx];
    AffineMap outMap  = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    newOuts = rewriter
                  .create<linalg::GenericOp>(
                      loc, matmulResultType,
                      ValueRange{bias}, ValueRange{broadcastDest},
                      SmallVector<AffineMap>{biasMap, outMap}, iters,
                      [](OpBuilder &b, Location loc, ValueRange args) {
                        b.create<linalg::YieldOp>(loc, args[0]);
                      })
                  .getResult(0);
  }

  // 6. Swap matmul outs from zero-fill to bias-initialised tensor.
  rewriter.modifyOpInPlace(matmulOp, [&]() {
    auto dpsIface = cast<DestinationStyleOpInterface>(matmulOp);
    matmulOp->setOperand(
        dpsIface.getDpsInitOperand(0)->getOperandNumber(), newOuts);
  });

  // 7. Replace the add op — matmul result now includes the bias.
  rewriter.replaceOp(addOp, mmResult);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern for named matmul ops (MatmulOp, BatchMatmulOp)
//===----------------------------------------------------------------------===//

template <typename MatmulOpTy>
struct FuseMatmulBiasIntoOuts : public OpRewritePattern<MatmulOpTy> {
  using OpRewritePattern<MatmulOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOpTy matmulOp,
                                PatternRewriter &rewriter) const override {
    Value outsVal = matmulOp.getDpsInitOperand(0)->get();
    Value mmResult = matmulOp.getResult(0);
    return fuseMatmulBias(matmulOp.getOperation(), outsVal, mmResult, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pattern for linalg.generic contractions (the nova.linear lowering path)
//===----------------------------------------------------------------------===//

struct FuseGenericContractionBias : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isContractionGeneric(genericOp))
      return failure();
    Value outsVal = genericOp.getDpsInitOperand(0)->get();
    Value mmResult = genericOp.getResult(0);
    return fuseMatmulBias(genericOp.getOperation(), outsVal, mmResult, rewriter);
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
                 FuseMatmulBiasIntoOuts<linalg::BatchMatmulOp>,
                 FuseGenericContractionBias>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "fuse-matmul-bias"; }
  StringRef getDescription() const final {
    return "Fuse bias-add into matmul/batch_matmul/generic-contraction outs initializer";
  }
};

} // namespace

namespace mlir::nova {

std::unique_ptr<Pass> createFuseMatmulBiasPass() {
  return std::make_unique<FuseMatmulBiasPass>();
}

} // namespace mlir::nova
