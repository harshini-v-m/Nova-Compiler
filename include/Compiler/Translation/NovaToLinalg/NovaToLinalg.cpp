#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

namespace mlir {
namespace nova {

// Conversion Patterns

struct NovaBroadcastInDimOpLowering
    : public OpConversionPattern<nova::BroadcastInDimOp> {
  using OpConversionPattern<nova::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getOperand();

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked tensor result type");
    }

    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked tensor input type");
    }

    auto loc = op.getLoc();
    auto dimsAttr = op.getBroadcastDimensions();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(),
        resultType.getEncoding());

    // Build affine map for input
    SmallVector<AffineExpr> inputExprs;
    for (auto [inputIdx, dimAttr] :
         llvm::enumerate(dimsAttr.getAsValueRange<IntegerAttr>())) {
      int64_t outputDim = dimAttr.getSExtValue();
      int64_t inputSize = inputType.getDimSize(inputIdx);
      int64_t outputSize = resultType.getDimSize(outputDim);

      // If broadcasting dimension (1 -> N), use constant 0
      if (inputSize == 1 && outputSize != 1) {
        inputExprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        inputExprs.push_back(rewriter.getAffineDimExpr(outputDim));
      }
    }

    // Build affine map for output (identity)
    SmallVector<AffineExpr> outputExprs;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }

    auto inputMap = AffineMap::get(resultType.getRank(), 0, inputExprs,
                                   rewriter.getContext());
    auto outputMap = AffineMap::get(resultType.getRank(), 0, outputExprs,
                                    rewriter.getContext());

    SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    // Create linalg.generic for broadcast
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, input, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct NovaMatmulOpLoweringgeneric
    : public OpConversionPattern<nova::MatmulOp> {
  using OpConversionPattern<nova::MatmulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    auto resultType = cast<RankedTensorType>(op.getType());

    int64_t rank = resultType.getRank();
    auto shape = resultType.getShape();

    // 1. Initialize Output (Zero Fill)
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, shape, resultType.getElementType(), resultType.getEncoding());
    Value out = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);

    // 2. Define Iteration Space and Indexing Maps
    SmallVector<AffineMap> maps;
    SmallVector<utils::IteratorType> iters;

    // totalLoops = Batch dims + M + N + K
    // For 3D (1x8x8), totalLoops = 1 (batch) + 3 (M,N,K) = 4
    int64_t batchRank = rank - 2;
    int64_t totalLoops = batchRank + 3;

    SmallVector<AffineExpr> lhsExprs, rhsExprs, outExprs;

    // Batch Dimensions
    for (int64_t i = 0; i < batchRank; ++i) {
      auto d = rewriter.getAffineDimExpr(i);
      lhsExprs.push_back(d);
      rhsExprs.push_back(d);
      outExprs.push_back(d);
      iters.push_back(utils::IteratorType::parallel);
    }

    // Define M, N, and K dimension expressions
    auto dimM = rewriter.getAffineDimExpr(batchRank);
    auto dimN = rewriter.getAffineDimExpr(batchRank + 1);
    auto dimK = rewriter.getAffineDimExpr(batchRank + 2);

    // LHS: (Batch..., M, K)
    lhsExprs.push_back(dimM);
    lhsExprs.push_back(dimK);

    // RHS: (Batch..., K, N)
    rhsExprs.push_back(dimK);
    rhsExprs.push_back(dimN);

    // Out: (Batch..., M, N)
    outExprs.push_back(dimM);
    outExprs.push_back(dimN);

    // Iterator types for M, N (parallel) and K (reduction)
    iters.push_back(utils::IteratorType::parallel);  // M
    iters.push_back(utils::IteratorType::parallel);  // N
    iters.push_back(utils::IteratorType::reduction); // K

    maps.push_back(AffineMap::get(totalLoops, 0, lhsExprs, ctx));
    maps.push_back(AffineMap::get(totalLoops, 0, rhsExprs, ctx));
    maps.push_back(AffineMap::get(totalLoops, 0, outExprs, ctx));

    // 3. Create the linalg.generic operation
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{lhs, rhs}, ValueRange{out}, maps, iters,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
          Value add = b.create<arith::AddFOp>(nestedLoc, mul, args[2]);
          b.create<linalg::YieldOp>(nestedLoc, add);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
struct NovaGatherOpLowering : public OpConversionPattern<nova::GatherOp> {
  using OpConversionPattern<nova::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value indices = adaptor.getIndices();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(),
        resultType.getEncoding());

    auto indexMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
    SmallVector<AffineMap> indexingMaps = {indexMap, indexMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, indices, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location l, ValueRange args) {
          // args[0] is the index value
          Value batchIdx = b.create<linalg::IndexOp>(l, 0);
          Value classIdx =
              b.create<arith::IndexCastOp>(l, b.getIndexType(), args[0]);
          Value extracted =
              b.create<tensor::ExtractOp>(l, input, ValueRange{batchIdx, classIdx});
          b.create<linalg::YieldOp>(l, extracted);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct NovaScatterAddOpLowering : public OpConversionPattern<nova::ScatterAddOp> {
  using OpConversionPattern<nova::ScatterAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ScatterAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto loc = op.getLoc();
    Value input = adaptor.getInput(); // self
    Value indices = adaptor.getIndices();
    Value src = adaptor.getSrc();

    auto ctx = rewriter.getContext();
    auto i = rewriter.getAffineDimExpr(0);
    auto j = rewriter.getAffineDimExpr(1);

    auto mapIndices = AffineMap::get(2, 0, {i}, ctx);
    auto mapSrc = AffineMap::get(2, 0, {i}, ctx);
    auto mapInput = AffineMap::get(2, 0, {i, j}, ctx);
    auto mapResult = AffineMap::get(2, 0, {i, j}, ctx);

    SmallVector<AffineMap> indexingMaps = {mapIndices, mapSrc, mapInput,
                                           mapResult};
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{indices, src, input}, input,
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location l, ValueRange args) {
          // args[0] is index[i]
          // args[1] is src[i]
          // args[2] is input[i, j]

          Value classIdx = b.create<linalg::IndexOp>(l, 1);
          Value targetClassIdx =
              b.create<arith::IndexCastOp>(l, b.getIndexType(), args[0]);
          Value isTarget = b.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::eq, classIdx, targetClassIdx);

          Value zero = b.create<arith::ConstantOp>(
              l, b.getZeroAttr(resultType.getElementType()));
          Value toAdd = b.create<arith::SelectOp>(l, isTarget, args[1], zero);

          Value res = b.create<arith::AddFOp>(l, args[2], toAdd);
          b.create<linalg::YieldOp>(l, res);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct NovaTransposeOpLowering : public OpConversionPattern<nova::TransposeOp> {
  using OpConversionPattern<nova::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    auto resultShape = resultType.getShape();
    int64_t rank = resultShape.size();

    int64_t axes1 = rank - 1;
    int64_t axes2 = rank - 2;

    if (auto attr = op->getAttrOfType<IntegerAttr>("axes1"))
      axes1 = attr.getInt();
    if (auto attr = op->getAttrOfType<IntegerAttr>("axes2"))
      axes2 = attr.getInt();

    if (axes1 < 0)
      axes1 += rank;
    if (axes2 < 0)
      axes2 += rank;

    llvm::SmallVector<int64_t> perms;
    for (int64_t i = 0; i < rank; i++) {
      if (i == axes1)
        perms.push_back(axes2);
      else if (i == axes2)
        perms.push_back(axes1);
      else
        perms.push_back(i);
    }

    Location loc = op.getLoc();
    auto permutedInit = rewriter.create<tensor::EmptyOp>(
        loc, resultShape, resultType.getElementType(),
        resultType.getEncoding());

    auto transposeOp = rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getInput(), permutedInit, perms);

    // Explicitly set the result type to ensure encoding is preserved
    transposeOp->getResult(0).setType(resultType);

    return success();
  }
};

struct NovaToDeviceOpLowering : public OpConversionPattern<nova::ToDeviceOp> {
  using OpConversionPattern<nova::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ToDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor with the new encoding (device)
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(),
        resultType.getEncoding());

    // Identity maps for linalg.generic
    auto identityMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, input, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

void populateNovaToLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaMatmulOpLoweringgeneric, NovaBroadcastInDimOpLowering,
               NovaTransposeOpLowering, NovaToDeviceOpLowering,
               NovaScatterAddOpLowering,NovaGatherOpLowering
               //  NovaDivopLowering
               //    ,NovaSquareOpLowering
               >(patterns.getContext());
}
} // namespace nova
} // namespace mlir