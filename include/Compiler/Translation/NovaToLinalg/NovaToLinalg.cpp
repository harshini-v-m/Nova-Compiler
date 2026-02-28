#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

using namespace mlir;
using namespace mlir::nova;
using namespace mlir::bufferization;

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
        loc, resultType.getShape(), resultType.getElementType());

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
// Helper function to broadcast a tensor to a target shape
static Value broadcastTensor(ConversionPatternRewriter &rewriter, Location loc,
                             Value input, ArrayRef<int64_t> targetShape) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  auto inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  int64_t targetRank = targetShape.size();

  // If shapes already match, no broadcast needed
  if (inputRank == targetRank &&
      std::equal(inputShape.begin(), inputShape.end(), targetShape.begin())) {
    return input;
  }

  Value current = input;

  // Handle rank expansion ( [M, K] -> [1, M, K])
  if (inputRank < targetRank) {
    int64_t rankDiff = targetRank - inputRank;

    // Build the expanded shape with leading 1s
    SmallVector<int64_t> expandedShape;
    for (int64_t i = 0; i < rankDiff; ++i) {
      expandedShape.push_back(1);
    }
    expandedShape.append(inputShape.begin(), inputShape.end());

    // Build reassociation indices for tensor.expand_shape
    SmallVector<ReassociationIndices> reassociation;

    // all the new dimensions (leading 1s) plus the first original dimension
    ReassociationIndices firstGroup;
    for (int64_t i = 0; i <= rankDiff; ++i) {
      firstGroup.push_back(i);
    }
    reassociation.push_back(firstGroup);

    // Remaining dimensions map 1:1
    for (int64_t i = 1; i < inputRank; ++i) {
      reassociation.push_back({rankDiff + i});
    }

    auto expandedType =
        RankedTensorType::get(expandedShape, inputType.getElementType());
    current = rewriter.create<tensor::ExpandShapeOp>(loc, expandedType, current,
                                                     reassociation);

    inputType = expandedType;
    inputShape = expandedShape;
    inputRank = targetRank;
  }

  // Handle broadcasting dimensions ([1, M, K] -> [B, M, K])
  bool needsBroadcast = false;
  for (int64_t i = 0; i < targetRank; ++i) {
    if (inputShape[i] != targetShape[i]) {
      needsBroadcast = true;
      break;
    }
  }

  if (!needsBroadcast) {
    return current;
  }

  // Build affine map for broadcasting
  SmallVector<AffineExpr> inputExprs;
  for (int64_t i = 0; i < targetRank; ++i) {
    int64_t inputSize = inputShape[i];
    int64_t targetSize = targetShape[i];

    // If broadcasting dimension (1 -> N), use constant 0
    if (inputSize == 1 && targetSize != 1) {
      inputExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }

  // Build affine map for output (identity)
  SmallVector<AffineExpr> outputExprs;
  for (int64_t i = 0; i < targetRank; ++i) {
    outputExprs.push_back(rewriter.getAffineDimExpr(i));
  }

  auto inputMap =
      AffineMap::get(targetRank, 0, inputExprs, rewriter.getContext());
  auto outputMap =
      AffineMap::get(targetRank, 0, outputExprs, rewriter.getContext());

  // Create empty output tensor
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, targetShape, inputType.getElementType());

  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
  SmallVector<utils::IteratorType> iteratorTypes(targetRank,
                                                 utils::IteratorType::parallel);

  // Create linalg.generic for broadcast
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{emptyTensor.getType()}, current, emptyTensor, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      });

  return genericOp.getResult(0);
}
struct NovaMatmulOpLowering : public OpConversionPattern<nova::MatmulOp> {
  using OpConversionPattern<nova::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();

    if (operands.size() != 2) {
      return rewriter.notifyMatchFailure(op, "expected exactly 2 operands");
    }

    Value lhs = operands[0];
    Value rhs = operands[1];

    auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());

    // Get result type and create empty output tensor
    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }

    auto resultShape = resultType.getShape();
    auto newShape = resultShape.drop_back(2);
    int64_t resultRank = resultShape.size();

    // For rank > 3: flatten batch dimensions
    if (resultRank > 3) {
      // Broadcast LHS to match result batch dimensions
      SmallVector<int64_t> lhsTargetShape(newShape.begin(), newShape.end());
      lhsTargetShape.push_back(lhsType.getShape()[lhsType.getRank() - 2]);
      lhsTargetShape.push_back(lhsType.getShape()[lhsType.getRank() - 1]);
      lhs = broadcastTensor(rewriter, op.getLoc(), lhs, lhsTargetShape);
      lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());

      // Broadcast RHS to match result batch dimensions
      SmallVector<int64_t> rhsTargetShape(newShape.begin(), newShape.end());
      rhsTargetShape.push_back(rhsType.getShape()[rhsType.getRank() - 2]);
      rhsTargetShape.push_back(rhsType.getShape()[rhsType.getRank() - 1]);
      rhs = broadcastTensor(rewriter, op.getLoc(), rhs, rhsTargetShape);
      rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());

      int64_t N = 1;
      for (int64_t i = 0; i < (resultRank - 2); i++) {
        N *= resultType.getShape()[i];
      }

      int64_t M = resultType.getShape()[resultRank - 2];
      int64_t K = lhsType.getShape()[lhsType.getRank() - 1];
      int64_t N_cols = resultType.getShape()[resultRank - 1];

      SmallVector<int64_t> rank3_lhs_shape({N, M, K});
      SmallVector<int64_t> rank3_rhs_shape({N, K, N_cols});
      SmallVector<int64_t> rank3_output_shape({N, M, N_cols});

      auto rank3LhsType =
          RankedTensorType::get(rank3_lhs_shape, lhsType.getElementType());
      auto rank3RhsType =
          RankedTensorType::get(rank3_rhs_shape, rhsType.getElementType());

      // Build reassociation to collapse batch dimensions into one
      SmallVector<ReassociationIndices> lhsReassociation;
      ReassociationIndices batchIndices;
      for (int64_t i = 0; i < resultRank - 2; ++i) {
        batchIndices.push_back(i); // All batch dims collapse to one
      }
      lhsReassociation.push_back(batchIndices);
      lhsReassociation.push_back({resultRank - 2}); // M dimension
      lhsReassociation.push_back({resultRank - 1}); // K dimension

      SmallVector<ReassociationIndices> rhsReassociation;
      ReassociationIndices rhsBatchIndices;
      for (int64_t i = 0; i < resultRank - 2; ++i) {
        rhsBatchIndices.push_back(i); // All batch dims collapse to one
      }
      rhsReassociation.push_back(rhsBatchIndices);
      rhsReassociation.push_back({resultRank - 2}); // K dimension
      rhsReassociation.push_back({resultRank - 1}); // N dimension

      Value lhsCollapsed = rewriter.create<tensor::CollapseShapeOp>(
          op.getLoc(), rank3LhsType, lhs, lhsReassociation);
      Value rhsCollapsed = rewriter.create<tensor::CollapseShapeOp>(
          op.getLoc(), rank3RhsType, rhs, rhsReassociation);

      Value cst = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getZeroAttr(resultType.getElementType()));
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), rank3_output_shape, resultType.getElementType());
      Value outputTensor =
          rewriter.create<linalg::FillOp>(op.getLoc(), cst, emptyTensor)
              .getResult(0);

      Value matmul3D =
          rewriter
              .create<linalg::BatchMatmulOp>(
                  op.getLoc(), ValueRange{lhsCollapsed, rhsCollapsed},
                  outputTensor)
              .getResult(0);

      // Build reassociation to expand N back to original batch dimensions
      SmallVector<ReassociationIndices> resultReassociation;
      ReassociationIndices expandedBatchIndices;
      for (int64_t i = 0; i < resultRank - 2; ++i) {
        expandedBatchIndices.push_back(i); // N expands to all batch dims
      }
      resultReassociation.push_back(expandedBatchIndices);
      resultReassociation.push_back({resultRank - 2}); // M dimension
      resultReassociation.push_back({resultRank - 1}); // N dimension

      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          op, resultType, matmul3D, resultReassociation);
      return success();
    }

    // Handle broadcasting for batch matmul
    if (resultRank == 3) {
      // Result shape is [B, M, N]
      int64_t batchDim = resultShape[0];

      // Broadcast lhs to [B, M, K] if needed
      SmallVector<int64_t> lhsTargetShape;
      if (lhsType.getRank() == 2) {
        // [M, K] -> [B, M, K]
        lhsTargetShape = {batchDim, lhsType.getShape()[0],
                          lhsType.getShape()[1]};
      } else if (lhsType.getRank() == 3) {
        // [B', M, K] -> [B, M, K] (B' might be 1 or different)
        lhsTargetShape = {batchDim, lhsType.getShape()[1],
                          lhsType.getShape()[2]};
      }

      if (!lhsTargetShape.empty()) {
        lhs = broadcastTensor(rewriter, op.getLoc(), lhs, lhsTargetShape);
      }

      // Broadcast rhs to [B, K, N] if needed
      SmallVector<int64_t> rhsTargetShape;
      if (rhsType.getRank() == 2) {
        rhsTargetShape = {batchDim, rhsType.getShape()[0],
                          rhsType.getShape()[1]};
      } else if (rhsType.getRank() == 3) {
        rhsTargetShape = {batchDim, rhsType.getShape()[1],
                          rhsType.getShape()[2]};
      }

      if (!rhsTargetShape.empty()) {
        rhs = broadcastTensor(rewriter, op.getLoc(), rhs, rhsTargetShape);
      }
    }

    // create a constant zero
    Value cst = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(resultType.getElementType()));
    // Create an empty tensor for the output
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    // create a fill op to initialize the output tensor to zero
    Value outputTensor =
        rewriter.create<linalg::FillOp>(op.getLoc(), cst, emptyTensor)
            .getResult(0);

    // batch matmul
    if (resultType.getRank() == 3) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, ValueRange{lhs, rhs}, outputTensor);
      return success();
    }
    // Create linalg.matmul with inputs and output
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, ValueRange{lhs, rhs}, // inputs
        outputTensor);            // outputs
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
        loc, resultType.getShape(), resultType.getElementType());

    auto indicesType = cast<RankedTensorType>(indices.getType());
    int64_t indicesRank = indicesType.getRank();
    int64_t resRank = resultType.getRank();
    int64_t axis = op.getAxis();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();

    if (axis < 0)
      axis += inputRank;
    // Map for indices: (d0, ..., d_{resRank-1}) -> (d_axis, ..., d_{axis +
    // indicesRank - 1})
    SmallVector<AffineExpr> indicesExprs;
    for (int i = 0; i < indicesRank; ++i) {
      indicesExprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }
    auto indicesMap =
        AffineMap::get(resRank, 0, indicesExprs, rewriter.getContext());

    // Map for output is identity
    auto outMap = rewriter.getMultiDimIdentityMap(resRank);

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(indicesMap);
    indexingMaps.push_back(outMap);

    SmallVector<utils::IteratorType> iteratorTypes(
        resRank, utils::IteratorType::parallel);

    Type indicesElemType = indicesType.getElementType();

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, indices, emptyTensor, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location l, ValueRange args) {
          Value indexVal = args[0];
          if (llvm::isa<FloatType>(indicesElemType)) {
            indexVal = b.create<arith::FPToSIOp>(l, b.getI32Type(), indexVal);
          }

          Value classIdx =
              b.create<arith::IndexCastOp>(l, b.getIndexType(), indexVal);

          SmallVector<Value> extractionIndices;
          // Before axis
          for (int64_t i = 0; i < axis; ++i) {
            extractionIndices.push_back(b.create<linalg::IndexOp>(l, i));
          }
          // At axis
          extractionIndices.push_back(classIdx);
          // After axis
          for (int64_t i = axis + 1; i < inputRank; ++i) {
            extractionIndices.push_back(
                b.create<linalg::IndexOp>(l, i + indicesRank - 1));
          }

          Value extracted =
              b.create<tensor::ExtractOp>(l, input, extractionIndices);
          b.create<linalg::YieldOp>(l, extracted);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct NovaScatterAddOpLowering
    : public OpConversionPattern<nova::ScatterAddOp> {
  using OpConversionPattern<nova::ScatterAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ScatterAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elementTy = resultType.getElementType();
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value indices = adaptor.getIndices();
    Value src = adaptor.getSrc();
    int64_t axis = op.getAxis();
    int64_t inputRank = resultType.getRank();

    if (axis < 0)
      axis += inputRank;

    auto indicesType = cast<RankedTensorType>(indices.getType());
    // check the element type of indices and cast to i32 if it in in float
    auto indicesElemType = indicesType.getElementType();
    Value processedIndices = indices;

    if (llvm::isa<FloatType>(indicesElemType)) {
      // Create target type with same shape but i32 element type
      auto i32Type = rewriter.getI32Type();
      auto castedIndicesType =
          RankedTensorType::get(indicesType.getShape(), i32Type);

      // Cast float indices to i32 using TOSA cast
      processedIndices =
          rewriter.create<tosa::CastOp>(loc, castedIndicesType, indices);
    }

    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcShape = srcType.getShape();
    int64_t srcRank = srcType.getRank();

    // Update indicesType to reflect the processed indices
    auto processedIndicesType =
        cast<RankedTensorType>(processedIndices.getType());

    // 1. Bufferize operands to MemRef
    auto inputMemType = MemRefType::get(resultType.getShape(), elementTy);
    auto srcMemType =
        MemRefType::get(srcType.getShape(), srcType.getElementType());
    auto indicesMemType = MemRefType::get(
        processedIndicesType.getShape(), processedIndicesType.getElementType());

    Value inputMem =
        rewriter.create<ToBufferOp>(loc, inputMemType, input, /*restrict=*/true)
            .getResult();
    Value srcMem =
        rewriter.create<ToBufferOp>(loc, srcMemType, src, /*restrict=*/true)
            .getResult();
    Value indicesMem =
        rewriter
            .create<ToBufferOp>(loc, indicesMemType, processedIndices,
                                /*restrict=*/true)
            .getResult();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // 3. Parallel Loop over ALL dimensions of src
    SmallVector<Value> lowerBounds(srcRank, zero);
    SmallVector<Value> upperBounds;
    for (int64_t i = 0; i < srcRank; ++i) {
      upperBounds.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, srcShape[i]));
    }
    SmallVector<Value> steps(srcRank, one);

    rewriter.create<scf::ParallelOp>(
        loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &b, Location l, ValueRange ivs) {
          Value updateIdx = ivs[axis];

          // Extract Index (already i32 from TOSA cast if was float)
          Value idxVal =
              b.create<memref::LoadOp>(l, indicesMem, ValueRange{updateIdx});
          Value targetIdx =
              b.create<arith::IndexCastOp>(l, b.getIndexType(), idxVal);

          Value val = b.create<memref::LoadOp>(l, srcMem, ivs);

          SmallVector<Value> dstCoords;
          for (int64_t d = 0; d < srcRank; ++d) {
            if (d == axis)
              dstCoords.push_back(targetIdx);
            else
              dstCoords.push_back(ivs[d]);
          }

          arith::AtomicRMWKind kind = llvm::isa<FloatType>(elementTy)
                                          ? arith::AtomicRMWKind::addf
                                          : arith::AtomicRMWKind::addi;
          b.create<memref::AtomicRMWOp>(l, kind, val, inputMem, dstCoords);
          b.create<scf::ReduceOp>(l);
        });

    Value resultTensor =
        rewriter
            .create<ToTensorOp>(loc, resultType, inputMem, /*restrict=*/true)
            .getResult();
    rewriter.replaceOp(op, resultTensor);
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
        loc, resultShape, resultType.getElementType());

    auto transposeOp = rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getInput(), permutedInit, perms);

    // Explicitly set the result type
    transposeOp->getResult(0).setType(resultType);

    return success();
  }
};

struct NovaToDeviceOpLowering : public OpConversionPattern<nova::ToDeviceOp> {
  using OpConversionPattern<nova::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ToDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

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
struct NovaRandomOpLowering : public OpConversionPattern<nova::Rndm2DOp> {
  using OpConversionPattern<nova::Rndm2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(nova::Rndm2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resulttype = cast<mlir::RankedTensorType>(op.getType());
    if (!resulttype) {
      return failure();
    }
    auto loc = op.getLoc();
    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resulttype.getShape(), resulttype.getElementType());
    ValueRange args = op.getOperands();
    // seed= (min+max)^max
    Value sum = rewriter.create<arith::AddFOp>(loc, args[0], args[1]);
    Value mySeed = rewriter.create<math::PowFOp>(loc, sum, args[1]);
    Value seedVal =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI32Type(), mySeed);

    auto linalgop = rewriter.create<linalg::FillRng2DOp>(
        loc, ValueRange{args[0], args[1], seedVal}, ValueRange{output});

    rewriter.replaceOp(op, linalgop);
    return success();
  }
};
//----------------------------------------------------------------
//                          Argmin
//----------------------------------------------------------------
static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<nova::ArgMinOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       cast<FloatType>(elementTy).getFloatSemantics(), false));

  if (isa<nova::ArgMinOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}

inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}
class ArgMinConverter : public OpRewritePattern<nova::ArgMinOp> {
public:
  using OpRewritePattern<nova::ArgMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ArgMinOp argminOp,
                                PatternRewriter &rewriter) const final {
    auto loc = argminOp.getLoc();
    Value input = argminOp.getInput();
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<RankedTensorType>(argminOp.getType());
    auto inElementTy = inputTy.getElementType();
    auto outElementTy = resultTy.getElementType();
    int axis = static_cast<int>(argminOp.getDimension().value());
    auto resultMinTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

    if (!isa<IntegerType>(outElementTy))
      return rewriter.notifyMatchFailure(
          argminOp,
          "nova.arg_min to linalg.* requires integer-like result type");

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index.
    auto emptyTensorIdx = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       outElementTy, dynDims)
                              .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(outElementTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running min.
    auto emptyTensorMin = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       inElementTy, dynDims)
                              .getResult();
    auto fillValueMinAttr =
        createInitialValueForReduceOp(argminOp, inElementTy, rewriter);

    if (!fillValueMinAttr)
      return rewriter.notifyMatchFailure(
          argminOp, "unsupported nova.argmin element type");

    auto fillValueMin =
        rewriter.create<arith::ConstantOp>(loc, fillValueMinAttr);
    auto filledTensorMin =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueMin},
                                    ValueRange{emptyTensorMin})
            .result();

    // We need to reduce along the arg-min axis, with parallel operations along
    // the rest.
    SmallVector<utils::IteratorType, 4> iteratorTypes;
    iteratorTypes.resize(inputTy.getRank(), utils::IteratorType::parallel);
    iteratorTypes[axis] = utils::IteratorType::reduction;

    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    for (int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
      srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (axis != i)
        dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }

    bool didEncounterError = false;
    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs},
                                             rewriter.getContext());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy, resultMinTy}), input,
        ValueRange({filledTensorIdx, filledTensorMin}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, axis));

          Value predicate;
          if (isa<FloatType>(inElementTy)) {
            if (argminOp.getIgnoreNan()) {
              // Only update index & min value for non NaN values. If all
              // values are NaNs, the initial index will be return which is 0.
              predicate = rewriter.create<arith::CmpFOp>(
                  nestedLoc, arith::CmpFPredicate::OLT, newValue, oldValue);
            } else {
              // Update min value if either of the following is true:
              // - new value is bigger
              // - cur min is not NaN and new value is NaN
              Value lt = rewriter.create<arith::CmpFOp>(
                  nestedLoc, arith::CmpFPredicate::ULT, newValue, oldValue);
              Value oldNonNaN = rewriter.create<arith::CmpFOp>(
                  nestedLoc, arith::CmpFPredicate::ORD, oldValue, oldValue);
              predicate = rewriter.create<arith::AndIOp>(
                  nestedLoc, rewriter.getI1Type(), lt, oldNonNaN);
            }
          } else if (isa<IntegerType>(inElementTy)) {
            predicate = rewriter.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::slt, newValue, oldValue);
          } else {
            didEncounterError = true;
            return;
          }

          auto resultMin = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultIndex, resultMin}));
        });

    if (didEncounterError)
      return rewriter.notifyMatchFailure(
          argminOp, "unsupported nova.argmin element type");

    rewriter.replaceOp(argminOp, linalgOp.getResult(0));
    return success();
  }
};

//----------------------------------------------------------------
//                          ArgMax
//----------------------------------------------------------------
static TypedAttr createInitialValueForArgMaxOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<nova::ArgmaxOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy,
        APFloat::getInf(cast<FloatType>(elementTy).getFloatSemantics(), true));

  if (isa<nova::ArgmaxOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}

class ArgMaxConverter : public OpRewritePattern<nova::ArgmaxOp> {
public:
  using OpRewritePattern<nova::ArgmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ArgmaxOp argmaxOp,
                                PatternRewriter &rewriter) const final {
    auto loc = argmaxOp.getLoc();
    Value input = argmaxOp.getInput();

    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<RankedTensorType>(argmaxOp.getType());
    auto inElementTy = inputTy.getElementType();
    auto outElementTy = resultTy.getElementType();
    int axis = static_cast<int>(argmaxOp.getDimension().value_or(0));
    if (axis < 0)
      axis += inputTy.getRank();

    auto resultMaxTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

    if (!isa<IntegerType>(outElementTy))
      return rewriter.notifyMatchFailure(
          argmaxOp,
          "nova.argmax to linalg.* requires integer-like result type");

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index.
    auto emptyTensorIdx = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       outElementTy, dynDims)
                              .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(outElementTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running max.
    auto emptyTensorMax = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       inElementTy, dynDims)
                              .getResult();

    auto fillValueMaxAttr =
        createInitialValueForArgMaxOp(argmaxOp, inElementTy, rewriter);
    auto fillValueMax =
        rewriter.create<arith::ConstantOp>(loc, fillValueMaxAttr);

    auto filledTensorMax =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueMax},
                                    ValueRange{emptyTensorMax})
            .result();

    // We need to reduce along the arg-max axis, with parallel operations
    // along the rest.
    SmallVector<utils::IteratorType, 4> iteratorTypes;
    iteratorTypes.resize(inputTy.getRank(), utils::IteratorType::parallel);
    iteratorTypes[axis] = utils::IteratorType::reduction;

    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    for (int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
      srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (axis != i)
        dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }

    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs},
                                             rewriter.getContext());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy, resultMaxTy}), input,
        ValueRange({filledTensorIdx, filledTensorMax}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, axis));

          Value predicate;
          if (isa<FloatType>(inElementTy)) {
            // For Max: newValue > oldValue
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          } else if (isa<IntegerType>(inElementTy)) {
            predicate = rewriter.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::sgt, newValue, oldValue);
          }

          auto resultMax = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultIndex, resultMax}));
        });

    rewriter.replaceOp(argmaxOp, linalgOp.getResult(0));
    return success();
  }
};

//----------------------------------------------------------------
//                          ReduceOp
//----------------------------------------------------------------
// Helper to get identity value for reduction operations
static TypedAttr getReduceIdentity(nova::ReductionKind kind, Type elemType,
                                   PatternRewriter &rewriter) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    switch (kind) {
    case nova::ReductionKind::SUM:
    case nova::ReductionKind::MEAN:
      return rewriter.getFloatAttr(elemType, 0.0);
    case nova::ReductionKind::PRODUCT:
      return rewriter.getFloatAttr(elemType, 1.0);
    case nova::ReductionKind::MAX:
      return rewriter.getFloatAttr(
          elemType,
          APFloat::getInf(floatType.getFloatSemantics(), /*neg=*/true));
    case nova::ReductionKind::MIN:
      return rewriter.getFloatAttr(
          elemType,
          APFloat::getInf(floatType.getFloatSemantics(), /*neg=*/false));
    default:
      return {};
    }
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    unsigned bw = intType.getWidth();
    switch (kind) {
    case nova::ReductionKind::SUM:
    case nova::ReductionKind::MEAN:
      return rewriter.getIntegerAttr(elemType, 0);
    case nova::ReductionKind::PRODUCT:
      return rewriter.getIntegerAttr(elemType, 1);
    case nova::ReductionKind::MAX:
      return rewriter.getIntegerAttr(elemType, APInt::getSignedMinValue(bw));
    case nova::ReductionKind::MIN:
      return rewriter.getIntegerAttr(elemType, APInt::getSignedMaxValue(bw));
    case nova::ReductionKind::ALL:
      return rewriter.getIntegerAttr(elemType, 1);
    case nova::ReductionKind::ANY:
      return rewriter.getIntegerAttr(elemType, 0);
    }
  }
  return {};
}

// Helper to create reduction combiner operation
static Value createReduceCombiner(OpBuilder &b, Location loc,
                                  nova::ReductionKind kind, Value lhs,
                                  Value rhs, Type elemType) {
  switch (kind) {
  case nova::ReductionKind::SUM:
  case nova::ReductionKind::MEAN:
    return isa<FloatType>(elemType)
               ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::AddIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::PRODUCT:
    return isa<FloatType>(elemType)
               ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MulIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::MAX:
    return isa<FloatType>(elemType)
               ? b.create<arith::MaximumFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::MIN:
    return isa<FloatType>(elemType)
               ? b.create<arith::MinimumFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MinSIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::ALL:
    return b.create<arith::AndIOp>(loc, lhs, rhs).getResult();
  case nova::ReductionKind::ANY:
    return b.create<arith::OrIOp>(loc, lhs, rhs).getResult();
  }
}

// Linalg.generic based lowering for keepdims=true
static LogicalResult
lowerWithLinalgGeneric(nova::ReduceOp op, PatternRewriter &rewriter,
                       Location loc, Value input, RankedTensorType inputType,
                       RankedTensorType resultType, Type elemType, int64_t rank,
                       nova::ReductionKind kind, SmallVector<int64_t> &axes) {

  llvm::sort(axes);
  llvm::SmallDenseSet<int64_t> axisSet(axes.begin(), axes.end());

  // Handle ALL/ANY: cast to i1 first
  Type reductionElemType = elemType;
  Value current = input;
  if (kind == nova::ReductionKind::ALL || kind == nova::ReductionKind::ANY) {
    reductionElemType = rewriter.getI1Type();
    if (elemType != reductionElemType) {
      auto boolType =
          RankedTensorType::get(inputType.getShape(), reductionElemType);
      current = rewriter.create<tosa::CastOp>(loc, boolType, current);
    }
  }

  // Compute total reduced elements for MEAN
  int64_t totalReducedElements = 1;
  for (int64_t axis : axes) {
    totalReducedElements *= inputType.getDimSize(axis);
  }

  // Get identity value
  auto identityAttr = getReduceIdentity(kind, reductionElemType, rewriter);
  if (!identityAttr)
    return rewriter.notifyMatchFailure(op, "unsupported reduction kind");
  Value identity = rewriter.create<arith::ConstantOp>(loc, identityAttr);

  // Compute keepdims output shape (reduced dims become 1)
  SmallVector<int64_t> keepdimsShape;
  for (int64_t i = 0; i < rank; ++i) {
    keepdimsShape.push_back(axisSet.contains(i) ? 1 : inputType.getDimSize(i));
  }

  // Create output tensor with keepdims shape
  auto keepdimsType = RankedTensorType::get(keepdimsShape, reductionElemType);
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, keepdimsShape, reductionElemType, ValueRange{});
  Value filledTensor =
      rewriter.create<linalg::FillOp>(loc, identity, emptyTensor).result();

  // Build indexing maps
  // We want iterators: [parallel_0, ..., parallel_M, reduction_0, ...,
  // reduction_K]
  SmallVector<int64_t> parallelAxes;
  SmallVector<int64_t> reductionAxes;
  for (int64_t i = 0; i < rank; ++i) {
    if (axisSet.contains(i))
      reductionAxes.push_back(i);
    else
      parallelAxes.push_back(i);
  }

  // logicalToLoop[logical_dim] = generic_loop_index
  SmallVector<int64_t> logicalToLoop(rank);
  for (size_t i = 0; i < parallelAxes.size(); ++i)
    logicalToLoop[parallelAxes[i]] = i;
  for (size_t i = 0; i < reductionAxes.size(); ++i)
    logicalToLoop[reductionAxes[i]] = parallelAxes.size() + i;

  SmallVector<AffineExpr> inputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    inputExprs.push_back(rewriter.getAffineDimExpr(logicalToLoop[i]));
  }

  SmallVector<AffineExpr> outputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    if (axisSet.contains(i)) {
      outputExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      outputExprs.push_back(rewriter.getAffineDimExpr(logicalToLoop[i]));
    }
  }

  SmallVector<utils::IteratorType> iteratorTypes;
  for (size_t i = 0; i < parallelAxes.size(); ++i)
    iteratorTypes.push_back(utils::IteratorType::parallel);
  for (size_t i = 0; i < reductionAxes.size(); ++i)
    iteratorTypes.push_back(utils::IteratorType::reduction);

  auto inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
  auto outputMap = AffineMap::get(rank, 0, outputExprs, rewriter.getContext());

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, keepdimsType, current, filledTensor,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value result = createReduceCombiner(b, nestedLoc, kind, args[0],
                                            args[1], reductionElemType);
        b.create<linalg::YieldOp>(nestedLoc, result);
      });

  Value reduced = genericOp.getResult(0);

  // Handle MEAN: divide by total reduced elements
  if (kind == nova::ReductionKind::MEAN && isa<FloatType>(reductionElemType)) {
    double divisor = static_cast<double>(totalReducedElements);
    Value divisorVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(reductionElemType, divisor));

    Value divisorTensor = rewriter.create<tensor::EmptyOp>(
        loc, keepdimsShape, reductionElemType, ValueRange{});
    Value filledDivisor =
        rewriter.create<linalg::FillOp>(loc, divisorVal, divisorTensor)
            .result();

    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, keepdimsShape, reductionElemType, ValueRange{});

    auto divOp = rewriter.create<linalg::GenericOp>(
        loc, keepdimsType, ValueRange{reduced, filledDivisor}, outputTensor,
        maps, getNParallelLoopsAttrs(rank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = b.create<arith::DivFOp>(nestedLoc, args[0], args[1]);
          b.create<linalg::YieldOp>(nestedLoc, result);
        });
    reduced = divOp.getResult(0);
  }

  // Final type cast/reshape if needed
  if (cast<RankedTensorType>(reduced.getType()) != resultType) {
    auto reducedType = cast<RankedTensorType>(reduced.getType());
    if (reducedType.getRank() != resultType.getRank()) {
      auto shapeType = RankedTensorType::get({resultType.getRank()},
                                             rewriter.getIndexType());
      auto shapeAttr =
          DenseIntElementsAttr::get(shapeType, resultType.getShape());
      auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
          loc,
          mlir::tosa::shapeType::get(rewriter.getContext(),
                                     resultType.getRank()),
          shapeAttr);
      reduced = rewriter.create<tosa::ReshapeOp>(loc, resultType, reduced,
                                                 shapeConst);
    } else {
      reduced = rewriter.create<tensor::CastOp>(loc, resultType, reduced);
    }
  }

  rewriter.replaceOp(op, reduced);
  return success();
}

class ReduceOpConverter : public OpRewritePattern<nova::ReduceOp> {
public:
  using OpRewritePattern<nova::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elemType = inputType.getElementType();
    int64_t rank = inputType.getRank();

    nova::ReductionKind kind = op.getKind();

    // Collect reduction axes
    SmallVector<int64_t> axes;
    if (auto dims = op.getDimension()) {
      for (auto attr : *dims) {
        int64_t axis = cast<IntegerAttr>(attr).getInt();
        if (axis < 0)
          axis += rank;
        axes.push_back(axis);
      }
    } else {
      // Reduce all dimensions
      for (int64_t i = 0; i < rank; ++i)
        axes.push_back(i);
    }

    // Use linalg.generic path
    return lowerWithLinalgGeneric(op, rewriter, loc, input, inputType,
                                  resultType, elemType, rank, kind, axes);

    // Sort axes in descending order so we can reduce from higher dims first
    // (this prevents axis index shifting issues)
    llvm::sort(axes, std::greater<int64_t>());

    Value current = input;
    int64_t totalReducedElements = 1;

    // Reduce each axis using TOSA reduce ops
    for (int64_t axis : axes) {
      auto currentType = cast<RankedTensorType>(current.getType());
      Type currentElemType = currentType.getElementType();

      // Compute intermediate result shape (reduced axis becomes 1)
      SmallVector<int64_t> resultShape =
          llvm::to_vector(currentType.getShape());
      totalReducedElements *= resultShape[axis];
      resultShape[axis] = 1;

      auto axisAttr = rewriter.getI32IntegerAttr(axis);

      switch (kind) {
      case nova::ReductionKind::SUM:
      case nova::ReductionKind::MEAN: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceSumOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::MAX: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceMaxOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::MIN: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceMinOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::PRODUCT: {
        auto intermediateType =
            RankedTensorType::get(resultShape, currentElemType);
        current = rewriter.create<tosa::ReduceProductOp>(loc, intermediateType,
                                                         current, axisAttr);
        break;
      }
      case nova::ReductionKind::ALL: {
        // ALL requires boolean input - cast to i1 if needed
        Type i1Type = rewriter.getI1Type();
        if (currentElemType != i1Type) {
          auto boolType = RankedTensorType::get(currentType.getShape(), i1Type);
          current = rewriter.create<tosa::CastOp>(loc, boolType, current);
        }
        auto intermediateType = RankedTensorType::get(resultShape, i1Type);
        current = rewriter.create<tosa::ReduceAllOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      case nova::ReductionKind::ANY: {
        // ANY requires boolean input - cast to i1 if needed
        Type i1Type = rewriter.getI1Type();
        if (currentElemType != i1Type) {
          auto boolType = RankedTensorType::get(currentType.getShape(), i1Type);
          current = rewriter.create<tosa::CastOp>(loc, boolType, current);
        }
        auto intermediateType = RankedTensorType::get(resultShape, i1Type);
        current = rewriter.create<tosa::ReduceAnyOp>(loc, intermediateType,
                                                     current, axisAttr);
        break;
      }
      }
    }

    // Handle MEAN: divide by product of reduced dimensions
    if (kind == nova::ReductionKind::MEAN) {
      auto currentType = cast<RankedTensorType>(current.getType());

      // Create scalar divisor with shape [1, 1, ...] matching current rank
      SmallVector<int64_t> divisorShape(currentType.getRank(), 1);
      auto divisorType = RankedTensorType::get(divisorShape, elemType);

      Value divisor;
      if (llvm::isa<FloatType>(elemType)) {
        auto attr = DenseElementsAttr::get(
            divisorType,
            rewriter.getFloatAttr(elemType, (double)totalReducedElements));
        divisor = rewriter.create<tosa::ConstOp>(loc, divisorType, attr);
      } else {
        auto attr = DenseElementsAttr::get(
            divisorType,
            rewriter.getIntegerAttr(elemType, totalReducedElements));
        divisor = rewriter.create<tosa::ConstOp>(loc, divisorType, attr);
      }

      // Use tosa.reciprocal + tosa.mul instead of division for better
      // performance
      auto reciprocal =
          rewriter.create<tosa::ReciprocalOp>(loc, divisorType, divisor);

      // Create shift tensor for mul
      auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
      auto shiftAttr =
          DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
      auto shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

      current = rewriter.create<tosa::MulOp>(loc, currentType, current,
                                             reciprocal, shift);
    }

    // Reshape to final result shape if needed
    // If keepdims=false, we need to drop the size-1 dimensions
    auto currentType = cast<RankedTensorType>(current.getType());
    if (currentType.getShape() != resultType.getShape()) {
      auto shapeType = RankedTensorType::get({resultType.getRank()},
                                             rewriter.getIndexType());
      auto shapeAttr =
          DenseIntElementsAttr::get(shapeType, resultType.getShape());
      auto shapeConst = rewriter.create<tosa::ConstShapeOp>(
          loc,
          mlir::tosa::shapeType::get(rewriter.getContext(),
                                     resultType.getRank()),
          shapeAttr);
      current = rewriter.create<tosa::ReshapeOp>(loc, resultType, current,
                                                 shapeConst);
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

struct AdamOpConverter : public OpConversionPattern<::mlir::nova::AdamOp> {
  using OpConversionPattern<::mlir::nova::AdamOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::mlir::nova::AdamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value param = adaptor.getParam();
    Value m = adaptor.getM();
    Value v = adaptor.getV();
    Value grad = adaptor.getGrad();

    // Load hyperparameters
    double beta1 = op.getBeta1().convertToDouble();
    double beta2 = op.getBeta2().convertToDouble();
    double epsilon = op.getEpsilon().convertToDouble();
    double lr = op.getLr().convertToDouble();
    int64_t t = op.getT();

    // Team Logic: Compute alpha_eff and bias corrections
    // Note: t is assumed to be the already-incremented step count (1, 2, ...)
    double bias_corr1 = 1.0 - std::pow(beta1, t);
    double bias_corr2 = 1.0 - std::pow(beta2, t);

    // alpha_eff = alpha * sqrt(bias_corr2) / bias_corr1
    double alpha_eff = lr * std::sqrt(bias_corr2) / bias_corr1;
    double sqrt_bias_corr2 = std::sqrt(bias_corr2);

    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    auto elementType = resultType.getElementType();

    // Create separate empty tensors for results
    Value empty_param = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);
    Value empty_m = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                     elementType);
    Value empty_v = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                     elementType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType, resultType, resultType},
        ValueRange{param, m, v, grad},
        ValueRange{empty_param, empty_m, empty_v},
        SmallVector<AffineMap>(
            7, rewriter.getMultiDimIdentityMap(resultType.getRank())),
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value current_p = args[0];
          Value current_m = args[1];
          Value current_v = args[2];
          Value current_g = args[3];

          // Constants based on team implementation
          Value c_beta1 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, beta1));
          Value c_beta2 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, beta2));
          Value c_one_minus_beta1 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, 1.0 - beta1));
          Value c_one_minus_beta2 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, 1.0 - beta2));
          Value c_alpha_eff = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, alpha_eff));
          Value c_epsilon = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, epsilon));
          Value c_sqrt_bias_corr2 = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, sqrt_bias_corr2));

          // 1. Update first moment: m = beta1*m + (1-beta1)*grad
          Value m_term1 = b.create<arith::MulFOp>(loc, current_m, c_beta1);
          Value m_term2 =
              b.create<arith::MulFOp>(loc, current_g, c_one_minus_beta1);
          Value m_new = b.create<arith::AddFOp>(loc, m_term1, m_term2);

          // 2. Update second moment: v = beta2*v + (1-beta2)*grad^2
          Value g_sq = b.create<arith::MulFOp>(loc, current_g, current_g);
          Value v_term1 = b.create<arith::MulFOp>(loc, current_v, c_beta2);
          Value v_term2 = b.create<arith::MulFOp>(loc, g_sq, c_one_minus_beta2);
          Value v_new = b.create<arith::AddFOp>(loc, v_term1, v_term2);

          // 3. Update parameter: param -= alpha_eff * m / (sqrt(v) + epsilon *
          // sqrt(bias_corr2))
          Value sqrt_v = b.create<math::SqrtOp>(loc, v_new);
          Value eps_term =
              b.create<arith::MulFOp>(loc, c_epsilon, c_sqrt_bias_corr2);
          Value denom = b.create<arith::AddFOp>(loc, sqrt_v, eps_term);

          Value num = b.create<arith::MulFOp>(loc, c_alpha_eff, m_new);
          Value update = b.create<arith::DivFOp>(loc, num, denom);

          Value p_new = b.create<arith::SubFOp>(loc, current_p, update);

          b.create<linalg::YieldOp>(loc, ValueRange{p_new, m_new, v_new});
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
struct ReshapeOpConverter : public OpConversionPattern<nova::ReshapeOp> {
  using OpConversionPattern<nova::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto resultType = cast<RankedTensorType>(op.getType());
    Location loc = op.getLoc();

    // Create a constant tensor for the output shape
    auto resultShape = resultType.getShape();
    auto shapeType =
        RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType());

    // Convert shape to attribute
    auto shapeAttr = rewriter.getIndexTensorAttr(resultShape);
    auto shapeConst =
        rewriter.create<arith::ConstantOp>(loc, shapeType, shapeAttr);

    // Create tensor.reshape
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType, input,
                                                   shapeConst);
    return success();
  }
};

struct NovaLinearOpLowering : public OpConversionPattern<nova::LinearOp> {
  using OpConversionPattern<nova::LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nova::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    if (operands.size() != 3) {
      return rewriter.notifyMatchFailure(op, "expected exactly 3 operands");
    }

    Value input = operands[0];
    Value weight = operands[1];
    Value bias = operands[2];

    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto weightType = llvm::dyn_cast<RankedTensorType>(weight.getType());
    auto biasType = llvm::dyn_cast<RankedTensorType>(bias.getType());

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !weightType || !biasType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }

    auto loc = op.getLoc();
    auto elementType = resultType.getElementType();
    int64_t inputRank = inputType.getRank();

    // 1. Initialize output with Bias
    SmallVector<Value> dynDims;
    for (int i = 0; i < inputRank; ++i) {
      if (resultType.isDynamicDim(i)) {
        // Find dim dynamically. We could just use tensor::DimOp if input has
        // dynamic dims matching result.
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   elementType, dynDims);

    // Map for Bias to Broadcast: bias is 1D -> (C)
    // Result is [A, B, ..., C]
    SmallVector<AffineExpr> biasExprs = {
        rewriter.getAffineDimExpr(inputRank - 1)};
    AffineMap biasMap =
        AffineMap::get(inputRank, 0, biasExprs, rewriter.getContext());
    AffineMap resultMap = rewriter.getMultiDimIdentityMap(inputRank);

    SmallVector<utils::IteratorType> broadcastIters(
        inputRank, utils::IteratorType::parallel);

    auto broadcastBias = rewriter.create<linalg::GenericOp>(
        loc, empty.getType(), bias, empty,
        ArrayRef<AffineMap>{biasMap, resultMap}, broadcastIters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });

    Value initTensor = broadcastBias.getResult(0);

    // 2. Fused Matmul+Bias
    // Iterators: [parallel..., parallel, reduction]
    SmallVector<utils::IteratorType> matmulIters(inputRank,
                                                 utils::IteratorType::parallel);
    matmulIters.push_back(utils::IteratorType::reduction);

    int numLoops = inputRank + 1;

    // Input Map: D, A, B -> [d0, d1, d3] (where d3 is B, the reduction dim)
    SmallVector<AffineExpr> inputExprs;
    for (int i = 0; i < inputRank - 1; ++i) {
      inputExprs.push_back(
          rewriter.getAffineDimExpr(i)); // Parallel dims of LHS
    }
    inputExprs.push_back(
        rewriter.getAffineDimExpr(inputRank)); // Reduction dim K
    AffineMap inputMap =
        AffineMap::get(numLoops, 0, inputExprs, rewriter.getContext());

    // Weight Map: B, C -> [d3, d2] (where d3 is reduction dim K, d2 is C, the
    // last parallel dim)
    SmallVector<AffineExpr> weightExprs;
    weightExprs.push_back(rewriter.getAffineDimExpr(inputRank));     // B (K)
    weightExprs.push_back(rewriter.getAffineDimExpr(inputRank - 1)); // C (N)
    AffineMap weightMap =
        AffineMap::get(numLoops, 0, weightExprs, rewriter.getContext());

    // Output Map: D, A, C -> [d0, d1, d2]
    SmallVector<AffineExpr> outputExprs;
    for (int i = 0; i < inputRank; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i)); // Parallel dims
    }
    AffineMap outputMap =
        AffineMap::get(numLoops, 0, outputExprs, rewriter.getContext());

    auto genericMatmul = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input, weight}, initTensor,
        ArrayRef<AffineMap>{inputMap, weightMap, outputMap}, matmulIters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value inVal = args[0];
          Value wtVal = args[1];
          Value outVal = args[2];
          Value prod;
          if (isa<FloatType>(elementType))
            prod = b.create<arith::MulFOp>(loc, inVal, wtVal);
          else
            prod = b.create<arith::MulIOp>(loc, inVal, wtVal);

          Value sum;
          if (isa<FloatType>(elementType))
            sum = b.create<arith::AddFOp>(loc, outVal, prod);
          else
            sum = b.create<arith::AddIOp>(loc, outVal, prod);

          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericMatmul.getResult(0));
    return success();
  }
};

void populateNovaToLinalgPatterns(RewritePatternSet &patterns) {
  patterns
      .add<NovaMatmulOpLowering, NovaBroadcastInDimOpLowering,
           NovaTransposeOpLowering, NovaToDeviceOpLowering,
           NovaScatterAddOpLowering, NovaGatherOpLowering, NovaRandomOpLowering,
           ArgMinConverter, ArgMaxConverter, ReduceOpConverter, AdamOpConverter,
           ReshapeOpConverter, NovaLinearOpLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// NovaToLinalg Pass (structural ops only)
//===----------------------------------------------------------------------===//

struct NovaToLinalgPass
    : public PassWrapper<NovaToLinalgPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, tensor::TensorDialect,
                arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect>();
  }

  StringRef getArgument() const final { return "convert-nova-to-linalg"; }

  StringRef getDescription() const final {
    return "Lower Nova structural ops (matmul, gather, transpose, etc.) to "
           "Linalg";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();
    ConversionTarget target(*context);

    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::bufferization::BufferizationDialect>();

    // Only the structural ops this file has patterns for are illegal.
    // Elementwise ops and nova.reduce remain legal here (handled elsewhere).
    target.addIllegalOp<nova::MatmulOp, nova::BroadcastInDimOp,
                        nova::TransposeOp, nova::ToDeviceOp, nova::ScatterAddOp,
                        nova::GatherOp, nova::Rndm2DOp, nova::ReshapeOp,
                        nova::ArgMinOp, nova::ArgmaxOp, nova::ReduceOp,
                        nova::AdamOp, nova::LinearOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    populateNovaToLinalgPatterns(patterns);
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

void registerNovaToLinalgPass() { PassRegistration<NovaToLinalgPass>(); }

std::unique_ptr<Pass> createNovaToLinalgPass() {
  return std::make_unique<NovaToLinalgPass>();
}

} // namespace nova
} // namespace mlir