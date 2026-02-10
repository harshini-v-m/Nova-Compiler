#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
//check the element type of indices and cast to i32 if it in in float
    auto indicesElemType = indicesType.getElementType();
    Value processedIndices = indices;
    
    if (llvm::isa<FloatType>(indicesElemType)) {
      // Create target type with same shape but i32 element type
      auto i32Type = rewriter.getI32Type();
      auto castedIndicesType = RankedTensorType::get(
          indicesType.getShape(), i32Type);
      
      // Cast float indices to i32 using TOSA cast
      processedIndices = rewriter.create<tosa::CastOp>(
          loc, castedIndicesType, indices);
    }
    
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcShape = srcType.getShape();
    int64_t srcRank = srcType.getRank();
    
    // Update indicesType to reflect the processed indices
    auto processedIndicesType = cast<RankedTensorType>(processedIndices.getType());
    
    // 1. Bufferize operands to MemRef
    auto inputMemType = MemRefType::get(resultType.getShape(), elementTy);
    auto srcMemType =
        MemRefType::get(srcType.getShape(), srcType.getElementType());
    auto indicesMemType =
        MemRefType::get(processedIndicesType.getShape(), processedIndicesType.getElementType());


    Value inputMem =
        rewriter.create<ToBufferOp>(loc, inputMemType, input, /*restrict=*/true)
            .getResult();
    Value srcMem =
        rewriter.create<ToBufferOp>(loc, srcMemType, src, /*restrict=*/true)
            .getResult();
    Value indicesMem =
        rewriter
            .create<ToBufferOp>(loc, indicesMemType, processedIndices, /*restrict=*/true)
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
void populateNovaToLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaMatmulOpLowering, NovaBroadcastInDimOpLowering,
               NovaTransposeOpLowering, NovaToDeviceOpLowering,
               NovaScatterAddOpLowering, NovaGatherOpLowering,
               NovaRandomOpLowering>(patterns.getContext());
}
} // namespace nova
} // namespace mlir