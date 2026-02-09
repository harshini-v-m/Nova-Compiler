#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "Compiler/Transforms/Passes.h"

namespace mlir {
namespace nova {
#define GEN_PASS_DECL_VECTOROPT
#define GEN_PASS_DEF_VECTOROPT
#include "Compiler/Transforms/Passes.h.inc"
} // namespace nova
} // namespace mlir

using namespace mlir;

namespace {

// Pattern to convert affine.for reduction loop with mulf/addf to vector.contract
struct FuseMulAddToContractPattern : public OpRewritePattern<affine::AffineForOp> {
  FuseMulAddToContractPattern(MLIRContext *context, int unrollFactor)
      : OpRewritePattern<affine::AffineForOp>(context), unrollFactor(unrollFactor) {}

  int unrollFactor;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp, PatternRewriter &rewriter) const override {
    // 1. Check if the loop has exactly one iter_arg (the accumulator)
    if (forOp.getRegionIterArgs().size() != 1)
      return failure();

    Block *body = forOp.getBody();
    auto yieldOp = cast<affine::AffineYieldOp>(body->getTerminator());
    Value yieldVal = yieldOp.getOperand(0);

    // 2. Look for the addf operation
    auto addOp = yieldVal.getDefiningOp<arith::AddFOp>();
    if (!addOp)
      return failure();

    // The accumulator should be one of the operands
    Value iterArg = forOp.getRegionIterArgs()[0];
    Value mulResult;
    if (addOp.getLhs() == iterArg) {
      mulResult = addOp.getRhs();
    } else if (addOp.getRhs() == iterArg) {
      mulResult = addOp.getLhs();
    } else {
      return failure();
    }

    // 3. Look for the mulf operation
    auto mulOp = mulResult.getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();

    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    // 4. Verify operands are vector types
    auto lhsType = dyn_cast<VectorType>(lhs.getType());
    auto rhsType = dyn_cast<VectorType>(rhs.getType());
    auto accType = dyn_cast<VectorType>(iterArg.getType());

    if (!lhsType || !rhsType || !accType)
      return failure();

    
    // Fix: Set insertion point to the addOp inside the loop body
    rewriter.setInsertionPoint(addOp);



    auto vecType = cast<VectorType>(lhsType);
    int64_t rank = vecType.getRank();

    Value lhsInput = lhs;
    Value rhsInput = rhs;
    Value accInput = iterArg;
    
    SmallVector<AffineMap, 3> maps;
    SmallVector<Attribute, 3> iteratorTypes;
    VectorType resType;

    // Advanced: Check if we can upgrade 1D broadcasts (from vectorizer) to 2D Blocked Loads
    // This happens if the input is a broadcast (Rank 1 logic) or explicit broadcast (Rank 2 logic with permutation)
    // We want to achieve K=8 blocked contraction if possible.
    
    // Check definition of LHS and RHS
    auto lhsDefOp = lhs.getDefiningOp<vector::TransferReadOp>();
    auto rhsDefOp = rhs.getDefiningOp<vector::TransferReadOp>();
    
    bool canUpgradeToBlocked = false;
    if (lhsDefOp && rhsDefOp && rank == 2) {
       // Check if maps are broadcasting
       // LHS should be (d0, d1) -> (d0, 0) [Col Broadcast] -> Result 1 is constant
       // RHS should be (d0, d1) -> (0, d1) [Row Broadcast] -> Result 0 is constant
       
       auto lhsMap = lhsDefOp.getPermutationMap();
       auto rhsMap = rhsDefOp.getPermutationMap();
       
       bool lhsIsBroadcast = false;
       bool rhsIsBroadcast = false;

       if (lhsMap.getNumResults() == 2) {
           if (auto cst = dyn_cast<AffineConstantExpr>(lhsMap.getResult(1))) {
                if (cst.getValue() == 0) lhsIsBroadcast = true;
           }
       }
       if (rhsMap.getNumResults() == 2) {
           if (auto cst = dyn_cast<AffineConstantExpr>(rhsMap.getResult(0))) {
                if (cst.getValue() == 0) rhsIsBroadcast = true;
           }
       }
       
       if (lhsIsBroadcast && rhsIsBroadcast) {
           // We found the pattern! The vectorizer loaded slices.
           // Let's try to rewrite them to load FULL TILES.
           // Original: vector<8x8> (broadcasted from 8x1)
           // Desired:  vector<8x8> (contiguous 8x8 tile)
            auto lb = forOp.getConstantLowerBound();
            auto ub = forOp.getConstantUpperBound();
            auto step = forOp.getStep().getSExtValue();

            // Use unrollFactor directly for the reduction loop
            int64_t localUnrollFactor = unrollFactor;
            int64_t blockFactor = vecType.getDimSize(vecType.getRank() - 1); // e.g. 16 for vector<16x16>
            int64_t totalStride = blockFactor * localUnrollFactor; 
            
            // We assume step is 1 originally
            if (step == 1 && localUnrollFactor > 1) {
              // Upgrading to Step N*blockFactor
              // Upgrading to Step N*blockFactor
              
              // Initial Accumulator is the operand passed to the loop (outside value)
              iterArg = forOp.getInits()[0];

              // Create Main Loop
              // Range: [lb, ub - (ub-lb)%totalStride]
              int64_t tripCount = ub - lb;
              int64_t mainUb = lb + (tripCount / totalStride) * totalStride;
             
             if (mainUb > lb) {
                 rewriter.setInsertionPoint(forOp);
                 // Fix: Must pass initial values (operands) not block args for new loop creation
                 auto mainLoop = rewriter.create<affine::AffineForOp>(forOp.getLoc(), lb, mainUb, totalStride, forOp.getInits());
                 mainLoop->setAttr("vector_opt.unrolled", rewriter.getUnitAttr());
                 
                 // Generate Body for Main Loop (4x Unroll)
                 // Use PatternRewriter inside the loop
                 // Save insertion point to restore later? No need, we return success().
                 
                 Block *body = mainLoop.getBody();
                 rewriter.setInsertionPointToStart(body);
                 
                 Value mainAcc = mainLoop.getRegionIterArgs()[0];
                  Value iv = mainLoop.getInductionVar();
                  
                  // Generate N contracts
                  for (int i = 0; i < localUnrollFactor; ++i) {
                      int64_t offset = i * blockFactor;
                     
                     // Create Identity Map (d0, d1) -> (d0, d1)
                     auto identityMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

                     // Rewrite LHS Load (Offset iv + offset)

                     
                     // We need to clone the transfer read but replace the K index.
                     SmallVector<Value, 4> newLhsIndices(lhsDefOp.getIndices().begin(), lhsDefOp.getIndices().end());
                     SmallVector<Value, 4> newRhsIndices(rhsDefOp.getIndices().begin(), rhsDefOp.getIndices().end());
                     
                     AffineExpr d0 = rewriter.getAffineDimExpr(0);
                     auto mapPlusOffset = AffineMap::get(1, 0, d0 + offset);
                     Value ivPlusOffset = rewriter.create<affine::AffineApplyOp>(forOp.getLoc(), mapPlusOffset, iv);

                     // Replace occurrences of old IV in indices
                     for (auto &idx : newLhsIndices) if (idx == forOp.getInductionVar()) idx = ivPlusOffset;
                     for (auto &idx : newRhsIndices) if (idx == forOp.getInductionVar()) idx = ivPlusOffset;

                     auto newLhs = rewriter.create<vector::TransferReadOp>(
                        lhsDefOp.getLoc(), 
                        vecType, 
                        lhsDefOp.getBase(), 
                        newLhsIndices, 
                        identityMap, 
                        lhsDefOp.getPadding(), 
                        /*mask=*/Value(), 
                        lhsDefOp.getInBoundsAttr());

                     auto newRhs = rewriter.create<vector::TransferReadOp>(
                        rhsDefOp.getLoc(), 
                        vecType, 
                        rhsDefOp.getBase(), 
                        newRhsIndices, 
                        identityMap, 
                        rhsDefOp.getPadding(), 
                        /*mask=*/Value(), 
                        rhsDefOp.getInBoundsAttr());

                     // Contract K=8
                     SmallVector<AffineMap, 3> maps;
                     auto ctx = rewriter.getContext();
                     auto m = getAffineDimExpr(0, ctx);
                     auto n = getAffineDimExpr(1, ctx);
                     auto k = getAffineDimExpr(2, ctx);
                     maps.push_back(AffineMap::get(3, 0, {m, k}, ctx));
                     maps.push_back(AffineMap::get(3, 0, {k, n}, ctx));
                     maps.push_back(AffineMap::get(3, 0, {m, n}, ctx));
                     
                     SmallVector<Attribute, 3> iterTypes;
                     iterTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel));
                     iterTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel));
                     iterTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction));
                     
                     auto contract = rewriter.create<vector::ContractionOp>(
                         addOp.getLoc(), newLhs, newRhs, mainAcc,
                         rewriter.getAffineMapArrayAttr(maps), rewriter.getArrayAttr(iterTypes));
                         
                     mainAcc = contract.getResult();
                 }
                 
                 rewriter.create<affine::AffineYieldOp>(forOp.getLoc(), mainAcc);
                 
                 // Update iterArg for potential cleanup loop
                 iterArg = mainLoop.getResult(0);
                 rewriter.setInsertionPointAfter(mainLoop);
             }
             
             // Create Cleanup Loop (Step 16)
             // Range: [mainUb, ub]
              if (mainUb < ub) {
                  auto cleanupLoop = rewriter.create<affine::AffineForOp>(forOp.getLoc(), mainUb, ub, blockFactor, iterArg);
                  cleanupLoop->setAttr("vector_opt.unrolled", rewriter.getUnitAttr());
                 
                 Block *body = cleanupLoop.getBody();
                 rewriter.setInsertionPointToStart(body);
                 
                 Value cleanupAcc = cleanupLoop.getRegionIterArgs()[0];
                 Value iv = cleanupLoop.getInductionVar();
                 
                 // Single Contract (K=8)
                 
                     // Create Identity Map (d0, d1) -> (d0, d1)
                     auto identityMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

                     // Indices use IV directly
                     SmallVector<Value, 4> newLhsIndices(lhsDefOp.getIndices().begin(), lhsDefOp.getIndices().end());
                     SmallVector<Value, 4> newRhsIndices(rhsDefOp.getIndices().begin(), rhsDefOp.getIndices().end());
                     for (auto &idx : newLhsIndices) if (idx == forOp.getInductionVar()) idx = iv;
                     for (auto &idx : newRhsIndices) if (idx == forOp.getInductionVar()) idx = iv;

                     // CRITICAL FIX: The cleanup loop is handling the tail (out-of-bounds reads).
                     // We must force 'in_bounds' to false for the reduction dimension (K) 
                     // to use padding (0.0) instead of segfaulting.
                     // LHS (M, K) -> [true, false]
                     // RHS (K, N) -> [false, true]
                     // Note: We assume M and N loops are tiled correctly (divisible by 8) for now.
                     
                     SmallVector<bool, 2> lhsInBounds = {true, false};
                     SmallVector<bool, 2> rhsInBounds = {false, true};
                     
                     auto newLhs = rewriter.create<vector::TransferReadOp>(
                        lhsDefOp.getLoc(), 
                        lhsType, 
                        lhsDefOp.getBase(), 
                        newLhsIndices, 
                        identityMap, 
                        lhsDefOp.getPadding(), 
                        /*mask=*/Value(), 
                        rewriter.getBoolArrayAttr(lhsInBounds));

                     auto newRhs = rewriter.create<vector::TransferReadOp>(
                        rhsDefOp.getLoc(), 
                        rhsType, 
                        rhsDefOp.getBase(), 
                        newRhsIndices, 
                        identityMap, 
                        rhsDefOp.getPadding(), 
                        /*mask=*/Value(), 
                        rewriter.getBoolArrayAttr(rhsInBounds));

                     // Contract K=8
                     SmallVector<AffineMap, 3> maps;
                     auto ctx = rewriter.getContext();
                     auto m = getAffineDimExpr(0, ctx);
                     auto n = getAffineDimExpr(1, ctx);
                     auto k = getAffineDimExpr(2, ctx);
                     maps.push_back(AffineMap::get(3, 0, {m, k}, ctx));
                     maps.push_back(AffineMap::get(3, 0, {k, n}, ctx));
                     maps.push_back(AffineMap::get(3, 0, {m, n}, ctx));
                     
                     SmallVector<Attribute, 3> iterTypes;
                     iterTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel));
                     iterTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel));
                     iterTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction));
                     
                     auto contract = rewriter.create<vector::ContractionOp>(
                         addOp.getLoc(), newLhs, newRhs, cleanupAcc,
                         rewriter.getAffineMapArrayAttr(maps), rewriter.getArrayAttr(iterTypes));
                 
                 rewriter.create<affine::AffineYieldOp>(forOp.getLoc(), contract.getResult());
                 
                 iterArg = cleanupLoop.getResult(0); // For final replacement
             }
             
             // Replace Everything
             rewriter.replaceOp(forOp, iterArg);
             return success();
           } else {
             // If step is already modified or non-standard, fallback to just K=8 (non-unrolled or previously handled)
             // For now, let's just stick to the single loop update if step > 1
             if (forOp.getStep().getSExtValue() < blockFactor) forOp.setStep(blockFactor);
             canUpgradeToBlocked = true;
           }

           // Set insertion point back
           rewriter.setInsertionPoint(addOp);
       }
    }
    
    if (canUpgradeToBlocked) {
       // K=8 logic for fallback cases

       // LHS (M, K) [8x8], RHS (K, N) [8x8], Acc (M, N) [8x8]
       
       auto ctx = rewriter.getContext();
       auto m = getAffineDimExpr(0, ctx);
       auto n = getAffineDimExpr(1, ctx);
       auto k = getAffineDimExpr(2, ctx);
      
       maps.push_back(AffineMap::get(3, 0, {m, k}, ctx)); // LHS: (d0, d2)
       maps.push_back(AffineMap::get(3, 0, {k, n}, ctx)); // RHS: (d2, d1)
       maps.push_back(AffineMap::get(3, 0, {m, n}, ctx)); // Acc: (d0, d1)
       
       // Iterators: Parallel, Parallel, Reduction
       iteratorTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel)); 
       iteratorTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel)); 
       iteratorTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction));

       resType = vecType;
       
    } else if (rank == 1) {
      // 1D Case: Cast to 2D (1xN) and dummy reduction
      int64_t width = vecType.getDimSize(0);
      SmallVector<int64_t, 2> type2DConfig = {1, width};
      auto type2D = VectorType::get(type2DConfig, vecType.getElementType());

      lhsInput = rewriter.create<vector::ShapeCastOp>(addOp.getLoc(), type2D, lhs);
      rhsInput = rewriter.create<vector::ShapeCastOp>(addOp.getLoc(), type2D, rhs);
      accInput = rewriter.create<vector::ShapeCastOp>(addOp.getLoc(), type2D, iterArg);
      
      auto map = rewriter.getMultiDimIdentityMap(2);
      maps.push_back(map);
      maps.push_back(map);
      maps.push_back(map);

      iteratorTypes.push_back(vector::IteratorTypeAttr::get(rewriter.getContext(), vector::IteratorType::parallel));
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(rewriter.getContext(), vector::IteratorType::parallel));
      
      resType = type2D;
    } else if (rank == 2) {
      // 2D Case: NxN with explicit broadcast structure
      // We assume LHS is (N, 1) broadcasted to (N, N) -> Take column 0
      // We assume RHS is (1, N) broadcasted to (N, N) -> Take row 0
      
      int64_t dim0 = vecType.getDimSize(0);
      int64_t dim1 = vecType.getDimSize(1);
      
      // Target shapes: LHS (dim0, 1), RHS (1, dim1)
      SmallVector<int64_t, 2> lhsSliceShape = {dim0, 1};
      SmallVector<int64_t, 2> rhsSliceShape = {1, dim1};
      
      auto lhsSliceType = VectorType::get(lhsSliceShape, vecType.getElementType());
      auto rhsSliceType = VectorType::get(rhsSliceShape, vecType.getElementType());
      
      // Extract slices from (0,0)
      // LHS: sizes = [dim0, 1], strides = [1, 1]
      SmallVector<int64_t, 2> offsets = {0, 0};
      SmallVector<int64_t, 2> lhsSizes = {dim0, 1};
      SmallVector<int64_t, 2> rhsSizes = {1, dim1};
      SmallVector<int64_t, 2> strides = {1, 1};

      lhsInput = rewriter.create<vector::ExtractStridedSliceOp>(
          addOp.getLoc(), lhsSliceType, lhs, 
          rewriter.getI64ArrayAttr(offsets), 
          rewriter.getI64ArrayAttr(lhsSizes), 
          rewriter.getI64ArrayAttr(strides));
          
      rhsInput = rewriter.create<vector::ExtractStridedSliceOp>(
          addOp.getLoc(), rhsSliceType, rhs, 
          rewriter.getI64ArrayAttr(offsets), 
          rewriter.getI64ArrayAttr(rhsSizes), 
          rewriter.getI64ArrayAttr(strides));
          
      accInput = iterArg; // Accumulator is already (dim0, dim1)
      
      // Maps for (m, n, k) -> (m, k), (k, n), (m, n) 
      // k is dim 2 (reduction)
      auto ctx = rewriter.getContext();
      auto m = getAffineDimExpr(0, ctx);
      auto n = getAffineDimExpr(1, ctx);
      auto k = getAffineDimExpr(2, ctx);
      
      maps.push_back(AffineMap::get(3, 0, {m, k}, ctx)); // LHS
      maps.push_back(AffineMap::get(3, 0, {k, n}, ctx)); // RHS
      maps.push_back(AffineMap::get(3, 0, {m, n}, ctx)); // Acc/Res
      
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel)); // m
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel)); // n
      iteratorTypes.push_back(vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)); // k

      resType = vecType; // Result matches accumulator
    } else {
      return failure();
    }

    // Create contract conversion
    auto contractOp = rewriter.create<vector::ContractionOp>(
        addOp.getLoc(),
        lhsInput, rhsInput, accInput,
        rewriter.getAffineMapArrayAttr(maps),
        rewriter.getArrayAttr(iteratorTypes)
    );
     
    Value result = contractOp.getResult();
    
    // If we casted for 1D, cast back
    if (rank == 1) {
       result = rewriter.create<vector::ShapeCastOp>(addOp.getLoc(), vecType, result);
    }
    
    rewriter.replaceOp(addOp, result);
    return success();
  }
};

struct VectorOptPass : public nova::impl::VectorOptBase<VectorOptPass> {
  using nova::impl::VectorOptBase<VectorOptPass>::VectorOptBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseMulAddToContractPattern>(context, unrollFactor);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace nova {
std::unique_ptr<Pass> createVectorOptPass() {
  return std::make_unique<VectorOptPass>();
}
} // namespace nova
} // namespace mlir
