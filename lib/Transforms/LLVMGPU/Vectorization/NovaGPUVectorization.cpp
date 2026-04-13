//===- NovaGPUVectorization.cpp - Nova GPU Vectorization Passes -----------===//
//
// Defines the vectorization and hoisting passes for the Nova GPU pipeline,
// following the IREE-style SIMD-to-SIMT strategy.
//
// Passes defined here:
//   NovaGPUPackToIntrinsicsPass     — pack linalg.generic contractions into
//                                     MMA-compatible inner tile shapes
//   NovaGPUGenericVectorizationPass — linalg → vector dialect (SIMD)
//   NovaGPUSubsetHoistingPass       — hoist redundant transfers out of loops
//   NovaGPUVectorDistributePass     — distribute warp-level vectors to threads
//                                     using vector.warp_execute_on_lane_0
//   NovaGPUVectorizeMemrefCopyPass  — vectorize memref.copy (global→shared)
//   NovaGPUUnrollToIntrinsicsPass   — unroll vector.contract to MMA-native
//                                     shape read from the op's lowering_config
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the MMA intrinsic shape {M, N, K} for the given intrinsic enum
/// value. Returns {16, 16, 16} (WMMA default) if unrecognised.
static SmallVector<int64_t, 3> getMMAShape(int32_t mmaKindRaw) {
  switch (static_cast<NVMMAIntrinsicValues>(mmaKindRaw)) {
    case NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16:
    case NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16:
      return {16, 8, 16};
    case NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8:
      return {16, 8, 8};
    case NVMMAIntrinsicValues::WMMA_TF32_16x16x8:
      return {16, 16, 8};
    case NVMMAIntrinsicValues::WMMA_F32_16x16x16:
    case NVMMAIntrinsicValues::WMMA_F16_16x16x16:
      return {16, 16, 16};
    case NVMMAIntrinsicValues::NONE:
    default:
      // Default fallback for NVIDIA Tensor Cores if no specific kind matched.
      return {16, 16, 16};
  }
}

//===----------------------------------------------------------------------===//
// NovaGPUPackToIntrinsicsPass
//
// Repacks linalg.generic matmul ops whose lowering_config carries a non-zero
// mma_kind into inner tiles matching the MMA intrinsic shape.
//
// Before packing:  linalg.generic<(M, N, K)>  (arbitrary tile sizes)
// After packing:   linalg.generic<(M/m, N/n, K/k, m, n, k)>
//                  where {m, n, k} = getMMAShape(mma_kind)
//
// This is a prerequisite for NovaGPUUnrollToIntrinsicsPass — the unroll pass
// can only fire once the inner contraction loop has the exact native MMA shape.
// Mirrors IREE's GPUTensorTileToSerialLoops + GPUPadMultiByList sequence.
//
// NOTE: This pass is currently bypassed in the main pipeline because linalg.pack
// creates rank-6 tensors that the linalg vectorizer cannot handle as contractions.
// We instead vectorize the rank-2 op first and then unroll at the vector level.
//===----------------------------------------------------------------------===//

struct NovaGPUPackToIntrinsicsPass
    : public PassWrapper<NovaGPUPackToIntrinsicsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUPackToIntrinsicsPass)

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    SmallVector<linalg::LinalgOp> contractions;
    funcOp.walk([&](linalg::LinalgOp op) {
      // Only process ops that have an MMA kind set.
      DictionaryAttr cfg = getLoweringConfig(op);
      if (!cfg) return;
      int32_t mmaKind = getMmaKindRaw(cfg);
      if (mmaKind == 0) return;
      // Only matmul-like contractions (exactly one reduction iterator).
      unsigned numReduction = 0;
      for (utils::IteratorType it : op.getIteratorTypesArray())
        if (it == utils::IteratorType::reduction) ++numReduction;
      if (numReduction != 1) return;
      contractions.push_back(op);
    });

    for (linalg::LinalgOp op : contractions) {
      DictionaryAttr cfg = getLoweringConfig(op);
      SmallVector<int64_t, 3> mmaShape = getMMAShape(getMmaKindRaw(cfg));

      // Build innerTiles: one entry per loop dimension.
      // For a rank-3 (M, N, K) contraction: pack the last three dims.
      // For rank > 3 (batch dims present): skip leading batch dims.
      unsigned rank = op.getNumLoops();
      if (rank < 3) continue;

      // Pack the trailing M, N, K dims with the MMA tile sizes.
      // linalg::pack expects one inner-tile size per dimension to pack.
      // We pack dims [rank-3, rank-2, rank-1] with sizes mmaShape[0..2].
      SmallVector<OpFoldResult> innerTiles;
      innerTiles.reserve(rank);
      for (unsigned i = 0; i < rank - 3; ++i)
        innerTiles.push_back(rewriter.getIndexAttr(1)); // batch: tile by 1
      innerTiles.push_back(rewriter.getIndexAttr(mmaShape[0])); // M
      innerTiles.push_back(rewriter.getIndexAttr(mmaShape[1])); // N
      innerTiles.push_back(rewriter.getIndexAttr(mmaShape[2])); // K

      rewriter.setInsertionPoint(op);
      FailureOr<linalg::PackResult> packResult =
          linalg::pack(rewriter, op, innerTiles);
      if (failed(packResult)) continue;

      // Propagate lowering_config to the replacement op.
      if (packResult->packedLinalgOp)
        setLoweringConfig(packResult->packedLinalgOp, cfg);
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-pack-to-intrinsics";
  }
  StringRef getDescription() const override {
    return "Pack linalg contraction ops into MMA-compatible inner tile shapes.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUGenericVectorizationPass
//===----------------------------------------------------------------------===//

struct NovaGPUGenericVectorizationPass
    : public PassWrapper<NovaGPUGenericVectorizationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUGenericVectorizationPass)
  // Detect whether a linalg.generic has the standard matmul contraction body:
  //   %0 = arith.mulf %arg0, %arg1 : f32
  //   %1 = arith.addf %arg2, %0   : f32   (or arith.addf %0, %arg2)
  //   linalg.yield %1
  static bool isMatmulBody(linalg::GenericOp op) {
    Block &body = op.getRegion().front();
    SmallVector<Operation*> ops;
    for (Operation &innerOp : body)
      if (!isa<linalg::YieldOp>(innerOp))
        ops.push_back(&innerOp);
    if (ops.size() != 2) return false;
    auto *mul = ops[0], *add = ops[1];
    if (!isa<arith::MulFOp>(mul) && !isa<arith::MulIOp>(mul)) return false;
    if (!isa<arith::AddFOp>(add) && !isa<arith::AddIOp>(add)) return false;
    // add must consume mul's result
    Value mulResult = mul->getResult(0);
    for (Value operand : add->getOperands())
      if (operand == mulResult) return true;
    return false;
  }

  // Directly vectorize a matmul-shaped LinalgOp to vector.contract.
  // Handles both linalg.generic (with matmul body) and named linalg.matmul.
  // Returns true if conversion happened.
  // Maps must be: [d0,d2], [d2,d1], [d0,d1] with iterators [par,par,red].
  //
  // WHY: linalg::vectorize() produces vector.multi_reduction for BOTH
  // linalg.generic and named linalg.matmul in MLIR 21.  vector.multi_reduction
  // is then lowered by InnerParallel to broadcast+arith.mulf+sequential arith.addf
  // which has TWO roundings per multiply-add.  vector.contract lowered via
  // vector.outerproduct uses llvm.fma (ONE rounding), matching the eager
  // custom CUDA kernel and eliminating the precision mismatch.
  static bool tryVectorizeAsContraction(IRRewriter &rewriter,
                                        linalg::LinalgOp op) {
    // Accept 3-loop [par, par, red] for 2D matmul, or
    //        4-loop [par(batch), par, par, red] for 3D batch_matmul with batch=1.
    auto iterTypes = op.getIteratorTypesArray();
    bool isBatch = (iterTypes.size() == 4);
    if (iterTypes.size() != 3 && iterTypes.size() != 4) return false;

    if (isBatch) {
      if (iterTypes[0] != utils::IteratorType::parallel ||
          iterTypes[1] != utils::IteratorType::parallel ||
          iterTypes[2] != utils::IteratorType::parallel ||
          iterTypes[3] != utils::IteratorType::reduction) return false;
    } else {
      if (iterTypes[0] != utils::IteratorType::parallel ||
          iterTypes[1] != utils::IteratorType::parallel ||
          iterTypes[2] != utils::IteratorType::reduction) return false;
    }

    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) return false;
    // For linalg.generic, verify the body is a matmul (mulf+addf).
    // Named ops (linalg.matmul, linalg.batch_matmul) are matmuls by definition.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation()))
      if (!isMatmulBody(genericOp)) return false;

    auto ins  = op.getDpsInputs();
    auto outs = op.getDpsInits();
    auto aTy = dyn_cast<RankedTensorType>(ins[0].getType());
    auto bTy = dyn_cast<RankedTensorType>(ins[1].getType());
    auto cTy = dyn_cast<RankedTensorType>(outs[0].getType());
    if (!aTy || !bTy || !cTy) return false;
    if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !cTy.hasStaticShape()) return false;

    // In a 4-loop (batch) context the weight matrix B can be either:
    //   • Fully batched:  A=[1,M,K], B=[1,K,N], C=[1,M,N]  (all rank-3, batch=1)
    //   • Broadcast batch: A=[1,M,K], B=[K,N],   C=[1,M,N]  (B is rank-2, shared)
    //
    // The broadcast layout is by far the most common in transformer models
    // (the weight matrix is not replicated per batch element).
    // We handle both here so that linalg::vectorize() is not used as a fallback,
    // which would lose the mma_kind config and produce SIMT contracts.
    bool bIsBroadcast = isBatch && (bTy.getRank() == 2);

    if (!isBatch) {
      // Pure 2D path: all operands must be rank-2.
      if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
        return false;
    } else {
      // Batch path: A and C must be rank-3; B may be rank-2 (broadcast) or rank-3.
      if (aTy.getRank() != 3 || cTy.getRank() != 3) return false;
      if (!bIsBroadcast && bTy.getRank() != 3) return false;
      // The batch dimension (shape[0]) of A and C must be 1 so the whole tile
      // collapses into a single 2-D vector.contract that MMA can lower.
      if (aTy.getShape()[0] != 1 || cTy.getShape()[0] != 1) return false;
      // For fully-batched B (rank-3) the batch dim must also be 1.
      if (!bIsBroadcast && bTy.getShape()[0] != 1) return false;
    }

    MLIRContext *ctx = op.getContext();
    auto maps = op.getIndexingMapsArray();

    // Verify indexing maps.
    //   2D: A=[d0,d2], B=[d2,d1], C=[d0,d1]                    (3 iter dims)
    //   3D fully-batched: A=[d0,d1,d3], B=[d0,d3,d2], C=[d0,d1,d2]  (4 iter dims)
    //   3D broadcast:     A=[d0,d1,d3], B=[d3,d2],   C=[d0,d1,d2]  (4 iter dims, B no batch)
    AffineMap expectedA, expectedB, expectedC;
    if (!isBatch) {
      expectedA = AffineMap::get(3, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx);
      expectedB = AffineMap::get(3, 0,
          {getAffineDimExpr(2, ctx), getAffineDimExpr(1, ctx)}, ctx);
      expectedC = AffineMap::get(3, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
    } else if (bIsBroadcast) {
      expectedA = AffineMap::get(4, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(3, ctx)}, ctx);
      // B has no batch dim: its map skips d0.
      expectedB = AffineMap::get(4, 0,
          {getAffineDimExpr(3, ctx), getAffineDimExpr(2, ctx)}, ctx);
      expectedC = AffineMap::get(4, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx);
    } else {
      // Fully-batched B
      expectedA = AffineMap::get(4, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(3, ctx)}, ctx);
      expectedB = AffineMap::get(4, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(3, ctx), getAffineDimExpr(2, ctx)}, ctx);
      expectedC = AffineMap::get(4, 0,
          {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx);
    }
    if (maps[0] != expectedA || maps[1] != expectedB || maps[2] != expectedC) return false;

    // Extract logical M, N, K from the appropriate shape dimensions.
    int64_t M, N, K;
    if (!isBatch) {
      M = cTy.getShape()[0]; N = cTy.getShape()[1]; K = aTy.getShape()[1];
    } else if (bIsBroadcast) {
      M = cTy.getShape()[1]; N = cTy.getShape()[2]; K = aTy.getShape()[2];
    } else {
      M = cTy.getShape()[1]; N = cTy.getShape()[2]; K = aTy.getShape()[2];
    }

    Type elemTy = cTy.getElementType();
    auto lhsVT = VectorType::get({M, K}, elemTy);
    auto rhsVT = VectorType::get({K, N}, elemTy);
    auto accVT = VectorType::get({M, N}, elemTy);

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemTy));
    Value zeroI = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // in_bounds has one entry per *vector* dimension (always 2D here).
    SmallVector<bool> inBoundsVec = {true, true};

    Value lhs, rhs, acc;
    if (!isBatch) {
      SmallVector<Value> zeroIdx = {zeroI, zeroI};
      // Implicit identity permutation map for same-rank 2D transfers.
      lhs = rewriter.create<vector::TransferReadOp>(loc, lhsVT, ins[0],  zeroIdx, zero, inBoundsVec);
      rhs = rewriter.create<vector::TransferReadOp>(loc, rhsVT, ins[1],  zeroIdx, zero, inBoundsVec);
      acc = rewriter.create<vector::TransferReadOp>(loc, accVT, outs[0], zeroIdx, zero, inBoundsVec);
    } else {
      // A and C are rank-3 [1, M, K] / [1, M, N].  Use a projection map that
      // drops the batch d0 (always 0), mapping src dims (d0,d1,d2) → (d1,d2).
      SmallVector<Value> zeroIdx3 = {zeroI, zeroI, zeroI};
      auto projAC = AffineMap::get(3, 0,
          {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx);
      // Builder: (loc, vectorType, source, indices, padding, permutationMap, inBounds)
      lhs = rewriter.create<vector::TransferReadOp>(loc, lhsVT, ins[0],  zeroIdx3, zero, projAC, inBoundsVec);
      acc = rewriter.create<vector::TransferReadOp>(loc, accVT, outs[0], zeroIdx3, zero, projAC, inBoundsVec);

      if (bIsBroadcast) {
        // B is rank-2 [K, N] with no batch dim — use plain 2D transfer_read.
        SmallVector<Value> zeroIdx2 = {zeroI, zeroI};
        rhs = rewriter.create<vector::TransferReadOp>(loc, rhsVT, ins[1], zeroIdx2, zero, inBoundsVec);
      } else {
        // B is rank-3 [1, K, N] — same projection as A/C.
        rhs = rewriter.create<vector::TransferReadOp>(loc, rhsVT, ins[1], zeroIdx3, zero, projAC, inBoundsVec);
      }
    }

    // The vector.contract always uses 2-D maps regardless of source rank.
    auto contractA = AffineMap::get(3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx);
    auto contractB = AffineMap::get(3, 0, {getAffineDimExpr(2, ctx), getAffineDimExpr(1, ctx)}, ctx);
    auto contractC = AffineMap::get(3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
    SmallVector<AffineMap> cMaps = {contractA, contractB, contractC};
    SmallVector<Attribute> iterAttrs = {
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)};
    auto contractOp = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, acc,
        rewriter.getAffineMapArrayAttr(cMaps),
        rewriter.getArrayAttr(iterAttrs));

    // Propagate lowering_config (including mma_kind) from the source linalg op
    // to the new vector.contract.  UnrollToIntrinsicsPass and
    // VectorDistributePass both walk upward from vector.contract to find
    // mma_kind; without this copy they see no config, fall back to the SIMT
    // (outer-product) path, and the TF32 WMMA path is never taken.
    if (auto cfg = getLoweringConfig(op.getOperation()))
      setLoweringConfig(contractOp, cfg);

    // Write the contraction result back, using the same projection for 3D.
    Value writeResult;
    if (isBatch) {
      SmallVector<Value> zeroIdx3 = {zeroI, zeroI, zeroI};
      auto projW = AffineMap::get(3, 0,
          {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx);
      auto writeOp = rewriter.create<vector::TransferWriteOp>(
          loc, contractOp.getResult(), outs[0], zeroIdx3, projW, inBoundsVec);
      if (auto cfg = getLoweringConfig(op.getOperation()))
        setLoweringConfig(writeOp, cfg);
      writeResult = writeOp->getResult(0);
    } else {
      SmallVector<Value> zeroIdx = {zeroI, zeroI};
      auto writeOp = rewriter.create<vector::TransferWriteOp>(
          loc, contractOp.getResult(), outs[0], zeroIdx, inBoundsVec);
      if (auto cfg = getLoweringConfig(op.getOperation()))
        setLoweringConfig(writeOp, cfg);
      writeResult = writeOp->getResult(0);
    }
    rewriter.replaceOp(op, writeResult);
    return true;
  }

  // Fix the dBeta undercount caused by LinalgElementwiseOpFusionPass.
  //
  // Background:
  //   NovaLinalgHorizontalFusionPass creates a dual-output reduction generic
  //   for dgamma+dbeta with gy (dense<1.0>) as an explicit tensor input:
  //       ins(..., gy : tensor<4x4x2xf32>)   map (d0,d1,d2)->(d0,d1,d2)
  //       body: dbeta_acc += %in_gy           ← correct: 16 values per tile
  //
  //   LinalgElementwiseOpFusionPass then folds the constant tensor producer
  //   into the body, removing gy from `ins` and capturing a scalar 1.0f:
  //       ins(...)                            ← gy input GONE
  //       body: dbeta_acc += %cst_1.0f        ← wrong: adds 1 once per tile
  //
  //   linalg::vectorize() then generates:
  //       %v = arith.addf %acc<2>, dense<1.0> : vector<2xf32>
  //   This adds 1.0 per tile call, giving 512×1=512 instead of 512×16=8192.
  //
  // Fix:
  //   Before vectorizing, identify captured float constants that are directly
  //   accumulated into a reduction output block arg via addf.  Scale each such
  //   constant by the product of the reduction tile sizes (4×4=16 for this
  //   case), so that linalg::vectorize() generates:
  //       %v = arith.addf %acc<2>, dense<16.0> : vector<2xf32>
  //   After 512 tile iterations: acc += 16.0 * 512 = 8192.  ✓
  //
  //   This avoids modifying the linalg.generic structure (no new inputs),
  //   which would crash the vectorizer.
  static bool tryScaleDirectlyAccumulatedConstants(IRRewriter &rewriter,
                                                   linalg::GenericOp op) {
    // Only apply to generics with at least one reduction iterator.
    auto iterTypes = op.getIteratorTypesArray();
    if (llvm::none_of(iterTypes, [](utils::IteratorType t) {
          return t == utils::IteratorType::reduction;
        }))
      return false;

    Block &body = op.getRegion().front();
    unsigned numIns = op.getNumDpsInputs();

    // Collect captured float constants that are DIRECTLY accumulated into a
    // reduction output block arg via addf.  Skip constants used only in mulf
    // or other non-accumulation ops (e.g. scale/epsilon in variance).
    SmallVector<Value> capturedConsts;
    for (Operation &innerOp : body) {
      if (!isa<arith::AddFOp>(innerOp))
        continue;
      Value capturedCandidate;
      bool hasReductionOutputArg = false;
      for (Value v : innerOp.getOperands()) {
        if (auto ba = dyn_cast<BlockArgument>(v)) {
          if (ba.getOwner() == &body && ba.getArgNumber() >= numIns) {
            unsigned outIdx = ba.getArgNumber() - numIns;
            AffineMap outMap = op.getIndexingMapsArray()[numIns + outIdx];
            // Output map has fewer results than loops → this dim is reduced.
            if (outMap.getNumResults() < op.getNumLoops())
              hasReductionOutputArg = true;
          }
        } else {
          auto *defOp = v.getDefiningOp();
          if (defOp && isa<arith::ConstantOp>(defOp) &&
              isa<FloatType>(v.getType()))
            capturedCandidate = v;
        }
      }
      if (hasReductionOutputArg && capturedCandidate &&
          !llvm::is_contained(capturedConsts, capturedCandidate))
        capturedConsts.push_back(capturedCandidate);
    }
    if (capturedConsts.empty())
      return false;

    // Infer the iteration-space tile shape from operand shapes + affine maps.
    unsigned numLoops = op.getNumLoops();
    SmallVector<int64_t> iterShape(numLoops, ShapedType::kDynamic);
    {
      auto allMaps = op.getIndexingMapsArray();
      SmallVector<Value> allOperands;
      llvm::append_range(allOperands, op.getDpsInputs());
      llvm::append_range(allOperands, op.getDpsInits());
      for (auto [operand, map] : llvm::zip(allOperands, allMaps)) {
        auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorTy) continue;
        for (auto [resultIdx, result] : llvm::enumerate(map.getResults())) {
          auto dimExpr = dyn_cast<AffineDimExpr>(result);
          if (!dimExpr) continue;
          unsigned dim = dimExpr.getPosition();
          if (dim < numLoops && iterShape[dim] == ShapedType::kDynamic)
            iterShape[dim] = tensorTy.getShape()[resultIdx];
        }
      }
    }
    if (llvm::any_of(iterShape,
                     [](int64_t d) { return d == ShapedType::kDynamic; }))
      return false;

    // Total reduction elements per tile = product of all reduction dim sizes.
    // For [red=4, red=4, par=2]: 4×4 = 16.  Each tile should contribute
    // (original_val × 16) to the accumulator, not original_val × 1.
    int64_t totalRedElems = 1;
    for (unsigned i = 0; i < numLoops; ++i)
      if (iterTypes[i] == utils::IteratorType::reduction)
        totalRedElems *= iterShape[i];

    if (totalRedElems == 1)
      return false; // Nothing to scale.

    // For each captured constant, create a new scalar = original_val * totalRedElems
    // and replace all its uses inside this linalg body.
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    for (Value captured : capturedConsts) {
      auto floatTy = cast<FloatType>(captured.getType());
      auto constOp = cast<arith::ConstantOp>(captured.getDefiningOp());
      auto floatAttr = cast<FloatAttr>(constOp.getValue());
      double scaledVal =
          floatAttr.getValueAsDouble() * static_cast<double>(totalRedElems);
      Value scaledConst = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(floatTy, scaledVal));

      // Replace uses of the captured constant inside the linalg body only.
      for (Operation &innerOp : body)
        innerOp.replaceUsesOfWith(captured, scaledConst);
    }
    return true;
  }

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // Pre-scan: save the lowering_config (containing mma_kind) from any linalg
    // op that targets an MMA intrinsic.  linalg::vectorize() does NOT copy
    // op attributes to the vector.contract it creates, so the config is lost
    // once the linalg op is replaced.  We re-attach it after all vectorization.
    DictionaryAttr savedMmaConfig;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (auto cfg = getLoweringConfig(op.getOperation())) {
        if (getMmaKindRaw(cfg) != 0) {
          savedMmaConfig = cfg;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // 1a. For matmul-shaped LinalgOps (both linalg.generic with matmul body
    //    AND named linalg.matmul), directly emit vector.contract.
    //
    //    WHY: linalg::vectorize() in MLIR 21 produces vector.multi_reduction
    //    for ALL linalg matmul ops (generic or named).  The InnerParallel
    //    multi_reduction lowering (step 2) then converts that to:
    //        broadcast(A) × broadcast(B) → sequential arith.addf per K
    //    which has TWO roundings per multiply-add vs ONE for llvm.fma.
    //    This causes systematic divergence from the eager (non-JIT) path.
    //
    //    By emitting vector.contract here instead, the VectorDistribute pass
    //    will lower it to vector.outerproduct (with accumulator), which
    //    VectorToLLVM maps to llvm.fma — matching the eager FP32 behavior.
    SmallVector<linalg::LinalgOp> contractionOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (linalg::isaContractionOpInterface(op))
        contractionOps.push_back(op);
    });
    for (linalg::LinalgOp op : contractionOps)
      tryVectorizeAsContraction(rewriter, op);

    // 1b-pre. Re-introduce captured scalar constants as explicit tensor inputs
    //    in reduction linalg.generics so linalg::vectorize() emits proper
    //    vector.multi_reduction instead of a plain scalar-broadcast addf.
    //    (Undoes what LinalgElementwiseOpFusionPass does to dbeta.)
    {
      SmallVector<linalg::GenericOp> reductionGenerics;
      funcOp.walk([&](linalg::GenericOp op) {
        reductionGenerics.push_back(op);
      });
      for (auto op : reductionGenerics)
        tryScaleDirectlyAccumulatedConstants(rewriter, op);
    }


    // 1b. Vectorize remaining Linalg operations and replace them with the results.
    //    Skip fill-like ops to avoid vector<NxMxf32> when the fill tile is large.
    //    NVPTX can't lower vectors wider than 128 bits → llvm.mlir.poison →
    //    all __shared__ memory demoted → GPU kernel hangs.
    //
    //    Detection: a fill-like generic body yields only values captured from
    //    the outer scope (no block argument is used by any op in the body).
    //    Contractions and elementwise ops always use at least one block arg.
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      // Explicit linalg.fill: always skip vectorization.
      if (isa<linalg::FillOp>(linalgOp.getOperation()))
        return;

      // Generalized fill (linalg.generic with no ins or yield-from-constant):
      // skip if no block argument appears as an operand in the body.
      if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
        bool usesBlockArg = false;
        for (Operation &innerOp : genericOp.getRegion().front()) {
          for (Value operand : innerOp.getOperands()) {
            if (isa<BlockArgument>(operand)) {
              usesBlockArg = true;
              break;
            }
          }
          if (usesBlockArg)
            break;
        }
        if (!usesBlockArg)
          return; // fill-like: body is a constant write, not a computation
      }

      rewriter.setInsertionPoint(linalgOp);
      auto result = linalg::vectorize(rewriter, linalgOp);
      if (succeeded(result)) {
        rewriter.replaceOp(linalgOp, result->replacements);
      }
    });


    // 2. Lower multi-dimensional reductions via InnerReduction then fold into
    //    vector.contract so they follow the same outer-product → llvm.fma
    //    path as matmul contractions in VectorDistributePass.
    //
    //    WHY InnerReduction over InnerParallel:
    //    InnerParallel expands vector.multi_reduction directly into a chain of
    //    vector.extract + arith.addf ops (sequential scalar adds).  InnerReduction
    //    instead emits vector.reduction ops (1-D hardware reductions) for the
    //    inner dim.  populateVectorReductionToContractPatterns then converts
    //    each vector.reduction to a vector.contract (dot-product with a unit
    //    vector), which VectorDistributePass lowers via outer-product →
    //    llvm.fma (single rounding, matching the eager path for reductions).
    //
    //    NOTE: Do NOT lower vector.contract here. Surviving contracts must reach
    //    PrepareVectorToGPU (post-GPU-outlining) so they can be converted to
    //    gpu.subgroup_mma_compute → TF32 mma.sync. Lowering them to outer-product
    //    here would destroy the TF32 tensor-core path and produce plain FP32 results.
    {
      RewritePatternSet patterns(context);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorReductionToContractPatterns(patterns);
      vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
      vector::populateVectorShapeCastLoweringPatterns(patterns);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Post-vectorization: propagate the saved mma_kind config to any
    // vector.contract that didn't receive it during vectorization.
    // linalg::vectorize() replaces the linalg.generic (which had the config)
    // without copying attributes to the new vector.contract.  Without this,
    // UnrollToIntrinsicsPass and VectorDistributePass both see mma_kind == 0
    // and fall back to the SIMT outer-product path.
    if (savedMmaConfig) {
      funcOp.walk([&](vector::ContractionOp contractOp) {
        if (!getLoweringConfig(contractOp))
          setLoweringConfig(contractOp, savedMmaConfig);
      });
    }

  }
  StringRef getArgument() const override { return "nova-gpu-generic-vectorization"; }
  StringRef getDescription() const override {
    return "Vectorize linalg operations in the Nova GPU pipeline.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUSubsetHoistingPass
//===----------------------------------------------------------------------===//

struct NovaGPUSubsetHoistingPass
    : public PassWrapper<NovaGPUSubsetHoistingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUSubsetHoistingPass)
  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    // Hoist redundant transfer_read/transfer_write out of loops.
    linalg::hoistRedundantVectorTransfers(funcOp);
  }
  StringRef getArgument() const override { return "nova-gpu-subset-hoisting"; }
  StringRef getDescription() const override {
    return "Hoist redundant vector.transfer_read/write out of scf.for loops.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUVectorizeMemrefCopyPass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorizeMemrefCopyPass
    : public PassWrapper<NovaGPUVectorizeMemrefCopyPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorizeMemrefCopyPass)
  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    IRRewriter rewriter(&getContext());
    MLIRContext *context = &getContext();
    funcOp.walk([&](memref::CopyOp copyOp) {
      // Skip copies where source or destination is a function argument.
      // These are typically D2H output copies (e.g. memref.copy device_buf,
      // func_arg). They must stay as memref.copy until createConvertMemRefToGpuPass
      // (Step 12) converts them to gpu.memcpy → cudaMemcpyAsync. If vectorized
      // here, the host-side vector.transfer_read later reads from a device
      // pointer (after ConvertMemRefToGpu upgrades the alloc) = illegal access.
      // Also skip copies to view-like ops (expand_shape etc.) of func args.
      auto isFuncArg = [](Value v) -> bool {
        if (isa<BlockArgument>(v))
          return true;
        // Follow view-like ops (expand_shape, subview, cast) back to the root.
        Operation *defOp = v.getDefiningOp();
        while (defOp && isa<memref::ExpandShapeOp, memref::SubViewOp,
                            memref::CastOp, memref::ReinterpretCastOp,
                            memref::CollapseShapeOp>(defOp)) {
          v = defOp->getOperand(0);
          if (isa<BlockArgument>(v))
            return true;
          defOp = v.getDefiningOp();
        }
        return false;
      };
      if (isFuncArg(copyOp.getSource()) || isFuncArg(copyOp.getTarget()))
        return;
      (void)linalg::vectorizeCopy(rewriter, copyOp);
    });

    // Lower reductions via InnerReduction + ReductionToContract (same as
    // GenericVectorizationPass step 2) so any remaining multi_reductions
    // from copy vectorization also follow the outer-product → llvm.fma path.
    // NOTE: Do NOT lower vector.contract here — must survive to gpuPm.
    {
      RewritePatternSet patterns(context);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorReductionToContractPatterns(patterns);
      vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
      vector::populateVectorShapeCastLoweringPatterns(patterns);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
  StringRef getArgument() const override { return "nova-gpu-vectorize-memref-copy"; }
  StringRef getDescription() const override {
    return "Vectorize memref.copy ops, especially between global and shared memory.";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUVectorDistributePass
//
// Distributes warp-level vector ops across the 32 threads of a CUDA warp using
// the vector.warp_execute_on_lane_0 wrapper pattern, then lowers it to an
// scf.if guarded by gpu.lane_id == 0 with explicit __shfl_sync distribution.
//
// Pipeline position: AFTER GenericVectorization (SIMD vectors exist),
//                    BEFORE bufferization (still tensor form so warp ops can
//                    be cleanly outlined per thread).
//
// Strategy:
//   1. Wrap each vector.transfer_read/write pair that operates on a warp tile
//      inside vector.warp_execute_on_lane_0 with the correct warp size (32).
//   2. Apply populateWarpExecuteOnLane0OpToScfForPattern to lower the wrapper
//      to a lane-0 branch with warp shuffles for accumulator distribution.
//   3. Canonicalize to fold identity broadcasts.
//===----------------------------------------------------------------------===//

struct NovaGPUVectorDistributePass
    : public PassWrapper<NovaGPUVectorDistributePass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorDistributePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    MLIRContext *ctx = &getContext();

    // -----------------------------------------------------------------------
    // Step 1: Lower multi-dimensional reductions via InnerReduction then fold
    // into vector.contract.  InnerReduction produces vector.reduction (1-D
    // hardware reductions) instead of InnerParallel's scalar-extract chains.
    // populateVectorReductionToContractPatterns then converts each
    // vector.reduction to a vector.contract (dot-product with unit vector),
    // which step 2 below lowers via outer-product → llvm.fma.
    // -----------------------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorReductionToContractPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    // -----------------------------------------------------------------------
    // Step 1b: Fold extract(insert_strided_slice chain) + reduction chains.
    //
    // When a matmul tile is vectorized via linalg::vectorize() (rather than
    // tryVectorizeAsContraction — e.g. because the op has a batch dim of 1),
    // the K-loop is unrolled into:
    //   %p0 = mulf(A_k0, B_k0) : vector<16x8x8xf32>
    //   %v0 = insert_strided_slice %p0, %zero  {offsets=[0,0,0]}  → vector<16x8x64xf32>
    //   %p1 = mulf(A_k8, B_k8) : vector<16x8x8xf32>
    //   %v1 = insert_strided_slice %p1, %v0   {offsets=[0,0,8]}   → vector<16x8x64xf32>
    //   ... (8 tiles total) ...
    //   %slice = vector.extract %final[i, j]   → vector<64xf32>
    //   %sum   = vector.reduction<add> %slice  → f32
    //
    // The 3D accumulator vector<16x8x64xf32> = 8,192 f32 = 32 KB stays alive
    // through GPU outlining and becomes a by-value kernel parameter, triggering
    // CUDA_ERROR_INVALID_PTX (parameter space overflow + register explosion).
    //
    // This pattern rewrites to:
    //   %r0  = vector.extract %p0[i, j]          → vector<8xf32>
    //   %s0  = vector.reduction<add> %r0, %init  → f32
    //   %r1  = vector.extract %p1[i, j]          → vector<8xf32>
    //   %s1  = vector.reduction<add> %r1, %s0    → f32
    //   ... → %sum
    //
    // This eliminates the 3D accumulator entirely; each per-tile partial
    // product (vector<16x8x8xf32> = 128 f32 = 512 bytes) is reduced and
    // discarded immediately, keeping the working set at vector<16x8xf32>.
    // -----------------------------------------------------------------------
    {
      struct FoldExtractInsertChainReduction
          : OpRewritePattern<vector::ReductionOp> {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(vector::ReductionOp redOp,
                                      PatternRewriter &rewriter) const override {
          if (redOp.getKind() != vector::CombiningKind::ADD)
            return failure();

          // The vector being reduced must come from vector.extract [i, j, ...].
          auto extractOp =
              redOp.getVector().getDefiningOp<vector::ExtractOp>();
          if (!extractOp)
            return failure();

          // All extract positions must be static (constant indices).
          // getMixedPosition() returns SmallVector<OpFoldResult>; each entry
          // is either an Attribute (static) or a Value (dynamic).
          auto mixedPos = extractOp.getMixedPosition();
          SmallVector<int64_t> iPos;
          for (auto p : mixedPos) {
            if (!isa<Attribute>(p))
              return failure(); // dynamic index — can't fold statically
            iPos.push_back(cast<IntegerAttr>(cast<Attribute>(p)).getInt());
          }

          // Walk the insert_strided_slice chain, collecting all tiles with
          // their full offset vectors.  Tiles may vary in ALL dimensions —
          // e.g. vector<32x16x64xf32> assembled from vector<16x8x8xf32> tiles
          // at offsets [d0_off, d1_off, d2_off].
          //
          // For a given extract position iPos[0..N-2] (covering all dims
          // except the last inner dim), only tiles where
          //   offset[d] ≤ iPos[d] < offset[d] + tileDim[d]   (for d < rank-1)
          // contribute.  The contributing local index inside that tile is
          //   localPos[d] = iPos[d] - offset[d].
          struct TileEntry {
            SmallVector<int64_t> offsets;
            Value tile;
          };
          SmallVector<TileEntry> allTiles;
          Value cur = extractOp.getVector();
          VectorType tileTy;
          while (auto insertOp =
                     cur.getDefiningOp<vector::InsertStridedSliceOp>()) {
            // getOffsets() / getStrides() return ArrayAttr of IntegerAttr.
            SmallVector<int64_t> offsets, strides;
            for (auto a : insertOp.getOffsets().getAsRange<IntegerAttr>())
              offsets.push_back(a.getInt());
            for (auto a : insertOp.getStrides().getAsRange<IntegerAttr>())
              strides.push_back(a.getInt());
            // Require unit strides.
            for (auto s : strides)
              if (s != 1)
                return failure();
            Value tile = insertOp.getValueToStore();
            auto tTy = dyn_cast<VectorType>(tile.getType());
            if (!tTy)
              return failure();
            if (tileTy && tileTy != tTy)
              return failure(); // all tiles must be the same shape
            tileTy = tTy;
            allTiles.push_back({offsets, tile});
            cur = insertOp.getDest();
          }
          if (allTiles.size() < 2)
            return failure();

          // iPos covers the first (rank-1) dimensions of the tile.
          if ((int64_t)iPos.size() != tileTy.getRank() - 1)
            return failure();

          // Filter to tiles that contain the extract position iPos in their
          // first (rank-1) dimensions, and compute the local sub-index.
          struct ContribEntry {
            SmallVector<int64_t> localPos; // per-tile extract index (rank-1)
          };
          SmallVector<std::pair<ContribEntry, Value>> contribs;
          for (auto &te : allTiles) {
            bool inBounds = true;
            SmallVector<int64_t> localPos;
            for (size_t d = 0; d < iPos.size(); ++d) {
              int64_t lo = te.offsets[d];
              int64_t hi = lo + tileTy.getDimSize(d);
              if (iPos[d] < lo || iPos[d] >= hi) {
                inBounds = false;
                break;
              }
              localPos.push_back(iPos[d] - lo);
            }
            if (inBounds)
              contribs.push_back({{localPos}, te.tile});
          }
          if (contribs.empty())
            return failure();

          // No sort needed — vector.reduction<add> is commutative so
          // accumulation order does not affect the result.

          // Build: acc = init; for each contributing tile:
          //   subVec = extract(tile, localPos)  → 1-D vector (inner K-sub dim)
          //   acc    = reduce<add>(subVec, acc)
          Location loc = redOp.getLoc();
          Value acc = redOp.getAcc();
          if (!acc)
            acc = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getZeroAttr(redOp.getType()));

          for (auto &[ce, tile] : contribs) {
            Value subVec =
                rewriter.create<vector::ExtractOp>(loc, tile, ce.localPos);
            acc = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, subVec, acc);
          }
          rewriter.replaceOp(redOp, acc);
          return success();
        }
      };

      RewritePatternSet patterns(ctx);
      patterns.add<FoldExtractInsertChainReduction>(ctx);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    // -----------------------------------------------------------------------
    // Step 2 (MMA check): Detect whether this function targets tensor-core
    // (mma.sync) intrinsics via a non-NONE mma_kind in lowering_config.
    //
    //   MMA path  (hasMMAContract == true):
    //     Preserve vector.contract ops here.  They flow to PrepareVectorToGPU
    //     + ConvertVectorToGPU in the GPU pass pipeline (post-GPU-outlining),
    //     which wraps them per-thread in WarpExecuteOnLane0 and lowers to
    //     gpu.subgroup_mma_compute → mma.sync PTX.
    //
    //   SIMT path (hasMMAContract == false):
    //     Lower vector.contract → vector.outerproduct → llvm.fma.  This path
    //     is numerically correct (FP32 FMA, single rounding) but MUST NOT be
    //     used for large matmuls (e.g. 5120×5120): the outer-product chains
    //     for K=5120 create enormous register pressure that causes GPU kernel
    //     stalling regardless of whether PTX compilation succeeds.
    // -----------------------------------------------------------------------
    bool hasMMAContract = false;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      Operation *cur = contractOp.getOperation();
      while (cur) {
        if (auto cfg = getLoweringConfig(cur)) {
          if (getMmaKindRaw(cfg) != 0) {
            hasMMAContract = true;
            return WalkResult::interrupt();
          }
        }
        cur = cur->getParentOp();
      }
      return WalkResult::advance();
    });

    if (!hasMMAContract) {
      // SIMT (CUDA-core) path: lower vector.contract → outer-product → llvm.fma.
      // Safe for small tile sizes where the outerproduct chain fits in registers.
      RewritePatternSet patterns(ctx);
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     /*maxTransferRank=*/1);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
    // MMA path: vector.contract ops are preserved for ConvertVectorToGPU
    // (mma.sync / tensor-core lowering in the GPU pass pipeline).

    // -----------------------------------------------------------------------
    // Step 3: Flatten shape-cast chains introduced by outer-product lowering.
    // Skip on the MMA path — contracts must keep their 2-D tile shapes so
    // ConvertVectorToGPU pattern matching succeeds.
    // -----------------------------------------------------------------------
    if (!hasMMAContract) {
      RewritePatternSet patterns(ctx);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    // -----------------------------------------------------------------------
    // Step 4: Wrap warp-level vector ops inside gpu.warp_execute_on_lane_0.
    //
    // Each gpu.launch body runs with warpSize=32 threads per warp.  We wrap
    // the entire set of vector ops that feed a warp-level reduction inside
    // a WarpExecuteOnLane0Op.  The distribution passes then propagate
    // individual vector elements to each thread lane, replacing the wide
    // warp-level vector with a per-lane vector<1xT>.
    //
    // Allocation function: uses a small shared memory allocation to hold
    // warp-level temporaries during distribution.  The alloc is issued inside
    // the gpu.launch body (as a gpu.alloc of private address space) so it
    // lives in registers via the NVPTX backend.
    // -----------------------------------------------------------------------
    static constexpr int64_t kWarpSize = 32;

    // Walk all gpu.launch bodies and wrap vector ops.
    funcOp.walk([&](gpu::LaunchOp launchOp) {
      Block &body = launchOp.getBody().front();
      OpBuilder builder(&body, body.begin());
      Location loc = launchOp.getLoc();

      // Get lane ID = threadIdx.x (assumes linear thread mapping).
      Value laneId;
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        laneId = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      }

      // Collect all vector.reduction ops in the launch body — these drive
      // the warp-to-SIMT conversion.  Each reduction becomes a warp shuffle.
      SmallVector<vector::ReductionOp> reductions;
      launchOp.walk([&](vector::ReductionOp rop) {
        reductions.push_back(rop);
      });

      // For each reduction, wrap its data flow in a warp op if the source
      // vector is warp-sized (numElements == kWarpSize).  Smaller vectors
      // are already per-lane and don't need distribution.
      for (vector::ReductionOp rop : reductions) {
        Value src = rop.getVector();
        auto srcTy = dyn_cast<VectorType>(src.getType());
        if (!srcTy || srcTy.getNumElements() != kWarpSize)
          continue; // not a warp-sized vector, skip

        // Emit gpu.warp_execute_on_lane_0 wrapping the reduction source.
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(rop);
        Type elemTy = srcTy.getElementType();
        VectorType perLaneTy = VectorType::get({1}, elemTy);

        // Warp op: yields the entire warp vector, distributes to per-lane ×1.
        auto warpOp = builder.create<gpu::WarpExecuteOnLane0Op>(
            loc, TypeRange{perLaneTy}, laneId, kWarpSize,
            /*args=*/ValueRange{src}, /*argTypes=*/TypeRange{srcTy});

        // Populate warp body: yield the argument back (identity — propagation
        // patterns below will distribute it to individual lanes).
        Block *warpBody = warpOp.getBody(0);
        {
          OpBuilder::InsertionGuard wg(builder);
          builder.setInsertionPointToEnd(warpBody);
          builder.create<gpu::YieldOp>(loc,
                                       ValueRange{warpBody->getArgument(0)});
        }
        // Replace the wide reduction source with the per-lane warp result.
        src.replaceUsesWithIf(warpOp.getResult(0), [&](OpOperand &use) {
          return use.getOwner() == rop.getOperation();
        });
      }

      // MMA path: also wrap vector.contract ops in WarpExecuteOnLane0 so
      // ConvertVectorToGPU can distribute them to per-thread MMA fragments.
      // Each contract (e.g. vector<16x8xf32>) is wrapped with a warp-level
      // argument; the distribution pass below produces per-lane fragment
      // vectors that ConvertVectorToGPU converts to gpu.subgroup_mma_compute.
      if (hasMMAContract) {
        SmallVector<vector::ContractionOp> contracts;
        launchOp.walk([&](vector::ContractionOp cop) {
          contracts.push_back(cop);
        });
        for (vector::ContractionOp cop : contracts) {
          auto accTy = dyn_cast<VectorType>(cop.getAcc().getType());
          if (!accTy) continue;

          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(cop);

          // Per-lane accumulator: each lane owns 1 element of the output tile.
          int64_t laneElems = std::max<int64_t>(1,
              accTy.getNumElements() / kWarpSize);
          VectorType perLaneAccTy = VectorType::get({laneElems},
                                                    accTy.getElementType());

          // Wrap the full contraction (lhs, rhs, acc) in the warp op.
          // The warp body yields the contraction result; distribution
          // propagation patterns split it per lane.
          SmallVector<Value>    warpArgs = {cop.getLhs(), cop.getRhs(),
                                            cop.getAcc()};
          SmallVector<Type>     warpArgTys = {cop.getLhs().getType(),
                                              cop.getRhs().getType(),
                                              cop.getAcc().getType()};
          auto warpOp = builder.create<gpu::WarpExecuteOnLane0Op>(
              loc, TypeRange{perLaneAccTy}, laneId, kWarpSize,
              warpArgs, warpArgTys);

          Block *warpBody = warpOp.getBody(0);
          {
            OpBuilder::InsertionGuard wg(builder);
            builder.setInsertionPointToEnd(warpBody);
            // Re-emit the contraction inside the warp body using the
            // warp-body block arguments (which carry warp-level types).
            Value innerResult = builder.create<vector::ContractionOp>(
                loc, warpBody->getArgument(0),
                warpBody->getArgument(1),
                warpBody->getArgument(2),
                cop.getIndexingMaps(), cop.getIteratorTypes());
            builder.create<gpu::YieldOp>(loc, ValueRange{innerResult});
          }
          cop.getResult().replaceAllUsesWith(warpOp.getResult(0));
          cop.erase();
        }
      }
    });

    // -----------------------------------------------------------------------
    // Step 5: Apply warp distribution propagation patterns.
    //
    // These patterns propagate warp ops through the IR, eventually reducing
    // every vector.transfer_read/write to per-lane accesses and emitting
    // warp shuffles for reduction communication.
    // -----------------------------------------------------------------------
    funcOp.walk([&](gpu::WarpExecuteOnLane0Op warpOp) {
      // Move any scalar ops (loop-invariant constants, index arithmetic)
      // outside the warp region — they run identically on every lane.
      vector::moveScalarUniformCode(warpOp);
    });

    {
      RewritePatternSet patterns(ctx);

      // Distribution map: simple identity (element i → lane i for 1-D vectors).
      auto distributionMapFn = [](Value val) -> AffineMap {
        auto vt = cast<VectorType>(val.getType());
        // Map the outermost dimension to the warp lane.
        return AffineMap::getMultiDimIdentityMap(
            vt.getRank(), val.getContext());
      };

      // Warp shuffle: emit a gpu.shuffle_xor to broadcast the value from
      // the source lane.  PTX lowers this to shfl.sync.bfly.
      auto warpShuffleFn = [](Location loc, OpBuilder &b, Value val,
                              Value srcLane, int64_t warpSize) -> Value {
        Type i32 = b.getI32Type();
        // Mask: all 32 threads participate.
        Value mask = b.create<arith::ConstantIntOp>(loc, 0xffffffff, 32u);
        // Offset = srcLane (XOR shuffle)
        Value offset = b.create<arith::IndexCastOp>(loc, i32, srcLane);
        auto shuffleResult = b.create<gpu::ShuffleOp>(
            loc, val, offset, mask, gpu::ShuffleMode::IDX);
        return shuffleResult.getShuffleResult();
      };

      // Distributed reduction: use warp XOR butterfly for maximum bandwidth.
      auto distributedReductionFn = [](Location loc, OpBuilder &b, Value acc,
                                       vector::CombiningKind kind,
                                       uint32_t size) -> Value {
        // Emit a butterfly-reduction using gpu.shuffle_xor for each power-of-2
        // step. This matches what cuBLAS / CUTLASS emit for warp reductions.
        Value result = acc;
        Type i32 = b.getI32Type();
        Value fullMask =
            b.create<arith::ConstantIntOp>(loc, 0xffffffff, 32u);
        for (uint32_t offset = size / 2; offset > 0; offset >>= 1) {
          Value offsetVal =
              b.create<arith::ConstantIntOp>(loc, (int64_t)offset, 32u);
          auto shuffled = b.create<gpu::ShuffleOp>(
              loc, result, offsetVal, fullMask, gpu::ShuffleMode::XOR);
          Value other = shuffled.getShuffleResult();
          // Combine result and other according to the reduction kind.
          switch (kind) {
          case vector::CombiningKind::ADD:
            result = isa<FloatType>(acc.getType())
                         ? b.create<arith::AddFOp>(loc, result, other)
                               .getResult()
                         : b.create<arith::AddIOp>(loc, result, other)
                               .getResult();
            break;
          case vector::CombiningKind::MAXIMUMF:
            result = b.create<arith::MaximumFOp>(loc, result, other);
            break;
          case vector::CombiningKind::MAXNUMF:
            result = b.create<arith::MaxNumFOp>(loc, result, other);
            break;
          default:
            // For mul/min and other kinds fall back to an XOR add.
            result = isa<FloatType>(acc.getType())
                         ? b.create<arith::AddFOp>(loc, result, other)
                               .getResult()
                         : b.create<arith::AddIOp>(loc, result, other)
                               .getResult();
            break;
          }
        }
        return result;
      };

      // Propagate warp distribution across all vector ops.
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionMapFn, warpShuffleFn,
          /*benefit=*/1, /*readBenefit=*/0);

      // Distribute transfer_write ops with the highest priority so that
      // writes are resolved before reads (prevents spurious aliasing).
      vector::populateDistributeTransferWriteOpPatterns(
          patterns, distributionMapFn,
          /*maxNumElementsToExtract=*/1, /*benefit=*/2);

      // Distribute warp reductions using our XOR butterfly function.
      vector::populateDistributeReduction(patterns, distributedReductionFn,
                                          /*benefit=*/1);

      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    // -----------------------------------------------------------------------
    // Step 6: Lower remaining WarpExecuteOnLane0 ops to scf.if guarded by
    // lane-0 check.  The warp allocation function uses ub.poison as a
    // stand-in (no actual shared memory needed after distribution).
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Step 3: NO OP.  Wait for hardware-native legalization.
    //
    // PREVIOUSLY: We lowered vector.warp_execute_on_lane_0 to scf.for loops
    // here.  That "shattered" the distribution metadata too early.
    // We now align with the IREE strategy by preserving the warp_execute ops
    // until ConvertVectorToGPU (Step 12.1 in Passes.cpp) handles them.
    // -----------------------------------------------------------------------
  }

  StringRef getArgument() const override {
    return "nova-gpu-vector-distribute";
  }
  StringRef getDescription() const override {
    return "Distribute warp-level vector ops across 32 threads using "
           "gpu.warp_execute_on_lane_0 and XOR-butterfly warp shuffles (SIMD-to-SIMT).";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPUUnrollToIntrinsicsPass
//
// Unrolls vector.contract ops to the native MMA shape read from the
// surrounding op's lowering_config (mma_kind attribute).  Falls back to
// {16, 16, 16} (WMMA_F32) when no config is present.
//
// Must run AFTER GenericVectorization (so vector.contract ops exist) and
// AFTER PackToIntrinsics (so operand shapes are already multiples of the
// MMA shape).  Runs BEFORE VectorDistribute so that each lane sees a single
// MMA-sized fragment after distribution.
//===----------------------------------------------------------------------===//

struct NovaGPUUnrollToIntrinsicsPass
    : public PassWrapper<NovaGPUUnrollToIntrinsicsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUUnrollToIntrinsicsPass)

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    llvm::errs() << "Checking Func: " << funcOp.getName() << "\n";

    // Collect per-op MMA shapes so the filter can look them up.
    // Walk upward from each vector.contract to find the nearest linalg op
    // carrying a non-zero mma_kind in its lowering_config.
    llvm::DenseMap<Operation *, SmallVector<int64_t, 3>> opShapeMap;
    bool anyMMAKind = false;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      Operation *cur = contractOp.getOperation();
      while (cur) {
        if (auto cfg = getLoweringConfig(cur)) {
          int32_t k = getMmaKindRaw(cfg);
          if (k != 0) {
            opShapeMap[contractOp.getOperation()] = getMMAShape(k);
            anyMMAKind = true;
            return;
          }
        }
        cur = cur->getParentOp();
      }
      // mma_kind == 0 means SIMT / CUDA-core path — record with a sentinel
      // so we know the op exists but must NOT be unrolled.
      opShapeMap[contractOp.getOperation()] = {};
    });

    // When every vector.contract in this function is on the SIMT (CUDA-core)
    // path (mma_kind == 0), skip unrolling entirely.  The vector.contract ops
    // will be lowered by the standard MLIR vector-to-loops path which emits
    // FMA instructions, matching the precision of the eager (non-JIT) path.
    // Unrolling with the {16,16,16} fallback would instead produce a sequence
    // of arith.mulf + arith.addf with two roundings per multiply-add, causing
    // systematic divergence from the reference output for large (e.g. 256x256)
    // matmuls.
    if (!anyMMAKind)
      return;

    // Use a single native shape derived from the first NON-EMPTY MMA contraction found
    // (all contractions in one MMA kernel share the same intrinsic shape).
    SmallVector<int64_t, 3> nativeShape;
    for (auto &kv : opShapeMap) {
      if (!kv.second.empty()) {
        nativeShape = kv.second;
        break;
      }
    }

    if (nativeShape.empty())
      return;

    // Save the MMA lowering config BEFORE unrolling erases the original ops.
    // populateVectorUnrollPatterns replaces original contracts with smaller
    // tiled ones but does NOT copy the lowering_config attribute.  We re-attach
    // it after rewriting so VectorDistributePass detects hasMMAContract=true.
    DictionaryAttr mmaConfigToPropagate;
    for (auto &kv : opShapeMap) {
      if (!kv.second.empty()) {
        if (auto cfg = getLoweringConfig(kv.first)) {
          mmaConfigToPropagate = cfg;
          break;
        }
        // [AUDIT] If the config is on a parent op (e.g. linalg.matmul outside the unrolled loop),
        // we must find it.
        Operation *cur = kv.first;
        while (cur) {
          if (auto cfg = getLoweringConfig(cur)) {
            mmaConfigToPropagate = cfg;
            break;
          }
          cur = cur->getParentOp();
        }
        if (mmaConfigToPropagate) break;
      }
    }

    vector::UnrollVectorOptions options;
    options.setNativeShape(nativeShape);
    // Unroll all vector.contract ops in this function to the shared native shape.
    // This handles unrolling even for newly created ops during the process.
    vector::populateVectorUnrollPatterns(patterns, options);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    // Re-attach MMA config to all newly-created unrolled vector.contract ops.
    if (mmaConfigToPropagate) {
      funcOp.walk([&](vector::ContractionOp contractOp) {
        if (!getLoweringConfig(contractOp))
          setLoweringConfig(contractOp, mmaConfigToPropagate);
      });
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-unroll-to-intrinsics";
  }
  StringRef getDescription() const override {
    return "Unroll vector.contract to the MMA-native shape from lowering_config.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory Functions and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUPackToIntrinsicsPass() {
  return std::make_unique<NovaGPUPackToIntrinsicsPass>();
}
void registerNovaGPUPackToIntrinsicsPass() {
  PassRegistration<NovaGPUPackToIntrinsicsPass>();
}

std::unique_ptr<Pass> createNovaGPUGenericVectorizationPass() {
  return std::make_unique<NovaGPUGenericVectorizationPass>();
}
void registerNovaGPUGenericVectorizationPass() {
  PassRegistration<NovaGPUGenericVectorizationPass>();
}

std::unique_ptr<Pass> createNovaGPUSubsetHoistingPass() {
  return std::make_unique<NovaGPUSubsetHoistingPass>();
}
void registerNovaGPUSubsetHoistingPass() {
  PassRegistration<NovaGPUSubsetHoistingPass>();
}

std::unique_ptr<Pass> createNovaGPUVectorizeMemrefCopyPass() {
  return std::make_unique<NovaGPUVectorizeMemrefCopyPass>();
}
void registerNovaGPUVectorizeMemrefCopyPass() {
  PassRegistration<NovaGPUVectorizeMemrefCopyPass>();
}

std::unique_ptr<Pass> createNovaGPUVectorDistributePass() {
  return std::make_unique<NovaGPUVectorDistributePass>();
}
void registerNovaGPUVectorDistributePass() {
  PassRegistration<NovaGPUVectorDistributePass>();
}

std::unique_ptr<Pass> createNovaGPUUnrollToIntrinsicsPass() {
  return std::make_unique<NovaGPUUnrollToIntrinsicsPass>();
}
void registerNovaGPUUnrollToIntrinsicsPass() {
  PassRegistration<NovaGPUUnrollToIntrinsicsPass>();
}

//===----------------------------------------------------------------------===//
// NovaGPULowerPackOpsPass
//
// Lowers linalg.pack / linalg.unpack ops to standard tensor dialect ops
// (tensor.pad, tensor.expand_shape, linalg.transpose, tensor.extract_slice)
// that implement BufferizableOpInterface.
//
// linalg.pack and linalg.unpack do NOT implement BufferizableOpInterface in
// MLIR 21. The NovaGPUPackToIntrinsicsPass creates linalg.pack ops on tensors
// during the MMA vectorization path. These must be lowered to bufferizable
// tensor ops BEFORE OneShotBufferize (Step 8) runs.
//
// Must run AFTER all vectorization passes (PackToIntrinsics, Generic,
// UnrollToIntrinsics, VectorDistribute) and BEFORE addNovaGPUBufferizePasses.
//===----------------------------------------------------------------------===//

struct NovaGPULowerPackOpsPass
    : public PassWrapper<NovaGPULowerPackOpsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPULowerPackOpsPass)

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    IRRewriter rewriter(funcOp.getContext());

    // Collect linalg.pack ops first (walk to avoid iterator invalidation).
    SmallVector<linalg::PackOp> packOps;
    funcOp.walk([&](linalg::PackOp op) { packOps.push_back(op); });
    for (linalg::PackOp op : packOps) {
      rewriter.setInsertionPoint(op);
      if (failed(linalg::lowerPack(rewriter, op))) {
        op.emitError("NovaGPULowerPackOpsPass: failed to lower linalg.pack");
        return signalPassFailure();
      }
    }

    // Collect linalg.unpack ops.
    SmallVector<linalg::UnPackOp> unpackOps;
    funcOp.walk([&](linalg::UnPackOp op) { unpackOps.push_back(op); });
    for (linalg::UnPackOp op : unpackOps) {
      rewriter.setInsertionPoint(op);
      if (failed(linalg::lowerUnPack(rewriter, op))) {
        op.emitError("NovaGPULowerPackOpsPass: failed to lower linalg.unpack");
        return signalPassFailure();
      }
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-lower-pack-ops";
  }
  StringRef getDescription() const override {
    return "Lower linalg.pack/linalg.unpack to bufferizable tensor ops "
           "(tensor.pad, tensor.expand_shape, linalg.transpose) before "
           "GPU-aware OneShotBufferize.";
  }
};

std::unique_ptr<Pass> createNovaGPULowerPackOpsPass() {
  return std::make_unique<NovaGPULowerPackOpsPass>();
}
void registerNovaGPULowerPackOpsPass() {
  PassRegistration<NovaGPULowerPackOpsPass>();
}

} // namespace mlir::nova
