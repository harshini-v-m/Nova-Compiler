//===- NovaGPUVectorization.cpp - Nova GPU Vectorization Passes -----------===//
//
// Defines vectorization passes for the Nova GPU pipeline.
//
// Passes defined here:
//   NovaGPUVectorizeMemrefCopyPass — vectorize memref.copy (global→shared)
//   NovaGPUUnrollToIntrinsicsPass  — unroll vector.contract to MMA-native
//                                    shape read from the op's lowering_config
//   NovaGPULowerPackOpsPass        — lower linalg.pack/unpack before bufferize
//
// NovaGPUGenericVectorizationPass lives in NovaGPUGenericVectorization.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APInt.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"

using namespace mlir;

namespace mlir::nova {

namespace {



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
    // Must have exactly 2 ins and 1 out, 3 loops [par, par, red].
    auto iterTypes = op.getIteratorTypesArray();
    if (iterTypes.size() != 3) return false;
    if (iterTypes[0] != utils::IteratorType::parallel ||
        iterTypes[1] != utils::IteratorType::parallel ||
        iterTypes[2] != utils::IteratorType::reduction) return false;
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
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2) return false;

    // Verify indexing maps are exactly: [d0,d2], [d2,d1], [d0,d1].
    auto maps = op.getIndexingMapsArray();
    MLIRContext *ctx = op.getContext();
    auto expectedA = AffineMap::get(3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx);
    auto expectedB = AffineMap::get(3, 0, {getAffineDimExpr(2, ctx), getAffineDimExpr(1, ctx)}, ctx);
    auto expectedC = AffineMap::get(3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
    if (maps[0] != expectedA || maps[1] != expectedB || maps[2] != expectedC) return false;

    int64_t M = cTy.getShape()[0], N = cTy.getShape()[1], K = aTy.getShape()[1];
    Type elemTy = cTy.getElementType();
    auto lhsVT = VectorType::get({M, K}, elemTy);
    auto rhsVT = VectorType::get({K, N}, elemTy);
    auto accVT = VectorType::get({M, N}, elemTy);

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemTy));
    SmallVector<Value> zeroIdx = {rewriter.create<arith::ConstantIndexOp>(loc, 0),
                                  rewriter.create<arith::ConstantIndexOp>(loc, 0)};
    // in_bounds = [true, true] since shapes are static and we verified sizes match.
    SmallVector<bool> inBounds2D = {true, true};
    Value lhs = rewriter.create<vector::TransferReadOp>(loc, lhsVT, ins[0], zeroIdx,
                                                        zero, inBounds2D);
    Value rhs = rewriter.create<vector::TransferReadOp>(loc, rhsVT, ins[1], zeroIdx,
                                                        zero, inBounds2D);
    Value acc = rewriter.create<vector::TransferReadOp>(loc, accVT, outs[0], zeroIdx,
                                                        zero, inBounds2D);

    // Build vector.contract with the same maps.
    SmallVector<AffineMap> cMaps = {expectedA, expectedB, expectedC};
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

    // [RECTIFICATION] Also carry the config to the transfer_write to ensure 
    // stability through bufferization/hoisting.
    auto writeOp = rewriter.create<vector::TransferWriteOp>(loc, contractOp.getResult(), outs[0], zeroIdx,
                                                             inBounds2D);
    if (auto cfg = getLoweringConfig(op.getOperation()))
      setLoweringConfig(writeOp, cfg);
    rewriter.replaceOp(op, writeOp->getResult(0));
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
// NovaGPUUnrollToIntrinsicsPass
//
// NOTE: This pass is now a no-op stub. M/N-batch unrolling to the native MMA
// intrinsic shape has been absorbed into NovaGPUVectorDistributePass (§4.5),
// which performs the unroll inline per K-step immediately after distributing
// per-thread slices. Running a separate pass-over of already-distributed IR
// is unnecessary and was the root cause of the massive static unroll explosion
// (512+ contracts, 24KB register spill) seen in linear.log.
//
// The registration is kept so existing pipeline strings using
// --nova-gpu-unroll-to-intrinsics don't break, but the pass is a no-op.
//===----------------------------------------------------------------------===//

struct NovaGPUUnrollToIntrinsicsPass
    : public PassWrapper<NovaGPUUnrollToIntrinsicsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUUnrollToIntrinsicsPass)

  void runOnOperation() override {
    // No-op: unrolling is now handled inside NovaGPUVectorDistributePass.
  }

  StringRef getArgument() const override {
    return "nova-gpu-unroll-to-intrinsics";
  }
  StringRef getDescription() const override {
    return "(no-op) M/N-batch unroll is now inline in nova-gpu-vector-distribute.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory Functions and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUVectorizeMemrefCopyPass() {
  return std::make_unique<NovaGPUVectorizeMemrefCopyPass>();
}
void registerNovaGPUVectorizeMemrefCopyPass() {
  PassRegistration<NovaGPUVectorizeMemrefCopyPass>();
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
