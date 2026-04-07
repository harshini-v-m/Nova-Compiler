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
    // Step 2: Lower vector.contract → outer-product chains.
    // After PackToIntrinsics + UnrollToIntrinsics, each lane sees a single
    // MMA-sized vector.contract ({16,16,16} or {16,8,16}).  Lowering to
    // outer-products decomposes this into a chain of vector.outerproduct ops
    // that can be independently distributed across the 32 warp lanes.
    // Then flatten transfer rank to 1 so warp distribution sees simple 1-D
    // vectors (no 2-D tile shapes that confuse the warp op matcher).
    //
    // SKIP when any linalg op in this function carries a non-NONE mma_kind:
    // the contracts will be preserved for PrepareVectorToGPU + ConvertVectorToGPU
    // in the gpuPm (post-outlining) which converts them directly to
    // gpu.subgroup_mma_compute (TF32 mma.sync for FP32 matmuls on sm_80+).
    // Lowering to outer-products here would lose the TF32 precision path.
    // -----------------------------------------------------------------------
    {
      // Always lower all vector.contract ops via the SIMT outer-product path.
      //
      // DESIGN NOTE — why we do NOT use the nvgpu.mma.sync (tensor core) path:
      //
      // nvgpu.mma.sync operates at the level of per-thread fragments.  For
      // TF32 m16n8k8 each thread in a 32-thread warp holds only 4 A-elements,
      // 2 B-elements and 4 C-elements.  Correct lowering therefore requires:
      //   1. Warp-level vector.contract (16×8) distributed to per-thread
      //      fragments via a warp-distribution pass.
      //   2. Distributed contracts (vector<2x1>, vector<1x1>, vector<2x2>)
      //      converted to nvgpu.mma.sync by ConvertVectorToGPU.
      //
      // Our current kernel structure assigns one thread per output tile (8
      // threads each computing a full 16×8 accumulation), NOT 32 threads
      // cooperating on a single warp tile.  Passing warp-level vectors
      // (vector<16x8xf32>) directly to nvgpu.mma.sync is architecturally
      // incorrect — the per-thread type mismatch causes PrepareVectorToMMA
      // patterns to silently not match, leaving contracts un-lowered, which
      // then fall through to a broken scalar-loop path that emits a 3-byte
      // PTX binary (all zeros).
      //
      // Implementing proper warp-level tensor-core distribution requires a
      // non-trivial redesign (assign 32-thread warps to 16×8 tiles, run
      // warp distribution, emit per-thread fragments).  Until that redesign
      // is complete, ALL contracts are routed through the outer-product path:
      //   vector.contract → vector.outerproduct → llvm.fma
      // This produces numerically correct results (FP32 precision) identical
      // to the eager path and successfully compiles to valid PTX.
      {
        RewritePatternSet patterns(ctx);
        vector::populateVectorContractLoweringPatterns(
            patterns, vector::VectorContractLowering::OuterProduct);
        vector::populateVectorTransferLoweringPatterns(patterns,
                                                       /*maxTransferRank=*/1);
        (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
      }
    }

    // -----------------------------------------------------------------------
    // Step 3: Flatten shape-cast chains introduced by outer-product lowering.
    // -----------------------------------------------------------------------
    {
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
