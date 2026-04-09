//===- Passes.cpp - Nova GPU Pass Pipeline --------------------------------===//
//
// Defines the Nova GPU optimized compilation pipeline and the
// addNovaGPUBufferizePasses helper.
//
// The main entry point is addNovaGPUOptimizedPipeline(), which mirrors IREE's
// addGPUTileAndFusePassPipeline in LLVMGPU/Passes.cpp.  It drives the full
// tensor → PTX lowering sequence: Nova dialect lowering, TOSA conversion,
// tiling and fusion, GPU-aware bufferization, forall-to-gpu.launch conversion,
// NVVM lowering, and final LLVM IR generation.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Transforms/Passes.h"
#include "Compiler/Transforms/GenerateDynamicWrapper.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "Compiler/Transforms/FixGpuLaunch.h"
#include "Compiler/Transforms/LLVMGPU/LoopSplit.h"
#include "Compiler/Transforms/LLVMGPU/NovaScfLoopUnroll.h"
#include "Compiler/Transforms/LLVMGPU/NovaScfLoopVectorize.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;

namespace mlir::nova
{

  void addNovaGPUOptimizedPipeline(OpPassManager &pm, StringRef cudaArch)
  {
    StringRef arch = cudaArch.empty() ? "sm_86" : cudaArch;

    // ── Nova dialect → Linalg/Arith/Tosa lowering ──────────────────────────
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createRemDevAttrPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::nova::createNovaToLinalgGenericLoweringPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::nova::createNovaToLinalgNamedPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToTensorPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // ── Step -1: Pre-tiling elementwise fusion ─────────────────────────────
    pm.addNestedPass<mlir::func::FuncOp>(createFuseMatmulBiasPass());
    pm.addNestedPass<mlir::func::FuncOp>(createNovaElementwiseOpFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(createNovaCheckInsParallelFuse());
    pm.addNestedPass<mlir::func::FuncOp>(
        createNovaLinalgHorizontalFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(createNovaMultiConsumerFusion());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step -0.5: Fold unit-extent dims ───────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 0: Select lowering strategy ───────────────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUSelectLoweringStrategyPass(arch));

    // ── Step 1: Tile and distribute to workgroups ───────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaTileAndDistributeToWorkgroupsPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 2: Pad operands ────────────────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPUPadOperandsPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 3: Promote to shared memory [BEFORE K-tiling] ─────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUPromoteMatmulOperandsPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 4: Tile K (reduction) dimension [AFTER promotion] ─────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelReductionPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 5: Tile threads ────────────────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPUApplyTilingLevelThreadPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 5.5: Configure tensor layouts ─────────────────────────────────
    // NEW: Computes per-operand NovaVectorLayout from lowering_config and
    // attaches as nova.layout_0/1/2 attributes on each MMA linalg op.
    // Must run AFTER thread tiling (iteration space is final) and BEFORE
    // vectorization (attributes must be on linalg ops so GenericVectorization
    // can copy them to vector.contract).
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUConfigureTensorLayoutsPass());


    // ── Step 6: Normalize loop bounds ──────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 6.5: Fuse and hoist parallel loops ─────────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUFuseAndHoistParallelLoopsPass());

    // ── Step 6.6: Distribute orphaned ops ──────────────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUGreedilyDistributeToThreadsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 6.75: Lower barrier regions ───────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPULowerBarrierRegionPass());

    // ── Step 7: Post-tiling linalg cleanup ─────────────────────────────────
    // sm_80+: keep named contraction ops for the vector.contract path.
    // GenericVectorization (Stage 18) vectorizes them directly to
    // vector.contract. Generalizing here would route them through
    // vector.multi_reduction instead, losing the MMA path.
    // pre-sm_80: generalize first for the SIMT outer-product path.
    if (arch.starts_with("sm_") && arch.substr(3).compare("80") >= 0) {
      pm.addPass(createLinalgElementwiseOpFusionPass());
    } else {
      pm.addPass(createLinalgGeneralizeNamedOpsPass());
      pm.addPass(createLinalgElementwiseOpFusionPass());
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Stage 18: Vectorization ─────────────────────────────────────────────
    // GenericVectorization:
    //   MMA path  (mma_kind != NONE): linalg contraction → vector.contract
    //   SIMT path (mma_kind == NONE): linalg.generic → vector.transfer + arith
    // OptimizeVectorTransfer: fold redundant transfer_read/write pairs.
    // DropVectorUnitDims: vector<1x16xf32> → vector<16xf32> for MMA matching.
    //
    // NOTE: CastTypeToFitMMA removed from here — it is a no-op for the
    // nvgpu.mma.sync path (sm_80+). It remains at 38b inside gpuPm where
    // it handles the WMMA path (sm_70–sm_75) post-outlining.
    pm.addNestedPass<func::FuncOp>(createNovaGenericVectorizationPass());
    // pm.addNestedPass<func::FuncOp>(createNovaGPUOptimizeVectorTransferPass());
    // pm.addNestedPass<func::FuncOp>(createNovaGPUDropVectorUnitDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(
        createNovaGPUHoistVectorExtractInsertSlicePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Stage 18.5: Shared-memory staging for MMA operands ─────────────────
    // Reads shared_mem=true from nova.layout_* attrs (set by Step 5.5) on
    // vector.contract ops and stages those operands through workgroup memory:
    //   gpu.barrier → alloc_tensor(workgroup) → transfer_write →
    //   nova.value_barrier → transfer_read → replace operand
    // Must run AFTER GenericVectorization (vector.contract ops must exist)
    // and BEFORE bufferization (alloc_tensor is lowered to __shared__ alloca).
    pm.addNestedPass<func::FuncOp>(createNovaGPUVectorAllocPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Stage 18.6: Combine value_barrier pairs ─────────────────────────────
    // VectorAllocPass emits two value_barrier pairs per A/B operand (one write
    // barrier on the vector, one read barrier on the workgroup tensor).  This
    // pass merges same-semantics barriers in each block into a single op so
    // that only ONE gpu.barrier is emitted after bufferization instead of N.
    // Must run AFTER VectorAllocPass and BEFORE bufferization.
    pm.addNestedPass<func::FuncOp>(createNovaGPUCombineValueSemanticBarriersPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Stage 19: Unroll vector.contract to MMA intrinsic shapes ───────────
    // Reads mma_kind from lowering_config. Unrolls to:
    //   MMA_SYNC_TF32_16x8x8  → 16×8×8 fragments
    //   MMA_SYNC_F16_16x8x16  → 16×8×16 fragments
    //   WMMA_F32_16x16x16     → 16×16×16 fragments
    // The extract_strided_slice ops produced here are hoisted/folded by
    // Stage 19.5 (HoistVectorExtractInsertSlice) and then by
    // // FoldExtractStridedSliceFromTransferRead in GpuHardwareMappingPass.
    // pm.addNestedPass<func::FuncOp>(createNovaGPUUnrollToIntrinsicsPass());
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(createCSEPass());

    // ── Stage 19.5: Hoist vector/tensor extract+insert slice pairs ──────────
    // Hoists loop-invariant extract_slice / insert_slice (both tensor and
    // vector dialects) out of scf.for loops and folds identity no-op slices.
    // 6-step pipeline:
    //   1. Move extracts to earliest valid position (improves LICM coverage).
    //   2. hoistRedundantVectorTransfers — transfer_read/write pairs.
    //   3. moveLoopInvariantCode        — standard MLIR LICM.
    //   4. hoistLoopInvariantSubsets    — extract+insert sharing iter_arg.
    //   5. hoistSubsetWithLoopInvariantTensor — insert into loop-invariant dest.
    //   6. Cleanup: CastLike{Extract,Insert}SliceFolder + Vector{Extract,Insert}
    //      StridedFolder + scf/vector canonicalization patterns.
    


    // ── Stage 21: Host-side vector scalarization [BEFORE bufferization] ────
    // targetRank=0 scalarizes all host-side vector ops.
    // Device-side vector.contract ops survive inside gpu.launch bodies and
    // are lowered by GpuHardwareMappingPass + ConvertVectorToGPU post-outlining.
    {
      auto &funcPm = pm.nest<func::FuncOp>();
      VectorTransferToSCFOptions opts;
      opts.setTargetRank(0);
      funcPm.addPass(createConvertVectorToSCFPass(opts));
      funcPm.addPass(createCanonicalizerPass());
      funcPm.addPass(createCSEPass());
    }

    // ── Step 8: GPU-aware bufferization (tensor → memref) ──────────────────
    addNovaGPUBufferizePasses(pm);
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertBufferizationToMemRefPass());

    // ── Stage 23: Vectorize shared memory copies [AFTER bufferization] ──────
    // Vectorizes linalg.copy on shared memory buffers → wide vector loads
    // targeting 128-bit LDS.128 instructions on sm_80+.
    pm.addNestedPass<func::FuncOp>(createNovaGPUVectorizeMemrefCopyPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 8.5: Eliminate degenerate single-iteration foralls ────────────
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Fill-copy forwarding and buffer coalescing ──────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPUFillCopyForwardingPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUCoalesceWorkgroupBuffersPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());


    // ── Step 9: scf.forall → gpu.launch ────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPUMapForallToGPUPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 10: Lower linalg → scf loops ──────────────────────────────────
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Scalar accumulator + warp shuffle reduction ─────────────────────────
    pm.addNestedPass<func::FuncOp>(
        mlir::nova::createSCFScalarizeAccumulatorPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(
        mlir::nova::createNovaWarpShuffleReductionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Loop optimizations ──────────────────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(mlir::nova::createNovaScfLoopSplitPass());
    pm.addPass(createCanonicalizerPass());

    // ── Reposition stores ───────────────────────────────────────────────────
    pm.addNestedPass<mlir::func::FuncOp>(createNovaRepositionStorePass());

    // ── Step 11: Insert workgroup barriers ──────────────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUInsertWorkgroupBarriersPass());

    // ── Step 12: Outline gpu.launch → gpu.module kernels ───────────────────
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(mlir::nova::createRenameGpuKernelsPass());
    pm.addPass(createCanonicalizerPass());

    // ── Stage 35: Device-side hardware mapping (inside gpu.GPUModuleOp) ────
    {
      auto &gpuHwPm = pm.nest<gpu::GPUModuleOp>();

      gpuHwPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct GpuHardwareMappingPass
            : public PassWrapper<GpuHardwareMappingPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuHardwareMappingPass)

          void runOnOperation() override {
            auto funcOp = getOperation();

            // Step 1: Shield non-indexing divui/remui from the legalizer.
            funcOp.walk([&](Operation *op) {
              auto isIndexingMath = [](Value v) -> bool {
                for (auto &use : v.getUses()) {
                  if (auto r = dyn_cast<vector::TransferReadOp>(
                          use.getOwner()))
                    for (auto idx : r.getIndices())
                      if (idx == v) return true;
                  if (auto w = dyn_cast<vector::TransferWriteOp>(
                          use.getOwner()))
                    for (auto idx : w.getIndices())
                      if (idx == v) return true;
                }
                return false;
              };
              if (auto divOp = dyn_cast<arith::DivUIOp>(op)) {
                if (!isIndexingMath(divOp.getResult())) {
                  OpBuilder b(divOp);
                  auto map = AffineMap::get(
                      2, 0,
                      b.getAffineDimExpr(0).floorDiv(
                          b.getAffineDimExpr(1)),
                      b.getContext());
                  auto applyOp = b.create<affine::AffineApplyOp>(
                      divOp.getLoc(), map,
                      ValueRange{divOp.getLhs(), divOp.getRhs()});
                  divOp.replaceAllUsesWith(applyOp.getResult());
                  divOp.erase();
                }
              } else if (auto remOp = dyn_cast<arith::RemUIOp>(op)) {
                if (!isIndexingMath(remOp.getResult())) {
                  OpBuilder b(remOp);
                  auto map = AffineMap::get(
                      2, 0,
                      b.getAffineDimExpr(0) % b.getAffineDimExpr(1),
                      b.getContext());
                  auto applyOp = b.create<affine::AffineApplyOp>(
                      remOp.getLoc(), map,
                      ValueRange{remOp.getLhs(), remOp.getRhs()});
                  remOp.replaceAllUsesWith(applyOp.getResult());
                  remOp.erase();
                }
              }
            });

            // Step 2: Fold extract_strided_slice(transfer_read) → narrowed
            //         transfer_read so LHS reaches MMA prepare patterns.
            {
              struct FoldExtractStridedSliceFromTransferRead
                  : public OpRewritePattern<vector::ExtractStridedSliceOp> {
                using OpRewritePattern::OpRewritePattern;
                LogicalResult matchAndRewrite(
                    vector::ExtractStridedSliceOp extractOp,
                    PatternRewriter &rewriter) const override {
                  auto extractType =
                      cast<VectorType>(extractOp.getType());
                  if (extractType.getRank() != 2) return failure();
                  auto readOp =
                      extractOp.getVector()
                          .getDefiningOp<vector::TransferReadOp>();
                  if (!readOp) return failure();
                  if (!isa<MemRefType>(readOp.getBase().getType()))
                    return failure();
                  if (readOp.getMask() || readOp.hasOutOfBoundsDim())
                    return failure();
                  auto strides = extractOp.getStrides()
                                     .getAsValueRange<IntegerAttr>();
                  if (!llvm::all_of(strides, [](const APInt &v) {
                        return v.isOne();
                      }))
                    return failure();
                  SmallVector<int64_t> offsets;
                  for (auto attr :
                       extractOp.getOffsets()
                           .getAsValueRange<IntegerAttr>())
                    offsets.push_back(attr.getSExtValue());
                  if ((int64_t)offsets.size() !=
                      (int64_t)readOp.getIndices().size())
                    return failure();
                  Location loc = extractOp.getLoc();
                  SmallVector<Value> newIndices =
                      llvm::to_vector(readOp.getIndices());
                  for (size_t i = 0; i < offsets.size(); ++i) {
                    if (offsets[i] != 0) {
                      Value off =
                          rewriter.create<arith::ConstantIndexOp>(
                              loc, offsets[i]);
                      newIndices[i] =
                          rewriter.create<arith::AddIOp>(
                              loc, newIndices[i], off);
                    }
                  }
                  SmallVector<bool> inBounds(2, true);
                  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
                      extractOp, extractType, readOp.getBase(),
                      newIndices, readOp.getPermutationMapAttr(),
                      readOp.getPadding(), /*mask=*/Value{},
                      rewriter.getBoolArrayAttr(inBounds));
                  return success();
                }
              };
              RewritePatternSet foldPatterns(&getContext());
              foldPatterns
                  .add<FoldExtractStridedSliceFromTransferRead>(
                      &getContext());
              (void)applyPatternsAndFoldGreedily(
                  funcOp, std::move(foldPatterns));
            }

            // Step 3: Prepare transfers → nvgpu fragment form.
            {
              RewritePatternSet patterns(&getContext());
              populatePrepareVectorToMMAPatterns(patterns,
                                                 /*useNvGpu=*/true);
              if (failed(applyPatternsAndFoldGreedily(
                      funcOp, std::move(patterns))))
                return signalPassFailure();
            }
          }
          StringRef getArgument() const override {
            return "nova-gpu-hardware-mapping";
          }
        };
        return std::make_unique<GpuHardwareMappingPass>();
      }());

      // vector.contract → nvgpu.mma.sync
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>(
          createConvertVectorToGPUPass(/*useNvGpu=*/true));

      // Fix missing tf32_enabled on f32 nvgpu.mma.sync ops
      gpuHwPm.addNestedPass<gpu::GPUFuncOp>(
          [&]() -> std::unique_ptr<Pass> {
        struct FixMmaSyncTF32Pass
            : public PassWrapper<FixMmaSyncTF32Pass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixMmaSyncTF32Pass)
          void runOnOperation() override {
            getOperation().walk([](nvgpu::MmaSyncOp mmaSyncOp) {
              auto aType = dyn_cast<VectorType>(
                  mmaSyncOp.getMatrixA().getType());
              if (aType && aType.getElementType().isF32() &&
                  !mmaSyncOp.getTf32Enabled())
                mmaSyncOp.setTf32Enabled(true);
            });
          }
          StringRef getArgument() const override {
            return "nova-fix-mma-sync-tf32";
          }
        };
        return std::make_unique<FixMmaSyncTF32Pass>();
      }());

      gpuHwPm.addPass(createCanonicalizerPass());
      gpuHwPm.addPass(createCSEPass());
    }

    // ── Step 12.5: Convert workgroup alloc → memref.global ─────────────────
    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      gpuPm.addPass(createNovaConvertSharedMemAllocsPass());
      gpuPm.addPass(createCanonicalizerPass());
    }

    // ── Step 12.75: Convert host allocs → gpu.alloc ─────────────────────────
    pm.addPass(nova::createConvertMemRefToGpuPass());
    pm.addPass(createCanonicalizerPass());

    // ── Step 13: NVVM lowering ──────────────────────────────────────────────
    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip = arch.str();
    nvvmTargetOptions.features = "+ptx76";
    nvvmTargetOptions.optLevel = 3;
    nvvmTargetOptions.fastFlag = true;
    nvvmTargetOptions.ftzFlag = true;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

    {
      auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
      gpuPm.addPass(memref::createExpandStridedMetadataPass());
      gpuPm.addPass(memref::createFoldMemRefAliasOpsPass());
      // CastTypeToFitMMA: correct position — handles WMMA strided-memref
      // index normalization for sm_70–sm_75. No-op for nvgpu path (sm_80+).
      gpuPm.addPass(createNovaGPUCastTypeToFitMMAPass());
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createLowerAffinePass());
      gpuPm.addPass(createNovaGPULowerMemorySpacePass());
      gpuPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct VectorToSCFOnGPUFuncPass
            : public PassWrapper<VectorToSCFOnGPUFuncPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
              VectorToSCFOnGPUFuncPass)
          void runOnOperation() override {
            RewritePatternSet patterns(&getContext());
            VectorTransferToSCFOptions opts;
            populateVectorToSCFConversionPatterns(patterns, opts);
            if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                    std::move(patterns))))
              signalPassFailure();
          }
          StringRef getArgument() const override {
            return "nova-vector-to-scf-gpu-func";
          }
        };
        return std::make_unique<VectorToSCFOnGPUFuncPass>();
      }());
      gpuPm.addNestedPass<gpu::GPUFuncOp>(createConvertVectorToLLVMPass());
      gpuPm.addPass(createSCFToControlFlowPass());
      gpuPm.addPass(createConvertNVGPUToNVVMPass());
      ConvertGpuOpsToNVVMOpsOptions nvvmOpts;
      gpuPm.addPass(createConvertGpuOpsToNVVMOps(nvvmOpts));
      gpuPm.addPass(createLowerAffinePass());
      gpuPm.addPass(createConvertIndexToLLVMPass());
      gpuPm.addPass(createArithToLLVMConversionPass());
      gpuPm.addPass(createConvertMathToLLVMPass());
      gpuPm.addPass(createConvertVectorToLLVMPass());
      gpuPm.addPass(createReconcileUnrealizedCastsPass());
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    GpuModuleToBinaryPassOptions binaryOptions;
    binaryOptions.toolkitPath = "/usr/local/cuda-13.0";
    binaryOptions.compilationTarget = "isa";
    pm.addPass(createGpuModuleToBinaryPass(binaryOptions));

    pm.addPass(createNovaGPULowerMemorySpacePass());
    GpuToLLVMConversionPassOptions hostOpts;
    pm.addPass(createGpuToLLVMConversionPass(hostOpts));
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(mlir::nova::createGenerateDynamicWrapperPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  void registerNovaLLVMGPUPasses()
  {
    registerNovaConfigTrackingCanonicalizerPass();
    registerNovaGPUSelectLoweringStrategyPass();
    registerNovaTileAndDistributePass();
    registerNovaGPUPadOperandsPass();
    registerNovaGPUPromoteMatmulOperandsPass();
    registerNovaGPUApplyTilingLevelReductionPass();
    registerNovaGPUApplyTilingLevelThreadPass();
    registerNovaGPUApplyTilingLevelSubgroupPass();
    registerNovaGPUFuseAndHoistParallelLoopsPass();
    registerNovaGPUGreedilyDistributeToThreadsPass();
    registerNovaGPULowerBarrierRegionPass();
    registerNovaGPUEraseFusionBarriersPass();
    registerNovaGPUInferMemorySpacePass();
    registerNovaGPUComprehensiveBufferizePass();
    registerNovaGPUInsertWorkgroupBarriersPass();
    registerNovaNormalizeLoopBoundsPass();
    registerNovaEliminateEmptyTensorsPass();
    registerNovaConvertSharedMemAllocsPass();
    registerNovaGPUMapForallToGPUPass();
    registerNovaGPULowerMemorySpacePass();
    registerNovaWarpShuffleReductionPass();
    registerNovaGPUFillCopyForwardingPass();
    registerNovaGPUCoalesceWorkgroupBuffersPass();
    registerNovaGPUConfigureTensorLayoutsPass();
    registerNovaGPUVectorAllocPass();
    registerNovaGPUCombineValueSemanticBarriersPass();

    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return mlir::nova::createNovaScfLoopUnrollPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return mlir::createForLoopSpecializationPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return mlir::createForLoopPeelingPass();
    });

    PassPipelineRegistration<>(
        "nova-gpu-optimized-pipeline",
        "Nova GPU Optimized Pipeline (tile+fuse → normalize → bufferize → "
        "gpu.launch → linalg-to-loops)",
        [](OpPassManager &pm) {
          addNovaGPUOptimizedPipeline(pm, "");
        });
  }

  void addNovaGPUBufferizePasses(OpPassManager &pm)
  {
    pm.addNestedPass<func::FuncOp>(createNovaEliminateEmptyTensorsPass());
    pm.addNestedPass<func::FuncOp>(
        bufferization::createEmptyTensorToAllocTensorPass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUInferMemorySpacePass());
    pm.addPass(createNovaGPUComprehensiveBufferizePass());
    pm.addNestedPass<func::FuncOp>(
        memref::createResolveShapedTypeResultDimsPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    bufferization::BufferDeallocationPipelineOptions deallocOpts;
    bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);
  }

} // namespace mlir::nova