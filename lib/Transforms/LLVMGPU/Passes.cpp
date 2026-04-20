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
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

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

    // ── Step 3: Tile K (reduction) dimension ───────────────────────────────
    // Must happen BEFORE operand promotion so that linalg.copy inserted by
    // the promote pass covers only [wgM × kStep] (one K-tile) rather than
    // the full [wgM × K] buffer.  K-tiling fuses the copy into the K-loop
    // so each iteration cooperatively loads a kStep-wide slice into shared
    // memory, enabling proper double-buffering and register reuse.
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelReductionPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 4: Promote operands to shared memory [INSIDE K-loop] ──────────
    // Runs after K-tiling so the promoted linalg.copy lives inside the K-loop
    // and covers exactly [wgM × kStep] per iteration, not the full K extent.
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUPromoteMatmulOperandsPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 4.5: Tile subgroups (MMA ops → #gpu.warp scf.forall) ──────────
    // Only fires for ops where "subgroup" tiles != 0 (i.e. mma_kind != 0).
    // Copy/fill/elementwise ops have subgroup=0 and are skipped entirely.
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelSubgroupPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 5: Tile threads (copy/fill/elementwise → #gpu.thread forall) ──
    // MMA ops have threadTiles=0 after Step 0 and are skipped here.
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
    pm.addNestedPass<func::FuncOp>(createNovaGenericVectorizationPass());
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

    // ── Step 8: GPU-aware bufferization (tensor → memref) ──────────────────
    addNovaGPUBufferizePasses(pm);
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertBufferizationToMemRefPass());

    pm.addNestedPass<func::FuncOp>(createNovaGPUVectorDistributePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 8.5: Eliminate degenerate single-iteration foralls ────────────
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());

    // ── Fill-copy forwarding and buffer coalescing ──────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPUFillCopyForwardingPass());
    pm.addNestedPass<func::FuncOp>(createNovaGPUCoalesceWorkgroupBuffersPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 9: scf.forall → gpu.launch ────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPUMapForallToGPUPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 9.5: Sink arith.constant ops into gpu.launch bodies ───────────
    // Prevents dense<true> mask constants (e.g. vector.gather all-true mask)
    // from being captured as kernel arguments by the outliner (Step 12).
    // Without this, the PTX backend sees an opaque runtime predicate and emits
    // 4 serial @%p-guarded ld.global instead of 4 parallel unconditional loads.
    pm.addPass(mlir::createGpuLaunchSinkIndexComputationsPass());

    // ── Step 10: Lower linalg → scf loops ──────────────────────────────────
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Scalar accumulator: scalarize loop accumulators, atomicize cross-block
    // stores, and insert gpu.memset before distributed reduction launches.
    pm.addNestedPass<func::FuncOp>(
        mlir::nova::createSCFScalarizeAccumulatorPass());
     pm.addNestedPass<func::FuncOp>(
        mlir::vector::createLowerVectorMultiReductionPass(
            mlir::vector::VectorMultiReductionLowering::InnerReduction));
    pm.addPass(createCanonicalizerPass());
    
    // pm.addNestedPass<func::FuncOp>(
    //     mlir::nova::createNovaWarpShuffleReductionPass());
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(createCSEPass());
   
    // ── Reposition stores ───────────────────────────────────────────────────
    // pm.addNestedPass<mlir::func::FuncOp>(createNovaRepositionStorePass());

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

            {
              RewritePatternSet castPatterns(&getContext());
              vector::populateCastAwayVectorLeadingOneDimPatterns(
                  castPatterns);
              (void)applyPatternsAndFoldGreedily(
                  funcOp, std::move(castPatterns));
            }

            // Step 3: Prepare transfers → nvgpu fragment form.
            {
              // Collect original contracts and their configs before unrolling
              // replaces them.
              llvm::DenseMap<Operation *, DictionaryAttr> configs;
              funcOp.walk([&](vector::ContractionOp op) {
                if (auto cfg = op->getAttrOfType<DictionaryAttr>(
                        "lowering_config"))
                  configs[op] = cfg;
              });

              RewritePatternSet patterns(&getContext());
              populatePrepareVectorToMMAPatterns(patterns,
                                                 /*useNvGpu=*/true);
              if (failed(applyPatternsAndFoldGreedily(
                      funcOp, std::move(patterns))))
                return signalPassFailure();

              // Re-attach configs to the new prepared contracts.
              // In this stage there's typically only one matmul being prepared.
              if (!configs.empty()) {
                DictionaryAttr sharedCfg = configs.begin()->second;
                funcOp.walk([&](vector::ContractionOp op) {
                  if (!op->hasAttr("lowering_config"))
                    op->setAttr("lowering_config", sharedCfg);
                });
              }
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

      gpuPm.addNestedPass<gpu::GPUFuncOp>([&]() -> std::unique_ptr<Pass> {
        struct FoldStructMemrefRoundtripsPass
            : public PassWrapper<FoldStructMemrefRoundtripsPass,
                                 OperationPass<gpu::GPUFuncOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
              FoldStructMemrefRoundtripsPass)
          void runOnOperation() override {
            getOperation().walk([](UnrealizedConversionCastOp castToMemref) {
              // Match: struct → memref (first cast)
              if (castToMemref.getInputs().size() != 1 ||
                  castToMemref.getOutputs().size() != 1)
                return;
              if (!isa<MemRefType>(castToMemref.getResult(0).getType()))
                return;
              if (!isa<LLVM::LLVMStructType>(
                      castToMemref.getOperand(0).getType()))
                return;
              // Check: single user is memref → struct (second cast)
              Value memrefVal = castToMemref.getResult(0);
              for (Operation *user : llvm::make_early_inc_range(
                       memrefVal.getUsers())) {
                auto castBack =
                    dyn_cast<UnrealizedConversionCastOp>(user);
                if (!castBack || castBack.getInputs().size() != 1 ||
                    castBack.getOutputs().size() != 1)
                  continue;
                if (!isa<LLVM::LLVMStructType>(
                        castBack.getResult(0).getType()))
                  continue;
                // Replace uses of the round-trip result with the original struct
                castBack.getResult(0).replaceAllUsesWith(
                    castToMemref.getOperand(0));
                castBack.erase();
              }
            });
          }
          StringRef getArgument() const override {
            return "nova-fold-struct-memref-roundtrips";
          }
        };
        return std::make_unique<FoldStructMemrefRoundtripsPass>();
      }());
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
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(mlir::createUBToLLVMConversionPass());
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
    registerNovaGPUVectorDistributePass();
    registerNovaGPUUnrollToIntrinsicsPass();

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