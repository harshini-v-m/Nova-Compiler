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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
// Forward-declare pass factories defined in MLIRNovaVectorExt library.
namespace mlir::nova {
std::unique_ptr<mlir::Pass> createNovaVectorExtFoldUnitExtentDimsPass();
void registerNovaVectorExtFoldUnitExtentDimsPass();
std::unique_ptr<mlir::Pass> createNovaVectorizeVectorExtOpsPass();
void registerNovaVectorizeVectorExtOpsPass();
} // namespace mlir::nova

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


    pm.addNestedPass<mlir::func::FuncOp>(createFuseMatmulBiasPass());
    // pm.addNestedPass<mlir::func::FuncOp>(createNovaElementwiseOpFusionPass());
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


    pm.addNestedPass<func::FuncOp>(
        createNovaGPUApplyTilingLevelReductionPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(
        createNovaGPUPromoteMatmulOperandsPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaConfigTrackingCanonicalizerPass());
    pm.addPass(createCSEPass());

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

    // Fold unit dims first so ConfigureTensorLayouts wraps already-rank-reduced
    // tensors. If ConfigureTensorLayouts ran first, FoldUnitExtentDims would
    // insert extract_slice + empty + insert_slice around to_layout ops,
    // breaking the shared_outs alias chain needed for in-place bufferization.
    pm.addNestedPass<func::FuncOp>(
        createNovaVectorExtFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUConfigureTensorLayoutsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 6: Normalize loop bounds ──────────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
 

    // ── Step 6.6: Distribute orphaned ops ──────────────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUGreedilyDistributeToThreadsPass());
    pm.addPass(createCanonicalizerPass());


    // ── Step 6.75: Lower barrier regions ───────────────────────────────────
    pm.addNestedPass<func::FuncOp>(createNovaGPULowerBarrierRegionPass());

    pm.addNestedPass<func::FuncOp>(createNovaVectorizeVectorExtOpsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createNovaGenericVectorizationPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(
        createNovaGPUHoistVectorExtractInsertSlicePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createNovaGPUVectorAllocPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createNovaGPUCombineValueSemanticBarriersPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ── Step 8: GPU-aware bufferization (tensor → memref) ──────────────────
    addNovaGPUBufferizePasses(pm);

    pm.addNestedPass<func::FuncOp>(createNovaGPUVectorDistributePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
     pm.addNestedPass<func::FuncOp>(createNovaGPUReduceBankConflictsPass());
    // ── Step 8.5: Eliminate degenerate single-iteration foralls ────────────
    pm.addNestedPass<func::FuncOp>(createNovaNormalizeLoopBoundsPass());
    pm.addPass(createCanonicalizerPass());

    // Generalize remaining named linalg ops (linalg.copy {nova.promote_to_workgroup}
    // is already gone after bufferization, so this is safe here).
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    
    pm.addNestedPass<func::FuncOp>(createNovaGPUMapForallToGPUPass());

    // ── Step 11: Insert workgroup barriers ──────────────────────────────────
    pm.addNestedPass<func::FuncOp>(
        createNovaGPUInsertWorkgroupBarriersPass());

    // Safety net: sink any remaining index/scalar constants into launch bodies.
    pm.addPass(mlir::createGpuLaunchSinkIndexComputationsPass());

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
            // Fold extract/insert strided-slice ↔ transfer read/write chains
            // and cast away vector leading unit dims left by VectorDistribute.
            // PrepareVectorToMMAPatterns is intentionally NOT run here —
            // mma.sync ops are already emitted by NovaGPUVectorDistributePass
            // and PrepareVectorToMMA would introduce scalar smem loads for B
            // fragments instead of the vectorized loads VectorDistribute emits.
            {
              RewritePatternSet hwPatterns(&getContext());
              nova::populateGpuHardwareMappingStridedSlicePatterns(hwPatterns);
              vector::populateCastAwayVectorLeadingOneDimPatterns(hwPatterns);
              (void)applyPatternsAndFoldGreedily(funcOp,
                                                 std::move(hwPatterns));
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
    nvvmTargetOptions.features = "+ptx86";
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
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::nova::createFixHostGpuMemoryPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(mlir::nova::createGenerateDynamicWrapperPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

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
    registerNovaGPUCoalesceWorkgroupBuffersPass();
    registerNovaGPUConfigureTensorLayoutsPass();
    registerNovaVectorExtFoldUnitExtentDimsPass();
    registerNovaVectorizeVectorExtOpsPass();
    registerNovaGPUVectorAllocPass();
    registerNovaGPUCombineValueSemanticBarriersPass();
    registerNovaGPUVectorDistributePass();
    // NOTE: NovaGPUUnrollToIntrinsicsPass is a no-op stub — unrolling to MMA
    // intrinsics is handled inside NovaGPUVectorDistributePass (§4.5 MMAEmitter).
    // The pass registration is intentionally removed here to avoid confusion.
    registerNovaStrideReductionPass();

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
    addNovaComprehensiveBufferizePasses(pm);
  }

} // namespace mlir::nova