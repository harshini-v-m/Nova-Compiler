// convert to llvm includes
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
// other conversion includes from mlir
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
// utils
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#include "Compiler/Transforms/AddGpuMemoryCopies.h"
#include "Compiler/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/PassRegistry.h"
// buffer includes
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

// nova dialect includes
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
// optimization passes includes
#include "Compiler/Transforms/FixGpuLaunch.h"
#include "Compiler/Transforms/FuseMatmulBias.h"
#include "Compiler/Transforms/RenameGpuKernels.h"

// gpu
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"    // For GPUModuleOp
#include "mlir/Dialect/GPU/Pipelines/Passes.h" // For GpuNVVMAttachTarget
#include "mlir/Dialect/GPU/Transforms/Passes.h" // For createGpuKernelOutliningPass
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

// lowering passes
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

// header of this file
#include "Compiler/Pipeline/Gpupipeline.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"

using namespace mlir;

namespace mlir
{
    namespace nova
    {
        namespace
        {
            // Custom passes removed in favor of standard MLIR passes.
        }
        void createNovaGPUPipelines(mlir::OpPassManager &pm)
        {
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createRemDevAttrPass());
            pm.addPass(mlir::nova::createNovaToTosaLoweringPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createNovaToGpuPass());
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::nova::createNovaToLinalgLoweringPass());

            // 2. TOSA TO LINALG (Named and regular)
            pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
            // This enables the 2:4 structured sparsity hardware path on your RTX 3060.
            // pm.addPass(mlir::createSparsificationPass());
            // pm.addPass(mlir::createSparseTensorConversionPass());

            // 3. TOSA TO ARITH/TENSOR/SCF
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToTensorPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToSCFPass());
            pm.addPass(mlir::createCanonicalizerPass());


            // 4. LINALG OPT to linalg generalize pass
            pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createFuseMatmulBiasPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());

            // Tiling is handled in Section 6 after parallel loop conversion
            pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
            // pm.addNestedPass<mlir::func::FuncOp>(
            //     mlir::createLinalgGeneralizeNamedOpsPass());

            // 5. BUFFERIZATION & DEALLOCATION
            bufferization::OneShotBufferizePassOptions bufferizeOptions;
            bufferizeOptions.bufferizeFunctionBoundaries = true;
            bufferizeOptions.functionBoundaryTypeConversion = bufferization::LayoutMapOption::IdentityLayoutMap;
            bufferizeOptions.useEncodingForMemorySpace = true;
            bufferizeOptions.allowUnknownOps = true;
            pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

            bufferization::BufferDeallocationPipelineOptions deallocationOptions;
            bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);
            pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferHoistingPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());
            pm.addPass(mlir::createConvertBufferizationToMemRefPass());
            // device attribute handling pass
            // pm.addPass(mlir::nova::createConvertMemRefToGpuPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());

            // lowering through Affine
            pm.addPass(mlir::createCanonicalizerPass());

            pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
            pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
            //add this pass --affine-expand-index-ops-as-affine
            pm.addPass(mlir::affine::createAffineExpandIndexOpsAsAffinePass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            //affine fusion pass
            pm.addPass(mlir::affine::createLoopFusionPass(1, 1024, true, mlir::affine::FusionMode::Greedy));
            pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createAffineScalarizeAccumulatorPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopInvariantCodeMotionPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createRaiseMemrefToAffine());
            
            // Setting explicit tile sizes {128, 8, ...} via text pipeline 
            // since createLoopTilingPass(cacheSize) uses a heuristic.
            if (failed(mlir::parsePassPipeline("func.func(affine-loop-tile{tile-sizes=128,8})", pm)))
                llvm::errs() << "Pipeline parsing failed for affine-loop-tile\n";

            // pm.addNestedPass<mlir::func::FuncOp>(  
            //     mlir::affine::createAffineDataCopyGenerationPass(  
            //         /*slowMemorySpace=*/0,  
            //         /*fastMemorySpace=*/3,  
            //         /*tagMemorySpace=*/0,  
            //         /*minDmaTransferSize=*/1024,  
            //         /*fastMemCapacityBytes=*/32*1024  
            //     )  
            // );

            if (failed(mlir::parsePassPipeline(
                    "func.func(affine-super-vectorize{virtual-vector-size=8 "
                    "vectorize-reductions=true})",
                    pm))) {
                llvm::errs() << "Failed to parse vectorize pipeline\n";  
            }  
            pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopNormalizePass());  
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createAddGpuMemoryCopiesPass());
            
            // Map tiled loops to parallel loops. We use parsePassPipeline to avoid 
            // the linker error for createAffineParallelizePass.
            if (failed(mlir::parsePassPipeline("func.func(affine-parallelize)", pm)))
                llvm::errs() << "Pipeline parsing failed for affine-parallelize\n";

            pm.addPass(mlir::createLowerAffinePass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopFusionPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // Use the modern SCF-based mapping which is much more robust than 
            // ConvertAffineForToGPU for tiled and non-perfectly nested loops.
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertParallelLoopToGpuPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            pm.addPass(mlir::nova::createConvertMemRefToGpuPass());
            pm.addPass(mlir::createGpuKernelOutliningPass());
            pm.addPass(mlir::nova::createRenameGpuKernelsPass());

            mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
            nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
            nvvmTargetOptions.chip = "sm_86";
            nvvmTargetOptions.optLevel = 3;
            nvvmTargetOptions.fastFlag = true;
            nvvmTargetOptions.ftzFlag = true;
            pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

            // Lower vectors on host before they become llvm.func
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToLLVMPass());

            // Lowering INSIDE the GPU Module (Fixes 'index' in kernels)
            auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
            gpuPm.addPass(mlir::createConvertVectorToSCFPass());
            gpuPm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToGPUPass(true));
            gpuPm.addPass(mlir::createLowerAffinePass());

            gpuPm.addPass(mlir::createSCFToControlFlowPass());
            gpuPm.addPass(mlir::createArithToLLVMConversionPass());
            gpuPm.addPass(mlir::createConvertVectorToLLVMPass());

            mlir::ConvertGpuOpsToNVVMOpsOptions nvvmOptions;
            nvvmOptions.useBarePtrCallConv = true; // Use bare pointers to match host
            gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvmOptions));
            gpuPm.addPass(mlir::nova::createFixGpuKernelSignaturePass());
            gpuPm.addPass(mlir::createConvertIndexToLLVMPass());
            gpuPm.addPass(mlir::createConvertMathToLLVMPass());
            gpuPm.addPass(mlir::createReconcileUnrealizedCastsPass());

            // MAIN LOWERING: gpu.launch_func -> runtime calls (Host side)
            // Run this AFTER device lowering so it sees the fixed kernel signatures (AS 0)
            mlir::GpuToLLVMConversionPassOptions hostOptions;
            hostOptions.kernelBarePtrCallConv = true; // Use bare pointers to avoid descriptor expansion
            pm.addPass(mlir::createGpuToLLVMConversionPass(hostOptions));
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // Binary generation (Stage 2)
            mlir::GpuModuleToBinaryPassOptions binaryOptions;
            binaryOptions.toolkitPath = "/usr/local/cuda-13.0";
            binaryOptions.compilationTarget = "isa"; 
            pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));

            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createSCFToControlFlowPass());
            pm.addPass(mlir::createConvertControlFlowToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            
            pm.addPass(mlir::memref::createExpandStridedMetadataPass());
            pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
            pm.addPass(mlir::nova::createGpuRuntimeLoweringPass());
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::nova::createGenerateDynamicWrapperPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

        }
        void registerNovaGPUPipelines()
        {
            PassPipelineRegistration<>("nova-gpu-pipeline",
                                       "Nova GPU Pipeline",
                                       createNovaGPUPipelines);
        }
    } // namespace nova
} // namespace mlir