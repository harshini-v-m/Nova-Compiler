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

// gpu
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
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

            // 4. NOVA TRANSFORMS & LINALG OPT
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createFuseMatmulBiasPass());
            pm.addPass(mlir::createCanonicalizerPass());
            // Tiling is handled in Section 6 after parallel loop conversion
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::createLinalgElementwiseOpFusionPass());
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::createLinalgGeneralizeNamedOpsPass());

            // 5. BUFFERIZATION & DEALLOCATION
            bufferization::OneShotBufferizePassOptions bufferizeOptions;
            bufferizeOptions.bufferizeFunctionBoundaries = true;
            bufferizeOptions.functionBoundaryTypeConversion = bufferization::LayoutMapOption::IdentityLayoutMap;
            bufferizeOptions.useEncodingForMemorySpace = true;
            bufferizeOptions.allowUnknownOps = true;
            pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

            bufferization::BufferDeallocationPipelineOptions deallocationOptions;
            bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);
            pm.addPass(mlir::createConvertBufferizationToMemRefPass());
            // device attribute handling pass
            pm.addPass(mlir::nova::createConvertMemRefToGpuPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());

            // 6. LINALG OPTIMIZATION & TILING
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgFoldUnitExtentDimsPass());
            pm.addPass(mlir::createCanonicalizerPass());

            // Map Linalg to Parallel Loops
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToParallelLoopsPass());
            // Apply Tiling HERE on the parallel loops
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopTilingPass({32, 32, 1}));
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopFusionPass());
            pm.addPass(mlir::createCanonicalizerPass());
            // 8. GPU MAPPINGcreateParallelLoopFusionPass
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertParallelLoopToGpuPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // Add custom memory management pass BEFORE outlining to capture gpu.launch
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createAddGpuMemoryCopiesPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::nova::createConvertMemRefToGpuPass());
            pm.addPass(mlir::createGpuKernelOutliningPass());

            mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
            nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
            nvvmTargetOptions.chip = "sm_86";
            pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

            // Lowering INSIDE the GPU Module (Fixes 'index' in kernels)
            auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
            gpuPm.addPass(mlir::createLowerAffinePass());
            gpuPm.addPass(mlir::createSCFToControlFlowPass());
            mlir::ConvertGpuOpsToNVVMOpsOptions nvvmOptions;
            // nvvmOptions.useBarePtrCallConv = true; // Disabled to match dynamic wrapper
            gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvmOptions));
            gpuPm.addPass(mlir::createConvertIndexToLLVMPass());
            gpuPm.addPass(mlir::createArithToLLVMConversionPass());
            gpuPm.addPass(mlir::createConvertMathToLLVMPass());
            gpuPm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // Binary generation (Stage 2)
            mlir::GpuModuleToBinaryPassOptions binaryOptions;
            binaryOptions.toolkitPath = "/usr/local/cuda-13.0";
            binaryOptions.compilationTarget = "isa"; 
            pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));

            // MAIN LOWERING: gpu.launch_func -> runtime calls
            mlir::GpuToLLVMConversionPassOptions hostOptions;
            // hostOptions.kernelBarePtrCallConv = true; // Disabled to match dynamic wrapper
            pm.addPass(mlir::createGpuToLLVMConversionPass(hostOptions));
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