//convert to llvm includes
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
//other conversion includes from mlir
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
//utils
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
//buffer includes
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
// Affine optimization passes
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"


//nova dialect includes
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

//optimization passes includes
#include "Compiler/Transforms/FuseMatmulBias.h"
#include "Compiler/Transforms/GenerateDynamicWrapper.h"
#include "Compiler/Transforms/VectorOpt.h"

//lowering passes
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

// header of this file
#include "Compiler/Pipeline/Pipeline.h"
#include "Compiler/Transforms/Affine/DependencyAnalysisTestPass.h"

using namespace mlir;

namespace mlir {
namespace nova {}
} // namespace mlir::nova

void mlir::nova::createNovaPipelines(OpPassManager &pm) {
  // pm.addPass(createCanonicalizerPass());
  // pm.addNestedPass<func::FuncOp>(std::make_unique<DependencyAnalysisTestPass>());
  
  // Lower Nova dialect to standard dialects 
//pm.addNestedPass<func::FuncOp>(ceateNovaToLinalg());  
  pm.addPass(createCanonicalizerPass()); 
  pm.addPass(createNovaToTosaLoweringPass());
  pm.addNestedPass<func::FuncOp>(createNovaToLinalgLoweringPass());
  pm.addPass(createNovaToArithLoweringPass());
  pm.addPass(createCanonicalizerPass());
//  pm.addPass(mlir::createTensorBufferizePass());  
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(mlir::createTosaToTensorPass());
  pm.addPass(mlir::createTosaToArithPass());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalgNamed()); 
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalg());  
  pm.addPass(mlir::createTosaToTensorPass());
  pm.addPass(mlir::createCanonicalizerPass());


  // Convert elementwise operations to Linalg
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  
  // TILE NAMED OPS (linalg.matmul, linalg.add) - BEFORE generalization
  // CPU optimized for Intel i7-14700K cache hierarchy
  std::string transformFileName = std::string(NOVA_SOURCE_DIR) + "/include/Compiler/Transforms/Tiling/tiling_multilevel.mlir";
  mlir::transform::PreloadLibraryPassOptions preloadOptions;
  preloadOptions.transformLibraryPaths = {transformFileName};
 //pm.addPass(mlir::transform::createPreloadLibraryPass(preloadOptions));

  mlir::transform::InterpreterPassOptions interpOptions;
  //pm.addPass(mlir::transform::createInterpreterPass(interpOptions));
//  pm.addPass(createCanonicalizerPass());

  // GENERALIZE NAMED OPS (after tiling, for vectorization)
  pm.addNestedPass<mlir::func::FuncOp>(
                mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createCanonicalizerPass());

  // FUSE ELEMENTWISE (operates on generalized ops)
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  //  Bufferization (Tensor -> MemRef) - After tiling
  bufferization::OneShotBufferizePassOptions bufferizeOptions; 
  bufferizeOptions.bufferizeFunctionBoundaries = true; 
  bufferizeOptions.functionBoundaryTypeConversion= 
                bufferization::LayoutMapOption::IdentityLayoutMap; 

  // so when we bufferize a function that takes a tensor as input, it will be converted to a memref with an identity layout map
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));
  
  //  Buffer deallocation pipeline - this will make sure that all allocated buffers are deallocated properly based on the analysis done during bufferization
  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);
  
  //Convert remaining bufferization ops to memref
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());

  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

  OpPassManager &funcPM = pm.nest<func::FuncOp>();


  if (failed(parsePassPipeline("func.func(affine-parallelize{max-nested=2})", pm))) {}
  
  // Unroll and Jam (factor=2)
  funcPM.addPass(mlir::affine::createLoopUnrollAndJamPass(2));

  // Loop Unroll (factor=8)
  funcPM.addPass(mlir::affine::createLoopUnrollPass(8));

  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(mlir::createCSEPass());
  funcPM.addPass(mlir::math::createMathUpliftToFMA());  

  /*
   mlir::affine::AffineVectorizeOptions vectorOptions;
   vectorOptions.vectorSizes = {8};
   funcPM.addPass(mlir::affine::createAffineVectorize(vectorOptions));
 
  funcPM.addPass(mlir::nova::createVectorOptPass());
  */
  funcPM.addPass(mlir::createCanonicalizerPass());

  // Lower affine to standard control flow
  pm.addPass(createLowerAffinePass());
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(createLowerAffinePass());

  // Lower Linalg to loops (SCF dialect)
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  
  // Convert SCF to CF (Control Flow)
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(createCanonicalizerPass()); 
  
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());

  //Lower to LLVM dialect
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass()); // Convert functions lastly
  
  // Generate dynamic wrapper for unlimited args (after C interface is created)
  pm.addPass(mlir::nova::createGenerateDynamicWrapperPass());
  
  pm.addPass(mlir::createUBToLLVMConversionPass());
  // reconcile unrealized casts
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  
  // Final canonicalization
  //pm.addPass(createCanonicalizerPass());
}

void mlir::nova::registerNovaPipelines() {
  PassPipelineRegistration<>("nova-opt-pipeline",
                            "Nova Optimizer Pipeline",
                            createNovaPipelines);
}