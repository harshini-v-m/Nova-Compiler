#include "Compiler/API/NovaCompilerAPI.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Pipeline/Gpupipeline.h"
#include "Compiler/Pipeline/Pipeline.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
// ValueBoundsOpInterface implementations (needed for tiling transforms)
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Target/LLVM/NVVM/Target.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>

using namespace mlir;
using namespace mlir::nova;

namespace {
struct AllReduceOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          AllReduceOpMemEffectModel, mlir::gpu::AllReduceOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // No memory effects (operating on values)
  }
};
} // namespace

struct DynamicSharedMemoryOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          DynamicSharedMemoryOpMemEffectModel, mlir::gpu::DynamicSharedMemoryOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // No memory side effects
  }
};

struct BarrierOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          BarrierOpMemEffectModel, mlir::gpu::BarrierOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // Barrier has effects on execution order, but here we mean memory side effects
    // that might block outlining. Barrier doesn't read/write memory itself.
  }
};

struct DeviceAsyncCreateGroupOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          DeviceAsyncCreateGroupOpMemEffectModel, mlir::nvgpu::DeviceAsyncCreateGroupOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // No direct memory effects (commits async groups)
  }
};

struct DeviceAsyncWaitOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          DeviceAsyncWaitOpMemEffectModel, mlir::nvgpu::DeviceAsyncWaitOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // No direct memory effects (waits for async groups)
  }
};

struct DeviceAsyncCopyOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          DeviceAsyncCopyOpMemEffectModel, mlir::nvgpu::DeviceAsyncCopyOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // Portably report Read and Write effects without binding to a specific Value
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
  }
};

struct CpAsyncCommitGroupOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          CpAsyncCommitGroupOpMemEffectModel, mlir::NVVM::CpAsyncCommitGroupOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // Declare Write so the canonicalizer/DCE does not remove this op.
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
  }
};

struct CpAsyncWaitGroupOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          CpAsyncWaitGroupOpMemEffectModel, mlir::NVVM::CpAsyncWaitGroupOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // Declare Write so the canonicalizer/DCE does not remove this op.
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
  }
};

NovaCompilerAPI::NovaCompilerAPI() {
  // Register all MLIR passes globally so they can be parsed from strings
  // Register specific passes needed for parsed pipelines
  mlir::affine::registerAffinePasses();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  context = std::make_unique<MLIRContext>();

  // Create and populate dialect registry first
  DialectRegistry registry;

  // Register necessary dialects
  registry
      .insert<mlir::nova::NovaDialect, mlir::func::FuncDialect,
              mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
              mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
              mlir::tosa::TosaDialect, mlir::memref::MemRefDialect,
              mlir::vector::VectorDialect,
              mlir::bufferization::BufferizationDialect, mlir::gpu::GPUDialect,
              mlir::NVVM::NVVMDialect, mlir::nvgpu::NVGPUDialect>();

  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register BufferDeallocationOpInterface external models
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);

  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertOpenMPToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  // Register LLVM IR translation
  registerLLVMDialectTranslation(registry);
  registerAllToLLVMIRTranslations(registry);

  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

NovaCompilerAPI::~NovaCompilerAPI() = default;

std::unique_ptr<llvm::Module>
NovaCompilerAPI::compileFile(const std::string &inputFile,
                             llvm::LLVMContext &llvmContext,
                             const CompilerOptions &options) {
  // Parse the input file
  auto fileOrErr = llvm::MemoryBuffer::getFile(inputFile);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Failed to open file: " << ec.message() << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, context.get());
  if (!module) {
    llvm::errs() << "Failed to parse MLIR file\n";
    return nullptr;
  }

  return compileToLLVMModule(*module, llvmContext, options);
}

//---------------------------------------------------------------------------------------------------
// IMPORTANT
//---------------------------------------------------------------------------------------------------

void NovaCompilerAPI::registerAllDialects(DialectRegistry &registry) {
  // Register necessary dialects
  registry.insert<
      mlir::nova::NovaDialect, mlir::func::FuncDialect,
      mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::math::MathDialect, mlir::tensor::TensorDialect,
      mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
      mlir::tosa::TosaDialect, mlir::memref::MemRefDialect,
      mlir::vector::VectorDialect, mlir::bufferization::BufferizationDialect,
      mlir::gpu::GPUDialect, mlir::NVVM::NVVMDialect, mlir::nvgpu::NVGPUDialect,
      mlir::LLVM::LLVMDialect, mlir::transform::TransformDialect>();

  // Register Transform Dialect extensions
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::memref::registerTransformDialectExtension(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::func::registerInlinerExtension(registry);

  // Register Tiling Interface external models
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);

  // Register ValueBoundsOpInterface external models (required for tiling)
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::linalg::registerValueBoundsOpInterfaceExternalModels(registry);

  // Register external models and conversion interfaces
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  // Register BufferDeallocationOpInterface external models
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);

  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertOpenMPToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);

  registerLLVMDialectTranslation(registry);
  registerAllToLLVMIRTranslations(registry);
    // Attach the interface to gpu::AllReduceOp when GPU dialect is loaded
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            mlir::gpu::GPUDialect *dialect) {
    mlir::gpu::AllReduceOp::attachInterface<AllReduceOpMemEffectModel>(*ctx);
    mlir::gpu::DynamicSharedMemoryOp::attachInterface<DynamicSharedMemoryOpMemEffectModel>(*ctx);
    mlir::gpu::BarrierOp::attachInterface<BarrierOpMemEffectModel>(*ctx);
  });

  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            mlir::nvgpu::NVGPUDialect *dialect) {
    mlir::nvgpu::DeviceAsyncCreateGroupOp::attachInterface<
        DeviceAsyncCreateGroupOpMemEffectModel>(*ctx);
    mlir::nvgpu::DeviceAsyncWaitOp::attachInterface<
        DeviceAsyncWaitOpMemEffectModel>(*ctx);
    mlir::nvgpu::DeviceAsyncCopyOp::attachInterface<
        DeviceAsyncCopyOpMemEffectModel>(*ctx);
  });

  // NVVM cp.async ops used directly for synchronisation (bypasses nvgpu tokens)
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            mlir::NVVM::NVVMDialect *dialect) {
    mlir::NVVM::CpAsyncCommitGroupOp::attachInterface<
        CpAsyncCommitGroupOpMemEffectModel>(*ctx);
    mlir::NVVM::CpAsyncWaitGroupOp::attachInterface<
        CpAsyncWaitGroupOpMemEffectModel>(*ctx);
  });
}

std::unique_ptr<llvm::Module>
NovaCompilerAPI::compileToLLVMModule(ModuleOp module,
                                     llvm::LLVMContext &llvmContext,
                                     const CompilerOptions &options) {

  // Register required dialect interfaces on the module's context
  // This is necessary because the module may have been created by a different
  // context
  mlir::MLIRContext *ctx = module.getContext();
  DialectRegistry registry;
  registerAllDialects(registry);

  // Re-register side-effect interfaces needed by buffer deallocation
  registry.addExtension(+[](mlir::MLIRContext *c,
                            mlir::gpu::GPUDialect *) {
    mlir::gpu::AllReduceOp::attachInterface<AllReduceOpMemEffectModel>(*c);
    mlir::gpu::DynamicSharedMemoryOp::attachInterface<DynamicSharedMemoryOpMemEffectModel>(*c);
    mlir::gpu::BarrierOp::attachInterface<BarrierOpMemEffectModel>(*c);
  });
  registry.addExtension(+[](mlir::MLIRContext *c,
                            mlir::nvgpu::NVGPUDialect *) {
    mlir::nvgpu::DeviceAsyncCreateGroupOp::attachInterface<
        DeviceAsyncCreateGroupOpMemEffectModel>(*c);
    mlir::nvgpu::DeviceAsyncWaitOp::attachInterface<
        DeviceAsyncWaitOpMemEffectModel>(*c);
    mlir::nvgpu::DeviceAsyncCopyOp::attachInterface<
        DeviceAsyncCopyOpMemEffectModel>(*c);
  });
  registry.addExtension(+[](mlir::MLIRContext *c,
                            mlir::NVVM::NVVMDialect *) {
    mlir::NVVM::CpAsyncCommitGroupOp::attachInterface<
        CpAsyncCommitGroupOpMemEffectModel>(*c);
    mlir::NVVM::CpAsyncWaitGroupOp::attachInterface<
        CpAsyncWaitGroupOpMemEffectModel>(*c);
  });

  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  // Extensions only fire when a dialect is *first* loaded. If the NVVM/NVGPU/GPU
  // dialects were loaded before appendDialectRegistry, the callbacks never ran.
  // Attach interfaces directly to already-loaded dialects as a safety net.
  mlir::NVVM::CpAsyncCommitGroupOp::attachInterface<CpAsyncCommitGroupOpMemEffectModel>(*ctx);
  mlir::NVVM::CpAsyncWaitGroupOp::attachInterface<CpAsyncWaitGroupOpMemEffectModel>(*ctx);
  mlir::nvgpu::DeviceAsyncCreateGroupOp::attachInterface<DeviceAsyncCreateGroupOpMemEffectModel>(*ctx);
  mlir::nvgpu::DeviceAsyncWaitOp::attachInterface<DeviceAsyncWaitOpMemEffectModel>(*ctx);
  mlir::nvgpu::DeviceAsyncCopyOp::attachInterface<DeviceAsyncCopyOpMemEffectModel>(*ctx);
  mlir::gpu::AllReduceOp::attachInterface<AllReduceOpMemEffectModel>(*ctx);
  mlir::gpu::DynamicSharedMemoryOp::attachInterface<DynamicSharedMemoryOpMemEffectModel>(*ctx);
  mlir::gpu::BarrierOp::attachInterface<BarrierOpMemEffectModel>(*ctx);

  if (failed(verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return nullptr;
  }

  if (failed(runPipeline(module, options))) {
    llvm::errs() << "Pipeline execution failed\n";
    return nullptr;
  }

  auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
  return llvmModule;
}

LogicalResult NovaCompilerAPI::runPipeline(ModuleOp module,
                                           const CompilerOptions &options) {
  // Create a PassManager that operates on ModuleOp
  // IMPORTANT: Use the module's own context, not our internal context
  PassManager pm(module.getContext());

  // Enable verbose mode to trace pipeline execution
  if (options.verbose) {
    pm.enableIRPrinting([](Pass *, Operation *) { return false; }, // before
                        [](Pass *, Operation *) { return true; },  // after
                        true,  // print module scope
                        false, // print after only on change
                        false, // print after only on failure
                        llvm::errs());
  }

  if (options.runFullPipeline) {
    // Add the Nova optimization pipeline based on target device
    if (options.device == "gpu" || options.device == "cuda") {
      createNovaGPUPipelines(pm);
    } else {
      createNovaPipelines(pm);
    }
  }

  // Run custom pipeline if specified
  if (!options.customPipeline.empty()) {
    if (failed(parsePassPipeline(options.customPipeline, pm))) {
      return failure();
    }
  }

  if (failed(pm.run(module))) {
    llvm::errs() << "Pipeline failed. Dumping IR to failed_module.mlir\n";
    std::error_code ec;
    llvm::raw_fd_ostream dest("failed_module.mlir", ec, llvm::sys::fs::OF_Text);
    if (!ec) {
      module.print(dest);
      dest.flush();
    }
    return failure();
  }
  return success();
}
//----------------------------------------------------------------------------//
// NovaCompilerSystemAPI Implementation
//----------------------------------------------------------------------------//

std::string NovaCompilerSystemAPI::executeCommand(const std::string &command) {
  std::array<char, 128> buffer;
  std::string result;

  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return "";
  }

  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }

  pclose(pipe);
  return result;
}

std::string NovaCompilerSystemAPI::findNovaOpt(const std::string &hint) {
  if (!hint.empty()) {
    return hint;
  }

  // Try common locations
  std::vector<std::string> paths = {
      "./build/tools/nova-opt/nova-opt", "../build/tools/nova-opt/nova-opt",
      "nova-opt" // In PATH
  };

  for (const auto &path : paths) {
    std::string cmd = "which " + path + " 2>/dev/null";
    std::string result = executeCommand(cmd);
    if (!result.empty()) {
      // Remove trailing newline
      if (!result.empty() && result.back() == '\n') {
        result.pop_back();
      }
      return result;
    }
  }

  return "nova-opt"; // Fallback to PATH
}

std::string NovaCompilerSystemAPI::getLLVMIR(const std::string &inputFile,
                                             const std::string &novaOptPath,
                                             const std::string &device) {
  std::string novaOpt = findNovaOpt(novaOptPath);

  // Build command based on device
  std::string pipeline =
      (device == "gpu") ? "--nova-gpu-pipeline" : "--nova-opt-pipeline";
  std::string cmd = novaOpt + " " + inputFile + " " + pipeline + " | " +
                    "mlir-translate --mlir-to-llvmir";

  return executeCommand(cmd);
}