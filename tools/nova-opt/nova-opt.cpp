#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/ViewOpGraph.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Transforms/FuseMatmulBias.h"
#include "Compiler/Transforms/Passes.h"

#include "Compiler/Pipeline/Gpupipeline.h"
#include "Compiler/Pipeline/Pipeline.h"
#include "Compiler/Transforms/Affine/DependencyAnalysisTestPass.h"
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

namespace mlir {
namespace nova {}
} // namespace mlir

#include "Compiler/Transforms/AddGpuMemoryCopies.h"

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

struct BarrierOpMemEffectModel
    : public mlir::MemoryEffectOpInterface::ExternalModel<
          BarrierOpMemEffectModel, mlir::gpu::BarrierOp> {
  void getEffects(mlir::Operation *op,
                  llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) const {
    // No memory effects (synchronization only)
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
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
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
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
  }
};
} // namespace

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  // Attach the interface to gpu::AllReduceOp and gpu::BarrierOp when GPU
  // dialect is loaded
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            mlir::gpu::GPUDialect *dialect) {
    mlir::gpu::AllReduceOp::attachInterface<AllReduceOpMemEffectModel>(*ctx);
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

  // Register the AddGpuMemoryCopies pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::nova::createAddGpuMemoryCopiesPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::nova::createNovaToGpuPass();
  });

  // Register the ViewOpGraph pass specifically
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createPrintOpGraphPass();
  });

  // Register only the dialects we need
  registry.insert<mlir::nova::NovaDialect>();
  registry.insert<mlir::transform::TransformDialect>();
  mlir::registerAllDialects(registry);

  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::memref::registerTransformDialectExtension(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::func::registerInlinerExtension(registry);
  // Register LLVM IR translation
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::nova::registerNovaPipelines();
  mlir::nova::registerNovaGPUPipelines();
  mlir::nova::registerAffinePasses();

  mlir::nova::registerNovaToArithLoweringPass();
  mlir::nova::registerNovaToTosaLoweringPass();
  mlir::nova::registerNovaElementwiseToLinalgPass();
  mlir::nova::registerNovaToLinalgPass();
  mlir::nova::registerNovaLinalgHorizontalFusionPass();
  mlir::nova::registerNovaFusionKernelEmitterPass();
  mlir::registerDependencyAnalysisTestPass();

  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::NVVM::registerConvertGpuToNVVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Nova dialect optimizer\n", registry));
}
