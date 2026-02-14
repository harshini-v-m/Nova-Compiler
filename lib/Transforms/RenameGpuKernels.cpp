#include "Compiler/Transforms/RenameGpuKernels.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace nova {

namespace {

struct RenameGpuKernelsPass
    : public PassWrapper<RenameGpuKernelsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RenameGpuKernelsPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);

    module.walk([&](gpu::GPUModuleOp gpuModule) {
      StringRef moduleName = gpuModule.getName();
      SymbolTable gpuModuleSymbolTable(gpuModule);

      gpuModule.walk([&](gpu::GPUFuncOp gpuFunc) {
        if (!gpuFunc.isKernel())
          return;

        StringRef oldName = gpuFunc.getName();
        // Rename the kernel to match the module name to ensure uniqueness
        // in the generated binary output (PTX).
        // If the module name is "main_kernel_0", the kernel becomes "main_kernel_0".
        StringRef newName = moduleName;

        if (oldName == newName)
          return;

        if (failed(gpuModuleSymbolTable.rename(gpuFunc, newName))) {
            return; 
        }

        // Fix up gpu.launch_func in the main module
        module.walk([&](gpu::LaunchFuncOp launchOp) {
          if (launchOp.getKernelModuleName().getValue() == moduleName &&
              launchOp.getKernelName().getValue() == oldName) {
            // Update the kernel name attribute
            auto ctx = launchOp.getContext();
            auto rootName = StringAttr::get(ctx, moduleName);
            auto nestedName = FlatSymbolRefAttr::get(ctx, newName);
            auto newSymbolRef = SymbolRefAttr::get(ctx, rootName, {nestedName});
            launchOp->setAttr("kernel", newSymbolRef);
          }
        });
      });
    });
  }

  StringRef getArgument() const final { return "rename-gpu-kernels"; }
  StringRef getDescription() const final {
    return "Rename GPU kernels to match their module name for unique profiling symbols";
  }
};

} // namespace

std::unique_ptr<Pass> createRenameGpuKernelsPass() {
  return std::make_unique<RenameGpuKernelsPass>();
}

} // namespace nova
} // namespace mlir
