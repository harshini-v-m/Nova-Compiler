//===- NovaGPUSelectLoweringStrategy.cpp - strategy pass ------------------===//
//
// Mirrors IREE's LLVMGPUSelectLoweringStrategy pass.
// Reads the target CUDA arch from a pass option (default: "sm_80") and
// calls initNovaGPULaunchConfig() to attach LoweringConfig attrs to all
// linalg contraction ops inside the function.
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaKernelConfig.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-select-lowering-strategy"

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NovaGPUSelectLoweringStrategyPass
    : public PassWrapper<NovaGPUSelectLoweringStrategyPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUSelectLoweringStrategyPass)

  NovaGPUSelectLoweringStrategyPass() = default;
  NovaGPUSelectLoweringStrategyPass(const NovaGPUSelectLoweringStrategyPass &pass)
      : PassWrapper(pass) {}
  NovaGPUSelectLoweringStrategyPass(StringRef arch) {
    cudaArch = arch.str();
  }

  // Pass option: which CUDA arch to target (e.g. "sm_80", "sm_75", "ampere").
  Option<std::string> cudaArch{*this, "cuda-arch",
      llvm::cl::desc("CUDA SM architecture (e.g. sm_80, sm_75, volta)."),
      llvm::cl::init("sm_80")};

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    NVIDIATargetInfo target = getNVIDIATargetInfo(cudaArch);

    if (!target.isValid()) {
      funcOp.emitWarning()
          << "[nova-gpu-select-lowering-strategy] Unrecognized CUDA arch '"
          << cudaArch
          << "'; falling back to sm_80 defaults.";
      target = getNVIDIATargetInfo("sm_80");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-gpu-select-lowering-strategy] Using target: "
               << target.archName << " (smCount=" << target.smCount
               << " maxShmem=" << target.maxWorkgroupMemBytes
               << " MMA intrinsics=" << target.mmaIntrinsics.size() << ")\n");

    initNovaGPULaunchConfig(funcOp, target);
  }

  StringRef getArgument() const override {
    return "nova-gpu-select-lowering-strategy";
  }
  StringRef getDescription() const override {
    return "Select and attach GPU lowering configs for Nova kernel pipeline";
  }
};

//===----------------------------------------------------------------------===//
// Public factory functions
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
createNovaGPUSelectLoweringStrategyPass(StringRef cudaArch) {
  return std::make_unique<NovaGPUSelectLoweringStrategyPass>(cudaArch);
}

void registerNovaGPUSelectLoweringStrategyPass() {
  PassRegistration<NovaGPUSelectLoweringStrategyPass>();
}

} // namespace mlir::nova
