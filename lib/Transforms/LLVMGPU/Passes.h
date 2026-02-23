#ifndef NOVA_TRANSFORMS_LLVMGPU_PASSES_H_
#define NOVA_TRANSFORMS_LLVMGPU_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace nova {

// --- Pipeline ---
// Adds the Nova GPU optimized pipeline (strategy → tile → pad → promote → K-tile →...)
// |cudaArch|: CUDA SM arch string forwarded to the strategy pass, e.g. "sm_80".
void addNovaGPUOptimizedPipeline(OpPassManager &pm,
                                  StringRef cudaArch = "sm_80");

// --- Passes ---

// Runs the standard canonicalizer but propagates `lowering_config` attrs
// to replacement ops when the original op is folded/replaced.
// Mirrors IREE's ConfigTrackingCanonicalizerPass.
// Must be used instead of createCanonicalizerPass() at all pipeline stages
// where tiled ops may lose their config during rewriting.
std::unique_ptr<Pass> createNovaConfigTrackingCanonicalizerPass();
void registerNovaConfigTrackingCanonicalizerPass();

// Selects tile sizes and MMA intrinsic based on NVIDIA target info, then
// attaches a #nova.lowering_config dict attribute to each linalg matmul op.
// Mirrors IREE's LLVMGPUSelectLoweringStrategy pass.
// |cudaArch|: SM architecture string, e.g. "sm_80", "sm_75", "ampere".
std::unique_ptr<Pass> createNovaGPUSelectLoweringStrategyPass(
    StringRef cudaArch = "sm_80");
void registerNovaGPUSelectLoweringStrategyPass();

// Tiles compute operations and distributes them to workgroups using scf.forall
std::unique_ptr<Pass> createNovaTileAndDistributeToWorkgroupsPass();
void registerNovaTileAndDistributePass();

// Pads linalg operands to static multiples of tile sizes.
// Ported from IREE's GPUPadOperands.cpp.
// TODO: When LoweringConfig is available, padding sizes will be read from the
// config attribute instead of using heuristics.
std::unique_ptr<Pass> createNovaGPUPadOperandsPass();
void registerNovaGPUPadOperandsPass();

// Promotes matmul A/B operands to GPU shared memory (workgroup address space).
// Ported/adapted from IREE's GPUPromoteMatmulOperands.cpp.
// TODO: When LoweringConfig is available, read which operands to promote from
// the config attribute instead of using heuristics.
std::unique_ptr<Pass> createNovaGPUPromoteMatmulOperandsPass();
void registerNovaGPUPromoteMatmulOperandsPass();

// Tiles reduction (K) dimensions of matmul inside workgroup scf.forall into
// an inner scf.for loop. Fuses A/B producer copies into the loop.
// Mirrors IREE's GPUApplyTilingLevel(Reduction).
std::unique_ptr<Pass> createNovaGPUApplyTilingLevelReductionPass();
void registerNovaGPUApplyTilingLevelReductionPass();

// Tiles parallel (M/N) dimensions of matmul to per-thread register tiles.
// Mirrors IREE's GPUApplyTilingLevel(Thread).
std::unique_ptr<Pass> createNovaGPUApplyTilingLevelThreadPass();
void registerNovaGPUApplyTilingLevelThreadPass();

// Tiles parallel (M/N) dimensions of matmul to per-subgroup (warp) tiles.
// Mirrors IREE's GPUApplyTilingLevel(Subgroup).
std::unique_ptr<Pass> createNovaGPUApplyTilingLevelSubgroupPass();
void registerNovaGPUApplyTilingLevelSubgroupPass();

} // namespace nova
} // namespace mlir

#endif // NOVA_TRANSFORMS_LLVMGPU_PASSES_H_
