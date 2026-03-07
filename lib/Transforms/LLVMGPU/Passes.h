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

// Greedily fuses and hoists parallel scf.forall loops after thread/subgroup
// tiling. Runs 3 rounds of patterns (forall fusion, consumer fusion, producer
// fusion) to produce a single flat thread-mapped forall per compute region.
// Mirrors IREE's GPUFuseAndHoistParallelLoops pass.
std::unique_ptr<Pass> createNovaGPUFuseAndHoistParallelLoopsPass();
void registerNovaGPUFuseAndHoistParallelLoopsPass();

// Erases all nova.fusion_barrier ops by replacing them with their source
// operand. Must run as the first step of addNovaGPUBufferizePasses.
// Mirrors what IREE's IREEComprehensiveBufferizePass does at its start.
std::unique_ptr<Pass> createNovaGPUEraseFusionBarriersPass();
void registerNovaGPUEraseFusionBarriersPass();

// Infers GPU memory spaces for `bufferization.alloc_tensor` ops.
// Any alloc used as the shared_outs init of a thread-mapped scf.forall is
// tagged as workgroup (shared) memory; all others become private memory.
// Must run immediately before bufferization.
// Mirrors IREE's GPUInferMemorySpacePass.
std::unique_ptr<Pass> createNovaGPUInferMemorySpacePass();
void registerNovaGPUInferMemorySpacePass();

// Eliminates tensor.empty ops by finding existing destination tensors.
// Runs linalg convert-to-DPS patterns then bufferization empty tensor
// elimination analysis. Reduces unnecessary allocations.
// Mirrors IREE's EliminateEmptyTensorsPass.
std::unique_ptr<Pass> createNovaEliminateEmptyTensorsPass();
void registerNovaEliminateEmptyTensorsPass();

// GPU-aware bufferization helper.
// Erases nova.fusion_barrier ops, infers memory spaces, then runs
// OneShotBufferize with GPU alloc / memcpy functions (workgroup → memref.alloc,
// private → memref.alloca; barriers inserted around workgroup copies).
// Mirrors IREE's addGPUBufferizePasses().
void addNovaGPUBufferizePasses(OpPassManager &pm);

// GPU comprehensive bufferize pass.
// Erases nova.fusion_barrier (GAP 2) and runs OneShotBufferize with
// GPU-aware alloc/copy fns (GAP 3). Called by addNovaGPUBufferizePasses.
// Mirrors IREE's IREEComprehensiveBufferizePass.
std::unique_ptr<Pass> createNovaGPUComprehensiveBufferizePass();
void registerNovaGPUComprehensiveBufferizePass();

// Inserts gpu.barrier before/after memref.copy ops involving workgroup memory.
// Must run AFTER bufferization (gpu.barrier breaks OneShotBufferize analysis).
std::unique_ptr<Pass> createNovaGPUInsertWorkgroupBarriersPass();
void registerNovaGPUInsertWorkgroupBarriersPass();

// Normalizes scf.forall loop bounds to lb=0, step=1.
// Inserts affine.apply ops to compute denormalized induction variable values.
// Must run before GPU distribution (map_forall_to_blocks requires normalized foralls).
// Mirrors IREE's NormalizeLoopBoundsPass.
std::unique_ptr<Pass> createNovaNormalizeLoopBoundsPass();
void registerNovaNormalizeLoopBoundsPass();

// Converts workgroup memref.alloc → memref.global + memref.get_global and
// erases workgroup memref.dealloc. GPU shared memory must be statically
// declared at module level (not dynamically allocated via malloc).
// Ported from IREE's ConvertSharedMemAllocOp + DropSharedMemoryDeallocOp.
std::unique_ptr<Pass> createNovaConvertSharedMemAllocsPass();
void registerNovaConvertSharedMemAllocsPass();
// Wrapper around GpuMapParallelLoops and ConvertParallelLoopToGpu that skips
// existing gpu.launch blocks to prevent illegal nested launches.
std::unique_ptr<Pass> createNovaGpuMapParallelLoopPass();
void registerNovaGpuMapParallelLoopPass();


} // namespace nova
} // namespace mlir

#endif // NOVA_TRANSFORMS_LLVMGPU_PASSES_H_
