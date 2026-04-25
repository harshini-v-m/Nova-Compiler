//===- Passes.h - Nova GPU LLVM-GPU Transform Pass Declarations -----------===//
//
// Public declarations for every individual pass and for the two pipeline
// builder functions in the Nova GPU LLVM-GPU transform library.
//
// Consumers only need to include this header; implementation details and
// MLIR PassWrapper boilerplate live in the corresponding .cpp files.
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_PASSES_H_
#define NOVA_TRANSFORMS_LLVMGPU_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace nova {

//===----------------------------------------------------------------------===//
// Pipeline builders
//===----------------------------------------------------------------------===//

// CORE LOGIC — addNovaGPUOptimizedPipeline
// Adds the entire Nova GPU optimized pipeline to `pm`:
//   Nova dialect lowering → TOSA → Linalg → tiling & fusion →
//   GPU-aware bufferization → gpu.launch → NVVM → PTX binary.
// |cudaArch|: CUDA SM arch string forwarded to the strategy pass, e.g. "sm_86".
// Defaults to "sm_86" (Ampere / RTX 3060) when empty.
void addNovaGPUOptimizedPipeline(OpPassManager &pm,
                                  StringRef cudaArch = "sm_86");

// GPU-aware bufferization helper — mirrors IREE's addGPUBufferizePasses().
// Called internally by addNovaGPUOptimizedPipeline (Step 8).
// Order: EliminateEmptyTensors → AllocTensor → InferMemorySpace →
//        ComprehensiveBufferize → cleanup → BufferLoopHoisting → Deallocation.
void addNovaGPUBufferizePasses(OpPassManager &pm);

//===----------------------------------------------------------------------===//
// Individual pass declarations
//===----------------------------------------------------------------------===//

// Runs the standard canonicalizer but propagates `lowering_config` attrs
// to replacement ops when the original op is folded/replaced.
// Mirrors IREE's ConfigTrackingCanonicalizer.
// IMPORTANT: Must be used instead of createCanonicalizerPass() at all
// pipeline stages where tiled ops may lose their config during rewriting.
std::unique_ptr<Pass> createNovaConfigTrackingCanonicalizerPass();
void registerNovaConfigTrackingCanonicalizerPass();

// Selects tile sizes and MMA intrinsic based on NVIDIA target info, then
// attaches a #nova.lowering_config dict attribute to each linalg matmul op.
// Mirrors IREE's LLVMGPUSelectLoweringStrategy pass.
// |cudaArch|: SM architecture string, e.g. "sm_80", "sm_75", "ampere".
std::unique_ptr<Pass> createNovaGPUSelectLoweringStrategyPass(
    StringRef cudaArch = "sm_86");
void registerNovaGPUSelectLoweringStrategyPass();

// Tiles compute operations and distributes them to workgroups using scf.forall.
std::unique_ptr<Pass> createNovaTileAndDistributeToWorkgroupsPass();
void registerNovaTileAndDistributePass();

// Pads linalg operands to static multiples of tile sizes.
// Ported from IREE's GPUPadOperands.cpp.
// IMPORTANT: Also pads the K (reduction) dimension to the reduction tile
// size — without this a non-aligned K silently truncates the last elements.
std::unique_ptr<Pass> createNovaGPUPadOperandsPass();
void registerNovaGPUPadOperandsPass();

// Promotes matmul A/B operands to GPU shared memory (workgroup address space).
// Uses a two-stage copy pattern (global→shared linalg.copy + nova.fusion_barrier
// + per-thread linalg.copy). Ported from IREE's GPUPromoteMatmulOperands.cpp.
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

// Greedily distributes orphaned parallel ops (not inside thread-mapped
// scf.forall) to GPU threads.  Derives thread tile sizes from loop ranges.
// Must run AFTER FuseAndHoist, BEFORE LowerBarrierRegion.
// Mirrors IREE's GPUGreedilyDistributeToThreads pass.
std::unique_ptr<Pass> createNovaGPUGreedilyDistributeToThreadsPass();
void registerNovaGPUGreedilyDistributeToThreadsPass();

// Lowers nova.barrier_region ops into two nova.value_barrier ops (one for
// writes, one for reads) with the body inlined between them.
// Must run after FuseAndHoistParallelLoops and before bufferization.
// Mirrors IREE's LowerBarrierRegion pattern.
std::unique_ptr<Pass> createNovaGPULowerBarrierRegionPass();
void registerNovaGPULowerBarrierRegionPass();

// Erases all nova.fusion_barrier ops by replacing them with their source
// operand. Must run as the first step of addNovaGPUBufferizePasses.
// Mirrors what IREE's IREEComprehensiveBufferizePass does at its start.
std::unique_ptr<Pass> createNovaGPUEraseFusionBarriersPass();
void registerNovaGPUEraseFusionBarriersPass();

// Infers GPU memory spaces for `bufferization.alloc_tensor` ops.
// Decision: alloc used as shared_outs init of a thread-mapped scf.forall →
// workgroup (shared) memory; all others → private memory.
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

// GPU comprehensive bufferize pass.
// Erases nova.fusion_barrier (step 1) and runs OneShotBufferize with
// GPU-aware alloc/copy functions (step 2). Called by addNovaGPUBufferizePasses.
// Mirrors IREE's IREEComprehensiveBufferizePass.
std::unique_ptr<Pass> createNovaGPUComprehensiveBufferizePass();
void registerNovaGPUComprehensiveBufferizePass();

// Inserts gpu.barrier before/after memref.copy ops involving workgroup memory.
// IMPORTANT: Must run AFTER bufferization — gpu.barrier has "unknown side
// effects" that break OneShotBufferize analysis.
std::unique_ptr<Pass> createNovaGPUInsertWorkgroupBarriersPass();
void registerNovaGPUInsertWorkgroupBarriersPass();

// Normalizes scf.forall loop bounds to lb=0, step=1.
// Inserts affine.apply ops to compute denormalized induction variable values.
// Must run before GPU distribution (map_forall_to_blocks requires normalized
// foralls). Mirrors IREE's NormalizeLoopBoundsPass.
std::unique_ptr<Pass> createNovaNormalizeLoopBoundsPass();
void registerNovaNormalizeLoopBoundsPass();

// Converts workgroup memref.alloc → memref.global + memref.get_global and
// erases workgroup memref.dealloc.
// GPU shared memory must be statically declared at module level (not
// dynamically allocated via malloc).
// Ported from IREE's ConvertSharedMemAllocOp + DropSharedMemoryDeallocOp.
std::unique_ptr<Pass> createNovaConvertSharedMemAllocsPass();
void registerNovaConvertSharedMemAllocsPass();

// Converts block/thread-mapped scf.forall ops to gpu.launch with dynamic
// block dims computed from the actual thread forall iteration bounds.
// Replaces the transform dialect script (gpu_forall_to_launch.mlir).
std::unique_ptr<Pass> createNovaGPUMapForallToGPUPass();
void registerNovaGPUMapForallToGPUPass();

// Converts #gpu.address_space attributes on memref types to integer address
// spaces appropriate for NVVM/LLVM lowering.
// private → generic address space 0 (avoids LLVM LSR ScalarEvolution issues
// with non-default address spaces on NVPTX).
// Must run inside gpu.module BEFORE finalizeMemRefToLLVMConversionPass.
std::unique_ptr<Pass> createNovaGPULowerMemorySpacePass();
void registerNovaGPULowerMemorySpacePass();




std::unique_ptr<Pass> createNovaWarpShuffleReductionPass();
void registerNovaWarpShuffleReductionPass();

// Lowers vector.reduction / vector.multi_reduction ops inside gpu.launch bodies
// to warp butterfly shuffle reductions (all-reduce via XOR butterfly + thread-0
// write). Currently a no-op stub — the Nova vectorizer does not yet emit
// vector.reduction; this pass is the extensibility hook for that future work.
// Shared utilities (buildShuffleReductionTree, emitTypedShuffleXOR, etc.) live
// in NovaVectorReduction.cpp and are also used by NovaWarpShuffleReduction.


std::unique_ptr<Pass> createNovaGPUFillCopyForwardingPass();
void registerNovaGPUFillCopyForwardingPass();

std::unique_ptr<Pass> createNovaGPUCoalesceWorkgroupBuffersPass();
void registerNovaGPUCoalesceWorkgroupBuffersPass();

// Multi-buffers every workgroup-memory `memref.alloc` used inside an `scf.for`
// loop. Rewrites `memref<TxS, #shared>` → `memref<N x TxS, #shared>` and indexes
// uses by `(loopIV mod N)`, enabling a later software-pipelining pass to overlap
// `cp.async` of iteration i+N-1 with compute of iteration i without a RAW
// hazard on the same shared-memory tile.
//
// Must run AFTER bufferization and BEFORE NovaConvertSharedMemAllocs (which
// turns the alloc into memref.global and would block the rewrite).
// Ported from IREE's GPUMultiBuffering.cpp.
std::unique_ptr<Pass> createNovaGPUMultiBufferingPass(unsigned numBuffers = 2);
void registerNovaGPUMultiBufferingPass();

// Rewrites every gmem→smem `vector.transfer_read`/`vector.transfer_write` pair
// into one or more `nvgpu.device_async_copy` ops, terminated by a
// `nvgpu.device_async_create_group` + `nvgpu.device_async_wait` (numGroups
// unset → wait-all). Preserves serial semantics; the downstream pipelining
// pass replaces the wait-all with `wait(depth-1)` to actually overlap.
//
// Must run AFTER NovaGPUMultiBuffering (so the smem subviews carry the
// stage dim) and BEFORE NovaGPUPipelining.
//
// Ported from IREE's `createAsyncGroups` in
// compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.cpp.
std::unique_ptr<Pass> createNovaGPUCreateAsyncCopiesPass();
void registerNovaGPUCreateAsyncCopiesPass();

// Software-pipelines K-loops that contain `nvgpu.device_async_copy`.
// Time-shifts cp.async + deps to stage 0 and the compute to stage `depth-1`,
// rewriting `wait` counts so depth-1 groups stay in flight per iteration.
//
// Wraps upstream `mlir::scf::pipelineForLoop`. Bails (leaves loop intact)
// when the K-loop body contains region-bearing children — in that case the
// cp.async ops still run via `wait(0)`, which is correct but un-overlapped.
//
// Ported from IREE's GPUPipelining.cpp (loadGlobalStage0 strategy).
std::unique_ptr<Pass> createNovaGPUPipeliningPass(unsigned depth = 2);
void registerNovaGPUPipeliningPass();

std::unique_ptr<Pass> createNovaRepositionStorePass();
void registerNovaRepositionStorePass();

std::unique_ptr<Pass> createNovaMultiConsumerFusion();
void registerNovaMultiConsumerFusion();

std::unique_ptr<Pass> createNovaGPUReduceBankConflictsPass(
    StringRef arch = "sm_86");
void registerNovaGPUReduceBankConflictsPass();
std::unique_ptr<Pass> createNovaGPUApplySwizzlePass();
void registerNovaGPUApplySwizzlePass();

// ---------------------------------------------------------------------------
// Vectorization Passes  (Batch 1: declarations; wired into pipeline in Batch 2+)
// ---------------------------------------------------------------------------

/// Repacks linalg.generic contraction ops whose lowering_config carries a
/// non-zero mma_kind into inner tile shapes matching the MMA intrinsic
/// (e.g. {16,16,16} for WMMA_F32, {16,8,16} for MMA_SYNC_F16).
/// Must run BEFORE GenericVectorization (still linalg form).
std::unique_ptr<Pass> createNovaGPUPackToIntrinsicsPass();
void registerNovaGPUPackToIntrinsicsPass();

/// Vectorizes linalg operations: matmul-shaped ops → vector.contract,
/// others → vector.multi_reduction / vector.transfer.
/// Also lowers multi_reduction → inner-reduction → vector.contract.
std::unique_ptr<Pass> createNovaGPUGenericVectorizationPass();
void registerNovaGPUGenericVectorizationPass();

/// Hoists vector.transfer_read/write out of sequential loops (subset hoisting).
std::unique_ptr<Pass> createNovaGPUSubsetHoistingPass();
void registerNovaGPUSubsetHoistingPass();

/// Distributes warp-level vector ops across threads (SIMD-to-SIMT).
/// Lowers vector.contract → outer-product chains.
/// Must run AFTER UnrollToIntrinsics.
std::unique_ptr<Pass> createNovaGPUVectorDistributePass();
void registerNovaGPUVectorDistributePass();

/// Vectorizes memref.copy ops targeting shared memory (128-bit LDS.128).
std::unique_ptr<Pass> createNovaGPUVectorizeMemrefCopyPass();
void registerNovaGPUVectorizeMemrefCopyPass();

/// Unrolls vector.contract to the native MMA shape from lowering_config.
/// Must run AFTER GenericVectorization, BEFORE VectorDistribute.
std::unique_ptr<Pass> createNovaGPUUnrollToIntrinsicsPass();
void registerNovaGPUUnrollToIntrinsicsPass();

/// Lowers linalg.pack / linalg.unpack to tensor ops before bufferization.
/// linalg.pack/unpack have no BufferizableOpInterface so must be lowered
/// BEFORE OneShotBufferize.
std::unique_ptr<Pass> createNovaGPULowerPackOpsPass();
void registerNovaGPULowerPackOpsPass();

/// Aggressive transfer hoisting, sinking and folding patterns.
std::unique_ptr<Pass> createNovaGPUOptimizeVectorTransferPass();
void registerNovaGPUOptimizeVectorTransferPass();

/// Drops unit dimensions from vector types.
std::unique_ptr<Pass> createNovaGPUDropVectorUnitDimsPass();
void registerNovaGPUDropVectorUnitDimsPass();

/// Hoists vector.extract_slice / vector.insert_slice out of loops.
std::unique_ptr<Pass> createNovaGPUHoistVectorExtractInsertSlicePass();
void registerNovaGPUHoistVectorExtractInsertSlicePass();

/// Normalizes strided-memref indices for WMMA load/store ops before NVVM legalization.
/// No-op for the nvgpu.mma.sync (Ampere native) path.
std::unique_ptr<Pass> createNovaGPUCastTypeToFitMMAPass();
void registerNovaGPUCastTypeToFitMMAPass();

// ---------------------------------------------------------------------------
// Warp-Level Vector Distribution Passes
// ---------------------------------------------------------------------------

/// Wraps each vector.contract in gpu.warp_execute_on_lane_0 (step 1 of warp
/// distribution). Must run BEFORE ConvertVectorToGPU so warp regions exist.
std::unique_ptr<Pass> createNovaGPUVectorWrapPass();
void registerNovaGPUVectorWrapPass();

/// Lowers gpu.warp_execute_on_lane_0 regions: inlines MMA regions (all 32
/// lanes must reach mma.sync), lowers others to scf.if + alloca + barrier.
/// Must run inside gpuPm AFTER ConvertVectorToGPU(useNvGpu=true).
std::unique_ptr<Pass> createNovaGPUWarpToSCFPass();
void registerNovaGPUWarpToSCFPass();

std::unique_ptr<Pass> createNovaFoldTransposeIntoConsumerPass();
void registerNovaFoldTransposeIntoConsumerPass();

std::unique_ptr<Pass> createNovaStrideReductionPass();
void registerNovaStrideReductionPass();

std::unique_ptr<Pass> createNovaDirectReductionLoweringPass();
void registerNovaDirectReductionLoweringPass();
} // namespace nova
} // namespace mlir

#endif // NOVA_TRANSFORMS_LLVMGPU_PASSES_H_
