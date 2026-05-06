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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace nova {

// Marker attribute set by NovaGPUMultiBuffering on every scf.for whose
// workgroup buffers were successfully widened. NovaGPUPipelining only
// time-shifts loops carrying this attribute: pipelining a non-multibuffered
// loop would race on shared memory because successive cp.async groups would
// alias the same buffer.
inline constexpr llvm::StringLiteral kNovaMultiBufferedLoopMarker =
    "__nova_multibuffered__";

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
std::unique_ptr<Pass>
createNovaGPUSelectLoweringStrategyPass(StringRef cudaArch = "sm_86");
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
// Uses a two-stage copy pattern (global→shared linalg.copy +
// nova.fusion_barrier
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

// Multi-buffers workgroup-memory allocations whose uses live inside an scf.for,
// to enable cp.async software pipelining. Widens `memref<TxSxf32, #ws>` →
// `memref<NxTxSxf32, #ws>` and indexes per iteration with `(iv mod N)`.
// Tags the rewritten scf.for with a marker attribute so the downstream
// pipelining pass only operates on loops whose buffers were actually widened.
std::unique_ptr<Pass> createNovaGPUMultiBufferingPass(unsigned numBuffers = 3);
void registerNovaGPUMultiBufferingPass();

// Software-pipelines K-loops containing nvgpu.device_async_copy ops. Time-
// shifts the cp.async ops by `depth-1` iterations (Stage 0 = transfers,
// Stage `depth-1` = compute) and rewrites the wait_group count from
// wait-all to `depth-1`. Only operates on loops marked by NovaGPUMultiBuffering
// — pipelining a loop without widened buffers would race on shared memory.
std::unique_ptr<Pass> createNovaGPUPipeliningPass(unsigned depth = 3);
void registerNovaGPUPipeliningPass();

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

std::unique_ptr<Pass> createNovaRepositionStorePass();
void registerNovaRepositionStorePass();

std::unique_ptr<Pass> createSCFScalarizeAccumulatorPass();

std::unique_ptr<Pass> createNovaMultiConsumerFusion();
void registerNovaMultiConsumerFusion();

// Computes per-operand thread distribution layouts for MMA contraction ops
// and attaches them as nova.layout_0/1/2 attributes on the linalg op.
// MUST run after thread tiling (Step 5) and before vectorization (Step 8).
std::unique_ptr<Pass> createNovaGPUConfigureTensorLayoutsPass();
void registerNovaGPUConfigureTensorLayoutsPass();

std::unique_ptr<Pass>
createNovaGPUReduceBankConflictsPass(StringRef arch = "sm_86");
void registerNovaGPUReduceBankConflictsPass();

std::unique_ptr<Pass> createNovaGPUSwizzleSharedMemoryPass();
void registerNovaGPUSwizzleSharedMemoryPass();

// ---------------------------------------------------------------------------
// Vectorization Passes  (Batch 1: declarations; wired into pipeline in Batch
// 2+)
// ---------------------------------------------------------------------------

/// Vectorizes linalg ops to vector.contract / vector.transfer_*.
/// MMA ops: vectorize + transfer nova.layout_* onto vector.contract.
/// Static ops: plain vectorize. Dynamic ops: masked vectorize via ValueBounds.
std::unique_ptr<Pass> createNovaGenericVectorizationPass();
void registerNovaGenericVectorizationPass();

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

/// Populates patterns that fold insert/extract strided-slice chains into direct
/// vector.transfer_read/write ops.  Used by GpuHardwareMappingPass (Stage 35)
/// so the patterns are not re-implemented inline in Passes.cpp:
///   • FoldExtractStridedSliceFromTransferRead  — rank-2 extract fold
///   • FoldInsertStridedSliceIntoTransferWrite  — insert chain → per-tile
///   writes • SplitTransferReadExtract                 — general minor-identity
///   extract fold
void populateGpuHardwareMappingStridedSlicePatterns(
    mlir::RewritePatternSet &patterns);

/// Normalizes strided-memref indices for WMMA load/store ops before NVVM
/// legalization. No-op for the nvgpu.mma.sync (Ampere native) path.
std::unique_ptr<Pass> createNovaGPUCastTypeToFitMMAPass();
void registerNovaGPUCastTypeToFitMMAPass();

std::unique_ptr<Pass> createNovaGPUVectorAllocPass();
void registerNovaGPUVectorAllocPass();

std::unique_ptr<Pass> createNovaGPUCombineValueSemanticBarriersPass();
void registerNovaGPUCombineValueSemanticBarriersPass();

std::unique_ptr<Pass> createNovaGPUVectorDistributePass();
void registerNovaGPUVectorDistributePass();

std::unique_ptr<Pass> createNovaStrideReductionPass();
void registerNovaStrideReductionPass();

/// Converts tensor-semantic nova_vector_ext.to_layout ops to vector-semantic
/// by wrapping each with vector.transfer_read + to_layout + transfer_write.
/// Must run BEFORE GenericVectorization and BEFORE bufferization so that
/// OneShotBufferize never sees tensor-semantic to_layout ops (which have no
/// BufferizableOpInterface and would be conservatively wrapped with
/// bufferization.to_tensor / to_buffer, surviving into LLVM).
std::unique_ptr<Pass> createNovaVectorizeVectorExtOpsPass();
void registerNovaVectorizeVectorExtOpsPass();

// Pipeline helpers — declared here so Passes.cpp and external callers can use
// them.
void addNovaPostBufferizationPasses(OpPassManager &pm);
void addNovaComprehensiveBufferizePasses(OpPassManager &pm);

// Registers BufferizableOpInterface external model on nova::ValueBarrierOp.
// Must be called during dialect registry setup (before any bufferization pass
// runs).
void registerNovaValueBarrierBufferizationInterface(
    mlir::DialectRegistry &registry);

} // namespace nova
} // namespace mlir

#endif // NOVA_TRANSFORMS_LLVMGPU_PASSES_H_
