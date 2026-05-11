//===- NovaKernelConfig.cpp - GPU kernel config heuristic -----------------===//
//
// Implements the heuristic that picks workgroup tile sizes, thread tiles,
// MMA intrinsic, and promoted operands for linalg compute ops, then attaches
// a "lowering_config" DictionaryAttr to each root op.
//
// Config dispatch (mirroring IREE's KernelConfig.cpp TileAndFuse path):
//
//   1. Contractions (setContractConfig):
//      a. Try MMA-based config (Volta+): sqrt/GCD/min tile distribution.
//      b. Fall back to SIMT tile table (8 entries from IREE).
//      Thread tiles = intrinsic tile shape (one MMA per thread slot).
//      Subgroup tiles = intrinsic * mnTileCountPerSubgroup.
//      C promotion = true only when accumulator carries live data from a
//                    prior kernel (readingExistingAccBuffer). Epilogue users
//                    do not trigger C promotion — C stays in registers.
//
//   2. Default (setDefaultConfig): For reductions and elementwise ops.
//      Mirrors IREE's setRootDefaultConfig:
//      - workgroupThreads = kTargetThreadsPerBlock = 256
//      - vectorSize = 4 (128-bit loads for f32)
//      - inner parallel dim tile = threads * vectorSize (up to 1024)
//      - outer parallel dims: remaining thread budget distributed inward->outward
//      - reduction dims tile = 4
//      - thread tile for inner dim = vectorSize
//
//   3. initNovaGPULaunchConfig: Single-root model per compute cluster.
//      Walks all linalg ops, finds roots, stamps configs.
//      Root ops = true contractions + any op with a reduction iterator.
//      Non-root ops (prologues, epilogues, fills) fuse into their root.
//
// Fix log:
//   [Fix2] Accumulator upcasting allowed at getMmaSchedule filter stage so
//          f16-problem + f32-accumulator intrinsics are not rejected too early.
//   [Fix3] sortMMAIntrinsics now breaks volume ties by preferring mma.sync over
//          wmma on Ampere+ (sm >= 80) so the faster instruction wins for large
//          problems instead of the one with the larger tile volume.
//   [Fix4] adjustSeedsForTarget re-runs Pass 1 after Pass 2 boost to ensure
//          workgroup count never drops below smCount after a boost.
//   [Fix5] threadTileM/N set to intrinsic size (correct for one MMA per thread
//          slot); comment added explaining why subgroupTile encodes multi-tile
//          ownership for the lowering backend.
//   [Fix6] FusedOpMemoryInfo: fused leading op shared memory cost is now
//          analysed and subtracted from the available smem budget before tile
//          selection, so MMA/SIMT tile sizes are shrunk to fit the real limit.
//          Simple elementwise ops (relu, bias_add, gelu) need no extra SMEM
//          (evaluated in registers). Row-reduction ops (layernorm) force
//          tileK == K and reserve per-row mean+variance scratch space.
//===----------------------------------------------------------------------===//


#include "Compiler/Transforms/LLVMGPU/NovaKernelConfig.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"


#define DEBUG_TYPE "nova-kernel-config"


namespace mlir::nova {


//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//


/// Preferred maximum threads per block for this target (Ampere RTX 3060).
/// CUDA's hard limit is 1024; we cap at 256 to leave register file headroom
/// and improve occupancy on the 28-SM GA106 die.
static constexpr int64_t kTargetThreadsPerBlock = 256;


/// Warp size for all NVIDIA GPUs.
static constexpr int64_t kWarpSize = 32;


/// Threshold below which a matmul M or N dimension is considered "very skinny"
/// and should be redirected to the vector reduction pipeline.
static constexpr int64_t kVerySkinnyDimThreshold = 4;


//===----------------------------------------------------------------------===//
// SIMT fallback tile table
// Mirrors IREE's getMatmulConfig(). Listed largest-to-smallest; we pick the
// first entry whose tile sizes divide the problem dimensions.
//===----------------------------------------------------------------------===//


struct SimtTilePair {
 std::array<int64_t, 3> tileMNK;   // Workgroup tile for M, N, K
 std::array<int64_t, 3> workgroup; // Thread block dim (x, y, z)
};


static constexpr SimtTilePair kSimtTable[] = {
   {{ 32, 128, 32}, {32,  8, 1}},  // 256 threads
   {{128,  64,  8}, {16,  8, 1}},  // 128 threads
   {{ 16, 256, 32}, {64,  2, 1}},  // 128 threads
   {{  8,  32, 32}, { 8,  8, 1}},  //  64 threads
   {{ 32, 128,  4}, {32,  8, 1}},  // 256 threads
   {{  8, 128,  4}, {32,  1, 1}},  //  32 threads
   {{ 16,  64,  4}, {16,  2, 1}},  //  32 threads
   {{  1, 128,  8}, {32,  1, 1}},  //  32 threads
};


//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//


enum class GemmSize { SmallGemm, MediumGemm, LargeGemm, VeryLargeGemm };


static llvm::StringRef gemmSizeName(GemmSize s) {
 switch (s) {
 case GemmSize::SmallGemm:     return "SmallGemm";
 case GemmSize::MediumGemm:    return "MediumGemm";
 case GemmSize::LargeGemm:     return "LargeGemm";
 case GemmSize::VeryLargeGemm: return "VeryLargeGemm";
 }
 return "Unknown";
}


struct GemmCutoffs {
 int64_t smallGemmCutoff;
 int64_t largeGemmCutoff;
 int64_t veryLargeGemmCutoff;
};


struct GPUMMAHeuristicSeeds {
 int64_t bestMNTileCountPerSubgroup;
 int64_t bestKElementCountPerSubgroup;
 int64_t bestSubgroupCountPerWorkgroup;
 int64_t bestKTileCountPerSubgroup;
 std::optional<int64_t> boostMNTileCountPerSubgroup = std::nullopt;
 std::optional<double>  minUtilizationThreshold     = std::nullopt;
};


struct GPUMMASchedule {
 NVMMAIntrinsicInfo intrinsic;
 int64_t wgM, wgN, wgK;
 // Total warps per workgroup = subgroupCountM * subgroupCountN.
 // Kept as (legacy) subgroupCount for downstream passes that still read it.
 int64_t subgroupCount;
 int64_t subgroupCountM;              // warps splitting M within a workgroup
 int64_t subgroupCountN;              // warps splitting N within a workgroup
 // Per-warp MMA tile counts. For the legacy (1-D warp split) path these are
 // equal to mnTileCountPerSubgroup. The 2-D warp grid path (VeryLarge
 // balanced GEMMs targeting a matmul.cpp-style 128x128 tile) sets them
 // independently.
 int64_t mnTileCountPerSubgroup;      // legacy; = max(mTile, nTile)
 int64_t mTileCountPerSubgroup;       // MMA tiles per warp along M
 int64_t nTileCountPerSubgroup;       // MMA tiles per warp along N
 int64_t kTileCountPerSubgroup;
};


/// All static problem parameters needed by the MMA heuristic.
struct ContractProblem {
 int64_t mSize, nSize, kSize;
 int32_t lhsKind, rhsKind, accKind; // 0=f16, 1=bf16, 2=f32, 3=tf32
 int64_t inBitWidth;
 GemmSize gemmSize;
 bool transposedLhs;
 bool transposedRhs;
};


struct MatmulDims {
 int64_t M = -1, N = -1, K = -1;
 bool valid() const { return M > 0 && N > 0 && K > 0; }
};


/// [Fix6] Describes the extra shared memory cost and tiling constraints
/// imposed by a fused leading op (e.g. layernorm, gelu, bias_add).
///
/// For simple elementwise ops (gelu, relu, bias_add):
///   - extraBytesPerInputElem > 0  (one extra copy of the A tile)
///   - hasRowReduction == false
///   - forcedTileK == -1           (no K constraint)
///
/// For row-reduction ops (layernorm):
///   - extraBytesPerInputElem > 0  (raw input tile must be staged)
///   - hasRowReduction == true     (mean + variance scratch needed per row)
///   - rowScratchBytesPerRow > 0   (sizeof(mean) + sizeof(var) = 8 bytes f32)
///   - forcedTileK == fullK        (must cover entire reduction dim at once)
struct FusedOpMemoryInfo {
 /// Extra bytes per element of the matmul A (LHS) tile needed to stage
 /// the fused op's input before writing the result into the MMA staging area.
 int64_t extraBytesPerInputElem = 0;


 /// True when the fused op requires a full-row reduction (e.g. layernorm).
 /// When true, tileK must equal the full K dimension — it cannot be tiled.
 bool hasRowReduction = false;


 /// Bytes of per-row scratch needed for row-reduction intermediates
 /// (mean + variance for layernorm = 2 * sizeof(float) = 8 bytes).
 /// Only meaningful when hasRowReduction == true.
 int64_t rowScratchBytesPerRow = 0;


 /// When >= 0, the tileK dimension is forced to this value (full K for
 /// layernorm). Ignored when == -1.
 int64_t forcedTileK = -1;


 /// True when any leading fused op was detected (even if trivially cheap).
 bool hasFusedOp = false;


 /// Compute the total extra shared memory (bytes) for a tile of size
 /// wgM x wgK (the A-tile footprint).
 int64_t extraSmemBytes(int64_t wgM, int64_t wgK) const {
   int64_t inputTileBytes = extraBytesPerInputElem * wgM * wgK;
   int64_t scratchBytes   = hasRowReduction ? rowScratchBytesPerRow * wgM : 0;
   return inputTileBytes + scratchBytes;
 }
};


//===----------------------------------------------------------------------===//
// Small utilities
//===----------------------------------------------------------------------===//


static int64_t intSqrt(int64_t n) {
 if (n <= 0) return 0;
 int64_t s = static_cast<int64_t>(std::sqrt(static_cast<double>(n)));
 while (s * s > n) --s;
 while ((s + 1) * (s + 1) <= n) ++s;
 return s;
}


static int64_t gcd(int64_t a, int64_t b) {
 while (b) { int64_t t = b; b = a % b; a = t; }
 return a;
}


//===----------------------------------------------------------------------===//
// inferMatmulDims
//===----------------------------------------------------------------------===//


static MatmulDims inferMatmulDims(linalg::LinalgOp op) {
 auto contractionDims = mlir::linalg::inferContractionDims(op);
 if (failed(contractionDims))
   return {};
 if (contractionDims->m.empty() || contractionDims->n.empty() ||
     contractionDims->k.empty())
   return {};


 SmallVector<int64_t> bounds = op.getStaticLoopRanges();
 int64_t mDim = contractionDims->m.back();
 int64_t nDim = contractionDims->n.back();
 int64_t kDim = contractionDims->k.back();


 if (ShapedType::isDynamic(bounds[mDim]) ||
     ShapedType::isDynamic(bounds[nDim]) ||
     ShapedType::isDynamic(bounds[kDim]))
   return {};


 return {bounds[mDim], bounds[nDim], bounds[kDim]};
}


//===----------------------------------------------------------------------===//
// computePaddingSizes
//===----------------------------------------------------------------------===//


static SmallVector<int64_t>
computePaddingSizes(linalg::LinalgOp op,
                   ArrayRef<int64_t> workgroupTiles,
                   ArrayRef<int64_t> reductionTiles) {
 int numLoops = op.getNumLoops();
 SmallVector<int64_t> padding(numLoops, 1);
 auto iterTypes = op.getIteratorTypesArray();
 for (int i = 0; i < numLoops; ++i) {
   if (linalg::isParallelIterator(iterTypes[i])) {
     int64_t tile = (i < (int)workgroupTiles.size()) ? workgroupTiles[i] : 0;
     if (tile > 0) padding[i] = tile;
   } else {
     int64_t tile = (i < (int)reductionTiles.size()) ? reductionTiles[i] : 0;
     if (tile > 0) padding[i] = tile;
   }
 }
 return padding;
}


//===----------------------------------------------------------------------===//
// clampThreadTilesToMaxThreads
//===----------------------------------------------------------------------===//


static void clampThreadTilesToMaxThreads(ArrayRef<int64_t> workgroupTiles,
                                        SmallVectorImpl<int64_t> &threadTiles) {
 int n = (int)threadTiles.size();


 auto computeTotal = [&]() -> int64_t {
   int64_t total = 1;
   for (int i = 0; i < n; ++i) {
     if (threadTiles[i] <= 0) continue;
     if (i >= (int)workgroupTiles.size() || workgroupTiles[i] <= 0) continue;
     int64_t trips = (workgroupTiles[i] + threadTiles[i] - 1) / threadTiles[i];
     total *= trips;
   }
   return total;
 };


 while (computeTotal() > kTargetThreadsPerBlock) {
   int worstDim = -1;
   int64_t worstTrips = 0;
   for (int i = 0; i < n; ++i) {
     if (threadTiles[i] <= 0) continue;
     if (i >= (int)workgroupTiles.size() || workgroupTiles[i] <= 0) continue;
     int64_t trips = (workgroupTiles[i] + threadTiles[i] - 1) / threadTiles[i];
     if (trips > worstTrips) { worstTrips = trips; worstDim = i; }
   }
   if (worstDim < 0) break;
   threadTiles[worstDim] *= 2;
   if (worstDim < (int)workgroupTiles.size())
     threadTiles[worstDim] = std::min(threadTiles[worstDim],
                                       workgroupTiles[worstDim]);
   if (computeTotal() > kTargetThreadsPerBlock && worstTrips <= 1) break;
 }
}


//===----------------------------------------------------------------------===//
// Transposition detection helpers
//===----------------------------------------------------------------------===//


static bool isTransposedLhs(linalg::LinalgOp op,
                            const linalg::ContractionDimensions &cd) {
 AffineMap lhsMap = op.getMatchingIndexingMap(op.getDpsInputOperand(0));
 if (lhsMap.getNumResults() == 0) return false;
 unsigned lastDim =
     llvm::cast<AffineDimExpr>(
         lhsMap.getResult(lhsMap.getNumResults() - 1)).getPosition();
 for (int64_t k : cd.k)
   if ((unsigned)k == lastDim) return true;
 return false;
}


static bool isTransposedRhs(linalg::LinalgOp op,
                            const linalg::ContractionDimensions &cd) {
 AffineMap rhsMap = op.getMatchingIndexingMap(op.getDpsInputOperand(1));
 if (rhsMap.getNumResults() == 0) return false;
 unsigned lastDim =
     llvm::cast<AffineDimExpr>(
         rhsMap.getResult(rhsMap.getNumResults() - 1)).getPosition();
 for (int64_t n : cd.n)
   if ((unsigned)n == lastDim) return true;
 return false;
}


//===----------------------------------------------------------------------===//
// readingExistingAccBuffer
//===----------------------------------------------------------------------===//


bool readingExistingAccBuffer(linalg::LinalgOp op) {
 // Returns true only when the init operand (C) is a live accumulated value
 // rather than a freshly allocated or zero-initialized tensor.
 for (OpOperand &initOperand : op.getDpsInitsMutable()) {
   Value initVal = initOperand.get();
   Operation *defOp = initVal.getDefiningOp();
   if (defOp && isa<tensor::EmptyOp>(defOp)) continue;  // fresh alloc
   if (defOp && isa<linalg::FillOp>(defOp))  continue;  // zero-filled
   return true; // existing accumulated buffer — must promote C
 }
 return false;
}


//===----------------------------------------------------------------------===//
// analyzeFusedLeadingOp
//
// Walks producers of the matmul's LHS (operand 0). Classifies what kind of
// fused leading op is present and returns the corresponding memory info.
//
//   - Row-reduction ops (layernorm): forces tileK = fullK, reserves 8 bytes/row
//     mean+var scratch, stages input tile (4 bytes/elem extra SMEM).
//   - Pure-parallel ops (gelu, bias_add, relu): evaluated in registers while
//     loading the A tile — no extra SMEM beyond the A tile itself.
//
// Only looks one level up the def chain.
//===----------------------------------------------------------------------===//


static FusedOpMemoryInfo analyzeFusedLeadingOp(linalg::LinalgOp matmul,
                                               const MatmulDims &dims) {
 FusedOpMemoryInfo info;


 // All ops at this stage are linalg.*  — there are no nova.* ops in the IR.
 // A fused leading op is a linalg.generic (or linalg.add) that:
 //   (a) feeds the matmul's LHS directly, AND
 //   (b) has no lowering_config yet (it will be fused into the matmul kernel,
 //       not dispatched as its own kernel).
 Value lhs = matmul.getDpsInputOperand(0)->get();
 Operation *producer = lhs.getDefiningOp();
 if (!producer)
   return info;


 // Fills and empty allocations are not fused ops.
 if (isa<linalg::FillOp, tensor::EmptyOp>(producer))
   return info;


 // If the producer already has a lowering_config it is an independent root,
 // not a fused prologue — nothing to account for.
 if (getLoweringConfig(producer))
   return info;


 auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
 if (!producerLinalgOp)
   return info;


 auto iterTypes = producerLinalgOp.getIteratorTypesArray();
 bool hasReduction = llvm::any_of(iterTypes, [](utils::IteratorType t) {
   return linalg::isReductionIterator(t);
 });


 info.hasFusedOp = true;


 if (hasReduction) {
   // Inline row-reduction (e.g. layernorm stats computed inside the same
   // tile as the matmul). tileK must cover the full K so all elements are
   // visible for mean/variance; reserve 8 bytes/row of scratch (f32 mean +
   // f32 variance). Input tile must also be staged → 4 bytes/elem.
   info.hasRowReduction        = true;
   info.forcedTileK            = dims.K;
   info.extraBytesPerInputElem = 4;
   info.rowScratchBytesPerRow  = 8;
   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] Fused leading op: linalg.generic "
              << "(row-reduction, forcedTileK=" << dims.K << ")\n");
 } else {
   // Pure parallel: elementwise (GELU, bias-add, relu) or a normalization
   // epilogue that reads pre-computed mean/var scalars.
   // These are evaluated in registers while loading the A tile — no extra
   // shared memory is required beyond the A tile itself.
   info.hasRowReduction        = false;
   info.forcedTileK            = -1;
   info.extraBytesPerInputElem = 0;
   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] Fused leading op: "
              << producer->getName() << " (parallel, in-register)\n");
 }


 return info;
}


//===----------------------------------------------------------------------===//
// GemmCutoffs + heuristic seeds
//===----------------------------------------------------------------------===//
static GemmCutoffs getGemmCutoffs(const NVIDIATargetInfo &target) {
 if (target.maxWorkgroupMemBytes >  48 * 1024) return {2, 16, 64}; // was >= 100*1024
 if (target.maxWorkgroupMemBytes >= 48 * 1024) return {1,  8, 32};
 return {1, 4, 16};
}




static std::optional<GPUMMAHeuristicSeeds>
getContractionHeuristicSeeds(const ContractProblem &problem,
                             bool isGemm, bool scaled) {
 switch (problem.gemmSize) {
 case GemmSize::SmallGemm:
   if (problem.inBitWidth <= 8)  return GPUMMAHeuristicSeeds{2, 64, 1, 4};
   if (problem.inBitWidth <= 16) return GPUMMAHeuristicSeeds{2, 32, 1, 2};
   return GPUMMAHeuristicSeeds{2, 16, 1, 1};


 case GemmSize::MediumGemm:
   if (problem.inBitWidth <= 8)  return GPUMMAHeuristicSeeds{4, 64, 2, 4};
   if (problem.inBitWidth <= 16) return GPUMMAHeuristicSeeds{4, 32, 2, 2};
   return GPUMMAHeuristicSeeds{4, 16, 2, 1};


 case GemmSize::LargeGemm:
   if (problem.inBitWidth <= 8)
     return GPUMMAHeuristicSeeds{4, 64, 4, 4, /*boostMNT=*/8, /*util=*/0.85};
   if (problem.inBitWidth <= 16)
     return GPUMMAHeuristicSeeds{4, 32, 4, 2, /*boostMNT=*/8, /*util=*/0.85};
   return GPUMMAHeuristicSeeds{4, 16, 4, 1, /*boostMNT=*/8, /*util=*/0.85};


 case GemmSize::VeryLargeGemm:
   if (problem.inBitWidth <= 8)
     return GPUMMAHeuristicSeeds{4, 128, 4, 8,
                                 /*boostMNT=*/std::nullopt, /*util=*/0.80};
   if (problem.inBitWidth <= 16)
     return GPUMMAHeuristicSeeds{4, 64, 4, 4,
                                 /*boostMNT=*/std::nullopt, /*util=*/0.80};
   // f32 / TF32: target a 128x128x32 workgroup tile with a 2x2 warp grid.
   //
   // Seeds: {MNT=4, kElem=8, S=4, kTiles=4}
   //
   // Before 2x2 reshape: wgM = 16*4*4 = 256, wgN = 8*4 = 32.
   // After 2x2 reshape (fires for VeryLarge when nTiles has headroom):
   //   subgroupCountM=2, subgroupCountN=2, nTileCount=8
   //   wgM = 16*4*2 = 128,  wgN = 8*8*2 = 128.
   //
   // K-step: bestKTileCountPerSubgroup=4 → wgK = 8*4 = 32.
   // Raises arithmetic intensity from ~1 FLOP/byte (K=8) to ~8 FLOP/byte,
   // keeping the 3-stage async pipeline full.
   // SMEM: 2*(128*32*4 + 32*128*4) = ~49 KB; fitScheduleInSharedMemory
   // clamps wgK to 16 if target is strictly 48 KB.
   //
   // Register spill note: per-warp tile is 64x64, FR_M=4, FR_N=8 → 128
   // acc regs/warp. Resolve this in matmul.cpp by reducing WARP_N to 32
   // (FR_N=4, 64 regs/warp). The workgroup tile stays 128x128 for occupancy.
   //
   // Single canonical tile: all matmul ops in a function share this seed,
   // so forward/dX/dW all produce the same tile regime regardless of shape.
   return GPUMMAHeuristicSeeds{4, 8, 4, 4,
                               /*boostMNT=*/std::nullopt, /*util=*/0.80};
 }
 return std::nullopt;
}


//===----------------------------------------------------------------------===//
// computeSharedMemory
//===----------------------------------------------------------------------===//


static int64_t computeSharedMemory(int64_t wgM, int64_t wgN, int64_t wgK,
                                  int32_t lhsKind, int32_t rhsKind,
                                  bool doCPromotion,
                                  const FusedOpMemoryInfo &fusedInfo = {}) {
 auto elemBytes = [](int32_t kind) -> int64_t {
   return (kind == 0 || kind == 1) ? 2 : 4;
 };
 int64_t lhsBytes = elemBytes(lhsKind);
 int64_t rhsBytes = elemBytes(rhsKind);
 int64_t accBytes = 4;
 // Bank-conflict avoidance pads the contiguous (K) dimension by 4 elements
 // for both A (KxN-major) and B (MxK-major) tiles. Without modeling this in
 // the SMEM estimate the fitter will accept a tile whose actual allocation
 // exceeds the dynamic-SMEM cap (~12% bigger than the unpadded estimate),
 // and the kernel launches with insufficient SMEM → wrong values / illegal
 // memory access. See NovaGPUReduceBankConflicts.cpp for the padding pass.
 constexpr int64_t kBankPadElems = 4;
 int64_t paddedK = wgK + kBankPadElems;
 // A tile + B tile, 3-buffered for the cp.async pipeline.
 int64_t smem = 3 * (wgM * paddedK * lhsBytes + paddedK * wgN * rhsBytes);
 if (doCPromotion)
   smem += wgM * wgN * accBytes;
 // [Fix6] Add shared memory cost of fused leading op:
 //   - Raw input tile staging (one extra A-tile-sized buffer).
 //   - Per-row mean + variance scratch for layernorm.
 smem += fusedInfo.extraSmemBytes(wgM, wgK);
 return smem;
}


//===----------------------------------------------------------------------===//
// computeEstimatedWorkgroupCount
//===----------------------------------------------------------------------===//


static int64_t computeEstimatedWorkgroupCount(
   const GPUMMAHeuristicSeeds &seeds, const ContractProblem &problem,
   const NVMMAIntrinsicInfo &intrinsic, int64_t splitReductionTripCnt) {
 int64_t wgM = intrinsic.mSize * seeds.bestMNTileCountPerSubgroup *
               seeds.bestSubgroupCountPerWorkgroup;
 int64_t wgN = intrinsic.nSize * seeds.bestMNTileCountPerSubgroup;
 wgM = std::max<int64_t>(wgM, 1);
 wgN = std::max<int64_t>(wgN, 1);
 int64_t numM = (problem.mSize + wgM - 1) / wgM;
 int64_t numN = (problem.nSize + wgN - 1) / wgN;
 return numM * numN * splitReductionTripCnt;
}


//===----------------------------------------------------------------------===//
// adjustSeedsForTarget
// [Fix4] Re-runs Pass 1 after Pass 2 boost so workgroup count never falls
//         below smCount after an MNT boost.
//===----------------------------------------------------------------------===//


static void adjustSeedsForTarget(GPUMMAHeuristicSeeds &seeds,
                                 const ContractProblem &problem,
                                 const NVMMAIntrinsicInfo &intrinsic,
                                 const NVIDIATargetInfo &target,
                                 int64_t splitReductionTripCnt) {
 int64_t wgpCount = target.smCount;
 if (wgpCount == 0) {
   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] SM count unavailable, "
                 "skipping seed adjustment.\n");
   return;
 }


 if (problem.gemmSize == GemmSize::SmallGemm) {
   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] Arithmetic intensity too low, "
                 "skipping seed adjustment.\n");
   return;
 }


 // Helper: reduce MNT until numWorkgroups >= wgpCount.
 auto fillCUs = [&]() {
   int64_t numWG = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                                  splitReductionTripCnt);
   LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Estimated workgroups: "
                            << numWG << ", SM count: " << wgpCount << "\n");
   while (numWG < wgpCount) {
     if (seeds.bestMNTileCountPerSubgroup <= 1) {
       LLVM_DEBUG(llvm::dbgs()
                  << "[nova-kernel-config] Cannot decrease MNT further.\n");
       break;
     }
     seeds.bestMNTileCountPerSubgroup /= 2;
     LLVM_DEBUG(llvm::dbgs()
                << "[nova-kernel-config] Decreasing MNT to "
                << seeds.bestMNTileCountPerSubgroup << "\n");
     numWG = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                            splitReductionTripCnt);
   }
 };


 // Pass 1: Fill CUs.
 fillCUs();


 // Pass 2: Boost MNT for large balanced GEMMs (K not dominant).
 if (seeds.boostMNTileCountPerSubgroup) {
   int64_t boostMNT = *seeds.boostMNTileCountPerSubgroup;
   int64_t boostedWGSize = boostMNT * intrinsic.mSize * intrinsic.nSize *
                           seeds.bestSubgroupCountPerWorkgroup;
   bool kDominated = problem.kSize > std::max(problem.mSize, problem.nSize);
   bool enoughOutput =
       problem.mSize * problem.nSize >= 2 * wgpCount * boostedWGSize;
   if (!kDominated && enoughOutput) {
     seeds.bestMNTileCountPerSubgroup =
         std::max(seeds.bestMNTileCountPerSubgroup, boostMNT);
     LLVM_DEBUG(llvm::dbgs()
                << "[nova-kernel-config] Boosting MNT to "
                << seeds.bestMNTileCountPerSubgroup << "\n");
     // [Fix4] Re-run Pass 1: boost may have shrunk workgroup count below smCount.
     fillCUs();
   }
 }


 // Pass 3: Utilisation trim — halve MNT when last wave is mostly idle.
 if (seeds.minUtilizationThreshold) {
   double threshold = *seeds.minUtilizationThreshold;
   int64_t numWG = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                                  splitReductionTripCnt);
   auto computeUtilization = [&]() -> double {
     int64_t waves = (numWG + wgpCount - 1) / wgpCount;
     if (waves == 0) return 0.0;
     return static_cast<double>(numWG) / (waves * wgpCount);
   };
   while (computeUtilization() < threshold) {
     if (seeds.bestMNTileCountPerSubgroup <= 1) break;
     seeds.bestMNTileCountPerSubgroup /= 2;
     numWG = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                            splitReductionTripCnt);
     LLVM_DEBUG(llvm::dbgs()
                << "[nova-kernel-config] Low utilisation, MNT -> "
                << seeds.bestMNTileCountPerSubgroup << "\n");
   }
 }
}


//===----------------------------------------------------------------------===//
// canTargetIntrinsic
//===----------------------------------------------------------------------===//


static LogicalResult canTargetIntrinsic(const ContractProblem &problem,
                                        const NVMMAIntrinsicInfo &intrinsic,
                                        int32_t subgroupSize,
                                        bool canUpcastAcc,
                                        bool mustBeAligned) {
 if (intrinsic.warpSize != subgroupSize)
   return failure();


 if (intrinsic.accKind != problem.accKind) {
   if (!canUpcastAcc || intrinsic.accKind < problem.accKind)
     return failure();
 }


 if (mustBeAligned) {
   if (problem.mSize % intrinsic.mSize != 0) return failure();
   if (problem.nSize % intrinsic.nSize != 0) return failure();
   if (problem.kSize % intrinsic.kSize != 0) return failure();
 }
 return success();
}


//===----------------------------------------------------------------------===//
// sortMMAIntrinsics
// [Fix3] Break volume ties by preferring mma.sync over wmma on Ampere+.
//===----------------------------------------------------------------------===//


static SmallVector<NVMMAIntrinsicInfo>
sortMMAIntrinsics(const ContractProblem &problem,
                 ArrayRef<NVMMAIntrinsicInfo> intrinsics,
                 const NVIDIATargetInfo &target) {
 SmallVector<NVMMAIntrinsicInfo> sorted(intrinsics.begin(), intrinsics.end());


 auto tileVolume = [](const NVMMAIntrinsicInfo &info) -> int64_t {
   return info.mSize * info.nSize * info.kSize;
 };


 // [Fix3] On Ampere+ (sm >= 80), mma.sync is strictly preferred over wmma
 // because it has lower latency, higher throughput, and better pipelining.
 // Assign a generation rank: mma.sync > wmma so tie-breaking favours it.
 auto generationRank = [](const NVMMAIntrinsicInfo &info) -> int {
   switch (info.intrinsic) {
   case NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16:
   case NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16:
   case NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8:
     return 2; // Ampere mma.sync — highest preference
   case NVMMAIntrinsicValues::WMMA_F32_16x16x16:
   case NVMMAIntrinsicValues::WMMA_F16_16x16x16:
     return 1; // Volta/Turing wmma — lower preference
   default:
     return 0;
   }
 };


 bool preferMmaSync = (target.preferredSubgroupSize == 32) &&
                      (target.maxWorkgroupMemBytes >= 48 * 1024);


 if (problem.gemmSize == GemmSize::LargeGemm ||
     problem.gemmSize == GemmSize::VeryLargeGemm) {
   llvm::stable_sort(sorted, [&](const NVMMAIntrinsicInfo &a,
                                  const NVMMAIntrinsicInfo &b) {
     if (preferMmaSync) {
       int ra = generationRank(a), rb = generationRank(b);
       if (ra != rb) return ra > rb; // higher generation first
     }
     return tileVolume(a) > tileVolume(b); // then larger tile first
   });
 } else if (problem.gemmSize == GemmSize::SmallGemm) {
   llvm::stable_sort(sorted, [&](const NVMMAIntrinsicInfo &a,
                                  const NVMMAIntrinsicInfo &b) {
     return tileVolume(a) < tileVolume(b); // smaller tile for small GEMMs
   });
 }
 // MediumGemm: preserve hardware-table order.
 return sorted;
}


//===----------------------------------------------------------------------===//
// getOptimalMMASchedule
//===----------------------------------------------------------------------===//


static GPUMMASchedule
getOptimalMMASchedule(const ContractProblem &problem,
                     const NVMMAIntrinsicInfo &intrinsic,
                     const GPUMMAHeuristicSeeds &seeds,
                     int64_t effectiveK) {
 int64_t S = seeds.bestSubgroupCountPerWorkgroup;
 int64_t T = seeds.bestMNTileCountPerSubgroup;


 int64_t mTiles = (problem.mSize + intrinsic.mSize - 1) / intrinsic.mSize;
 int64_t nTiles = (problem.nSize + intrinsic.nSize - 1) / intrinsic.nSize;


 int64_t subgroupCount = S;
 int64_t mnTileCount   = T;


 // Strategy 1: square-root — balanced M/N coverage.
 int64_t splitFactor = intSqrt(S * T);
 bool usedSqrt = false;
 if (splitFactor > 1) {
   if (mTiles % splitFactor == 0 && splitFactor <= mTiles) {
     subgroupCount = std::min(splitFactor, mTiles);
     mnTileCount   = std::min(splitFactor,
                              std::min(mTiles / subgroupCount, nTiles));
     usedSqrt = true;
   } else if (nTiles % splitFactor == 0 && splitFactor <= nTiles) {
     subgroupCount = 1;
     mnTileCount   = std::min(splitFactor, nTiles);
     usedSqrt = true;
   }
 }


 if (!usedSqrt) {
   // Strategy 2: GCD fallback — clean M split without remainder.
   int64_t g = gcd(S, mTiles);
   if (g > 1) {
     subgroupCount = std::min(g, mTiles);
     mnTileCount   = std::min(T, nTiles);
   } else {
     // Strategy 3: min fallback — conservative clamp.
     subgroupCount = std::min(S, mTiles);
     mnTileCount   = std::min(T, nTiles);
   }
 }


 // Final clamp.
 subgroupCount = std::max<int64_t>(subgroupCount, 1);
 mnTileCount   = std::max<int64_t>(mnTileCount,   1);
 subgroupCount = std::min(subgroupCount, mTiles);
 mnTileCount   = std::min(mnTileCount,
                           std::min((mTiles + subgroupCount - 1) / subgroupCount,
                                    nTiles));


 // K tiling.
 int64_t kStep = (seeds.bestKElementCountPerSubgroup / intrinsic.kSize) *
                 intrinsic.kSize;
 kStep = std::max(kStep, intrinsic.kSize);
 int64_t wgK = std::min(kStep * seeds.bestKTileCountPerSubgroup, effectiveK);
 wgK = (wgK / intrinsic.kSize) * intrinsic.kSize;
 wgK = std::max(wgK, intrinsic.kSize);


 int64_t wgM = intrinsic.mSize * mnTileCount * subgroupCount;
 int64_t wgN = intrinsic.nSize * mnTileCount;


 // Default: 1-D warp split — all warps along M, one warp wide in N.
 int64_t subgroupCountM = subgroupCount;
 int64_t subgroupCountN = 1;
 int64_t mTileCount     = mnTileCount;
 int64_t nTileCount     = mnTileCount;


 // ── 2-D warp grid reshape (matmul.cpp-style 2x2) ─────────────────────────
 // For Large/VeryLarge GEMMs with room on the N-side, redistribute the
 // warp group from [S warps in M × 1 in N] into a balanced 2-D grid.
 // Mirrors the eager/direct-lowering layout: 2 warps in M, 2 in N, each
 // warp covering per-warp tile (wgM / 2) × (wgN / 2).
 //
 // Conditions:
 //   * subgroupCount is even (>= 2)
 //   * nTiles has headroom to double the N-side tile
 //   * K is not dominant (column-heavy problems prefer the old split)
 if ((problem.gemmSize == GemmSize::LargeGemm ||
      problem.gemmSize == GemmSize::VeryLargeGemm) &&
     subgroupCount >= 2 && (subgroupCount % 2) == 0 &&
     nTiles >= mnTileCount * 2 &&
     problem.kSize <= std::max(problem.mSize, problem.nSize)) {
   subgroupCountM = subgroupCount / 2;
   subgroupCountN = 2;
   // Per-warp MMA tile counts preserve the per-warp footprint in M and
   // double the per-warp footprint in N (new N-warp absorbs its share).
   mTileCount = mnTileCount;
   nTileCount = mnTileCount * 2;
   // Clamp to available tiles so we don't over-cover.
   mTileCount = std::min(mTileCount,
                         std::max<int64_t>(mTiles / std::max<int64_t>(subgroupCountM, 1),
                                           int64_t{1}));
   nTileCount = std::min(nTileCount,
                         std::max<int64_t>(nTiles / subgroupCountN, int64_t{1}));
   wgM = intrinsic.mSize * mTileCount * subgroupCountM;
   wgN = intrinsic.nSize * nTileCount * subgroupCountN;
 }


 return GPUMMASchedule{intrinsic, wgM, wgN, wgK,
                       subgroupCount,
                       subgroupCountM, subgroupCountN,
                       std::max(mTileCount, nTileCount),  // mnTileCount legacy
                       mTileCount, nTileCount,
                       seeds.bestKTileCountPerSubgroup};
}


//===----------------------------------------------------------------------===//
// fitScheduleInSharedMemory
//===----------------------------------------------------------------------===//


// [Fix6] fitScheduleInSharedMemory — now fused-op-aware
//
// rowScratchBytesPerRow scales with wgM (which changes as we shrink MNT),
// so we re-evaluate the full cost inside the loop rather than pre-subtracting.
static std::optional<GPUMMASchedule>
fitScheduleInSharedMemory(GPUMMASchedule sched,
                         const NVMMAIntrinsicInfo &intrinsic,
                         int64_t maxSmem,
                         bool doCPromotion,
                         const FusedOpMemoryInfo &fusedInfo = {}) {
 while (true) {
   int64_t smem = computeSharedMemory(sched.wgM, sched.wgN, sched.wgK,
                                      intrinsic.lhsKind, intrinsic.rhsKind,
                                      doCPromotion, fusedInfo);
   if (smem <= maxSmem) return sched;


   // Halve whichever side has the larger per-warp MMA tile count first.
   // Fall through to halving the other side when equal.
   bool halveM = sched.mTileCountPerSubgroup >= sched.nTileCountPerSubgroup;
   if (halveM && sched.mTileCountPerSubgroup <= 1)
     halveM = false;
   if (!halveM && sched.nTileCountPerSubgroup <= 1) {
     if (sched.mTileCountPerSubgroup <= 1)
       return std::nullopt;
     halveM = true;
   }

   if (halveM) {
     int64_t newMT = sched.mTileCountPerSubgroup / 2;
     if (newMT < 1) return std::nullopt;
     sched.mTileCountPerSubgroup = newMT;
   } else {
     int64_t newNT = sched.nTileCountPerSubgroup / 2;
     if (newNT < 1) return std::nullopt;
     sched.nTileCountPerSubgroup = newNT;
   }

   sched.wgM = std::max(intrinsic.mSize * sched.mTileCountPerSubgroup *
                        sched.subgroupCountM,
                        intrinsic.mSize);
   sched.wgN = std::max(intrinsic.nSize * sched.nTileCountPerSubgroup *
                        sched.subgroupCountN,
                        intrinsic.nSize);
   sched.mnTileCountPerSubgroup =
       std::max(sched.mTileCountPerSubgroup, sched.nTileCountPerSubgroup);


   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] fitScheduleInSharedMemory: smem="
              << smem << " > limit=" << maxSmem
              << " (incl. fused-op overhead), shrinking "
              << (halveM ? "M" : "N") << "-tile. New mT="
              << sched.mTileCountPerSubgroup
              << " nT=" << sched.nTileCountPerSubgroup
              << " wgM=" << sched.wgM << " wgN=" << sched.wgN << "\n");
 }
}


//===----------------------------------------------------------------------===//
// deduceMMASchedule
//===----------------------------------------------------------------------===//


static std::optional<GPUMMASchedule>
deduceMMASchedule(const ContractProblem &problem,
                 const NVIDIATargetInfo &target,
                 ArrayRef<NVMMAIntrinsicInfo> intrinsics,
                 const GPUMMAHeuristicSeeds &seeds,
                 bool mustBeAligned,
                 bool doCPromotion,
                 const FusedOpMemoryInfo &fusedInfo = {},
                 int64_t splitReductionTripCnt = 1) {
 splitReductionTripCnt = std::max<int64_t>(splitReductionTripCnt, 1);


 // [Fix3] Pass target to sortMMAIntrinsics for generation-aware tie-breaking.
 SmallVector<NVMMAIntrinsicInfo> sortedIntrinsics =
     sortMMAIntrinsics(problem, intrinsics, target);


 for (const NVMMAIntrinsicInfo &intrinsic : sortedIntrinsics) {
   // [Fix2] canUpcastAcc=true: allow f16 problem with f32 accumulator intrinsic.
   if (failed(canTargetIntrinsic(problem, intrinsic,
                                 target.preferredSubgroupSize,
                                 /*canUpcastAcc=*/true, mustBeAligned)))
     continue;


   int64_t effectiveK = problem.kSize / splitReductionTripCnt;


   // [Fix6] If layernorm forces a full-K tile, enforce it here.
   if (fusedInfo.forcedTileK >= 0)
     effectiveK = fusedInfo.forcedTileK;


   if (mustBeAligned && effectiveK % intrinsic.kSize != 0)
     continue;


   GPUMMAHeuristicSeeds localSeeds = seeds;
   adjustSeedsForTarget(localSeeds, problem, intrinsic, target,
                        splitReductionTripCnt);


   GPUMMASchedule sched =
       getOptimalMMASchedule(problem, intrinsic, localSeeds, effectiveK);


   if (sched.wgM < intrinsic.mSize) continue;
   if (sched.wgN < intrinsic.nSize) continue;
   if (sched.wgK < intrinsic.kSize) continue;


   if (mustBeAligned) {
     if (problem.mSize % sched.wgM != 0) continue;
     if (problem.nSize % sched.wgN != 0) continue;
     if (effectiveK   % sched.wgK != 0) continue;
   }


   // Use the dynamic SMEM budget here. The 3-stage matmul pipeline emits
   // gpu.dynamic_shared_memory + view (NovaGPUMapForallToGPU) and the launcher
   // calls cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) to opt into the
   // larger carve-out. With static-only budget (48 KB) the heuristic would
   // shrink (128, 64, 32) tiles unnecessarily even though they fit dynamically
   // (~72 KB at depth=3). fusedInfo is still passed so the fitter accounts for
   // any extra fused-op SMEM staging.
   std::optional<GPUMMASchedule> fitted =
       fitScheduleInSharedMemory(sched, intrinsic,
                                 target.maxWorkgroupDynamicMemBytes,
                                 doCPromotion, fusedInfo);
   if (!fitted) continue;
   sched = *fitted;


   if (mustBeAligned) {
     if (problem.mSize % sched.wgM != 0) continue;
     if (problem.nSize % sched.wgN != 0) continue;
     if (effectiveK   % sched.wgK != 0) continue;
   }


   LLVM_DEBUG({
     int64_t smem = computeSharedMemory(sched.wgM, sched.wgN, sched.wgK,
                                        intrinsic.lhsKind, intrinsic.rhsKind,
                                        doCPromotion, fusedInfo);
     llvm::dbgs() << "[nova-kernel-config] deduceMMASchedule: intrinsic="
                  << (int)intrinsic.intrinsic
                  << " wgM=" << sched.wgM << " wgN=" << sched.wgN
                  << " wgK=" << sched.wgK
                  << " subgroups=" << sched.subgroupCount
                  << " mnTile=" << sched.mnTileCountPerSubgroup
                  << " smem=" << smem
                  << " fusedOpOverhead="
                  << fusedInfo.extraSmemBytes(sched.wgM, sched.wgK)
                  << " transLhs=" << problem.transposedLhs
                  << " transRhs=" << problem.transposedRhs << "\n";
   });
   return sched;
 }
 return std::nullopt;
}


//===----------------------------------------------------------------------===//
// getMmaSchedule
// [Fix2] Intrinsic filter now passes canUpcastAcc=true at deduceMMASchedule
//         level; the pre-filter here is relaxed to also accept wider acc types.
//===----------------------------------------------------------------------===//


static std::optional<GPUMMASchedule>
getMmaSchedule(const NVIDIATargetInfo &target, ContractProblem &problem,
              bool isGemm, bool mustBeAligned = true,
              bool doCPromotion = false,
              const FusedOpMemoryInfo &fusedInfo = {},
              int64_t splitReductionTripCnt = 1) {
 const int32_t targetSubgroupSize = target.preferredSubgroupSize;
 SmallVector<NVMMAIntrinsicInfo> intrinsics;


 for (const NVMMAIntrinsicInfo &mma : target.mmaIntrinsics) {
   if (!mma.getDistributionMappingKind()) continue;
   if (mma.warpSize != targetSubgroupSize) continue;


   auto [mSize, nSize, kSize] = mma.getMNKShape();
   auto [aKind, bKind, cKind] = mma.getABCElementKinds();


   // Input types must match exactly.
   if (aKind != problem.lhsKind || bKind != problem.rhsKind) continue;


   // [Fix2] Accumulator: allow exact match OR wider accumulator (upcasting).
   // e.g., f16 problem (accKind=0) can use f32 accumulator intrinsic (cKind=2).
   if (cKind != problem.accKind && cKind < problem.accKind) continue;


   intrinsics.push_back({mma.intrinsic, mSize, nSize, kSize,
                         mma.warpSize, aKind, bKind, cKind,
                         mma.distribution});
 }
 if (intrinsics.empty()) return std::nullopt;


 // Classify problem by arithmetic intensity.
 int64_t flops = 2 * problem.mSize * problem.nSize * problem.kSize;
 int64_t bytes = problem.mSize * problem.kSize +
                 problem.kSize * problem.nSize +
                 problem.mSize * problem.nSize;
 int64_t computeIntensity = (bytes > 0) ? flops / bytes : 0;


 GemmCutoffs cutoffs = getGemmCutoffs(target);
 if (computeIntensity <= cutoffs.smallGemmCutoff)
   problem.gemmSize = GemmSize::SmallGemm;
 else if (computeIntensity >= cutoffs.veryLargeGemmCutoff)
   problem.gemmSize = GemmSize::VeryLargeGemm;
 else if (computeIntensity >= cutoffs.largeGemmCutoff)
   problem.gemmSize = GemmSize::LargeGemm;
 else
   problem.gemmSize = GemmSize::MediumGemm;


 LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] This config is "
                          << gemmSizeName(problem.gemmSize)
                          << " (intensity=" << computeIntensity << ")\n");


 std::optional<GPUMMAHeuristicSeeds> maybeSeeds =
     getContractionHeuristicSeeds(problem, isGemm, /*scaled=*/false);
 if (!maybeSeeds) return std::nullopt;


 return deduceMMASchedule(problem, target, intrinsics, *maybeSeeds,
                          mustBeAligned, doCPromotion,
                          fusedInfo, splitReductionTripCnt);
}


//===----------------------------------------------------------------------===//
// tryPickMMASchedule
//===----------------------------------------------------------------------===//


static std::optional<GPUMMASchedule>
tryPickMMASchedule(linalg::LinalgOp op,
                  const NVIDIATargetInfo &target,
                  const MatmulDims &dims,
                  int32_t lhsKind, int32_t rhsKind, int32_t accKind,
                  bool doCPromotion,
                  const FusedOpMemoryInfo &fusedInfo) {
 auto cd = mlir::linalg::inferContractionDims(op);
 bool isGemm = succeeded(cd) && cd->batch.empty();


 ContractProblem problem;
 problem.mSize         = dims.M;
 problem.nSize         = dims.N;
 problem.kSize         = dims.K;
 problem.lhsKind       = lhsKind;
 problem.rhsKind       = rhsKind;
 problem.accKind       = accKind;
 problem.inBitWidth    = (lhsKind == 0 || lhsKind == 1) ? 16 : 32;
 problem.gemmSize      = GemmSize::MediumGemm; // overwritten by getMmaSchedule
 problem.transposedLhs = succeeded(cd) ? isTransposedLhs(op, *cd) : false;
 problem.transposedRhs = succeeded(cd) ? isTransposedRhs(op, *cd) : false;


 // Pass 1: strict alignment — no padding required.
 if (auto sched = getMmaSchedule(target, problem, isGemm,
                                 /*mustBeAligned=*/true, doCPromotion,
                                 fusedInfo))
   return sched;


 // Pass 2: unaligned retry — allows tiles that do not evenly divide M/N/K.
 LLVM_DEBUG(llvm::dbgs()
            << "[nova-kernel-config] Aligned pass failed, "
               "retrying without alignment requirement\n");
 return getMmaSchedule(target, problem, isGemm,
                       /*mustBeAligned=*/false, doCPromotion, fusedInfo);
}


//===----------------------------------------------------------------------===//
// trySetMMAConfig
// [Fix5] threadTile = intrinsic size (one MMA slot per thread);
//         subgroupTile = intrinsic * mnTileCount encodes multi-tile ownership.
//===----------------------------------------------------------------------===//
static LogicalResult trySetMMAConfig(linalg::LinalgOp matmul,
                                     const NVIDIATargetInfo &target,
                                     const MatmulDims &dims,
                                     int numLoops,
                                     bool doCPromotion,
                                     const FusedOpMemoryInfo &fusedInfo) {
  Type lhsType = getElementTypeOrSelf(matmul.getDpsInputOperand(0)->get());
  Type rhsType = getElementTypeOrSelf(matmul.getDpsInputOperand(1)->get());
  Type accType = getElementTypeOrSelf(matmul.getDpsInitOperand(0)->get());

  int32_t lhsKind = typeToElementKind(lhsType);
  int32_t rhsKind = typeToElementKind(rhsType);
  int32_t accKind = typeToElementKind(accType);
  if (lhsKind < 0 || rhsKind < 0 || accKind < 0)
    return failure();

  auto maybeSchedule = tryPickMMASchedule(matmul, target, dims,
                                          lhsKind, rhsKind, accKind,
                                          doCPromotion, fusedInfo);
  if (!maybeSchedule)
    return failure();

  const GPUMMASchedule &sched = *maybeSchedule;
  auto contractionDims = mlir::linalg::inferContractionDims(matmul);

  SmallVector<int64_t> workgroupTiles(numLoops, 0);
  SmallVector<int64_t> reductionTiles(numLoops, 0);
  SmallVector<int64_t> threadTiles(numLoops, 0);
  SmallVector<int64_t> subgroupTiles(numLoops, 0);
  // Per-workgroup subgroup counts (warps within ONE workgroup tile).
  // Read by NovaGPUConfigureTensorLayouts to build correct per-warp layouts.
  // Separate from subgroupTiles (global = numWorkgroups × warpsPerWG).
  SmallVector<int64_t> wgSubgroupTiles(numLoops, 0);

  // ── Batch and leading M/N/K dims: tile by 1 ──────────────────────────────
  for (int64_t b : contractionDims->batch) {
    workgroupTiles[b]   = 1;
    threadTiles[b]      = 1;
    subgroupTiles[b]    = 1;
    wgSubgroupTiles[b]  = 1;
  }
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTiles[m]   = 1;
    threadTiles[m]      = 1;
    subgroupTiles[m]    = 1;
    wgSubgroupTiles[m]  = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTiles[n]   = 1;
    threadTiles[n]      = 1;
    subgroupTiles[n]    = 1;
    wgSubgroupTiles[n]  = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims->k))
    reductionTiles[k] = 1;

  // ── Innermost M, N, K dims ───────────────────────────────────────────────
  int mDim = contractionDims->m.back();
  int nDim = contractionDims->n.back();
  int kDim = contractionDims->k.back();

  workgroupTiles[mDim] = sched.wgM;
  workgroupTiles[nDim] = sched.wgN;
  reductionTiles[kDim] = sched.wgK;

  // threadTiles for MMA M/N = 0: the Subgroup pass owns MMA tiling.
  // The Thread pass checks anyNonZero and skips this op entirely.
  // Copy/fill ops use derived_thread config (separate code path) and are
  // unaffected by this zeroing.
  threadTiles[mDim] = 0;
  threadTiles[nDim] = 0;

  // ── Per-warp tile step ("subgroup" field) ────────────────────────────────
  // = wgTile / warpsPerWG along each axis — what the Subgroup scf.forall
  // iterates by. With a 2-D warp grid (matmul.cpp-style 2x2), both M and N
  // are split across warps; the legacy 1-D split has subgroupCountN == 1.
  subgroupTiles[mDim] = sched.wgM / sched.subgroupCountM;
  subgroupTiles[nDim] = sched.wgN / sched.subgroupCountN;
  subgroupTiles[kDim] = 0;           // K not distributed across subgroups

  for (int64_t b : contractionDims->batch)
    subgroupTiles[b] = workgroupTiles[b]; // one warp owns full batch dim

  // ── Per-workgroup subgroup counts ("wg_subgroup" field) ──────────────────
  // Number of warps along each axis within ONE workgroup tile.
  // NovaGPUConfigureTensorLayouts uses this to compute per-warp
  // batch_counts / sg_counts correctly from the wg-level tile bounds.
  wgSubgroupTiles[mDim] = sched.subgroupCountM;
  wgSubgroupTiles[nDim] = sched.subgroupCountN;
  wgSubgroupTiles[kDim] = 0;                   // K not subgroup-distributed

  for (int64_t b : contractionDims->batch)
    wgSubgroupTiles[b] = 1; // one subgroup per batch dim within WG

  // ── Promoted operands ────────────────────────────────────────────────────
  // C (operand 2) goes to SMEM only when it carries live data from a prior
  // kernel. Each warp owns a disjoint subslice of C and accumulates in registers,
  // so epilogue users are not a reason to stage C through SMEM.
  SmallVector<int64_t> promotedOps = {0, 1};
  if (doCPromotion)
    promotedOps.push_back(2);

  // ── Padding ──────────────────────────────────────────────────────────────
  SmallVector<int64_t> padding =
      computePaddingSizes(matmul, workgroupTiles, reductionTiles);

  // ── Write config ─────────────────────────────────────────────────────────
  MLIRContext *ctx = matmul.getContext();
  setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                               workgroupTiles, reductionTiles,
                               threadTiles, subgroupTiles,
                               static_cast<int32_t>(sched.intrinsic.intrinsic),
                               promotedOps, padding,
                               wgSubgroupTiles);

  LLVM_DEBUG(llvm::dbgs()
             << "[nova-kernel-config] MMA config:"
             << " wgM=" << sched.wgM
             << " wgN=" << sched.wgN
             << " wgK=" << sched.wgK
             << " warpGrid=" << sched.subgroupCountM
             << "x" << sched.subgroupCountN
             << " mTilePerWarp=" << sched.mTileCountPerSubgroup
             << " nTilePerWarp=" << sched.nTileCountPerSubgroup
             << " sgStepM=" << subgroupTiles[mDim]
             << " sgStepN=" << subgroupTiles[nDim]
             << " wgSgM=" << wgSubgroupTiles[mDim]
             << " wgSgN=" << wgSubgroupTiles[nDim]
             << " doCPromo=" << doCPromotion
             << " fusedOverhead="
             << fusedInfo.extraSmemBytes(sched.wgM, sched.wgK) << "\n");
  return success();
}

//===----------------------------------------------------------------------===//
// setSimtConfig — SIMT fallback, fused-op-aware [Fix6]
//
// After picking a tile from kSimtTable, checks whether the A+B tiles plus
// the fused-op overhead fit within maxSmem. If not, tileM is halved until
// they fit (or we hit the minimum entry).
//===----------------------------------------------------------------------===//
static LogicalResult setSimtConfig(linalg::LinalgOp matmul,
                                  const MatmulDims &dims,
                                  int numLoops,
                                  const NVIDIATargetInfo &target,
                                  const FusedOpMemoryInfo &fusedInfo) {
 // Phase 1: find the best-aligned tile from the table.
 const SimtTilePair *chosen = nullptr;
 for (const auto &entry : kSimtTable) {
   if (dims.M % entry.tileMNK[0] == 0 &&
       dims.N % entry.tileMNK[1] == 0 &&
       dims.K % entry.tileMNK[2] == 0) {
     chosen = &entry;
     break;
   }
 }
 if (!chosen) {
   for (const auto &entry : kSimtTable) {
     if (dims.M % entry.tileMNK[0] == 0 &&
         dims.N % entry.tileMNK[1] == 0) {
       chosen = &entry;
       break;
     }
   }
 }
 if (!chosen) chosen = &kSimtTable[0];


 int64_t tileM = chosen->tileMNK[0];
 int64_t tileN = chosen->tileMNK[1];
 int64_t tileK = chosen->tileMNK[2];
 int64_t wgX   = chosen->workgroup[0];
 int64_t wgY   = chosen->workgroup[1];


 // [Fix6] If layernorm forces the full K, override tileK.
 if (fusedInfo.forcedTileK >= 0)
   tileK = fusedInfo.forcedTileK;
 else
   while (tileK > 1 && dims.K % tileK != 0) tileK >>= 1;


 // [Fix-P1-4] Target at most half of maxSmem per block so at least 2 blocks
 // fit per SM, doubling warp slots available for latency hiding (occupancy
 // goes from ~17% to ~33% on sm_86 with 48KB shared memory).
 constexpr int32_t kF32Kind = 2;
 int64_t maxSmem = target.maxWorkgroupMemBytes / 2;

 // [Fix6] Phase 2: shrink tileM (then tileK) until A+B+fused overhead fits
 // maxSmem. Use f32 (4 bytes/elem) conservatively.
 while (tileM > 1) {
   int64_t smem = computeSharedMemory(tileM, tileN, tileK,
                                      kF32Kind, kF32Kind,
                                      /*doCPromotion=*/false, fusedInfo);
   if (smem <= maxSmem) break;
   tileM /= 2;
   wgX = std::max<int64_t>(1, wgX / 2);
   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] SIMT smem too large (" << smem
              << " > " << maxSmem << "), shrinking tileM to " << tileM
              << "\n");
 }
 // If tileM is already 1 and smem is still over budget, halve tileK.
 // This keeps the occupancy budget (maxSmem/2) respected even for wide K.
 if (fusedInfo.forcedTileK < 0) {
   while (tileK > 1) {
     int64_t smem = computeSharedMemory(tileM, tileN, tileK,
                                        kF32Kind, kF32Kind,
                                        /*doCPromotion=*/false, fusedInfo);
     if (smem <= maxSmem) break;
     tileK /= 2;
     LLVM_DEBUG(llvm::dbgs()
                << "[nova-kernel-config] SIMT smem still too large (" << smem
                << " > " << maxSmem << "), shrinking tileK to " << tileK
                << "\n");
   }
 }

 // [Fix-P0-1] Cap per-thread output tile to 4×4 = 16 accumulators maximum.
 // The SIMT table entries can produce threadTileM=8, threadTileN=8 (64
 // accumulators) for shapes like [128,64] which overflows the 255-register
 // budget and spills 24KB of local memory per thread, destroying performance.
 // 4×4 stays well within 64 live registers and eliminates all local mem spill.
 static constexpr int64_t kMaxThreadTile = 4;
 int64_t threadTileM = std::min(kMaxThreadTile,
                                std::max<int64_t>(1, tileM / wgX));
 int64_t threadTileN = std::min(kMaxThreadTile,
                                std::max<int64_t>(1, tileN / wgY));


 auto contractionDims = mlir::linalg::inferContractionDims(matmul);


 SmallVector<int64_t> workgroupTiles(numLoops, 0);
 SmallVector<int64_t> reductionTiles(numLoops, 0);
 SmallVector<int64_t> threadTiles(numLoops, 0);
 SmallVector<int64_t> subgroupTiles(numLoops, 0);


 for (int64_t b : contractionDims->batch) {
   workgroupTiles[b] = 1; threadTiles[b] = 1;
 }
 for (int64_t m : llvm::drop_end(contractionDims->m)) {
   workgroupTiles[m] = 1; threadTiles[m] = 1;
 }
 for (int64_t n : llvm::drop_end(contractionDims->n)) {
   workgroupTiles[n] = 1; threadTiles[n] = 1;
 }
 for (int64_t k : llvm::drop_end(contractionDims->k))
   reductionTiles[k] = 1;


 workgroupTiles[contractionDims->m.back()] = tileM;
 workgroupTiles[contractionDims->n.back()] = tileN;
 reductionTiles[contractionDims->k.back()] = tileK;
 threadTiles[contractionDims->m.back()]    = threadTileM;
 threadTiles[contractionDims->n.back()]    = threadTileN;


 LLVM_DEBUG(llvm::dbgs()
            << "[nova-kernel-config] SIMT config: tileM=" << tileM
            << " tileN=" << tileN << " tileK=" << tileK
            << " threadM=" << threadTileM << " threadN=" << threadTileN
            << " fusedOpOverhead="
            << fusedInfo.extraSmemBytes(tileM, tileK) << "\n");


 SmallVector<int64_t> padding =
     computePaddingSizes(matmul, workgroupTiles, reductionTiles);


 MLIRContext *ctx = matmul.getContext();
 setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                              workgroupTiles, reductionTiles,
                              threadTiles, subgroupTiles,
                              static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                              /*promotedOperands=*/{0, 1}, padding);
 return success();
}


//===----------------------------------------------------------------------===//
// setContractConfig — Public API for contractions
//===----------------------------------------------------------------------===//


LogicalResult setContractConfig(linalg::LinalgOp op,
                               const NVIDIATargetInfo &target) {
 if (!linalg::isaContractionOpInterface(op)) return failure();
 if (op.getNumParallelLoops() < 2)            return failure();


 // Reject ops where ALL indexing maps are permutations (pure permutation ops
 // have no contraction structure and should go through the reduction pipeline).
 // Using any_of was wrong: standard linalg.matmul has a RHS map that is a
 // permutation of its two operand dims, which would incorrectly reject it.
 if (llvm::all_of(op.getIndexingMapsArray(),
                  [](AffineMap m) { return m.isPermutation(); }))
   return failure();


 MatmulDims dims = inferMatmulDims(op);
 if (!dims.valid())          return failure();
 if (dims.M == 1 || dims.N == 1) return failure();


 // Send very skinny matmuls to the vector reduction pipeline.
 FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
     mlir::linalg::inferContractionDims(op);
 assert(succeeded(contractionDims) && "Could not infer contraction dims");
 if (llvm::all_equal({contractionDims->m.size(), contractionDims->n.size(),
                      contractionDims->k.size(), size_t{1}}) &&
     contractionDims->batch.empty()) {
   int64_t preferredSubgroupSize = target.preferredSubgroupSize;
   if ((dims.M <= kVerySkinnyDimThreshold &&
        (dims.N > preferredSubgroupSize || ShapedType::isDynamic(dims.N))) ||
       (dims.N <= kVerySkinnyDimThreshold &&
        (dims.M > preferredSubgroupSize || ShapedType::isDynamic(dims.M))))
     return failure();
 }


 FusedOpMemoryInfo fusedInfo = analyzeFusedLeadingOp(op, dims);
 // C-promotion is never needed for contraction ops: each warp accumulates
 // into registers and writes directly to global memory in the epilogue.
 // The promotion pass (NovaGPUPromoteMatmulOperands) already guards against
 // this — it silently skips result operands for isaContractionOpInterface ops
 // (see promoteResultToShared guard). Setting doCPromotion=true only adds
 // wgM*wgN*4 bytes to the SMEM budget, shrinking the tile for no benefit.
 bool doCPromotion = false;


 int numLoops = op.getNumLoops();


 // Tiny GEMM fast path: M*N fits in one warp, no MMA needed.
 if (dims.M * dims.N <= kWarpSize) {
   SmallVector<int64_t> workgroupTiles(numLoops, 0);
   SmallVector<int64_t> reductionTiles(numLoops, 0);
   SmallVector<int64_t> threadTiles(numLoops, 0);
   SmallVector<int64_t> subgroupTiles(numLoops, 0);


   workgroupTiles[contractionDims->m.back()] = dims.M;
   workgroupTiles[contractionDims->n.back()] = dims.N;
   reductionTiles[contractionDims->k.back()] = 4;
   threadTiles[contractionDims->m.back()]    = 1;
   threadTiles[contractionDims->n.back()]    = 1;
   for (int64_t b : contractionDims->batch) {
     workgroupTiles[b] = 1;
     threadTiles[b]    = 1;
   }


   SmallVector<int64_t> padding =
       computePaddingSizes(op, workgroupTiles, reductionTiles);
   MLIRContext *ctx = op.getContext();
   setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                                workgroupTiles, reductionTiles,
                                threadTiles, subgroupTiles,
                                static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                                /*promotedOperands=*/{}, padding);
   return success();
 }


 // MMA path.
 if (!target.mmaIntrinsics.empty()) {
   if (succeeded(trySetMMAConfig(op, target, dims, numLoops,
                                 doCPromotion, fusedInfo)))
     return success();
 }


 // SIMT fallback — pass target + fusedInfo so smem budget is respected.
 return setSimtConfig(op, dims, numLoops, target, fusedInfo);
}


//===----------------------------------------------------------------------===//
// setDefaultConfig — For reductions and elementwise ops
//===----------------------------------------------------------------------===//


LogicalResult setDefaultConfig(linalg::LinalgOp op,
                              const NVIDIATargetInfo &target) {
 int numLoops = op.getNumLoops();
 SmallVector<int64_t> loopBounds = op.getStaticLoopRanges();
 if (loopBounds.size() != static_cast<size_t>(numLoops))
   return failure();
 for (int64_t b : loopBounds)
   if (ShapedType::isDynamic(b)) return failure();


 auto iterTypes = op.getIteratorTypesArray();
 SmallVector<unsigned> parallelDims, reductionDims;
 for (int i = 0; i < numLoops; ++i) {
   if (linalg::isParallelIterator(iterTypes[i]))
     parallelDims.push_back(i);
   else if (linalg::isReductionIterator(iterTypes[i]))
     reductionDims.push_back(i);
 }


 SmallVector<int64_t> workgroupTiles(numLoops, 0);
 SmallVector<int64_t> threadTiles(numLoops, 0);
 SmallVector<int64_t> reductionTiles(numLoops, 0);
 SmallVector<int64_t> subgroupTiles(numLoops, 0);


 constexpr int64_t kWorkgroupThreads = kTargetThreadsPerBlock;
 constexpr int64_t kVectorSize = 4;


 if (parallelDims.empty() && reductionDims.empty()) {
   // Rank-0 scalar op — nothing to tile.


 } else if (parallelDims.empty() && !reductionDims.empty()) {
   // Full reduction: split best dim across one warp (32 threads).
   constexpr int64_t kWarpThreads = 32;
   unsigned bestRedDim = reductionDims[0];
   int64_t bestNumThreads = 0;
   int64_t bestThreadTile = INT64_MAX;
   for (unsigned dim : reductionDims) {
     int64_t bound = loopBounds[dim];
     int64_t nt = std::min(kWarpThreads, bound);
     while (nt > 1 && bound % nt != 0) nt /= 2;
     int64_t tt = (nt > 1) ? bound / nt : bound;
     if (nt > bestNumThreads ||
         (nt == bestNumThreads && tt < bestThreadTile)) {
       bestRedDim    = dim;
       bestNumThreads = nt;
       bestThreadTile = tt;
     }
   }
   if (bestNumThreads > 1) {
     threadTiles[bestRedDim]    = bestThreadTile;
     workgroupTiles[bestRedDim] = loopBounds[bestRedDim];
   } else {
     int64_t bound = loopBounds[bestRedDim];
     reductionTiles[bestRedDim] = (bound % 4 == 0) ? 4 :
                                  (bound % 2 == 0) ? 2 : 1;
   }
   for (unsigned dim : reductionDims) {
     if (dim == bestRedDim) continue;
     int64_t bound = loopBounds[dim];
     reductionTiles[dim] = (bound % 4 == 0) ? 4 :
                           (bound % 2 == 0) ? 2 : 1;
   }


 } else if (!reductionDims.empty() && parallelDims.size() >= 2 &&
            reductionDims.back() == (unsigned)(numLoops - 1)) {
    // Row-reduction: tile inner parallel dim; reduction is sequential.
    unsigned innerParallelDim = parallelDims.back();
    int64_t innerSize = loopBounds[innerParallelDim];
    int64_t innerTile = std::min(innerSize, (int64_t)kWorkgroupThreads);
    while (innerTile > 1 && innerSize % innerTile != 0) innerTile /= 2;
    workgroupTiles[innerParallelDim] = innerTile;
    threadTiles[innerParallelDim]   = 1;
    // Distribute remaining thread budget to outer parallel dims (inward→outward),
    // mirroring the general-purpose else branch below.  Without this, a
    // [par, par, red] op with innerTile=64 only uses 64/256 threads.
    {
      int64_t remainingBudget = kWorkgroupThreads / innerTile;
      for (int i = (int)parallelDims.size() - 2; i >= 0; --i) {
        unsigned dim = parallelDims[i];
        int64_t dimSize = loopBounds[dim];
        if (remainingBudget > 1 && dimSize > 1) {
          int64_t tile = std::min(dimSize, remainingBudget);
          while (tile > 1 && dimSize % tile != 0) tile /= 2;
          workgroupTiles[dim] = tile;
          threadTiles[dim]    = 1;
          remainingBudget    /= tile;
        } else {
          workgroupTiles[dim] = 1;
          threadTiles[dim]    = 0;
        }
      }
    }
    // DO NOT set threadTiles on reduction dims.
    for (unsigned dim : reductionDims) {
      int64_t bound = loopBounds[dim];
      reductionTiles[dim] = (bound % 4 == 0) ? 4 :
                            (bound % 2 == 0) ? 2 : 1;
    }

  } else {
   // Has parallel dims — distribute across workgroup.
   unsigned innerParallelDim = parallelDims.back();
   int64_t innerSize = loopBounds[innerParallelDim];


   // Two-pass: prefer vectorSize >= 2 for meaningful thread tiles.
   int64_t bestThreads = 0, bestVS = 1;
   for (int64_t vs = kVectorSize; vs >= 2; vs /= 2) {
     for (int64_t thr = kWorkgroupThreads; thr >= kWarpSize; thr /= 2) {
       int64_t tile = thr * vs;
       if (tile <= innerSize && innerSize % tile == 0) {
         if (thr > bestThreads || (thr == bestThreads && vs > bestVS)) {
           bestThreads = thr; bestVS = vs;
         }
         break;
       }
     }
   }
   if (bestThreads == 0) {
     for (int64_t thr = kWorkgroupThreads; thr >= kWarpSize; thr /= 2) {
       if (thr <= innerSize && innerSize % thr == 0) {
         bestThreads = thr; bestVS = 1; break;
       }
     }
   }
   if (bestThreads == 0) {
     bestVS = kVectorSize;
     while (bestVS > 1 && innerSize % bestVS != 0) bestVS /= 2;
     bestThreads = std::min(innerSize / bestVS, kWorkgroupThreads);
   }


   int64_t innerTile = std::min(bestThreads * bestVS, innerSize);
   workgroupTiles[innerParallelDim] = innerTile;
   threadTiles[innerParallelDim]    = bestVS;


   int64_t remainingBudget = kWorkgroupThreads / bestThreads;
   for (int i = (int)parallelDims.size() - 2; i >= 0; --i) {
     unsigned dim = parallelDims[i];
     int64_t dimSize = loopBounds[dim];
     if (remainingBudget > 1 && dimSize > 1) {
       int64_t tile = std::min(dimSize, remainingBudget);
       while (tile > 1 && dimSize % tile != 0) tile /= 2;
       workgroupTiles[dim] = tile;
       threadTiles[dim]    = 1;
       remainingBudget    /= tile;
     } else {
       workgroupTiles[dim] = 1;
       threadTiles[dim]    = 0;
     }
   }


   for (unsigned dim : reductionDims) {
     int64_t bound = loopBounds[dim];
     reductionTiles[dim] = (bound % 4 == 0) ? 4 :
                           (bound % 2 == 0) ? 2 : 1;
   }


   clampThreadTilesToMaxThreads(workgroupTiles, threadTiles);
 }


 LLVM_DEBUG({
   llvm::dbgs() << "[nova-kernel-config] Default config for "
                << op->getName() << ": workgroup=[";
   llvm::interleaveComma(workgroupTiles, llvm::dbgs());
   llvm::dbgs() << "] reduction=[";
   llvm::interleaveComma(reductionTiles, llvm::dbgs());
   llvm::dbgs() << "] thread=[";
   llvm::interleaveComma(threadTiles, llvm::dbgs());
   llvm::dbgs() << "]\n";
 });


 SmallVector<int64_t> padding =
     computePaddingSizes(op, workgroupTiles, reductionTiles);


 // Promote input operands that need tiled staging in workgroup shared memory.
 //
 // Budget: use at most half of the target's shared memory per block (matching
 // the occupancy constraint in setSimtConfig). Divide by the element size
 // derived from the op's element type so this works for f16/bf16/f32/f64.
 //
 // Decision per operand (inputs only, in index order):
 //   1. Broadcast inputs (map rank < loop count) — always promote; they are
 //      small by definition and must be staged for correctness.
 //   2. Full-rank inputs — compute the tile footprint from workgroup/reduction
 //      tiles. Promote if the cumulative promoted bytes stay within budget.
 //      Skip operands whose map contains non-dim expressions (e.g. constants
 //      or affine offsets) — those can't be straightforwardly staged.
 SmallVector<int64_t> promotedOps;
 auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
 auto maps = linalgOp.getIndexingMapsArray();
 unsigned numInputs = linalgOp.getNumDpsInputs();

 // Derive element size in bytes from the op's element type.
 int64_t elemBytes = 4; // default f32
 if (auto shaped = dyn_cast<ShapedType>(op->getOperand(0).getType()))
   if (auto floatTy = dyn_cast<FloatType>(shaped.getElementType()))
     elemBytes = (floatTy.getWidth() + 7) / 8;

 // Half of the target SMEM budget — same fraction setSimtConfig uses.
 const int64_t smemBudget = target.maxWorkgroupMemBytes / 2;
 int64_t smemUsed = 0;

 // Tile footprint for one operand. Returns -1 if the map contains non-dim
 // expressions (non-trivial affine), signalling that promotion is unsafe.
 auto getTileBytes = [&](AffineMap map) -> int64_t {
   int64_t vol = 1;
   for (unsigned r = 0; r < map.getNumResults(); ++r) {
     auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(r));
     if (!dimExpr)
       return -1; // non-trivial map — cannot stage
     unsigned d = dimExpr.getPosition();
     int64_t t = workgroupTiles[d] > 0 ? workgroupTiles[d]
               : reductionTiles[d] > 0 ? reductionTiles[d] : 1;
     vol *= t;
   }
   return vol * elemBytes;
 };
 bool isFullReduction = parallelDims.empty() && !reductionDims.empty();
 for (unsigned i = 0; i < numInputs; ++i) {
  if (isFullReduction)
     break; // no operand promotion for full reductions
   if (maps[i].getNumResults() < (unsigned)numLoops) {
     // Broadcast input — always promote regardless of budget.
     promotedOps.push_back(i);
   } else {
     int64_t tileBytes = getTileBytes(maps[i]);
     if (tileBytes > 0 && smemUsed + tileBytes <= smemBudget) {
       promotedOps.push_back(i);
       smemUsed += tileBytes;
     }
   }
 }


 MLIRContext *ctx = op.getContext();
 setMatmulLoweringConfigAttrs(op.getOperation(), ctx,
                              workgroupTiles, reductionTiles,
                              threadTiles, subgroupTiles,
                              static_cast<int32_t>(NVMMAIntrinsicValues::NONE),
                              promotedOps, padding);
 return success();
}


//===----------------------------------------------------------------------===//
// Root / prologue / epilogue classification helpers
//===----------------------------------------------------------------------===//


static bool isTrueContraction(Operation *op) {
 if (isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::MatvecOp,
         linalg::VecmatOp, linalg::BatchMatvecOp>(op))
   return true;
 auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
 if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) return false;
 auto indexMaps = linalgOp.getIndexingMapsArray();
 if (indexMaps.size() < 3) return false;
 auto iterTypes = linalgOp.getIteratorTypesArray();
 unsigned numLoops = iterTypes.size();
 for (unsigned i = 0; i < numLoops; ++i) {
   if (iterTypes[i] != utils::IteratorType::parallel) continue;
   bool inInput0 = indexMaps[0].isFunctionOfDim(i);
   bool inInput1 = indexMaps[1].isFunctionOfDim(i);
   if (inInput0 != inInput1) return true;
 }
 return false;
}


static bool isRootOp(Operation *op) {
 auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
 if (!linalgOp) return false;
 if (isTrueContraction(op)) return true;
 for (auto iterType : linalgOp.getIteratorTypesArray())
   if (iterType != utils::IteratorType::parallel) return true;
 return false;
}


static bool isEpilogue(Operation *op) {
 auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
 if (!linalgOp) return false;
 for (auto iterType : linalgOp.getIteratorTypesArray())
   if (iterType != utils::IteratorType::parallel) return false;
 // All indexing maps must be projected permutations for safe consumer fusion.
 for (AffineMap map : linalgOp.getIndexingMapsArray())
   if (!map.isProjectedPermutation()) return false;
 // Op's operand must trace to a root (up to 2 levels deep).
 for (Value operand : op->getOperands()) {
   Operation *defOp = operand.getDefiningOp();
   if (!defOp) continue;
   if (isRootOp(defOp)) return true;
   for (Value innerOp : defOp->getOperands()) {
     Operation *innerDef = innerOp.getDefiningOp();
     if (innerDef && isRootOp(innerDef)) return true;
   }
 }
 return false;
}


static bool isPrologue(Operation *op) {
 auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
 if (!linalgOp) return false;
 for (auto iterType : linalgOp.getIteratorTypesArray())
   if (iterType != utils::IteratorType::parallel) return false;
 for (OpResult result : op->getResults()) {
   for (Operation *user : result.getUsers()) {
     if (!isRootOp(user)) continue;
     auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user);
     if (!consumerLinalgOp) continue;
     for (OpOperand &operand : consumerLinalgOp->getOpOperands()) {
       if (operand.get() != result) continue;
       AffineMap map = consumerLinalgOp.getMatchingIndexingMap(&operand);
       if (map.isProjectedPermutation() && !map.isEmpty()) return true;
       break;
     }
   }
 }
 return false;
}


//===----------------------------------------------------------------------===//
// initNovaGPULaunchConfig — Main entry point
//===----------------------------------------------------------------------===//


void initNovaGPULaunchConfig(mlir::func::FuncOp funcOp,
                             const NVIDIATargetInfo &target) {
 funcOp.walk([&](linalg::LinalgOp op) {
   if (getLoweringConfig(op.getOperation())) return;
   if (isa<linalg::FillOp>(op.getOperation())) return;


   if (isEpilogue(op.getOperation())) {
     LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping epilogue: "
                              << op->getName() << "\n");
     return;
   }
   if (isPrologue(op.getOperation())) {
     LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping prologue: "
                              << op->getName() << "\n");
     return;
   }


   if (linalg::isaContractionOpInterface(op)) {
     if (succeeded(setContractConfig(op, target))) return;
   }


   (void)setDefaultConfig(op, target);
 });
}


} // namespace mlir::nova
