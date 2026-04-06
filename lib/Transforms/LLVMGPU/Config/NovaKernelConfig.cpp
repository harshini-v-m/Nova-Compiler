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
//      C promotion = true when accumulator buffer is live (matmul_accumulate)
//                    OR when prologue ops feed non-matmul operands into the
//                    kernel (promotePrologueOperands).
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
//   [Fix1] promotePrologueOperands now propagated to setMatmulLoweringConfigAttrs
//          instead of being computed and silently discarded.
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
//          Simple elementwise ops (relu, bias_add, gelu) are handled directly.
//          Row-reduction ops (layernorm) force tileK == K (full reduction dim)
//          and reserve per-row mean+variance scratch space.
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
 int64_t subgroupCount;
 int64_t mnTileCountPerSubgroup;
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
// checkUserHaveNewOperands / readingExistingAccBuffer
//===----------------------------------------------------------------------===//


bool checkUserHaveNewOperands(linalg::LinalgOp op) {
 // Returns true when any user of any result of `op` is itself a LinalgOp.
 // Fixed: removed the dead inner loop that dereferenced a null linalgop
 // after a failed cast.
 for (auto result : op->getResults())
   for (Operation *user : result.getUsers())
     if (dyn_cast<linalg::LinalgOp>(user))
       return true;
 return false;
}


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
// [Fix6] analyzeFusedLeadingOp
//
// Walks producers of the matmul's LHS (operand 0). Classifies what kind of
// fused leading op is present and returns the corresponding memory info.
//
// Currently handled:
//   - nova.layer_norm  → row-reduction, forces tileK = fullK,
//                        reserves 8 bytes/row mean+var scratch
//   - nova.gelu        → elementwise, extra f32 input tile copy
//   - nova.add (bias)  → elementwise, extra f32 input tile copy
//   - anything else    → treated as generic elementwise (conservative)
//
// Only looks one level up the def chain. Deeper chains can be added later.
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
   return GPUMMAHeuristicSeeds{4, 32, 4, 2,
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
 // A tile + B tile, double-buffered
 int64_t smem = 2 * (wgM * wgK * lhsBytes + wgK * wgN * rhsBytes);
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


 return GPUMMASchedule{intrinsic, wgM, wgN, wgK,
                       subgroupCount, mnTileCount,
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


   int64_t newMNT = sched.mnTileCountPerSubgroup / 2;
   if (newMNT < 1) return std::nullopt;


   sched.mnTileCountPerSubgroup = newMNT;
   sched.wgM = std::max(intrinsic.mSize * newMNT * sched.subgroupCount,
                        intrinsic.mSize);
   sched.wgN = std::max(intrinsic.nSize * newMNT, intrinsic.nSize);


   LLVM_DEBUG(llvm::dbgs()
              << "[nova-kernel-config] fitScheduleInSharedMemory: smem="
              << smem << " > limit=" << maxSmem
              << " (incl. fused-op overhead), shrinking MNT to " << newMNT
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


   // [Fix6] Pass fusedInfo so fitScheduleInSharedMemory uses the real budget.
   std::optional<GPUMMASchedule> fitted =
       fitScheduleInSharedMemory(sched, intrinsic,
                                 target.maxWorkgroupMemBytes,
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
// setContractConfig (MMA path)
// [Fix1] promotePrologueOperands now propagated to promoted operand list.
// [Fix5] threadTile = intrinsic size (one MMA slot per thread);
//         subgroupTile = intrinsic * mnTileCount encodes multi-tile ownership.
//===----------------------------------------------------------------------===//


static LogicalResult trySetMMAConfig(linalg::LinalgOp matmul,
                                    const NVIDIATargetInfo &target,
                                    const MatmulDims &dims,
                                    int numLoops,
                                    bool doCPromotion,
                                    bool promotePrologueOperands,
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


 for (int64_t b : contractionDims->batch) {
   workgroupTiles[b] = 1; threadTiles[b] = 1; subgroupTiles[b] = 1;
 }
 for (int64_t m : llvm::drop_end(contractionDims->m)) {
   workgroupTiles[m] = 1; threadTiles[m] = 1; subgroupTiles[m] = 1;
 }
 for (int64_t n : llvm::drop_end(contractionDims->n)) {
   workgroupTiles[n] = 1; threadTiles[n] = 1; subgroupTiles[n] = 1;
 }
 for (int64_t k : llvm::drop_end(contractionDims->k))
   reductionTiles[k] = 1;


 workgroupTiles[contractionDims->m.back()] = sched.wgM;
 workgroupTiles[contractionDims->n.back()] = sched.wgN;
 reductionTiles[contractionDims->k.back()] = sched.wgK;


 // [Fix5] threadTile = one MMA instruction shape.
 // The lowering backend derives per-warp multi-tile ownership from subgroupTile.
 threadTiles[contractionDims->m.back()] = sched.intrinsic.mSize;
 threadTiles[contractionDims->n.back()] = sched.intrinsic.nSize;


 // subgroupTile = intrinsic × mnTileCount encodes how many MMA tiles each
 // subgroup (warp) owns. The backend pipelines these for register-level ILP.
 subgroupTiles[contractionDims->m.back()] =
     sched.intrinsic.mSize * sched.mnTileCountPerSubgroup;
 subgroupTiles[contractionDims->n.back()] =
     sched.intrinsic.nSize * sched.mnTileCountPerSubgroup;


 SmallVector<int64_t> padding =
     computePaddingSizes(matmul, workgroupTiles, reductionTiles);


 // [Fix1] Build promoted operand list.
 // Always promote A and B (operands 0 and 1) into shared memory.
 // Also promote C (operand 2 / init) when C promotion is needed due to:
 //   - live accumulator buffer (doCPromotion / matmul_accumulate), OR
 //   - prologue ops that feed non-matmul data into the kernel.
 SmallVector<int64_t> promotedOps = {0, 1};
 if (doCPromotion || promotePrologueOperands)
   promotedOps.push_back(2);


 MLIRContext *ctx = matmul.getContext();
 setMatmulLoweringConfigAttrs(matmul.getOperation(), ctx,
                              workgroupTiles, reductionTiles,
                              threadTiles, subgroupTiles,
                              static_cast<int32_t>(sched.intrinsic.intrinsic),
                              promotedOps, padding);


 LLVM_DEBUG(llvm::dbgs()
            << "[nova-kernel-config] MMA config: wgM=" << sched.wgM
            << " wgN=" << sched.wgN << " wgK=" << sched.wgK
            << " subgroups=" << sched.subgroupCount
            << " mnTile=" << sched.mnTileCountPerSubgroup
            << " doCPromo=" << doCPromotion
            << " promoProlog=" << promotePrologueOperands
            << " fusedOpOverhead="
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


 // [Fix6] Phase 2: shrink tileM until A+B+fused overhead fits maxSmem.
 // Use f32 (4 bytes/elem) conservatively since type info is not threaded here.
 constexpr int32_t kF32Kind = 2;
 int64_t maxSmem = target.maxWorkgroupMemBytes;
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


 int64_t threadTileM = std::max<int64_t>(1, tileM / wgX);
 int64_t threadTileN = std::max<int64_t>(1, tileN / wgY);


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


 // Reject ops where all indexing maps are permutations (no broadcast).
 // Those should go through the reduction pipeline, not contract.
 if (llvm::any_of(op.getIndexingMapsArray(),
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


 // [Fix6] Analyse fused leading op and derive shared memory requirements.
 // Simple elementwise ops (gelu, bias_add) are handled inline.
 // Row-reduction ops (layernorm) force tileK = fullK and reserve scratch.
 FusedOpMemoryInfo fusedInfo = analyzeFusedLeadingOp(op, dims);


 // [Fix1] Compute both promotion flags here and pass them downstream.
 bool promotePrologueOperands = checkUserHaveNewOperands(op);
 bool doCPromotion             = readingExistingAccBuffer(op);


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
   // [Fix1] Pass both flags into trySetMMAConfig.
   if (succeeded(trySetMMAConfig(op, target, dims, numLoops,
                                 doCPromotion, promotePrologueOperands,
                                 fusedInfo)))
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
   threadTiles[innerParallelDim]    = 1;
   for (unsigned dim : parallelDims)
     if (dim != innerParallelDim)
       workgroupTiles[dim] = 1;
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


 // Promote input operands whose indexing maps have fewer dims than the loop
 // count (i.e., they are broadcast inputs that need tiled staging).
 SmallVector<int64_t> promotedOps;
 auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
 auto maps = linalgOp.getIndexingMapsArray();
 unsigned numInputs = linalgOp.getNumDpsInputs();
 for (unsigned i = 0; i < numInputs; ++i)
   if (maps[i].getNumResults() < (unsigned)numLoops)
     promotedOps.push_back(i);


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

