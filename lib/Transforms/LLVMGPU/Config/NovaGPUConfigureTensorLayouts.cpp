//===- NovaGPUConfigureTensorLayouts.cpp ----------------------------------===//
//
// Computes per-operand thread distribution layouts for MMA contraction ops
// and attaches them as DictionaryAttr attributes directly on the linalg op.
//
// Replaces IREE's LLVMGPUConfigureTensorLayoutsPass which requires the
// VectorExt dialect (ToLayoutOp / NestedLayoutAttr). This pass achieves
// the same result by storing layout info as plain DictionaryAttr under
// the keys "nova.layout_0", "nova.layout_1", "nova.layout_2".
//
// Layout attribute format (DictionaryAttr) — per-operand, post-projection:
//   sg_counts     : [i64] — per-WG subgroup counts (warps per dim in WG tile)
//   batch_counts  : [i64] — MMA tiles each subgroup processes per dim
//   outer_counts  : [i64] — outer unroll factor within one MMA tile (usually 1)
//   thread_counts : [i64] — threads within one MMA tile per dim
//   elem_counts   : [i64] — elements each thread owns per dim
//   sg_strides    : [i64] — row-major strides to linearise the subgroup index
//   thread_strides: [i64] — row-major strides to linearise the thread index
//   shared_mem    : bool  — true if this operand is in LDS / workgroup memory
//
// Key strings must stay in sync with those in NovaVectorLayoutAttr.cpp
// (toAttr / fromAttr).  Both files use the same "sg_counts", "batch_counts",
// etc. keys; there is intentionally no shared header constant so the layout
// attribute helper stays self-contained.
//
// These attributes are read by:
//   1. NovaGPUGenericVectorization  — propagates them onto vector.contract
//   2. NovaGPUUnrollToIntrinsics    — uses batch_counts to determine unroll
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-configure-tensor-layouts"

using namespace mlir;
using namespace mlir::nova;

namespace {

//===----------------------------------------------------------------------===//
// §1  Pure utility helpers
//     (no hardware knowledge, no config reads — depend on nothing above)
//===----------------------------------------------------------------------===//

/// Compute row-major strides from a counts vector.
/// A dimension with count == 1 gets stride 0 (it contributes nothing to the
/// flat index).  Otherwise strides are assigned innermost-first.
///
/// Example: counts = [1, 4, 1, 1]  →  strides = [0, 1, 0, 0]
static SmallVector<int64_t> computeStrides(ArrayRef<int64_t> counts) {
  int n = counts.size();
  SmallVector<int64_t> strides(n, 0);
  int64_t stride = 1;
  for (int i = n - 1; i >= 0; i--) {
    if (counts[i] > 1)
      strides[i] = stride;
    stride *= counts[i];
  }
  return strides;
}

/// Project a full-rank array through an AffineMap.
/// For each result dimension i of `map`, if the corresponding AffineExpr is a
/// plain AffineDimExpr(j), copy fullRankVec[j] into projected[i]; otherwise
/// leave the defaultVal.  Only simple dim-permutation / projection maps are
/// supported (no arithmetic expressions in results).
static void projectThroughMap(AffineMap map,
                               ArrayRef<int64_t> fullRankVec,
                               SmallVectorImpl<int64_t> &projected,
                               int64_t defaultVal = 1) {
  int resultRank = map.getNumResults();
  projected.assign(resultRank, defaultVal);
  for (int i = 0; i < resultRank; i++) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i))) {
      int src = dimExpr.getPosition();
      if (src < (int)fullRankVec.size())
        projected[i] = fullRankVec[src];
    }
  }
}

/// Build the DictionaryAttr that represents one operand's vector layout.
/// Field order is alphabetical so MLIR's DictionaryAttr sorts them
/// deterministically when printing.
static DictionaryAttr buildLayoutAttr(MLIRContext *ctx,
                                       ArrayRef<int64_t> sgCounts,
                                       ArrayRef<int64_t> batchCounts,
                                       ArrayRef<int64_t> outerCounts,
                                       ArrayRef<int64_t> threadCounts,
                                       ArrayRef<int64_t> elemCounts,
                                       ArrayRef<int64_t> sgStrides,
                                       ArrayRef<int64_t> threadStrides,
                                       bool sharedMem) {
  auto i64Array = [&](ArrayRef<int64_t> vals) -> Attribute {
    SmallVector<Attribute> attrs;
    attrs.reserve(vals.size());
    for (int64_t v : vals)
      attrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), v));
    return ArrayAttr::get(ctx, attrs);
  };

  SmallVector<NamedAttribute> fields;
  fields.emplace_back(StringAttr::get(ctx, "batch_counts"),  i64Array(batchCounts));
  fields.emplace_back(StringAttr::get(ctx, "elem_counts"),   i64Array(elemCounts));
  fields.emplace_back(StringAttr::get(ctx, "outer_counts"),  i64Array(outerCounts));
  fields.emplace_back(StringAttr::get(ctx, "sg_counts"),     i64Array(sgCounts));
  fields.emplace_back(StringAttr::get(ctx, "sg_strides"),    i64Array(sgStrides));
  fields.emplace_back(StringAttr::get(ctx, "shared_mem"),    BoolAttr::get(ctx, sharedMem));
  fields.emplace_back(StringAttr::get(ctx, "thread_counts"), i64Array(threadCounts));
  fields.emplace_back(StringAttr::get(ctx, "thread_strides"),i64Array(threadStrides));
  return DictionaryAttr::get(ctx, fields);
}

//===----------------------------------------------------------------------===//
// §2  Hardware layout table
//     Maps (MMA intrinsic, operand index) → per-dimension thread distribution.
//
//     Each DimLayout describes ONE dimension of ONE operand:
//       outer   — outer unroll count (usually 1)
//       thread  — threads along this dim within the warp
//       element — contiguous elements each thread holds on this dim
//       tstride — stride for computing the thread's dim index from warp lane ID
//                 (0 = dim not distributed across threads)
//
//     Layout convention:
//       LHS (operand 0): [outerDim=M, innerDim=K]
//       RHS (operand 1): [outerDim=K, innerDim=N]
//       ACC (operand 2): [outerDim=M, innerDim=N]
//
//     Source: PTX ISA, NVIDIA CUDA Programming Guide,
//             IREE IREEGPUAttrs.cpp getSingleSubgroupLayout().
//===----------------------------------------------------------------------===//

struct DimLayout {
  int64_t outer;
  int64_t thread;
  int64_t element;
  int64_t tstride;
};

struct OperandHWLayout {
  DimLayout outerDimLayout; // M for LHS/ACC, K for RHS
  DimLayout innerDimLayout; // K for LHS, N for RHS/ACC
};

static FailureOr<OperandHWLayout>
getHardwareLayout(int32_t mmaKind, int operandIdx) {
  switch (static_cast<NVMMAIntrinsicValues>(mmaKind)) {

  // ── Ampere mma.sync f16/bf16: m16n8k16 ──────────────────────────────────
  // PTX: mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32
  //   LHS [M=16, K=16]: 16 threads on M (1 elem each), 1 thread on K (16 elem)
  //   RHS [K=16, N=8]:  1 thread on K (16 elem), 1 thread on N (8 elem)
  //   ACC [M=16, N=8]:  16 threads on M (1 elem each), 1 thread on N (8 elem)
  case NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16:
  case NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16:
    switch (operandIdx) {
    case 0: return OperandHWLayout{{1, 16, 1,  1}, {1, 1,  16, 0}}; // LHS
    case 1: return OperandHWLayout{{1, 1,  16, 0}, {1, 1,  8,  0}}; // RHS
    case 2: return OperandHWLayout{{1, 16, 1,  1}, {1, 1,  8,  0}}; // ACC
    }
    break;

  // ── Ampere mma.sync tf32: m16n8k8 ───────────────────────────────────────
  // PTX: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
  // nvgpu.mma.sync fragment shapes (2D, per warp-lane, 32 lanes):
  //   A  → vector<4x1xf32>   (mTile=2, kTile=2, numElemA=1: 2×2 × 1 = 4 elems)
  //   B  → vector<2x1xf32>   (kTile=2, nTile=1, numElemB=1: 2×1 × 1 = 2 elems)
  //   C  → vector<2x2xf32>   (mTile=2, nTile=1, numElemC=2: 2×1 × 2 = 4 elems)
  //
  // Layout convention for VectorDistribute per-thread slices:
  //   thread × elem coverage must equal the MMA tile dimension.
  //   All K-batching is done by the K-loop in distributeContractOp; per
  //   K-step the contract sees one kTile=8 slice.
  //
  //   LHS [M=16, K=8]:  M: 16 threads × 1 elem, K: 2 threads × 4 elem
  //                     per-thread: vector<1x1x4xf32>  (total 4 elem/thread ✓)
  //   RHS [K=8,  N=8]:  K: 2 threads × 4 elem, N: 8 threads × 1 elem
  //                     per-thread: vector<1x4x1xf32>  (total 4 elem/thread)
  //                     NOTE: K coverage matches LHS K (2×4=8) ✓
  //   ACC [M=16, N=8]:  M: 16 threads × 1 elem, N: 8 threads × 1 elem
  //                     per-thread: vector<1x1x1xf32>  (total 1 elem/thread)
  //                     The 4-elem nvgpu C fragment is produced by shape-casting
  //                     the per-batch acc slices in emitMNBatchUnroll.
  //
  // emitMNBatchUnroll detects mma_kind != 0 and emits nvgpu.mma.sync with
  // shape-casts: (1x1x4xf32→4x1xf32, 1x4x1xf32→2x1xf32, 1x1x1xf32→2x2xf32).
  // ── Ampere mma.sync tf32: m16n8k8 ───────────────────────────────────────
  // nvgpu.mma.sync fragment shapes (2D per lane, 32 lanes, tf32/f32):
  //   A → vector<4x1xf32>  mTile=2, kTile=2, numElemA=1  → 4 regs/lane ✓
  //   B → vector<2x1xf32>  kTile=2, nTile=1, numElemB=1  → 2 regs/lane ✓
  //   C → vector<2x2xf32>  mTile=2, nTile=1, numElemC=2  → 4 regs/lane ✓
  //
  // K is NOT distributed across threads (thread_K=1); one thread holds all
  // elem_K values for its row.  The K-loop in distributeContractOp steps by
  // mmaShape_K=8 (fixed separately); batchCounts_K = totalK / mmaShape_K.
  //
  // Per-thread 3D slice shapes → shape-cast to 2D nvgpu fragment:
  //   LHS [M=16, K=8]:  M: t=16,e=1  K: t=1,e=4  → [1,1,4] (4 elems) → [4,1] ✓
  //   RHS [K=8,  N=8]:  K: t=1, e=2  N: t=8,e=1  → [1,2,1] (2 elems) → [2,1] ✓
  //   ACC [M=16, N=8]:  M: t=8, e=2  N: t=4,e=2  → [1,2,2] (4 elems) → [2,2] ✓
  //     (8×2=16 M-rows ✓, 4×2=8 N-cols ✓, kThreadCount=1 → no warp shuffle)
  case NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8:
    switch (operandIdx) {
    case 0: return OperandHWLayout{{1, 16, 1, 1}, {1, 2, 4, 16}}; // LHS: M={t=16,e=1,s=1}, K={t=2,e=4,s=16}
    case 1: return OperandHWLayout{{1, 4, 2, 8}, {1, 8, 1, 1}};   // RHS: K={t=4,e=2,s=8},  N={t=8,e=1,s=1}
    case 2: return OperandHWLayout{{1, 8, 2, 1}, {1, 4, 2, 8}};   // ACC: M={t=8,e=2,s=1},  N={t=4,e=2,s=8}
    }
    break;

  // ── Volta/Turing WMMA f16: m16n16k16 ────────────────────────────────────
  // PTX: wmma.mma.sync.aligned.row.col.m16n16k16.f32.{f16,f16}.f32
  //   LHS [M=16, K=16]: 16 threads on M (1 elem), 1 thread on K (16 elem)
  //   RHS [K=16, N=16]: 1 thread on K (16 elem), 16 threads on N (1 elem)
  //   ACC [M=16, N=16]: 16 threads on M (1 elem), 1 thread on N (16 elem)
  case NVMMAIntrinsicValues::WMMA_F32_16x16x16:
  case NVMMAIntrinsicValues::WMMA_F16_16x16x16:
    switch (operandIdx) {
    case 0: return OperandHWLayout{{1, 16, 1,  1}, {1, 1,  16, 0}}; // LHS
    case 1: return OperandHWLayout{{1, 1,  16, 0}, {1, 16, 1,  1}}; // RHS
    case 2: return OperandHWLayout{{1, 16, 1,  1}, {1, 1,  16, 0}}; // ACC
    }
    break;

  // ── Volta/Turing WMMA tf32: m16n16k8 ────────────────────────────────────
  // PTX: wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32
  //   LHS [M=16, K=8]:  16 threads on M (1 elem), 1 thread on K (8 elem)
  //   RHS [K=8,  N=16]: 1 thread on K (8 elem), 16 threads on N (1 elem)
  //   ACC [M=16, N=16]: 16 threads on M (1 elem), 1 thread on N (16 elem)
  case NVMMAIntrinsicValues::WMMA_TF32_16x16x8:
    switch (operandIdx) {
    case 0: return OperandHWLayout{{1, 16, 1, 1}, {1, 1,  8,  0}}; // LHS
    case 1: return OperandHWLayout{{1, 1,  8, 0}, {1, 16, 1,  1}}; // RHS
    case 2: return OperandHWLayout{{1, 16, 1, 1}, {1, 1,  16, 0}}; // ACC
    }
    break;

  default:
    break;
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// §3  Core layout computation — one operand at a time
//===----------------------------------------------------------------------===//

static FailureOr<DictionaryAttr>
computeOperandLayout(MLIRContext *ctx,
                     linalg::LinalgOp op,
                     int operandIdx,
                     ArrayRef<int64_t> tiledBounds, // loop ranges of warp-level op
                     DictionaryAttr config) {
  int rank = tiledBounds.size();
  int32_t mmaKind = getMmaKindRaw(config);

  // ── 1. MMA intrinsic shape ───────────────────────────────────────────────
  // Resolved once and reused in steps 3 and 6.
  SmallVector<int64_t, 3> mmaShape = mlir::nova::getMMAShape(mmaKind);
  // [M-size, N-size, K-size], e.g. [16, 8, 8] for MMA_SYNC_TF32_16x8x8

  // ── 2. Per-workgroup subgroup counts ─────────────────────────────────────
  // "wg_subgroup" = how many warps tile each dim WITHIN one workgroup tile.
  // This is distinct from "subgroup" (global counts = numWg × warpsPerWg).
  //
  // The layout pass runs after ALL tiling: tiledBounds are the warp-level
  // loop ranges, not the workgroup tile.  We need per-WG counts to correctly
  // compute batch_counts and sg_counts relative to the workgroup tile.
  SmallVector<int64_t> perWgSgCounts =
      getConfigField(config, kWgSubgroupKey);

  if (perWgSgCounts.empty() || (int)perWgSgCounts.size() != rank) {
    // Fallback for configs that predate the wg_subgroup field.
    // Approximate subgroupCounts from wg tile sizes / MMA tile sizes.
    perWgSgCounts.assign(rank, 1);
    SmallVector<int64_t> wgField = getConfigField(config, kWorkgroupKey);
    auto contractionDimsFB = linalg::inferContractionDims(op);
    if (succeeded(contractionDimsFB) && !wgField.empty()) {
      // M dimension
      if (!contractionDimsFB->m.empty()) {
        int mDimFB = contractionDimsFB->m.back();
        if (mDimFB < (int)wgField.size() && wgField[mDimFB] > 0)
          perWgSgCounts[mDimFB] =
              llvm::divideCeil(wgField[mDimFB], mmaShape[0]);
      }
      // N dimension — must also be estimated or all warps alias on N
      if (!contractionDimsFB->n.empty()) {
        int nDimFB = contractionDimsFB->n.back();
        if (nDimFB < (int)wgField.size() && wgField[nDimFB] > 0)
          perWgSgCounts[nDimFB] =
              llvm::divideCeil(wgField[nDimFB], mmaShape[1]);
      }
    }
  }
  // Zero means "not distributed across subgroups" — normalise to 1.
  for (auto &v : perWgSgCounts)
    if (v == 0) v = 1;

  // ── 3. Subgroup strides — deferred until after projection (see step 10) ──
  // Computing strides on the full iteration-space counts and then projecting
  // gives wrong stride values when the operand map permutes dimensions (e.g.
  // RHS map swaps K and N).  We compute strides on the projected counts instead.

  // ── 4. Reconstruct workgroup-level bounds from config ────────────────────
  // tiledBounds reflects the warp-level op (after thread tiling).
  // We need workgroup bounds to compute how many MMA tiles each warp owns.
  //   parallel dims  → workgroup tile size
  //   reduction dim  → reduction tile size (one K step)
  SmallVector<int64_t> wgField  = getConfigField(config, kWorkgroupKey);
  SmallVector<int64_t> redField = getConfigField(config, kReductionKey);

  SmallVector<int64_t> wgBounds(rank);
  for (int i = 0; i < rank; i++) {
    if (i < (int)wgField.size() && wgField[i] > 0)
      wgBounds[i] = wgField[i];        // M, N, B parallel dims
    else if (i < (int)redField.size() && redField[i] > 0)
      wgBounds[i] = redField[i];       // K reduction dim (one loop step)
    else
      wgBounds[i] = tiledBounds[i];   // safe fallback
  }

  // ── 5. Per-subgroup bounds = wgBounds / perWgSgCounts ────────────────────
  // How much data each warp processes from the workgroup tile per dim.
  // e.g. wgM=64, perWgSgCounts_M=4  →  perSgBounds_M = 16
  SmallVector<int64_t> perSgBounds(rank);
  for (int i = 0; i < rank; i++)
    perSgBounds[i] = llvm::divideCeil(wgBounds[i], perWgSgCounts[i]);

  // ── 6. Identify innermost M, N, K iteration dims ─────────────────────────
  auto contractionDims = linalg::inferContractionDims(op);
  if (failed(contractionDims)) return failure();

  int innerM = contractionDims->m.empty() ? -1
                                          : (int)contractionDims->m.back();
  int innerN = contractionDims->n.empty() ? -1
                                          : (int)contractionDims->n.back();
  int innerK = contractionDims->k.empty() ? -1
                                          : (int)contractionDims->k.back();

  // ── 7. Full-rank MMA shape vector ────────────────────────────────────────
  SmallVector<int64_t> mmaFullShape(rank, 1);
  if (innerM >= 0) mmaFullShape[innerM] = mmaShape[0]; // M tile
  if (innerN >= 0) mmaFullShape[innerN] = mmaShape[1]; // N tile
  if (innerK >= 0) mmaFullShape[innerK] = mmaShape[2]; // K tile

  // ── 8. Batch counts = ceil(perSgBounds / mmaFullShape) ───────────────────
  // Number of MMA tiles each warp processes per dimension.
  // e.g. perSgBounds_K=64, mmaShape_K=8  →  batch_K = 8
  SmallVector<int64_t> batchCounts(rank);
  for (int i = 0; i < rank; i++)
    batchCounts[i] = llvm::divideCeil(perSgBounds[i], mmaFullShape[i]);

  // ── 9. Per-MMA-tile thread layout (hardware-defined) ─────────────────────
  SmallVector<int64_t> outerCounts(rank, 1);
  SmallVector<int64_t> threadCounts(rank, 1);
  SmallVector<int64_t> threadStrides(rank, 0);
  SmallVector<int64_t> elemCounts(rank, 1);

  auto hwLayout = getHardwareLayout(mmaKind, operandIdx);
  if (failed(hwLayout)) return failure();

  // Map hardware outerDim/innerDim onto the full iteration space dims:
  //   LHS (0): outerDim = M,  innerDim = K
  //   RHS (1): outerDim = K,  innerDim = N
  //   ACC (2): outerDim = M,  innerDim = N
  int outerDim = -1, innerDim = -1;
  switch (operandIdx) {
  case 0: outerDim = innerM; innerDim = innerK; break;
  case 1: outerDim = innerK; innerDim = innerN; break;
  case 2: outerDim = innerM; innerDim = innerN; break;
  default: return failure();
  }

  if (outerDim >= 0) {
    outerCounts[outerDim]   = hwLayout->outerDimLayout.outer;
    threadCounts[outerDim]  = hwLayout->outerDimLayout.thread;
    threadStrides[outerDim] = hwLayout->outerDimLayout.tstride;
    elemCounts[outerDim]    = hwLayout->outerDimLayout.element;
  }
  if (innerDim >= 0) {
    outerCounts[innerDim]   = hwLayout->innerDimLayout.outer;
    threadCounts[innerDim]  = hwLayout->innerDimLayout.thread;
    threadStrides[innerDim] = hwLayout->innerDimLayout.tstride;
    elemCounts[innerDim]    = hwLayout->innerDimLayout.element;
  }

  // ── 10. Project all full-rank arrays through the operand's indexing map ──
  // Drops iteration dims that don't appear in this operand (e.g. N from LHS).
  // Strides are computed AFTER projection so that row-major ordering is based
  // on the operand's physical axis order, not the iteration-space order.
  AffineMap operandMap = op.getIndexingMapsArray()[operandIdx];

  SmallVector<int64_t> projSgCounts,    projBatchCounts,  projOuterCounts;
  SmallVector<int64_t> projThreadCounts,projElemCounts;

  projectThroughMap(operandMap, perWgSgCounts, projSgCounts,     1);
  projectThroughMap(operandMap, batchCounts,   projBatchCounts,  1);
  projectThroughMap(operandMap, outerCounts,   projOuterCounts,  1);
  projectThroughMap(operandMap, threadCounts,  projThreadCounts, 1);
  projectThroughMap(operandMap, elemCounts,    projElemCounts,   1);

  // Compute strides on the projected counts so stride values respect the
  // operand's physical axis ordering after any map permutation.
  SmallVector<int64_t> projSgStrides     = computeStrides(projSgCounts);
  SmallVector<int64_t> projThreadStrides;
projectThroughMap(operandMap, threadStrides, projThreadStrides, 0);

  // ── 11. Shared memory flag ────────────────────────────────────────────────
  bool sharedMem = llvm::is_contained(getPromotedOperands(config),
                                       (int64_t)operandIdx);

  // ── 12. Assemble and return the layout attribute ──────────────────────────
  return buildLayoutAttr(ctx,
                         projSgCounts,    projBatchCounts,
                         projOuterCounts, projThreadCounts,
                         projElemCounts,  projSgStrides,
                         projThreadStrides, sharedMem);
}

//===----------------------------------------------------------------------===//
// §4  Pass definition
//===----------------------------------------------------------------------===//

struct NovaGPUConfigureTensorLayoutsPass
    : public PassWrapper<NovaGPUConfigureTensorLayoutsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUConfigureTensorLayoutsPass)

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    funcOp.walk([&](linalg::LinalgOp op) {
      // Only MMA ops have layouts to configure.
      auto config = getLoweringConfig(op.getOperation());
      if (!config || getMmaKindRaw(config) == 0)
        return;

      // Layout computation is only meaningful for contraction ops.
      if (!linalg::isaContractionOpInterface(op))
        return;

      // Requires a fully static iteration space (dynamic shapes are padded
      // away earlier in the pipeline by NovaGPUPadOperandsPass).
      SmallVector<int64_t> bounds = op.getStaticLoopRanges();
      if (ShapedType::isDynamicShape(bounds)) {
        op.emitWarning() << "nova-gpu-configure-tensor-layouts: "
                            "skipping op with dynamic iteration space";
        return;
      }

      // Attach nova.layout_0 / nova.layout_1 / nova.layout_2 for
      // LHS, RHS, ACC respectively.
      int layoutCount = std::min(op->getNumOperands(), 3u);
      for (int i = 0; i < layoutCount; i++) {
        auto layout = computeOperandLayout(ctx, op, i, bounds, config);
        if (failed(layout)) {
          op.emitError() << "nova-gpu-configure-tensor-layouts: "
                            "failed to compute layout for operand "
                         << i << " (mma_kind=" << getMmaKindRaw(config) << ")";
          return signalPassFailure();
        }
        op->setAttr("nova.layout_" + std::to_string(i), *layout);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[nova-configure-layouts] " << op->getName() << "\n";
        for (int i = 0; i < layoutCount; i++)
          llvm::dbgs() << "  operand " << i << ": "
                       << op->getAttr("nova.layout_" + std::to_string(i))
                       << "\n";
      });
    });
  }

  StringRef getArgument() const override {
    return "nova-gpu-configure-tensor-layouts";
  }
  StringRef getDescription() const override {
    return "Attach per-operand MMA thread distribution layouts (nova.layout_*)"
           " to MMA contraction ops.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

namespace mlir::nova {

std::unique_ptr<Pass> createNovaGPUConfigureTensorLayoutsPass() {
  return std::make_unique<NovaGPUConfigureTensorLayoutsPass>();
}

void registerNovaGPUConfigureTensorLayoutsPass() {
  PassRegistration<NovaGPUConfigureTensorLayoutsPass>();
}

} // namespace mlir::nova
