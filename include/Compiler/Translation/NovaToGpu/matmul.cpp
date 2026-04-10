#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace nova {

// ============================================================================
// Scalar CUDA-core path (used for f64 or as fallback)
// ============================================================================
static LogicalResult lowerScalarMatmul(nova::MatmulOp op,
                                       PatternRewriter &rewriter,
                                       Value lhsFlat, Value rhsFlat,
                                       Value resFlat,
                                       int64_t M, int64_t K, int64_t N,
                                       int64_t totalBatches,
                                       int64_t lhsBatchStride,
                                       int64_t rhsBatchStride,
                                       int64_t resBatchStride,
                                       RankedTensorType resultType,
                                       SmallVector<int64_t> resultShape,
                                       Value resultMemRef) {
  Location loc = op.getLoc();
  int64_t tileSize = 16;
  int64_t elemByteSize =
      cast<RankedTensorType>(op.getLhs().getType()).getElementType().getIntOrFloatBitWidth() / 8;
  int64_t tileByteSize = tileSize * tileSize * elemByteSize;
  int64_t totalSharedBytes = 2 * tileByteSize;

  Value cM = rewriter.create<arith::ConstantIndexOp>(loc, M);
  Value cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
  Value cK = rewriter.create<arith::ConstantIndexOp>(loc, K);
  Value cTileSize = rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value cTileBytes = rewriter.create<arith::ConstantIndexOp>(loc, tileByteSize);
  Value cTotalBatches = rewriter.create<arith::ConstantIndexOp>(loc, totalBatches);
  Value cLhsBatchStride = rewriter.create<arith::ConstantIndexOp>(loc, lhsBatchStride);
  Value cRhsBatchStride = rewriter.create<arith::ConstantIndexOp>(loc, rhsBatchStride);
  Value cResBatchStride = rewriter.create<arith::ConstantIndexOp>(loc, resBatchStride);
  Value cLhsKStride = rewriter.create<arith::ConstantIndexOp>(loc, K);
  Value cRhsNStride = rewriter.create<arith::ConstantIndexOp>(loc, N);

  Value dynamicSharedMemSize = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(),
      rewriter.create<arith::ConstantIndexOp>(loc, totalSharedBytes));

  Value gridX = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, cN,
          rewriter.create<arith::ConstantIndexOp>(loc, tileSize - 1)), cTileSize);
  Value gridY = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, cM,
          rewriter.create<arith::ConstantIndexOp>(loc, tileSize - 1)), cTileSize);
  Value gridZ = cTotalBatches;

  auto launchOp = rewriter.create<gpu::LaunchOp>(
      loc,
      gridX, gridY, gridZ,
      cTileSize, cTileSize, c1,
      /*dynamicSharedMemorySize=*/dynamicSharedMemSize);

  rewriter.setInsertionPointToStart(&launchOp.getBody().front());

  MLIRContext *ctx = rewriter.getContext();
  auto workgroupAddrSpace = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto i8DynMemRefType = MemRefType::get({ShapedType::kDynamic},
                                         rewriter.getIntegerType(8),
                                         AffineMap{},
                                         workgroupAddrSpace);
  Value shmem = rewriter.create<gpu::DynamicSharedMemoryOp>(loc, i8DynMemRefType);

  auto tileMemRefType = MemRefType::get({tileSize, tileSize},
                                        cast<RankedTensorType>(op.getLhs().getType()).getElementType(),
                                        AffineMap{},
                                        workgroupAddrSpace);
  Value c0k = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value tileA = rewriter.create<memref::ViewOp>(loc, tileMemRefType, shmem, c0k, ValueRange{});
  Value tileB = rewriter.create<memref::ViewOp>(loc, tileMemRefType, shmem, cTileBytes, ValueRange{});

  Value tx = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value ty = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value bx = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value by = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value bz = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::z);

  Value col = rewriter.create<arith::AddIOp>(loc, rewriter.create<arith::MulIOp>(loc, bx, cTileSize), tx);
  Value row = rewriter.create<arith::AddIOp>(loc, rewriter.create<arith::MulIOp>(loc, by, cTileSize), ty);

  Value lhsBatchOff = rewriter.create<arith::MulIOp>(loc, bz, cLhsBatchStride);
  Value rhsBatchOff = rewriter.create<arith::MulIOp>(loc, bz, cRhsBatchStride);
  Value resBatchOff = rewriter.create<arith::MulIOp>(loc, bz, cResBatchStride);

  Value sum_init = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultType.getElementType()));

  auto kLoop = rewriter.create<scf::ForOp>(loc, c0k, cK, cTileSize, ValueRange{sum_init});
  rewriter.setInsertionPointToStart(kLoop.getBody());

  Value kOffset = kLoop.getInductionVar();
  Value currentSum = kLoop.getRegionIterArgs()[0];

  Value aCol = rewriter.create<arith::AddIOp>(loc, kOffset, tx);
  Value aInBounds = rewriter.create<arith::AndIOp>(loc,
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, cM),
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, aCol, cK));

  auto ifA = rewriter.create<scf::IfOp>(loc, aInBounds, true);
  rewriter.setInsertionPointToStart(ifA.thenBlock());
  {
    Value rowOffset = rewriter.create<arith::MulIOp>(loc, row, cLhsKStride);
    Value aLinear = rewriter.create<arith::AddIOp>(loc,
                      rewriter.create<arith::AddIOp>(loc, lhsBatchOff, rowOffset), aCol);
    Value aVal = rewriter.create<memref::LoadOp>(loc, lhsFlat, ValueRange{aLinear});
    rewriter.create<memref::StoreOp>(loc, aVal, tileA, ValueRange{ty, tx});
  }
  rewriter.setInsertionPointToStart(ifA.elseBlock());
  rewriter.create<memref::StoreOp>(loc, sum_init, tileA, ValueRange{ty, tx});
  rewriter.setInsertionPointAfter(ifA);

  Value bRow = rewriter.create<arith::AddIOp>(loc, kOffset, ty);
  Value bInBounds = rewriter.create<arith::AndIOp>(loc,
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, bRow, cK),
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, cN));

  auto ifB = rewriter.create<scf::IfOp>(loc, bInBounds, true);
  rewriter.setInsertionPointToStart(ifB.thenBlock());
  {
    Value bRowOffset = rewriter.create<arith::MulIOp>(loc, bRow, cRhsNStride);
    Value bLinear = rewriter.create<arith::AddIOp>(loc,
                      rewriter.create<arith::AddIOp>(loc, rhsBatchOff, bRowOffset), col);
    Value bVal = rewriter.create<memref::LoadOp>(loc, rhsFlat, ValueRange{bLinear});
    rewriter.create<memref::StoreOp>(loc, bVal, tileB, ValueRange{ty, tx});
  }
  rewriter.setInsertionPointToStart(ifB.elseBlock());
  rewriter.create<memref::StoreOp>(loc, sum_init, tileB, ValueRange{ty, tx});
  rewriter.setInsertionPointAfter(ifB);

  rewriter.create<NVVM::Barrier0Op>(loc);

  auto innerLoop = rewriter.create<scf::ForOp>(loc, c0k, cTileSize, c1, ValueRange{currentSum});
  Block *kBlock = kLoop.getBody();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *innerBody = innerLoop.getBody();
    if (!innerBody->empty() && innerBody->back().hasTrait<mlir::OpTrait::IsTerminator>())
      rewriter.eraseOp(&innerBody->back());
    rewriter.setInsertionPointToEnd(innerBody);

    Value i   = innerLoop.getInductionVar();
    Value acc = innerLoop.getRegionIterArgs()[0];
    Value tA  = rewriter.create<memref::LoadOp>(loc, tileA, ValueRange{ty, i});
    Value tB  = rewriter.create<memref::LoadOp>(loc, tileB, ValueRange{i, tx});

    Value mul, nextAcc;
    if (llvm::isa<FloatType>(resultType.getElementType())) {
      mul     = rewriter.create<arith::MulFOp>(loc, tA, tB);
      nextAcc = rewriter.create<arith::AddFOp>(loc, acc, mul);
    } else {
      mul     = rewriter.create<arith::MulIOp>(loc, tA, tB);
      nextAcc = rewriter.create<arith::AddIOp>(loc, acc, mul);
    }
    rewriter.create<scf::YieldOp>(loc, nextAcc);
  }

  Value loopResult = innerLoop.getResult(0);

  {
    if (!kBlock->empty() && kBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
      rewriter.eraseOp(&kBlock->back());
    rewriter.setInsertionPoint(kBlock, kBlock->end());
  }
  rewriter.create<NVVM::Barrier0Op>(loc);
  rewriter.create<scf::YieldOp>(loc, loopResult);

  rewriter.setInsertionPointAfter(kLoop);
  Value finalResult = kLoop.getResult(0);

  Value outBounds = rewriter.create<arith::AndIOp>(loc,
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, cM),
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, cN));

  auto ifOut = rewriter.create<scf::IfOp>(loc, outBounds, false);
  rewriter.setInsertionPointToStart(ifOut.thenBlock());
  {
    Value rowOffsetR = rewriter.create<arith::MulIOp>(loc, row, cRhsNStride);
    Value rLinear    = rewriter.create<arith::AddIOp>(loc,
                          rewriter.create<arith::AddIOp>(loc, resBatchOff, rowOffsetR), col);
    rewriter.create<memref::StoreOp>(loc, finalResult, resFlat, ValueRange{rLinear});
  }
  rewriter.setInsertionPointAfter(ifOut);

  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointAfter(launchOp);

  Value resultTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, resultType, resultMemRef, true).getResult();
  rewriter.replaceOp(op, resultTensor);
  return success();
}

// ============================================================================
// Tensor Core path — V18-style Pointer Induction Edition
//
// Key changes vs. previous version:
//  - All thread-invariant math (laneId/4, warpId*WARP_N, rowBase, etc.)
//    is hoisted BEFORE the loop — computed once, reused forever.
//  - 128 threads / 4 warps in a 2×2 grid, matching sgemm_v18 layout.
//  - 3-stage async pipeline with prolog loads before loop entry.
//  - Global addresses recomputed from kBase each iteration (pointer
//    induction via iter_args was removed — cAStride/cBStride are dead).
// ============================================================================
static LogicalResult lowerTensorCoreMatmul(nova::MatmulOp op,
                                           PatternRewriter &rewriter,
                                           Value lhsFlat, Value rhsFlat,
                                           Value resFlat,
                                           int64_t M, int64_t K, int64_t N,
                                           int64_t totalBatches,
                                           int64_t lhsBatchStride,
                                           int64_t rhsBatchStride,
                                           int64_t resBatchStride,
                                           RankedTensorType resultType,
                                           SmallVector<int64_t> resultShape,
                                           Value resultMemRef,
                                           bool useTF32) {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // ── Tile / warp constants ──
  // Warp layout: 2 warps in M-dim, 2 warps in N-dim (4 warps total, 128 threads)
  const int64_t TILE_M      = 128;
  const int64_t TILE_N      = 128;
  // TF32 MMA uses m16n8k8 → TILE_K = 8
  // FP16 MMA uses m16n8k16 → TILE_K = 16
  const int64_t TILE_K      = useTF32 ? 8 : 16;
  const int64_t MMA_K       = TILE_K;
  const int64_t KS_STEPS    = 1;
  const int64_t WARPS       [[maybe_unused]] = 4;
  const int64_t WARP_M_CNT  = 2;
  const int64_t WARP_N_CNT  = 2;
  const int64_t WARP_M      = TILE_M / WARP_M_CNT;  // 64
  const int64_t WARP_N      = TILE_N / WARP_N_CNT;  // 64
  const int64_t THREADS     = 128;
  const int64_t STAGES      = 3;

  // MMA shape: m16n8k8 (TF32) or m16n8k16 (FP16)
  const int64_t MMA_M = 16, MMA_N = 8;
  const int64_t FR_M  = WARP_M / MMA_M;  // 4
  const int64_t FR_N  = WARP_N / MMA_N;  // 8

  // Total elements per stage
  const int64_t A_ELEMS = TILE_M * TILE_K;
  const int64_t B_ELEMS = TILE_K * TILE_N;

  // 128-bit vectorized async copy: 4×f32 or 8×f16 per cp.async instruction.
  // Vectorized path requires ALL tiles to be fully in-bounds (no boundary
  // partial tiles), so that we can omit the runtime srcElements parameter.
  // The _s intrinsic variant (cp.async ... 16, %reg) with a runtime src_size
  // register causes non-deterministic results — likely an NVPTX codegen bug
  // in instruction scheduling of the src_size register.
  const int64_t VEC_ELEMS  = useTF32 ? 4 : 8;
  const bool tileAligned   = (M % TILE_M == 0) && (K % TILE_K == 0) && (N % TILE_N == 0);
  const bool useVecCopy    = tileAligned && (K % VEC_ELEMS == 0) && (N % VEC_ELEMS == 0);
  const int64_t COPY_ELEMS = useVecCopy ? VEC_ELEMS : 1;

  // A tile: COPY_ELEMS-wide groups assigned linearly to threads
  const int64_t A_GROUPS_PER_ROW = TILE_K / COPY_ELEMS;
  const int64_t A_TOTAL_GROUPS   = TILE_M * A_GROUPS_PER_ROW;
  const int64_t A_PASSES         = A_TOTAL_GROUPS / THREADS;

  // B tile: COPY_ELEMS-wide groups assigned linearly to threads
  const int64_t B_GROUPS_PER_ROW = TILE_N / COPY_ELEMS;
  const int64_t B_TOTAL_GROUPS   = TILE_K * B_GROUPS_PER_ROW;
  const int64_t B_LOAD_ITERS     = B_TOTAL_GROUPS / THREADS;

  // Swizzle config (must be consistent between write and MMA-read sides)
  // A: stride = COPY_ELEMS, mask = groups_per_row - 1
  // B: stride = TILE_N/TILE_K (unchanged — already a multiple of COPY_ELEMS)
  const int64_t A_SWIZZLE_MASK   = A_GROUPS_PER_ROW - 1;
  const int64_t A_SWIZZLE_STRIDE = COPY_ELEMS;
  const int64_t B_SWIZZLE_STRIDE = TILE_N / TILE_K;

  SmallVector<int64_t> mmaShapeVals = {MMA_M, MMA_N, MMA_K};


  // ── Element types and MMA fragment types ──
  Type mmaInputTy  = useTF32 ? rewriter.getF32Type() : rewriter.getF16Type();
  Type mmaAccTy    = rewriter.getF32Type();

  // Fragment dimensions per k-dimension (per single MMA invocation, NOT per SMEM tile)
  int64_t aK = MMA_K / 8;   // 1 for TF32 (k8/8), 2 for FP16 (k16/8)
  int64_t bK = MMA_K / 8;   // 1 for TF32 (k8/8), 2 for FP16 (k16/8)

  // TF32 (m16n8k8): A=vector<4x1xf32>, B=vector<2x1xf32> (4=m16/4, 1=k8/8, 2=m16/8, 1=k8/8)
  // FP16 (m16n8k16): A=vector<4x2xf16>, B=vector<2x2xf16> (4=m16/4, 2=k16/8, 2=m16/8, 2=k16/8)
  auto aFragTy = VectorType::get({4, aK}, mmaInputTy);
  auto bFragTy = VectorType::get({2, bK}, mmaInputTy);
  // Accumulator holds FR_M x FR_N MMA output tiles, each is 2x2 floats
  auto cFragTy = VectorType::get({FR_M, FR_N, 2, 2}, mmaAccTy);

  // Shared memory per stage: A tile + B tile (bytes)
  int64_t inputElemBytes  = mmaInputTy.getIntOrFloatBitWidth() / 8;
  int64_t stageABytes     = A_ELEMS * inputElemBytes;
  int64_t stageBBytes     = B_ELEMS * inputElemBytes;
  int64_t stageTotalBytes = stageABytes + stageBBytes;
  int64_t totalSmemBytes  = STAGES * stageTotalBytes;


  // ── Index constants emitted ONCE (fully hoisted) ──
  auto ci = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), c2 = ci(2), c3 = ci(3), c4 = ci(4), c7 = ci(7);
  Value cTileK  = ci(TILE_K),  cTileM  = ci(TILE_M),  cTileN  = ci(TILE_N);
  Value cWarpM  = ci(WARP_M),  cWarpN  = ci(WARP_N),  cThreads = ci(THREADS), c32 = ci(32);
  Value cWarpMCnt = ci(WARP_M_CNT), cWarpNCnt = ci(WARP_N_CNT);
  Value cM = ci(M), cN = ci(N), cK = ci(K);
  Value cNStride     = ci(N);
  Value cKStride     = ci(K);
  Value cStages      = ci(STAGES);
  Value cAElems      = ci(A_ELEMS);
  Value cBElems      = ci(B_ELEMS);
  // Pointer induction strides (in elements)
  Value cAStride = cTileK;            // per K-step advance for A
  Value cBStride = ci(TILE_K * N);    // per K-step advance for B (row-major)

  // Launch grid / block
  Value dynamicSmem = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), ci(totalSmemBytes));
  Value gridX = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, cN, ci(TILE_N - 1)), cTileN);
  Value gridY = rewriter.create<arith::DivUIOp>(loc,
      rewriter.create<arith::AddIOp>(loc, cM, ci(TILE_M - 1)), cTileM);
  Value gridZ = ci(totalBatches);

  auto launchOp = rewriter.create<gpu::LaunchOp>(
      loc, gridX, gridY, gridZ, cThreads, c1, c1,
      /*dynamicSharedMemorySize=*/dynamicSmem);

  rewriter.setInsertionPointToStart(&launchOp.getBody().front());

  // ── Shared memory ──
  auto workgroupAS = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto i8SmemTy = MemRefType::get(
      {ShapedType::kDynamic}, rewriter.getIntegerType(8), AffineMap{}, workgroupAS);
  Value shmem = rewriter.create<gpu::DynamicSharedMemoryOp>(loc, i8SmemTy);

  auto sATy = MemRefType::get({TILE_M, TILE_K}, mmaInputTy, AffineMap{}, workgroupAS);
  auto sBTy = MemRefType::get({TILE_K, TILE_N}, mmaInputTy, AffineMap{}, workgroupAS);

  // Get shared memory views for a given stage index (returns sA, sB views)
  auto getStageViews = [&](OpBuilder &b, Value stageIdx) -> std::pair<Value, Value> {
    Value cStageTotV    = b.create<arith::ConstantIndexOp>(loc, stageTotalBytes);
    Value cStageABytesV = b.create<arith::ConstantIndexOp>(loc, stageABytes);
    Value stageBase = b.create<arith::MulIOp>(loc, stageIdx, cStageTotV);
    Value bBase     = b.create<arith::AddIOp>(loc, stageBase, cStageABytesV);
    Value vA = b.create<memref::ViewOp>(loc, sATy, shmem, stageBase, ValueRange{});
    Value vB = b.create<memref::ViewOp>(loc, sBTy, shmem, bBase,     ValueRange{});
    return {vA, vB};
  };


  // ── Thread/warp IDs — hoisted, computed once ──
  Value tid    = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value laneId = rewriter.create<arith::RemUIOp>(loc, tid, c32);
  Value warpId = rewriter.create<arith::DivUIOp>(loc, tid, c32);

  // ── Block coordinates (direct mapping) ──
  Value bx = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value by = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);

  // Batch offset (loop-invariant)
  Value bz          = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::z);
  Value lhsBatchOff = rewriter.create<arith::MulIOp>(loc, bz, ci(lhsBatchStride));
  Value rhsBatchOff = rewriter.create<arith::MulIOp>(loc, bz, ci(rhsBatchStride));
  Value resBatchOff = rewriter.create<arith::MulIOp>(loc, bz, ci(resBatchStride));

  // Block base row/col (needed by tile loading and epilogue)
  Value rowBase      = rewriter.create<arith::MulIOp>(loc, by, cTileM);
  Value blockColBase = rewriter.create<arith::MulIOp>(loc, bx, cTileN);

  // ── Block / warp coordinate decomposition (hoisted) ──
  // Warp layout in block: warpRow = warpId / WARP_N_CNT, warpCol = warpId % WARP_N_CNT
  Value warpRow    = rewriter.create<arith::DivUIOp>(loc, warpId, cWarpNCnt);
  Value warpColIdx = rewriter.create<arith::RemUIOp>(loc, warpId, cWarpNCnt);
  Value warpRowOff = rewriter.create<arith::MulIOp>(loc, warpRow, cWarpM);
  Value warpColOff = rewriter.create<arith::MulIOp>(loc, warpColIdx, cWarpN);
  // Absolute warp origins within the global matrix
  Value warpGlobalRowBase = rewriter.create<arith::AddIOp>(loc, rowBase, warpRowOff);
  Value warpColBase       = rewriter.create<arith::AddIOp>(loc, blockColBase, warpColOff);

  // ── issueStageLoad: loads one async stage from global → shared ──
  // kBase = absolute K-column where this tile begins.
  // Uses COPY_ELEMS-wide async copies (128-bit when K,N aligned, scalar fallback).
  // Each thread is assigned a linear group index; groups of COPY_ELEMS
  // contiguous elements are loaded per cp.async instruction.
  auto issueStageLoad = [&](OpBuilder &b, Value stageIdx, Value kBase) {
    auto [sA, sB] = getStageViews(b, stageIdx);
    SmallVector<Value> tokens;

    // ── A copies: A_PASSES passes, COPY_ELEMS elements per instruction ──
    for (int64_t i = 0; i < A_PASSES; ++i) {
      Value linIdx = b.create<arith::AddIOp>(loc, tid,
          b.create<arith::ConstantIndexOp>(loc, i * THREADS));
      // Decompose linear index → (row, vector-group within row)
      Value aTileRowI = b.create<arith::DivUIOp>(loc, linIdx,
                            b.create<arith::ConstantIndexOp>(loc, A_GROUPS_PER_ROW));
      Value aGroupIdx = b.create<arith::RemUIOp>(loc, linIdx,
                            b.create<arith::ConstantIndexOp>(loc, A_GROUPS_PER_ROW));
      Value aTileColI = b.create<arith::MulIOp>(loc, aGroupIdx,
                            b.create<arith::ConstantIndexOp>(loc, COPY_ELEMS));
      Value aGlobalRowI = b.create<arith::AddIOp>(loc, rowBase, aTileRowI);
      Value aKColI      = b.create<arith::AddIOp>(loc, kBase, aTileColI);

      // Bounds: row < M and kCol < K (K % COPY_ELEMS == 0 guarantees full group)
      Value aRowOkI    = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, aGlobalRowI, cM);
      Value aColOkI    = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, aKColI, cK);
      Value aInBoundsI = b.create<arith::AndIOp>(loc, aRowOkI, aColOkI);

      Value aPtrI = b.create<arith::AddIOp>(loc, lhsBatchOff,
          b.create<arith::AddIOp>(loc,
              b.create<arith::MulIOp>(loc, aGlobalRowI, cKStride),
              aKColI));

      // srcElements = COPY_ELEMS when in-bounds, 0 for hardware zero-fill
      Value aSrcElems = b.create<arith::SelectOp>(loc, aInBoundsI,
          b.create<arith::ConstantIndexOp>(loc, COPY_ELEMS), c0);

      // A swizzle: col ^ ((row & A_SWIZZLE_MASK) * A_SWIZZLE_STRIDE)
      // Stride is a multiple of COPY_ELEMS → destination stays COPY_ELEMS-aligned
      Value aRowMod  = b.create<arith::AndIOp>(loc, aTileRowI,
                           b.create<arith::ConstantIndexOp>(loc, A_SWIZZLE_MASK));
      Value aXorMask = b.create<arith::MulIOp>(loc, aRowMod,
                           b.create<arith::ConstantIndexOp>(loc, A_SWIZZLE_STRIDE));
      Value aColSwiz = b.create<arith::XOrIOp>(loc, aTileColI, aXorMask);

      // When useVecCopy (tileAligned): all accesses in-bounds, omit srcElements
      // to avoid the _s intrinsic variant (non-deterministic NVPTX codegen bug).
      // Also omit bypassL1 to use cp.async.ca (not cg — cg causes corruption).
      // Scalar path: runtime srcElements needed for boundary zero-fill.
      tokens.push_back(b.create<nvgpu::DeviceAsyncCopyOp>(
          loc, sA, ValueRange{aTileRowI, aColSwiz},
          lhsFlat, ValueRange{aPtrI},
          rewriter.getIndexAttr(COPY_ELEMS),
          useVecCopy ? nullptr : aSrcElems,
          /*bypassL1=*/nullptr));
    }

    // ── B copies: B_LOAD_ITERS passes, COPY_ELEMS elements per instruction ──
    for (int64_t i = 0; i < B_LOAD_ITERS; ++i) {
      Value linIdx = b.create<arith::AddIOp>(loc, tid,
          b.create<arith::ConstantIndexOp>(loc, i * THREADS));
      // Decompose → (row within TILE_K, vector-group within TILE_N)
      Value bTileRowI = b.create<arith::DivUIOp>(loc, linIdx,
                            b.create<arith::ConstantIndexOp>(loc, B_GROUPS_PER_ROW));
      Value bGroupIdx = b.create<arith::RemUIOp>(loc, linIdx,
                            b.create<arith::ConstantIndexOp>(loc, B_GROUPS_PER_ROW));
      Value bTileColI = b.create<arith::MulIOp>(loc, bGroupIdx,
                            b.create<arith::ConstantIndexOp>(loc, COPY_ELEMS));

      // B swizzle: col ^ ((row & (TILE_K-1)) * B_SWIZZLE_STRIDE)
      // B_SWIZZLE_STRIDE = TILE_N/TILE_K ≥ COPY_ELEMS → preserves alignment
      Value bRowMod  = b.create<arith::AndIOp>(loc, bTileRowI,
                           b.create<arith::ConstantIndexOp>(loc, TILE_K - 1));
      Value bXorMask = b.create<arith::MulIOp>(loc, bRowMod,
                           b.create<arith::ConstantIndexOp>(loc, B_SWIZZLE_STRIDE));
      Value bColSwiz = b.create<arith::XOrIOp>(loc, bTileColI, bXorMask);

      // Global source address
      Value bKColI      = b.create<arith::AddIOp>(loc, kBase, bTileRowI);
      Value bGlobalColI = b.create<arith::AddIOp>(loc, blockColBase, bTileColI);
      Value bGlobal     = b.create<arith::AddIOp>(loc, rhsBatchOff,
          b.create<arith::AddIOp>(loc,
              b.create<arith::MulIOp>(loc, bKColI, cNStride),
              bGlobalColI));

      // Bounds: K-row < K and N-col < N
      Value bKOkI    = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, bKColI, cK);
      Value bColOkI  = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, bGlobalColI, cN);
      Value inBounds = b.create<arith::AndIOp>(loc, bKOkI, bColOkI);
      Value bSrcElems = b.create<arith::SelectOp>(loc, inBounds,
          b.create<arith::ConstantIndexOp>(loc, COPY_ELEMS), c0);

      tokens.push_back(b.create<nvgpu::DeviceAsyncCopyOp>(
          loc, sB, ValueRange{bTileRowI, bColSwiz},
          rhsFlat, ValueRange{bGlobal},
          rewriter.getIndexAttr(COPY_ELEMS),
          useVecCopy ? nullptr : bSrcElems,
          /*bypassL1=*/nullptr));
    }

    // Use NVVM commit_group directly instead of nvgpu.device_async_create_group.
    // The nvgpu token type (!nvgpu.device.async.token) cannot survive the
    // SCF-to-CF + NVGPU-to-NVVM pass pipeline when flowing through scf.for
    // iter_args or scf.if results — the commit/wait ops get silently dropped,
    // leaving cp.async with NO synchronisation → non-deterministic results.
    (void)tokens;  // cp.async ops are side-effecting; tokens unused
    b.create<NVVM::CpAsyncCommitGroupOp>(loc);
  };


  // ── Prolog: Load stages 0 and 1 ──
  // issueStageLoad emits cp.async + NVVM::CpAsyncCommitGroupOp internally.
  // No token values flow through scf.if — avoids the nvgpu token lowering bug.
  issueStageLoad(rewriter, c0, c0);

  Value hasStage1 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                    cK, cTileK);
  auto ifS1 = rewriter.create<scf::IfOp>(loc, hasStage1, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ifS1.thenBlock());
    if (!ifS1.thenBlock()->empty()) rewriter.eraseOp(&ifS1.thenBlock()->back());
    issueStageLoad(rewriter, c1, cTileK);
    rewriter.create<scf::YieldOp>(loc);
  }

  // ── Accumulator init: FR_M x FR_N x 2 x 2 floats = 0 ──
  Value cZeroAcc = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(mmaAccTy));
  Value cInit    = rewriter.create<vector::SplatOp>(loc, cFragTy, cZeroAcc);

  // ── Wait for prolog loads to complete before MMA ──
  // cp.async.wait_group 0 — drain ALL outstanding async copies
  rewriter.create<NVVM::CpAsyncWaitGroupOp>(loc, rewriter.getI32IntegerAttr(0));
  rewriter.create<NVVM::Barrier0Op>(loc);  // Sync all warps before MMA

  // ── Main K-loop — iter_args: (acc, writeStage, readStage) ──
  Value cWriteStage0 = rewriter.create<arith::RemUIOp>(loc, c2, cStages);  // =2
  Value cReadStage0  = c0;

  auto mainLoop = rewriter.create<scf::ForOp>(
      loc, c0, cK, cTileK,
      ValueRange{cInit, cWriteStage0, cReadStage0});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mainLoop.getBody());
    if (!mainLoop.getBody()->empty())
      rewriter.eraseOp(&mainLoop.getBody()->back());

    Value k          = mainLoop.getInductionVar();
    Value acc        = mainLoop.getRegionIterArgs()[0];
    Value writeStage = mainLoop.getRegionIterArgs()[1];
    Value readStage  = mainLoop.getRegionIterArgs()[2];

    // CRITICAL: sync all warps before issuing new cp.async.
    rewriter.create<NVVM::Barrier0Op>(loc);

    // Issue next stage load (k + 2*TILE_K) if still in range
    Value nextK   = rewriter.create<arith::AddIOp>(loc, k, ci(2 * TILE_K));
    Value hasNext = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, nextK, cK);
    auto ifNext = rewriter.create<scf::IfOp>(loc,
        TypeRange{}, hasNext, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(rewriter);
      rewriter.setInsertionPointToStart(ifNext.thenBlock());
      if (!ifNext.thenBlock()->empty()) rewriter.eraseOp(&ifNext.thenBlock()->back());
      {
        issueStageLoad(rewriter, writeStage, nextK);
        rewriter.create<NVVM::CpAsyncWaitGroupOp>(loc,
            rewriter.getI32IntegerAttr(STAGES - 2));  // =1
        rewriter.create<NVVM::Barrier0Op>(loc);
        rewriter.create<scf::YieldOp>(loc);
      }

      rewriter.setInsertionPointToStart(ifNext.elseBlock());
      if (!ifNext.elseBlock()->empty()) rewriter.eraseOp(&ifNext.elseBlock()->back());
      {
        rewriter.create<NVVM::CpAsyncCommitGroupOp>(loc);
        rewriter.create<NVVM::CpAsyncWaitGroupOp>(loc, rewriter.getI32IntegerAttr(0));
        rewriter.create<NVVM::Barrier0Op>(loc);
        rewriter.create<scf::YieldOp>(loc);
      }
    }

    auto [vA, vB] = getStageViews(rewriter, readStage);

    // ── MMA: 4 (M-frags) x 8 (N-frags) per warp ──
    Value aLaneRow = rewriter.create<arith::DivUIOp>(loc, laneId, ci(4));
    Value aLaneCol = rewriter.create<arith::RemUIOp>(loc, laneId, ci(4));
    Value bLaneRow = rewriter.create<arith::RemUIOp>(loc, laneId, ci(4));
    Value bLaneCol = rewriter.create<arith::DivUIOp>(loc, laneId, ci(4));

    Value nextAcc = acc;

    for (int fi = 0; fi < (int)FR_M; ++fi) {
      Value aFrag = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(aFragTy));
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < (int)aK; ++c) {
          int rowAdd  = (r % 2) * 8;
          int colBase = (r / 2) * ((int)TILE_K / 2) + c * 4;
          Value sR = rewriter.create<arith::AddIOp>(loc, warpRowOff,
              rewriter.create<arith::AddIOp>(loc,
                  ci(fi * (int)MMA_M + rowAdd), aLaneRow));
          Value sC = rewriter.create<arith::AddIOp>(loc, ci(colBase), aLaneCol);
          Value sRMod  = rewriter.create<arith::AndIOp>(loc, sR, rewriter.create<arith::ConstantIndexOp>(loc, A_SWIZZLE_MASK));
          Value sXorA  = rewriter.create<arith::MulIOp>(loc, sRMod, rewriter.create<arith::ConstantIndexOp>(loc, A_SWIZZLE_STRIDE));
          Value sCSwizA = rewriter.create<arith::XOrIOp>(loc, sC, sXorA);
          Value e = rewriter.create<memref::LoadOp>(loc, vA, ValueRange{sR, sCSwizA});
          if (e.getType() != mmaInputTy)
            e = rewriter.create<arith::TruncFOp>(loc, mmaInputTy, e);
          aFrag = rewriter.create<vector::InsertOp>(loc, e, aFrag,
                      SmallVector<int64_t>{r, c});
        }
      }

      for (int fj = 0; fj < (int)FR_N; ++fj) {
        Value bFrag = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bFragTy));
        for (int r = 0; r < 2; ++r) {
          for (int c = 0; c < (int)bK; ++c) {
            int kRowBase = r * 4 + c * ((int)TILE_K / 2);
            Value sR = rewriter.create<arith::AddIOp>(loc, ci(kRowBase), bLaneRow);
            Value sC = rewriter.create<arith::AddIOp>(loc, warpColOff,
                rewriter.create<arith::AddIOp>(loc, ci(fj * (int)MMA_N), bLaneCol));
            Value sRMod = rewriter.create<arith::AndIOp>(loc, sR,
                              rewriter.create<arith::ConstantIndexOp>(loc, TILE_K - 1));
            Value sXorMsk = rewriter.create<arith::MulIOp>(loc, sRMod,
                                rewriter.create<arith::ConstantIndexOp>(loc, TILE_N / TILE_K));
            Value sCSwiz  = rewriter.create<arith::XOrIOp>(loc, sC, sXorMsk);
            Value e = rewriter.create<memref::LoadOp>(loc, vB, ValueRange{sR, sCSwiz});
            if (e.getType() != mmaInputTy)
              e = rewriter.create<arith::TruncFOp>(loc, mmaInputTy, e);
            bFrag = rewriter.create<vector::InsertOp>(loc, e, bFrag,
                        SmallVector<int64_t>{r, c});
          }
        }

        Value cSlice = rewriter.create<vector::ExtractOp>(loc, nextAcc,
                           SmallVector<int64_t>{fi, fj});
        Value mmaOut = rewriter.create<nvgpu::MmaSyncOp>(
            loc, aFrag, bFrag, cSlice, mmaShapeVals, useTF32).getResult();
        nextAcc = rewriter.create<vector::InsertOp>(loc, mmaOut, nextAcc,
                      SmallVector<int64_t>{fi, fj});
      }
    }

    // Advance stage indices
    Value nextWriteStage = rewriter.create<arith::RemUIOp>(loc,
        rewriter.create<arith::AddIOp>(loc, writeStage, c1), cStages);
    Value nextReadStage = rewriter.create<arith::RemUIOp>(loc,
        rewriter.create<arith::AddIOp>(loc, readStage, c1), cStages);

    rewriter.create<scf::YieldOp>(loc,
        ValueRange{nextAcc, nextWriteStage, nextReadStage});
  }

  Value fAcc = mainLoop.getResult(0);

  // ── Epilogue: store FR_M x FR_N x 2x2 accumulator tiles ──
  // Each thread in a warp writes 2x2 output elements per (fi, fj) MMA tile.
  // The 2 column elements (fc=0,1) are adjacent in memory → store as float2
  // (64-bit coalesced write) when tiles are aligned, else scalar with bounds check.
  Value laneDiv4 = rewriter.create<arith::DivUIOp>(loc, laneId, ci(4));
  Value laneMod4 = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::RemUIOp>(loc, laneId, ci(4)), ci(2));

  // Vector store type: vector<2 x resultElemTy> for float2 coalesced writes
  auto vecStoreTy = VectorType::get({2}, resultType.getElementType());

  for (int fi = 0; fi < (int)FR_M; ++fi) {
    for (int fj = 0; fj < (int)FR_N; ++fj) {
      // The MMA output is a vector<2x2xf32>
      Value mmaSlice = rewriter.create<vector::ExtractOp>(loc, fAcc,
                           SmallVector<int64_t>{fi, fj});
      // 2 rows: fr=0 (row offset 0) and fr=1 (row offset 8)
      for (int fr = 0; fr < 2; ++fr) {
        // Global row: warpGlobalRowBase + fi*MMA_M + fr*8 + laneId/4
        Value oR = rewriter.create<arith::AddIOp>(loc, warpGlobalRowBase,
            rewriter.create<arith::AddIOp>(loc,
                ci(fi * (int)MMA_M + fr * 8), laneDiv4));
        // Global col base (for both fc=0 and fc=1):
        // warpColBase + fj*MMA_N + (laneId%4)*2
        Value oC = rewriter.create<arith::AddIOp>(loc, warpColBase,
            rewriter.create<arith::AddIOp>(loc,
                ci(fj * (int)MMA_N), laneMod4));

        if (tileAligned) {
          // All tiles fully in-bounds: unconditional float2 store
          // Extract the 2-element row: mmaSlice[fr] → vector<2xf32>
          Value rowVec = rewriter.create<vector::ExtractOp>(loc, mmaSlice,
                             SmallVector<int64_t>{fr});
          // Truncate if result type differs from accumulator (e.g. f16 output)
          if (resultType.getElementType() != mmaAccTy &&
              llvm::isa<FloatType>(resultType.getElementType()))
            rowVec = rewriter.create<arith::TruncFOp>(loc, vecStoreTy, rowVec);

          Value fI = rewriter.create<arith::AddIOp>(loc, resBatchOff,
              rewriter.create<arith::AddIOp>(loc,
                  rewriter.create<arith::MulIOp>(loc, oR, cNStride), oC));
          // vector.store writes 2 consecutive elements → 64-bit coalesced
          rewriter.create<vector::StoreOp>(loc, rowVec, resFlat, ValueRange{fI});
        } else {
          // Boundary tiles: scalar stores with per-element bounds check
          for (int fc = 0; fc < 2; ++fc) {
            Value e = rewriter.create<vector::ExtractOp>(loc, mmaSlice,
                          SmallVector<int64_t>{fr, fc});
            Value tW = e;
            if (resultType.getElementType() != mmaAccTy &&
                llvm::isa<FloatType>(resultType.getElementType()))
              tW = rewriter.create<arith::TruncFOp>(loc, resultType.getElementType(), e);

            Value oCfc = rewriter.create<arith::AddIOp>(loc, oC, ci(fc));
            Value inR = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oR, cM);
            Value inC = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oCfc, cN);
            auto sIf = rewriter.create<scf::IfOp>(loc,
                rewriter.create<arith::AndIOp>(loc, inR, inC), false);
            {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(sIf.thenBlock());
              if (!sIf.thenBlock()->empty()) rewriter.eraseOp(&sIf.thenBlock()->back());
              Value fI = rewriter.create<arith::AddIOp>(loc, resBatchOff,
                  rewriter.create<arith::AddIOp>(loc,
                      rewriter.create<arith::MulIOp>(loc, oR, cNStride), oCfc));
              rewriter.create<memref::StoreOp>(loc, tW, resFlat, ValueRange{fI});
              rewriter.create<scf::YieldOp>(loc);
            }
          }
        }
      }
    }
  }

  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointAfter(launchOp);

  Value rT = rewriter.create<bufferization::ToTensorOp>(
      loc, resultType, resultMemRef, true).getResult();
  rewriter.replaceOp(op, rT);
  return success();
}
// ============================================================================
// Pattern entry point: three-way dispatch based on element type
// ============================================================================
struct NovaToGpuMatmulPattern : public OpRewritePattern<nova::MatmulOp> {
  using OpRewritePattern<nova::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();

    if (lhsRank < 2 || rhsRank < 1)
      return rewriter.notifyMatchFailure(op, "Unsupported matmul rank");

    int64_t M = lhsType.getShape()[lhsRank - 2];
    int64_t K = lhsType.getShape()[lhsRank - 1];
    int64_t N = rhsType.getShape()[rhsRank - 1];

    ArrayRef<int64_t> lhsBatchDims = lhsType.getShape().drop_back(2);
    int64_t totalBatches = 1;
    for (int64_t d : lhsBatchDims) totalBatches *= d;

    int64_t lhsBatchStride = (lhsRank > 2 && totalBatches > 1) ? (M * K) : 0;
    int64_t rhsBatchStride = 0;
    if (rhsRank > 2) {
      int64_t rhsBatches = 1;
      for (int64_t d : rhsType.getShape().drop_back(2))
        if (d != ShapedType::kDynamic) rhsBatches *= d;
      if (rhsBatches > 1) rhsBatchStride = K * N;
    }
    int64_t resBatchStride = M * N;

    SmallVector<int64_t> resultShape;
    if (totalBatches > 1) resultShape.push_back(totalBatches);
    resultShape.push_back(M);
    resultShape.push_back(N);

    auto i1Attr = rewriter.getI64IntegerAttr(1);
    int64_t lhsFlatSize = totalBatches * M * K;
    int64_t rhsFlatSize = (rhsRank > 2 ? totalBatches : 1) * K * N;
    int64_t resFlatSize = totalBatches * M * N;

    auto lhsFlatType = MemRefType::get({lhsFlatSize}, lhsType.getElementType(),
                                       MemRefLayoutAttrInterface{}, i1Attr);
    auto rhsFlatType = MemRefType::get({rhsFlatSize}, rhsType.getElementType(),
                                       MemRefLayoutAttrInterface{}, i1Attr);
    auto resFlatType = MemRefType::get({resFlatSize}, resultType.getElementType(),
                                       MemRefLayoutAttrInterface{}, i1Attr);
    auto lhsMemRefType = MemRefType::get(lhsType.getShape(), lhsType.getElementType(),
                                         MemRefLayoutAttrInterface{}, i1Attr);
    auto rhsMemRefType = MemRefType::get(rhsType.getShape(), rhsType.getElementType(),
                                         MemRefLayoutAttrInterface{}, i1Attr);
    auto resultMemRefType = MemRefType::get(resultShape, resultType.getElementType(),
                                            MemRefLayoutAttrInterface{}, i1Attr);

    Value lhsMemRef = rewriter.create<bufferization::ToBufferOp>(loc, lhsMemRefType, lhs).getResult();
    Value rhsMemRef = rewriter.create<bufferization::ToBufferOp>(loc, rhsMemRefType, rhs).getResult();
    // Allocate device output directly — avoids tensor.empty → to_buffer(space 1)
    // chain that bufferization resolves as host alloc + device alloc + memcpy.
    Value resultMemRef = rewriter.create<memref::AllocOp>(loc, resultMemRefType).getResult();

    Value lhsFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, lhsFlatType, lhsMemRef, (int64_t)0,
        ArrayRef<int64_t>{lhsFlatSize}, ArrayRef<int64_t>{1});
    Value rhsFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, rhsFlatType, rhsMemRef, (int64_t)0,
        ArrayRef<int64_t>{rhsFlatSize}, ArrayRef<int64_t>{1});
    Value resFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, resFlatType, resultMemRef, (int64_t)0,
        ArrayRef<int64_t>{resFlatSize}, ArrayRef<int64_t>{1});

    Type elemType = lhsType.getElementType();

    // ── Tensor Core tile-size thresholds ──
    // lowerTensorCoreMatmul needs at least one full TILE_M × TILE_N block to
    // be useful, and the 3-stage async pipeline requires K ≥ TILE_K.  For
    // matrices smaller than a single tile on any relevant dimension, fall back
    // to the scalar CUDA-core path which handles all boundary sizes correctly.
    const int64_t TC_TILE_M  = 128;
    const int64_t TC_TILE_N  = 128;
    const int64_t TC_TILE_K_TF32 = 8;
    const int64_t TC_TILE_K_FP16 = 16;

    // ── Three-way dispatch ──
    if (elemType.isF64()) {
      // f64: no Tensor Core support → scalar CUDA core tiled matmul
      return lowerScalarMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                               M, K, N, totalBatches, lhsBatchStride,
                               rhsBatchStride, resBatchStride,
                               resultType, resultShape, resultMemRef);
    } else if (elemType.isF32()) {
      // f32: TF32 Tensor Cores (Ampere sm_80+), mmaShape=[16,8,8]
      // Fall back to scalar for sub-tile dimensions to avoid pipeline issues.
      bool tcViable = (M >= TC_TILE_M) && (N >= TC_TILE_N) && (K >= TC_TILE_K_TF32);
      if (!tcViable)
        return lowerScalarMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                                 M, K, N, totalBatches, lhsBatchStride,
                                 rhsBatchStride, resBatchStride,
                                 resultType, resultShape, resultMemRef);
      return lowerTensorCoreMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                                   M, K, N, totalBatches, lhsBatchStride,
                                   rhsBatchStride, resBatchStride,
                                   resultType, resultShape, resultMemRef,
                                   /*useTF32=*/true);
    } else {
      // f16 / bf16: FP16 Tensor Cores, mmaShape=[16,8,16], no casting
      // Fall back to scalar for sub-tile dimensions.
      bool tcViable = (M >= TC_TILE_M) && (N >= TC_TILE_N) && (K >= TC_TILE_K_FP16);
      if (!tcViable)
        return lowerScalarMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                                 M, K, N, totalBatches, lhsBatchStride,
                                 rhsBatchStride, resBatchStride,
                                 resultType, resultShape, resultMemRef);
      return lowerTensorCoreMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                                   M, K, N, totalBatches, lhsBatchStride,
                                   rhsBatchStride, resBatchStride,
                                   resultType, resultShape, resultMemRef,
                                   /*useTF32=*/false);
    }
  }
};

void populateMatmulPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaToGpuMatmulPattern>(patterns.getContext());
}

} // namespace nova
} // namespace mlir