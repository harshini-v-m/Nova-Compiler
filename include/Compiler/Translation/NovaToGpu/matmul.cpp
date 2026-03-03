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
//  - aPtr / bPtr (flat offsets into global memory) are carried as
//    scf.for iter_args and incremented by TILE_K each iteration.
//    No multiply-heavy index recomputation inside the loop body.
//  - 3-stage async pipeline with prolog loads before loop entry.
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
  // Warp layout: 2 warps in M-dim, 4 warps in N-dim (8 warps total, 256 threads)
  const int64_t TILE_M      = 128;
  const int64_t TILE_N      = 128;
  // TF32 MMA uses m16n8k8 → TILE_K = 8
  // FP16 MMA uses m16n8k16 → TILE_K = 16
  const int64_t TILE_K      = useTF32 ? 8 : 16;
  const int64_t WARPS       [[maybe_unused]] = 8;
  const int64_t WARP_M_CNT  = 2;
  const int64_t WARP_N_CNT  = 4;
  const int64_t WARP_M      = TILE_M / WARP_M_CNT;  // 64
  const int64_t WARP_N      = TILE_N / WARP_N_CNT;  // 32
  const int64_t THREADS     = 256;
  const int64_t STAGES      = 3;

  // MMA shape: m16n8k8 (TF32) or m16n8k16 (FP16)
  const int64_t MMA_M = 16, MMA_N = 8;
  const int64_t FR_M  = WARP_M / MMA_M;  // 4
  const int64_t FR_N  = WARP_N / MMA_N;  // 4

  // Total elements per stage
  const int64_t A_ELEMS = TILE_M * TILE_K;  // 128*8=1024 (TF32) or 128*16=2048 (FP16)
  const int64_t B_ELEMS = TILE_K * TILE_N;  // 8*128=1024 (TF32) or 16*128=2048 (FP16)

  // B loading: B_ROWS_PER_PASS = THREADS/TILE_N = 256/128 = 2
  // B_LOAD_ITERS = TILE_K / B_ROWS_PER_PASS = 8/2=4 (TF32) or 16/2=8 (FP16)
  const int64_t B_ROWS_PER_PASS = THREADS / TILE_N;          // 2
  const int64_t B_LOAD_ITERS   = TILE_K / B_ROWS_PER_PASS;  // 4 (TF32) or 8 (FP16)
  const int64_t B_TOTAL_ROWS   [[maybe_unused]] = TILE_K;

  SmallVector<int64_t> mmaShapeVals = {MMA_M, MMA_N, TILE_K};


  // ── Element types and MMA fragment types ──
  Type mmaInputTy  = useTF32 ? rewriter.getF32Type() : rewriter.getF16Type();
  Type mmaAccTy    = rewriter.getF32Type();
  int64_t aK = useTF32 ? 1 : 2;  // k-tiles per A fragment
  int64_t bK = useTF32 ? 1 : 2;  // k-tiles per B fragment
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
  Value cBRowsPerPass = ci(B_ROWS_PER_PASS);

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
  Value bx     = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value by     = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
  Value laneId = rewriter.create<arith::RemUIOp>(loc, tid, c32);
  Value warpId = rewriter.create<arith::DivUIOp>(loc, tid, c32);

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



  // ── Per-thread tile load offsets for A (hoisted) ──
  // A tile is TILE_M x TILE_K = 2048 elements, 256 threads → each thread loads 8 A-elements.
  // We unroll 8 passes: pass i loads row = (tid + i*256) / TILE_K, col = (tid+i*256)%TILE_K
  // For simplicity in MLIR: treat each element independently with in-bounds guard.
  Value aTid       = rewriter.create<arith::RemUIOp>(loc, tid, ci(A_ELEMS));
  Value aInBounds  = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tid, ci(A_ELEMS));
  Value aTileRow   = rewriter.create<arith::DivUIOp>(loc, aTid, cTileK);
  Value aTileCol   = rewriter.create<arith::RemUIOp>(loc, aTid, cTileK);
  Value aGlobalRow = rewriter.create<arith::AddIOp>(loc, rowBase, aTileRow);

  // ── Per-thread tile load offsets for B (hoisted) ──
  // B tile is TILE_K x TILE_N = 2048 elements, 256 threads → each thread maps to 1 B element
  // at bTileRow = tid / TILE_N, bTileCol = tid % TILE_N.
  Value bTid       = rewriter.create<arith::RemUIOp>(loc, tid, ci(B_ELEMS));
  Value bInBounds  = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tid, ci(B_ELEMS));
  Value bTileRow   = rewriter.create<arith::DivUIOp>(loc, bTid, cTileN);
  Value bTileCol   = rewriter.create<arith::RemUIOp>(loc, bTid, cTileN);
  Value bGlobalCol = rewriter.create<arith::AddIOp>(loc, blockColBase, bTileCol);

  // ── Pointer induction base offsets ──
  // aPtr0 = lhsBatchOff + aGlobalRow * K + aTileCol  (points to k=0)
  // bPtr0 = rhsBatchOff + bTileRow   * N + bGlobalCol

  Value aPtr0 = rewriter.create<arith::AddIOp>(loc, lhsBatchOff,
      rewriter.create<arith::AddIOp>(loc,
          rewriter.create<arith::MulIOp>(loc, aGlobalRow, cKStride),
          aTileCol));
  Value bPtr0 = rewriter.create<arith::AddIOp>(loc, rhsBatchOff,
      rewriter.create<arith::AddIOp>(loc,
          rewriter.create<arith::MulIOp>(loc, bTileRow, cNStride),
          bGlobalCol));

  // ── Async copy helpers ──
  // asyncCopyA: copies one A element from global to shared (bounds-clamped).
  auto asyncCopyA = [&](OpBuilder &b, Value sA, Value aPtr,
                        Value inBoundsRow, Value inBoundsK) {
    Value inBounds = b.create<arith::AndIOp>(loc,
        aInBounds,
        b.create<arith::AndIOp>(loc, inBoundsRow, inBoundsK));
    Value safePtr = b.create<arith::SelectOp>(loc, inBounds, aPtr, c0);
    return b.create<nvgpu::DeviceAsyncCopyOp>(
        loc, sA, ValueRange{aTileRow, aTileCol},
        lhsFlat, ValueRange{safePtr},
        rewriter.getIndexAttr(1), Value{}, nullptr);
  };

  // asyncCopyB: copies one B element into swizzled shared mem position.
  auto asyncCopyB = [&](OpBuilder &b, Value sB, Value bPtr,
                        Value inBoundsRow, Value inBoundsCol) {
    Value inBounds = b.create<arith::AndIOp>(loc,
        bInBounds,
        b.create<arith::AndIOp>(loc, inBoundsRow, inBoundsCol));
    Value safePtr = b.create<arith::SelectOp>(loc, inBounds, bPtr, c0);
    // XOR swizzle: swizzled_col = col ^ ((row & 15) * 8)
    // For TILE_N=128 (7 bits): XOR by values 0,8,16..120 keeps result in [0,127].
    Value bRowMod16 = b.create<arith::AndIOp>(loc, bTileRow,
                          b.create<arith::ConstantIndexOp>(loc, 15));
    Value bXorMask  = b.create<arith::MulIOp>(loc, bRowMod16,
                          b.create<arith::ConstantIndexOp>(loc, 8));
    Value bColSwiz  = b.create<arith::XOrIOp>(loc, bTileCol, bXorMask);
    return b.create<nvgpu::DeviceAsyncCopyOp>(
        loc, sB, ValueRange{bTileRow, bColSwiz},
        rhsFlat, ValueRange{safePtr},
        rewriter.getIndexAttr(1), Value{}, nullptr);
  };

  // ── issueStageLoad: loads one async stage from global → shared ──
  // Emits B_LOAD_ITERS passes to cover all TILE_K rows of sB.
  // Pass i covers sB rows [i*B_ROWS_PER_PASS .. (i+1)*B_ROWS_PER_PASS - 1].
  // bPtr points to the K-step base (row bTileRow of B in global mem).
  // For pass i the global src offset is bPtr + i*B_ROWS_PER_PASS*N.
  auto issueStageLoad = [&](OpBuilder &b, Value stageIdx,
                             Value aPtr, Value bPtr) -> Value {
    auto [sA, sB] = getStageViews(b, stageIdx);

    // A copy (one element per thread)
    Value aRowOk = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                            aGlobalRow, cM);
    Value tokA = asyncCopyA(b, sA, aPtr, aRowOk,
                             b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                 aTileCol, cTileK));

    Value bColOk = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                            bGlobalCol, cN);

    // B copies: unrolled over B_LOAD_ITERS passes
    SmallVector<Value> tokens{tokA};
    for (int64_t i = 0; i < B_LOAD_ITERS; ++i) {
      // Shared mem destination row for this pass
      Value bRowI = b.create<arith::AddIOp>(loc, bTileRow,
                        b.create<arith::ConstantIndexOp>(loc, i * B_ROWS_PER_PASS));
      // XOR swizzle for bank-conflict-free store
      // Pattern: col ^ ((row & (TILE_K-1)) * (TILE_N / TILE_K))
      // With TILE_K=8, TILE_N=128: col ^ ((row & 7) * 16)
      int64_t swizzleStride = TILE_N / TILE_K; // 16 for TF32, 8 for FP16
      Value bRowMod = b.create<arith::AndIOp>(loc, bRowI,
                          b.create<arith::ConstantIndexOp>(loc, TILE_K - 1));
      Value bXorMask = b.create<arith::MulIOp>(loc, bRowMod,
                           b.create<arith::ConstantIndexOp>(loc, swizzleStride));
      Value bColSwiz = b.create<arith::XOrIOp>(loc, bTileCol, bXorMask);
      // Global source: bPtr + i * B_ROWS_PER_PASS * N elements from base
      Value bPtrI = (i == 0) ? bPtr
                              : b.create<arith::AddIOp>(loc, bPtr,
                                    b.create<arith::ConstantIndexOp>(loc, i * B_ROWS_PER_PASS * N));
      // B row bounds: row i must be < TILE_K (always true by construction)
      Value bRowOk = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                              bRowI, cTileK);
      // Combined in-bounds for this element
      Value inBounds = b.create<arith::AndIOp>(loc,
          bInBounds,
          b.create<arith::AndIOp>(loc, bRowOk, bColOk));
      Value safePtr = b.create<arith::SelectOp>(loc, inBounds, bPtrI, c0);
      tokens.push_back(b.create<nvgpu::DeviceAsyncCopyOp>(
          loc, sB, ValueRange{bRowI, bColSwiz},
          rhsFlat, ValueRange{safePtr},
          rewriter.getIndexAttr(1), Value{}, nullptr));
    }

    return b.create<nvgpu::DeviceAsyncCreateGroupOp>(
        loc, nvgpu::DeviceAsyncTokenType::get(ctx), tokens);
  };


  // ── Prolog: Load stages 0 and 1 ──
  Value tok0 = issueStageLoad(rewriter, c0, aPtr0, bPtr0);
  rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange{tok0});

  Value aPtr1 = rewriter.create<arith::AddIOp>(loc, aPtr0, cAStride);
  Value bPtr1 = rewriter.create<arith::AddIOp>(loc, bPtr0, cBStride);
  Value hasStage1 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                    cK, cTileK);
  auto ifS1 = rewriter.create<scf::IfOp>(loc, nvgpu::DeviceAsyncTokenType::get(ctx),
                                          hasStage1, /*withElse=*/true);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ifS1.thenBlock());
    if (!ifS1.thenBlock()->empty()) rewriter.eraseOp(&ifS1.thenBlock()->back());
    Value t1 = issueStageLoad(rewriter, c1, aPtr1, bPtr1);
    rewriter.create<scf::YieldOp>(loc, t1);

    rewriter.setInsertionPointToStart(ifS1.elseBlock());
    if (!ifS1.elseBlock()->empty()) rewriter.eraseOp(&ifS1.elseBlock()->back());
    rewriter.create<scf::YieldOp>(loc, tok0);
  }
  Value prologToken = ifS1.getResult(0);

  // ── Accumulator init: FR_M x FR_N x 2 x 2 floats = 0 ──
  Value cZeroAcc = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(mmaAccTy));
  Value cInit    = rewriter.create<vector::SplatOp>(loc, cFragTy, cZeroAcc);

  // Starting pointer values for k=2*TILE_K (what the loop will issue first)
  Value aPtr2 = rewriter.create<arith::AddIOp>(loc, aPtr1, cAStride);
  Value bPtr2 = rewriter.create<arith::AddIOp>(loc, bPtr1, cBStride);

  // ── Main K-loop — iter_args: (acc, token, writeStage, readStage, aPtr, bPtr) ──
  Value cWriteStage0 = rewriter.create<arith::RemUIOp>(loc, c2, cStages);  // =2
  Value cReadStage0  = c0;

  auto mainLoop = rewriter.create<scf::ForOp>(
      loc, c0, cK, cTileK,
      ValueRange{cInit, prologToken, cWriteStage0, cReadStage0, aPtr2, bPtr2});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mainLoop.getBody());
    if (!mainLoop.getBody()->empty())
      rewriter.eraseOp(&mainLoop.getBody()->back());

    Value k          = mainLoop.getInductionVar();
    Value acc        = mainLoop.getRegionIterArgs()[0];
    Value currToken  = mainLoop.getRegionIterArgs()[1];
    Value writeStage = mainLoop.getRegionIterArgs()[2];
    Value readStage  = mainLoop.getRegionIterArgs()[3];
    Value aPtr       = mainLoop.getRegionIterArgs()[4];
    Value bPtr       = mainLoop.getRegionIterArgs()[5];

    // Issue next stage load (k + 2*TILE_K) if still in range
    Value nextK   = rewriter.create<arith::AddIOp>(loc, k, ci(2 * TILE_K));
    Value hasNext = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, nextK, cK);
    auto ifNext = rewriter.create<scf::IfOp>(loc,
        nvgpu::DeviceAsyncTokenType::get(ctx), hasNext, true);
    {
      OpBuilder::InsertionGuard g2(rewriter);
      rewriter.setInsertionPointToStart(ifNext.thenBlock());
      if (!ifNext.thenBlock()->empty()) rewriter.eraseOp(&ifNext.thenBlock()->back());
      rewriter.create<scf::YieldOp>(loc, issueStageLoad(rewriter, writeStage, aPtr, bPtr));

      rewriter.setInsertionPointToStart(ifNext.elseBlock());
      if (!ifNext.elseBlock()->empty()) rewriter.eraseOp(&ifNext.elseBlock()->back());
      rewriter.create<scf::YieldOp>(loc, currToken);
    }
    Value nextToken = ifNext.getResult(0);

    // Wait for the read stage to be ready
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(
        loc, TypeRange{}, nextToken, rewriter.getI32IntegerAttr(1));
    rewriter.create<NVVM::Barrier0Op>(loc);

    auto [vA, vB] = getStageViews(rewriter, readStage);

    // ── MMA: 4 (M-frags) x 4 (N-frags) per warp ──
    // For TF32 m16n8k8: A frag = [4,1], B frag = [2,1], C frag = [2,2]
    // Lane layout (same as V18):
    //   A: laneDiv = lane/(TILE_K/aK), laneMod = lane%(TILE_K/aK)
    //      sR = warpRowFrag*MMA_M + r*(MMA_M/4) + laneDiv
    //      sC = laneMod
    //   B: kLane = lane/4, colLane = (lane%4)*2
    //      sR = kLane + bK_shift, sC_swizzled
    Value laneDiv  = rewriter.create<arith::DivUIOp>(loc, laneId, ci(TILE_K / aK));
    Value laneMod  = rewriter.create<arith::RemUIOp>(loc, laneId, ci(TILE_K / aK));
    Value kLane    = rewriter.create<arith::DivUIOp>(loc, laneId, ci(4));
    Value colLane  = rewriter.create<arith::MulIOp>(loc,
        rewriter.create<arith::RemUIOp>(loc, laneId, ci(4)), ci(2));

    Value nextAcc = acc;

    for (int fi = 0; fi < (int)FR_M; ++fi) {
      // A fragment for this M-fragment (shared across all FR_N N-fragments)
      Value aFrag = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(aFragTy));
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < (int)aK; ++c) {
          // Row in sA: warp M-offset + fragment M-offset + lane contribution
          // warpRowOff was computed above (warpRow * WARP_M)
          // within this warp: fi * MMA_M + r*(MMA_M/4) + laneDiv
          Value sR = rewriter.create<arith::AddIOp>(loc, warpRowOff,
              rewriter.create<arith::AddIOp>(loc,
                  ci(fi * (int)MMA_M + r * ((int)MMA_M / 4)), laneDiv));
          Value sC = rewriter.create<arith::AddIOp>(loc,
              ci(c * ((int)TILE_K / (int)aK)), laneMod);
          Value e = rewriter.create<memref::LoadOp>(loc, vA, ValueRange{sR, sC});
          if (e.getType() != mmaInputTy)
            e = rewriter.create<arith::TruncFOp>(loc, mmaInputTy, e);
          aFrag = rewriter.create<vector::InsertOp>(loc, e, aFrag,
                      SmallVector<int64_t>{r, c});
        }
      }

      for (int fj = 0; fj < (int)FR_N; ++fj) {
        // B fragment for this N-fragment of the warp
        Value bFrag = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bFragTy));
        for (int r = 0; r < 2; ++r) {
          for (int c = 0; c < (int)bK; ++c) {
            Value sR = rewriter.create<arith::RemUIOp>(loc,
                rewriter.create<arith::AddIOp>(loc, ci(r * ((int)TILE_K / 2)), kLane),
                cTileK);
            // sC within warp's N-fragment: fj*MMA_N + c*(WARP_N/bK) + colLane
            Value sC = rewriter.create<arith::AddIOp>(loc, warpColOff,
                rewriter.create<arith::AddIOp>(loc,
                    ci(fj * (int)MMA_N + c * ((int)WARP_N / (int)bK)),
                    colLane));
            // XOR swizzle (must match write-side: col ^ ((row & (TILE_K-1)) * (TILE_N/TILE_K)))
            // TF32: (row & 7) * 16,  FP16: (row & 15) * 8
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

        // Extract accumulator sub-tile, run MMA, insert result back
        Value cSlice = rewriter.create<vector::ExtractOp>(loc, nextAcc,
                           SmallVector<int64_t>{fi, fj});
        Value mmaOut = rewriter.create<nvgpu::MmaSyncOp>(
            loc, aFrag, bFrag, cSlice, mmaShapeVals, useTF32).getResult();
        nextAcc = rewriter.create<vector::InsertOp>(loc, mmaOut, nextAcc,
                      SmallVector<int64_t>{fi, fj});
      }
    }

    // Advance stage indices and pointers
    Value nextWriteStage = rewriter.create<arith::RemUIOp>(loc,
        rewriter.create<arith::AddIOp>(loc, writeStage, c1), cStages);
    Value nextReadStage = rewriter.create<arith::RemUIOp>(loc,
        rewriter.create<arith::AddIOp>(loc, readStage, c1), cStages);
    Value nextAPtr  = rewriter.create<arith::AddIOp>(loc, aPtr, cAStride);
    Value nextBPtr  = rewriter.create<arith::AddIOp>(loc, bPtr, cBStride);

    rewriter.create<scf::YieldOp>(loc,
        ValueRange{nextAcc, nextToken, nextWriteStage, nextReadStage, nextAPtr, nextBPtr});
  }

  Value fAcc = mainLoop.getResult(0);

  // ── Epilogue: store FR_M x FR_N x 2x2 accumulator tiles ──
  // Each thread in a warp writes 2x2 output elements per (fi, fj) MMA tile.
  // Row base for (fi, fj) tile: warpGlobalRowBase + fi*MMA_M + laneId/4
  // Col base for (fi, fj) tile: warpColBase + fj*MMA_N + (laneId%4)*2
  Value laneDiv4 = rewriter.create<arith::DivUIOp>(loc, laneId, ci(4));
  Value laneMod4 = rewriter.create<arith::MulIOp>(loc,
      rewriter.create<arith::RemUIOp>(loc, laneId, ci(4)), ci(2));

  for (int fi = 0; fi < (int)FR_M; ++fi) {
    for (int fj = 0; fj < (int)FR_N; ++fj) {
      // The MMA output is a vector<2x2xf32>
      Value mmaSlice = rewriter.create<vector::ExtractOp>(loc, fAcc,
                           SmallVector<int64_t>{fi, fj});
      // 2 rows (index 0,1) x 2 cols (index 0,1) in the 2x2 C fragment
      for (int fr = 0; fr < 2; ++fr) {
        for (int fc = 0; fc < 2; ++fc) {
          Value e = rewriter.create<vector::ExtractOp>(loc, mmaSlice,
                        SmallVector<int64_t>{fr, fc});
          Value tW = e;
          if (resultType.getElementType() != mmaAccTy &&
              llvm::isa<FloatType>(resultType.getElementType()))
            tW = rewriter.create<arith::TruncFOp>(loc, resultType.getElementType(), e);

          // Global row: warpGlobalRowBase + fi*MMA_M + fr*8 + laneId/4
          Value oR = rewriter.create<arith::AddIOp>(loc, warpGlobalRowBase,
              rewriter.create<arith::AddIOp>(loc,
                  ci(fi * (int)MMA_M + fr * 8), laneDiv4));
          // Global col: warpColBase + fj*MMA_N + fc + (laneId%4)*2
          Value oC = rewriter.create<arith::AddIOp>(loc, warpColBase,
              rewriter.create<arith::AddIOp>(loc,
                  ci(fj * (int)MMA_N + fc), laneMod4));

          Value inR = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oR, cM);
          Value inC = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oC, cN);
          auto sIf = rewriter.create<scf::IfOp>(loc,
              rewriter.create<arith::AndIOp>(loc, inR, inC), false);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(sIf.thenBlock());
            if (!sIf.thenBlock()->empty()) rewriter.eraseOp(&sIf.thenBlock()->back());
            Value fI = rewriter.create<arith::AddIOp>(loc, resBatchOff,
                rewriter.create<arith::AddIOp>(loc,
                    rewriter.create<arith::MulIOp>(loc, oR, cNStride), oC));
            rewriter.create<memref::StoreOp>(loc, tW, resFlat, ValueRange{fI});
            rewriter.create<scf::YieldOp>(loc);
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
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultShape, resultType.getElementType());
    Value resultMemRef = rewriter.create<bufferization::ToBufferOp>(loc, resultMemRefType, emptyTensor).getResult();

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

    // ── Three-way dispatch ──
    if (elemType.isF64()) {
      // f64: no Tensor Core support → scalar CUDA core tiled matmul
      return lowerScalarMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                               M, K, N, totalBatches, lhsBatchStride,
                               rhsBatchStride, resBatchStride,
                               resultType, resultShape, resultMemRef);
    } else if (elemType.isF32()) {
      // f32: TF32 Tensor Cores (Ampere sm_80+), mmaShape=[16,8,8]
      return lowerTensorCoreMatmul(op, rewriter, lhsFlat, rhsFlat, resFlat,
                                   M, K, N, totalBatches, lhsBatchStride,
                                   rhsBatchStride, resBatchStride,
                                   resultType, resultShape, resultMemRef,
                                   /*useTF32=*/true);
    } else {
      // f16 / bf16: FP16 Tensor Cores, mmaShape=[16,8,16], no casting
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