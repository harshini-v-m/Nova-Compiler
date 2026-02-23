#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace nova {

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

    // Require at least 2D on LHS and at least 1D on RHS.
    if (lhsRank < 2 || rhsRank < 1)
      return rewriter.notifyMatchFailure(op, "Unsupported matmul rank");

    // The last two dims of LHS are [M, K], last two dims of RHS are [K, N].
    int64_t M = lhsType.getShape()[lhsRank - 2];
    int64_t K = lhsType.getShape()[lhsRank - 1];
    int64_t N = rhsType.getShape()[rhsRank - 1];

    // --- Batch dimension flattening ---
    ArrayRef<int64_t> lhsBatchDims = lhsType.getShape().drop_back(2);

    int64_t totalBatches = 1;
    for (int64_t d : lhsBatchDims)
      totalBatches *= d;

    // Stride = number of elements to advance per batch
    int64_t lhsBatchStride = (lhsRank > 2 && totalBatches > 1) ? (M * K) : 0;
    int64_t rhsBatchStride = 0;
    if (rhsRank > 2) {
      int64_t rhsBatches = 1;
      for (int64_t d : rhsType.getShape().drop_back(2)) {
        if (d != ShapedType::kDynamic) {
          rhsBatches *= d;
        }
      }
      if (rhsBatches > 1) {
        rhsBatchStride = K * N;
      }
    }


    int64_t resBatchStride = M * N;

    // Result shape: [totalBatches, M, N] if batched, else [M, N]
    SmallVector<int64_t> resultShape;
    if (totalBatches > 1)
      resultShape.push_back(totalBatches);
    resultShape.push_back(M);
    resultShape.push_back(N);

    // ---- MemRef types (GPU global memory, addr space 1) ----
    // Flatten LHS + RHS to 2D for clean indexing inside the kernel.
    // For batched: lhsFlat=[totalBatches*M, K], rhsFlat=[totalBatches*K, N]
    // We actually keep them as-is at the type level; we'll index manually.
    // Use 1D flat memrefs for the batch + matrix index arithmetic.
    int64_t lhsFlatSize = totalBatches * M * K;
    int64_t rhsFlatSize = (rhsRank > 2 ? totalBatches : 1) * K * N;
    int64_t resFlatSize = totalBatches * M * N;

    auto i1Attr = rewriter.getI64IntegerAttr(1);
    auto lhsFlatType = MemRefType::get({lhsFlatSize}, lhsType.getElementType(),
                                       MemRefLayoutAttrInterface{}, i1Attr);
    auto rhsFlatType = MemRefType::get({rhsFlatSize}, rhsType.getElementType(),
                                       MemRefLayoutAttrInterface{}, i1Attr);
    auto resFlatType = MemRefType::get({resFlatSize}, resultType.getElementType(),
                                       MemRefLayoutAttrInterface{}, i1Attr);

    // Original typed memrefs for bufferization
    auto lhsMemRefType = MemRefType::get(lhsType.getShape(), lhsType.getElementType(),
                                         MemRefLayoutAttrInterface{}, i1Attr);
    auto rhsMemRefType = MemRefType::get(rhsType.getShape(), rhsType.getElementType(),
                                         MemRefLayoutAttrInterface{}, i1Attr);
    auto resultMemRefType = MemRefType::get(resultShape, resultType.getElementType(),
                                            MemRefLayoutAttrInterface{}, i1Attr);

    Value lhsMemRef = rewriter.create<bufferization::ToBufferOp>(loc, lhsMemRefType, lhs).getResult();
    Value rhsMemRef = rewriter.create<bufferization::ToBufferOp>(loc, rhsMemRefType, rhs).getResult();

    // Allocate result
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultShape, resultType.getElementType());
    Value resultMemRef = rewriter.create<bufferization::ToBufferOp>(loc, resultMemRefType, emptyTensor).getResult();

    // Cast lhs, rhs, and result to flat 1D views for index arithmetic
    Value lhsFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, lhsFlatType, lhsMemRef,
        /*offset=*/(int64_t)0,
        /*sizes=*/ArrayRef<int64_t>{lhsFlatSize},
        /*strides=*/ArrayRef<int64_t>{1});
    Value rhsFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, rhsFlatType, rhsMemRef,
        /*offset=*/(int64_t)0,
        /*sizes=*/ArrayRef<int64_t>{rhsFlatSize},
        /*strides=*/ArrayRef<int64_t>{1});
    Value resFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, resFlatType, resultMemRef,
        /*offset=*/(int64_t)0,
        /*sizes=*/ArrayRef<int64_t>{resFlatSize},
        /*strides=*/ArrayRef<int64_t>{1});

    // ---- Shared memory constants ----
    int64_t tileSize = 16;
    int64_t elemByteSize = lhsType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t tileByteSize = tileSize * tileSize * elemByteSize;
    int64_t totalSharedBytes = 2 * tileByteSize;

    Value cM         = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value cN         = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value cK         = rewriter.create<arith::ConstantIndexOp>(loc, K);
    Value cTileSize  = rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
    Value c1         = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cTileBytes = rewriter.create<arith::ConstantIndexOp>(loc, tileByteSize);
    Value cTotalBatches = rewriter.create<arith::ConstantIndexOp>(loc, totalBatches);
    Value cLhsBatchStride = rewriter.create<arith::ConstantIndexOp>(loc, lhsBatchStride);
    Value cRhsBatchStride = rewriter.create<arith::ConstantIndexOp>(loc, rhsBatchStride);
    Value cResBatchStride = rewriter.create<arith::ConstantIndexOp>(loc, resBatchStride);
    Value cLhsKStride     = rewriter.create<arith::ConstantIndexOp>(loc, K);   // stride along K for A
    Value cRhsNStride     = rewriter.create<arith::ConstantIndexOp>(loc, N);   // stride along N for B

    Value dynamicSharedMemSize = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(),
        rewriter.create<arith::ConstantIndexOp>(loc, totalSharedBytes));

    Value gridX = rewriter.create<arith::DivUIOp>(loc,
        rewriter.create<arith::AddIOp>(loc, cN,
            rewriter.create<arith::ConstantIndexOp>(loc, tileSize - 1)), cTileSize);
    Value gridY = rewriter.create<arith::DivUIOp>(loc,
        rewriter.create<arith::AddIOp>(loc, cM,
            rewriter.create<arith::ConstantIndexOp>(loc, tileSize - 1)), cTileSize);
    // Grid Z = totalBatches (1 for pure 2D matmul, >1 for batched)
    Value gridZ = cTotalBatches;

    // ---- GPU Launch ----
    auto launchOp = rewriter.create<gpu::LaunchOp>(
        loc,
        gridX, gridY, gridZ,
        cTileSize, cTileSize, c1,
        /*dynamicSharedMemorySize=*/dynamicSharedMemSize);

    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    // ---- Inside the kernel ----
    MLIRContext *ctx = rewriter.getContext();
    auto workgroupAddrSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
    auto i8DynMemRefType = MemRefType::get({ShapedType::kDynamic},
                                           rewriter.getIntegerType(8),
                                           AffineMap{},
                                           workgroupAddrSpace);
    Value shmem = rewriter.create<gpu::DynamicSharedMemoryOp>(loc, i8DynMemRefType);

    auto tileMemRefType = MemRefType::get({tileSize, tileSize},
                                          lhsType.getElementType(),
                                          AffineMap{},
                                          workgroupAddrSpace);
    Value c0k = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value tileA = rewriter.create<memref::ViewOp>(
        loc, tileMemRefType, shmem, c0k, /*sizes=*/ValueRange{});
    Value tileB = rewriter.create<memref::ViewOp>(
        loc, tileMemRefType, shmem, cTileBytes, /*sizes=*/ValueRange{});

    // Thread/block indices
    Value tx = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value ty = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value bx = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value by = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
    Value bz = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::z); // batch index

    // Global output row/col for this thread
    Value col = rewriter.create<arith::AddIOp>(loc, rewriter.create<arith::MulIOp>(loc, bx, cTileSize), tx);
    Value row = rewriter.create<arith::AddIOp>(loc, rewriter.create<arith::MulIOp>(loc, by, cTileSize), ty);

    // Compute batch base offsets for each operand
    Value lhsBatchOff = rewriter.create<arith::MulIOp>(loc, bz, cLhsBatchStride);
    Value rhsBatchOff = rewriter.create<arith::MulIOp>(loc, bz, cRhsBatchStride);
    Value resBatchOff = rewriter.create<arith::MulIOp>(loc, bz, cResBatchStride);

    Value sum_init = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));

    // ---- Main tiling loop over K ----
    auto kLoop = rewriter.create<scf::ForOp>(loc, c0k, cK, cTileSize, ValueRange{sum_init});
    rewriter.setInsertionPointToStart(kLoop.getBody());

    Value kOffset    = kLoop.getInductionVar();
    Value currentSum = kLoop.getRegionIterArgs()[0];

    // A element linear index = lhsBatchOff + row*K + aCol
    Value aCol = rewriter.create<arith::AddIOp>(loc, kOffset, tx);
    Value aInBounds = rewriter.create<arith::AndIOp>(loc,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, cM),
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, aCol, cK));

    auto ifA = rewriter.create<scf::IfOp>(loc, aInBounds, /*withElse=*/true);
    rewriter.setInsertionPointToStart(ifA.thenBlock());
    {
      // lhsFlat[ lhsBatchOff + row*K + aCol ]
      Value rowOffset  = rewriter.create<arith::MulIOp>(loc, row, cLhsKStride);
      Value aLinear    = rewriter.create<arith::AddIOp>(loc,
                            rewriter.create<arith::AddIOp>(loc, lhsBatchOff, rowOffset), aCol);
      Value aVal = rewriter.create<memref::LoadOp>(loc, lhsFlat, ValueRange{aLinear});
      rewriter.create<memref::StoreOp>(loc, aVal, tileA, ValueRange{ty, tx});
    }
    rewriter.setInsertionPointToStart(ifA.elseBlock());
    rewriter.create<memref::StoreOp>(loc, sum_init, tileA, ValueRange{ty, tx});
    rewriter.setInsertionPointAfter(ifA);

    // B element linear index = rhsBatchOff + bRow*N + col
    Value bRow = rewriter.create<arith::AddIOp>(loc, kOffset, ty);
    Value bInBounds = rewriter.create<arith::AndIOp>(loc,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, bRow, cK),
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, cN));

    auto ifB = rewriter.create<scf::IfOp>(loc, bInBounds, /*withElse=*/true);
    rewriter.setInsertionPointToStart(ifB.thenBlock());
    {
      // rhsFlat[ rhsBatchOff + bRow*N + col ]
      Value bRowOffset = rewriter.create<arith::MulIOp>(loc, bRow, cRhsNStride);
      Value bLinear    = rewriter.create<arith::AddIOp>(loc,
                            rewriter.create<arith::AddIOp>(loc, rhsBatchOff, bRowOffset), col);
      Value bVal = rewriter.create<memref::LoadOp>(loc, rhsFlat, ValueRange{bLinear});
      rewriter.create<memref::StoreOp>(loc, bVal, tileB, ValueRange{ty, tx});
    }
    rewriter.setInsertionPointToStart(ifB.elseBlock());
    rewriter.create<memref::StoreOp>(loc, sum_init, tileB, ValueRange{ty, tx});
    rewriter.setInsertionPointAfter(ifB);

    // Barrier 1: all threads done loading shared memory tiles
    {
      rewriter.create<NVVM::Barrier0Op>(loc);
    }

    // Compute partial dot product from shared memory tiles.
    // Use InsertionGuard to safely scope the innerLoop body build.
    auto innerLoop = rewriter.create<scf::ForOp>(loc, c0k, cTileSize, c1, ValueRange{currentSum});
    // Save the kLoop body block before entering innerLoop scope
    Block *kBlock = kLoop.getBody();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      // The ForOp block might already have an implicit yield — erase it only if present
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
    } // InsertionGuard restores insertion point

    Value loopResult = innerLoop.getResult(0);

    // EXPLICITLY insert barrier 2 and kLoop yield into kBlock using Block iterator.
    // This guarantees correct placement regardless of InsertionGuard restore.
    {
      // Erase implicit kLoop yield if it already exists
      if (!kBlock->empty() && kBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
        rewriter.eraseOp(&kBlock->back());
      // Set insertion point explicitly to the END of kBlock
      rewriter.setInsertionPoint(kBlock, kBlock->end());
    }
    // Barrier 2: all threads done using shared-memory tiles before next K-tile iteration.
    {
      rewriter.create<NVVM::Barrier0Op>(loc);
    }

    rewriter.create<scf::YieldOp>(loc, loopResult);

    // ---- Store result ----
    rewriter.setInsertionPointAfter(kLoop);
    Value finalResult = kLoop.getResult(0);

    Value outBounds = rewriter.create<arith::AndIOp>(loc,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, cM),
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, col, cN));

    auto ifOut = rewriter.create<scf::IfOp>(loc, outBounds, /*withElse=*/false);
    rewriter.setInsertionPointToStart(ifOut.thenBlock());
    {
      // resFlat[ resBatchOff + row*N + col ]
      Value rowOffsetR = rewriter.create<arith::MulIOp>(loc, row, cRhsNStride); // N stride
      Value rLinear    = rewriter.create<arith::AddIOp>(loc,
                            rewriter.create<arith::AddIOp>(loc, resBatchOff, rowOffsetR), col);
      rewriter.create<memref::StoreOp>(loc, finalResult, resFlat, ValueRange{rLinear});
    }
    rewriter.setInsertionPointAfter(ifOut);

    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    // ---- Wrap result memref back to tensor ----
    Value resultTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, resultMemRef, /*restrict=*/true).getResult();
    rewriter.replaceOp(op, resultTensor);

    return success();
  }
};

void populateMatmulPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaToGpuMatmulPattern>(patterns.getContext());
}

} // namespace nova
} // namespace mlir