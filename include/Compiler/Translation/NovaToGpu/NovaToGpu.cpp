#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::nova;

namespace mlir {
namespace nova {

namespace {

struct Barrier0OpMemoryEffects
    : public MemoryEffectOpInterface::ExternalModel<Barrier0OpMemoryEffects,
                                                    NVVM::Barrier0Op> {
  void
  getEffects(Operation *op,
             SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) const {
    // Report both Read and Write effects to satisfy bufferization analysis.
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Read::get(),
                         SideEffects::DefaultResource::get());
  }
};

} // namespace
// Warp-level reduction using butterfly shuffle (__shfl_xor_sync).
// After reduction, ALL 32 threads hold the final reduced value.
// Avoids gpu.all_reduce which doesn't lower correctly in the JIT pipeline.
Value createWarpReduce(OpBuilder &rewriter, Location loc, Value val,
                       gpu::AllReduceOperation op) {
  Value width = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(32));

  for (int mask = 16; mask >= 1; mask >>= 1) {
    Value maskVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(mask));
    auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
        loc, val, maskVal, width, gpu::ShuffleMode::XOR);
    Value other = shuffleOp.getShuffleResult();

    switch (op) {
    case gpu::AllReduceOperation::ADD:
      val = rewriter.create<arith::AddFOp>(loc, val, other);
      break;
    case gpu::AllReduceOperation::MAXIMUMF:
      val = rewriter.create<arith::MaximumFOp>(loc, val, other);
      break;
    default:
      llvm_unreachable("Unsupported reduction op for warp reduce");
    }
  }
  return val; // All 32 threads hold the same reduced value
}

// Helper to create shared memory reduction for max/sum (used by other patterns)
// Returns the reduced value for the block (only thread 0 has the valid result)
Value createBlockReduce(OpBuilder &rewriter, Location loc, Value val,
                        gpu::AllReduceOperation op, int blockDim) {
  mlir::gpu::AllReduceOperationAttr opAttr =
      mlir::gpu::AllReduceOperationAttr::get(rewriter.getContext(), op);
  return rewriter
      .create<gpu::AllReduceOp>(loc, val.getType(), val, opAttr,
                                /*uniform=*/true)
      .getResult();
}

struct SceOpLowering : public OpRewritePattern<mlir::nova::SceOp> {
  using OpRewritePattern<mlir::nova::SceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::nova::SceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value logits = op.getLogits();
    Value targets = op.getTargets();
    auto logitsType = cast<RankedTensorType>(logits.getType());
    auto targetsType = cast<RankedTensorType>(targets.getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto logitsShape = logitsType.getShape();
    int64_t rank = logitsType.getRank();
    int64_t C = logitsShape[rank - 1];
    int64_t N = 1;
    for (int i = 0; i < rank - 1; ++i) {
      if (logitsShape[i] != ShapedType::kDynamic)
        N *= logitsShape[i];
    }

    Type elemTy = resultType.getElementType();

    // 1. Bufferize inputs
    auto logitsMemRefType =
        MemRefType::get(logitsType.getShape(), logitsType.getElementType());
    Value logitsMemRef =
        rewriter
            .create<bufferization::ToBufferOp>(loc, logitsMemRefType, logits)
            .getResult();

    auto targetsMemType =
        MemRefType::get(targetsType.getShape(), targetsType.getElementType());
    Value targetsMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, targetsMemType, targets)
            .getResult();

    // 2. Allocate per-row loss array on device (avoids non-deterministic atomics)
    auto rowLossMemRefType = MemRefType::get(
        {N}, rewriter.getF32Type(), MemRefLayoutAttrInterface{},
        rewriter.getI64IntegerAttr(1));
    Value rowLossMemRef =
        rewriter
            .create<gpu::AllocOp>(loc, rowLossMemRefType,
                                  /*asyncDependencies=*/ValueRange{},
                                  /*dynamicSizes=*/ValueRange{},
                                  /*symbolOperands=*/ValueRange{})
            .getMemref();

    // ====================================================================
    // Kernel 1: N blocks x 32 threads (1 warp). Each block computes one
    // row's cross-entropy loss using warp shuffle reduction (XOR butterfly
    // via gpu.shuffle). Thread 0 stores the row loss to rowLossMemRef[bid].
    // ====================================================================
    constexpr int64_t blockDim = 32;
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value cBlockDim = rewriter.create<arith::ConstantIndexOp>(loc, blockDim);

    auto launchOp =
        rewriter.create<gpu::LaunchOp>(loc, cN, c1, c1,       // Grid: N, 1, 1
                                       cBlockDim, c1, c1      // Block: 32, 1, 1
        );

    rewriter.setInsertionPointToStart(&launchOp.getBody().front());
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);

    Value zero_f32 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value neg_inf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value cC = rewriter.create<arith::ConstantIndexOp>(loc, C);
    Value c0_k = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Delinearize block index for multi-dim batch indexing
    SmallVector<Value> rowIndices;
    Value rem = bid;
    for (int i = rank - 2; i >= 0; --i) {
      Value dSize =
          rewriter.create<arith::ConstantIndexOp>(loc, logitsShape[i]);
      if (i > 0) {
        rowIndices.push_back(rewriter.create<arith::RemUIOp>(loc, rem, dSize));
        rem = rewriter.create<arith::DivUIOp>(loc, rem, dSize);
      } else {
        rowIndices.push_back(rem);
      }
    }
    std::reverse(rowIndices.begin(), rowIndices.end());

    // A. Load target class index
    Value targetIdxVal =
        rewriter.create<memref::LoadOp>(loc, targetsMemRef, rowIndices);
    Value targetIdx = targetIdxVal;
    if (!targetIdx.getType().isIndex())
      targetIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), targetIdxVal);

    // B. Parallel Pass 1: strided max over C classes
    //    Each of 32 threads processes elements tid, tid+32, tid+64, ...
    auto maxLoop = rewriter.create<scf::ForOp>(
        loc, tid, cC, bdim, ValueRange{neg_inf},
        [&](OpBuilder &ib, Location il, Value c_idx, ValueRange iargs) {
          SmallVector<Value> lIdx = rowIndices;
          lIdx.push_back(c_idx);
          Value val = ib.create<memref::LoadOp>(il, logitsMemRef, lIdx);
          if (val.getType().isF16() || val.getType().isBF16())
            val = ib.create<arith::ExtFOp>(il, ib.getF32Type(), val);
          Value newMax = ib.create<arith::MaximumFOp>(il, iargs[0], val);
          ib.create<scf::YieldOp>(il, newMax);
        });
    Value threadMax = maxLoop.getResult(0);

    // Warp shuffle reduction for max (XOR butterfly, all threads get result)
    Value rowMax =
        createWarpReduce(rewriter, loc, threadMax,
                         gpu::AllReduceOperation::MAXIMUMF);

    // C. Parallel Pass 2: strided sum(exp) and target logit extraction
    auto sumLoop = rewriter.create<scf::ForOp>(
        loc, tid, cC, bdim, ValueRange{zero_f32, zero_f32},
        [&](OpBuilder &ib, Location il, Value c_idx, ValueRange iargs) {
          SmallVector<Value> lIdx = rowIndices;
          lIdx.push_back(c_idx);
          Value val = ib.create<memref::LoadOp>(il, logitsMemRef, lIdx);
          if (val.getType().isF16() || val.getType().isBF16())
            val = ib.create<arith::ExtFOp>(il, ib.getF32Type(), val);

          Value diff = ib.create<arith::SubFOp>(il, val, rowMax);
          Value expVal = ib.create<mlir::math::ExpOp>(il, diff);
          Value newSum = ib.create<arith::AddFOp>(il, iargs[0], expVal);

          // Only the thread owning the target class picks up non-zero
          Value isTarget = ib.create<arith::CmpIOp>(
              il, arith::CmpIPredicate::eq, c_idx, targetIdx);
          Value zero =
              ib.create<arith::ConstantOp>(il, ib.getF32FloatAttr(0.0f));
          Value valIfTarget =
              ib.create<arith::SelectOp>(il, isTarget, val, zero);
          Value newTargetLogit =
              ib.create<arith::AddFOp>(il, iargs[1], valIfTarget);

          ib.create<scf::YieldOp>(il, ValueRange{newSum, newTargetLogit});
        });
    Value threadSum = sumLoop.getResult(0);
    Value threadTargetLogit = sumLoop.getResult(1);

    // Warp shuffle reduction for sum and target logit
    Value rowSum = createWarpReduce(rewriter, loc, threadSum,
                                    gpu::AllReduceOperation::ADD);
    Value targetLogit = createWarpReduce(rewriter, loc, threadTargetLogit,
                                         gpu::AllReduceOperation::ADD);

    // D. Compute row loss: log(sum_exp) - (target_logit - max)
    Value logSum = rewriter.create<mlir::math::LogOp>(loc, rowSum);
    Value t2 = rewriter.create<arith::SubFOp>(loc, targetLogit, rowMax);
    Value rowLoss = rewriter.create<arith::SubFOp>(loc, logSum, t2);

    // E. Thread 0 stores row loss (no atomics, deterministic)
    Value isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, tid, c0_k);
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, isThread0, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    rewriter.create<memref::StoreOp>(loc, rowLoss, rowLossMemRef,
                                     ValueRange{bid});

    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    // ====================================================================
    // Kernel 2: 1 block x 1 thread. Sequentially sums the N row losses
    // and divides by N. Deterministic — no atomics, no reduction races.
    // ====================================================================
    auto lossMemRefType = MemRefType::get({1}, elemTy,
                                          MemRefLayoutAttrInterface{},
                                          rewriter.getI64IntegerAttr(1));
    Value lossMemRef =
        rewriter
            .create<gpu::AllocOp>(loc, lossMemRefType,
                                  /*asyncDependencies=*/ValueRange{},
                                  /*dynamicSizes=*/ValueRange{},
                                  /*symbolOperands=*/ValueRange{})
            .getMemref();

    auto launchOp2 =
        rewriter.create<gpu::LaunchOp>(loc, c1, c1, c1,   // Grid:  1,1,1
                                       c1, c1, c1         // Block: 1,1,1
        );

    rewriter.setInsertionPointToStart(&launchOp2.getBody().front());

    Value c0_k2 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1_k2 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cN_k2 = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value zero_f32_k2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));

    auto sumRedLoop = rewriter.create<scf::ForOp>(
        loc, c0_k2, cN_k2, c1_k2, ValueRange{zero_f32_k2},
        [&](OpBuilder &ib, Location il, Value idx, ValueRange iargs) {
          Value v = ib.create<memref::LoadOp>(il, rowLossMemRef,
                                              ValueRange{idx});
          Value acc = ib.create<arith::AddFOp>(il, iargs[0], v);
          ib.create<scf::YieldOp>(il, acc);
        });
    Value totalLoss = sumRedLoop.getResult(0);

    // Divide by N to get mean loss
    Value cN_f32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(static_cast<float>(N)));
    Value meanLoss =
        rewriter.create<arith::DivFOp>(loc, totalLoss, cN_f32);

    // Store final scalar loss, casting if needed
    Value storeLoss = meanLoss;
    if (elemTy.isF16() || elemTy.isBF16())
      storeLoss = rewriter.create<arith::TruncFOp>(loc, elemTy, meanLoss);

    rewriter.create<memref::StoreOp>(loc, storeLoss, lossMemRef,
                                     ValueRange{c0_k2});

    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp2);

    // 3. Result
    auto scalarTensorType = RankedTensorType::get({1}, elemTy);
    Value lossTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, scalarTensorType, lossMemRef, /*restrict=*/true,
        /*writable=*/true);

    rewriter.replaceOp(op, lossTensor);
    return success();
  }
};

class NovaToGpuReducePattern : public OpRewritePattern<nova::ReduceOp> {
public:
  using OpRewritePattern<nova::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // if it is not scalar rerduction return failure
    if (cast<RankedTensorType>(op.getOutput().getType()).getShape().size() !=
        0) {
      // if all shapes is not 1 return fail
      auto shape = cast<RankedTensorType>(op.getOutput().getType()).getShape();
      for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] != 1) {
          return failure();
        }
      }
    }
    Value input = op.getInput();
    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto outputType = llvm::dyn_cast<RankedTensorType>(op.getOutput().getType());

    if (!inputType || !outputType)
      return failure();

    // Only support reduction to scalar for now in this direct GPU mapper
    // Partial reductions should fall back to NovaToLinalg
    if (outputType.getRank() != 0) {
       bool all_dims_reduced = true;
       if (auto dimensionAttr = op.getDimension()) {
         if (dimensionAttr->size() != inputType.getRank()) {
            all_dims_reduced = false;
         }
       } else {
          // If dimension is null, it's a full reduction to scalar (if output rank is 0).
          // If output rank > 0 but dimension is null, it's weird, but technically full reduction leads to rank 0.
          all_dims_reduced = (outputType.getRank() == 0);
       }
       if (!all_dims_reduced)
         return failure();
    }

    // 2. Map Reduction Kind to GPU Op attribute
    gpu::AllReduceOperation gpuOp;
    ReductionKind kind = op.getKind();

    switch (kind) {
    case ReductionKind::SUM:
      gpuOp = gpu::AllReduceOperation::ADD;
      break;
    case ReductionKind::PRODUCT:
      gpuOp = gpu::AllReduceOperation::MUL;
      break;
    case ReductionKind::MAX:
      gpuOp = gpu::AllReduceOperation::MAXIMUMF;
      break;
    case ReductionKind::MIN:
      gpuOp = gpu::AllReduceOperation::MINIMUMF;
      break;
    case ReductionKind::MEAN:
      gpuOp = gpu::AllReduceOperation::ADD;
      break;
    default:
      return failure(); // Unsupported for direct GPU mapping yet
    }

    // Initialize accumulator memory with identity value before launch
    Location loc = op.getLoc();
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // 3. initializing accumulator memory

    // Result type is got from output type
    // and we are creating the memref for it
    // (eg: 0-D tensor -> 0-D memref)
    auto memRefType = MemRefType::get(
        outputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));

    Value alloc = rewriter.create<memref::AllocOp>(loc, memRefType).getResult();
    // selecting accumulator initial value based on the reduction kind
    Value initialValue;
    Type elementType = inputType.getElementType();

    // For f16/bf16, we use f32 for accumulation
    Type accType = elementType;
    if (elementType.isF16() || elementType.isBF16()) {
      accType = rewriter.getF32Type();
    }

    float initValFloat = 0.0f;
    if (kind == ReductionKind::PRODUCT)
      initValFloat = 1.0f;
    else if (kind == ReductionKind::MAX)
      initValFloat = -std::numeric_limits<float>::infinity();
    else if (kind == ReductionKind::MIN)
      initValFloat = std::numeric_limits<float>::infinity();
    else if (kind == ReductionKind::MEAN)
      initValFloat = 0.0f;

    if (accType.isF32()) {
      initialValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(initValFloat));
    } else {
      initialValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(accType));
    }
    Value memsetVal = initialValue;
    if (elementType != accType) {
      memsetVal =
          rewriter.create<arith::TruncFOp>(loc, elementType, initialValue);
    }

    rewriter.create<gpu::MemsetOp>(loc, Type(), ValueRange{}, alloc, memsetVal);

    // 3. Convert input tensor to memref BEFORE launch (avoids gpu.alloc inside
    // kernel)
    auto inputMemRefType =
        MemRefType::get(inputType.getShape(), inputType.getElementType(),
                        MemRefLayoutAttrInterface{});
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input)
            .getResult();

    // 4. Create gpu.launch parameters
    int64_t numElements = inputType.getNumElements();
    int64_t threadsPerBlock = 128;
    int64_t numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    Value cGridSize = rewriter.create<arith::ConstantIndexOp>(loc, numBlocks);
    Value cBlockSize =
        rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock);

    auto launchOp =
        rewriter.create<gpu::LaunchOp>(loc, cGridSize, c1, c1, // Grid: M, 1, 1
                                       cBlockSize, c1, c1 // Block: 1024, 1, 1
        );

    // 4. Generate Body
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    // Get thread/block/grid IDs
    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value gdim = rewriter.create<gpu::GridDimOp>(loc, gpu::Dimension::x);

    // Calculate global ID: tid + bid * bdim
    Value bid_x_bdim = rewriter.create<arith::MulIOp>(loc, bid, bdim);
    Value globalId = rewriter.create<arith::AddIOp>(loc, tid, bid_x_bdim);

    // Calculate stride: bdim * gdim
    Value stride = rewriter.create<arith::MulIOp>(loc, bdim, gdim);

    // Initialize accumulator
    Value accumulator = initialValue;

    // Loop over elements: for (i = globalId; i < numElements; i += stride)
    Value lowerBound = globalId;
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, numElements);
    Value step = stride; // Grid stride

    scf::ForOp loop = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{accumulator});

    // Body of the loop
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());

      Value idx = loop.getInductionVar();
      Value currentAcc = loop.getRegionIterArgs()[0];

      // 2. Delinearization index calculation
      auto shape = inputType.getShape();
      int64_t rank = shape.size();
      SmallVector<Value> indices(rank);
      Value rem = idx;

      for (int i = rank - 1; i >= 0; --i) {
        Value dimSize = rewriter.create<arith::ConstantIndexOp>(loc, shape[i]);
        if (i > 0) {
          indices[i] = rewriter.create<arith::RemUIOp>(loc, rem, dimSize);
          rem = rewriter.create<arith::DivUIOp>(loc, rem, dimSize);
        } else {
          indices[i] = rem;
        }
      }

      Value val = rewriter.create<memref::LoadOp>(loc, inputMemRef, indices);

      // Cast loaded value to f32 if necessary
      Value calcVal = val;
      if (elementType != accType) {
        calcVal = rewriter.create<arith::ExtFOp>(loc, accType, val);
      }

      Value newAcc;
      switch (kind) {
      case ReductionKind::SUM:
        newAcc = rewriter.create<arith::AddFOp>(loc, currentAcc, calcVal);
        break;
      case ReductionKind::PRODUCT:
        newAcc = rewriter.create<arith::MulFOp>(loc, currentAcc, calcVal);
        break;
      case ReductionKind::MAX:
        newAcc = rewriter.create<arith::MaximumFOp>(loc, currentAcc, calcVal);
        break;
      case ReductionKind::MIN:
        newAcc = rewriter.create<arith::MinimumFOp>(loc, currentAcc, calcVal);
        break;
      case ReductionKind::MEAN:
        newAcc = rewriter.create<arith::AddFOp>(loc, currentAcc, calcVal);
        break;
      default:
        newAcc = currentAcc;
      }

      rewriter.create<scf::YieldOp>(loc, newAcc);
    }

    Value finalAccumulator = loop.getResult(0);

    // GPU All Reduce (Block Level)
    mlir::gpu::AllReduceOperationAttr opAttr =
        mlir::gpu::AllReduceOperationAttr::get(op.getContext(), gpuOp);
    Value reduced = rewriter.create<gpu::AllReduceOp>(
        loc, finalAccumulator.getType(), finalAccumulator, opAttr,
        /*uniform=*/true);

    // Store Result (Thread 0 of each block ONLY)
    // Then Atomic Update to Global
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isBlockMaster =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);

    scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, isBlockMaster,
                                                /*withElseRegion=*/false);
    // if the tid is 0 then update the output memory with atomic rmw
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    // Atomic RMW on the output memory
    SmallVector<Value> indices;

    for (int i = 0; i < outputType.getRank(); ++i)
      indices.push_back(c0);

    arith::AtomicRMWKind atomicKind;
    switch (kind) {
    case ReductionKind::SUM:
      atomicKind = arith::AtomicRMWKind::addf;
      break;
    case ReductionKind::PRODUCT:
      atomicKind = arith::AtomicRMWKind::mulf;
      break;
    case ReductionKind::MAX:
      atomicKind = arith::AtomicRMWKind::maximumf;
      break;
    case ReductionKind::MIN:
      atomicKind = arith::AtomicRMWKind::minimumf;
      break;
    case ReductionKind::MEAN:
      atomicKind = arith::AtomicRMWKind::addf;
      break;
    default:
      atomicKind = arith::AtomicRMWKind::addf;
    }

    // Cast reduced value back to element type for atomic op if needed
    Value atomicVal = reduced;
    if (reduced.getType() != elementType) {
      atomicVal = rewriter.create<arith::TruncFOp>(loc, elementType, reduced);
    }

    rewriter.create<memref::AtomicRMWOp>(loc, atomicKind, atomicVal, alloc,
                                         indices);

    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);
    // 5. Handle MEAN reduction (Division) with a second kernel
    if (kind == ReductionKind::MEAN) {
      Value c1_k2 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto launchOp2 = rewriter.create<gpu::LaunchOp>(loc, c1_k2, c1_k2, c1_k2,
                                                      c1_k2, c1_k2, c1_k2);

      rewriter.setInsertionPointToStart(&launchOp2.getBody().front());

      // Load current sum
      // Indices for 0-D memref are empty
      SmallVector<Value> indices;
      for (int i = 0; i < outputType.getRank(); ++i) {
        indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      Value sumVal = rewriter.create<memref::LoadOp>(loc, alloc, indices);

      // Create divisor constant
      Type elemType = inputType.getElementType();

      // Perform computation in f32 if input was f16/bf16
      Type computeType = elemType;
      if (elemType.isF16() || elemType.isBF16()) {
        computeType = rewriter.getF32Type();
        sumVal = rewriter.create<arith::ExtFOp>(loc, computeType, sumVal);
      }

      Value divisor;
      if (computeType.isF32() || computeType.isF64()) {
        divisor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(computeType, numElements));
      } else {
        divisor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(
                     elemType, numElements)); // Integer division not really
                                              // supported for mean usually
      }

      // Perform Division
      Value meanVal;
      if (isa<FloatType>(computeType)) {
        meanVal = rewriter.create<arith::DivFOp>(loc, sumVal, divisor);
      } else {
        meanVal = rewriter.create<arith::DivSIOp>(loc, sumVal, divisor);
      }

      // Cast back if needed
      if (meanVal.getType() != elemType) {
        meanVal = rewriter.create<arith::TruncFOp>(loc, elemType, meanVal);
      }

      // Store back
      rewriter.create<memref::StoreOp>(loc, meanVal, alloc, indices);
      rewriter.create<gpu::TerminatorOp>(loc);

      rewriter.setInsertionPointAfter(launchOp2);
    }

    // 6. Wrap the result memref into a tensor
    bufferization::ToTensorOp toTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, op.getType(), alloc);
    toTensor.setRestrict(true);

    rewriter.replaceOp(op, toTensor.getResult());

    return success();
  }
};
// cretae rewriter pattern for the gather op to gpu kernel from the novatolinalg
// pass
struct NovaToGpuGatherPattern : public OpRewritePattern<nova::GatherOp> {
  using OpRewritePattern<nova::GatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(nova::GatherOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value indices = op.getIndices();
    int64_t axis = op.getAxis();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto indiceType = cast<RankedTensorType>(indices.getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // bufferize — preserve GPU memory space from tensor encoding
    auto inputMemRefType = MemRefType::get(
        inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    auto indicesMemRefType = MemRefType::get(
        indiceType.getShape(), indiceType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input);
    Value indicesMemRef = rewriter.create<bufferization::ToBufferOp>(
        loc, indicesMemRefType, indices);
    auto outputMemrefType = MemRefType::get(
        resultType.getShape(), resultType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));

    Value outputMemRef =
        rewriter
            .create<gpu::AllocOp>(loc, outputMemrefType,
                                  /*asyncDependencies=*/ValueRange{},
                                  /*dynamicSizes=*/ValueRange{},
                                  /*symbolOperands=*/ValueRange{})
            .getMemref();

    int64_t totalElements = resultType.getNumElements();
    int64_t threadsPerBlock = 256;
    int64_t numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    Value cGrid = rewriter.create<arith::ConstantIndexOp>(loc, numBlocks);
    Value cBlock =
        rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto launchOp =
        rewriter.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlock, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value globalId = rewriter.create<arith::AddIOp>(
        loc, tid, rewriter.create<arith::MulIOp>(loc, bid, bdim));
    Value cTotal = rewriter.create<arith::ConstantIndexOp>(loc, totalElements);
    Value inBounds = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalId, cTotal);
    // ── 6. Bounds-guarded body ──
    auto ifOp = rewriter.create<scf::IfOp>(loc, inBounds, false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    // Delinearize globalId → [i0, i1, ..., i_{resRank-1}]
    // For result [B, T, C]: globalId → (b, t, c)
    auto resultShape = resultType.getShape();
    int64_t resRank = resultType.getRank();
    SmallVector<Value> outIndices(resRank);
    Value rem = globalId;
    for (int i = resRank - 1; i >= 0; --i) {
      Value dimSize =
          rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]);
      if (i > 0) {
        outIndices[i] = rewriter.create<arith::RemUIOp>(loc, rem, dimSize);
        rem = rewriter.create<arith::DivUIOp>(loc, rem, dimSize);
      } else {
        outIndices[i] = rem;
      }
    }
    // Build input indices:
    //   Before axis dims: outIndices[0..axis-1]  (none if axis=0)
    //   At axis:         indices[outIndices[axis..axis+indicesRank-1]]  (the
    //   gather lookup) After axis:      outIndices[axis+indicesRank ..
    //   resRank-1]

    // Load the index value from indices tensor
    SmallVector<Value> idxAccessIndices;
    int64_t indicesRank = indiceType.getRank();
    for (int i = 0; i < indicesRank; ++i) {
      idxAccessIndices.push_back(outIndices[axis + i]);
    }
    Value gatherIdx =
        rewriter.create<memref::LoadOp>(loc, indicesMemRef, idxAccessIndices);

    // Cast to index type if needed
    if (!gatherIdx.getType().isIndex())
      gatherIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), gatherIdx);
    // Build the full input access indices
    SmallVector<Value> inputIndices;
    for (int64_t i = 0; i < axis; ++i)
      inputIndices.push_back(outIndices[i]); // before axis

    // Clamp the gathered index to prevent OOB access and ILLEGAL_ADDRESS errors
    int64_t axisDimSize = inputType.getShape()[axis];
    Value cAxisDim = rewriter.create<arith::ConstantIndexOp>(loc, axisDimSize);
    Value cZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cMaxIdx = rewriter.create<arith::ConstantIndexOp>(loc, axisDimSize - 1);

    // signed comparison for negative check
    Value isNeg = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gatherIdx, cZero);
    Value clampedPos = rewriter.create<arith::SelectOp>(loc, isNeg, cZero, gatherIdx);
    Value isOOB = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, clampedPos, cAxisDim);
    Value finalIdx = rewriter.create<arith::SelectOp>(loc, isOOB, cMaxIdx, clampedPos);

    inputIndices.push_back(finalIdx);       // at axis (gathered and clamped)

    for (int64_t i = axis + 1; i < inputType.getRank(); ++i)
      inputIndices.push_back(outIndices[i + indicesRank - 1]); // after axis
    // Load from input, store to output
    Value val = rewriter.create<memref::LoadOp>(loc, inputMemRef, inputIndices);
    rewriter.create<memref::StoreOp>(loc, val, outputMemRef, outIndices);
    // ── 7. Close kernel ──
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);
    // ── 8. Output ──
    Value resultTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, outputMemRef, /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, resultTensor);
    return success();
  }
};

// ── ScatterAdd: Direct GPU kernel ──────────────────────────────────
// Instead of lowering to scf::ParallelOp (which breaks in the
// scf→affine→gpu pipeline due to address-space mismatches),
// we emit a gpu.launch directly — one thread per source element,
// each doing AtomicRMWOp to scatter into the destination.
struct ScatterAddOpGpuLowering : public OpRewritePattern<nova::ScatterAddOp> {
  using OpRewritePattern<nova::ScatterAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ScatterAddOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elementTy = resultType.getElementType();
    Value input = op.getInput();
    Value indices = op.getIndices();
    Value src = op.getSrc();
    int64_t axis = op.getAxis();
    int64_t inputRank = resultType.getRank();

    if (axis < 0)
      axis += inputRank;

    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcShape = srcType.getShape();
    int64_t srcRank = srcType.getRank();
    auto indicesType = cast<RankedTensorType>(indices.getType());

    // 1. Bufferize operands to MemRef — preserve GPU memory space
    auto inputMemType = MemRefType::get(resultType.getShape(), elementTy,
                                        MemRefLayoutAttrInterface{},
                                        rewriter.getI64IntegerAttr(1));
    auto srcMemType = MemRefType::get(
        srcType.getShape(), srcType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    auto indicesMemType = MemRefType::get(
        indicesType.getShape(), indicesType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));

    Value inputMem =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemType, input,
                                                   /*restrict=*/true);
    Value srcMem =
        rewriter.create<bufferization::ToBufferOp>(loc, srcMemType, src,
                                                   /*restrict=*/true);
    Value indicesMem =
        rewriter.create<bufferization::ToBufferOp>(loc, indicesMemType, indices,
                                                   /*restrict=*/true);

    // 2. Compute total src elements
    int64_t totalSrcElements = 1;
    for (int64_t i = 0; i < srcRank; ++i)
      totalSrcElements *= srcShape[i];

    // 3. Launch kernel: one thread per src element
    int64_t threadsPerBlock = 256;
    int64_t numBlocks =
        (totalSrcElements + threadsPerBlock - 1) / threadsPerBlock;

    Value cGrid = rewriter.create<arith::ConstantIndexOp>(loc, numBlocks);
    Value cBlock =
        rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto launchOp =
        rewriter.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlock, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    // 4. Compute globalId = bid * blockDim + tid
    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value globalId = rewriter.create<arith::AddIOp>(
        loc, tid, rewriter.create<arith::MulIOp>(loc, bid, bdim));

    Value cTotal =
        rewriter.create<arith::ConstantIndexOp>(loc, totalSrcElements);
    Value inBounds = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalId, cTotal);

    // 5. Bounds-guarded body
    auto ifOp = rewriter.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    // 6. Delinearize globalId → src coordinates [s0, s1, ..., s_{srcRank-1}]
    SmallVector<Value> srcIndices(srcRank);
    Value rem = globalId;
    for (int i = srcRank - 1; i >= 0; --i) {
      Value dimSize = rewriter.create<arith::ConstantIndexOp>(loc, srcShape[i]);
      if (i > 0) {
        srcIndices[i] = rewriter.create<arith::RemUIOp>(loc, rem, dimSize);
        rem = rewriter.create<arith::DivUIOp>(loc, rem, dimSize);
      } else {
        srcIndices[i] = rem;
      }
    }

    // 7. Load index value: indices[srcIndices[axis]]
    //    For 1D indices: indices[srcIndices[axis]]
    //    For nD indices: same shape as src's axis dimensions
    Value idxVal = rewriter.create<memref::LoadOp>(
        loc, indicesMem, ValueRange{srcIndices[axis]});
    // Cast to index
    Value targetIdx = idxVal;
    if (!targetIdx.getType().isIndex())
      targetIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), idxVal);

    // 8. Load value from src
    Value val = rewriter.create<memref::LoadOp>(loc, srcMem, srcIndices);

    // 9. Build destination coordinates: replace axis dim with gathered index
    SmallVector<Value> dstCoords;
    for (int64_t d = 0; d < srcRank; ++d) {
      if (d == axis)
        dstCoords.push_back(targetIdx);
      else
        dstCoords.push_back(srcIndices[d]);
    }

    // 9b. Bounds Check for safety (prevents CUDA_ERROR_ILLEGAL_ADDRESS)
    Value axisDim = rewriter.create<arith::ConstantIndexOp>(
        loc, resultType.getShape()[axis]);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value lowerBound = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, targetIdx, c0);
    Value upperBound = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, targetIdx, axisDim);
    Value isSafe = rewriter.create<arith::AndIOp>(loc, lowerBound, upperBound);

    auto safeIfOp = rewriter.create<scf::IfOp>(loc, isSafe, /*withElse=*/false);
    rewriter.setInsertionPointToStart(safeIfOp.thenBlock());

    // 10. Atomic scatter-add into input
    arith::AtomicRMWKind kind = llvm::isa<FloatType>(elementTy)
                                    ? arith::AtomicRMWKind::addf
                                    : arith::AtomicRMWKind::addi;
    rewriter.create<memref::AtomicRMWOp>(loc, kind, val, inputMem, dstCoords);

    rewriter.setInsertionPointAfter(safeIfOp);

    // 11. Close kernel
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    // 12. Convert result back to tensor
    Value resultTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, inputMem, /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, resultTensor);
    return success();
  }
};

// Check if a value is provably uninitialized, meaning a host-to-device
// copy would just transfer garbage. In that case we can skip the copy
// and only allocate device memory.
static bool isUninitializedInput(Value input) {
  Operation *defOp = input.getDefiningOp();
  if (!defOp)
    return false;

  // tensor.empty produces uninitialized memory
  if (isa<tensor::EmptyOp>(defOp))
    return true;

  // bufferization.to_tensor wrapping a block argument memref that has
  // no stores — this is an output-only buffer from the JIT runtime
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defOp)) {
    Value memref = toTensor.getBuffer();
    if (isa<BlockArgument>(memref)) {
      bool hasStores = false;
      for (auto &use : memref.getUses()) {
        Operation *user = use.getOwner();
        if (user == defOp)
          continue;
        if (isa<memref::DeallocOp>(user))
          continue;
        // Any other use means the memref may be written to or read
        hasStores = true;
        break;
      }
      if (!hasStores)
        return true;
    }
  }

  return false;
}

struct ToDeviceOpLowering : public OpRewritePattern<nova::ToDeviceOp> {
  using OpRewritePattern<nova::ToDeviceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ToDeviceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());

    // GPU MemRef type with memory space 1 (device)
    auto gpuMemRefType =
        MemRefType::get(inputType.getShape(), inputType.getElementType(),
                        MemRefLayoutAttrInterface{},
                        rewriter.getI64IntegerAttr(1));

    // Fast path: if the input is provably uninitialized (e.g. tensor.empty
    // or an output-only block arg), skip the host alloc + copy entirely
    // and just allocate device memory.
    if (isUninitializedInput(input)) {
      Value gpuMemRef =
          rewriter
              .create<gpu::AllocOp>(loc, gpuMemRefType,
                                    /*asyncDeps=*/ValueRange{},
                                    /*dynSizes=*/ValueRange{},
                                    /*symbolOperands=*/ValueRange{})
              .getMemref();

      auto toTensor = rewriter.create<bufferization::ToTensorOp>(
          loc, op.getType(), gpuMemRef);
      toTensor.setRestrict(true);
      rewriter.replaceOp(op, toTensor.getResult());
      return success();
    }

    // Default path: full host-to-device copy

    // 1. Convert Input Tensor to Host MemRef
    auto hostMemRefType =
        MemRefType::get(inputType.getShape(), inputType.getElementType());
    Value hostMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, hostMemRefType, input);

    // 2. Allocate GPU Memory
    Value gpuMemRef = rewriter
                          .create<gpu::AllocOp>(loc, gpuMemRefType,
                                                /*asyncDeps=*/ValueRange{},
                                                /*dynSizes=*/ValueRange{},
                                                /*symbolOperands=*/ValueRange{})
                          .getMemref();

    // 3. Copy Host -> Device
    rewriter.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{}, gpuMemRef,
                                   hostMemRef);

    // 4. Convert Device MemRef back to Tensor
    bufferization::ToTensorOp toTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, op.getType(),
                                                   gpuMemRef);
    toTensor.setRestrict(true);

    rewriter.replaceOp(op, toTensor.getResult());
    return success();
  }
};

struct NovaToGpuPass
    : public PassWrapper<NovaToGpuPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToGpuPass)

  StringRef getArgument() const override { return "convert-nova-to-gpu"; }
  StringRef getDescription() const override {
    return "Lower nova to gpu dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, math::MathDialect,
                    func::FuncDialect, NVVM::NVVMDialect, nvgpu::NVGPUDialect,
                    vector::VectorDialect>();

    // Register the side-effect interface for nvvm.barrier0.
    registry.addExtension(
        +[](MLIRContext *context, NVVM::NVVMDialect *dialect) {
          NVVM::Barrier0Op::attachInterface<Barrier0OpMemoryEffects>(*context);
        });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<
        arith::ArithDialect, gpu::GPUDialect, scf::SCFDialect,
        memref::MemRefDialect, func::FuncDialect, math::MathDialect,
        tensor::TensorDialect, bufferization::BufferizationDialect,
        NVVM::NVVMDialect, nvgpu::NVGPUDialect, vector::VectorDialect>();
    target.addLegalOp<gpu::BarrierOp>();
    target.addIllegalOp<nova::ToDeviceOp, nova::SceOp, nova::MatmulOp,
                        nova::GatherOp, nova::ScatterAddOp>();
 //   patterns.add<NovaToGpuReducePattern>(context);
    patterns.add<ToDeviceOpLowering>(context);
    patterns.add<SceOpLowering>(context);
    patterns.add<NovaToGpuGatherPattern>(context);
    patterns.add<ScatterAddOpGpuLowering>(context);
    populateMatmulPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createNovaToGpuPass() {
  return std::make_unique<NovaToGpuPass>();
}

} // namespace nova
} // namespace mlir