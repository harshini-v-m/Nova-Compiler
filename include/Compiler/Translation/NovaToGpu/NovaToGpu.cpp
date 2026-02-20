#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::nova;

namespace {

// Helper to create shared memory reduction for max/sum
// Returns the reduced value for the block (only thread 0 has the valid result)
Value createBlockReduce(PatternRewriter &rewriter, Location loc, Value val,
                        gpu::AllReduceOperation op, int blockDim) {
  // Use gpu.all_reduce for simplicity and efficiency within the block
  // This generates the necessary shuffle/shared memory ops
  mlir::gpu::AllReduceOperationAttr opAttr =
      mlir::gpu::AllReduceOperationAttr::get(rewriter.getContext(), op);
  return rewriter.create<gpu::AllReduceOp>(loc, val.getType(), val, opAttr,
                                           /*uniform=*/true);
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

    // Generalized Shapes: logits [D1, D2, ..., Dn, C], targets [D1, D2, ...,
    // Dn]
    auto logitsShape = logitsType.getShape();
    int64_t rank = logitsType.getRank();
    int64_t C = logitsShape[rank - 1];
    int64_t N = 1;
    for (int i = 0; i < rank - 1; ++i) {
      if (logitsShape[i] != ShapedType::kDynamic)
        N *= logitsShape[i];
    }

    // 1. Allocate memory for the FINAL loss (scalar, size [1])
    auto lossMemRefType = MemRefType::get({1}, resultType.getElementType(),
                                          MemRefLayoutAttrInterface{},
                                          rewriter.getI64IntegerAttr(1));

    Value lossMemRef =
        rewriter
            .create<gpu::AllocOp>(loc, lossMemRefType,
                                  /*asyncDependencies=*/ValueRange{},
                                  /*dynamicSizes=*/ValueRange{},
                                  /*symbolOperands=*/ValueRange{})
            .getMemref();

    // Initialize to 0.0
    Value zero_init = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultType.getElementType(), 0.0));
    rewriter.create<gpu::MemsetOp>(loc, Type(), ValueRange{}, lossMemRef,
                                   zero_init);

    // 2. Prepare Inputs (Convert Tensors to MemRefs)
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

    int64_t threadsPerBlock = 128;

    Value cGridSize = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value cBlockSize =
        rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto launchOp =
        rewriter.create<gpu::LaunchOp>(loc, cGridSize, c1, c1, // Grid: N, 1, 1
                                       cBlockSize, c1, c1 // Block: 128, 1, 1
        );

    // 4. Kernel Body
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value bid = rewriter.create<gpu::BlockIdOp>(
        loc, gpu::Dimension::x); // Sample ID (flattened)
    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);

    // Delinearize bid for targets and logits indexing
    SmallVector<Value> targetIndices;
    Value rem = bid;
    for (int i = rank - 2; i >= 0; --i) {
      Value dimSize =
          rewriter.create<arith::ConstantIndexOp>(loc, logitsShape[i]);
      if (i > 0) {
        targetIndices.push_back(
            rewriter.create<arith::RemUIOp>(loc, rem, dimSize));
        rem = rewriter.create<arith::DivUIOp>(loc, rem, dimSize);
      } else {
        targetIndices.push_back(rem);
      }
    }
    std::reverse(targetIndices.begin(), targetIndices.end());

    // Constants
    Value cC = rewriter.create<arith::ConstantIndexOp>(loc, C);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value neg_inf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value zero_f32 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));

    // A. Load Target Class for this sample
    Value targetIdxVal = rewriter.create<memref::LoadOp>(
        loc, targetsMemRef, ValueRange(targetIndices));
    Value targetIdx = targetIdxVal;
    if (!targetIdx.getType().isIndex())
      targetIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), targetIdxVal);

    // B. Pass 1: Find Max (Row-wise)
    Value initialMax = neg_inf;
    auto ptr = rewriter.create<scf::ForOp>(
        loc, tid, cC, bdim, ValueRange{initialMax},
        [&](OpBuilder &b, Location l, Value idx, ValueRange args) {
          SmallVector<Value> logitsIdx = targetIndices;
          logitsIdx.push_back(idx);
          Value val =
              b.create<memref::LoadOp>(l, logitsMemRef, ValueRange(logitsIdx));
          if (val.getType().isF16() || val.getType().isBF16())
            val = b.create<arith::ExtFOp>(l, b.getF32Type(), val);

          Value newMax = b.create<arith::MaximumFOp>(l, args[0], val);
          b.create<scf::YieldOp>(l, newMax);
        });

    Value threadMax = ptr.getResult(0);
    Value rowMax =
        createBlockReduce(rewriter, loc, threadMax,
                          gpu::AllReduceOperation::MAXIMUMF, threadsPerBlock);

    // C. Pass 2: Compute Sum(Exp) and Capture Target Logit
    Value initialSum = zero_f32;
    Value initialTargetLogit = zero_f32;

    auto loop2 = rewriter.create<scf::ForOp>(
        loc, tid, cC, bdim, ValueRange{initialSum, initialTargetLogit},
        [&](OpBuilder &b, Location l, Value idx, ValueRange args) {
          Value currSum = args[0];
          Value currTargetLogit = args[1];

          SmallVector<Value> logitsIdx = targetIndices;
          logitsIdx.push_back(idx);
          Value val = b.create<memref::LoadOp>(l, logitsMemRef, logitsIdx);
          if (val.getType().isF16() || val.getType().isBF16())
            val = b.create<arith::ExtFOp>(l, b.getF32Type(), val);

          Value diff = b.create<arith::SubFOp>(l, val, rowMax);
          Value expVal = b.create<mlir::math::ExpOp>(l, diff);
          Value newSum = b.create<arith::AddFOp>(l, currSum, expVal);

          Value isTarget = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq,
                                                   idx, targetIdx);
          Value zero = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0.0f));
          Value valIfTarget = b.create<arith::SelectOp>(l, isTarget, val, zero);
          Value newTargetLogit =
              b.create<arith::AddFOp>(l, currTargetLogit, valIfTarget);

          b.create<scf::YieldOp>(l, ValueRange{newSum, newTargetLogit});
        });

    Value threadSum = loop2.getResult(0);
    Value threadTargetLogit = loop2.getResult(1);

    Value rowSum =
        createBlockReduce(rewriter, loc, threadSum,
                          gpu::AllReduceOperation::ADD, threadsPerBlock);
    Value rowTargetLogit =
        createBlockReduce(rewriter, loc, threadTargetLogit,
                          gpu::AllReduceOperation::ADD, threadsPerBlock);

    // D. Compute Loss (Only Thread 0 writes)
    Value isMaster =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);

    scf::IfOp writeOp =
        rewriter.create<scf::IfOp>(loc, isMaster, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(writeOp.thenBlock());

    Value logSum = rewriter.create<mlir::math::LogOp>(loc, rowSum);
    Value term2 = rewriter.create<arith::SubFOp>(loc, rowTargetLogit, rowMax);
    Value loss = rewriter.create<arith::SubFOp>(loc, logSum, term2);

    // NORMALIZE by Total Samples N
    Value cN_f32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(static_cast<float>(N)));
    Value normalizedLoss = rewriter.create<arith::DivFOp>(loc, loss, cN_f32);

    // Cast back to output element type if needed
    Value finalLoss = normalizedLoss;
    Type elemType = resultType.getElementType();
    if (elemType.isF16() || elemType.isBF16()) {
      finalLoss =
          rewriter.create<arith::TruncFOp>(loc, elemType, normalizedLoss);
    }

    // Atomic Add to Output[0]
    rewriter.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf,
                                         finalLoss, lossMemRef, ValueRange{c0});

    rewriter.setInsertionPointAfter(writeOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    // 5. Result
    auto scalarTensorType =
        RankedTensorType::get({1}, resultType.getElementType());
    Value lossTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, scalarTensorType, lossMemRef, /*restrict=*/true,
        /*writable=*/true);

    rewriter.replaceOp(op, lossTensor);
    return success();
  }
};
} // namespace

namespace mlir {
namespace nova {

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
    auto outputType = cast<RankedTensorType>(op.getOutput().getType());
    auto memRefType = MemRefType::get(
        outputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));

    Value alloc = rewriter
                      .create<gpu::AllocOp>(loc, memRefType,
                                            /*asyncDependencies=*/ValueRange{},
                                            /*dynamicSizes=*/ValueRange{},
                                            /*symbolOperands=*/ValueRange{})
                      .getMemref();
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

    // 3. Create gpu.launch parameters
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

    // 1. Convert the input tensor to a memref so we can use memref ops
    auto inputMemRefType =
        MemRefType::get(inputType.getShape(), inputType.getElementType(),
                        MemRefLayoutAttrInterface{});

    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input)
            .getResult();

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
//cretae rewriter pattern for the gather op to gpu kernel from the novatolinalg pass
struct NovaToGpuGatherPattern : public OpRewritePattern<nova::GatherOp>{
  using OpRewritePattern<nova::GatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(nova::GatherOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value indices = op.getIndices();
    int64_t axis = op.getAxis();  

    auto inputType = cast<RankedTensorType> (input.getType());
    auto indiceType =cast<RankedTensorType> (indices.getType());
    auto resultType = cast<RankedTensorType> (op.getResult().getType());
    
    // bufferize — preserve GPU memory space from tensor encoding
    auto inputMemRefType = MemRefType::get(inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    auto indicesMemRefType = MemRefType::get(indiceType.getShape(), indiceType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value inputMemRef = rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input);
    Value indicesMemRef = rewriter.create<bufferization::ToBufferOp>(loc, indicesMemRefType, indices);
    auto outputMemrefType=MemRefType::get(resultType.getShape(), resultType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));

    Value outputMemRef = rewriter.create<gpu::AllocOp>(loc, outputMemrefType,
        /*asyncDependencies=*/ValueRange{},
        /*dynamicSizes=*/ValueRange{},
        /*symbolOperands=*/ValueRange{}).getMemref();

    int64_t totalElements = resultType.getNumElements();
    int64_t threadsPerBlock = 256;
    int64_t numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    Value cGrid = rewriter.create<arith::ConstantIndexOp>(loc, numBlocks);
    Value cBlock = rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    auto launchOp = rewriter.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlock, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value tid=rewriter.create<gpu::ThreadIdOp> (loc ,gpu::Dimension::x);
    Value bid=rewriter.create<gpu::BlockIdOp>(loc,gpu::Dimension::x);
    Value bdim=rewriter.create<gpu::BlockDimOp>(loc,gpu::Dimension::x);
    Value globalId= rewriter.create<arith::AddIOp>(loc,tid,rewriter.create<arith::MulIOp>(loc,bid,bdim));
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
      Value dimSize = rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]);
      if (i > 0) {
        outIndices[i] = rewriter.create<arith::RemUIOp>(loc, rem, dimSize);
        rem = rewriter.create<arith::DivUIOp>(loc, rem, dimSize);
      } else {
        outIndices[i] = rem;
      }
    }
    // Build input indices:
    //   Before axis dims: outIndices[0..axis-1]  (none if axis=0)
    //   At axis:         indices[outIndices[axis..axis+indicesRank-1]]  (the gather lookup)
    //   After axis:      outIndices[axis+indicesRank .. resRank-1]
    
    // Load the index value from indices tensor
    SmallVector<Value> idxAccessIndices;
    int64_t indicesRank = indiceType.getRank();
    for (int i = 0; i < indicesRank; ++i) {
      idxAccessIndices.push_back(outIndices[axis + i]);
    }
    Value gatherIdx = rewriter.create<memref::LoadOp>(loc, indicesMemRef, idxAccessIndices);
    
    // Cast to index type if needed
    if (!gatherIdx.getType().isIndex())
      gatherIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), gatherIdx);
    // Build the full input access indices
    SmallVector<Value> inputIndices;
    for (int64_t i = 0; i < axis; ++i)
      inputIndices.push_back(outIndices[i]);           // before axis
    inputIndices.push_back(gatherIdx);                  // at axis (gathered)
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
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    auto srcMemType = MemRefType::get(srcType.getShape(), srcType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    auto indicesMemType = MemRefType::get(indicesType.getShape(), indicesType.getElementType(),
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
      Value dimSize =
          rewriter.create<arith::ConstantIndexOp>(loc, srcShape[i]);
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
    Value idxVal =
        rewriter.create<memref::LoadOp>(loc, indicesMem,
                                        ValueRange{srcIndices[axis]});
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
    Value axisDim = rewriter.create<arith::ConstantIndexOp>(loc, resultType.getShape()[axis]);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value lowerBound = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, targetIdx, c0);
    Value upperBound = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, targetIdx, axisDim);
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


struct ToDeviceOpLowering : public OpRewritePattern<nova::ToDeviceOp> {
  using OpRewritePattern<nova::ToDeviceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ToDeviceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());

    // 1. Convert Input Tensor to Host MemRef
    // We assume the input is on host (default memory space)
    auto hostMemRefType =
        MemRefType::get(inputType.getShape(), inputType.getElementType());
    Value hostMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, hostMemRefType, input);

    // 2. Allocate GPU Memory (Device MemRef)
    // We need a MemRef type with memory space 1 (GPU Global)
    auto gpuMemRefType =
        MemRefType::get(inputType.getShape(), inputType.getElementType(),
                        MemRefLayoutAttrInterface{},
                        rewriter.getI64IntegerAttr(1)); // Memory Space 1

    Value gpuMemRef = rewriter
                          .create<gpu::AllocOp>(loc, gpuMemRefType,
                                                /*asyncDeps=*/ValueRange{},
                                                /*dynSizes=*/ValueRange{},
                                                /*symbolOperands=*/ValueRange{})
                          .getMemref();

    // 3. Copy Host -> Device
    rewriter.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{}, gpuMemRef,
                                   hostMemRef);

    // 4. Convert Device MemRef back to Tensor (for output)
    // The output tensor type should match the op's result type
    bufferization::ToTensorOp toTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, op.getType(),
                                                   gpuMemRef);

    // Optional: Restrict aliasing if you know ownership occurs here
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
                    bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target
        .addLegalDialect<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                         memref::MemRefDialect, tensor::TensorDialect,
                         bufferization::BufferizationDialect>();
    target.addIllegalOp<nova::ToDeviceOp>();
    patterns.add<NovaToGpuReducePattern>(context);
    patterns.add<ToDeviceOpLowering>(context);
    patterns.add<SceOpLowering>(context);
    patterns.add<NovaToGpuGatherPattern>(context);
    patterns.add<ScatterAddOpGpuLowering>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createNovaToGpuPass() {
  return std::make_unique<NovaToGpuPass>();
}

} // namespace nova
} // namespace mlir
