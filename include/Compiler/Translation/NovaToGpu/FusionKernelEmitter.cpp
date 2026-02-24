#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>

using namespace mlir;
using namespace mlir::nova;

namespace {

// Helper to create shared memory reduction for max/sum
Value createBlockReduce(OpBuilder &rewriter, Location loc, Value val,
                        gpu::AllReduceOperation op, int blockDim) {
  mlir::gpu::AllReduceOperationAttr opAttr =
      mlir::gpu::AllReduceOperationAttr::get(rewriter.getContext(), op);
  return rewriter
      .create<gpu::AllReduceOp>(loc, val.getType(), val, opAttr,
                                /*uniform=*/true)
      .getResult();
}

// The greedy walker that absorbs linalg.generic users into the kernel
std::pair<Value, SmallVector<Operation *>>
AbsorbElementwise(Value tensorResult, // The tensor result of the previous op
                                      // (e.g. nova.reduce output)
                  Value scalarResult, // The local scalar result in the kernel
                  Value reduceInput,  // The original input to the reduce op
                  const SmallVectorImpl<Value>
                      &reduceIndices, // Indices for loading reduceInput
                  IRMapping &mapper, gpu::LaunchOp launchOp,
                  PatternRewriter &rewriter) {
  // Record the current scalar result for this tensor result
  mapper.map(tensorResult, scalarResult);
  Value latestScalar = scalarResult;
  SmallVector<Operation *> fusedOps;

  for (Operation *user : tensorResult.getUsers()) {
    auto genericOp = dyn_cast<linalg::GenericOp>(user);
    if (!genericOp)
      continue;

    // Check if all inputs are either in mapping or match the reduce input
    bool allOperandsAvailable = true;
    for (Value operand : genericOp.getInputs()) {
      if (mapper.contains(operand))
        continue;
      if (operand == reduceInput)
        continue; // We can handle this by loading it later
      allOperandsAvailable = false;
      break;
    }
    if (!allOperandsAvailable)
      continue;

    // Found a fuseable consumer!
    IRMapping bodyMapper;
    for (auto it : llvm::enumerate(genericOp.getInputs())) {
      Value inputTensor = it.value();
      if (!mapper.contains(inputTensor) && inputTensor == reduceInput) {
        // Generate a load for the original reduce input using the specified
        // indices
        auto tensorType = cast<RankedTensorType>(inputTensor.getType());
        auto memRefType = MemRefType::get(
            tensorType.getShape(), tensorType.getElementType(),
            MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
        Value memRef = rewriter.create<bufferization::ToBufferOp>(
            genericOp->getLoc(), memRefType, inputTensor);
        Value loadedVal = rewriter.create<memref::LoadOp>(
            genericOp->getLoc(), memRef, reduceIndices);
        mapper.map(inputTensor, loadedVal);
      }
      Value localScalar = mapper.lookup(inputTensor);
      bodyMapper.map(genericOp.getRegion().front().getArgument(it.index()),
                     localScalar);
    }

    for (auto &op : genericOp.getRegion().front().without_terminator()) {
      rewriter.clone(op, bodyMapper);
    }

    auto yieldOp =
        cast<linalg::YieldOp>(genericOp.getRegion().front().getTerminator());
    Value newScalarResult = bodyMapper.lookup(yieldOp.getOperand(0));

    // Recurse to find more consumers and update latestScalar
    auto recursed = AbsorbElementwise(
        genericOp->getResults()[0], newScalarResult, reduceInput, reduceIndices,
        mapper, launchOp, rewriter);
    latestScalar = recursed.first;
    fusedOps.push_back(user);
    fusedOps.append(recursed.second.begin(), recursed.second.end());
  }
  return {latestScalar, fusedOps};
}

static std::pair<Operation *, bool>
analyzeFusion(Value tensorResult, Value reduceInput,
              SmallPtrSetImpl<Value> &mappedTensors) {
  Operation *lastFusedOp = nullptr;
  bool isTwoPhase = false;

  mappedTensors.insert(tensorResult);

  for (Operation *user : tensorResult.getUsers()) {
    auto genericOp = dyn_cast<linalg::GenericOp>(user);
    if (!genericOp)
      continue;

    bool allOperandsAvailable = true;
    bool usesReduceInput = false;
    for (Value operand : genericOp.getInputs()) {
      if (mappedTensors.contains(operand))
        continue;
      if (operand == reduceInput) {
        usesReduceInput = true;
        continue;
      }
      allOperandsAvailable = false;
      break;
    }
    if (!allOperandsAvailable)
      continue;

    lastFusedOp = user;
    if (usesReduceInput)
      isTwoPhase = true;

    // Simulate mapping output
    Value newResult = genericOp.getResults()[0];
    auto recursed = analyzeFusion(newResult, reduceInput, mappedTensors);
    if (recursed.first) {
      lastFusedOp = recursed.first;
      isTwoPhase |= recursed.second;
    }
  }
  return {lastFusedOp, isTwoPhase};
}
struct FullReduceLowering : public OpRewritePattern<nova::ReduceOp> {
  using OpRewritePattern<nova::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto outputType = cast<RankedTensorType>(op.getOutput().getType());

    // 1. Check if this is a "Full" reduction
    bool isFull = (outputType.getRank() == 0);
    if (!isFull) {
      isFull = true;
      for (auto dim : outputType.getShape())
        if (dim != 1)
          isFull = false;
    }
    if (!isFull)
      return failure();

    Location loc = op.getLoc();
    ReductionKind kind = op.getKind();

    gpu::AllReduceOperation gpuOp;
    float initValFloat = 0.0f;
    switch (kind) {
    case ReductionKind::SUM:
      gpuOp = gpu::AllReduceOperation::ADD;
      initValFloat = 0.0f;
      break;
    case ReductionKind::PRODUCT:
      gpuOp = gpu::AllReduceOperation::MUL;
      initValFloat = 1.0f;
      break;
    case ReductionKind::MAX:
      gpuOp = gpu::AllReduceOperation::MAXIMUMF;
      initValFloat = -std::numeric_limits<float>::infinity();
      break;
    case ReductionKind::MIN:
      gpuOp = gpu::AllReduceOperation::MINIMUMF;
      initValFloat = std::numeric_limits<float>::infinity();
      break;
    case ReductionKind::MEAN:
      gpuOp = gpu::AllReduceOperation::ADD;
      initValFloat = 0.0f;
      break;
    default:
      return failure();
    }

    arith::AtomicRMWKind rmwKind = arith::AtomicRMWKind::addf;
    if (kind == ReductionKind::SUM || kind == ReductionKind::MEAN)
      rmwKind = arith::AtomicRMWKind::addf;
    else if (kind == ReductionKind::PRODUCT)
      rmwKind = arith::AtomicRMWKind::mulf;
    else if (kind == ReductionKind::MAX)
      rmwKind = arith::AtomicRMWKind::maximumf;
    else if (kind == ReductionKind::MIN)
      rmwKind = arith::AtomicRMWKind::minimumf;

    // 2. Prepare Buffers
    auto accMemRefType = MemRefType::get(
        outputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value alloc = rewriter
                      .create<gpu::AllocOp>(loc, accMemRefType, ValueRange{},
                                            ValueRange{}, ValueRange{})
                      .getMemref();

    Value initialValue =
        rewriter
            .create<arith::ConstantOp>(
                loc,
                rewriter.getFloatAttr(inputType.getElementType(), initValFloat))
            .getResult();
    rewriter.create<gpu::MemsetOp>(loc, Type(), ValueRange{}, alloc,
                                   initialValue);

    // 3. Launch Kernel
    int64_t numElements = inputType.getNumElements();
    int64_t threadsPerBlock = 128;
    int64_t numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
    Value cG =
        rewriter.create<arith::ConstantIndexOp>(loc, numBlocks).getResult();
    Value cB = rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock)
                   .getResult();

    auto launchOp = rewriter.create<gpu::LaunchOp>(loc, cG, c1, c1, cB, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value gdim = rewriter.create<gpu::GridDimOp>(loc, gpu::Dimension::x);

    Value globalId =
        rewriter
            .create<arith::AddIOp>(
                loc, tid,
                rewriter.create<arith::MulIOp>(loc, bid, bdim).getResult())
            .getResult();
    Value stride = rewriter.create<arith::MulIOp>(loc, bdim, gdim).getResult();
    Value cNumElements =
        rewriter.create<arith::ConstantIndexOp>(loc, numElements).getResult();

    auto inputMemRefType = MemRefType::get(
        inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input)
            .getResult();

    auto loop = rewriter.create<scf::ForOp>(loc, globalId, cNumElements, stride,
                                            ValueRange{initialValue});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value idx = loop.getInductionVar();
      Value currentAcc = loop.getRegionIterArgs()[0];

      SmallVector<Value> indices;
      Value rem = idx;
      auto shape = inputType.getShape();
      for (int i = shape.size() - 1; i >= 0; --i) {
        Value dimSize =
            rewriter.create<arith::ConstantIndexOp>(loc, shape[i]).getResult();
        if (i > 0) {
          indices.insert(
              indices.begin(),
              rewriter.create<arith::RemUIOp>(loc, rem, dimSize).getResult());
          rem = rewriter.create<arith::DivUIOp>(loc, rem, dimSize).getResult();
        } else {
          indices.insert(indices.begin(), rem);
        }
      }

      Value val = rewriter.create<memref::LoadOp>(loc, inputMemRef, indices)
                      .getResult();
      Value nextAcc;
      switch (kind) {
      case ReductionKind::SUM:
        nextAcc =
            rewriter.create<arith::AddFOp>(loc, currentAcc, val).getResult();
        break;
      case ReductionKind::PRODUCT:
        nextAcc =
            rewriter.create<arith::MulFOp>(loc, currentAcc, val).getResult();
        break;
      case ReductionKind::MAX:
        nextAcc = rewriter.create<arith::MaximumFOp>(loc, currentAcc, val)
                      .getResult();
        break;
      case ReductionKind::MIN:
        nextAcc = rewriter.create<arith::MinimumFOp>(loc, currentAcc, val)
                      .getResult();
        break;
      case ReductionKind::MEAN:
        nextAcc =
            rewriter.create<arith::AddFOp>(loc, currentAcc, val).getResult();
        break;
      default:
        nextAcc = currentAcc;
      }
      rewriter.create<scf::YieldOp>(loc, nextAcc);
    }

    Value finalAcc = loop.getResult(0);
    Value reduced =
        createBlockReduce(rewriter, loc, finalAcc, gpuOp, threadsPerBlock);

    // 4. Fusion and Store
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    Value isMaster =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0)
            .getResult();
    auto ifOp = rewriter.create<scf::IfOp>(loc, isMaster, false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    IRMapping mapper;
    // For FullReduce, indices is empty (or just zeroes if needed, but it's a
    // full reduce so shape is usually a scalar block) Actually, full reduce
    // over all elements. The indices for the element that resulted in 'reduced'
    // is not well-defined. Full reduce usually just produces a scalar. If there
    // is a fusion that requires the input, it's a bit ambiguous what index it
    // wants. However, if the fused op operates pointwise with the original
    // input, we'd need a two-phase loop. Let's pass empty indices for now, but
    // note that FullReduce + Binary might need a full 2-phase loop structure
    // which we haven't built into FullReduceLowering yet.
    SmallVector<Value> emptyIndices;
    auto fusionResult =
        AbsorbElementwise(op.getOutput(), reduced, input, emptyIndices, mapper,
                          launchOp, rewriter);
    Value resultToStore = fusionResult.first;
    SmallVector<Operation *> fusedOps = fusionResult.second;

    SmallVector<Value> storeIdx(outputType.getRank(), c0);
    rewriter.create<memref::AtomicRMWOp>(loc, rmwKind, resultToStore, alloc,
                                         storeIdx);

    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    if (kind == ReductionKind::MEAN) {
      Value c1_k2 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto launchOp2 = rewriter.create<gpu::LaunchOp>(loc, c1_k2, c1_k2, c1_k2,
                                                      c1_k2, c1_k2, c1_k2);
      rewriter.setInsertionPointToStart(&launchOp2.getBody().front());

      SmallVector<Value> indices(
          outputType.getRank(),
          rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult());
      Value sumVal = rewriter.create<memref::LoadOp>(loc, alloc, indices);

      Type elemType = inputType.getElementType();
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
            loc, rewriter.getIntegerAttr(elemType, numElements));
      }

      Value meanVal;
      if (isa<FloatType>(computeType)) {
        meanVal = rewriter.create<arith::DivFOp>(loc, sumVal, divisor);
      } else {
        meanVal = rewriter.create<arith::DivSIOp>(loc, sumVal, divisor);
      }

      if (meanVal.getType() != elemType) {
        meanVal = rewriter.create<arith::TruncFOp>(loc, elemType, meanVal);
      }

      rewriter.create<memref::StoreOp>(loc, meanVal, alloc, indices);
      rewriter.create<gpu::TerminatorOp>(loc);

      rewriter.setInsertionPointAfter(launchOp2);
    }

    auto toTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, outputType, alloc);
    toTensor.setRestrict(true);

    if (!fusedOps.empty()) {
      rewriter.replaceOp(fusedOps.back(), toTensor.getResult());
      for (auto it = fusedOps.rbegin(); it != fusedOps.rend(); ++it) {
        if (*it != fusedOps.back()) {
          rewriter.eraseOp(*it);
        }
      }
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, toTensor.getResult());
    }
    return success();
  }
};

struct PartialReduceLowering : public OpRewritePattern<nova::ReduceOp> {
  using OpRewritePattern<nova::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto outputType = cast<RankedTensorType>(op.getType());

    if (outputType.getRank() == 0)
      return failure(); // Full reduction handled elsewhere

    Location loc = op.getLoc();

    // 1. Identify Reduced and Parallel Dimensions
    SmallVector<int64_t> reducedDims;
    if (auto dimAttr = op.getDimension()) {
      for (auto attr : dimAttr.value()) {
        reducedDims.push_back(cast<IntegerAttr>(attr).getInt());
      }
    } else {
      // Fallback or assume row-wise if missing?
      // For now, if no dimension, we expect same rank implementation or fail.
      // Let's rely on dimension being present for rank-reducing.
      return failure();
    }

    // Sort reduced dims for easier processing
    std::sort(reducedDims.begin(), reducedDims.end());

    int64_t inputRank = inputType.getRank();
    SmallVector<int64_t> parallelDims;
    for (int64_t i = 0; i < inputRank; ++i) {
      bool isReduced = false;
      for (auto d : reducedDims)
        if (d == i)
          isReduced = true;
      if (!isReduced)
        parallelDims.push_back(i);
    }

    // Compute sizes
    int64_t numParallel = 1;
    for (auto d : parallelDims)
      numParallel *= inputType.getDimSize(d);

    int64_t numReduction = 1;
    for (auto d : reducedDims)
      numReduction *= inputType.getDimSize(d);

    ReductionKind kind = op.getKind();
    gpu::AllReduceOperation gpuOp;
    float initValFloat = 0.0f;
    arith::AtomicRMWKind rmwKind = arith::AtomicRMWKind::addf;

    switch (kind) {
    case ReductionKind::SUM:
      gpuOp = gpu::AllReduceOperation::ADD;
      initValFloat = 0.0f;
      rmwKind = arith::AtomicRMWKind::addf;
      break;
    case ReductionKind::MAX:
      gpuOp = gpu::AllReduceOperation::MAXIMUMF;
      initValFloat = -std::numeric_limits<float>::infinity();
      rmwKind = arith::AtomicRMWKind::maximumf;
      break;
    case ReductionKind::MIN:
      gpuOp = gpu::AllReduceOperation::MINIMUMF;
      initValFloat = std::numeric_limits<float>::infinity();
      rmwKind = arith::AtomicRMWKind::minimumf;
      break;
    case ReductionKind::MEAN:
      gpuOp = gpu::AllReduceOperation::ADD;
      initValFloat = 0.0f;
      rmwKind = arith::AtomicRMWKind::addf;
      break;
    default:
      return failure();
    }

    // 2. Prepare Buffers
    SmallPtrSet<Value, 4> mappedTensors;
    auto analysis = analyzeFusion(op.getOutput(), input, mappedTensors);
    Operation *finalFusedOp = analysis.first;
    bool isTwoPhase = analysis.second;

    RankedTensorType fusionOutputType = outputType;
    if (finalFusedOp) {
      fusionOutputType =
          cast<RankedTensorType>(finalFusedOp->getResults()[0].getType());
    }

    auto accMemRefType = MemRefType::get(
        fusionOutputType.getShape(), fusionOutputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value alloc = rewriter
                      .create<gpu::AllocOp>(loc, accMemRefType, ValueRange{},
                                            ValueRange{}, ValueRange{})
                      .getMemref();

    Value initialValue =
        rewriter
            .create<arith::ConstantOp>(
                loc,
                rewriter.getFloatAttr(inputType.getElementType(), initValFloat))
            .getResult();
    rewriter.create<gpu::MemsetOp>(loc, Type(), ValueRange{}, alloc,
                                   initialValue);

    // 3. Launch Kernel: One Block per Output Element (Parallel Element)
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
    Value cG =
        rewriter.create<arith::ConstantIndexOp>(loc, numParallel).getResult();
    int64_t threadsPerBlock = 128;
    Value cB = rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock)
                   .getResult();

    auto launchOp = rewriter.create<gpu::LaunchOp>(loc, cG, c1, c1, cB, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);

    Value cNumReduction =
        rewriter.create<arith::ConstantIndexOp>(loc, numReduction).getResult();

    auto inputMemRefType = MemRefType::get(
        inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input)
            .getResult();

    // Loop over reduction dimensions
    auto loop = rewriter.create<scf::ForOp>(loc, tid, cNumReduction, bdim,
                                            ValueRange{initialValue});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value reduceLinearIdx = loop.getInductionVar();
      Value currentAcc = loop.getRegionIterArgs()[0];

      // Reconstruct Input Indices
      // We have `bid` as linear index for parallel dims
      // We have `reduceLinearIdx` as linear index for reduced dims

      SmallVector<Value> inputIndices(inputRank);

      // Delinearize Parallel (Output) Index
      Value remParallel = bid;
      for (int i = parallelDims.size() - 1; i >= 0; --i) {
        int64_t dimIdx = parallelDims[i];
        int64_t dimSize = inputType.getDimSize(dimIdx);
        Value dimSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();

        if (i > 0) {
          Value coord =
              rewriter.create<arith::RemUIOp>(loc, remParallel, dimSizeVal)
                  .getResult();
          inputIndices[dimIdx] = coord;
          remParallel =
              rewriter.create<arith::DivUIOp>(loc, remParallel, dimSizeVal)
                  .getResult();
        } else {
          inputIndices[dimIdx] = remParallel;
        }
      }

      // Delinearize Reduction Index
      Value remReduce = reduceLinearIdx;
      for (int i = reducedDims.size() - 1; i >= 0; --i) {
        int64_t dimIdx = reducedDims[i];
        int64_t dimSize = inputType.getDimSize(dimIdx);
        Value dimSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();

        if (i > 0) {
          Value coord =
              rewriter.create<arith::RemUIOp>(loc, remReduce, dimSizeVal)
                  .getResult();
          inputIndices[dimIdx] = coord;
          remReduce =
              rewriter.create<arith::DivUIOp>(loc, remReduce, dimSizeVal)
                  .getResult();
        } else {
          inputIndices[dimIdx] = remReduce;
        }
      }

      Value val =
          rewriter.create<memref::LoadOp>(loc, inputMemRef, inputIndices)
              .getResult();
      Value nextAcc;
      if (kind == ReductionKind::MAX) {
        nextAcc = rewriter.create<arith::MaximumFOp>(loc, currentAcc, val)
                      .getResult();
      } else if (kind == ReductionKind::MIN) {
        nextAcc = rewriter.create<arith::MinimumFOp>(loc, currentAcc, val)
                      .getResult();
      } else if (kind == ReductionKind::PRODUCT) {
        nextAcc =
            rewriter.create<arith::MulFOp>(loc, currentAcc, val).getResult();
      } else { // SUM or MEAN
        nextAcc =
            rewriter.create<arith::AddFOp>(loc, currentAcc, val).getResult();
      }
      rewriter.create<scf::YieldOp>(loc, nextAcc);
    }

    Value finalAcc = loop.getResult(0);
    Value reduced =
        createBlockReduce(rewriter, loc, finalAcc, gpuOp, threadsPerBlock);

    if (kind == ReductionKind::MEAN) {
      Type elementType = inputType.getElementType();
      Type computeType = elementType;
      Value sumVal = reduced;

      if (elementType.isF16() || elementType.isBF16()) {
        computeType = rewriter.getF32Type();
        sumVal = rewriter.create<arith::ExtFOp>(loc, computeType, sumVal);
      }

      Value divisor;
      if (computeType.isF32() || computeType.isF64()) {
        divisor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(computeType, numReduction));
      } else {
        divisor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(elementType, numReduction));
      }

      Value meanVal;
      if (isa<FloatType>(computeType)) {
        meanVal = rewriter.create<arith::DivFOp>(loc, sumVal, divisor);
      } else {
        meanVal = rewriter.create<arith::DivSIOp>(loc, sumVal, divisor);
      }

      if (meanVal.getType() != elementType) {
        meanVal = rewriter.create<arith::TruncFOp>(loc, elementType, meanVal);
      }
      reduced = meanVal;
    }

    // 4. Fusion and Store
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    SmallVector<Operation *> allFusedOps;
    if (!isTwoPhase) {
      Value isMaster =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0)
              .getResult();
      auto ifOp = rewriter.create<scf::IfOp>(loc, isMaster, false);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());

      IRMapping mapper;
      SmallVector<Value> emptyIndices;
      auto fusionResult =
          AbsorbElementwise(op.getOutput(), reduced, input, emptyIndices,
                            mapper, launchOp, rewriter);
      Value resultToStore = fusionResult.first;
      allFusedOps = fusionResult.second;
      // We only use fused ops for erasing later. We can rely on finalFusedOp
      // for type.

      // Store index: Delinearize bid again to match output shape
      SmallVector<Value> storeIdx;
      Value remStore = bid;
      auto outputShape = outputType.getShape();
      storeIdx.resize(outputShape.size());
      for (int i = outputShape.size() - 1; i >= 0; --i) {
        int64_t dimSize = outputShape[i];
        Value dimSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();
        if (i > 0) {
          storeIdx[i] =
              rewriter.create<arith::RemUIOp>(loc, remStore, dimSizeVal)
                  .getResult();
          remStore = rewriter.create<arith::DivUIOp>(loc, remStore, dimSizeVal)
                         .getResult();
        } else {
          storeIdx[i] = remStore;
        }
      }

      rewriter.create<memref::AtomicRMWOp>(loc, rmwKind, resultToStore, alloc,
                                           storeIdx);

      rewriter.setInsertionPointAfter(ifOp);
    } else {
      // Two-Phase Fusion
      rewriter.create<gpu::BarrierOp>(loc);

      // We run a second loop where ALL threads participate to perform
      // elementwise ops.
      auto loop2 = rewriter.create<scf::ForOp>(loc, tid, cNumReduction, bdim,
                                               ValueRange{});
      {
        OpBuilder::InsertionGuard guard2(rewriter);
        rewriter.setInsertionPointToStart(loop2.getBody());
        Value reduceLinearIdx2 = loop2.getInductionVar();

        SmallVector<Value> inputIndices2(inputRank);

        // Delinearize Parallel (Output) Index
        Value remParallel2 = bid;
        for (int i = parallelDims.size() - 1; i >= 0; --i) {
          int64_t dimIdx = parallelDims[i];
          int64_t dimSize = inputType.getDimSize(dimIdx);
          Value dimSizeVal =
              rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();

          if (i > 0) {
            Value coord =
                rewriter.create<arith::RemUIOp>(loc, remParallel2, dimSizeVal)
                    .getResult();
            inputIndices2[dimIdx] = coord;
            remParallel2 =
                rewriter.create<arith::DivUIOp>(loc, remParallel2, dimSizeVal)
                    .getResult();
          } else {
            inputIndices2[dimIdx] = remParallel2;
          }
        }

        // Delinearize Reduction Index
        Value remReduce2 = reduceLinearIdx2;
        for (int i = reducedDims.size() - 1; i >= 0; --i) {
          int64_t dimIdx = reducedDims[i];
          int64_t dimSize = inputType.getDimSize(dimIdx);
          Value dimSizeVal =
              rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();

          if (i > 0) {
            Value coord =
                rewriter.create<arith::RemUIOp>(loc, remReduce2, dimSizeVal)
                    .getResult();
            inputIndices2[dimIdx] = coord;
            remReduce2 =
                rewriter.create<arith::DivUIOp>(loc, remReduce2, dimSizeVal)
                    .getResult();
          } else {
            inputIndices2[dimIdx] = remReduce2;
          }
        }

        IRMapping mapper2;
        auto fusionResult2 =
            AbsorbElementwise(op.getOutput(), reduced, input, inputIndices2,
                              mapper2, launchOp, rewriter);
        Value resultToStore2 = fusionResult2.first;
        allFusedOps = fusionResult2.second;

        // In a two-phase fusion, the output shape perfectly matches the input
        // shape, so we can directly store the result at the inputIndices!
        rewriter.create<memref::StoreOp>(loc, resultToStore2, alloc,
                                         inputIndices2);
      }
    }

    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    auto toTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, fusionOutputType, alloc);
    toTensor.setRestrict(true);

    if (finalFusedOp) {
      rewriter.replaceOp(finalFusedOp, toTensor.getResult());

      for (auto it = allFusedOps.rbegin(); it != allFusedOps.rend(); ++it) {
        if (*it != finalFusedOp) {
          rewriter.eraseOp(*it);
        }
      }

      // The original reduce op is now dead.
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, toTensor.getResult());
    }
    return success();
  }
};

struct NovaFusionKernelEmitterPass
    : public PassWrapper<NovaFusionKernelEmitterPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaFusionKernelEmitterPass)
  StringRef getArgument() const override {
    return "nova-fusion-kernel-emitter";
  }
  StringRef getDescription() const override {
    return "Fuse nova operations into gpu kernels";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FullReduceLowering>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<nova::NovaDialect>();
   // target.addIllegalOp<nova::ReduceOp>();
    target.addLegalDialect<gpu::GPUDialect, arith::ArithDialect,
                           scf::SCFDialect, memref::MemRefDialect,
                           math::MathDialect, linalg::LinalgDialect,
                           func::FuncDialect, tensor::TensorDialect,
                           bufferization::BufferizationDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace nova {
std::unique_ptr<Pass> createNovaFusionKernelEmitterPass() {
  return std::make_unique<NovaFusionKernelEmitterPass>();
}
void registerNovaFusionKernelEmitterPass() {
  PassRegistration<NovaFusionKernelEmitterPass>();
}
} // namespace nova
} // namespace mlir
