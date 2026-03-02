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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
                                /*uniform=*/false)
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

    // 3. Launch Kernel — single block for deterministic reduction
    int64_t numElements = inputType.getNumElements();
    int64_t threadsPerBlock = 32; // 1 warp — ensures AllReduceOp uses only
                                  // warp shuffles (no shared memory races)

    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
    Value cB = rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock)
                   .getResult();

    auto inputMemRefType = MemRefType::get(
        inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input)
            .getResult();

    auto launchOp = rewriter.create<gpu::LaunchOp>(loc, c1, c1, c1, cB, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);

    Value cNumElements =
        rewriter.create<arith::ConstantIndexOp>(loc, numElements).getResult();

    auto loop = rewriter.create<scf::ForOp>(loc, tid, cNumElements, bdim,
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
    Value resultToStore = reduced;
    SmallVector<Operation *> fusedOps;


    // For MEAN, divide by numElements before storing
    if (kind == ReductionKind::MEAN) {
      Type elemType = inputType.getElementType();
      Type computeType = elemType;
      Value sumVal = resultToStore;

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
      resultToStore = meanVal;
    }

    SmallVector<Value> storeIdx(outputType.getRank(), c0);
    rewriter.create<memref::StoreOp>(loc, resultToStore, alloc, storeIdx);

    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    auto toTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, outputType, alloc);
    toTensor.setRestrict(true);

    if (!fusedOps.empty()) {
      // Replace only the last fused op; its predecessors become dead naturally
      // and the greedy rewriter's DCE will clean them up.
      rewriter.replaceOp(fusedOps.back(), toTensor.getResult());
      // The nova.reduce result feeds fusedOps[0]; with fusedOps[0] now dead
      // (orphaned by its predecessor becoming dead), replace the reduce too.
      rewriter.replaceOp(op, toTensor.getResult());
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

    int64_t inputRank = inputType.getRank();

    // 1. Identify Reduced and Parallel Dimensions
    SmallVector<int64_t> reducedDims;
    if (auto dimAttr = op.getDimension()) {
      for (auto attr : dimAttr.value()) {
        int64_t d = cast<IntegerAttr>(attr).getInt();
        if (d < 0)
          d += inputRank;
        reducedDims.push_back(d);
      }
    } else {
      // For now, if no dimension, we expect same rank implementation or fail.
      return failure();
    }

    // Sort reduced dims for easier processing
    std::sort(reducedDims.begin(), reducedDims.end());
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
    Operation *finalFusedOp = nullptr; // analysis.first;
    bool isTwoPhase = false; // analysis.second;
    // Disable fusion: the greedy linalg.generic consumer matching in analyzeFusion
    // can accidentally absorb ops from unrelated chains (e.g. SCE backward),
    // and since we replaced the store path with a direct AtomicRMW to the
    // reduce result, we must ensure alloc and to_tensor use the reduce output shape.



    RankedTensorType fusionOutputType = outputType;
    if (finalFusedOp) {
      fusionOutputType =
          cast<RankedTensorType>(finalFusedOp->getResults()[0].getType());
    }

    // Fix: For two-phase fusion, alloc must be sized for outputType (the reduce
    // output), NOT fusionOutputType. In two-phase mode we store per-row results
    // at output coords, so the buffer shape must match outputType.
    auto allocShape = isTwoPhase ? outputType.getShape() : fusionOutputType.getShape();
    auto allocElem  = isTwoPhase ? outputType.getElementType() : fusionOutputType.getElementType();
    auto accMemRefType = MemRefType::get(
        allocShape, allocElem,
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
    int64_t threadsPerBlock = 32; // 1 warp — deterministic AllReduceOp
    Value cB = rewriter.create<arith::ConstantIndexOp>(loc, threadsPerBlock)
                   .getResult();

    auto inputMemRefType = MemRefType::get(
        inputType.getShape(), inputType.getElementType(),
        MemRefLayoutAttrInterface{}, rewriter.getI64IntegerAttr(1));
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, inputMemRefType, input)
            .getResult();

    auto launchOp = rewriter.create<gpu::LaunchOp>(loc, cG, c1, c1, cB, c1, c1);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);

    Value cNumReduction =
        rewriter.create<arith::ConstantIndexOp>(loc, numReduction).getResult();

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

    // 4. Store: AtomicRMW at the output coordinates for this block.
    // Fusion via AbsorbElementwise is disabled: it generates loads with 0
    // indices on N-D tensors and incorrectly absorbs unrelated linalg ops.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    SmallVector<Operation *> allFusedOps; // kept for op replacement below

    // Delinearize bid → output indices
    SmallVector<Value> storeIdx;
    auto outputShape = outputType.getShape();
    storeIdx.resize(outputShape.size());
    Value remStore = bid;
    for (int i = (int)outputShape.size() - 1; i >= 0; --i) {
      int64_t dimSize = outputShape[i];
      Value dimSizeVal =
          rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();
      if (i > 0) {
        storeIdx[i] =
            rewriter.create<arith::RemUIOp>(loc, remStore, dimSizeVal).getResult();
        remStore =
            rewriter.create<arith::DivUIOp>(loc, remStore, dimSizeVal).getResult();
      } else {
        storeIdx[i] = remStore;
      }
    }

    // Only thread 0 of each block writes (AllReduce already gave every thread the result)
    Value isMaster =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0).getResult();
    auto ifOp = rewriter.create<scf::IfOp>(loc, isMaster, false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    rewriter.create<memref::AtomicRMWOp>(loc, rmwKind, reduced, alloc, storeIdx);
    rewriter.setInsertionPointAfter(ifOp);


    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.setInsertionPointAfter(launchOp);

    // alloc is always sized with outputType in this path.
    RankedTensorType toTensorType = outputType;
    auto toTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, toTensorType, alloc);
    toTensor.setRestrict(true);


    if (finalFusedOp) {
      // Replace only the final fused op; predecessors become dead naturally.
      rewriter.replaceOp(finalFusedOp, toTensor.getResult());
      rewriter.replaceOp(op, toTensor.getResult());
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
    patterns.add<PartialReduceLowering>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
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