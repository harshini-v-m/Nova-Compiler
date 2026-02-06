#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nova;

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

    // Initialize output memory with identity value before launch
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
    float initValFloat = 0.0f;
    if (kind == ReductionKind::PRODUCT)
      initValFloat = 1.0f;
    else if (kind == ReductionKind::MAX)
      initValFloat = -std::numeric_limits<float>::infinity();
    else if (kind == ReductionKind::MIN)
      initValFloat = std::numeric_limits<float>::infinity();
    else if (kind == ReductionKind::MEAN)
      initValFloat = 0.0f;

    if (elementType.isF32()) {
      initialValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(initValFloat));
    } else {
      initialValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(elementType));
    }

    // Initialize output memory using gpu.memset
    rewriter.create<gpu::MemsetOp>(loc, Type(), ValueRange{}, alloc,
                                   initialValue);

    // 3. Create gpu.launch parameters
    int64_t numElements = inputType.getNumElements();
    int64_t threadsPerBlock = 1024;
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

      Value newAcc;
      switch (kind) {
      case ReductionKind::SUM:
        newAcc = rewriter.create<arith::AddFOp>(loc, currentAcc, val);
        break;
      case ReductionKind::PRODUCT:
        newAcc = rewriter.create<arith::MulFOp>(loc, currentAcc, val);
        break;
      case ReductionKind::MAX:
        newAcc = rewriter.create<arith::MaximumFOp>(loc, currentAcc, val);
        break;
      case ReductionKind::MIN:
        newAcc = rewriter.create<arith::MinimumFOp>(loc, currentAcc, val);
        break;
      case ReductionKind::MEAN:
        newAcc = rewriter.create<arith::AddFOp>(loc, currentAcc, val);
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

    rewriter.create<memref::AtomicRMWOp>(loc, atomicKind, reduced, alloc,
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
      Value divisor;
      if (elemType.isF32() || elemType.isF64() || elemType.isF16() ||
          elemType.isBF16()) {
        divisor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elemType, numElements));
      } else {
        divisor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(elemType, numElements));
      }

      // Perform Division
      Value meanVal;
      if (isa<FloatType>(elemType)) {
        meanVal = rewriter.create<arith::DivFOp>(loc, sumVal, divisor);
      } else {
        meanVal = rewriter.create<arith::DivSIOp>(loc, sumVal, divisor);
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
