#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// ... (skip down to getDependentDialects)

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nova;

namespace mlir {
namespace nova {

class NovaToGpuPattern : public OpRewritePattern<nova::ReduceOp> {
public:
  using OpRewritePattern<nova::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    //if it is not scalar rerduction return failure
    if (cast<RankedTensorType>(op.getOutput().getType()).getShape().size() != 0) {
      //if all shapes is not 1 return fail
      auto shape = cast<RankedTensorType>(op.getOutput().getType()).getShape();
      for (int i = 0; i < shape.size(); i++) {
        if (shape[i] != 1) {
          return failure();
        }
      }
    }
    // 1. Check if input has device attribute "1"
    Value input = op.getInput();
    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) return failure();

    auto deviceAttr = llvm::dyn_cast_or_null<nova::NovaDeviceAttr>(inputType.getEncoding());
    if (!deviceAttr || deviceAttr.getValue().getValue() != "1")
      return failure();

    // 2. Map Reduction Kind to GPU Op
    gpu::AllReduceOperation gpuOp;
    ReductionKind kind = op.getKind();
    
    // Simple mapping for demo purposes
    switch (kind) {
      case ReductionKind::SUM: gpuOp = gpu::AllReduceOperation::ADD; break;
      case ReductionKind::PRODUCT: gpuOp = gpu::AllReduceOperation::MUL; break;
      // case ReductionKind::MAX: gpuOp = gpu::AllReduceOperation::MAX; break;
      // case ReductionKind::MIN: gpuOp = gpu::AllReduceOperation::MIN; break;
      default: return failure(); // Unsupported for direct GPU mapping yet
    }

    // 3. Create gpu.launch
    // We launch 1 block with enough threads to cover the flattened input, 
    // or just 32 threads and let them loop (simplest for now is 1-1 mapping for small tensors).
    
    int64_t numElements = inputType.getNumElements();
    if (numElements > 1024) return failure(); // Too big for single block simple mapping

    Location loc = op.getLoc();
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cNumThreads = rewriter.create<arith::ConstantIndexOp>(loc, numElements);
    
    // Create Alloc BEFORE launch to be visible outside
    // Create Alloc BEFORE launch to be visible outside
    // Result type is scalar f32 (0-D tensor -> 0-D memref)
    auto outputType = cast<RankedTensorType>(op.getOutput().getType());
    Value alloc = rewriter.create<memref::AllocOp>(loc, MemRefType::get(outputType.getShape(), inputType.getElementType(), MemRefLayoutAttrInterface{}, deviceAttr.getValue()));

    auto launchOp = rewriter.create<gpu::LaunchOp>(
        loc, 
        c1, c1, c1,           // Grid: 1, 1, 1
        cNumThreads, c1, c1   // Block: N, 1, 1
    );

    // 4. Generate Body
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());
    
    // Get thread ID
    Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    
    // Linearize/Extract element at TID
    // (Assuming flattened 1D access or we need to compute multi-dim indices)
    // For simplicity, let's treat it as 1D collapse if needed, or extract with Indices.
    // Generating indices from linear TID for N-D tensor:
    SmallVector<Value> indices;
    Value linearId = tid;
    
    // Stride calculation to convert linearId to multi-dim indices
    auto shape = inputType.getShape();
   // int64_t stride = 1;
    // We need strides. Just doing extract(tid) if 1D. 
    // If 2D (4x8), we need to compute row/col.
    
    if (shape.size() == 1) {
       indices.push_back(tid);
    } else {
       // Reverse compute indices
       // shape [d0, d1] -> idx0 = tid / d1, idx1 = tid % d1
       SmallVector<int64_t> strides(shape.size(), 1);
       for (int i = shape.size()-1; i > 0; --i) {
         strides[i-1] = strides[i] * shape[i];
       }
       
       for (int i = 0; i < shape.size(); ++i) {
          //Value dimStride = rewriter.create<arith::ConstantIndexOp>(loc, strides[i]);
          Value idx;
          if (i == shape.size()-1) {
            // Last dim is mod
             idx = linearId; // Simplified, assuming previous dims subtracted.
             // Standard division/mod approach:
             // val = tid
             // idx_i = val / stride_i
             // val = val % stride_i
             // But wait, constructing this logic in IR is verbose.
          }
       }
       // Fallback: Using tensor.from_elements + tensor.extract is complex inside launch.
       // Let's rely on the prompt's simplification: "Extract scalar value from tensor"
    }

    // SIMPLIFICATION for correct IR generation without complex math: 
    // Use `tensor.extract` with pre-computed indices if constant? No, dynamic TID.
    // Let's assume input is 1D for now or we create a collapse_shape before launch?
    // We can't insert ops before launch easily here without moving logic out.
    
    // Let's implement the "extract" via a helper (assuming 1D or handling conversion).
    // Or just accept we only handle 1D tensors effectively for this demo.
    // If not 1D, fail? 
    // User example was 4x8 -> 32 elements.
    
    // Better strategy:
    // 1. Collapse shape to 1D before launch.
    // 2. Pass 1D tensor to launch (implicitly captured).
    // 3. Extract at `tid`.
    // 4. AllReduce.
    
    rewriter.setInsertionPoint(launchOp); // Step back out
    
    // Collapse to 1D
  //  Type flatType = RankedTensorType::get({numElements}, inputType.getElementType(), inputType.getEncoding());
    
    // We can't use ReassociativeIndices easily in C++ without helper construction.
    // Just assuming 1D for the MVP if complex.
    // BUT the user input is 4x8.
    
    // Ok, let's do the index math inside kernel. It's safe.
    // 4x8:
    // row = tid / 8
    // col = tid % 8
    
    Value extractedVal;
    {
        rewriter.setInsertionPointToStart(&launchOp.getBody().front());
        // Re-get Op params
        loc = op.getLoc();
        tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        
        Value curr = tid;
        SmallVector<Value> accessIndices;
        
        // This logic handles fundamental row-major index reconstruction
        if (shape.size() > 1) {
             SmallVector<int64_t> reversedStrides;
            // int64_t runningStride = 1;
             for (auto dim : llvm::reverse(shape)) {
                 reversedStrides.push_back(dim);
             }
             // Actually we want div/mod.
             // idx_N = tid % dim_N
             // tid = tid / dim_N
             
             SmallVector<Value> coords(shape.size());
             for (int i = shape.size() - 1; i >= 0; --i) {
                 Value dimSize = rewriter.create<arith::ConstantIndexOp>(loc, shape[i]);
                 Value rem = rewriter.create<arith::RemUIOp>(loc, curr, dimSize);
                 Value div = rewriter.create<arith::DivUIOp>(loc, curr, dimSize);
                 coords[i] = rem;
                 curr = div;
             }
             accessIndices = coords;
        } else {
            accessIndices.push_back(tid);
        }
        
        extractedVal = rewriter.create<tensor::ExtractOp>(loc, input, accessIndices);

        // GPU All Reduce
        mlir::gpu::AllReduceOperationAttr opAttr = mlir::gpu::AllReduceOperationAttr::get(op.getContext(), gpuOp);
        // Correct Builder signature: ResultType, Value, OpAttr, Uniform
        Value reduced = rewriter.create<gpu::AllReduceOp>(loc, extractedVal.getType(), extractedVal, opAttr, /*uniform=*/true);
        
        // Store Result (Thread 0 only)
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value isMaster = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
        
        scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, isMaster, /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());

        SmallVector<Value> indices;
        for (int i = 0; i < outputType.getRank(); ++i) {
             indices.push_back(c0);
        }
        rewriter.create<memref::StoreOp>(loc, reduced, alloc, indices);
        
        rewriter.setInsertionPointAfter(ifOp);
        rewriter.create<gpu::TerminatorOp>(loc);
        rewriter.setInsertionPointAfter(launchOp);
        
        // Load back to tensor
      //  Value memrefVal = alloc;
        // 5. Wrap the result memref into a tensor
        // We cannot load from GPU memref on host (usually). 
        // We use bufferization.to_tensor to indicate this buffer IS the tensor result.
        
        // Ensure inputs match: result type (tensor) and alloc (memref)
        bufferization::ToTensorOp toTensor = rewriter.create<bufferization::ToTensorOp>(loc, op.getType(), alloc);
        toTensor.setRestrict(true);
        // toTensor.setWritable(true); // Optional but good for optimization if written later
        
        rewriter.replaceOp(op, toTensor.getResult());
    }

    return success();
  }
};

struct NovaToGpuPass : public PassWrapper<NovaToGpuPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToGpuPass)

  StringRef getArgument() const override { return "nova-to-gpu"; }
  StringRef getDescription() const override { return "Lower nova to gpu dialect"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect, memref::MemRefDialect, tensor::TensorDialect, bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<NovaToGpuPattern>(context);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
       // signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createNovaToGpuPass() {
  return std::make_unique<NovaToGpuPass>();
}

} // namespace nova
} // namespace mlir
