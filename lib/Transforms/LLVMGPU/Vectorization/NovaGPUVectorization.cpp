//===- NovaGPUVectorization.cpp - Nova GPU Vectorization Passes -----------===//
//
// Defines vectorization passes for the Nova GPU pipeline.
//
// Passes defined here:
//   NovaGPUVectorizeMemrefCopyPass — vectorize memref.copy (global→shared)
//   NovaGPUUnrollToIntrinsicsPass  — unroll vector.contract to MMA-native
//                                    shape read from the op's lowering_config
//   NovaGPULowerPackOpsPass        — lower linalg.pack/unpack before bufferize
//
// NovaGPUGenericVectorizationPass lives in NovaGPUGenericVectorization.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APInt.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"

using namespace mlir;

namespace mlir::nova {

namespace {




//===----------------------------------------------------------------------===//
// NovaGPUVectorizeMemrefCopyPass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorizeMemrefCopyPass
    : public PassWrapper<NovaGPUVectorizeMemrefCopyPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorizeMemrefCopyPass)
  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    IRRewriter rewriter(&getContext());
    MLIRContext *context = &getContext();
    funcOp.walk([&](memref::CopyOp copyOp) {
      // Skip copies where source or destination is a function argument.
      // These are typically D2H output copies (e.g. memref.copy device_buf,
      // func_arg). They must stay as memref.copy until createConvertMemRefToGpuPass
      // (Step 12) converts them to gpu.memcpy → cudaMemcpyAsync. If vectorized
      // here, the host-side vector.transfer_read later reads from a device
      // pointer (after ConvertMemRefToGpu upgrades the alloc) = illegal access.
      // Also skip copies to view-like ops (expand_shape etc.) of func args.
      auto isFuncArg = [](Value v) -> bool {
        if (isa<BlockArgument>(v))
          return true;
        // Follow view-like ops (expand_shape, subview, cast) back to the root.
        Operation *defOp = v.getDefiningOp();
        while (defOp && isa<memref::ExpandShapeOp, memref::SubViewOp,
                            memref::CastOp, memref::ReinterpretCastOp,
                            memref::CollapseShapeOp>(defOp)) {
          v = defOp->getOperand(0);
          if (isa<BlockArgument>(v))
            return true;
          defOp = v.getDefiningOp();
        }
        return false;
      };
      if (isFuncArg(copyOp.getSource()) || isFuncArg(copyOp.getTarget()))
        return;
      (void)linalg::vectorizeCopy(rewriter, copyOp);
    });

    // Lower reductions via InnerReduction + ReductionToContract (same as
    // GenericVectorizationPass step 2) so any remaining multi_reductions
    // from copy vectorization also follow the outer-product → llvm.fma path.
    // NOTE: Do NOT lower vector.contract here — must survive to gpuPm.
    {
      RewritePatternSet patterns(context);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorReductionToContractPatterns(patterns);
      vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
      vector::populateVectorShapeCastLoweringPatterns(patterns);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
  StringRef getArgument() const override { return "nova-gpu-vectorize-memref-copy"; }
  StringRef getDescription() const override {
    return "Vectorize memref.copy ops, especially between global and shared memory.";
  }
};


//===----------------------------------------------------------------------===//
// NovaGPUUnrollToIntrinsicsPass
//
// NOTE: This pass is now a no-op stub. M/N-batch unrolling to the native MMA
// intrinsic shape has been absorbed into NovaGPUVectorDistributePass (§4.5),
// which performs the unroll inline per K-step immediately after distributing
// per-thread slices. Running a separate pass-over of already-distributed IR
// is unnecessary and was the root cause of the massive static unroll explosion
// (512+ contracts, 24KB register spill) seen in linear.log.
//
// The registration is kept so existing pipeline strings using
// --nova-gpu-unroll-to-intrinsics don't break, but the pass is a no-op.
//===----------------------------------------------------------------------===//

struct NovaGPUUnrollToIntrinsicsPass
    : public PassWrapper<NovaGPUUnrollToIntrinsicsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUUnrollToIntrinsicsPass)

  void runOnOperation() override {
    // No-op: unrolling is now handled inside NovaGPUVectorDistributePass.
  }

  StringRef getArgument() const override {
    return "nova-gpu-unroll-to-intrinsics";
  }
  StringRef getDescription() const override {
    return "(no-op) M/N-batch unroll is now inline in nova-gpu-vector-distribute.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory Functions and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUVectorizeMemrefCopyPass() {
  return std::make_unique<NovaGPUVectorizeMemrefCopyPass>();
}
void registerNovaGPUVectorizeMemrefCopyPass() {
  PassRegistration<NovaGPUVectorizeMemrefCopyPass>();
}

std::unique_ptr<Pass> createNovaGPUUnrollToIntrinsicsPass() {
  return std::make_unique<NovaGPUUnrollToIntrinsicsPass>();
}
void registerNovaGPUUnrollToIntrinsicsPass() {
  PassRegistration<NovaGPUUnrollToIntrinsicsPass>();
}

//===----------------------------------------------------------------------===//
// NovaGPULowerPackOpsPass
//
// Lowers linalg.pack / linalg.unpack ops to standard tensor dialect ops
// (tensor.pad, tensor.expand_shape, linalg.transpose, tensor.extract_slice)
// that implement BufferizableOpInterface.
//
// linalg.pack and linalg.unpack do NOT implement BufferizableOpInterface in
// MLIR 21. The NovaGPUPackToIntrinsicsPass creates linalg.pack ops on tensors
// during the MMA vectorization path. These must be lowered to bufferizable
// tensor ops BEFORE OneShotBufferize (Step 8) runs.
//
// Must run AFTER all vectorization passes (PackToIntrinsics, Generic,
// UnrollToIntrinsics, VectorDistribute) and BEFORE addNovaGPUBufferizePasses.
//===----------------------------------------------------------------------===//

struct NovaGPULowerPackOpsPass
    : public PassWrapper<NovaGPULowerPackOpsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPULowerPackOpsPass)

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    IRRewriter rewriter(funcOp.getContext());

    // Collect linalg.pack ops first (walk to avoid iterator invalidation).
    SmallVector<linalg::PackOp> packOps;
    funcOp.walk([&](linalg::PackOp op) { packOps.push_back(op); });
    for (linalg::PackOp op : packOps) {
      rewriter.setInsertionPoint(op);
      if (failed(linalg::lowerPack(rewriter, op))) {
        op.emitError("NovaGPULowerPackOpsPass: failed to lower linalg.pack");
        return signalPassFailure();
      }
    }

    // Collect linalg.unpack ops.
    SmallVector<linalg::UnPackOp> unpackOps;
    funcOp.walk([&](linalg::UnPackOp op) { unpackOps.push_back(op); });
    for (linalg::UnPackOp op : unpackOps) {
      rewriter.setInsertionPoint(op);
      if (failed(linalg::lowerUnPack(rewriter, op))) {
        op.emitError("NovaGPULowerPackOpsPass: failed to lower linalg.unpack");
        return signalPassFailure();
      }
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-lower-pack-ops";
  }
  StringRef getDescription() const override {
    return "Lower linalg.pack/linalg.unpack to bufferizable tensor ops "
           "(tensor.pad, tensor.expand_shape, linalg.transpose) before "
           "GPU-aware OneShotBufferize.";
  }
};

std::unique_ptr<Pass> createNovaGPULowerPackOpsPass() {
  return std::make_unique<NovaGPULowerPackOpsPass>();
}
void registerNovaGPULowerPackOpsPass() {
  PassRegistration<NovaGPULowerPackOpsPass>();
}

} // namespace mlir::nova
