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
// Unrolls vector.contract ops to the native MMA shape read from the
// surrounding op's lowering_config (mma_kind attribute).  Falls back to
// {16, 16, 16} (WMMA_F32) when no config is present.
//
// Must run AFTER GenericVectorization (so vector.contract ops exist) and
// AFTER PackToIntrinsics (so operand shapes are already multiples of the
// MMA shape).  Runs BEFORE VectorDistribute so that each lane sees a single
// MMA-sized fragment after distribution.
//===----------------------------------------------------------------------===//

struct NovaGPUUnrollToIntrinsicsPass
    : public PassWrapper<NovaGPUUnrollToIntrinsicsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUUnrollToIntrinsicsPass)

  void runOnOperation() override {
    auto funcOp = dyn_cast<FunctionOpInterface>(getOperation());
    if (!funcOp) return;
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    llvm::errs() << "Checking Func: " << funcOp.getName() << "\n";

    // Collect per-op MMA shapes so the filter can look them up.
    // Walk upward from each vector.contract to find the nearest linalg op
    // carrying a non-zero mma_kind in its lowering_config.
    llvm::DenseMap<Operation *, SmallVector<int64_t, 3>> opShapeMap;
    bool anyMMAKind = false;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      Operation *cur = contractOp.getOperation();
      while (cur) {
        if (auto cfg = getLoweringConfig(cur)) {
          int32_t k = getMmaKindRaw(cfg);
          if (k != 0) {
            opShapeMap[contractOp.getOperation()] = getMMAShape(k);
            anyMMAKind = true;
            return;
          }
        }
        cur = cur->getParentOp();
      }
      // mma_kind == 0 means SIMT / CUDA-core path — record with a sentinel
      // so we know the op exists but must NOT be unrolled.
      opShapeMap[contractOp.getOperation()] = {};
    });

    // When every vector.contract in this function is on the SIMT (CUDA-core)
    // path (mma_kind == 0), skip unrolling entirely.  The vector.contract ops
    // will be lowered by the standard MLIR vector-to-loops path which emits
    // FMA instructions, matching the precision of the eager (non-JIT) path.
    // Unrolling with the {16,16,16} fallback would instead produce a sequence
    // of arith.mulf + arith.addf with two roundings per multiply-add, causing
    // systematic divergence from the reference output for large (e.g. 256x256)
    // matmuls.
    if (!anyMMAKind)
      return;

    // Use a single native shape derived from the first NON-EMPTY MMA contraction found
    // (all contractions in one MMA kernel share the same intrinsic shape).
    SmallVector<int64_t, 3> nativeShape;
    for (auto &kv : opShapeMap) {
      if (!kv.second.empty()) {
        nativeShape = kv.second;
        break;
      }
    }

    if (nativeShape.empty())
      return;

    // Save the MMA lowering config BEFORE unrolling erases the original ops.
    // populateVectorUnrollPatterns replaces original contracts with smaller
    // tiled ones but does NOT copy the lowering_config attribute.  We re-attach
    // it after rewriting so VectorDistributePass detects hasMMAContract=true.
    DictionaryAttr mmaConfigToPropagate;
    for (auto &kv : opShapeMap) {
      if (!kv.second.empty()) {
        if (auto cfg = getLoweringConfig(kv.first)) {
          mmaConfigToPropagate = cfg;
          break;
        }
        // [AUDIT] If the config is on a parent op (e.g. linalg.matmul outside the unrolled loop),
        // we must find it.
        Operation *cur = kv.first;
        while (cur) {
          if (auto cfg = getLoweringConfig(cur)) {
            mmaConfigToPropagate = cfg;
            break;
          }
          cur = cur->getParentOp();
        }
        if (mmaConfigToPropagate) break;
      }
    }

    vector::UnrollVectorOptions options;
    options.setNativeShape(nativeShape);
    // Unroll all vector.contract ops in this function to the shared native shape.
    // This handles unrolling even for newly created ops during the process.
    vector::populateVectorUnrollPatterns(patterns, options);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    // Re-attach MMA config to all newly-created unrolled vector.contract ops.
    if (mmaConfigToPropagate) {
      funcOp.walk([&](vector::ContractionOp contractOp) {
        if (!getLoweringConfig(contractOp))
          setLoweringConfig(contractOp, mmaConfigToPropagate);
      });
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-unroll-to-intrinsics";
  }
  StringRef getDescription() const override {
    return "Unroll vector.contract to the MMA-native shape from lowering_config.";
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
