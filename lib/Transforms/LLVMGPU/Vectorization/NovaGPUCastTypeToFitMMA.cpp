#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"

using namespace mlir;
using namespace mlir::nova;

namespace {

/// Generic template to hold the logic using CRTP.
template <typename MmaOp, typename Derived>
struct MmaNormalizationPattern : public OpRewritePattern<MmaOp> {
  using OpRewritePattern<MmaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MmaOp op, PatternRewriter &rewriter) const override {
    Value memrefSource = static_cast<const Derived*>(this)->getMemref(op);
    
    // Walk through trivial casts.
    Value memref = memrefSource;
    while (auto cast = memref.getDefiningOp<memref::CastOp>()) {
      memref = cast.getSource();
    }

    auto stridedType = dyn_cast<MemRefType>(memref.getType());
    if (!stridedType) return failure();

    auto [staticStrides, staticOffset] = stridedType.getStridesAndOffset();
    
    auto castOp = memref.getDefiningOp<memref::ReinterpretCastOp>();
    auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();

    // Recursion Guard: If we already have a static offset 0 reinterpret_cast source,
    // and the type is identity, do not match again.
    if (castOp && stridedType.getLayout().isIdentity() && staticOffset == 0) {
      auto mixedOffsets = castOp.getMixedOffsets();
      if (!mixedOffsets.empty()) {
        if (auto val = getConstantIntValue(mixedOffsets[0])) {
          if (*val == 0) return failure();
        }
      }
    }

    // If it's already canonical and we have nothing to fold, we're done.
    if (!castOp && !subviewOp && stridedType.getLayout().isIdentity() && staticOffset == 0)
      return failure();

    Location loc = op.getLoc();
    Value base;
    SmallVector<OpFoldResult> offsets;
    if (castOp) {
      base = castOp.getSource();
      offsets = castOp.getMixedOffsets();
    } else if (subviewOp) {
      base = subviewOp.getSource();
      offsets = subviewOp.getMixedOffsets();
    } else {
      // If we have a non-identity layout but no recognized folder op, 
      // just cast the current memref to identity at the current indices.
      base = memref;
      for (unsigned i = 0; i < stridedType.getRank(); ++i)
        offsets.push_back(rewriter.getIndexAttr(0));
    }

    SmallVector<Value> newIndices;
    auto oldIndices = op.getIndices();
    if (oldIndices.size() != 2) return failure();

    if (offsets.size() == 2) {
      for (int i = 0; i < 2; ++i) {
        Value offsetVal = getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i]);
        Value combined = rewriter.create<arith::AddIOp>(loc, oldIndices[i], offsetVal);
        newIndices.push_back(combined);
      }
    } else if (offsets.size() == 1) {
      if (staticStrides.empty()) return failure();
      int64_t stride0 = staticStrides[0];
      if (stride0 == ShapedType::kDynamic || stride0 <= 0) return failure();

      Value offsetVal = getValueOrCreateConstantIndexOp(rewriter, loc, offsets[0]);
      Value cStride0 = rewriter.create<arith::ConstantIndexOp>(loc, stride0);
      Value offI = rewriter.create<arith::DivUIOp>(loc, offsetVal, cStride0);
      Value offJ = rewriter.create<arith::RemUIOp>(loc, offsetVal, cStride0);
      
      newIndices.push_back(rewriter.create<arith::AddIOp>(loc, oldIndices[0], offI));
      newIndices.push_back(rewriter.create<arith::AddIOp>(loc, oldIndices[1], offJ));
    } else if (offsets.size() == 0) {
        newIndices.push_back(oldIndices[0]);
        newIndices.push_back(oldIndices[1]);
    } else {
      return failure();
    }

    auto resultType = MemRefType::get(stridedType.getShape(), 
                                      stridedType.getElementType(), 
                                      MemRefLayoutAttrInterface{}, 
                                      stridedType.getMemorySpace());
    
    OpFoldResult identityOffset = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> sizes;
    for (auto s : stridedType.getShape()) sizes.push_back(rewriter.getIndexAttr(s));
    SmallVector<OpFoldResult> identityStrides;
    identityStrides.push_back(rewriter.getIndexAttr(stridedType.getShape()[1]));
    identityStrides.push_back(rewriter.getIndexAttr(1));

    auto identityCast = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, base, identityOffset, sizes, identityStrides);

    static_cast<const Derived*>(this)->replaceOp(op, identityCast.getResult(), newIndices, rewriter);
    return success();
  }
};

struct MmaLoadNormalization : public MmaNormalizationPattern<gpu::SubgroupMmaLoadMatrixOp, MmaLoadNormalization> {
  using MmaNormalizationPattern<gpu::SubgroupMmaLoadMatrixOp, MmaLoadNormalization>::MmaNormalizationPattern;
  Value getMemref(gpu::SubgroupMmaLoadMatrixOp op) const { return op.getSrcMemref(); }
  void replaceOp(gpu::SubgroupMmaLoadMatrixOp op, Value base, ValueRange newIndices, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(
        op, op.getType(), base, newIndices, op.getLeadDimensionAttr(), op.getTransposeAttr());
  }
};

struct MmaStoreNormalization : public MmaNormalizationPattern<gpu::SubgroupMmaStoreMatrixOp, MmaStoreNormalization> {
  using MmaNormalizationPattern<gpu::SubgroupMmaStoreMatrixOp, MmaStoreNormalization>::MmaNormalizationPattern;
  Value getMemref(gpu::SubgroupMmaStoreMatrixOp op) const { return op.getDstMemref(); }
  void replaceOp(gpu::SubgroupMmaStoreMatrixOp op, Value base, ValueRange newIndices, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        op, op.getSrc(), base, newIndices, op.getLeadDimensionAttr(), op.getTransposeAttr());
  }
};

struct NovaGPUCastTypeToFitMMAPass
    : public PassWrapper<NovaGPUCastTypeToFitMMAPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUCastTypeToFitMMAPass)

  void runOnOperation() override {
    // This pass normalizes strided-memref indices for WMMA load/store ops
    // (gpu::SubgroupMmaLoadMatrixOp / gpu::SubgroupMmaStoreMatrixOp).
    //
    // For the nvgpu.mma.sync path (useNvGpu=true, Ampere native, sm_80+):
    //   nvgpu.mma.sync accepts f32 operands directly and truncates to TF32 in
    //   hardware. No type casting is needed and these patterns are no-ops since
    //   gpu::SubgroupMma* ops are absent. The pass is safe to run on both paths.
    //
    // For the WMMA path (useNvGpu=false, gpu.subgroup_mma_*, sm_70–sm_75):
    //   The normalization patterns fold subview/reinterpret_cast offsets into the
    //   load/store indices, stripping the 'offset: ?' from operand types before
    //   GPUSubgroupMMAToNVVM legalization.
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MmaLoadNormalization>(ctx);
    patterns.add<MmaStoreNormalization>(ctx);
    (void)applyPatternsGreedily(op, std::move(patterns));
  }

  StringRef getArgument() const override { return "nova-gpu-cast-type-mma"; }
  StringRef getDescription() const override {
    return "Normalize MMA load/store memref indices for NVVM legalization. "
           "No-op for the nvgpu.mma.sync (Ampere native) path.";
  }
};

} // namespace

namespace mlir::nova {
std::unique_ptr<Pass> createNovaGPUCastTypeToFitMMAPass() {
  return std::make_unique<NovaGPUCastTypeToFitMMAPass>();
}
void registerNovaGPUCastTypeToFitMMAPass() {
  PassRegistration<NovaGPUCastTypeToFitMMAPass>();
}
} // namespace mlir::nova
