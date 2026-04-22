//===- NovaGPUVectorDistribute.cpp - GPU Vector Distribution Pass ----------===//
//
// Port of IREE's LLVMGPUVectorDistribute pass.
// (iree/compiler/Codegen/LLVMGPU/LLVMGPUVectorDistribute.cpp)
//
// This pass distributes vector ops from SIMD (per-workgroup) to SIMT
// (per-thread) form using the NestedLayoutAttr anchors placed by
// ConfigureTensorLayouts.
//
// Pipeline:
//   1. Read workgroup_size from function attr (set by KernelConfig).
//   2. Create gpu.thread_id z/y/x ops.
//   3. Linearize thread ID: affine.linearize_index([z,y,x], wgSize).
//   4. Read subgroup_size from function attr (default 32).
//   5. Build VectorLayoutOptions (NovaContractionVectorLayoutOptions) that
//      holds all distribution patterns.
//   6. Call distributeVectorOps(funcOp, patterns, options).
//
// Key differences from IREE:
//   - No WMMA / MFMA intrinsic distribution (only non-MMA path).
//   - workgroup_size / subgroup_size come from Nova function attrs set by
//     NovaKernelConfig.cpp instead of from IREE's HAL attributes.
//   - Uses nova::vec_ext namespace.
//
//===----------------------------------------------------------------------===//

#include "NovaGPUVectorDistribution.h"
#include "../Passes.h"

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-vector-distribute"

namespace mlir::nova {

// Forward declarations of pattern registration functions defined in
// NovaGPUDistributionPatterns.cpp and NovaGPUNestedLayoutDistributionPatterns.cpp.
void populateNovaGPUDistributionPatterns(RewritePatternSet &patterns);
void populateNovaGPUDistributeNestedLayoutAttrPatterns(
    RewritePatternSet &patterns, Value threadId, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize, int64_t maxBitsPerShuffle = 32);

namespace {

//===----------------------------------------------------------------------===//
// NovaContractionVectorLayoutOptions
// Holds per-pass state (threadId, sizes) and builds the full pattern set.
//===----------------------------------------------------------------------===//

class NovaContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  NovaContractionVectorLayoutOptions(Operation *root, Value threadId,
                                      int64_t subgroupSize,
                                      ArrayRef<int64_t> workgroupSize,
                                      int64_t maxBitsPerShuffle = 32,
                                      bool fullConversion = true)
      : VectorLayoutOptions(root, fullConversion), threadId(threadId),
        subgroupSize(subgroupSize),
        workgroupSize(workgroupSize.begin(), workgroupSize.end()),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  /// Return null layout for any rank-0 vector (scalars); skip everything else.
  /// The engine will skip ops with null layouts.
  VectorLayoutInterface getDefaultLayout(VectorType type) const override {
    return {};
  }

  /// Build the full pattern set for this pass.
  RewritePatternSet getPatterns(MLIRContext *ctx) const {
    RewritePatternSet patterns(ctx);
    populateNovaGPUDistributionPatterns(patterns);
    populateNovaGPUDistributeNestedLayoutAttrPatterns(
        patterns, threadId, subgroupSize, workgroupSize, maxBitsPerShuffle);
    return patterns;
  }

private:
  Value              threadId;
  int64_t            subgroupSize;
  SmallVector<int64_t> workgroupSize;
  int64_t            maxBitsPerShuffle;
};

//===----------------------------------------------------------------------===//
// NovaGPUVectorDistributePass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorDistributePass
    : PassWrapper<NovaGPUVectorDistributePass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorDistributePass)

  StringRef getArgument() const override {
    return "nova-gpu-vector-distribute";
  }
  StringRef getDescription() const override {
    return "Distribute SIMD vector ops to per-thread SIMT slices using "
           "NestedLayoutAttr annotations.";
  }

  // Declare NVGPU as a dependent dialect so the PassManager loads it before
  // running this pass in a multi-threaded context. Without this declaration,
  // NVIDIADistributeContract's nvgpu.mma.sync creation aborts with
  // "Loading a dialect while in a multi-threaded execution context".
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx     = &getContext();

    // -----------------------------------------------------------------------
    // Step 1: Read workgroup_size from the function attribute.
    // KernelConfig sets  workgroup_size = [totalThreads, 1, 1]  or
    //                    workgroup_size = [x, y, z]
    // We need them in [x, y, z] order as a DenseI64ArrayAttr.
    // -----------------------------------------------------------------------
    SmallVector<int64_t, 3> workgroupSize = {1, 1, 1};
    if (auto attr = funcOp->getAttrOfType<DenseI64ArrayAttr>("workgroup_size"))
      workgroupSize = llvm::to_vector(attr.asArrayRef());

    // -----------------------------------------------------------------------
    // Step 2: Create gpu.thread_id ops at the function entry block.
    // We need [x, y, z] (gpu.Dimension ordering) to linearize.
    // -----------------------------------------------------------------------
    OpBuilder builder(funcOp);
    Block &entryBlock = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);
    Location loc = funcOp.getLoc();

    Value tidX = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                          gpu::Dimension::x);
    Value tidY = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                          gpu::Dimension::y);
    Value tidZ = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                          gpu::Dimension::z);

    // -----------------------------------------------------------------------
    // Step 3: Linearize thread ID.
    //   linear = tidX + wgX * (tidY + wgY * tidZ)
    // We use affine::AffineLinearizeIndexOp (multi-index → linear).
    // Order must match: [z, y, x] with sizes [wgZ, wgY, wgX] (most-major first).
    // -----------------------------------------------------------------------
    SmallVector<Value, 3> tidVec  = {tidZ, tidY, tidX};
    SmallVector<int64_t, 3> wgSizes = {workgroupSize[2], workgroupSize[1],
                                        workgroupSize[0]};
    // Remove leading 1-sized dims to keep the index clean.
    while (tidVec.size() > 1 && wgSizes.front() == 1) {
      tidVec.erase(tidVec.begin());
      wgSizes.erase(wgSizes.begin());
    }

    Value linearThreadId;
    if (tidVec.size() == 1) {
      linearThreadId = tidVec[0];
    } else {
      linearThreadId = affine::AffineLinearizeIndexOp::create(
          builder, loc, tidVec, wgSizes, /*disjoint=*/true);
    }

    // -----------------------------------------------------------------------
    // Step 4: Read subgroup_size (default 32).
    // -----------------------------------------------------------------------
    int64_t subgroupSize = 32;
    if (auto attr = funcOp->getAttrOfType<IntegerAttr>("subgroup_size"))
      subgroupSize = attr.getInt();

    LLVM_DEBUG(llvm::dbgs()
               << "[VectorDistribute] workgroup_size=[" << workgroupSize[0]
               << "," << workgroupSize[1] << "," << workgroupSize[2]
               << "] subgroup_size=" << subgroupSize << "\n");

    // -----------------------------------------------------------------------
    // Step 5: Build options + patterns.
    // -----------------------------------------------------------------------
    // fullConversion=true: matches IREE's default. Any vec_ext::ToLayoutOp that
    // survives VectorDistribute un-consumed causes an immediate pass failure,
    // surfacing pattern gaps early rather than silently emitting undistributed IR.
    //
    // Which ops carry to_layout anchors at this point:
    //   • MMA contractions: ConfigureTensorLayouts stamps to_layout on all three
    //     operands + result with a NestedLayoutAttr encoding subgroup/thread/elt
    //     layout. DistributeContract + DistributeElementwise consume these.
    //   • Standalone reductions (flag-gated): setGPULoweringConfigLayout stamps
    //     to_layout on operand + result. DistributeMultiReduction consumes them
    //     via inter-thread butterfly (gpu.subgroup_reduce).
    //   • Elementwise/activation ops fused inside an MMA or reduction dispatch:
    //     NO to_layout anchor of their own; layout is propagated from their
    //     annotated predecessor by DistributeElementwise.
    //   • SIMT contractions (mmaKind == 0): ConfigureTensorLayouts explicitly
    //     skips them (returns success() with no annotation). SIMT dispatches
    //     never reach this pass — KernelConfig routes them to
    //     addGPUTileAndFusePassPipeline instead.
    //   • Standalone elementwise / transpose dispatches: also routed to
    //     addGPUTileAndFusePassPipeline by KernelConfig; no to_layout anchors.
    //
    // Net result: for any function that reaches this pass, every to_layout op
    // belongs to an MMA or reduction dispatch and must be consumed. fullConversion=
    // true enforces this invariant.
    NovaContractionVectorLayoutOptions options(funcOp, linearThreadId,
                                               subgroupSize, workgroupSize,
                                               /*maxBitsPerShuffle=*/32,
                                               /*fullConversion=*/true);
    RewritePatternSet patterns = options.getPatterns(ctx);

    // -----------------------------------------------------------------------
    // Step 5.5: Propagate nova.gpu.mma from result to_layout → vector.contract.
    //
    // ConfigureTensorLayouts sets mma_kind on the result to_layout op (which
    // survives vectorization). GenericVectorization destroys the linalg op so
    // nova.gpu.mma is lost from the vector.contract. Recover it here by walking
    // every to_layout whose mma_kind is set and stamping nova.gpu.mma on the
    // defining vector.contract.
    // -----------------------------------------------------------------------
    funcOp.walk([&](vec_ext::ToLayoutOp toLayout) {
      auto mmaKindAttr = toLayout.getMmaKind();
      if (!mmaKindAttr)
        return;
      Value input = toLayout.getInput();
      auto contract = input.getDefiningOp<vector::ContractionOp>();
      if (!contract)
        return;
      if (!contract->hasAttr("nova.gpu.mma"))
        contract->setAttr("nova.gpu.mma", *mmaKindAttr);
    });

    // -----------------------------------------------------------------------
    // Step 6: Run distribution.
    // -----------------------------------------------------------------------
    if (failed(distributeVectorOps(funcOp, patterns, options)))
      return signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass creation / registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUVectorDistributePass() {
  return std::make_unique<NovaGPUVectorDistributePass>();
}

void registerNovaGPUVectorDistributePass() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createNovaGPUVectorDistributePass();
  });
}

} // namespace mlir::nova
