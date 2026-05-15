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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    RewritePatternSet &patterns,
    const llvm::DenseMap<Block *, Value> &threadIdMap, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize, int64_t maxBitsPerShuffle = 32);

namespace {

//===----------------------------------------------------------------------===//
// NovaContractionVectorLayoutOptions
// Holds per-pass state (threadId, sizes) and builds the full pattern set.
//===----------------------------------------------------------------------===//

class NovaContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  NovaContractionVectorLayoutOptions(
      Operation *root, const llvm::DenseMap<Block *, Value> &threadIdMap,
      int64_t subgroupSize, ArrayRef<int64_t> workgroupSize,
      int64_t maxBitsPerShuffle = 32, bool fullConversion = true)
      : VectorLayoutOptions(root, fullConversion), threadIdMap(threadIdMap),
        subgroupSize(subgroupSize),
        workgroupSize(workgroupSize.begin(), workgroupSize.end()),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  VectorLayoutInterface getDefaultLayout(VectorType type) const override {
    return {};
  }

  RewritePatternSet getPatterns(MLIRContext *ctx) const {
    RewritePatternSet patterns(ctx);
    populateNovaGPUDistributionPatterns(patterns);
    populateNovaGPUDistributeNestedLayoutAttrPatterns(
        patterns, threadIdMap, subgroupSize, workgroupSize, maxBitsPerShuffle);
    return patterns;
  }

private:
  const llvm::DenseMap<Block *, Value> &threadIdMap;
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
    // Step 1: Determine workgroup_size (total threads per workgroup).
    //
    // Preferred source: "workgroup_size" DenseI64ArrayAttr on the function,
    // set by KernelConfig as [totalThreads, 1, 1] or [x, y, z].
    //
    // Fallback (Nova pipeline): KernelConfig does not write workgroup_size.
    // Derive totalThreads from NestedLayoutAttr anchors on to_layout ops:
    //   totalThreads = subgroupTile_product × subgroupSize × threadTile_product
    // This matches the number of threads the layout assigns to one workgroup tile.
    // The derived value is placed in workgroupSize[0] so the linearization and
    // numThreads computation in DistributeTransferWrite work correctly.
    // -----------------------------------------------------------------------
    int64_t subgroupSize = 32;
    if (auto attr = funcOp->getAttrOfType<IntegerAttr>("subgroup_size"))
      subgroupSize = attr.getInt();

    SmallVector<int64_t, 3> workgroupSize = {1, 1, 1};
    if (auto attr = funcOp->getAttrOfType<DenseI64ArrayAttr>("workgroup_size")) {
      workgroupSize = llvm::to_vector(attr.asArrayRef());
    } else {
      // Scan to_layout ops and take the max totalThreads across all layouts
      // (different layouts encode different sub-tiles; the warp-mapped to_layout
      // gives the true per-workgroup thread count).
      int64_t maxThreads = 1;
      funcOp.walk([&](vec_ext::ToLayoutOp toLayout) {
        auto layout = dyn_cast<vec_ext::NestedLayoutAttr>(toLayout.getLayout());
        if (!layout)
          return;
        int64_t sgProd = 1;
        for (int64_t s : layout.getSubgroupTile())
          sgProd *= (s > 0 ? s : 1);
        int64_t tProd = 1;
        for (int64_t t : layout.getThreadTile())
          tProd *= (t > 0 ? t : 1);
        // Total threads per workgroup = numWarps * warpSize.
        // threadTile is a spatial mapping, not a thread multiplier.
        (void)tProd;
        int64_t total = sgProd * subgroupSize;
        maxThreads = std::max(maxThreads, total);
      });
      workgroupSize[0] = maxThreads;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[VectorDistribute] workgroup_size=[" << workgroupSize[0]
               << "," << workgroupSize[1] << "," << workgroupSize[2]
               << "] subgroup_size=" << subgroupSize << "\n");

    // -----------------------------------------------------------------------
    // Step 5.0: Skip the pass entirely if there are no MMA to_layout anchors.
    // Non-MMA ops (fills, copies, elementwise) are distributed by thread-tiling
    // forall mapping. Vector distribution for non-MMA paths is not yet built.
    // TODO: enable non-MMA distribution when the patterns are complete.
    // -----------------------------------------------------------------------
    bool hasMMALayout = false;
    funcOp.walk([&](vec_ext::ToLayoutOp toLayout) {
      if (toLayout.getMmaKind())
        hasMMALayout = true;
    });
    if (!hasMMALayout)
      return;

    // -----------------------------------------------------------------------
    // Step 5.5: Propagate nova.gpu.mma from result to_layout → vector.contract.
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
    // Step 6: Build a per-warp-forall thread ID map, then distribute.
    //
    // gpu.thread_id must live inside each warp forall body so that
    // MapForallToGPU (which runs later) splices it into the gpu.launch body.
    // If thread_id were created at function scope, GpuKernelOutliningPass
    // would capture it as a kernel argument — always 0 on the host side.
    //
    // We cannot call distributeVectorOps(warpForall, ...) because
    // applyPatternsGreedily requires an IsolatedFromAbove root (only funcOp
    // qualifies). Instead:
    //   1. Walk every warp forall; insert gpu.thread_id + linearize once at
    //      the start of each body and record body→linearThreadId in a map.
    //   2. Pass the map to the patterns; each pattern walks up from its op to
    //      find the nearest enclosing warp forall body and looks up the right
    //      thread ID.
    //   3. Call distributeVectorOps(funcOp, ...) as normal.
    // -----------------------------------------------------------------------
    llvm::DenseMap<Block *, Value> threadIdMap;

    // Helper: build a linearized gpu.thread_id at the top of a forall body
    // and record it in threadIdMap.
    auto registerThreadId = [&](scf::ForallOp forallOp) {
      if (threadIdMap.count(forallOp.getBody()))
        return;
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(forallOp.getBody());
      Location loc = forallOp.getLoc();

      Value tidX = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                            gpu::Dimension::x);
      Value tidY = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                            gpu::Dimension::y);
      Value tidZ = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                            gpu::Dimension::z);
      SmallVector<Value, 3> tidVec    = {tidZ, tidY, tidX};
      SmallVector<int64_t, 3> wgSizes = {workgroupSize[2], workgroupSize[1],
                                          workgroupSize[0]};
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
      threadIdMap[forallOp.getBody()] = linearThreadId;
    };

    // Register thread IDs for warp-mapped foralls (primary distribution scope).
    funcOp.walk([&](scf::ForallOp forallOp) {
      auto mapping = forallOp.getMappingAttr();
      if (!mapping || mapping.empty() ||
          !isa<gpu::GPUWarpMappingAttr>(mapping.getValue().front()))
        return;
      registerThreadId(forallOp);
    });

    // Also register thread IDs for block-mapped foralls. Vector ops hoisted
    // above the warp forall (e.g. bias transfer_reads whose result is an
    // iter_arg) live in these blocks and need a thread ID to be distributed.
    funcOp.walk([&](scf::ForallOp forallOp) {
      auto mapping = forallOp.getMappingAttr();
      if (!mapping || mapping.empty() ||
          !isa<gpu::GPUBlockMappingAttr>(mapping.getValue().front()))
        return;
      registerThreadId(forallOp);
    });

    NovaContractionVectorLayoutOptions options(funcOp, threadIdMap,
                                               subgroupSize, workgroupSize,
                                               /*maxBitsPerShuffle=*/32,
                                               /*fullConversion=*/true);
    RewritePatternSet patterns = options.getPatterns(ctx);
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
