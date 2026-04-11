//===- NovaGPUMapForallToGPU.cpp - scf.forall → gpu.launch ---------------===//
//
// Replaces the gpu_forall_to_launch.mlir transform script with a C++ pass
// that dynamically computes thread block dimensions from nested thread-mapped
// forall bounds, instead of hardcoding block_dims = [32, 32, 1].
//
// Algorithm (per outermost block-mapped scf.forall):
//   Step 1: Compute grid dims from the block forall's upper bounds.
//   Step 2: Find nested thread-mapped foralls; compute block dims from their
//           actual iteration bounds (linear mapping → product; 3D → per-dim max).
//   Step 3: Clamp block dims to CUDA's 1024-thread limit.
//   Step 4: Create gpu.launch with correct grid/block dims.
//   Step 5: Map block forall IVs → gpu.block_id.
//   Step 6: Erase block forall terminator; splice body into launch.
//   Step 7: Convert thread foralls inside the launch body:
//           - Replace thread forall IVs → gpu.thread_id.
//           - Add predication when blockDims > forall bounds (idle threads).
//   Step 8: Erase original block forall.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-map-forall"

using namespace mlir;

namespace mlir::nova {

// ===== Helpers =============================================================

static bool hasBlockMapping(scf::ForallOp forall) {
  auto mapping = forall.getMappingAttr();
  return mapping && !mapping.empty() &&
         isa<gpu::GPUBlockMappingAttr>(mapping.getValue().front());
}

static bool hasThreadMapping(scf::ForallOp forall) {
  auto mapping = forall.getMappingAttr();
  return mapping && !mapping.empty() &&
         isa<gpu::GPUThreadMappingAttr>(mapping.getValue().front());
}

static bool hasWarpMapping(scf::ForallOp forall) {
  auto mapping = forall.getMappingAttr();
  return mapping && !mapping.empty() &&
         isa<gpu::GPUWarpMappingAttr>(mapping.getValue().front());
}

/// Maps MappingId 0→x, 1→y, 2→z for 3D (non-linear) mappings.
static gpu::Dimension mappingIdToDim(int64_t id) {
  switch (id) {
  case 0:
    return gpu::Dimension::x;
  case 1:
    return gpu::Dimension::y;
  case 2:
    return gpu::Dimension::z;
  default:
    llvm_unreachable("invalid 3D mapping ID");
  }
}

static bool isLinearThreadMapping(scf::ForallOp forall) {
  for (auto attr : forall.getMappingAttr().getValue()) {
    if (cast<gpu::GPUThreadMappingAttr>(attr).isLinearMapping())
      return true;
  }
  return false;
}

static bool isLinearBlockMapping(scf::ForallOp forall) {
  for (auto attr : forall.getMappingAttr().getValue()) {
    if (cast<gpu::GPUBlockMappingAttr>(attr).isLinearMapping())
      return true;
  }
  return false;
}

// ===== Core conversion =====================================================

/// Convert a single outermost block-mapped scf.forall to gpu.launch.
static LogicalResult convertBlockForallToLaunch(IRRewriter &rewriter,
                                                 scf::ForallOp blockForall) {
  Location loc = blockForall.getLoc();

  // ---- ALGORITHM STEP 1: Compute grid dims from block forall mapping ----
  auto blockMapping = blockForall.getMappingAttr().getValue();
  auto blockUBs = blockForall.getMixedUpperBound();
  int64_t gridDims[3] = {1, 1, 1};
  bool linearBlock = isLinearBlockMapping(blockForall);

  if (linearBlock) {
    // Linear block mapping: product of all bounds → gridDim.x.
    int64_t totalBlocks = 1;
    for (auto ub : blockUBs) {
      auto cst = getConstantIntValue(ub);
      if (!cst)
        return blockForall.emitError("non-static block forall upper bound");
      totalBlocks *= *cst;
    }
    gridDims[0] = totalBlocks;
  } else {
    // 3D block mapping: getMappingId() returns DimX=0, DimY=1, DimZ=2.
    for (auto [idx, attr] : llvm::enumerate(blockMapping)) {
      auto blockAttr = cast<gpu::GPUBlockMappingAttr>(attr);
      auto dimCst = getConstantIntValue(blockUBs[idx]);
      if (!dimCst)
        return blockForall.emitError("non-static block forall upper bound");
      int64_t mid = blockAttr.getMappingId();
      if (mid < 3)
        gridDims[mid] = *dimCst;
    }
  }

  // Validate grid dims: all must be > 0.
  for (int i = 0; i < 3; ++i) {
    if (gridDims[i] <= 0) {
      LLVM_DEBUG(llvm::dbgs() << "[nova-gpu-map-forall] grid dim " << i
                               << " is " << gridDims[i]
                               << ", clamping to 1\n");
      gridDims[i] = 1;
    }
  }

  // ---- ALGORITHM STEP 2: Find warp/thread foralls and compute block dims ----
  int64_t blockDims[3] = {1, 1, 1};
  SmallVector<scf::ForallOp> warpForalls;
  SmallVector<scf::ForallOp> threadForalls;

  auto walkResult = blockForall.walk([&](scf::ForallOp inner) {
    if (inner == blockForall)
      return WalkResult::advance();

    // ---- Collect warp-mapped foralls (GPUWarpMappingAttr) ----
    if (hasWarpMapping(inner)) {
      warpForalls.push_back(inner);
      auto warpUBs = inner.getMixedUpperBound();
      for (auto ub : warpUBs) {
        if (!getConstantIntValue(ub)) {
          inner.emitError("non-static warp forall upper bound");
          return WalkResult::interrupt();
        }
      }
      // Block dim must accommodate numWarps × warpSize (32).
      // All warp foralls use linear mapping → total warps = product of bounds.
      int64_t numWarps = 1;
      for (auto ub : warpUBs) {
        if (auto cst = getConstantIntValue(ub))
          numWarps *= *cst;
      }
      blockDims[0] = std::max(blockDims[0], numWarps * 32);
      return WalkResult::advance();
    }

    // ---- Collect thread-mapped foralls (GPUThreadMappingAttr) ----
    if (!hasThreadMapping(inner))
      return WalkResult::advance();

    threadForalls.push_back(inner);
    auto threadUBs = inner.getMixedUpperBound();

    for (auto ub : threadUBs) {
      if (!getConstantIntValue(ub)) {
        inner.emitError("non-static thread forall upper bound");
        return WalkResult::interrupt();
      }
    }

    if (isLinearThreadMapping(inner)) {
      // Linear mapping: total threads = product → blockDim.x.
      int64_t totalThreads = 1;
      for (auto ub : threadUBs) {
        if (auto cst = getConstantIntValue(ub))
          totalThreads *= *cst;
      }
      blockDims[0] = std::max(blockDims[0], totalThreads);
    } else {
      // 3D mapping: getMappingId() returns DimX=0, DimY=1, DimZ=2.
      auto threadMapping = inner.getMappingAttr().getValue();
      for (auto [tidx, tattr] : llvm::enumerate(threadMapping)) {
        auto threadAttr = cast<gpu::GPUThreadMappingAttr>(tattr);
        if (auto dimCst = getConstantIntValue(threadUBs[tidx])) {
          int64_t mid = threadAttr.getMappingId();
          if (mid < 3)
            blockDims[mid] = std::max(blockDims[mid], *dimCst);
        }
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return failure();
  }

  // ---- ALGORITHM STEP 3: Clamp block dims to CUDA's 1024-thread limit ----
  //
  // PERFORMANCE CRITICAL: CUDA's maximum threads per block is 1024. If the
  // computed blockDims exceed this (e.g., due to an oversized thread tile from
  // the config), the PTX JIT rejects the module with CUDA_ERROR_INVALID_PTX.
  // Excess threads are redistributed into the grid (blockDims[0] overflow goes
  // to gridDims[0]) and blockDims[0] is hard-clamped at 1024.
  static constexpr int64_t kMaxThreadsPerBlock = 1024;
  if (blockDims[0] > kMaxThreadsPerBlock) {
    blockForall.emitWarning()
        << "[nova-gpu-map-forall] blockDims.x=" << blockDims[0]
        << " exceeds CUDA limit of " << kMaxThreadsPerBlock
        << "; clamping to " << kMaxThreadsPerBlock
        << ". Grid will absorb the overflow.";
    // Overflow factor: push extra work into the block-dim grid.
    int64_t overflow = (blockDims[0] + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    gridDims[0] *= overflow;
    blockDims[0] = kMaxThreadsPerBlock;
  }

  // ---- ALGORITHM STEP 4: Create gpu.launch ----
  rewriter.setInsertionPoint(blockForall);
  auto cstIdx = [&](int64_t v) {
    return rewriter.create<arith::ConstantIndexOp>(loc, v);
  };

  auto launchOp = rewriter.create<gpu::LaunchOp>(
      loc, cstIdx(gridDims[0]), cstIdx(gridDims[1]), cstIdx(gridDims[2]),
      cstIdx(blockDims[0]), cstIdx(blockDims[1]), cstIdx(blockDims[2]));

  // gpu.launch body is created with block args but NO terminator.
  // We must add one explicitly.
  Block &launchBody = launchOp.getBody().front();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&launchBody);
    rewriter.create<gpu::TerminatorOp>(loc);
  }

  // ---- ALGORITHM STEP 5: Replace block forall IVs with gpu.block_id ----
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&launchBody);

    if (linearBlock) {
      // Linear block: linearize block_id and decompose.
      auto bidX = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
      Value linearBid = bidX.getResult();

      SmallVector<std::pair<int64_t, unsigned>> dimOrder;
      for (auto [idx, attr] : llvm::enumerate(blockMapping)) {
        auto bAttr = cast<gpu::GPUBlockMappingAttr>(attr);
        dimOrder.emplace_back(bAttr.getRelativeIndex(), idx);
      }
      llvm::sort(dimOrder,
                 [](auto &a, auto &b) { return a.first < b.first; });

      int64_t stride = 1;
      for (auto [linearDimIdx, forallIvIdx] : dimOrder) {
        int64_t bound = *getConstantIntValue(blockUBs[forallIvIdx]);
        Value strideVal = cstIdx(stride);
        Value div =
            rewriter.create<arith::DivUIOp>(loc, linearBid, strideVal);
        Value boundVal = cstIdx(bound);
        Value iv = rewriter.create<arith::RemUIOp>(loc, div, boundVal);
        blockForall.getInductionVar(forallIvIdx).replaceAllUsesWith(iv);
        stride *= bound;
      }
    } else {
      // 3D block mapping.
      for (auto [idx, attr] : llvm::enumerate(blockMapping)) {
        auto blockAttr = cast<gpu::GPUBlockMappingAttr>(attr);
        int64_t mid = blockAttr.getMappingId();
        auto dim = mappingIdToDim(mid);
        auto blockIdOp = rewriter.create<gpu::BlockIdOp>(loc, dim);
        blockForall.getInductionVar(idx).replaceAllUsesWith(blockIdOp);
      }
    }
  }

  // ---- ALGORITHM STEP 6: Erase block forall terminator; move body to launch ----
  // The in_parallel terminator (empty after bufferization) must be erased.
  rewriter.eraseOp(blockForall.getTerminator());

  Block *forallBody = blockForall.getBody();
  auto insertPt = launchBody.without_terminator().end();
  launchBody.getOperations().splice(insertPt, forallBody->getOperations());

  // ---- ALGORITHM STEP 7a: Convert warp foralls inside the launch body ----
  //
  // Warp foralls use GPUWarpMappingAttr (linear). Each warp forall IV is
  // replaced by: warpId = threadIdx.x / 32, then decomposed into per-dim
  // warp indices. Warp foralls are processed FIRST because they are outer
  // (thread foralls are nested inside them for SIMT ops, or absent for MMA).
  for (auto warpForall : warpForalls) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(warpForall);
    auto warpMapping = warpForall.getMappingAttr().getValue();
    auto warpUBs = warpForall.getMixedUpperBound();

    // warpId = threadIdx.x / 32
    auto tidX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value warpId = rewriter.create<arith::DivUIOp>(loc, tidX, cstIdx(32));

    // Decompose warpId into per-dim warp IVs (linear warp mapping).
    SmallVector<std::pair<int64_t, unsigned>> dimOrder;
    for (auto [idx, attr] : llvm::enumerate(warpMapping)) {
      auto wAttr = cast<gpu::GPUWarpMappingAttr>(attr);
      dimOrder.emplace_back(wAttr.getRelativeIndex(), idx);
    }
    llvm::sort(dimOrder,
               [](auto &a, auto &b) { return a.first < b.first; });

    int64_t stride = 1;
    for (auto [linearDimIdx, forallIvIdx] : dimOrder) {
      int64_t bound = *getConstantIntValue(warpUBs[forallIvIdx]);
      Value strideVal = cstIdx(stride);
      Value div = rewriter.create<arith::DivUIOp>(loc, warpId, strideVal);
      Value boundVal = cstIdx(bound);
      Value iv = rewriter.create<arith::RemUIOp>(loc, div, boundVal);
      warpForall.getInductionVar(forallIvIdx).replaceAllUsesWith(iv);
      stride *= bound;
    }

    // Splice warp forall body into parent, erase the forall shell.
    rewriter.eraseOp(warpForall.getTerminator());
    Block *warpBody = warpForall.getBody();
    Block *parentBlock = warpForall->getBlock();
    parentBlock->getOperations().splice(Block::iterator(warpForall),
                                        warpBody->getOperations());
    rewriter.eraseOp(warpForall);
  }

  // ---- ALGORITHM STEP 7b: Convert thread foralls inside the launch body ----
  for (auto threadForall : threadForalls) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(threadForall);
    auto threadMapping = threadForall.getMappingAttr().getValue();
    auto threadUBs = threadForall.getMixedUpperBound();

    // IMPORTANT: All thread forall bounds must be static constants.
    // Dynamic bounds arise from non-aligned dimensions that were not padded
    // (NovaGPUPadOperandsPass). This error indicates a missing padding step.
    for (auto [idx, ub] : llvm::enumerate(threadUBs)) {
      if (!getConstantIntValue(ub))
        return threadForall.emitError(
            "thread forall upper bound at dim ")
               << idx << " is not a static constant; "
               << "ensure all ops are padded to tile-aligned sizes";
    }

    if (isLinearThreadMapping(threadForall)) {
      // Linear thread mapping: tid = threadIdx.x.
      auto tidX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      Value linearTid = tidX.getResult();

      // Decompose linear tid into per-dim IVs.
      SmallVector<std::pair<int64_t, unsigned>> dimOrder;
      for (auto [idx, attr] : llvm::enumerate(threadMapping)) {
        auto tAttr = cast<gpu::GPUThreadMappingAttr>(attr);
        dimOrder.emplace_back(tAttr.getRelativeIndex(), idx);
      }
      llvm::sort(dimOrder,
                 [](auto &a, auto &b) { return a.first < b.first; });

      int64_t stride = 1;
      int64_t totalThreads = 1;
      for (auto ub : threadUBs)
        totalThreads *= *getConstantIntValue(ub);

      for (auto [linearDimIdx, forallIvIdx] : dimOrder) {
        int64_t bound = *getConstantIntValue(threadUBs[forallIvIdx]);
        Value strideVal = cstIdx(stride);
        Value div =
            rewriter.create<arith::DivUIOp>(loc, linearTid, strideVal);
        Value boundVal = cstIdx(bound);
        Value iv = rewriter.create<arith::RemUIOp>(loc, div, boundVal);
        threadForall.getInductionVar(forallIvIdx).replaceAllUsesWith(iv);
        stride *= bound;
      }

      // Erase thread forall terminator and move body.
      rewriter.eraseOp(threadForall.getTerminator());
      Block *threadBody = threadForall.getBody();

      // IMPORTANT: Predication for idle threads.
      // When the thread forall has fewer threads than the block dim (because
      // blockDims = max across all foralls in the launch), some threads in the
      // block have no work to do for this forall. We guard with an if-pred to
      // prevent out-of-bounds side effects.
      if (totalThreads < blockDims[0]) {
        Value pred = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, linearTid, cstIdx(totalThreads));
        auto ifOp =
            rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
        ifOp.thenBlock()->getOperations().splice(ifOp.thenBlock()->begin(),
                                                 threadBody->getOperations());
      } else {
        // Exact match — no predication needed.
        Block *parentBlock = threadForall->getBlock();
        parentBlock->getOperations().splice(Block::iterator(threadForall),
                                            threadBody->getOperations());
      }
      rewriter.eraseOp(threadForall);

    } else {
      // 3D thread mapping.
      for (auto [idx, attr] : llvm::enumerate(threadMapping)) {
        auto tAttr = cast<gpu::GPUThreadMappingAttr>(attr);
        int64_t mid = tAttr.getMappingId();
        auto dim = mappingIdToDim(mid);
        auto tidOp = rewriter.create<gpu::ThreadIdOp>(loc, dim);
        threadForall.getInductionVar(idx).replaceAllUsesWith(tidOp);
      }

      // Predication for dimensions where block dims > forall bounds.
      Value pred;
      for (auto [idx, attr] : llvm::enumerate(threadMapping)) {
        auto tAttr = cast<gpu::GPUThreadMappingAttr>(attr);
        int64_t bound = *getConstantIntValue(threadUBs[idx]);
        int64_t mid = tAttr.getMappingId();
        if (mid < 3 && bound < blockDims[mid]) {
          auto dim = mappingIdToDim(mid);
          auto tidOp = rewriter.create<gpu::ThreadIdOp>(loc, dim);
          auto cmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, tidOp, cstIdx(bound));
          pred = pred
                     ? rewriter.create<arith::AndIOp>(loc, pred, cmp)
                           .getResult()
                     : cmp.getResult();
        }
      }

      rewriter.eraseOp(threadForall.getTerminator());
      Block *threadBody = threadForall.getBody();

      if (pred) {
        auto ifOp =
            rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
        ifOp.thenBlock()->getOperations().splice(ifOp.thenBlock()->begin(),
                                                 threadBody->getOperations());
      } else {
        Block *parentBlock = threadForall->getBlock();
        parentBlock->getOperations().splice(Block::iterator(threadForall),
                                            threadBody->getOperations());
      }
      rewriter.eraseOp(threadForall);
    }
  }

  // ---- ALGORITHM STEP 8: Erase original block forall ----
  rewriter.eraseOp(blockForall);
  return success();
}

// ===== Pass ================================================================

struct NovaGPUMapForallToGPUPass
    : public PassWrapper<NovaGPUMapForallToGPUPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUMapForallToGPUPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Collect outermost block-mapped foralls (snapshot before modification).
    SmallVector<scf::ForallOp> blockForalls;
    funcOp.walk([&](scf::ForallOp forall) {
      if (!forall->getParentOfType<scf::ForallOp>() && hasBlockMapping(forall))
        blockForalls.push_back(forall);
    });

    for (auto forall : blockForalls) {
      if (failed(convertBlockForallToLaunch(rewriter, forall))) {
        signalPassFailure();
        return;
      }
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-map-forall-to-gpu";
  }
  StringRef getDescription() const override {
    return "Convert block/thread-mapped scf.forall to gpu.launch with dynamic "
           "block dims";
  }
};

std::unique_ptr<Pass> createNovaGPUMapForallToGPUPass() {
  return std::make_unique<NovaGPUMapForallToGPUPass>();
}

void registerNovaGPUMapForallToGPUPass() {
  PassRegistration<NovaGPUMapForallToGPUPass>();
}

} // namespace mlir::nova
