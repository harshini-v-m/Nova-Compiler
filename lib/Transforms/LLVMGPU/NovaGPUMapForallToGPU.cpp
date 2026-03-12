//===- NovaGPUMapForallToGPU.cpp - scf.forall → gpu.launch ----------------===//
//
// Replaces the gpu_forall_to_launch.mlir transform script with a C++ pass
// that dynamically computes thread block dimensions from nested thread-mapped
// forall bounds, instead of hardcoding block_dims = [32, 32, 1].
//
// Algorithm:
//   1. Walk all outermost block-mapped scf.forall ops.
//   2. For each, compute grid dims from the block forall's upper bounds.
//   3. Find nested thread-mapped foralls and compute block dims from their
//      actual iteration bounds (linear mapping → product; 3D → per-dim max).
//   4. Create gpu.launch with correct grid/block dims.
//   5. Map block forall IVs → gpu.block_id, thread forall IVs → gpu.thread_id.
//   6. Add predication for thread IDs when multiple thread foralls share a
//      launch (block dims = max across all foralls).
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

  // ----- Step 1: Compute grid dims from block forall mapping -----
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

  // ----- Step 2: Find thread foralls and compute block dims -----
  int64_t blockDims[3] = {1, 1, 1};
  SmallVector<scf::ForallOp> threadForalls;

  blockForall.walk([&](scf::ForallOp inner) {
    if (inner == blockForall)
      return WalkResult::advance();
    if (!hasThreadMapping(inner))
      return WalkResult::advance();

    threadForalls.push_back(inner);
    auto threadUBs = inner.getMixedUpperBound();

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

  // Validate block dims: total threads must not exceed 1024.
  {
    int64_t totalThreads = blockDims[0] * blockDims[1] * blockDims[2];
    if (totalThreads > 1024) {
      return blockForall.emitError("total thread count ")
             << totalThreads << " (blockDims=" << blockDims[0] << "x"
             << blockDims[1] << "x" << blockDims[2]
             << ") exceeds hardware max of 1024 threads/block; "
             << "check thread tile sizes in lowering config";
    }
    // Clamp any 0-dim block dims to 1.
    for (int i = 0; i < 3; ++i) {
      if (blockDims[i] <= 0)
        blockDims[i] = 1;
    }
  }

  // ----- Step 3: Create gpu.launch -----
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

  // ----- Step 4: Replace block forall IVs -----
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

  // ----- Step 5: Erase block forall terminator, move body to launch -----
  // The in_parallel terminator (empty after bufferization) must be erased.
  rewriter.eraseOp(blockForall.getTerminator());

  Block *forallBody = blockForall.getBody();
  auto insertPt = launchBody.without_terminator().end();
  launchBody.getOperations().splice(insertPt, forallBody->getOperations());

  // ----- Step 6: Convert thread foralls inside the launch body -----
  for (auto threadForall : threadForalls) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(threadForall);
    auto threadMapping = threadForall.getMappingAttr().getValue();
    auto threadUBs = threadForall.getMixedUpperBound();

    // All thread forall bounds must be static constants.  Dynamic bounds
    // arise from non-aligned dimensions that were not padded.
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

      if (totalThreads < blockDims[0]) {
        // Need predication: some threads are idle.
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

  // ----- Step 7: Erase original block forall -----
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
