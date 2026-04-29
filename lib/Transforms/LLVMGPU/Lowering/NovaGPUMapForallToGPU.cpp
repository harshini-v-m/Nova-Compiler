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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

static constexpr int64_t kWarpSize = 32;


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

// ===== Dynamic shared memory materialisation ================================

/// True when `t` carries #gpu.address_space<workgroup>.
static bool hasWorkgroupAddressSpace(MemRefType t) {
  auto as = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return as && as.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Replace every workgroup `memref.alloc` in `launch.getBody()` with a
/// `memref.view` over a single `gpu.dynamic_shared_memory` buffer; sum the
/// aligned byte sizes and plug that into `launch.dynamicSharedMemorySize`.
///
/// Safe no-op if the body contains no workgroup allocs.
///
/// Rationale: the static `memref.global` path
/// (NovaConvertSharedMemAllocsPass) caps shared memory at 48 KB on sm_86/89.
/// Dynamic shared memory via `gpu.dynamic_shared_memory` is required to
/// exceed that limit (cudaFuncSetAttribute is set by the runtime wrapper).
/// For the matmul tile (128x128 with 3-stage multibuffering this is ~24 KB)
/// dynamic SMEM is already below 48 KB, so this change is numerically a
/// no-op for the matmul workload — it just replaces the static globals with
/// a single dynamic region.
static void materializeDynamicSharedMemory(gpu::LaunchOp launch) {
  MLIRContext *ctx = launch.getContext();
  Block &body = launch.getBody().front();
  Location loc = launch.getLoc();

  // Collect workgroup allocs (preserving program order).
  SmallVector<memref::AllocOp> wgAllocs;
  for (Operation &op : body) {
    if (auto alloc = dyn_cast<memref::AllocOp>(&op)) {
      auto mrTy = alloc.getType();
      if (hasWorkgroupAddressSpace(mrTy) && mrTy.hasStaticShape())
        wgAllocs.push_back(alloc);
    }
  }
  if (wgAllocs.empty())
    return;

  // Compute per-alloc byte size (rounded up to 16 B for cp.async.cg alignment)
  // and the cumulative offset table.
  SmallVector<int64_t> byteOffsets(wgAllocs.size(), 0);
  SmallVector<int64_t> byteSizes(wgAllocs.size(), 0);
  int64_t running = 0;
  for (auto [i, alloc] : llvm::enumerate(wgAllocs)) {
    MemRefType ty = alloc.getType();
    int64_t elemBits = ty.getElementTypeBitWidth();
    int64_t elems    = ty.getNumElements();
    int64_t bytes    = (elems * elemBits + 7) / 8;
    // Round up to 16-byte boundary.
    bytes = llvm::alignTo(bytes, (int64_t)16);
    byteOffsets[i] = running;
    byteSizes[i]   = bytes;
    running += bytes;
  }
  int64_t totalBytes = running;

  // ── Attach dynamic_shared_memory_size to the gpu.launch op ───────────────
  OpBuilder hostBuilder(launch);
  Value dynSmemSize = hostBuilder.create<arith::ConstantIntOp>(
      loc, /*value=*/totalBytes, /*bitwidth=*/32);
  launch.getDynamicSharedMemorySizeMutable().assign(dynSmemSize);

  // ── Emit a single gpu.dynamic_shared_memory at top of the launch body ───
  OpBuilder inBody(ctx);
  inBody.setInsertionPointToStart(&body);
  auto workgroupAS = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto i8DynTy = MemRefType::get(
      {ShapedType::kDynamic}, inBody.getIntegerType(8),
      MemRefLayoutAttrInterface{}, workgroupAS);
  Value dynBase =
      inBody.create<gpu::DynamicSharedMemoryOp>(loc, i8DynTy).getResult();

  // ── Replace each alloc with a memref.view ───────────────────────────────
  for (auto [i, alloc] : llvm::enumerate(wgAllocs)) {
    OpBuilder b(alloc);
    Value offset = b.create<arith::ConstantIndexOp>(loc, byteOffsets[i]);
    // `memref.view` requires result memref to have identity layout and 0
    // offset — the alloc already satisfies both (no layout attr).
    MemRefType viewTy = alloc.getType();
    auto view = b.create<memref::ViewOp>(
        loc, viewTy, dynBase, offset, /*sizes=*/ValueRange{});
    alloc.getResult().replaceAllUsesWith(view.getResult());
    alloc.erase();
  }

  // ── Drop any matching memref.dealloc ops (shared memory auto-frees) ──────
  SmallVector<memref::DeallocOp> staleDeallocs;
  body.walk([&](memref::DeallocOp d) {
    auto mrTy = dyn_cast<MemRefType>(d.getMemref().getType());
    if (mrTy && hasWorkgroupAddressSpace(mrTy))
      staleDeallocs.push_back(d);
  });
  for (memref::DeallocOp d : staleDeallocs)
    d.erase();

  LLVM_DEBUG(llvm::dbgs() << "[nova-gpu-map-forall] dynamic SMEM: "
                          << wgAllocs.size() << " alloc(s), "
                          << totalBytes << " bytes total\n");
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

  // Collect thread-mapped foralls and accumulate their bounds into blockDims.
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

  if (walkResult.wasInterrupted())
    return failure();

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

  // ---- ALGORITHM STEP 7a: Convert thread foralls inside the launch body ----
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

  // ---- ALGORITHM STEP 7b: Convert warp foralls → warp_id = threadIdx.x/32 ----
  // After the Subgroup tiling pass, MMA ops live inside #gpu.warp foralls.
  // One warp forall iteration corresponds to one warp: the IV is the warp
  // index, so we replace it with threadIdx.x / kWarpSize.
  for (auto warpForall : warpForalls) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(warpForall);
    auto warpUBs = warpForall.getMixedUpperBound();

    // Compute total warp count (product of all forall upper bounds).
    int64_t totalWarps = 1;
    for (auto ub : warpUBs)
      totalWarps *= *getConstantIntValue(ub);

    // warp_id = threadIdx.x / kWarpSize
    auto tidX    = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value warpSz = cstIdx(kWarpSize);
    Value warpId = rewriter.create<arith::DivUIOp>(loc, tidX, warpSz);

    // Decompose linear warp_id into per-dim IVs, mirroring the linear thread
    // mapping case. Sort by relativeIndex ascending (fastest-varying first)
    // so stride accumulates correctly.
    auto warpMapping = warpForall.getMappingAttr().getValue();
    SmallVector<std::pair<int64_t, unsigned>> dimOrder;
    for (auto [idx, attr] : llvm::enumerate(warpMapping)) {
      auto warpAttr = cast<gpu::GPUWarpMappingAttr>(attr);
      dimOrder.emplace_back(warpAttr.getRelativeIndex(), idx);
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

    // Erase in_parallel terminator then move or predicate the body.
    rewriter.eraseOp(warpForall.getTerminator());
    Block *warpBody = warpForall.getBody();

    int64_t activeThreads = totalWarps * kWarpSize;
    if (activeThreads < blockDims[0]) {
      // Some threads have no warp work — guard with scf.if.
      Value pred = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, tidX, cstIdx(activeThreads));
      auto ifOp =
          rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
      ifOp.thenBlock()->getOperations().splice(
          ifOp.thenBlock()->begin(), warpBody->getOperations());
    } else {
      // All threads are active — splice directly without predication.
      Block *parent = warpForall->getBlock();
      parent->getOperations().splice(Block::iterator(warpForall),
                                     warpBody->getOperations());
    }
    rewriter.eraseOp(warpForall);
  }

  // ---- ALGORITHM STEP 8: Erase original block forall ----
  rewriter.eraseOp(blockForall);

  // ---- ALGORITHM STEP 9: Materialise workgroup allocs as dynamic SMEM ----
  // This replaces the two per-tile `memref.alloc(workgroup)` ops (A and B
  // SMEM tiles, each multi-buffered) with a single `gpu.dynamic_shared_memory`
  // buffer plus per-tile `memref.view`s. After this, no static workgroup
  // allocs remain inside the launch body — NovaConvertSharedMemAllocsPass
  // becomes a no-op for matmul.
  materializeDynamicSharedMemory(launchOp);
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