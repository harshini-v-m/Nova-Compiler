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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

/// Static SMEM cap. Allocations whose summed bytes fit within this stay on the
/// static `memref.global` path (NovaConvertSharedMemAllocsPass). When the sum
/// exceeds it, the only way to actually allocate that much SMEM on sm_86/sm_89
/// is `gpu.dynamic_shared_memory` + `cudaFuncSetAttribute(...)` at launch.
static constexpr int64_t kStaticSharedMemBudgetBytes = 48 * 1024;

/// True when `t` carries #gpu.address_space<workgroup>.
static bool hasWorkgroupAddressSpace(MemRefType t) {
  auto as = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return as && as.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// If the workgroup `memref.alloc` ops in `launch.getBody()` together exceed
/// 48 KB, replace them with views over a single `gpu.dynamic_shared_memory`
/// buffer and set `launch.dynamicSharedMemorySize`. Otherwise leave them
/// alone — the static memref.global path handles them within the 48 KB cap.
///
/// The byte threshold matches the per-kernel budget enforced by
/// NovaConvertSharedMemAllocsPass; below it we keep the simpler static path
/// (one memref.global per alloc), above it we must use dynamic SMEM because
/// PTX caps a single static workgroup allocation at 48 KB on sm_86/sm_89.
static void maybeMaterializeDynamicSharedMemory(gpu::LaunchOp launch) {
  MLIRContext *ctx = launch.getContext();
  Block &body = launch.getBody().front();
  Location loc = launch.getLoc();

  // Collect static workgroup allocs (preserving program order).
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
    bytes = llvm::alignTo(bytes, (int64_t)16);
    byteOffsets[i] = running;
    byteSizes[i]   = bytes;
    running += bytes;
  }
  int64_t totalBytes = running;

  // GATE: only switch to dynamic SMEM when the static path can't fit.
  // if (totalBytes <= kStaticSharedMemBudgetBytes) {
  //   LLVM_DEBUG(llvm::dbgs() << "[nova-gpu-map-forall] keeping static SMEM path: "
  //                           << wgAllocs.size() << " alloc(s), " << totalBytes
  //                           << " bytes (<= " << kStaticSharedMemBudgetBytes
  //                           << ")\n");
  //   return;
  // }

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

  LLVM_DEBUG(llvm::dbgs() << "[nova-gpu-map-forall] switched to dynamic SMEM: "
                          << wgAllocs.size() << " alloc(s), "
                          << totalBytes << " bytes total (> "
                          << kStaticSharedMemBudgetBytes << ")\n");
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

  int64_t totalBlocks = 1;
  for (auto ub : blockUBs) {
    auto cst = getConstantIntValue(ub);
    if (!cst)
      return blockForall.emitError("non-static block forall upper bound");
    totalBlocks *= *cst;
  }

  if (linearBlock) {
    // Linear block mapping: product of all bounds → gridDim.x.
    gridDims[0] = totalBlocks;
  } else {
    // 3D block mapping: getMappingId() returns DimX=0, DimY=1, DimZ=2.
    for (auto [idx, attr] : llvm::enumerate(blockMapping)) {
      auto blockAttr = cast<gpu::GPUBlockMappingAttr>(attr);
      auto dimCst = getConstantIntValue(blockUBs[idx]);
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

  // ---- ALGORITHM STEP 2b: Fall back to workgroup_size attr when no thread foralls ----
  //
  // VectorDistribute emits gpu.thread_id directly (no thread-mapped foralls),
  // so threadForalls may be empty and blockDims stays (1,1,1). Read the
  // authoritative workgroup_size = array<i64: 128, 1, 1> from the parent
  // func.func attribute instead.
  if (threadForalls.empty()) {
    if (auto funcOp = blockForall->getParentOfType<func::FuncOp>()) {
      if (auto wsAttr = funcOp->getAttrOfType<DenseI64ArrayAttr>("workgroup_size")) {
        auto ws = wsAttr.asArrayRef();
        for (size_t i = 0; i < ws.size() && i < 3; ++i)
          blockDims[i] = ws[i];
      }
    }
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

      if (gridDims[1] > 1) {
        auto bidY = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
        auto dimX = rewriter.create<gpu::GridDimOp>(loc, gpu::Dimension::x);
        Value mul = rewriter.create<arith::MulIOp>(loc, bidY, dimX);
        linearBid = rewriter.create<arith::AddIOp>(loc, mul, linearBid);
      }

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

      // Predicate the whole body if the linearized grid is larger than totalBlocks.
      // This happens when overflow-splitting created a 2D grid that isn't a perfect multiple.
      rewriter.setInsertionPointToEnd(&launchBody);
      Value totalBlocksVal = cstIdx(totalBlocks);
      Value isBlockActive = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, linearBid, totalBlocksVal);

      auto ifOp = rewriter.create<scf::IfOp>(loc, isBlockActive, /*withElseRegion=*/false);
      rewriter.eraseOp(blockForall.getTerminator());
      ifOp.thenBlock()->getOperations().splice(ifOp.thenBlock()->begin(),
                                              blockForall.getBody()->getOperations());

    } else {
      // 3D block mapping.
      for (auto [idx, attr] : llvm::enumerate(blockMapping)) {
        auto blockAttr = cast<gpu::GPUBlockMappingAttr>(attr);
        int64_t mid = blockAttr.getMappingId();
        auto dim = mappingIdToDim(mid);
        auto blockIdOp = rewriter.create<gpu::BlockIdOp>(loc, dim);
        blockForall.getInductionVar(idx).replaceAllUsesWith(blockIdOp);
      }
      rewriter.eraseOp(blockForall.getTerminator());
      launchBody.getOperations().splice(launchBody.without_terminator().end(),
                                        blockForall.getBody()->getOperations());
    }
  }

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
  // Only when the per-launch total exceeds the 48 KB static cap. Within budget
  // we keep the static memref.global path (NovaConvertSharedMemAllocsPass).
  maybeMaterializeDynamicSharedMemory(launchOp);

  return success();
}

// ===== Pass ================================================================

struct NovaGPUMapForallToGPUPass
    : public PassWrapper<NovaGPUMapForallToGPUPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUMapForallToGPUPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
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

    // ---- Post-conversion: rescue stranded device-only ops ----
    //
    // After all foralls are converted to gpu.launch, several op kinds remain at
    // function scope because Nova's pipeline ran VectorDistribute and
    // HoistStaticAllocations on the mixed host+device func.func *before*
    // MapForallToGPU drew the host/device boundary:
    //
    //   (a) ConstantLike ops (arith.constant dense<...> vector types, ub.poison)
    //       shared across multiple launches. Each launch that uses such a
    //       constant would capture it as a kernel argument after outlining.
    //       Dense vector constants (vector<AxBxf32>) are illegal PTX kernel
    //       parameters and cause CUDA_ERROR_INVALID_PTX / CUDA param budget
    //       overflow. Fix: clone into every gpu.launch body that uses them and
    //       erase the original if no host-scope uses remain.
    //
    //   (b) memref.alloc with #gpu.address_space<workgroup> hoisted out of
    //       foralls by HoistStaticAllocations. Move into the launch that uses it.
    //
    //   (c) memref.dealloc of workgroup allocs — erase (workgroup memory is
    //       statically sized per block; there is no runtime free).
    //
    //   (d) gpu.thread_id emitted by VectorDistribute before any gpu.launch
    //       existed. Move into each launch body that uses it.
    //
    // For (a): clone per-launch (not move) because the same constant may be
    // shared by multiple launches. After cloning into all users the original is
    // erased only when no uses remain outside any launch body.

    Block *funcBody = &funcOp.getBody().front();

    auto isWorkgroupMemref = [](Type ty) -> bool {
      auto memTy = dyn_cast<MemRefType>(ty);
      if (!memTy) return false;
      auto sp = dyn_cast_or_null<gpu::AddressSpaceAttr>(memTy.getMemorySpace());
      return sp && sp.getValue() == gpu::AddressSpace::Workgroup;
    };

    // Collect all gpu.launch ops now present in the function.
    SmallVector<gpu::LaunchOp> launches;
    funcOp.walk([&](gpu::LaunchOp launch) { launches.push_back(launch); });

    // (a) Clone ConstantLike and gpu.thread_id ops into each launch that uses them.
    // Snapshot the function-scope ops first; we mutate the block while iterating.
    SmallVector<Operation *> funcScopeOps;
    for (Operation &op : *funcBody)
      funcScopeOps.push_back(&op);

    for (Operation *op : funcScopeOps) {
      if (!op->hasTrait<OpTrait::ConstantLike>() && !isa<gpu::ThreadIdOp>(op))
        continue;

      for (gpu::LaunchOp launch : launches) {
        Region &launchRegion = launch.getBody();

        // Collect operands of this op that are used inside this launch.
        bool hasUseInLaunch = false;
        for (Value result : op->getResults())
          for (Operation *user : result.getUsers())
            if (launchRegion.isAncestor(user->getParentRegion())) {
              hasUseInLaunch = true;
              break;
            }
        if (!hasUseInLaunch)
          continue;

        // Clone at the start of the launch body and replace uses inside it.
        OpBuilder b(&launch.getBody().front(),
                    launch.getBody().front().begin());
        Operation *cloned = b.clone(*op);
        for (auto [origRes, clonedRes] :
             llvm::zip(op->getResults(), cloned->getResults()))
          origRes.replaceUsesWithIf(clonedRes, [&](OpOperand &use) {
            return launchRegion.isAncestor(use.getOwner()->getParentRegion());
          });
      }

      // Erase the original if all uses have been replaced (no host-scope users).
      if (op->use_empty())
        rewriter.eraseOp(op);
    }

    // (b) Move workgroup allocs into their launch body; (c) erase deallocs.
    SmallVector<Operation *> toErase;
    for (Operation &op : *funcBody) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        if (!isWorkgroupMemref(allocOp.getType()))
          continue;
        // Find the launch that uses this alloc and move into it.
        for (gpu::LaunchOp launch : launches) {
          Region &launchRegion = launch.getBody();
          bool usedHere = false;
          for (Operation *user : allocOp->getUsers())
            if (launchRegion.isAncestor(user->getParentRegion())) {
              usedHere = true;
              break;
            }
          if (usedHere) {
            allocOp->moveBefore(&launch.getBody().front(),
                                launch.getBody().front().begin());
            break;
          }
        }
      } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        if (isWorkgroupMemref(deallocOp.getMemref().getType()))
          toErase.push_back(&op);
      }
    }
    for (Operation *op : toErase)
      rewriter.eraseOp(op);
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


