//===- NovaGPUSwizzleSharedMemory.cpp - XOR swizzle for SMEM ops ------===//
//
// Applies an XOR swizzle to all loads/stores of opt-in shared-memory allocs.
// Complements NovaGPUReduceBankConflicts (padding) by giving a zero-SMEM-cost
// alternative for tiles that are wide enough to benefit from it.
//
// Who opts in:
//   A workgroup memref.alloc must carry `nova.swizzle` as a unit attribute.
//   The promotion/config passes decide which tiles get it; this pass only
//   performs the mechanical rewrite.  If `nova.swizzle` is absent, this pass
//   does nothing to that alloc (leaving it to the pad pass or to no treatment).
//
// What the swizzle does:
//   For row-major memref<R x C x elemT, workgroup>, every access A[r, c] is
//   rewritten to A[r, c XOR ((r & mask) << shift)] where
//       shift  = log2(16 / elemBytes)   -- keeps 16B-contiguous blocks intact
//       mask   = (C * elemBytes / 16) - 1 when C*elemBytes is a power of 2
//              else the largest power-of-2 minus 1 that fits in C*elemBytes
//   This preserves cp.async 16B-contiguous destinations (since the XOR only
//   toggles bits above the 4-f32 block boundary) while permuting column
//   addresses so that the warp-wide set of 32 lane addresses lands in 32
//   distinct banks.
//
// What the swizzle doesn't fix:
//   Narrow tiles (C*elemBytes < 32) have too few column bits for any XOR to
//   break warp-wide collisions.  The pass recognises this and emits a log
//   message; the alloc should use padding instead.
//
// Run order:
//   Before NovaConvertSharedMemAllocs (which rewrites alloc→global).
//   Before NovaGPUReduceBankConflicts (so pad doesn't touch swizzled allocs).
//   After bufferization, vector-distribute, fill-copy-forwarding — i.e.,
//   after all passes that introduce SMEM load/store ops.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-swizzle-shared-memory"

using namespace mlir;

namespace mlir::nova {

namespace {

// Returns true if the memref type lives in the workgroup address space.
static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

// Largest power-of-2 ≤ n.
static int64_t floorLog2PowerOf2(int64_t n) {
  int64_t p = 1, e = 0;
  while ((p << 1) <= n) { p <<= 1; ++e; }
  return e;
}

// Swizzle parameters derived from an alloc's element type and innermost extent.
struct SwizzleParams {
  // XOR = (((row / perPhase) & rowMask) << shiftBits)
  int64_t rowMask;    // bit mask on the (scaled) row index
  int64_t shiftBits;  // left-shift applied to the masked row bits before XOR
  int64_t perPhase;   // consecutive rows that share the same XOR pattern.
                      // perPhase=1 gives IREE's default; perPhase=2 is needed
                      // for ldmatrix.x4 that loads 2 rows per thread (A matrix
                      // in TF32 mma.sync 16x8x8).
  bool    viable;     // false if the tile is too narrow for any useful swizzle
};

// Derive the XOR for a 2D (or higher-rank) SMEM tile whose innermost dim has
// `innerDim` elements of size `elemBytes`.
//
// Design:
//   - The shift is log2(16 / elemBytes).  On f32 this is 2 (shift by 4),
//     which means the XOR toggles bits 2..k of the column index.  Since
//     cp.async writes 4 consecutive f32 (= bits 0..1 of the column), the
//     XOR never permutes within a 4-element block — cp.async stays
//     contiguous-destination and remains valid.
//   - The rowMask gives us as many XOR variants as there are 16B chunks
//     per row.  For an 8-wide f32 tile (32 bytes per row = two 16B chunks),
//     rowMask = 1 — only one bit of row variation, not enough to break
//     the mod-4 reader collision we analysed.  Mark as not-viable so the
//     caller can log and skip.
//   - For 16-wide f32 (64 B/row = four 16B chunks), rowMask = 3, shift = 2
//     → XOR cycles through {0, 4, 8, 12} bytes on consecutive rows.
//   - For 64-wide f32 (A-tile: 256 B/row = sixteen 16B chunks), rowMask = 7
//     and shift = 2 → the standard ldmatrix-compatible pattern.
static SwizzleParams computeSwizzle(int64_t innerDim, int64_t elemBytes,
                                    int64_t perPhase = 1) {
  SwizzleParams p{0, 0, 1, false};
  int64_t rowBytes = innerDim * elemBytes;
  // A useful XOR swizzle needs at least 4 16B chunks per row (= 64 B/row →
  // 16 f32 columns).  With only 2 chunks (32 B/row) the XOR has a single
  // bit of variation, which is not enough to break the warp-wide stride
  // pattern that produces bank conflicts in the first place — the result
  // is essentially the same conflict count, just shifted around.  Tiles
  // that don't meet this threshold should use padding (or remain narrow
  // and accept the cost) rather than rely on a degenerate swizzle.
  if (rowBytes < 64) return p;
  if (elemBytes > 16) return p;                        // exotic: skip
  int64_t chunkBytes = 16;
  int64_t chunksPerRow = rowBytes / chunkBytes;        // floor
  if (chunksPerRow < 4) return p;                      // <2 useful XOR bits
  // Use the largest power of 2 ≤ chunksPerRow as the XOR range, capped at 8.
  int64_t chunksLog2 = floorLog2PowerOf2(chunksPerRow);
  if (chunksLog2 > 3) chunksLog2 = 3;                  // cap: 8 rows of variation
  p.rowMask   = (1LL << chunksLog2) - 1;
  p.shiftBits = floorLog2PowerOf2(chunkBytes / elemBytes); // log2(16/elemBytes)
  // perPhase: number of consecutive rows that share the same XOR value.
  // For ldmatrix.x4 (TF32 mma.sync 16x8x8), each thread loads 2 consecutive
  // rows of A from SMEM; both rows must have the same swizzle pattern so that
  // the hardware can issue a single ldmatrix instruction. Caller passes the
  // MMA tile height (e.g. 2 for 16x8x8 A) as perPhase.
  p.perPhase  = (perPhase > 0) ? perPhase : 1;
  p.viable = true;
  return p;
}

// Walk the use-def graph starting at `root` and collect every op that reads or
// writes memory derived from `root` (via subview / reinterpret_cast chains).
// For each collected op we record the two operands that drive the innermost
// index arithmetic; those are the values the pass will rewrite.
//
// We keep this deliberately small: only the op types the pipeline is known to
// emit against SMEM in Nova (nvgpu.device_async_copy on dst, memref.load,
// memref.store, vector.transfer_read, vector.transfer_write, vector.load,
// vector.store).  Anything else short-circuits and aborts the rewrite so we
// don't silently miss an access.
struct Access {
  Operation *op;
  // Pointer to the (row, col) operands inside op.  For the rewrite we only
  // need the column; row is captured for the XOR mask derivation.
  Value rowIdx;
  Value colIdx;
  unsigned colOperandIdx;    // operand position of the col index in op
  // For cp.async writers: the dst memref is typically `memref.subview view[a, b]
  // [1, N] [1, 1]` (or a chain of such subviews) and the cp.async indexes
  // INTO the subview, so its row/col operands are local deltas. To get the
  // XOR identical to the reader's address function we need the ABSOLUTE
  // (row, col) in the swizzled tile, which is the cp.async delta plus the
  // chain of enclosing subview offsets, AND the cp.async dst itself must be
  // re-pointed at the swizzled root (otherwise `subview_col + (delta XOR
  // mask)` is a plain ADD, not an XOR — they only agree when the masked
  // bits don't overlap subview_col, which is not the general case).
  //
  // `rowAddends` and `colAddends` are the enclosing offsets to fold into the
  // cp.async indices; `rebindDstToRoot` and `rootView` are the new dst
  // memref to plug in.  Non-cp.async users leave these empty/null and the
  // rewrite loop just XORs the existing col with `(rowIdx & mask) << shift`.
  SmallVector<OpFoldResult> rowAddends;
  SmallVector<OpFoldResult> colAddends;
  // Leading offsets from the subview chain (everything *outside* the
  // (row, col) pair).  Populated when the root view has rank > 2 — e.g.
  // after multi-buffering widens the alloc to memref<NxRxCxT> and the
  // cp.async dst is a rank-reducing subview `view[i, 0, 0][1,R,C]`.
  // The rewrite rebuilds a fresh cp.async with dst indices
  //   [outerOffsets..., absRow, newCol]
  // so the destination index count matches the (now rank>2) root.
  SmallVector<OpFoldResult> outerOffsets;
  bool rebindDstToRoot = false;
  Value rootView;
  unsigned rowOperandIdx = 0; // position of cp.async row index in op
  unsigned dstOperandIdx = 0; // position of cp.async dst memref in op
};

} // anonymous namespace

struct NovaGPUSwizzleSharedMemoryPass
    : public PassWrapper<NovaGPUSwizzleSharedMemoryPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUSwizzleSharedMemoryPass)

  StringRef getArgument() const override {
    return "nova-gpu-swizzle-shared-memory";
  }
  StringRef getDescription() const override {
    return "XOR-swizzle indices of loads/stores to opt-in (nova.swizzle-tagged) "
           "workgroup memref allocs to avoid bank conflicts without padding";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect,
                    memref::MemRefDialect, nvgpu::NVGPUDialect,
                    vector::VectorDialect>();
  }

  // Rewrite `originalCol = (originalCol) XOR ((row & rowMask) << shift)` using
  // index-typed arith.  Returns the new column value.
  static Value buildSwizzledCol(OpBuilder &b, Location loc, Value row,
                                Value col, SwizzleParams p) {
    Value maskC  = b.create<arith::ConstantIndexOp>(loc, p.rowMask);
    Value shiftC = b.create<arith::ConstantIndexOp>(loc, p.shiftBits);
    // Apply per_phase: divide row by perPhase so that consecutive rows within
    // the same phase produce the same XOR offset. Required for ldmatrix.x4
    // which reads 2 consecutive rows with a single instruction — both rows
    // must hash to the same swizzled column to produce a contiguous fragment.
    Value phaseRow = row;
    if (p.perPhase > 1) {
      Value phaseC = b.create<arith::ConstantIndexOp>(loc, p.perPhase);
      phaseRow = b.create<arith::DivUIOp>(loc, row, phaseC);
    }
    Value masked = b.create<arith::AndIOp>(loc, phaseRow, maskC);
    Value shifted = b.create<arith::ShLIOp>(loc, masked, shiftC);
    return b.create<arith::XOrIOp>(loc, col, shifted);
  }

  // Rewrite the innermost index of every SMEM access derived from allocRoot.
  // Returns true on success; false if we hit an op we don't know how to
  // handle (in which case we leave the IR untouched so we don't break it).
  bool rewriteAccessesOf(Value allocRoot, SwizzleParams p) {
    // BFS through all users, transparently walking past subview / cast ops.
    //
    // We carry per-value (rowAddends, colAddends) accumulators so that when
    // a load/store is reached through a subview chain, we know the absolute
    // (row, col) in the swizzle root's coordinate system. Without this, the
    // reader path used the load's *local* indices (which are 0 when the load
    // is `transfer_read %sv[%c0, %c0]` over a subview that already absorbs
    // the row/col offset), so the XOR was a no-op while the cp.async writer
    // — which already walks the chain — was producing a swizzled address.
    // Mismatched writer/reader address functions produced garbage values.
    struct WorkItem {
      Value v;
      SmallVector<OpFoldResult> rowAddends;
      SmallVector<OpFoldResult> colAddends;
    };
    SmallVector<WorkItem> worklist;
    worklist.push_back({allocRoot, {}, {}});
    SmallPtrSet<Operation *, 32> visited;
    SmallVector<Access> hits;
    // Subviews whose innermost offset must be rewritten.  See case 2 in the
    // cp.async handler.
    llvm::SmallSetVector<memref::SubViewOp, 8> subviewRewrites;

    auto pushNonZero = [](SmallVector<OpFoldResult> &dst, OpFoldResult o) {
      if (auto attr = dyn_cast<Attribute>(o)) {
        if (auto ia = dyn_cast_or_null<IntegerAttr>(attr)) {
          if (ia.getInt() != 0) dst.push_back(o);
          return;
        }
      }
      dst.push_back(o);
    };

    while (!worklist.empty()) {
      WorkItem wi = worklist.pop_back_val();
      Value v = wi.v;
      for (Operation *user : v.getUsers()) {
        if (!visited.insert(user).second) continue;

        // Transparent view ops: accumulate offsets where applicable and push
        // the result through. memref.subview contributes its row/col offsets
        // to the running addends. The other view ops don't change the linear
        // address, but expand/collapse_shape DO change index space — bail
        // for safety because we don't track the index reshape.
        if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
          SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
          if (offs.size() < 2) {
            LLVM_DEBUG(llvm::dbgs() << "[nova-swizzle-shared-memory] "
                                       "subview rank<2 in chain\n");
            return false;
          }
          WorkItem child;
          child.v = sv.getResult();
          child.rowAddends = wi.rowAddends;
          child.colAddends = wi.colAddends;
          pushNonZero(child.rowAddends, offs[offs.size() - 2]);
          pushNonZero(child.colAddends, offs[offs.size() - 1]);
          worklist.push_back(std::move(child));
          continue;
        }
        if (isa<memref::ReinterpretCastOp, memref::CastOp>(user)) {
          for (Value r : user->getResults()) {
            WorkItem child;
            child.v = r;
            child.rowAddends = wi.rowAddends;
            child.colAddends = wi.colAddends;
            worklist.push_back(std::move(child));
          }
          continue;
        }
        if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(user)) {
          // Index-space changes — we can't track the reshape mapping safely.
          LLVM_DEBUG(llvm::dbgs() << "[nova-swizzle-shared-memory] "
                                     "expand/collapse_shape in chain — "
                                     "bailing\n");
          return false;
        }

        // Loads / stores we know how to rewrite. We use the last two index
        // operands as (row, col); the absolute coordinates are obtained by
        // adding the subview-chain addends accumulated above. The rewrite
        // happens by queuing the immediate parent subview (if any) so its
        // column offset becomes swizzled, OR — when there is no parent
        // subview — by XOR-ing the load's own column index directly using
        // the absolute row.
        auto recordAccess = [&](Operation *op, Value rowIdx, Value colIdx,
                                unsigned colOpIdx) {
          Access acc;
          acc.op = op;
          acc.rowIdx = rowIdx;
          acc.colIdx = colIdx;
          acc.colOperandIdx = colOpIdx;
          acc.rowAddends = wi.rowAddends;
          acc.colAddends = wi.colAddends;
          hits.push_back(std::move(acc));
        };

        if (auto ld = dyn_cast<memref::LoadOp>(user)) {
          auto indices = ld.getIndices();
          if (indices.size() < 2) continue;
          recordAccess(user, indices[indices.size() - 2], indices.back(),
                       (unsigned)(ld.getODSOperandIndexAndLength(1).first +
                                  indices.size() - 1));
          continue;
        }
        if (auto st = dyn_cast<memref::StoreOp>(user)) {
          auto indices = st.getIndices();
          if (indices.size() < 2) continue;
          recordAccess(user, indices[indices.size() - 2], indices.back(),
                       (unsigned)(st.getODSOperandIndexAndLength(2).first +
                                  indices.size() - 1));
          continue;
        }
        if (auto tr = dyn_cast<vector::TransferReadOp>(user)) {
          auto indices = tr.getIndices();
          if (indices.size() < 2) continue;
          recordAccess(user, indices[indices.size() - 2], indices.back(),
                       (unsigned)(tr.getODSOperandIndexAndLength(1).first +
                                  indices.size() - 1));
          continue;
        }
        if (auto tw = dyn_cast<vector::TransferWriteOp>(user)) {
          auto indices = tw.getIndices();
          if (indices.size() < 2) continue;
          recordAccess(user, indices[indices.size() - 2], indices.back(),
                       (unsigned)(tw.getODSOperandIndexAndLength(2).first +
                                  indices.size() - 1));
          continue;
        }
        if (auto cp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(user)) {
          // Only rewrite when allocRoot is the DESTINATION (shared-mem side).
          // Source-side (global) has no bank-conflict notion.
          if (cp.getDst() != v && !isDerivedFrom(cp.getDst(), allocRoot))
            continue;
          auto dstIndices = cp.getDstIndices();
          if (dstIndices.size() < 2) continue;

          // Case 1: cp.async's own row operand is non-constant — the standard
          // case where we swizzle the cp.async index directly.
          //
          // ODS operand layout of nvgpu.device_async_copy:
          //   group 0: dst             (single)
          //   group 1: dstIndices      (variadic)        ← we want this
          //   group 2: src             (single)
          //   group 3: srcIndices      (variadic)
          // Earlier this used getODSOperandIndexAndLength(2), which pointed
          // into the SOURCE indices instead of the destination indices, so
          // the swizzle was rewriting the global-memory src offset, leaving
          // the SMEM dst linear.
          //
          // The XOR formula is `col XOR ((absRow & mask) << shift)`. For
          // readers we get absRow directly (their load is on a flattened
          // reinterpret_cast). For cp.async writers the dst is a chain of
          // `memref.subview` ops over the swizzled root, and the cp.async's
          // own row operand is only the LAST delta — the absolute row equals
          // the sum of all enclosing subview row-offsets plus the cp.async
          // delta. Walk that chain here and record the row addends; the
          // rewrite loop materialises the absolute row before XOR-ing.
          //
          // Without this, the writer's XOR uses only `arg26 ∈ {0,1}`, which
          // ignores the per-thread base row carried by the subview offset.
          // The reader uses the full absolute row, so writer / reader
          // address different SMEM cells → wrong results. Matches the
          // eager kernel pattern: `sc = c ^ ((r & mask) << shift)` where r
          // is the absolute SMEM row, not a per-iter delta.
          // Walk the subview chain and accumulate BOTH row and col offsets.
          // Stop at the first op that isn't a transparent view; if we don't
          // reach the swizzled root view, leave rebindDstToRoot=false and
          // fall back to the simple "rewrite col only" path (better than
          // bailing — the simple path handles the no-subview case).
          SmallVector<OpFoldResult> rowAddends;
          SmallVector<OpFoldResult> colAddends;
          // Leading offsets in the *root* memref's coordinate system,
          // collected outermost-first.  We accumulate per-subview-step
          // contributions to each leading dim and emit them as additional
          // dst indices when the root rank exceeds 2 (multi-buffer case).
          SmallVector<SmallVector<OpFoldResult>> outerAddendsPerDim;
          bool reachedRoot = false;
          {
            Value src = cp.getDst();
            while (true) {
              if (src == allocRoot) { reachedRoot = true; break; }
              if (auto sv = src.getDefiningOp<memref::SubViewOp>()) {
                SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
                if (offs.size() >= 2) {
                  auto pushNonZero = [](SmallVector<OpFoldResult> &dst,
                                        OpFoldResult o) {
                    if (auto attr = dyn_cast<Attribute>(o)) {
                      if (auto ia = dyn_cast_or_null<IntegerAttr>(attr)) {
                        if (ia.getInt() != 0) dst.push_back(o);
                        return;
                      }
                    }
                    dst.push_back(o);
                  };
                  pushNonZero(rowAddends, offs[offs.size() - 2]);
                  pushNonZero(colAddends, offs[offs.size() - 1]);
                  // Capture every offset *before* (row, col).  The subview
                  // source's rank gives us the absolute leading-dim count
                  // for this step; rank-reducing subviews with dropped
                  // leading dims still appear in the source-rank offset
                  // list (mixed offsets are in source-rank space).
                  size_t numOuter = offs.size() - 2;
                  if (outerAddendsPerDim.size() < numOuter)
                    outerAddendsPerDim.resize(numOuter);
                  for (size_t i = 0; i < numOuter; ++i)
                    pushNonZero(outerAddendsPerDim[i], offs[i]);
                }
                src = sv.getSource();
                continue;
              }
              break; // unknown intermediate op — keep what we have
            }
          }

          Value rowOp = dstIndices[dstIndices.size() - 2];
          if (!matchPattern(rowOp, m_Constant()) || reachedRoot) {
            unsigned base = cp.getODSOperandIndexAndLength(1).first;
            Access acc;
            acc.op = user;
            acc.rowIdx = rowOp;
            acc.colIdx = dstIndices.back();
            acc.colOperandIdx = base + (unsigned)(dstIndices.size() - 1);
            acc.rowAddends = std::move(rowAddends);
            acc.colAddends = std::move(colAddends);
            // Only rebind to root when we successfully walked to the root
            // AND the cp.async accesses a 2-D location (rank ≥ 2).
            if (reachedRoot && dstIndices.size() >= 2) {
              acc.rebindDstToRoot = true;
              acc.rootView = allocRoot;
              acc.dstOperandIdx = cp.getODSOperandIndexAndLength(0).first;
              acc.rowOperandIdx = base + (unsigned)(dstIndices.size() - 2);
              // For multi-buffered roots (rank > 2) we need additional
              // leading dst indices that the cp.async currently lacks.
              // Stash the per-leading-dim addend lists; the rewrite loop
              // sums each list at the cp.async insertion point and uses
              // the sums as the leading dst indices.
              auto rootMR = dyn_cast<MemRefType>(allocRoot.getType());
              if (rootMR && rootMR.getRank() > 2) {
                size_t numLeading = (size_t)rootMR.getRank() - 2;
                outerAddendsPerDim.resize(numLeading);
                acc.outerOffsets.reserve(numLeading);
                // outerAddendsPerDim is indexed in source-rank space of the
                // OUTER subview (which equals the root rank for the chain
                // we walked, modulo any rank-reducing subview that drops
                // the leading dims — those drops are reflected as offsets
                // we already captured).  Materialise one OpFoldResult per
                // leading dim by folding its addend list into a single
                // expression at rewrite time; here we just pass the lists
                // through as a flattened sentinel-separated form by
                // delegating to the rewriter — see Edit 3.
                for (size_t i = 0; i < numLeading; ++i) {
                  // Empty addend list → leading offset is 0.  Encode that
                  // as a 0 IndexAttr so the rewriter doesn't need a
                  // separate empty-list code path.
                  if (outerAddendsPerDim[i].empty()) {
                    acc.outerOffsets.push_back(
                        OpFoldResult(IntegerAttr::get(
                            IndexType::get(allocRoot.getContext()), 0)));
                  } else if (outerAddendsPerDim[i].size() == 1) {
                    acc.outerOffsets.push_back(outerAddendsPerDim[i][0]);
                  } else {
                    // Multiple addends in one leading dim are unusual
                    // (would require nested subviews each contributing to
                    // the same outer dim).  Bail on the rebind for safety
                    // — falling back to col-only XOR is preferable to
                    // emitting a wrong index.
                    acc.rebindDstToRoot = false;
                    acc.outerOffsets.clear();
                    break;
                  }
                }
              }
            }
            hits.push_back(std::move(acc));
            continue;
          }

          // Case 2: cp.async row operand is constant 0 because the real
          // per-thread row lives in the subview offset (rank-reducing
          // subview with outer size 1, typical for Nova's B-writer).
          // Walk back to the subview and rewrite ITS innermost offset:
          //    subview src[%row, %col] [1, N]  →
          //    %col' = %col XOR ((%row & mask) << shift)
          //    subview src[%row, %col'] [1, N]
          // This gives the writer the same effective SMEM address function
          // the reader already gets from rewriting its load indices.
          auto svDst = cp.getDst().getDefiningOp<memref::SubViewOp>();
          if (!svDst) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[nova-swizzle-shared-memory] cp.async writer "
                          "row is constant but dst is not a subview — cannot "
                          "swizzle writer; bailing\n");
            return false;
          }
          // The subview must have outer size 1 for its offset to carry the
          // row (the cp.async addresses only within the 1-sized outer dim).
          auto dstType = cast<MemRefType>(svDst.getType());
          if (dstType.getRank() < 2 || dstType.getShape()[0] != 1) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[nova-swizzle-shared-memory] cp.async writer "
                          "subview shape unexpected; bailing\n");
            return false;
          }
          // Queue a subview-offset rewrite instead of an op-index rewrite.
          // We encode it via a sentinel: op = svDst, colOperandIdx = kSubviewColOffset.
          subviewRewrites.insert(svDst);
          continue;
        }

        // Unknown op on an SMEM-derived value — refuse to rewrite anything so
        // we don't desync some accesses relative to others.
        LLVM_DEBUG(llvm::dbgs() << "[nova-swizzle-shared-memory] unknown "
                                   "SMEM user: " << *user << "\n");
        return false;
      }
    }

    // Do the rewrites.
    //
    // Two cases:
    //  (1) Non-cp.async users (load/store/transfer): rowIdx is already the
    //      absolute SMEM row → just XOR the col with `(rowIdx & mask) << shift`.
    //  (2) cp.async writers with rebindDstToRoot=true: the cp.async dst is
    //      `subview view[r_off, c_off]` and its (row, col) operands are
    //      local deltas. We materialise:
    //          abs_row = row_delta + sum(rowAddends)
    //          abs_col = col_delta + sum(colAddends)
    //          swiz_col = abs_col XOR ((abs_row & mask) << shift)
    //      then re-point the cp.async dst at the root view itself with
    //      [abs_row, swiz_col], dropping the subview indirection. This is
    //      the only formulation that gives writer and reader the *same*
    //      address function, since the reader's `(absRow & mask) << shift`
    //      XORs against the absolute column too — applying XOR to a delta
    //      and then ADDING a non-zero subview col offset would produce a
    //      different address whenever the masked bits and the subview
    //      offset bits overlap.
    for (Access &a : hits) {
      OpBuilder b(a.op);
      Location loc = a.op->getLoc();

      auto sumAddends = [&](Value base,
                            ArrayRef<OpFoldResult> addends) -> Value {
        Value v = base;
        for (OpFoldResult addend : addends) {
          Value addV = getValueOrCreateConstantIndexOp(b, loc, addend);
          v = b.create<arith::AddIOp>(loc, v, addV);
        }
        return v;
      };

      Value absRow = sumAddends(a.rowIdx, a.rowAddends);
      Value absCol = sumAddends(a.colIdx, a.colAddends);
      Value newCol = buildSwizzledCol(b, loc, absRow, absCol, p);

      if (a.rebindDstToRoot) {
        // cp.async: re-point dst at root view with [absRow, newCol].
        // For 2-D roots an in-place setOperand suffices because the
        // index count already matches.  For rank>2 roots (multi-buffered
        // ring), the existing cp.async carries only 2 dst indices, but
        // the new dst memref needs N=root.rank() indices.  Rebuild the
        // op with the full leading-offsets list folded in.
        auto rootMR = cast<MemRefType>(a.rootView.getType());
        if (rootMR.getRank() <= 2) {
          a.op->setOperand(a.dstOperandIdx, a.rootView);
          a.op->setOperand(a.rowOperandIdx, absRow);
          a.op->setOperand(a.colOperandIdx, newCol);
        } else {
          auto cp = cast<nvgpu::DeviceAsyncCopyOp>(a.op);
          // Materialise leading dst indices from outerOffsets.
          SmallVector<Value> dstIndices;
          dstIndices.reserve(rootMR.getRank());
          for (OpFoldResult o : a.outerOffsets)
            dstIndices.push_back(getValueOrCreateConstantIndexOp(b, loc, o));
          dstIndices.push_back(absRow);
          dstIndices.push_back(newCol);
          // Carry the existing src + srcIndices verbatim; only the dst
          // side changes (rank, indices).
          OpBuilder rb(cp);
          auto newOp = rb.create<nvgpu::DeviceAsyncCopyOp>(
              cp.getLoc(), cp.getResult().getType(),
              a.rootView, dstIndices,
              cp.getSrc(), cp.getSrcIndices(),
              cp.getDstElementsAttr(),
              cp.getSrcElements(),
              cp.getBypassL1Attr());
          cp.getResult().replaceAllUsesWith(newOp.getResult());
          cp.erase();
          a.op = newOp; // keep Access valid in case any later code touches it
        }
      } else {
        // Non-cp.async users (load / store / vector.transfer_*).  The op's
        // own column operand (`a.colIdx`) is in *subview-local* space; the
        // absolute column is `localCol + sum(colAddends)`.  We just XOR-ed
        // the absolute column to get `newCol = absCol XOR mask_shift`.  We
        // can't write `newCol` straight into the local-col operand because
        // the subview will then add its own col offset on top, giving
        // `absCol + newCol` instead of `newCol`.
        //
        // Compute the delta `newCol - absCol` and add it to the local col
        // operand instead.  The strided-memref address calculation then
        // yields:
        //     base + absRow*rowStride + (subviewColOffset + (localCol + delta))
        //   = base + absRow*rowStride + (absCol + delta)
        //   = base + absRow*rowStride + newCol     ✓
        //
        // When colAddends is empty (no enclosing subview offset), absCol
        // equals localCol and `localCol + delta = localCol + newCol -
        // localCol = newCol`, recovering the original simple rewrite.
        Value delta = b.create<arith::SubIOp>(loc, newCol, absCol);
        Value newLocalCol =
            b.create<arith::AddIOp>(loc, a.colIdx, delta);
        a.op->setOperand(a.colOperandIdx, newLocalCol);
      }
    }

    // Rewrite subview offsets for writers where the real row lives in the
    // subview offset (Nova's B promotion pattern).  For each queued subview:
    //   old:  memref.subview src[%row, %col] [1, N] [1, 1]
    //   new:  memref.subview src[%row, %col XOR ((%row & mask) << shift)]
    //         [1, N] [1, 1]
    for (memref::SubViewOp sv : subviewRewrites) {
      OpBuilder b(sv);
      Location loc = sv.getLoc();

      // Gather mixed offsets (static attrs + dynamic operands).  We need
      // the row offset (index rank-2) and col offset (index rank-1).
      SmallVector<OpFoldResult> offsets = sv.getMixedOffsets();
      if (offsets.size() < 2) continue;
      OpFoldResult rowOfs = offsets[offsets.size() - 2];
      OpFoldResult colOfs = offsets[offsets.size() - 1];

      // Materialize both as Values so we can run arith ops on them.
      Value rowVal = getValueOrCreateConstantIndexOp(b, loc, rowOfs);
      Value colVal = getValueOrCreateConstantIndexOp(b, loc, colOfs);
      Value newColVal = buildSwizzledCol(b, loc, rowVal, colVal, p);

      // Build a fresh SubViewOp with the rewritten col offset.  Keep all
      // other offsets / sizes / strides identical.  Using the mixed
      // convention preserves static entries that weren't the col.
      offsets.back() = OpFoldResult(newColVal);
      SmallVector<OpFoldResult> sizes = sv.getMixedSizes();
      SmallVector<OpFoldResult> strides = sv.getMixedStrides();
      auto newSv = b.create<memref::SubViewOp>(
          loc, cast<MemRefType>(sv.getType()), sv.getSource(),
          offsets, sizes, strides);
      sv.getResult().replaceAllUsesWith(newSv.getResult());
      sv.erase();
    }
    return true;
  }

  // Walk up a use-def chain to see if `v` is derived from `root` via
  // transparent view ops.  Used to confirm a cp.async destination belongs to
  // the alloc we're swizzling.
  static bool isDerivedFrom(Value v, Value root) {
    SmallPtrSet<Value, 8> seen;
    SmallVector<Value> stack = {v};
    while (!stack.empty()) {
      Value x = stack.pop_back_val();
      if (x == root) return true;
      if (!seen.insert(x).second) continue;
      if (Operation *def = x.getDefiningOp()) {
        if (isa<memref::SubViewOp, memref::ReinterpretCastOp,
                memref::CastOp, memref::ExpandShapeOp,
                memref::CollapseShapeOp>(def))
          for (Value o : def->getOperands()) stack.push_back(o);
      }
    }
    return false;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp gpuModule = getOperation();

    // ── Collection path A: static memref.global ──────────────────────────
    // After NovaConvertSharedMemAllocsPass, statically-sized SMEM allocs live
    // as gpu.module-scope memref.global ops; the `nova.swizzle` attribute
    // placed on the pre-conversion alloc is forwarded to the global.
    SmallVector<memref::GlobalOp> wgGlobals;
    for (auto g : gpuModule.getOps<memref::GlobalOp>()) {
      MemRefType t = g.getType();
      if (!isWorkgroupMemref(t)) continue;
      if (t.getRank() < 2) continue;
      if (ShapedType::isDynamic(t.getShape().back())) continue;
      wgGlobals.push_back(g);
    }

    // ── Collection path B: dynamic SMEM views ─────────────────────────────
    // When NovaGPUMapForallToGPU runs `materializeDynamicSharedMemory`, the
    // workgroup allocs are replaced by a single `gpu.dynamic_shared_memory`
    // op plus per-tile `memref.view`s, BEFORE NovaConvertSharedMemAllocs
    // runs. In that case there are no memref.globals at all — instead each
    // SMEM tile is a 2-D `memref.view` over the byte-typed dynamic buffer.
    //
    // We collect those views so they can be treated equivalently to globals.
    // The auto-tag and rewrite logic below operate on the same set of "tile
    // roots" regardless of whether the root is a memref.get_global result or
    // a memref.view result.
    SmallVector<memref::ViewOp> wgViews;
    gpuModule.walk([&](memref::ViewOp v) {
      auto vt = dyn_cast<MemRefType>(v.getType());
      if (!vt || !isWorkgroupMemref(vt)) return;
      if (vt.getRank() < 2) return;
      if (ShapedType::isDynamic(vt.getShape().back())) return;
      // Only views rooted at gpu.dynamic_shared_memory.
      auto src = v.getSource().getDefiningOp<gpu::DynamicSharedMemoryOp>();
      if (!src) return;
      wgViews.push_back(v);
    });

    // Helper: shape-back accessor that works on either tile-root op kind.
    auto innerDim = [](Operation *op) -> int64_t {
      if (auto g = dyn_cast<memref::GlobalOp>(op))
        return g.getType().getShape().back();
      if (auto v = dyn_cast<memref::ViewOp>(op))
        return cast<MemRefType>(v.getType()).getShape().back();
      return 0;
    };

    // Build a uniform list of tile roots (Operation*) for tagging+rewriting.
    SmallVector<Operation *> tileRoots;
    for (auto &g : wgGlobals) tileRoots.push_back(g.getOperation());
    for (auto &v : wgViews)   tileRoots.push_back(v.getOperation());

    // Auto-tag every workgroup tile.  The viability check in computeSwizzle
    // (≥ 4 16B chunks per row) filters out tiles where the swizzle would be
    // degenerate, so wide tiles (e.g. 128×128 / 64×64 = 256 B–512 B per row,
    // 16–32 chunks) get a real 8-way XOR swizzle while narrow B tiles
    // (e.g. 128×8 = 32 B per row, 2 chunks) are skipped with a warning so
    // the user knows to pad them instead.
    //
    // The previous heuristic of tagging only the narrowest tile produced a
    // degenerate 1-bit swizzle on the narrow B tile and left the wide A tile
    // (where the warp-wide ld.shared bank-conflict pattern actually lives)
    // un-swizzled, defeating the whole point.
    for (Operation *r : tileRoots) {
      if (r->hasAttr("nova.swizzle") || r->hasAttr("nova.swizzled"))
        continue;
      r->setAttr("nova.swizzle", UnitAttr::get(&getContext()));
      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-swizzle-shared-memory] auto-tagged tile "
                    "(innermost=" << innerDim(r) << ")\n");
    }

    // Helper: detect whether the access tree rooted at `entry` contains any
    // `vector.transfer_read` / `vector.transfer_write` that reads/writes more
    // than one row at a time, i.e. a vector whose row-dim extent > 1.  Such
    // bulk transfers are NOT safe to swizzle: the XOR rewrites the start
    // index, but the read still spans contiguous rows, and contiguous-row
    // reads after an XOR no longer correspond to the same data the writer
    // produced (writer XORs each row independently).  These tiles must use
    // padding instead.
    //
    // We treat any transfer with a *vector type* whose 2nd-from-last extent
    // is > 1 (or 1st-from-last extent is > 1 for a rank-2 root) as bulk.
    // Element-at-a-time transfers (`vector<1x...x1xT>`) are fine.
    auto hasBulkTransfer = [](Value entry) -> bool {
      SmallVector<Value> stack = {entry};
      SmallPtrSet<Operation *, 32> seen;
      while (!stack.empty()) {
        Value v = stack.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (!seen.insert(user).second) continue;
          if (isa<memref::SubViewOp, memref::ReinterpretCastOp,
                  memref::CastOp>(user)) {
            for (Value r : user->getResults()) stack.push_back(r);
            continue;
          }
          // accessElems = 4 for f32 (128-bit / 4B per element).
          // Wide single-row loads (shape.back() > accessElems) span multiple
          // XOR groups in the same row and cannot be safely swizzled by
          // XOR-ing only the start column — elements beyond the first group
          // land at the wrong swizzled address. Only 4-element-wide (128-bit)
          // loads are group-aligned throughout.
          auto isBulkVec = [](VectorType vt) {
            if (!vt) return false;
            ArrayRef<int64_t> shape = vt.getShape();
            if (shape.empty()) return false;
            // Multi-row transfer.
            for (size_t i = 0; i + 1 < shape.size(); ++i)
              if (shape[i] > 1) return true;
            // Single-row but wider than one 128-bit group (4 f32 or 8 f16).
            // These span multiple XOR groups and cannot be start-col-only swizzled.
            Type elemTy = vt.getElementType();
            int64_t elemBits = elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 32;
            int64_t groupElems = 128 / elemBits; // 128-bit group size in elements
            if (shape.back() > groupElems) return true;
            return false;
          };
          if (auto tr = dyn_cast<vector::TransferReadOp>(user))
            if (isBulkVec(tr.getVectorType())) return true;
          if (auto tw = dyn_cast<vector::TransferWriteOp>(user))
            if (isBulkVec(tw.getVectorType())) return true;
        }
      }
      return false;
    };

    // Process each tagged tile root.
    for (Operation *root : tileRoots) {
      if (!root->hasAttr("nova.swizzle")) continue;
      MemRefType t;
      if (auto g = dyn_cast<memref::GlobalOp>(root)) t = g.getType();
      else t = cast<MemRefType>(cast<memref::ViewOp>(root).getType());

      int64_t elemBytes = t.getElementType().getIntOrFloatBitWidth() / 8;
      if (elemBytes == 0) continue;
      SwizzleParams p = computeSwizzle(t.getShape().back(), elemBytes);
      if (!p.viable) {
        // llvm::errs() << "[nova-swizzle-shared-memory] tile too narrow for "
        //                 "useful swizzle (innermost="
        //              << t.getShape().back() << " elemBytes=" << elemBytes
        //              << ") — consider padding instead\n";
        continue;
      }

      // Skip tiles whose downstream loads/stores are bulk vector transfers.
      // The XOR can only correctly permute element-at-a-time accesses; bulk
      // transfers would silently read/write contiguous spans that cross
      // swizzled and un-swizzled cells.  These tiles need padding to resolve
      // bank conflicts, not swizzle.
      Value entry;
      if (auto g = dyn_cast<memref::GlobalOp>(root)) {
        // Use the first get_global as the entry; all share the same use tree.
        bool found = false;
        gpuModule.walk([&](memref::GetGlobalOp gg) {
          if (found) return;
          if (gg.getName() == g.getSymName()) {
            entry = gg.getResult();
            found = true;
          }
        });
        if (!entry) continue;
      } else {
        entry = cast<memref::ViewOp>(root).getResult();
      }
      if (hasBulkTransfer(entry)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-swizzle-shared-memory] tile has bulk vector "
                      "transfers — skipping swizzle (use padding)\n");
        continue;
      }

      bool anyRewritten = false;
      bool anyFailed = false;

      if (auto g = dyn_cast<memref::GlobalOp>(root)) {
        // Find every memref.get_global referencing this symbol; rewrite the
        // access tree rooted at each.
        StringRef symName = g.getSymName();
        gpuModule.walk([&](memref::GetGlobalOp getG) {
          if (getG.getName() != symName) return;
          if (rewriteAccessesOf(getG.getResult(), p))
            anyRewritten = true;
          else
            anyFailed = true;
        });
      } else {
        // Dynamic SMEM view: the view itself is the unique entry point — its
        // result feeds every downstream subview / load / store of the tile.
        auto v = cast<memref::ViewOp>(root);
        if (rewriteAccessesOf(v.getResult(), p))
          anyRewritten = true;
        else
          anyFailed = true;
      }

      if (anyFailed) {
        llvm::errs() << "[nova-swizzle-shared-memory] unknown access pattern "
                        "— partial rewrite avoided\n";
        continue;
      }
      if (!anyRewritten) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-swizzle-shared-memory] no accesses found for "
                      "tile\n");
        continue;
      }
      root->setAttr("nova.swizzled", UnitAttr::get(&getContext()));
      root->removeAttr("nova.swizzle");
    }
  }
};

std::unique_ptr<Pass> createNovaGPUSwizzleSharedMemoryPass() {
  return std::make_unique<NovaGPUSwizzleSharedMemoryPass>();
}

void registerNovaGPUSwizzleSharedMemoryPass() {
  PassRegistration<NovaGPUSwizzleSharedMemoryPass>();
}

} // namespace mlir::nova