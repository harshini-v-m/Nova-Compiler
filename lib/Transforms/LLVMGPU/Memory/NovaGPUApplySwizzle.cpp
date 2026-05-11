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
//       mask   = (1 << rowBits) - 1
//       rowBits= min( log2(C * elemBytes / 16),    -- can't XOR past row end
//                     log2(min(warpSize,numBanks)))-- cap at hardware lanes
//   The shift keeps the XOR above the 16-byte boundary so cp.async / ldmatrix
//   vector contiguity is preserved.  The mask grows with the tile up to a
//   single hard cap of 32 unique permutations on every NVIDIA SM (W = B = 32),
//   which is the most a warp can use in one transaction — matching IREE's
//   target-parameterised swizzle policy rather than baking in special cases
//   per element type or tile shape.
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
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
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
#include "llvm/ADT/SmallBitVector.h"
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
 // XOR = ((row & rowMask) << shiftBits)
 int64_t rowMask;    // bit mask on the row index (outer-most of the 2D view)
 int64_t shiftBits;  // left-shift applied to the masked row bits before XOR
 bool    viable;     // false if the tile is too narrow for any useful swizzle
};


// Derive the XOR for a 2D (or higher-rank) SMEM tile whose innermost dim has
// `innerDim` elements of size `elemBytes`, given the architecture's warp size
// and shared-memory bank count.  Architecture-parameterised: caller queries
// the target description and passes the values in.  No element-type or tile-
// shape special cases live here.
//
// Generic principle (mirrors IREE's LLVMGPU swizzle policy):
//
//   1. Vector chunk preservation.
//      Shared-memory transactions are issued at a fixed granularity:
//      cp.async, ldmatrix and stmatrix all move 16-byte aligned vectors per
//      lane.  The XOR must NOT permute bits below log2(chunkBytes); otherwise
//      it would split a 16-byte lane chunk across non-contiguous addresses
//      and break those instructions' contract.
//      → shiftBits = log2(chunkBytes / elemBytes)
//
//   2. Conflict-free warp coverage.
//      A bank conflict happens only inside a single warp's transaction.
//      With B = numBanks and W = warpSize (32 on every NVIDIA SM since
//      Volta), the most permutations the swizzle ever needs is W: any
//      additional rotation is wasted because no warp can address more than
//      W banks in one cycle.  At the same time, the row-mask bitwidth is
//      bounded above by log2(chunksPerRow) — XOR-ing past that walks past
//      the row.
//      → rowBits = min( log2(chunksPerRow), log2(min(W, B)) )
//
//   3. Viability lower bound.
//      A swizzle with ≤ 1 row-bit only flips between two address sets —
//      not enough to break the warp-wide stride that creates the conflict
//      in the first place.  Tiles below this threshold opt out and the
//      caller falls through to padding.
//      → require chunksPerRow ≥ 4  (rowBits ≥ 2)
//
// Worked examples (warpSize = 32, numBanks = 32 → cap at 5 row-bits):
//   - f32 16-col (64 B/row, 4 chunks):  rowBits=2, shift=2  (4 variants)
//   - f32 32-col (128 B/row, 8 chunks): rowBits=3, shift=2  (8 variants)
//   - f32 64-col (256 B/row, 16 chunks):rowBits=4, shift=2  (16 variants)
//   - f32 128-col(512 B/row, 32 chunks):rowBits=5, shift=2  (32 variants
//                                       = full warp — 0 conflicts)
//   - f16 64-col (128 B/row, 8 chunks): rowBits=3, shift=3  (8 variants)
//   - f16 128-col(256 B/row, 16 chunks):rowBits=4, shift=3  (16 variants)
//   - bf16/fp8 follow elemBytes mechanically.
//   - Tile too narrow (rowBytes < 64): viable = false, padding handles it.
static SwizzleParams computeSwizzle(int64_t innerDim, int64_t elemBytes,
                                   int64_t warpSize, int64_t numBanks) {
 SwizzleParams p{0, 0, false};
 if (elemBytes <= 0 || elemBytes > 16) return p;      // exotic: skip
 int64_t rowBytes = innerDim * elemBytes;
 // ldmatrix / cp.async unit: 16-byte aligned vectors per lane.  This is
 // architecture-fixed for every NVIDIA SM that supports these ops.
 constexpr int64_t chunkBytes = 16;
 if (chunkBytes < elemBytes) return p;                // would need shift < 0
 int64_t chunksPerRow = rowBytes / chunkBytes;        // floor
 if (chunksPerRow < 4) return p;                      // <2 useful XOR bits
 int64_t chunksLog2 = floorLog2PowerOf2(chunksPerRow);
 // Conflict-free coverage cap: a warp can occupy at most min(W, B) banks
 // per transaction, so more permutations than that are wasted.  On every
 // current NVIDIA SM W = B = 32, hence the natural 5-bit cap.
 int64_t hwLanes = std::min<int64_t>(warpSize, numBanks);
 if (hwLanes < 1) hwLanes = 32;                       // safety
 int64_t hwLog2 = floorLog2PowerOf2(hwLanes);
 if (chunksLog2 > hwLog2) chunksLog2 = hwLog2;
 p.rowMask  = (1LL << chunksLog2) - 1;
 p.shiftBits = floorLog2PowerOf2(chunkBytes / elemBytes); // log2(16/elemBytes)
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
                        OperationPass<func::FuncOp>> {
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUSwizzleSharedMemoryPass)


 std::string archName;


 NovaGPUSwizzleSharedMemoryPass() = default;
 explicit NovaGPUSwizzleSharedMemoryPass(StringRef arch)
     : archName(arch.str()) {}
 NovaGPUSwizzleSharedMemoryPass(const NovaGPUSwizzleSharedMemoryPass &p)
     : PassWrapper(p), archName(p.archName) {}


 StringRef getArgument() const override {
   return "nova-gpu-swizzle-shared-memory";
 }
 StringRef getDescription() const override {
   return "Per workgroup memref.alloc, decide between XOR-swizzling its "
          "loads/stores (zero SMEM cost; matmul-tile case) and padding the "
          "innermost dim by 16 bytes (fallback for narrow tiles or bulk "
          "vector transfers).  Replaces both NovaGPUReduceBankConflicts and "
          "the previous opt-in-only swizzle pass.";
 }


 void getDependentDialects(DialectRegistry &registry) const override {
   registry.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                   memref::MemRefDialect, nvgpu::NVGPUDialect,
                   vector::VectorDialect>();
 }


 // Rewrite `originalCol = (originalCol) XOR ((row & rowMask) << shift)` using
 // index-typed arith.  Returns the new column value.
 static Value buildSwizzledCol(OpBuilder &b, Location loc, Value row,
                               Value col, SwizzleParams p) {
   Value maskC  = b.create<arith::ConstantIndexOp>(loc, p.rowMask);
   Value shiftC = b.create<arith::ConstantIndexOp>(loc, p.shiftBits);
   Value masked = b.create<arith::AndIOp>(loc, row, maskC);
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
       // nvgpu.ldmatrix reads a 16x16 (or smaller) fragment from SMEM into
       // registers; the address formula is independent of the `transpose`
       // attribute (transpose only affects how the loaded fragment is laid
       // out into the result vector, not which SMEM bytes are read), so the
       // single rewrite case below covers both ldmatrix and ldmatrix.trans.
       // ODS operand layout: group 0 = srcMemref (single), group 1 = indices
       // (variadic).
       if (auto lm = dyn_cast<nvgpu::LdMatrixOp>(user)) {
         auto indices = lm.getIndices();
         if (indices.size() < 2) continue;
         recordAccess(user, indices[indices.size() - 2], indices.back(),
                      (unsigned)(lm.getODSOperandIndexAndLength(1).first +
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


 // Returns true if any vector.transfer_read/write rooted at `entry` reads or
 // writes more than one row at a time.  XOR-swizzling such a transfer would
 // silently read/write contiguous spans that cross swizzled cells; these
 // tiles must fall back to padding.  Element-at-a-time transfers
 // (vector<1x...x1xT>), scalar memref.load/store, ldmatrix and cp.async are
 // all fine and return false.
 static bool hasBulkTransferDownstream(Value entry) {
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
       auto isBulkVec = [](VectorType vt) {
         if (!vt) return false;
         ArrayRef<int64_t> shape = vt.getShape();
         if (shape.empty()) return false;
         // Any non-trailing dim > 1 means the load covers multiple rows.
         for (size_t i = 0; i + 1 < shape.size(); ++i)
           if (shape[i] > 1) return true;
         return false;
       };
       if (auto tr = dyn_cast<vector::TransferReadOp>(user))
         if (isBulkVec(tr.getVectorType())) return true;
       if (auto tw = dyn_cast<vector::TransferWriteOp>(user))
         if (isBulkVec(tw.getVectorType())) return true;
     }
   }
   return false;
 }


 // Padding fallback: extend the innermost dim of `alloc` by one ldmatrix
 // chunk (16 B) worth of elements and replace its uses with a strided
 // subview presenting the original shape.  Used when XOR is not viable
 // (rowBytes < 64) or the access pattern is incompatible (bulk vector
 // transfers, unknown ops).
 //
 // Generic policy (mirrors the swizzle path's IREE-style parameterisation):
 //   - bankPeriod  = numBanks * bankStride   bytes  (32 × 4 = 128 on NVIDIA)
 //   - halfPeriod  = bankPeriod / 2                  (= 64 on NVIDIA)
 //   - chunkBytes  = 16                              (cp.async / ldmatrix unit)
 //
 // Trigger: pad when the row stride lands on a bank-period boundary.  We
 // catch both the 32-way collision (stride % bankPeriod == 0) AND the
 // 16-way collision (stride % halfPeriod == 0) — the latter still doubles
 // up lanes onto half the banks per row and is worth fixing wherever it
 // costs less than the SRAM budget.  Strides that are not period-aligned
 // already step through banks naturally and don't need padding.
 //
 // Padding amount: one chunkBytes (= one ldmatrix vector).  This is the
 // smallest aligned step that always breaks the period (since chunkBytes <
 // halfPeriod ≤ bankPeriod and the stride was period-aligned, the new
 // stride is necessarily not period-aligned and stays chunk-aligned, so
 // ldmatrix and cp.async vector loads remain valid).
 static bool applyPaddingFallback(memref::AllocOp alloc, OpBuilder &builder,
                                  int64_t maxSRAM, int64_t warpSize,
                                  int64_t numBanks) {
   MLIRContext *ctx = alloc.getContext();
   auto memrefType = alloc.getType();
   ArrayRef<int64_t> shape = memrefType.getShape();
   Type elemType = memrefType.getElementType();


   unsigned elemBitsCheck = elemType.getIntOrFloatBitWidth();
   if (elemBitsCheck == 0 || (elemBitsCheck % 8) != 0) return false;
   int64_t elemBytesCheck = elemBitsCheck / 8;
   int64_t rowBytes = shape.back() * elemBytesCheck;


   // Hardware bank topology.  bankStride is invariant on NVIDIA (4 bytes
   // per bank since Volta) and is not exposed in the target struct yet —
   // pulling it out as a named constant keeps the formula readable for a
   // future port to architectures with different layouts.
   constexpr int64_t bankStride = 4;
   constexpr int64_t chunkBytes = 16;
   int64_t hwLanes = std::min<int64_t>(warpSize, numBanks);
   if (hwLanes < 1) hwLanes = 32;
   int64_t bankPeriod = hwLanes * bankStride;          // 128 on NVIDIA
   int64_t halfPeriod = bankPeriod / 2;                // 64  on NVIDIA


   // halfPeriod divides bankPeriod, so a stride aligned to bankPeriod is
   // automatically aligned to halfPeriod — checking the half period covers
   // both the 32-way (full period) and 16-way (half period) collision cases.
   if (halfPeriod <= 0 || (rowBytes % halfPeriod) != 0) {
     LLVM_DEBUG(llvm::dbgs()
                << "[nova-swizzle-shared-memory] padding skipped: row stride "
                << rowBytes << " B not aligned to bank period "
                << bankPeriod << " B (or half " << halfPeriod << " B)\n");
     return false;
   }


   int64_t paddingElems =
       std::max<int64_t>(1, chunkBytes / static_cast<int64_t>(elemBytesCheck));
   int64_t newInnerDim = shape.back() + paddingElems;
   SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
   paddedShape.back() = newInnerDim;


   int64_t paddedBytes = 1;
   for (int64_t d : paddedShape) paddedBytes *= d;
   paddedBytes *= elemBytesCheck;
   if (paddedBytes > maxSRAM) {
     LLVM_DEBUG(llvm::dbgs()
                << "[nova-swizzle-shared-memory] padding skipped: padded "
                << paddedBytes << " B exceeds SRAM budget " << maxSRAM
                << " B\n");
     return false;
   }


   auto workgroupSpace = gpu::AddressSpaceAttr::get(
       ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
   auto paddedMemrefType =
       MemRefType::get(paddedShape, elemType,
                       /*layout=*/MemRefLayoutAttrInterface{}, workgroupSpace);


   builder.setInsertionPoint(alloc);
   Location loc = alloc.getLoc();
   auto paddedAlloc = memref::AllocOp::create(
       builder, loc, paddedMemrefType, alloc.getDynamicSizes(),
       alloc.getAlignmentAttr());


   SmallVector<OpFoldResult> offsets(shape.size(), builder.getIndexAttr(0));
   SmallVector<OpFoldResult> sizes;
   for (int64_t d : shape) sizes.push_back(builder.getIndexAttr(d));
   SmallVector<OpFoldResult> strides(shape.size(), builder.getIndexAttr(1));
   auto subview = memref::SubViewOp::create(
       builder, loc, paddedAlloc.getResult(), offsets, sizes, strides);


   // Re-point dealloc ops at the padded base (deallocing a subview is UB).
   SmallVector<memref::DeallocOp> deallocsToFix;
   for (Operation *user :
        llvm::make_early_inc_range(alloc.getResult().getUsers())) {
     if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
       deallocsToFix.push_back(dealloc);
   }
   for (memref::DeallocOp dealloc : deallocsToFix)
     dealloc.getMemrefMutable().assign(paddedAlloc.getResult());


   alloc.getResult().replaceAllUsesWith(subview.getResult());
   alloc.erase();


   // Propagate the new stride through any downstream chained subviews.
   SmallVector<Value> worklist = {subview.getResult()};
   SmallPtrSet<Operation *, 16> visited;
   while (!worklist.empty()) {
     Value v = worklist.pop_back_val();
     for (Operation *user : v.getUsers()) {
       if (!visited.insert(user).second) continue;
       auto sv = dyn_cast<memref::SubViewOp>(user);
       if (!sv) continue;
       auto srcType = cast<MemRefType>(sv.getSource().getType());
       auto srcLayout = dyn_cast<StridedLayoutAttr>(srcType.getLayout());
       if (!srcLayout) continue;
       ArrayRef<int64_t> srcStrides = srcLayout.getStrides();
       SmallVector<int64_t> svStaticStrides =
           llvm::to_vector(sv.getStaticStrides());
       SmallVector<int64_t> svStaticOffsets =
           llvm::to_vector(sv.getStaticOffsets());
       SmallVector<int64_t> allStrides;
       allStrides.reserve(srcStrides.size());
       for (size_t i = 0; i < srcStrides.size(); ++i) {
         if (ShapedType::isDynamic(srcStrides[i]) ||
             ShapedType::isDynamic(svStaticStrides[i]))
           allStrides.push_back(ShapedType::kDynamic);
         else
           allStrides.push_back(srcStrides[i] * svStaticStrides[i]);
       }
       int64_t newOffset = srcLayout.getOffset();
       for (size_t i = 0; i < srcStrides.size(); ++i) {
         if (ShapedType::isDynamic(newOffset) ||
             ShapedType::isDynamic(svStaticOffsets[i]) ||
             ShapedType::isDynamic(srcStrides[i])) {
           newOffset = ShapedType::kDynamic;
           break;
         }
         newOffset += svStaticOffsets[i] * srcStrides[i];
       }
       llvm::SmallBitVector droppedDims = sv.getDroppedDims();
       SmallVector<int64_t> newStrides;
       newStrides.reserve(allStrides.size() - droppedDims.count());
       for (size_t i = 0; i < allStrides.size(); ++i) {
         if (!droppedDims.test(i)) newStrides.push_back(allStrides[i]);
       }
       auto oldResult = sv.getType();
       if (static_cast<int64_t>(newStrides.size()) != oldResult.getRank())
         continue;
       auto newResultType = MemRefType::get(
           oldResult.getShape(), oldResult.getElementType(),
           StridedLayoutAttr::get(ctx, newOffset, newStrides),
           oldResult.getMemorySpace());
       sv.getResult().setType(newResultType);
       worklist.push_back(sv.getResult());
     }
   }
   return true;
 }


 void runOnOperation() override {
   func::FuncOp funcOp = getOperation();
   MLIRContext *ctx = funcOp.getContext();


   // Resolve the SRAM limit so the padding-fallback pre-flight is target
   // accurate.  Fall back to a 48 KB Ampere default when arch is unknown.
   // Also resolve warpSize / numBanks for the swizzle: every NVIDIA SM since
   // Volta uses W = B = 32, but plumbing them through keeps the swizzle
   // policy honest for future targets (and avoids hard-coded magic numbers
   // in computeSwizzle).
   int64_t maxSRAM   = 48 * 1024;
   int64_t warpSize  = 32;
   int64_t numBanks  = 32;
   if (!archName.empty()) {
     NVIDIATargetInfo info = getNVIDIATargetInfo(archName);
     if (info.isValid()) {
       maxSRAM  = static_cast<int64_t>(info.maxWorkgroupMemBytes);
       warpSize = static_cast<int64_t>(info.preferredSubgroupSize);
       // numBanks is invariant on NVIDIA (32 banks, 4-byte stride) — not
       // exposed in the target struct yet; keep the architectural default.
     }
   }


   auto workgroupSpace = gpu::AddressSpaceAttr::get(
       ctx, gpu::GPUDialect::getWorkgroupAddressSpace());


   // Collect all workgroup memref.allocs up front: both rewrite paths (XOR
   // and padding) rewire users of the alloc and would invalidate a live
   // walk.
   SmallVector<memref::AllocOp> wgAllocs;
   funcOp.walk([&](memref::AllocOp alloc) {
     auto t = alloc.getType();
     if (t.getMemorySpace() != workgroupSpace) return;
     if (t.getRank() < 2) return;
     if (ShapedType::isDynamic(t.getShape().back())) return;
     wgAllocs.push_back(alloc);
   });


   if (wgAllocs.empty()) return;


   OpBuilder builder(ctx);
   for (memref::AllocOp alloc : wgAllocs) {
     MemRefType t = alloc.getType();
     Type elemType = t.getElementType();
     unsigned elemBits = elemType.getIntOrFloatBitWidth();
     if (elemBits == 0 || (elemBits % 8) != 0) continue;
     int64_t elemBytes = elemBits / 8;
     int64_t innerDim = t.getShape().back();


     SwizzleParams p = computeSwizzle(innerDim, elemBytes,
                                      warpSize, numBanks);
     bool bulk = hasBulkTransferDownstream(alloc.getResult());


     // Decision: XOR-swizzle if the tile is wide enough AND no bulk vector
     // transfers exist on the def-use tree; otherwise fall through to
     // padding.  rewriteAccessesOf returns false WITHOUT mutating IR if it
     // hits an unknown user, so the fall-through is safe.
     bool swizzled = false;
     if (p.viable && !bulk) {
       if (rewriteAccessesOf(alloc.getResult(), p)) {
         alloc->setAttr("nova.swizzled", UnitAttr::get(ctx));
         swizzled = true;
         LLVM_DEBUG(llvm::dbgs()
                    << "[nova-swizzle-shared-memory] swizzled alloc "
                       "(innerDim=" << innerDim
                    << " elemBytes=" << elemBytes << ")\n");
       } else {
         LLVM_DEBUG(llvm::dbgs()
                    << "[nova-swizzle-shared-memory] XOR rewrite failed; "
                       "falling back to padding\n");
       }
     }


     if (swizzled) continue;


     // Padding fallback.  Skips silently if the row stride isn't aligned
     // to a bank period (full or half) or the padded footprint exceeds
     // maxSRAM.  Tiles that don't fit either path stay in their natural
     // layout — they don't have a warp-wide bank conflict to fix.
     applyPaddingFallback(alloc, builder, maxSRAM, warpSize, numBanks);
   }
 }
};


std::unique_ptr<Pass> createNovaGPUSwizzleSharedMemoryPass(StringRef arch) {
 return std::make_unique<NovaGPUSwizzleSharedMemoryPass>(arch);
}


void registerNovaGPUSwizzleSharedMemoryPass() {
 PassRegistration<NovaGPUSwizzleSharedMemoryPass>();
}


} // namespace mlir::nova

