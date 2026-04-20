//===- NovaGPUApplySwizzle.cpp - XOR swizzle for shared memory -----------===//
//
// Post-bufferization pass. Runs after ConvertLinalgToLoops (Step 10) so that
// all linalg.copy / linalg.generic ops have been lowered to
// scf.for + memref.load/store.
//
// Goal
// ────
// Eliminate shared memory bank conflicts on workgroup-address-space allocs by
// applying XOR swizzle to the load/store column indices.
//
// Background
// ──────────
// NVIDIA shared memory has 32 banks, each 4 bytes wide. A row-major tile
// wider than 128 bytes repeats bank assignments across rows: column c in row r
// and column c in row r+32 hit the same bank. When consecutive threads read
// the same column across rows they serialize. XOR swizzle breaks this by
// rotating the column assignment per row.
//
// Why the swizzle is applied at the subview level
// ───────────────────────────────────────────────
// After ConvertLinalgToLoops every access to a workgroup alloc goes through
// a memref.subview whose row/col base offsets are runtime loop IVs. Those
// offsets are folded into the subview's strided layout and are invisible at
// the downstream load/store — the load/store only sees subview-local indices.
//
// The XOR phase must use the ABSOLUTE row in the alloc:
//
//   absRow = subviewRowBase + localRow
//   absCol = subviewColBase + localCol
//   swizzledAbsCol = swizzle(absCol, absRow)
//   newLocalCol = swizzledAbsCol - subviewColBase
//              = swizzle(absCol, absRow) - subviewColBase
//
// Both subviewRowBase and subviewColBase are available as operands of the
// SubViewOp, so the pass walks SubViewOps (not load/store ops directly).
//
// For each SubViewOp whose source is a workgroup alloc the pass:
//   1. Extracts rowBase and colBase from the subview's offset operands.
//   2. For every load/store on that subview, computes:
//        absRow      = rowBase + localRow      (insert arith.addi if rowBase≠0)
//        absCol      = colBase + localCol      (insert arith.addi if colBase≠0)
//        swizzledCol = XOR-swizzle(absCol, absRow)
//        newLocalCol = swizzledCol - colBase   (insert arith.subi)
//      then replaces localCol with newLocalCol.
//
// Static-zero bases (e.g. row=0 or col=0 in the subview) are handled without
// emitting arithmetic — the add/sub are folded out.
//
// Conflict-free early-out
// ───────────────────────
// If innerDim * elemBytes <= 128, each row occupies at most one full 32-bank
// cycle and no conflicts are possible. The alloc is skipped entirely.
//
// Swizzle formula (bank-group-aware, for the absolute col):
//
//   elemsPerBank  = max(1, 4 / elemBytes)
//   bankGroup     = absCol / elemsPerBank
//   phase         = absRow AND (maxPhase - 1)   (maxPhase is power-of-2)
//   swizzledBG    = bankGroup XOR phase
//   swizzledCol   = swizzledBG * elemsPerBank + (absCol % elemsPerBank)
//
// For f32 (elemsPerBank==1):  swizzledCol = absCol XOR phase
// For f16 (elemsPerBank==2):  full bank-group path.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::nova;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool hasWorkgroupAddressSpace(MemRefType type) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Returns {maxPhase, elemsPerBank}. Returns {0,1} if swizzle is not needed.
/// Caller must have already confirmed innerDim * elemBytes > 128.
static std::pair<int64_t, int64_t>
computeSwizzleParams(int64_t innerDim, int64_t elemBytes) {
  int64_t elemsPerBank = std::max<int64_t>(1, 4 / elemBytes);
  int64_t bankGroups   = innerDim / elemsPerBank;
  if (bankGroups <= 1)
    return {0, 1};
  int64_t capped   = std::min(bankGroups, (int64_t)32);
  int64_t maxPhase = (int64_t)llvm::NextPowerOf2(capped - 1);
  if (maxPhase > capped) maxPhase >>= 1;
  if (maxPhase < 2)      return {0, 1};
  return {maxPhase, elemsPerBank};
}

/// Add a compile-time constant to a Value, folding the add when c == 0.
static Value addConst(OpBuilder &b, Location loc, Value v, int64_t c) {
  if (c == 0) return v;
  Value cv = b.create<arith::ConstantIndexOp>(loc, c);
  return b.create<arith::AddIOp>(loc, cv, v);
}

/// Subtract a compile-time constant from a Value, folding when c == 0.
static Value subConst(OpBuilder &b, Location loc, Value v, int64_t c) {
  if (c == 0) return v;
  Value cv = b.create<arith::ConstantIndexOp>(loc, c);
  return b.create<arith::SubIOp>(loc, v, cv);
}

/// Build swizzledAbsCol from absCol and absRow.
///
///   phase        = absRow AND (maxPhase - 1)
///   For elemsPerBank == 1:
///     swizzledCol = absCol XOR phase
///   For elemsPerBank > 1:
///     bg          = absCol / elemsPerBank
///     swizzledCol = (bg XOR phase) * elemsPerBank + absCol % elemsPerBank
static Value buildSwizzledAbsCol(OpBuilder &b, Location loc,
                                 Value absRow, Value absCol,
                                 int64_t maxPhase, int64_t elemsPerBank) {
  Value mask  = b.create<arith::ConstantIndexOp>(loc, maxPhase - 1);
  Value phase = b.create<arith::AndIOp>(loc, absRow, mask);

  if (elemsPerBank == 1)
    return b.create<arith::XOrIOp>(loc, absCol, phase);

  Value epb    = b.create<arith::ConstantIndexOp>(loc, elemsPerBank);
  Value bg     = b.create<arith::DivUIOp>(loc, absCol, epb);
  Value rem    = b.create<arith::RemUIOp>(loc, absCol, epb);
  Value bgSwiz = b.create<arith::XOrIOp>(loc, bg, phase);
  Value scaled = b.create<arith::MulIOp>(loc, bgSwiz, epb);
  return b.create<arith::AddIOp>(loc, scaled, rem);
}

/// Holds the resolved base offset for one dimension of a subview.
/// Either a compile-time constant (dynVal == nullptr) or a runtime value.
struct DimBase {
  int64_t staticVal = 0;
  Value   dynVal;   // null when the offset is a compile-time constant

  bool isZero() const { return !dynVal && staticVal == 0; }

  Value toValue(OpBuilder &b, Location loc) const {
    if (dynVal) return dynVal;
    return b.create<arith::ConstantIndexOp>(loc, staticVal);
  }
};

/// Extract the row and col base offsets for the innermost two dimensions of
/// a SubViewOp using getMixedOffsets (the standard ViewLikeInterface API).
/// Each OpFoldResult is either an IntegerAttr (static) or a Value (dynamic).
static std::pair<DimBase, DimBase>
extractRowColBase(memref::SubViewOp sv, int64_t srcRank) {
  SmallVector<OpFoldResult> mixed = sv.getMixedOffsets();

  auto toBase = [&](OpFoldResult ofr) -> DimBase {
    DimBase base;
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      base.staticVal = cast<IntegerAttr>(attr).getInt();
      base.dynVal    = nullptr;
    } else {
      base.dynVal    = cast<Value>(ofr);
      base.staticVal = 0;
    }
    return base;
  };

  DimBase rowBase = toBase(mixed[srcRank - 2]);
  DimBase colBase = toBase(mixed[srcRank - 1]);
  return {rowBase, colBase};
}

/// Rewrite the column index of one scalar load/store op.
///
/// The load/store uses subview-local indices. We reconstruct:
///   absRow      = rowBase + localRow
///   absCol      = colBase + localCol
///   swizzledCol = XOR-swizzle(absCol, absRow)
///   newLocalCol = swizzledCol - colBase
///
/// then replace localCol with newLocalCol in the op's index list.
///
/// NOTE: vector.transfer_read/write ops are intentionally NOT rewritten here.
/// Swizzle scatters each element to a different (XOR'd) address. A wide
/// vector transfer reads/writes N *contiguous* elements from a single
/// starting index — offsetting that index by phase gives the wrong layout
/// (elements are no longer contiguous after swizzle). Only scalar
/// memref.load/store accesses individual elements and can be correctly
/// swizzled on a per-element basis.
static void rewriteSubviewAccess(Operation *op,
                                 const DimBase &rowBase,
                                 const DimBase &colBase,
                                 int64_t maxPhase, int64_t elemsPerBank) {
  OpBuilder b(op);
  Location loc = op->getLoc();

  // Only rewrite scalar load/store ops. Wide vector transfers load/store
  // contiguous ranges and cannot be corrected by a single start-index XOR.
  SmallVector<Value> indices;
  if (auto ld = dyn_cast<memref::LoadOp>(op))
    indices.assign(ld.getIndices().begin(), ld.getIndices().end());
  else if (auto st = dyn_cast<memref::StoreOp>(op))
    indices.assign(st.getIndices().begin(), st.getIndices().end());
  else
    return; // skip vector.transfer_read/write

  int64_t rank = (int64_t)indices.size();
  if (rank < 2) return;

  Value localRow = indices[rank - 2];
  Value localCol = indices[rank - 1];

  // absRow = rowBase + localRow
  Value absRow;
  if (rowBase.isZero()) {
    absRow = localRow;
  } else {
    Value rb = rowBase.toValue(b, loc);
    absRow   = b.create<arith::AddIOp>(loc, rb, localRow);
  }

  // absCol = colBase + localCol
  Value absCol;
  if (colBase.isZero()) {
    absCol = localCol;
  } else {
    Value cb = colBase.toValue(b, loc);
    absCol   = b.create<arith::AddIOp>(loc, cb, localCol);
  }

  // swizzledAbsCol = XOR-swizzle(absCol, absRow)
  Value swizzledAbsCol =
      buildSwizzledAbsCol(b, loc, absRow, absCol, maxPhase, elemsPerBank);

  // newLocalCol = swizzledAbsCol - colBase
  // (subtract the same colBase that was added, keeping the result in the
  //  subview's local coordinate space so the strided-layout offset is correct)
  Value newLocalCol;
  if (colBase.isZero()) {
    newLocalCol = swizzledAbsCol;
  } else {
    Value cb    = colBase.toValue(b, loc);
    newLocalCol = b.create<arith::SubIOp>(loc, swizzledAbsCol, cb);
  }

  indices[rank - 1] = newLocalCol;

  if (auto ld = dyn_cast<memref::LoadOp>(op))
    ld.getIndicesMutable().assign(indices);
  else if (auto st = dyn_cast<memref::StoreOp>(op))
    st.getIndicesMutable().assign(indices);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUApplySwizzlePass
    : public PassWrapper<NovaGPUApplySwizzlePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUApplySwizzlePass)

  StringRef getArgument() const override { return "nova-gpu-apply-swizzle"; }
  StringRef getDescription() const override {
    return "Apply XOR swizzle to shared memory accesses to reduce bank conflicts";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // -----------------------------------------------------------------------
    // Phase 1: collect workgroup allocs that need swizzling.
    //
    // Key insight: ReduceBankConflicts pads the alloc's innermost dim (e.g.
    // 64→68 for f32), so the raw alloc shape is no longer a power-of-2.  The
    // swizzle parameters must be derived from the ORIGINAL (unpadded) inner
    // dimension, which is the size presented by the first SubViewOp that
    // slices the padded alloc back to the unpadded shape.
    //
    // Strategy:
    //   • Walk all workgroup AllocOps.
    //   • For each one, find its first SubViewOp user that presents a
    //     rank-preserving, power-of-2 inner dim — that is the "logical" tile.
    //   • Compute swizzle params from that logical inner dim.
    //   • Map the alloc → params so the SubView walk below can look it up.
    // -----------------------------------------------------------------------
    llvm::DenseMap<Operation *, std::pair<int64_t, int64_t>> swizzleMap;

    funcOp.walk([&](memref::AllocOp alloc) {
      auto memType = alloc.getType();
      if (!hasWorkgroupAddressSpace(memType)) return;
      if (memType.getRank() < 2) return;

      int64_t elemBytes =
          memType.getElementType().getIntOrFloatBitWidth() / 8;

      // Find the first SubViewOp on this alloc to get the logical inner dim.
      // After ReduceBankConflicts the alloc is padded; the subview strips the
      // padding back to the original shape.
      int64_t logicalInnerDim = ShapedType::kDynamic;
      for (Operation *user : alloc->getUsers()) {
        auto sv = dyn_cast<memref::SubViewOp>(user);
        if (!sv) continue;
        auto svType = sv.getType();
        if (svType.getRank() < 2) continue;
        int64_t d = svType.getShape().back();
        if (ShapedType::isDynamic(d) || d <= 0) continue;
        if (!llvm::isPowerOf2_64(d)) continue;
        logicalInnerDim = d;
        break;
      }

      // If no subview found, fall back to the alloc's own inner dim (no
      // padding was applied or it was already power-of-2).
      if (logicalInnerDim == ShapedType::kDynamic) {
        int64_t d = memType.getShape().back();
        if (ShapedType::isDynamic(d) || d <= 0 ||
            !llvm::isPowerOf2_64(d)) return;
        logicalInnerDim = d;
      }

      // Conflict-free: entire logical row fits in one 128-byte cache line.
      if (logicalInnerDim * elemBytes <= 128) return;

      auto [maxPhase, elemsPerBank] =
          computeSwizzleParams(logicalInnerDim, elemBytes);
      if (maxPhase < 2) return;

      // ---- Guard: skip allocs that have ANY vector.transfer_read/write user ----
      // When vector.transfer_read compiles to ldmatrix, it reads contiguous
      // physical addresses — it does NOT apply hardware XOR swizzle unless a
      // swizzle descriptor (TMA) is used.  If we XOR-swizzle the scalar stores
      // but leave the vector reads un-swizzled, reads get wrong elements.
      //
      // Walk all subviews of this alloc transitively and bail out if any user
      // is a vector.transfer_read or vector.transfer_write.
      bool hasVectorUser = false;
      // BFS over the subview tree rooted at this alloc.
      SmallVector<Operation *> worklist(alloc->getUsers().begin(),
                                        alloc->getUsers().end());
      while (!worklist.empty() && !hasVectorUser) {
        Operation *cur = worklist.pop_back_val();
        if (isa<vector::TransferReadOp, vector::TransferWriteOp>(cur)) {
          hasVectorUser = true;
          break;
        }
        if (auto sv = dyn_cast<memref::SubViewOp>(cur)) {
          for (Operation *u : sv.getResult().getUsers())
            worklist.push_back(u);
        }
      }
      if (hasVectorUser) return;

      swizzleMap[alloc.getOperation()] = {maxPhase, elemsPerBank};
    });

    if (swizzleMap.empty()) return;

    // -----------------------------------------------------------------------
    // Phase 2: for every SubViewOp that ultimately traces back to a swizzle-
    // eligible alloc, rewrite the column indices of its load/store users.
    //
    // We follow multi-level subview chains (subview of a subview) because
    // the pipeline produces:
    //   %paddedAlloc = memref.alloc() : memref<64x68xf32, workgroup>
    //   %sv0 = memref.subview %paddedAlloc[0,0][64,64][1,1]  ← padding subview
    //   %sv1 = memref.subview %sv0[row, col][…]               ← tile subview
    //   memref.store …, %sv1[localRow, localCol]
    //
    // The swizzle map is keyed on the AllocOp; we resolve the root alloc by
    // walking up the source chain until we hit an AllocOp.
    // -----------------------------------------------------------------------
    funcOp.walk([&](memref::SubViewOp sv) {
      // Resolve root alloc by chasing source subviews.
      Value src = sv.getSource();
      memref::AllocOp rootAlloc;
      // depth == 0 means sv's source is the alloc directly (the padding
      // subview inserted by ReduceBankConflicts). depth >= 1 means sv is
      // a real tile-level subview. We skip depth-0 subviews: they have
      // offset [0,0] and exist only to present the unpadded shape — swizzling
      // them would corrupt every downstream tile subview.
      int depth = 0;
      while (src) {
        if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
          rootAlloc = alloc;
          break;
        }
        auto parentSv = src.getDefiningOp<memref::SubViewOp>();
        if (!parentSv) break;
        src = parentSv.getSource();
        ++depth;
      }
      if (!rootAlloc) return;

      auto it = swizzleMap.find(rootAlloc.getOperation());
      if (it == swizzleMap.end()) return;

      // Skip the padding subview itself (it just exposes the logical shape;
      // its only users are further subviews and memref.dealloc, not loads).
      if (depth == 0) return;

      auto [maxPhase, elemsPerBank] = it->second;

      // Use the sv's own source type rank for offset extraction.
      auto srcType = cast<MemRefType>(sv.getSource().getType());
      int64_t srcRank = srcType.getRank();
      if (srcRank < 2) return;

      auto [rowBase, colBase] = extractRowColBase(sv, srcRank);

      // Rewrite scalar load/store ops on this subview.
      // vector.transfer_read/write are intentionally skipped — they access
      // contiguous ranges and cannot be corrected by a single XOR offset.
      for (Operation *user : llvm::make_early_inc_range(sv.getResult().getUsers())) {
        if (isa<memref::LoadOp, memref::StoreOp>(user))
          rewriteSubviewAccess(user, rowBase, colBase, maxPhase, elemsPerBank);
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaGPUApplySwizzlePass() {
  return std::make_unique<NovaGPUApplySwizzlePass>();
}

void mlir::nova::registerNovaGPUApplySwizzlePass() {
  PassRegistration<NovaGPUApplySwizzlePass>();
}
