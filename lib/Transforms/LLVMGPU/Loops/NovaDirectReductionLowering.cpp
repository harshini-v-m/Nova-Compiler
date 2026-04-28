//===- NovaDirectReductionLowering.cpp - Direct GPU reduction lowering ====//
//
// Replaces the two-pass SCFScalarizeAccumulator + NovaWarpShuffleReduction
// roundabout with a single pass that matches the raw IR produced by
// NovaScfLoopVectorize and lowers it directly.
//
// WHY TWO PASSES EXIST TODAY
// ==========================
// SCFScalarizeAccumulator (Phase 2) sees this pattern after vectorization:
//
//   vector.store %partial_vec, %scratch[]    // all N threads write same addr
//   %val = memref.load %scratch[]
//   memref.store %val, %output[]            // last writer wins — WRONG
//
// It converts the final store to atomic_rmw so all threads accumulate:
//   memref.atomic_rmw addf %val, %output[]
//
// Then NovaWarpShuffleReduction matches that atomic_rmw, recovers the true
// per-thread partial by tracing through the scratch load-store roundtrip,
// and emits:
//   gpu.all_reduce add %partial
//   scf.if (tid == 0) { memref.store %total, %output[] }
//
// This pass collapses that into one step: match the raw memref.store to an
// external rank-0 output memref and emit the gpu.all_reduce directly.
//
// WHAT PATTERN WE MATCH
// =====================
// Inside a gpu.launch body with >1 threads (bY=bZ=1):
//
//   %partial  = vector.extract %forResult[0]  : f32   ← true per-thread value
//   %vec      = vector.insert  %partial, %p []         ← wrap into vector<f32>
//   vector.store %vec, %scratch[]                      ← scratch write
//   %val      = memref.load   %scratch[]               ← scratch readback
//   memref.store %val, %output[indices...]             ← TARGET: erase this
//
// where %output is defined outside the launch (external output memref).
// The scratch (%alloc_2 in practice) is internal to the function but external
// to the launch body.
//
// We erase the 4 ops (vstore, mload, mstore, insert) and emit in their place:
//
//   %total = gpu.all_reduce <kind> %partial   // intra-block cross-thread reduce
//   scf.if (tid.x == 0) {
//     <write %total to %output[indices]>      // see below for write strategy
//   }
//
// WRITE STRATEGY (derived from IR, not hardcoded)
// ================================================
// The correct write for thread-0 depends on whether multiple blocks can
// write to the SAME output index concurrently.
//
//   Case A — single block (grid = 1,1,1):
//     Only one block exists. gpu.all_reduce gives the complete total.
//     Thread-0 does a plain memref.store. No atomic needed.
//
//   Case B — multiple blocks, each writing a UNIQUE index:
//     (Partial reduction: e.g. reduce along one dim, output shape = [B,...])
//     Each block's thread-0 writes to a different output location.
//     No two blocks touch the same address → plain memref.store is correct.
//     Detection: output indices contain at least one gpu.block_id operand,
//     OR the output rank > 0 (non-scalar output per block).
//
//   Case C — multiple blocks, ALL writing the SAME index:
//     (Full reduction split across blocks: output is a scalar, all blocks
//     contribute to the same memref<f32> slot.)
//     Thread-0 must use memref.atomic_rmw so contributions don't race.
//     Detection: grid > 1 AND output indices have NO gpu.block_id component
//     AND output is rank-0 (scalar).
//
// SCOPE
// =====
// - Full reduction single-block (current config): Case A.
// - Partial reduction multi-block (if tiling ever creates it): Case B.
// - Full reduction multi-block (if tiling splits reduction across blocks): Case C.
// - Elementwise kernels: no external scalar memref.store matched → no-op.
// - The pass does NOT touch intra-kernel private/workgroup stores.
//
// PIPELINE PLACEMENT
// ==================
// Insert AFTER NovaScfLoopVectorize (which creates the pattern we match),
// INSTEAD OF SCFScalarizeAccumulator + NovaWarpShuffleReduction.
//
// The gpu.memset that SCFScalarize used to emit is also handled here:
// emitted before the gpu.launch if the output is not already zeroed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <limits>

#define DEBUG_TYPE "nova-direct-reduction-lowering"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static std::optional<int64_t> getConstIdx(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>())   return c.value();
  return std::nullopt;
}

/// Returns true if `mem` (or its subview root) is defined outside the launch.
static bool isExternalToLaunch(Value mem, gpu::LaunchOp launch) {
  // Walk through subview chains to find root.
  Value cur = mem;
  while (auto sv = cur.getDefiningOp<memref::SubViewOp>())
    cur = sv.getSource();
  Operation *def = cur.getDefiningOp();
  if (!def)
    if (auto arg = dyn_cast<BlockArgument>(cur))
      def = arg.getOwner()->getParentOp();
  return def && !launch->isAncestor(def);
}

/// True if all three grid dims are constant 1.
static bool isSingleBlock(gpu::LaunchOp launch) {
  auto gx = getConstIdx(launch.getGridSizeX());
  auto gy = getConstIdx(launch.getGridSizeY());
  auto gz = getConstIdx(launch.getGridSizeZ());
  return gx && gy && gz && *gx == 1 && *gy == 1 && *gz == 1;
}

/// Returns true if any value in `indices` is (or transitively depends on)
/// a gpu.block_id op — meaning each block writes to a distinct output slot.
static bool indicesContainBlockId(ValueRange indices) {
  // BFS/DFS through defining ops looking for gpu.BlockIdOp.
  SmallVector<Value> worklist(indices.begin(), indices.end());
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !visited.insert(def).second) continue;
    if (isa<gpu::BlockIdOp>(def)) return true;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

/// Infer gpu::AllReduceOperation from the arith combiner op inside scf.for
/// bodies within the launch. Defaults to ADD.
static gpu::AllReduceOperation inferReduceKind(gpu::LaunchOp launch) {
  gpu::AllReduceOperation kind = gpu::AllReduceOperation::ADD;
  launch.walk([&](scf::ForOp forOp) {
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yield.getNumOperands() == 0) return;
    Operation *combiner = yield.getOperand(0).getDefiningOp();
    if (!combiner) return;
    // Peel vector.broadcast wrappers.
    if (auto bcast = dyn_cast<vector::BroadcastOp>(combiner))
      combiner = bcast.getSource().getDefiningOp();
    if (!combiner) return;
    if (isa<arith::MulFOp>(combiner))     { kind = gpu::AllReduceOperation::MUL;      return; }
    if (isa<arith::MaximumFOp>(combiner)) { kind = gpu::AllReduceOperation::MAXIMUMF; return; }
    if (isa<arith::MinimumFOp>(combiner)) { kind = gpu::AllReduceOperation::MINIMUMF; return; }
    if (isa<arith::MaxNumFOp>(combiner))  { kind = gpu::AllReduceOperation::MAXNUMF;  return; }
    if (isa<arith::MinNumFOp>(combiner))  { kind = gpu::AllReduceOperation::MINNUMF;  return; }
    if (isa<arith::MulIOp>(combiner))     { kind = gpu::AllReduceOperation::MUL;      return; }
    if (isa<arith::MaxSIOp>(combiner))    { kind = gpu::AllReduceOperation::MAXSI;    return; }
    if (isa<arith::MaxUIOp>(combiner))    { kind = gpu::AllReduceOperation::MAXUI;    return; }
    if (isa<arith::MinSIOp>(combiner))    { kind = gpu::AllReduceOperation::MINSI;    return; }
    if (isa<arith::MinUIOp>(combiner))    { kind = gpu::AllReduceOperation::MINUI;    return; }
    if (isa<arith::OrIOp>(combiner))      { kind = gpu::AllReduceOperation::OR;       return; }
    if (isa<arith::AndIOp>(combiner))     { kind = gpu::AllReduceOperation::AND;      return; }
    // addf / addi → default ADD.
  });
  return kind;
}

/// Map gpu::AllReduceOperation back to arith::AtomicRMWKind for the
/// inter-block atomic case (Case C). Float types prefer the float variant.
static arith::AtomicRMWKind toAtomicKind(gpu::AllReduceOperation op,
                                          Type elemTy) {
  bool isFloat = isa<FloatType>(elemTy);
  switch (op) {
  case gpu::AllReduceOperation::MUL:      return isFloat ? arith::AtomicRMWKind::mulf : arith::AtomicRMWKind::muli;
  case gpu::AllReduceOperation::MAXIMUMF: return arith::AtomicRMWKind::maximumf;
  case gpu::AllReduceOperation::MINIMUMF: return arith::AtomicRMWKind::minimumf;
  case gpu::AllReduceOperation::MAXNUMF:  return arith::AtomicRMWKind::maxnumf;
  case gpu::AllReduceOperation::MINNUMF:  return arith::AtomicRMWKind::minnumf;
  case gpu::AllReduceOperation::MAXSI:    return arith::AtomicRMWKind::maxs;
  case gpu::AllReduceOperation::MAXUI:    return arith::AtomicRMWKind::maxu;
  case gpu::AllReduceOperation::MINSI:    return arith::AtomicRMWKind::mins;
  case gpu::AllReduceOperation::MINUI:    return arith::AtomicRMWKind::minu;
  case gpu::AllReduceOperation::OR:       return arith::AtomicRMWKind::ori;
  case gpu::AllReduceOperation::AND:      return arith::AtomicRMWKind::andi;
  default:
    return isFloat ? arith::AtomicRMWKind::addf : arith::AtomicRMWKind::addi;
  }
}

//===----------------------------------------------------------------------===//
// Pattern description
//===----------------------------------------------------------------------===//

/// Describes the matched tail pattern inside a gpu.launch body:
///
///   vector.store %vec,  %scratch[]         (vstore)
///   %val = memref.load  %scratch[]         (mload)
///   memref.store %val,  %output[indices]   (mstore — to be replaced)
///
/// `partial` is the scalar extracted from the iter_arg result that feeds
/// vector.insert → vstore. All four ops are stored as Operation* to avoid
/// the const-qualifier issue with typed op wrappers.
struct ReductionTail {
  Value partial;                    // per-thread scalar (true partial sum)
  Value outputMem;                  // external output memref
  SmallVector<Value> outputIndices; // indices into outputMem
  gpu::AllReduceOperation reduceOp;
  Type elemTy;

  Operation *vstore;    // vector.store %vec, %scratch[]
  Operation *mload;     // memref.load %scratch[]
  Operation *mstore;    // memref.store %val, %output[]  ← replaced
  Operation *insertOp;  // vector.insert %partial, %p [] ← may be erased
};

//===----------------------------------------------------------------------===//
// Pattern matching
//===----------------------------------------------------------------------===//

/// Scan the top-level block of the launch body for all matching tails.
///
/// We look for memref.store to an external output, then trace backward:
///   mstore ← mload ← scratch ← vstore ← vector.insert ← partial scalar
static SmallVector<ReductionTail> matchTails(gpu::LaunchOp launch) {
  SmallVector<ReductionTail> results;
  Block &body = launch.getBody().front();
  gpu::AllReduceOperation kind = inferReduceKind(launch);

  for (Operation &op : body) {
    auto mstore = dyn_cast<memref::StoreOp>(&op);
    if (!mstore) continue;

    // Output must be external (defined outside this launch).
    Value outMem = mstore.getMemRef();
    if (!isExternalToLaunch(outMem, launch)) continue;

    // The stored value must come from a memref.load (the scratch readback).
    auto mload = mstore.getValueToStore().getDefiningOp<memref::LoadOp>();
    if (!mload) continue;

    // The load's source must have a vector.store writing to it before the load.
    Value scratch = mload.getMemRef();
    vector::StoreOp vstore;
    for (Operation &candidate : body) {
      if (&candidate == mload.getOperation()) break;
      if (auto vs = dyn_cast<vector::StoreOp>(&candidate))
        if (vs.getBase() == scratch)
          vstore = vs;
    }
    if (!vstore) continue;

    // The vector stored must be a 0-d vector.insert wrapping a scalar.
    // Pattern: vector.insert %scalar, %poison [] : f32 into vector<f32>
    auto insertOp = vstore.getValueToStore().getDefiningOp<vector::InsertOp>();
    if (!insertOp || !insertOp.getStaticPosition().empty()) continue;

    // The inserted scalar is the per-thread partial.
    Value partial = insertOp.getOperand(0);
    Type elemTy = partial.getType();

    // Only handle float and integer scalar types.
    if (!isa<FloatType>(elemTy) && !isa<IntegerType>(elemTy)) continue;

    ReductionTail tail;
    tail.partial   = partial;
    tail.outputMem = outMem;
    tail.outputIndices.assign(mstore.getIndices().begin(),
                               mstore.getIndices().end());
    tail.reduceOp  = kind;
    tail.elemTy    = elemTy;
    tail.vstore    = vstore.getOperation();
    tail.mload     = mload.getOperation();
    tail.mstore    = mstore.getOperation();
    tail.insertOp  = insertOp.getOperation();
    results.push_back(tail);
  }
  return results;
}

//===----------------------------------------------------------------------===//
// Output zero-initialization
//===----------------------------------------------------------------------===//

/// Emit gpu.memset to zero-initialize the output before the launch, if not
/// already done. This replaces the gpu.memset that SCFScalarizeAccumulator
/// used to emit. Only needed for reductions whose identity is zero (ADD),
/// but conservative: always emit for float (0.0f is always correct for addf).
/// For non-zero identities (max → -inf, min → +inf, mul → 1) the output must
/// have been pre-initialized by the user or an earlier pipeline stage — we
/// don't try to emit those here since gpu.memset only fills byte-by-byte and
/// non-zero float patterns aren't byte-uniform.
static void emitZeroMemsetIfNeeded(Value outputMem, gpu::LaunchOp launch,
                                    gpu::AllReduceOperation kind, Type elemTy) {
  // Only ADD reductions need a zero-init from us (identity = 0).
  // MUL needs 1.0, MAX needs -inf, MIN needs +inf — these require the caller
  // to pre-initialize, or an earlier pipeline stage to handle it.
  if (kind != gpu::AllReduceOperation::ADD) return;

  // Check if there is already a gpu.memset for this memref before this launch.
  Block *block = launch->getBlock();
  for (Operation &op : *block) {
    if (&op == launch.getOperation()) break;
    if (auto ms = dyn_cast<gpu::MemsetOp>(&op))
      if (ms.getDst() == outputMem) return;
  }

  OpBuilder b(launch);
  Location loc = launch.getLoc();

  // Emit gpu.memset with the identity value (0 for ADD).
  // Matches the SCFScalarizeAccumulator pattern exactly:
  //   pre.create<gpu::MemsetOp>(loc, TypeRange{}, ValueRange{}, target, zero);
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(ft, 0.0));
    b.create<gpu::MemsetOp>(loc, TypeRange{}, ValueRange{}, outputMem, zero);
  } else {
    Value zero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(elemTy, 0));
    b.create<gpu::MemsetOp>(loc, TypeRange{}, ValueRange{}, outputMem, zero);
  }
}

//===----------------------------------------------------------------------===//
// Core transformation
//===----------------------------------------------------------------------===//

/// Determine the write strategy for thread-0 and emit the replacement code.
///
///   Case A (single block):
///     gpu.all_reduce %partial + tid==0 plain store
///
///   Case B (multi-block, index contains block_id → unique per block):
///     gpu.all_reduce %partial + tid==0 plain store
///     (Each block's thread-0 writes to its own unique index, no conflict.)
///
///   Case C (multi-block, same index for all blocks):
///     gpu.all_reduce %partial + tid==0 atomic_rmw
///     (Multiple blocks compete on the same output slot; atomic prevents race.)
///
/// Inserted immediately after the op that produced `partial`. Erases the
/// vstore / mload / mstore / insertOp chain.
static void lowerTail(const ReductionTail &tail, gpu::LaunchOp launch) {
  bool singleBlock = isSingleBlock(launch);

  // Case B: multi-block but each block writes a distinct index.
  // Detected by the presence of gpu.block_id in the index computation.
  bool uniquePerBlock = !singleBlock &&
                        indicesContainBlockId(tail.outputIndices);

  // Case C: multi-block and all blocks write to the same index.
  bool needsInterBlockAtomic = !singleBlock && !uniquePerBlock;

  LLVM_DEBUG(llvm::dbgs()
             << "[direct-reduction] launch: singleBlock=" << singleBlock
             << " uniquePerBlock=" << uniquePerBlock
             << " needsInterBlockAtomic=" << needsInterBlockAtomic
             << " reduceOp=" << (int)tail.reduceOp << "\n");

  // Insert after the op that produced the per-thread partial.
  Operation *anchor = tail.partial.getDefiningOp();
  if (!anchor)
    anchor = &launch.getBody().front().front();
  OpBuilder b(anchor->getNextNode());
  Location loc = anchor->getLoc();

  // ── Intra-block cross-thread reduction ──
  // gpu.all_reduce combines all threads' partials within the block.
  auto reduceAttr = gpu::AllReduceOperationAttr::get(b.getContext(),
                                                      tail.reduceOp);
  Value blockTotal = b.create<gpu::AllReduceOp>(
      loc, tail.partial.getType(), tail.partial, reduceAttr,
      /*uniform=*/false);

  // ── Thread-0 guard: only one thread per block writes the result ──
  Value tidX   = b.create<gpu::ThreadIdOp>(loc, b.getIndexType(),
                                            gpu::Dimension::x);
  Value c0     = b.create<arith::ConstantIndexOp>(loc, 0);
  Value isT0   = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          tidX, c0);

  b.create<scf::IfOp>(loc, isT0, [&](OpBuilder &ib, Location il) {
    if (!needsInterBlockAtomic) {
      // Case A or B: plain store — either single block or unique index per block.
      ib.create<memref::StoreOp>(il, blockTotal, tail.outputMem,
                                  tail.outputIndices);
    } else {
      // Case C: inter-block atomic to accumulate contributions from all blocks.
      arith::AtomicRMWKind atomicKind = toAtomicKind(tail.reduceOp, tail.elemTy);
      ib.create<memref::AtomicRMWOp>(il, atomicKind, blockTotal,
                                      tail.outputMem, tail.outputIndices);
    }
    ib.create<scf::YieldOp>(il);
  });

  // ── Erase the racy chain in reverse dependency order ──
  tail.mstore->erase();
  tail.mload->erase();
  tail.vstore->erase();
  if (tail.insertOp && tail.insertOp->use_empty())
    tail.insertOp->erase();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaDirectReductionLoweringPass
    : public PassWrapper<NovaDirectReductionLoweringPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaDirectReductionLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect, scf::SCFDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<gpu::LaunchOp> launches;
    funcOp.walk([&](gpu::LaunchOp l) { launches.push_back(l); });

    bool changed = false;

    for (auto launch : launches) {
      // Must be a multi-thread launch on the x-dim only (bY=bZ=1).
      auto bx = getConstIdx(launch.getBlockSizeX());
      auto by = getConstIdx(launch.getBlockSizeY());
      auto bz = getConstIdx(launch.getBlockSizeZ());
      if (!bx || !by || !bz || *bx <= 1 || *by != 1 || *bz != 1)
        continue;

      // Elementwise kernels have no external scalar store in the launch body —
      // matchTails returns empty and we skip cleanly.
      auto tails = matchTails(launch);
      if (tails.empty()) continue;

      for (auto &tail : tails) {
        // Emit zero-init for the output (ADD only; matches what SCFScalarize did).
        emitZeroMemsetIfNeeded(tail.outputMem, launch,
                                tail.reduceOp, tail.elemTy);

        lowerTail(tail, launch);
        changed = true;
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }

  StringRef getArgument() const override {
    return "nova-direct-reduction-lowering";
  }
  StringRef getDescription() const override {
    return "Directly lower reduction tails to gpu.all_reduce + thread-0 store, "
           "replacing SCFScalarizeAccumulator + NovaWarpShuffleReduction. "
           "Single-block and unique-per-block cases use plain store; "
           "multi-block same-index case uses atomic_rmw for inter-block safety.";
  }
};

std::unique_ptr<Pass> createNovaDirectReductionLoweringPass() {
  return std::make_unique<NovaDirectReductionLoweringPass>();
}

void registerNovaDirectReductionLoweringPass() {
  PassRegistration<NovaDirectReductionLoweringPass>();
}

} // namespace mlir::nova
