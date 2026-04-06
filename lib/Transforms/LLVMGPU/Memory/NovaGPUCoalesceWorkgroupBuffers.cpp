//===- NovaGPUCoalesceWorkgroupBuffers.cpp --------------------------------===//
//
// Post-bufferization pass.  Runs after OneShotBufferize.
//
// Goal
// ────
// Reuse workgroup SRAM allocs that have the same MemRefType and whose
// live ranges do not overlap.  "Live range" is defined simply as
// [allocIdx, lastUseIdx] — the sequential index of the alloc among direct
// body children, to the maximum index of any direct-body child that
// (transitively through view-like ops) uses the alloc.
//
// No epoch splitting.  The original pass split each buffer's lifetime into
// "epochs" separated by linalg.fill / write-only linalg.generic kills.
// After bufferization that structure is unreliable:
//   • linalg.fill on a subview is no longer a kill of the parent alloc.
//   • iter_arg copies blur the write frontier.
//   • The greedy scheduler already handles reuse correctly without epochs:
//     if two allocs truly overlap in time, their plain [def,lastUse]
//     intervals overlap too and they will not be merged.
//
// Steps
// ─────
//   1. Index every direct child of the forall body (for ordering).
//   2. Walk ALL nested ops (excluding nested forall scopes) to collect
//      workgroup allocs — captures bufferizer-generated allocs inside
//      scf.if / scf.for that the original direct-children walk missed.
//   3. Compute [allocIdx, lastUseIdx] for each alloc by anchoring every
//      transitive user to its enclosing direct-body child.
//   4. Group allocs by identical MemRefType (shape + elem + address space).
//   5. Within each group: sort by def, greedy earliest-free scheduling.
//   6. Apply: redirect uses of victims to their repr, erase victims and
//      their deallocs, move repr deallocs to block terminator.
//   7. Sweep-line peak-bytes calculation on surviving intervals.
//   8. Enforce 48 KB budget — signalPassFailure() on violation.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-coalesce-workgroup-buffers"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static constexpr int64_t kMaxWorkgroupSRAMBytes = 48 * 1024;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// True if op produces a result that aliases its source operand (view-like).
static bool isViewLike(Operation *op) {
  return isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp,
             memref::ExpandShapeOp, memref::CollapseShapeOp,
             memref::TransposeOp>(op);
}

/// Walk up the parent chain and return the ancestor of `op` whose parent
/// block is `blk`, or nullptr if `op` is not nested inside `blk`.
static Operation *ancestorInBlock(Operation *op, Block *blk) {
  while (op) {
    if (op->getBlock() == blk) return op;
    op = op->getParentOp();
  }
  return nullptr;
}

/// Byte size of a fully static workgroup memref alloc.
static int64_t memrefBytes(memref::AllocOp alloc) {
  auto mtype = alloc.getType();
  int64_t elems = 1;
  for (int64_t d : mtype.getShape())
    elems *= (d > 0 ? d : 1);
  Type elem = mtype.getElementType();
  int64_t eb = 4;
  if (elem.isF16() || elem.isBF16() || elem.isInteger(16))      eb = 2;
  else if (elem.isF64() || elem.isInteger(64))                   eb = 8;
  else if (elem.isInteger(8) || elem.isInteger(1))               eb = 1;
  return elems * eb;
}

//===----------------------------------------------------------------------===//
// Live interval  [def, lastUse]
//===----------------------------------------------------------------------===//

struct LiveInterval {
  unsigned def;
  unsigned lastUse;
  bool overlaps(const LiveInterval &o) const {
    return !(lastUse < o.def || o.lastUse < def);
  }
};

/// Compute the live interval for `allocVal` by:
///   • Starting at the sequential index of the alloc op itself (def).
///   • BFS-ing through all transitive users (following view-like chains).
///   • Anchoring each user to its enclosing direct-body child via
///     ancestorInBlock(), taking the max index seen (lastUse).
///
/// This correctly handles allocs whose users are inside nested scf.if /
/// scf.for — those are anchored to the containing direct-body op, which
/// is the right conservative bound for liveness.
static LiveInterval
computeLiveInterval(Value allocVal, Block *bodyBlock,
                    const DenseMap<Operation *, unsigned> &opIdx) {
  unsigned allocIdx = opIdx.lookup(allocVal.getDefiningOp());
  unsigned lastUse  = allocIdx;

  SmallVector<Value> worklist = {allocVal};
  SmallPtrSet<Value, 16> visited;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;

    for (Operation *user : v.getUsers()) {
      if (isa<memref::DeallocOp>(user)) continue; // deallocs don't extend live range
      Operation *anc = ancestorInBlock(user, bodyBlock);
      if (!anc) continue;
      auto it = opIdx.find(anc);
      if (it == opIdx.end()) continue;
      lastUse = std::max(lastUse, it->second);
      // Transitively follow view-like ops — they alias the same storage.
      if (isViewLike(user))
        for (Value res : user->getResults())
          worklist.push_back(res);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]  interval alloc@"
                           << allocIdx << "  [" << allocIdx
                           << "," << lastUse << "]\n");
  return {allocIdx, lastUse};
}

//===----------------------------------------------------------------------===//
// Post-merge peak bytes  (sweep-line)
//===----------------------------------------------------------------------===//

struct SurvivorInfo {
  memref::AllocOp repr;
  LiveInterval    interval;
  int64_t         bytes;
};

static int64_t computePeakBytes(ArrayRef<SurvivorInfo> survivors) {
  if (survivors.empty()) return 0;

  struct Event { unsigned idx; int64_t delta; };
  SmallVector<Event> events;
  events.reserve(survivors.size() * 2);

  for (const auto &s : survivors) {
    events.push_back({s.interval.def,         +s.bytes});
    events.push_back({s.interval.lastUse + 1, -s.bytes});
  }

  llvm::sort(events, [](const Event &a, const Event &b) {
    if (a.idx != b.idx) return a.idx < b.idx;
    return a.delta < b.delta; // process frees before allocs at same index
  });

  int64_t current = 0, peak = 0;
  for (const auto &e : events) {
    current += e.delta;
    peak = std::max(peak, current);
  }
  return peak;
}

//===----------------------------------------------------------------------===//
// Per-forall coalescing
//===----------------------------------------------------------------------===//

struct CoalesceResult { unsigned coalesced; int64_t peakBytesAfter; };

static CoalesceResult
coalesceWorkgroupAllocsInForall(scf::ForallOp forallOp) {
  Block      *body = forallOp.getBody();
  MLIRContext *ctx  = forallOp.getContext();

  auto wgSpace = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

  // ── Step 1: index direct children ────────────────────────────────────
  DenseMap<Operation *, unsigned> opIdx;
  unsigned seq = 0;
  for (Operation &op : body->getOperations())
    opIdx[&op] = seq++;

  // ── Step 2: collect all static WG allocs, direct AND nested ──────────
  //
  // Walk the entire forall body recursively so we catch allocs that
  // OneShotBufferize emitted inside scf.if / scf.for regions (iter_arg
  // copy allocs, padding buffers, etc.).  The original pass iterated only
  // direct body children and silently missed these, causing:
  //   (a) missed coalescing opportunities, and
  //   (b) an incomplete peak estimate → silent SRAM budget overruns.
  //
  // Stop descent at nested scf.forall — those are separate block scopes
  // handled by their own pass invocation (and their allocs are filtered
  // out by the address-space check anyway for thread-mapped foralls).
  SmallVector<memref::AllocOp> wgAllocs;
  forallOp.walk([&](memref::AllocOp alloc) {
    if (alloc.getType().getMemorySpace() != wgSpace) return;
    if (!alloc.getDynamicSizes().empty())             return;
    // Skip allocs inside a nested forall (different block scope).
    Operation *p = alloc->getParentOp();
    while (p && p != forallOp.getOperation()) {
      if (isa<scf::ForallOp>(p)) return;
      p = p->getParentOp();
    }
    wgAllocs.push_back(alloc);
  });

  if (wgAllocs.size() < 2) {
    SmallVector<SurvivorInfo> sv;
    for (auto a : wgAllocs) {
      sv.push_back({a, computeLiveInterval(a.getResult(), body, opIdx),
                    memrefBytes(a)});
    }
    return {0, computePeakBytes(sv)};
  }

  // ── Step 3: compute live intervals ───────────────────────────────────
  struct AllocInfo {
    memref::AllocOp allocOp;
    LiveInterval    interval;
    int64_t         bytes;
  };
  SmallVector<AllocInfo> infos;
  infos.reserve(wgAllocs.size());
  for (auto a : wgAllocs)
    infos.push_back({a, computeLiveInterval(a.getResult(), body, opIdx),
                     memrefBytes(a)});

  // ── Step 4: group by MemRefType (shape + elem + address space) ────────
  //
  // Only allocs with identical MemRefType can share physical storage.
  // Mixing types would require a reinterpret_cast which is unsafe for
  // differently-shaped workgroup buffers.
  DenseMap<Type, SmallVector<AllocInfo *>> byType;
  for (auto &info : infos)
    byType[info.allocOp.getType()].push_back(&info);

  // ── Step 5: greedy interval scheduling within each type group ─────────
  SmallVector<std::pair<memref::AllocOp, memref::AllocOp>> toCoalesce;
  SmallPtrSet<Operation *, 8>                               reprsThatAbsorbed;
  DenseMap<Operation *, SurvivorInfo>                       survivorMap;

  for (auto &[type, group] : byType) {
    if (group.size() < 2) {
      survivorMap[group.front()->allocOp.getOperation()] = {
          group.front()->allocOp,
          group.front()->interval,
          group.front()->bytes};
      continue;
    }

    // Sort by def so we try to pack earliest-starting allocs first.
    llvm::sort(group, [](const AllocInfo *a, const AllocInfo *b) {
      return a->interval.def < b->interval.def;
    });

    struct Slot {
      LiveInterval    merged; // union of all absorbed intervals in this slot
      memref::AllocOp repr;
      int64_t         bytes;
    };
    SmallVector<Slot> slots;

    for (AllocInfo *info : group) {
      // Find the first existing slot whose merged interval doesn't overlap.
      unsigned chosen = slots.size(); // sentinel → open new slot
      for (unsigned s = 0; s < slots.size(); ++s) {
        if (!slots[s].merged.overlaps(info->interval)) {
          chosen = s; break;
        }
      }

      if (chosen < slots.size()) {
        // Absorb into existing slot: extend its merged interval.
        Slot &slot = slots[chosen];
        slot.merged.def     = std::min(slot.merged.def,     info->interval.def);
        slot.merged.lastUse = std::max(slot.merged.lastUse, info->interval.lastUse);

        toCoalesce.push_back({info->allocOp, slot.repr});
        reprsThatAbsorbed.insert(slot.repr.getOperation());
        // Update the survivor map with the extended interval.
        survivorMap[slot.repr.getOperation()].interval = slot.merged;

        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  coalescing alloc@"
                   << info->interval.def << " → repr@"
                   << opIdx.lookup(slot.repr.getOperation()) << "\n");
      } else {
        // Open a new slot — this alloc becomes the repr.
        slots.push_back({info->interval, info->allocOp, info->bytes});
        survivorMap[info->allocOp.getOperation()] = {
            info->allocOp, info->interval, info->bytes};
      }
    }
  }

  if (toCoalesce.empty()) {
    SmallVector<SurvivorInfo> sv;
    for (auto &[op, info] : survivorMap) sv.push_back(info);
    return {0, computePeakBytes(sv)};
  }

  // ── Step 6: apply coalescing ──────────────────────────────────────────
  for (auto [victim, repr] : toCoalesce) {
    // Erase victim's deallocs first to avoid dangling uses.
    SmallVector<Operation *> toErase;
    for (Operation *u : victim.getResult().getUsers())
      if (isa<memref::DeallocOp>(u)) toErase.push_back(u);
    for (Operation *d : toErase) d->erase();

    victim.getResult().replaceAllUsesWith(repr.getResult());
    victim.erase();
  }

  // ── Step 7: move repr deallocs to just before the block terminator ────
  //
  // After absorbing victims, the repr's uses now span further than its
  // original dealloc position.  Moving the dealloc to the terminator is
  // safe: workgroup allocs are implicitly freed at kernel exit anyway.
  Operation *terminator = body->getTerminator();
  for (Operation *reprOp : reprsThatAbsorbed) {
    auto repr = cast<memref::AllocOp>(reprOp);
    for (Operation *u : repr.getResult().getUsers()) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
        dealloc->moveBefore(terminator);
        break;
      }
    }
  }

  // ── Step 8: post-merge peak for budget verification ───────────────────
  SmallVector<SurvivorInfo> survivors;
  survivors.reserve(survivorMap.size());
  for (auto &[op, info] : survivorMap) {
    if (!info.repr.getOperation()->getBlock()) continue; // victim — erased
    survivors.push_back(info);
  }

  int64_t peakBytes = computePeakBytes(survivors);

  LLVM_DEBUG(llvm::dbgs()
             << "[" DEBUG_TYPE "]  post-merge peak: " << peakBytes
             << " / " << kMaxWorkgroupSRAMBytes << " B  ("
             << survivors.size() << " surviving allocs)\n");

  return {static_cast<unsigned>(toCoalesce.size()), peakBytes};
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NovaGPUCoalesceWorkgroupBuffersPass
    : public PassWrapper<NovaGPUCoalesceWorkgroupBuffersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUCoalesceWorkgroupBuffersPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect,
                    func::FuncDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    unsigned totalCoalesced = 0;
    bool     budgetViolation = false;

    // Collect all block-mapped (workgroup-level) scf.forall ops.
    SmallVector<scf::ForallOp> blockForalls;
    funcOp.walk([&](scf::ForallOp op) {
      auto mapping = op.getMappingAttr();
      if (!mapping) return;
      bool hasBlock = llvm::any_of(mapping, [](Attribute a) {
        return isa<gpu::GPUBlockMappingAttr>(a);
      });
      if (hasBlock) blockForalls.push_back(op);
    });

    for (scf::ForallOp op : blockForalls) {
      auto [coalesced, peakBytes] = coalesceWorkgroupAllocsInForall(op);
      totalCoalesced += coalesced;

      if (peakBytes > kMaxWorkgroupSRAMBytes) {
        op.emitError()
            << "[nova-coalesce-workgroup-buffers] workgroup SRAM budget "
               "exceeded after coalescing: peak "
            << peakBytes << " B > limit " << kMaxWorkgroupSRAMBytes
            << " B. Reduce tile sizes, split the kernel, or ensure "
               "NovaGPUInferMemorySpace ran before bufferization.";
        budgetViolation = true;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] eliminated "
                             << totalCoalesced
                             << " redundant workgroup alloc(s)\n");

    if (budgetViolation) signalPassFailure();
  }

  StringRef getArgument() const override {
    return "nova-gpu-coalesce-workgroup-buffers";
  }
  StringRef getDescription() const override {
    return "Coalesces same-type workgroup allocs with non-overlapping "
           "[def,lastUse] live ranges (direct + nested, epoch-free) and "
           "verifies the 48 KB SRAM budget post-merge.";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUCoalesceWorkgroupBuffersPass() {
  return std::make_unique<NovaGPUCoalesceWorkgroupBuffersPass>();
}

void registerNovaGPUCoalesceWorkgroupBuffersPass() {
  PassRegistration<NovaGPUCoalesceWorkgroupBuffersPass>();
}

} // namespace mlir::nova