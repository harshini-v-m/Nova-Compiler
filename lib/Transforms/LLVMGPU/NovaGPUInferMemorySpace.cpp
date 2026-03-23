//===- NovaGPUInferMemorySpace.cpp - Tiered tensor address-space annotation ===//
//
// Pre-bufferization pass.  Annotates bufferization.alloc_tensor ops with
// the appropriate GPU address space before OneShotBufferize runs.
//
// Three-tier cascade (applied in order, fallback to next tier on failure):
//
//   Tier 1 — private  (#gpu.address_space<private>)
//     • Thread-local: NOT accessed by multiple distinct thread-mapped foralls
//       and NOT declared at workgroup scope.
//     • Static size ≤ kMaxPrivateBytes (1 KB).
//     • Becomes memref.alloca after bufferization — zero SRAM consumed.
//
//   Tier 2 — workgroup  (#gpu.address_space<workgroup>)
//     • Fits within the effective SRAM budget:
//         effective_budget = kMaxWorkgroupSRAMBytes
//                          - estimatedPhantomBytes   (bufferizer temporaries)
//     • Candidates sorted largest-first ("most profitable" = most bytes
//       saved from slow global accesses) before the greedy fill loop.
//     • Dynamic-shape tensors: conservatively skipped (size unknown → global).
//       They cannot be statically accounted in the budget.
//
//   Tier 3 — global  (no annotation, bufferize emits device DRAM alloc)
//     • Cross-workgroup tensors (feed workgroup-forall outputs / materialize).
//     • Tensors that do not fit in the workgroup budget.
//     • Dynamic-shape tensors (cannot bound their size statically).
//
// Phantom headroom
// ─────────────────
// OneShotBufferize injects new alloc_tensor ops for:
//   • scf.for iter_arg copies (one per tensor-typed iter_arg)
//   • scf.if result staging buffers (one per tensor-typed result)
// These are invisible to this pass but will become workgroup allocs after
// bufferization.  We estimate their total size and subtract it from the
// workgroup budget so the final post-bufferization peak stays under 48 KB.
// Static phantom sizes are counted exactly; dynamic-shape tensors get a
// conservative per-tensor estimate (4 KB each) capped at kMaxPhantomHeadroomBytes.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "nova-gpu-infer-memory-space"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

/// RTX 3060 / Ampere: 48 KB shared memory per SM block.
static constexpr int64_t kMaxWorkgroupSRAMBytes = 48 * 1024;

/// Max size for private (register/alloca) promotion.
/// A warp can use ~4 KB of register file per thread on Ampere, but exceeding
/// ~1 KB per allocation causes the register allocator to spill to L1 scratch,
/// corrupting compute throughput. Stay at 1 KB.
static constexpr int64_t kMaxPrivateBytes = 1 * 1024;

/// Headroom reserved for phantom allocs OneShotBufferize will inject
/// (iter_arg copy buffers, scf.if staging buffers). Capped at 24 KB.
/// Raised from 8 KB: complex chains (LN->GELU->Linear->SCE) inject up to
/// ~22.5 KB of staging buffers via scf.if; the old 8 KB cap caused the
/// greedy loop to over-commit the budget and trigger illegal memory access.
static constexpr int64_t kMaxPhantomHeadroomBytes = 24 * 1024;

/// Maximum alias-chain recursion depth in collectThreadForallUsers.
/// Chains longer than this are treated conservatively as cross-thread.
static constexpr int kMaxAliasDepth = 12;

//===----------------------------------------------------------------------===//
// Byte-size helpers
//===----------------------------------------------------------------------===//

static int64_t getElemBytes(Type elemType) {
  if (elemType.isF16() || elemType.isBF16() || elemType.isInteger(16)) return 2;
  if (elemType.isF64() || elemType.isInteger(64))                       return 8;
  if (elemType.isInteger(8) || elemType.isInteger(1))                   return 1;
  return 4; // f32, i32, index, etc.
}

/// Returns the static byte size of the alloc_tensor, or -1 if dynamic.
static int64_t allocBytes(bufferization::AllocTensorOp alloc) {
  auto type = cast<ShapedType>(alloc.getResult().getType());
  if (!type.hasStaticShape()) return -1;
  return type.getNumElements() * getElemBytes(type.getElementType());
}

/// Returns the static byte size of a shaped type, or -1 if dynamic.
static int64_t shapedBytes(ShapedType st) {
  if (!st.hasStaticShape()) return -1;
  return st.getNumElements() * getElemBytes(st.getElementType());
}

//===----------------------------------------------------------------------===//
// forall mapping helpers
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUThreadMappingAttr>(a);
  });
}

static bool isWorkgroupForall(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUBlockMappingAttr>(a);
  });
}

//===----------------------------------------------------------------------===//
// isCrossWorkgroupUsed
//
// Returns true if the alloc_tensor (or any alias) is:
//   (a) passed as a shared_out / init to a workgroup-mapped forall, OR
//   (b) used in a bufferization.materialize_in_destination
//
// Both cases require the buffer to survive across workgroup launches and
// must therefore live in global (device) memory.
//
// NOTE: we do NOT short-circuit when the alloc is nested inside a forall.
// A tensor allocated inside a workgroup forall can still be a shared_out
// of a nested workgroup forall, which would be a cross-scope use.
//===----------------------------------------------------------------------===//

static bool isCrossWorkgroupUsed(bufferization::AllocTensorOp alloc) {
  SmallVector<Value> worklist = {alloc.getResult()};
  SmallPtrSet<Value, 16> visited;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    for (Operation *user : v.getUsers()) {
      // Used as output of a workgroup-mapped forall → must be global.
      if (auto forallOp = dyn_cast<scf::ForallOp>(user)) {
        if (isWorkgroupForall(forallOp)) {
          for (Value out : forallOp.getOutputs())
            if (out == v) return true;
        }
      }

      // Used in materialize_in_destination → survives across kernel → global.
      if (isa<bufferization::MaterializeInDestinationOp>(user))
        return true;

      // Chase aliases.
      if (isa<tensor::CastOp, tensor::InsertSliceOp, tensor::ExtractSliceOp,
              tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user)) {
        for (Value res : user->getResults())
          worklist.push_back(res);
        continue;
      }

      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
        for (int i = 0, e = linalgOp.getNumDpsInits(); i < e; ++i) {
          OpOperand *init = linalgOp.getDpsInitOperand(i);
          if (init->get() == v)
            worklist.push_back(linalgOp.getTiedOpResult(init));
        }
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// isCrossThreadAccess
//
// Returns true if the alloc is accessed by MULTIPLE distinct thread-mapped
// foralls (different threads read AND write it → needs shared memory).
//
// Fix vs original: removed the last condition
//   `if (!hasNonForallUser && !writers.empty()) return true`
// That condition incorrectly forced single-writer, zero-reader allocs into
// workgroup even when they are purely local to one thread. A tensor only
// written by one thread-mapped forall and never read from another forall is
// thread-local and private-eligible.
//===----------------------------------------------------------------------===//

static void collectThreadForallUsers(Value val,
                                     SmallVectorImpl<scf::ForallOp> &writers,
                                     SmallVectorImpl<scf::ForallOp> &readers,
                                     bool &depthLimitHit, int depth = 0) {
  if (depth > kMaxAliasDepth) {
    depthLimitHit = true;
    return;
  }
  for (Operation *user : val.getUsers()) {
    // Case 1: thread-forall — classify as writer (val in outputs) or reader.
    if (auto forallOp = dyn_cast<scf::ForallOp>(user)) {
      if (hasThreadMapping(forallOp)) {
        bool isWriter = false;
        for (Value init : forallOp.getOutputs())
          if (init == val) { isWriter = true; break; }
        (isWriter ? writers : readers).push_back(forallOp);
      }
      continue;
    }
    // Case 2: linalg DPS init — chase alias through the tied result.
    bool handled = false;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
      for (int i = 0, e = linalgOp.getNumDpsInits(); i < e; ++i) {
        OpOperand *init = linalgOp.getDpsInitOperand(i);
        if (init->get() == val) {
          collectThreadForallUsers(linalgOp.getTiedOpResult(init),
                                   writers, readers, depthLimitHit, depth + 1);
          handled = true;
          break;
        }
      }
    }
    // Case 3: tensor alias ops — chase through all results.
    if (!handled &&
        isa<tensor::CastOp, tensor::InsertSliceOp, tensor::ExtractSliceOp,
            tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user)) {
      for (Value res : user->getResults())
        collectThreadForallUsers(res, writers, readers,
                                 depthLimitHit, depth + 1);
    }
    // Case 4: any other user — not an alias, stop chasing this branch.
  }
}

static bool isCrossThreadAccess(bufferization::AllocTensorOp alloc) {
  SmallVector<scf::ForallOp> writers, readers;
  bool depthLimitHit = false;
  collectThreadForallUsers(alloc.getResult(), writers, readers, depthLimitHit);

  // Alias chain hit the depth limit — conservatively treat as cross-thread to
  // avoid a race caused by truncated analysis.
  if (depthLimitHit) return true;

  // Cross-thread iff at least one forall writes AND at least one reads.
  // This covers both same-forall read+write races and distinct-forall
  // producer/consumer cases.  The earlier cross-product loop (w != r for any
  // pair) was fully subsumed by this single check.
  return !writers.empty() && !readers.empty();
}

static bool isAtWorkgroupScope(bufferization::AllocTensorOp alloc) {
  bool insideWorkgroupForall = false;
  Operation *parent = alloc->getParentOp();
  while (parent) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(parent)) {
      if (hasThreadMapping(forallOp)) return false;
      if (isWorkgroupForall(forallOp)) insideWorkgroupForall = true;
    }
    parent = parent->getParentOp();
  }
  return insideWorkgroupForall;
}

//===----------------------------------------------------------------------===//
// findScopeBlock
//
// Walks the parent-op chain from `op` upward to find the body Block of the
// innermost workgroup-mapped scf.forall.  Returns the function entry block
// if no workgroup forall is found.  Optionally sets *scopeForallOut.
//===----------------------------------------------------------------------===//

static Block *findScopeBlock(Operation *op, func::FuncOp funcOp,
                              scf::ForallOp *scopeForallOut = nullptr) {
  Block *scopeBlock = &funcOp.getBody().front();
  if (scopeForallOut)
    *scopeForallOut = scf::ForallOp{};
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(parent)) {
      if (isWorkgroupForall(forallOp)) {
        scopeBlock = forallOp.getBody();
        if (scopeForallOut)
          *scopeForallOut = forallOp;
        break;
      }
    }
    parent = parent->getParentOp();
  }
  return scopeBlock;
}

//===----------------------------------------------------------------------===//
// estimatePhantomBytes
//
// Estimate the workgroup SRAM that OneShotBufferize will consume for
// alloc_tensor ops it injects invisibly:
//
//   • scf.for  : one iter_arg copy buffer per tensor-typed iter_arg.
//   • scf.if   : one staging buffer per tensor-typed result.
//
// Both are created inside the forall body and land in workgroup space after
// bufferization if their containing region is inside a workgroup forall.
//
// Descent stops at nested scf.forall boundaries (those are separate scopes).
// Result is capped at kMaxPhantomHeadroomBytes.
//===----------------------------------------------------------------------===//

static int64_t estimatePhantomBytes(scf::ForallOp forallOp) {
  // Static phantom sizes are exact — do NOT cap them.
  // Dynamic-shape tensors each contribute 4 KB, capped at kMaxPhantomHeadroomBytes
  // total, to avoid over-reserving when there are many dynamic iter_args.
  int64_t staticTotal  = 0;
  int64_t dynamicCount = 0;

  forallOp.walk([&](Operation *op) -> WalkResult {
    // Do not descend into nested forall ops (separate scope).
    if (isa<scf::ForallOp>(op) && op != forallOp.getOperation())
      return WalkResult::skip();

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      for (Value iterArg : forOp.getInitArgs()) {
        auto st = dyn_cast<ShapedType>(iterArg.getType());
        if (!st) continue;
        int64_t b = shapedBytes(st);
        if (b > 0) staticTotal += b;
        else       ++dynamicCount;
      }
      return WalkResult::advance();
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      for (Type resTy : ifOp.getResultTypes()) {
        auto st = dyn_cast<ShapedType>(resTy);
        if (!st) continue;
        int64_t b = shapedBytes(st);
        if (b > 0) staticTotal += b;
        else       ++dynamicCount;
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  // Cap only the dynamic estimate; static bytes are exact.
  int64_t dynamicEstimate =
      std::min(dynamicCount * int64_t{4 * 1024}, kMaxPhantomHeadroomBytes);
  int64_t total = staticTotal + dynamicEstimate;

  LLVM_DEBUG(llvm::dbgs()
             << "[" DEBUG_TYPE "]  phantom: static=" << staticTotal
             << " B  dynamic=" << dynamicCount << "×4 KB (capped "
             << dynamicEstimate << " B)  total=" << total << " B\n");
  return total;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUInferMemorySpacePass
    : public PassWrapper<NovaGPUInferMemorySpacePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUInferMemorySpacePass)

  NovaGPUInferMemorySpacePass() = default;
  NovaGPUInferMemorySpacePass(const NovaGPUInferMemorySpacePass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
                    scf::SCFDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();

    auto privateSpace   = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getPrivateAddressSpace());
    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    bool failed = false;

    // ── Phase 1: classify every alloc_tensor ─────────────────────────────
    //
    // Group workgroup candidates by the body block of their innermost
    // workgroup forall (= their SRAM scope).  Private promotions are
    // applied immediately; global tensors are left untagged.

    // Map: workgroup-forall body block → list of WG candidates in that scope.
    DenseMap<Block *, SmallVector<bufferization::AllocTensorOp>> wgCandsByScope;
    // Map: scope block → the forall op itself (for phantom estimation).
    // Populated for every scope — including pre-committed-only scopes — so
    // phantom headroom is always available for the greedy loop.
    DenseMap<Block *, scf::ForallOp> scopeToForall;
    // Map: scope block → bytes already consumed by pre-annotated WG allocs.
    DenseMap<Block *, int64_t> preCommittedBytes;

    // ── Single classification walk ────────────────────────────────────────
    //
    // Handles all three cases in one pass:
    //   • Pre-annotated allocs → verify and record pre-committed bytes.
    //   • Cross-workgroup allocs → leave untagged (Tier 3 / global).
    //   • Tier 1 eligible → tag private immediately.
    //   • Everything else → stage as Tier-2 workgroup candidate.

    funcOp.walk([&](bufferization::AllocTensorOp alloc) {
      scf::ForallOp scopeForall;
      Block *scopeBlock = findScopeBlock(alloc, funcOp, &scopeForall);
      // Always record the forall so phantom estimation works even when a
      // scope's only allocs are pre-committed (and thus return early below).
      if (scopeForall)
        scopeToForall.try_emplace(scopeBlock, scopeForall);

      // ── Already annotated ──────────────────────────────────────────────
      std::optional<Attribute> existingSpace = alloc.getMemorySpace();
      if (existingSpace.has_value()) {
        if (*existingSpace == workgroupSpace) {
          // Record against the budget so the greedy loop sees the real headroom.
          int64_t b = allocBytes(alloc);
          if (b > 0) {
            preCommittedBytes[scopeBlock] += b;
            LLVM_DEBUG(llvm::dbgs()
                       << "[" DEBUG_TYPE "]  pre-committed workgroup bytes=" << b
                       << "  scope running=" << preCommittedBytes[scopeBlock]
                       << "\n");
          }
          return; // leave as-is
        }
        if (*existingSpace == privateSpace)
          return; // leave as-is
        alloc.emitOpError(
            "unexpected memory space — must be private or workgroup (got ")
            << *existingSpace << ")";
        failed = true;
        return;
      }

      // ── Tier 3 fast-path: must live in global memory ───────────────────
      if (isCrossWorkgroupUsed(alloc))
        return; // leave untagged → global

      // ── Tier 1: private ───────────────────────────────────────────────
      bool threadLocal = !isCrossThreadAccess(alloc) &&
                         !isAtWorkgroupScope(alloc);
      int64_t bytes = allocBytes(alloc);

      if (threadLocal && bytes >= 0 && bytes <= kMaxPrivateBytes) {
        alloc.setMemorySpaceAttr(privateSpace);
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  private  bytes=" << bytes << "\n");
        return;
      }

      // ── Tier 2 candidate ─────────────────────────────────────────────
      wgCandsByScope[scopeBlock].push_back(alloc);
    });

    if (failed) { signalPassFailure(); return; }

    // ── Phase 2: greedy workgroup promotion per SRAM scope ────────────────
    //
    // For each scope:
    //   1. Phantom bytes (bufferizer-injected iter_arg copies and scf.if
    //      staging buffers) are estimated using exact static sizes; only the
    //      dynamic-shape portion is capped.
    //   2. effective budget = 48 KB − phantom − pre-committed.
    //   3. Dynamic-shape candidates are demoted to global (unknown size).
    //   4. Static candidates are sorted largest-first.
    //   5. Greedy fill until the budget is exhausted.

    for (auto &[scopeBlock, candidates] : wgCandsByScope) {
      // Phantom headroom: only computable if we have the forall op.
      int64_t phantomBytes = 0;
      auto it = scopeToForall.find(scopeBlock);
      if (it != scopeToForall.end())
        phantomBytes = estimatePhantomBytes(it->second);

      // Subtract both phantom headroom AND bytes already committed by
      // pre-annotated workgroup allocs that the greedy loop never sees.
      int64_t preCommitted = preCommittedBytes.lookup(scopeBlock);
      int64_t effectiveBudget =
          kMaxWorkgroupSRAMBytes - phantomBytes - preCommitted;

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]  scope budget: " << effectiveBudget
                 << " B  (48 KB - " << phantomBytes << " B phantom - "
                 << preCommitted << " B pre-committed)\n");

      if (effectiveBudget <= 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  WARNING: effectiveBudget exhausted by"
                      " phantom/pre-committed bytes (" << effectiveBudget
                   << " B); all candidates will be demoted to global.\n");
      }

      // Separate dynamic from static candidates.
      SmallVector<bufferization::AllocTensorOp> staticCands;
      for (auto alloc : candidates) {
        int64_t b = allocBytes(alloc);
        if (b < 0) {
          // Dynamic shape — size unknown statically.
          // Promoting blindly risks blowing the budget. Demote to global.
          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE "]  global (dynamic shape)\n");
          // Leave untagged.
        } else {
          staticCands.push_back(alloc);
        }
      }

      // Sort largest-first: most bytes saved per slot = most profitable.
      llvm::sort(staticCands, [](bufferization::AllocTensorOp a,
                                 bufferization::AllocTensorOp b) {
        return allocBytes(a) > allocBytes(b);
      });

      int64_t cumBytes = 0;
      for (auto alloc : staticCands) {
        int64_t b = allocBytes(alloc);
        if (cumBytes + b <= effectiveBudget) {
          alloc.setMemorySpaceAttr(workgroupSpace);
          cumBytes += b;
          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE "]  workgroup bytes=" << b
                     << "  running=" << cumBytes
                     << " / " << effectiveBudget << " B\n");
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE "]  global (budget full): bytes=" << b
                     << "  running=" << cumBytes
                     << " / " << effectiveBudget << " B\n");
          // Leave untagged → global.
        }
      }
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-infer-memory-space";
  }
  StringRef getDescription() const override {
    return "Three-tier GPU memory space inference: private (≤1 KB, thread-"
           "local) → workgroup (greedy largest-first, effective budget = "
           "48 KB minus phantom headroom for bufferizer temporaries) → "
           "global (cross-workgroup, dynamic shape, or budget overflow)";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUInferMemorySpacePass() {
  return std::make_unique<NovaGPUInferMemorySpacePass>();
}

void registerNovaGPUInferMemorySpacePass() {
  PassRegistration<NovaGPUInferMemorySpacePass>();
}

} // namespace mlir::nova