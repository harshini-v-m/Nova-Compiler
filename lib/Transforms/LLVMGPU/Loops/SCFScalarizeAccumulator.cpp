//===- SCFScalarizeAccumulator.cpp - Scalarize loop-carried accumulators --===//
//
// Converts buffer-based loop accumulation patterns in scf.for loops to
// register-based iteration arguments (iter_args), reducing memory traffic.
//
// Two-phase transformation:
//
//   Phase 1 — Scalarize intra-kernel accumulators:
//     Detects scf.for loops that load from and store to the same memref
//     with loop-invariant indices. Converts the load/store pair into an
//     iter_arg that carries the scalar value in registers across iterations.
//     After the loop, writes the final scalar back via store (or atomic_rmw
//     if the memref is shared/cross-kernel).
//
//   Phase 2 — Atomicize cross-block stores:
//     Detects memref.store ops inside gpu.launch that target memrefs defined
//     outside the launch (i.e., distributed reduction outputs like "mean").
//     Each block computes a partial result, so stores must use atomic_rmw
//     (addf) to accumulate correctly. Also inserts gpu.memset to zero-init
//     the target before the launch so the accumulation starts from zero.
//
// PERFORMANCE CRITICAL: Without Phase 1, each iteration of the accumulation
// loop reads from and writes to shared or L2 memory, adding significant
// latency. Scalarizing keeps the accumulator in a register for the entire
// loop, only writing back once.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;

namespace {

/// Extract constant integer value from a Value, or return std::nullopt.
/// Mirrors the implementation in NovaWarpShuffleReduction.cpp.
static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return std::nullopt;
}

/// Returns true if the launch has exactly 1 total thread:
/// gridSize=(1,1,1) AND blockSize=(1,1,1). With a single thread there is
/// zero contention, so atomic_rmw is unnecessary.
static bool isSingleThreadLaunch(gpu::LaunchOp launch) {
  auto gx = getConstantIndex(launch.getGridSizeX());
  auto gy = getConstantIndex(launch.getGridSizeY());
  auto gz = getConstantIndex(launch.getGridSizeZ());
  auto bx = getConstantIndex(launch.getBlockSizeX());
  auto by = getConstantIndex(launch.getBlockSizeY());
  auto bz = getConstantIndex(launch.getBlockSizeZ());
  return gx && gy && gz && bx && by && bz &&
         *gx == 1 && *gy == 1 && *gz == 1 &&
         *bx == 1 && *by == 1 && *bz == 1;
}

/// Returns true if `memref` has at least one use that is ordered AFTER
/// `launchOp` in the same block (i.e., a subsequent op reads/writes it).
/// If there are no post-launch uses the store inside the launch is dead.
/// Conservative: returns true when the ordering cannot be determined.
static bool hasPostLaunchUses(Value memref, gpu::LaunchOp launchOp) {
  Block *launchBlock = launchOp->getBlock();
  if (!launchBlock)
    return true; // conservative

  for (Operation *user : memref.getUsers()) {
    // Skip uses inside this launchOp — they are the stores being considered.
    if (launchOp->isAncestor(user))
      continue;

    // Deallocs free memory but don't read content — ignore them.
    if (isa<memref::DeallocOp>(user))
      continue;

    // Walk up to find the top-level ancestor in launchBlock.
    Operation *ancestor = user;
    while (ancestor && ancestor->getBlock() != launchBlock)
      ancestor = ancestor->getParentOp();

    if (!ancestor)
      return true; // conservative: different region

    if (ancestor == launchOp.getOperation())
      continue; // already handled above

    // If ancestor is ordered after launchOp, this is a genuine post-launch use.
    if (launchOp->isBeforeInBlock(ancestor))
      return true;
  }
  return false;
}

/// Return the identity element for the given reduction kind and element type.
///   addf/addi/ori -> 0    mulf -> 1.0    maximumf/maxnumf -> -inf
///   minimumf/minnumf -> +inf   maxs -> INT_MIN   maxu -> 0
///   mins -> INT_MAX   minu/andi -> all-ones
static Value buildIdentityValue(OpBuilder &builder, Location loc,
                                arith::AtomicRMWKind kind, Type elemTy) {
  if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
    double val = 0.0;
    switch (kind) {
    case arith::AtomicRMWKind::mulf:
      val = 1.0; break;
    case arith::AtomicRMWKind::maximumf:
    case arith::AtomicRMWKind::maxnumf:
      val = -std::numeric_limits<double>::infinity(); break;
    case arith::AtomicRMWKind::minimumf:
    case arith::AtomicRMWKind::minnumf:
      val = std::numeric_limits<double>::infinity(); break;
    default:
      val = 0.0; break;
    }
    return builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(floatTy, val));
  }
  auto intTy = cast<IntegerType>(elemTy);
  unsigned w = intTy.getWidth();
  int64_t val = 0;
  switch (kind) {
  case arith::AtomicRMWKind::muli:
    val = 1; break;
  case arith::AtomicRMWKind::andi:
  case arith::AtomicRMWKind::minu:
    val = -1; break; // all-ones: UINT_MAX in unsigned representation
  case arith::AtomicRMWKind::maxs:
    val = (w >= 64) ? std::numeric_limits<int64_t>::min()
                    : -(int64_t(1) << (w - 1)); break;
  case arith::AtomicRMWKind::mins:
    val = (w >= 64) ? std::numeric_limits<int64_t>::max()
                    : (int64_t(1) << (w - 1)) - 1; break;
  default:
    val = 0; break;
  }
  return builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(intTy, val));
}

// Checks whether `mem` is a memref defined outside `launchOp`.
// Returns true when the defining op (or the block argument owner) is an
// ancestor of launchOp — meaning the allocation lives in host memory and
// is shared across all GPU blocks.
static bool isExternalMemRef(Value mem, gpu::LaunchOp launchOp) {
  Operation *defOp = mem.getDefiningOp();
  if (!defOp) {
    if (auto arg = dyn_cast<BlockArgument>(mem))
      defOp = arg.getOwner()->getParentOp();
  }
  return defOp && !launchOp->isAncestor(defOp);
}

/// Like isExternalMemRef but traces through memref.subview to the root.
/// Used only for stride-32 launches where the accumulator is a subview
/// of an external alloc.
static bool isExternalMemRefThroughSubview(Value mem, gpu::LaunchOp launchOp) {
  Value current = mem;
  while (auto subview = current.getDefiningOp<memref::SubViewOp>())
    current = subview.getSource();
  return isExternalMemRef(current, launchOp);
}

// Checks whether a loop's store-to-load pattern is scalarisable:
//   - Last store in the body targets the same memref + indices as a load.
//   - The shared indices are loop-invariant (not the induction variable).
static bool matchScalarizePattern(scf::ForOp forOp,
                                   memref::LoadOp &outLoad,
                                   memref::StoreOp &outStore) {
  auto &bodyOps = forOp.getBody()->getOperations();
  if (bodyOps.empty())
    return false;

  // Find the last store in the body.
  memref::StoreOp storeOp;
  for (auto &op : llvm::reverse(bodyOps)) {
    if (auto s = dyn_cast<memref::StoreOp>(op)) { storeOp = s; break; }
  }
  if (!storeOp)
    return false;

  // Find a matching load to the same memref + indices.
  memref::LoadOp loadOp;
  for (auto &op : bodyOps) {
    auto l = dyn_cast<memref::LoadOp>(op);
    if (!l || l.getMemRef() != storeOp.getMemRef()) continue;
    if (l.getIndices().size() != storeOp.getIndices().size()) continue;
    bool indicesMatch = llvm::all_of(
        llvm::zip(l.getIndices(), storeOp.getIndices()),
        [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); });
    if (indicesMatch) { loadOp = l; break; }
  }
  if (!loadOp)
    return false;

  // Indices must be loop-invariant (none equals the induction variable).
  Value loopIV = forOp.getInductionVar();
  for (Value idx : loadOp.getIndices())
    if (idx == loopIV) return false;

  outLoad  = loadOp;
  outStore = storeOp;
  return true;
}

/// Tries to fix the "captured-initial-value" store pattern that bufferization
/// sometimes emits.  The bufferizer can generate:
///
///   [outside gpu.launch]:
///     %captured = memref.load %scratchB[]   // captures the initial zero
///   [inside gpu.launch]:
///     %innerB  = memref.load %scratchB[]
///     %X       = <per-thread computation or load from another scratch>
///     %acc     = arith.addf %X, %innerB     // accumulates into scratchB
///     memref.store %acc, %scratchB[]
///     memref.store %captured, %target[]     // BUG: stores initial zero
///
/// Without the fix, Phase 2 converts the buggy store to
///   atomic_rmw addf 0.0, %target[]          // no-op → target stays 0
/// leading to incorrect results (e.g. mean cross-entropy = 0).
///
/// The fix replaces %captured with the true per-thread contribution:
/// • If %X = memref.load %scratchA[], we find the last value written to
///   %scratchA before that load (store or atomic_rmw) and use that value,
///   so the atomic_rmw that Phase 2 emits sees the actual per-thread datum.
/// • Otherwise %X itself is the contribution.
///
/// Returns the replacement Value, or a null Value when the pattern does not
/// match (no false positives — the original behaviour is preserved).
static Value tryFixCapturedValueStore(Value storeVal, gpu::LaunchOp launchOp) {
  // storeVal must originate OUTSIDE the launch (a captured value).
  Operation *storeValDef = storeVal.getDefiningOp();
  if (!storeValDef || launchOp->isAncestor(storeValDef))
    return {};

  // It must be a memref.load of a buffer defined outside the launch.
  auto capturedLoad = dyn_cast<memref::LoadOp>(storeValDef);
  if (!capturedLoad)
    return {};
  Value scratchB = capturedLoad.getMemRef();

  // Inside the launch, look for:
  //   %innerB = memref.load %scratchB[]
  //   %acc    = arith.addf %X, %innerB  (or arith.addf %innerB, %X)
  //   memref.store %acc, %scratchB[]
  Value contribution;
  launchOp.walk([&](memref::LoadOp innerLoad) -> WalkResult {
    if (contribution)
      return WalkResult::interrupt();
    if (innerLoad.getMemRef() != scratchB)
      return WalkResult::advance();
    // innerLoad reads from scratchB inside the launch.
    for (Operation *user : innerLoad.getResult().getUsers()) {
      auto addfOp = dyn_cast<arith::AddFOp>(user);
      if (!addfOp)
        continue;
      // The other addf operand is the per-thread contribution candidate.
      Value X = (addfOp.getLhs() == innerLoad.getResult()) ? addfOp.getRhs()
                                                           : addfOp.getLhs();
      // Confirm that the addf result is stored back into scratchB.
      bool confirmed = false;
      for (Operation *stUser : addfOp.getResult().getUsers()) {
        auto accStore = dyn_cast<memref::StoreOp>(stUser);
        if (accStore && accStore.getMemRef() == scratchB) {
          confirmed = true;
          break;
        }
      }
      if (!confirmed)
        continue;

      // If X is itself a load from another scratch (scratchA), find the
      // last value written to scratchA before X's load.  This avoids using
      // a stale racy-load value and instead uses the actual stored datum
      // (e.g. the per-row loss value fed into the C-step atomic for scratchA).
      auto innerLoadA = X.getDefiningOp<memref::LoadOp>();
      if (innerLoadA && launchOp->isAncestor(innerLoadA)) {
        Value scratchA = innerLoadA.getMemRef();
        Block *block = innerLoadA->getBlock();
        Value lastWrite;
        for (Operation &op : *block) {
          if (&op == innerLoadA.getOperation())
            break;
          if (auto s = dyn_cast<memref::StoreOp>(&op)) {
            if (s.getMemRef() == scratchA)
              lastWrite = s.getValueToStore();
          } else if (auto rmw = dyn_cast<memref::AtomicRMWOp>(&op)) {
            if (rmw.getMemref() == scratchA)
              lastWrite = rmw.getValue();
          }
        }
        contribution = lastWrite ? lastWrite : X;
      } else {
        contribution = X;
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return contribution;
}

// PERFORMANCE CRITICAL — scf.for accumulator scalarization.
// Converts:
//   scf.for %i = ... {
//     %v = memref.load %acc[]
//     %v2 = arith.addf %v, ...
//     memref.store %v2, %acc[]
//   }
// Into:
//   scf.for %i = ... iter_args(%r = <init>) -> ... {
//     %r2 = arith.addf %r, ...
//     scf.yield %r2
//   }
//   memref.store %final, %acc[]          // (or atomic_rmw for shared)
struct SCFScalarizeAccumulatorPass
    : public PassWrapper<SCFScalarizeAccumulatorPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFScalarizeAccumulatorPass)

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // ALGORITHM STEP 1: Scalarize loop-carried accumulator load/store pairs
    // inside scf.for loops. Process into a snapshot list first to avoid
    // iterator invalidation during transformation.
    SmallVector<scf::ForOp> loopsToTransform;
    funcOp.walk([&](scf::ForOp forOp) {
      memref::LoadOp load; memref::StoreOp store;
      if (matchScalarizePattern(forOp, load, store))
        loopsToTransform.push_back(forOp);
    });
    for (auto forOp : loopsToTransform)
      scalarizeLoop(forOp);

    // ALGORITHM STEP 2: Atomicize cross-block stores inside gpu.launch ops.
    // Stores that target memrefs defined outside the launch must use atomic
    // operations because multiple blocks run concurrently and would race on
    // a plain store.
    SmallVector<gpu::LaunchOp> launches;
    funcOp.walk([&](gpu::LaunchOp launch) { launches.push_back(launch); });
    for (auto launch : launches)
      atomicizeCrossBlockStores(launch);
  }

  // PERFORMANCE CRITICAL — Replace cross-block memref.store with atomic_rmw.
  // For each float store outside a scf.for that targets an external (host)
  // memref, replace it with memref::AtomicRMWOp(addf) and prepend a
  // gpu.memset to zero-init the target buffer before the launch.
  //
  // Two optimizations applied before the default atomic path:
  //   1. Dead-use elimination: if the target memref has no uses after the
  //      launch, the store result is never consumed — erase it, skip memset.
  //   2. Single-thread bypass: if grid=(1,1,1) and block=(1,1,1), there is
  //      exactly 1 thread with zero contention — leave the plain store in
  //      place, skip memset.
  void atomicizeCrossBlockStores(gpu::LaunchOp launchOp) {
    SmallVector<memref::StoreOp> toAtomicize;
    launchOp.walk([&](memref::StoreOp storeOp) {
      // Phase 1 already handled stores inside scf.for; skip those.
      if (storeOp->getParentOfType<scf::ForOp>())
        return;
      auto memTy = dyn_cast<MemRefType>(storeOp.getMemRef().getType());
      if (!memTy || !isa<FloatType>(memTy.getElementType()))
        return;
      if (isExternalMemRef(storeOp.getMemRef(), launchOp))
        toAtomicize.push_back(storeOp);
    });

    if (toAtomicize.empty())
      return;

    // Pre-compute single-thread status once — depends only on launchOp.
    bool singleThread = isSingleThreadLaunch(launchOp);

    // Pre-compute captured-value fixes for all stores BEFORE any erasures.
    // tryFixCapturedValueStore confirms the pattern by checking existing
    // users of arith.addf results; those users (accumulation stores) may be
    // erased during the main loop, so we must snapshot the fixes first.
    llvm::DenseMap<memref::StoreOp, Value> capturedFixes;
    for (auto storeOp : toAtomicize) {
      if (Value fixed =
              tryFixCapturedValueStore(storeOp.getValueToStore(), launchOp))
        capturedFixes[storeOp] = fixed;
    }

    // Track memrefs already zeroed to avoid duplicate gpu.memset ops.
    llvm::DenseSet<Value> zeroed;

    for (auto storeOp : toAtomicize) {
      Value targetMemref = storeOp.getMemRef();
      Value storeValue   = storeOp.getValueToStore();
      auto elemTy = cast<MemRefType>(targetMemref.getType()).getElementType();
      Location loc = storeOp.getLoc();

      // ----------------------------------------------------------------
      // Check 1 (Bug 1): Dead-use elimination.
      // The target memref has no consumers after this launch — the store
      // result is never read. Erase the dead store; no memset needed.
      // ----------------------------------------------------------------
      if (!hasPostLaunchUses(targetMemref, launchOp)) {
        storeOp.erase();
        continue;
      }

      // ----------------------------------------------------------------
      // Check 2 (Bug 2): Single-thread bypass.
      // Exactly 1 thread => zero contention. The original memref.store is
      // already correct. No memset needed (store overwrites the target).
      // ----------------------------------------------------------------
      if (singleThread) {
        // Leave the memref.store untouched.
        continue;
      }

      // ----------------------------------------------------------------
      // Check 3 (Captured-initial-value fix):
      // If storeValue was captured from outside the launch (e.g. a
      // pre-launch host load of a zero-initialised scratch buffer), it
      // carries the initial zero rather than the per-thread contribution.
      // Use the pre-computed fix to substitute the actual per-thread datum
      // so the atomic_rmw below accumulates meaningful data.
      // ----------------------------------------------------------------
      auto fixIt = capturedFixes.find(storeOp);
      if (fixIt != capturedFixes.end())
        storeValue = fixIt->second;

      // ----------------------------------------------------------------
      // Default path: atomicize (existing behavior).
      // Replace the store with an atomic add so all blocks accumulate
      // safely. Insert gpu.memset before the launch to zero-init.
      // ----------------------------------------------------------------
      OpBuilder builder(storeOp);
      builder.create<memref::AtomicRMWOp>(
          loc, arith::AtomicRMWKind::addf, storeValue,
          targetMemref, storeOp.getIndices());
      storeOp.erase();

      // Zero-initialize the target before the launch (once per unique memref).
      if (zeroed.insert(targetMemref).second) {
        OpBuilder pre(launchOp);
        Value zero = pre.create<arith::ConstantOp>(
            loc, pre.getFloatAttr(elemTy, 0.0));
        pre.create<gpu::MemsetOp>(loc, TypeRange{}, ValueRange{},
                                  targetMemref, zero);
      }
    }
  }

  void scalarizeLoop(scf::ForOp forOp) {
    memref::LoadOp loadOp;
    memref::StoreOp storeOp;
    if (!matchScalarizePattern(forOp, loadOp, storeOp))
      return;

    auto &bodyOps = forOp.getBody()->getOperations();

    // Determine if the target memref is shared across threads.
    // Shared accumulators need a final atomic_rmw instead of a plain store.
    //
    // "Shared" means: multiple threads write to the same memref location.
    // This happens in two cases:
    //   (a) The memref is defined outside a forall/launch (cross-block sharing)
    //   (b) The launch has multiple threads per block AND the memref indices
    //       are loop-invariant (all threads in the block access the same slot)
    //       — this is the stride-32 reduction pattern from NovaStrideReduction.
    bool isShared = false;
    if (auto forallOp = forOp->getParentOfType<scf::ForallOp>()) {
      Operation *defOp = storeOp.getMemRef().getDefiningOp();
      if (!defOp)
        if (auto arg = dyn_cast<BlockArgument>(storeOp.getMemRef()))
          defOp = arg.getOwner()->getParentOp();
      if (defOp && !forallOp->isAncestor(defOp))
        isShared = true;
    } else if (auto launchOp = forOp->getParentOfType<gpu::LaunchOp>()) {
      Operation *defOp = storeOp.getMemRef().getDefiningOp();
      if (!defOp)
        if (auto arg = dyn_cast<BlockArgument>(storeOp.getMemRef()))
          defOp = arg.getOwner()->getParentOp();
      if (defOp && !launchOp->isAncestor(defOp))
        isShared = true;

      // Case (b): stride-32 reduction (NovaStrideReduction).
      // After the stride pass, blockSizeX=32 and all 32 threads stride
      // through the SAME reduction loop, accumulating into the SAME
      // memref location. Detect: blockSizeX=32 AND the scf.for lower
      // bound is gpu.thread_id (signature of the stride rewrite).
      if (!isShared) {
        auto bx = getConstantIndex(launchOp.getBlockSizeX());
        if (bx && *bx == 32 &&
            forOp.getLowerBound().getDefiningOp<gpu::ThreadIdOp>())
          isShared = true;
      }
    }

    // Pick the atomic kind from the op that feeds the store (the actual
    // reduction combiner), NOT just any arith op in the body.  For fused
    // elementwise+reduction bodies like relu+sum the body contains both
    // maxnumf (relu) and addf (sum); picking the first one gives the wrong
    // atomic kind.
    arith::AtomicRMWKind atomicKind = arith::AtomicRMWKind::addf;
    if (Operation *storeValDef = storeOp.getValueToStore().getDefiningOp()) {
      // Float reductions — NaN-returning variants
      if (isa<arith::AddFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::addf;
      else if (isa<arith::MulFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::mulf;
      else if (isa<arith::MaximumFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::maximumf;
      else if (isa<arith::MinimumFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::minimumf;
      // Float reductions — NaN-propagating variants
      else if (isa<arith::MaxNumFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::maxnumf;
      else if (isa<arith::MinNumFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::minnumf;
      // Integer reductions
      else if (isa<arith::AddIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::addi;
      else if (isa<arith::MulIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::muli;
      else if (isa<arith::MaxSIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::maxs;
      else if (isa<arith::MaxUIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::maxu;
      else if (isa<arith::MinSIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::mins;
      else if (isa<arith::MinUIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::minu;
      else if (isa<arith::OrIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::ori;
      else if (isa<arith::AndIOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::andi;
    }

    // Look for an initialisation store immediately before the loop.
    memref::StoreOp initStore;
    if (!isShared) {
      Operation *prevOp = forOp->getPrevNode();
      while (prevOp && prevOp->getBlock() == forOp->getBlock()) {
        if (auto s = dyn_cast<memref::StoreOp>(prevOp)) {
          if (s.getMemRef() == loadOp.getMemRef() &&
              s.getIndices().size() == loadOp.getIndices().size()) {
            bool match = llvm::all_of(
                llvm::zip(s.getIndices(), loadOp.getIndices()),
                [](auto p) { return std::get<0>(p) == std::get<1>(p); });
            if (match) { initStore = s; break; }
          }
        }
        prevOp = prevOp->getPrevNode();
      }
    }

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value initVal;

    if (isShared) {
      auto elemTy = cast<MemRefType>(storeOp.getMemRef().getType()).getElementType();
      initVal = buildIdentityValue(builder, loc, atomicKind, elemTy);
    } else if (initStore) {
      initVal = initStore.getValueToStore();
    } else {
      initVal = builder.create<memref::LoadOp>(
          loc, loadOp.getMemRef(), loadOp.getIndices());
    }

    // Build the replacement scf.for with the scalar accumulator as iter_arg.
    SmallVector<Value> iterArgs = forOp.getInitArgs();
    iterArgs.push_back(initVal);
    auto newForOp = builder.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        iterArgs);

    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
      mapping.map(forOp.getRegionIterArg(i), newForOp.getRegionIterArg(i));

    // Map the old load result to the new iter_arg (register accumulator).
    Value newIterArg = newForOp.getRegionIterArgs().back();
    mapping.map(loadOp.getResult(), newIterArg);

    builder.setInsertionPointToStart(newForOp.getBody());

    Value yieldValue;
    for (auto &op : bodyOps) {
      if (&op == loadOp.getOperation()) {
        // Skip — replaced by iter_arg.
      } else if (&op == storeOp.getOperation()) {
        // Capture the value to store so we can yield it.
        yieldValue = mapping.lookupOrDefault(storeOp.getValueToStore());
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        SmallVector<Value> yields;
        for (Value v : yieldOp.getOperands())
          yields.push_back(mapping.lookupOrDefault(v));
        if (yieldValue)
          yields.push_back(yieldValue);
        builder.create<scf::YieldOp>(loc, yields);
      } else {
        builder.clone(op, mapping);
      }
    }

    // Write the final accumulated value back after the loop.
    builder.setInsertionPointAfter(newForOp);
    if (isShared) {
      // Atomic write-back for shared accumulators to avoid inter-block races.
      builder.create<memref::AtomicRMWOp>(
          loc, atomicKind, newForOp.getResults().back(),
          storeOp.getMemRef(), storeOp.getIndices());
    } else {
      builder.create<memref::StoreOp>(
          loc, newForOp.getResults().back(),
          storeOp.getMemRef(), storeOp.getIndices());
    }

    // Replace old loop results and erase.
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    forOp.erase();

    if (initStore)
      initStore.erase();
  }

  StringRef getArgument() const final { return "scf-scalarize-accumulator"; }
  StringRef getDescription() const final {
    return "Convert buffer-based scf.for loop accumulation to register-based "
           "iter_args, reducing memory traffic for loop-carried scalars";
  }
};

} // namespace

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createSCFScalarizeAccumulatorPass() {
  return std::make_unique<SCFScalarizeAccumulatorPass>();
}

} // namespace nova
} // namespace mlir
