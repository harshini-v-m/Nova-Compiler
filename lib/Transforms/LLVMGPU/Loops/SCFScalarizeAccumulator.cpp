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

using namespace mlir;

namespace {

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
for (Value idx : loadOp.getIndices()) {
    if (idx == loopIV) return false;
    // NEW: also reject indices whose defining op is inside the loop
    if (Operation *defOp = idx.getDefiningOp())
        if (forOp->isProperAncestor(defOp)) return false;
}

  outLoad  = loadOp;
  outStore = storeOp;
  return true;
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

    // Track memrefs already zeroed to avoid duplicate gpu.memset ops.
    llvm::DenseSet<Value> zeroed;

    for (auto storeOp : toAtomicize) {
      Value targetMemref = storeOp.getMemRef();
      Value storeValue   = storeOp.getValueToStore();
      auto elemTy = cast<MemRefType>(targetMemref.getType()).getElementType();
      Location loc = storeOp.getLoc();

      // Replace the store with an atomic add so all blocks accumulate safely.
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

    // Determine if the target memref is shared (cross-forall or cross-launch).
    // Shared accumulators need a final atomic_rmw instead of a plain store.
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
    }

    // Pick the atomic kind from the op that feeds the store (the actual
    // reduction combiner), NOT just any arith op in the body.  For fused
    // elementwise+reduction bodies like relu+sum the body contains both
    // maxnumf (relu) and addf (sum); picking the first one gives the wrong
    // atomic kind.
    arith::AtomicRMWKind atomicKind = arith::AtomicRMWKind::addf;
    if (Operation *storeValDef = storeOp.getValueToStore().getDefiningOp()) {
      if (isa<arith::AddFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::addf;
      else if (isa<arith::MulFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::mulf;
      else if (isa<arith::MaxNumFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::maximumf;
      else if (isa<arith::MinNumFOp>(storeValDef))
        atomicKind = arith::AtomicRMWKind::minimumf;
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
      // For shared accumulators, choose the identity value for the operation.
      auto elemTy = cast<MemRefType>(storeOp.getMemRef().getType()).getElementType();
      double identity = (atomicKind == arith::AtomicRMWKind::mulf) ? 1.0 : 0.0;
      initVal = builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(elemTy, identity));
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
