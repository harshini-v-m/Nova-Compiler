//===- NovaWarpShuffleReduction.cpp - Warp butterfly shuffle for reductions ===//
//
// Replaces the "double-atomic" reduction pattern produced by
// SCFScalarizeAccumulator with a warp butterfly shuffle reduction.
//
// The buggy pattern (all N threads execute):
//   (A) memref.atomic_rmw addf %partial, %scratch[]   — correct total in scratch
//   (B) %val = memref.load %scratch[]                  — all threads read total
//   (C) memref.atomic_rmw addf %val, %output[]         — N× the correct result
//
// Replaced with:
//   log2(N) rounds of gpu.shuffle xor to combine partials in registers
//   Thread 0 writes the final result to output (plain store or atomic_rmw)
//
// When (A) is nested inside scf.for loops (from reduction tiling), the triad
// spans two different blocks.  In that case we use barrier + thread-0 guard:
// keep (A) inside the loop, barrier after the outermost loop, thread-0 reads
// scratch and writes to output.
//
// Constraints:
//   - Single warp only (thread count ≤ 32, power of 2)
//   - blockSizeY == blockSizeZ == 1
//   - f32 element type
//
// Shuffle utilities (emitTypedShuffleXOR, buildReductionCombineOp,
// buildShuffleReductionTree, buildReductionIdentity) are defined in
// NovaVectorReduction.cpp and declared in NovaVectorReduction.h.
// This file calls them via the mlir::nova:: namespace.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

#define DEBUG_TYPE "nova-warp-shuffle-reduction"

using namespace mlir;

namespace {

#include <optional>

static std::optional<gpu::AllReduceOperation> mapToGPUAllReduce(arith::AtomicRMWKind kind) {
  switch (kind) {
    case arith::AtomicRMWKind::addf:
    case arith::AtomicRMWKind::addi:   return gpu::AllReduceOperation::ADD;
    case arith::AtomicRMWKind::mulf:
    case arith::AtomicRMWKind::muli:   return gpu::AllReduceOperation::MUL;
    case arith::AtomicRMWKind::maximumf: return gpu::AllReduceOperation::MAXIMUMF;
    case arith::AtomicRMWKind::minimumf: return gpu::AllReduceOperation::MINIMUMF;
    case arith::AtomicRMWKind::maxnumf:return gpu::AllReduceOperation::MAXNUMF;
    case arith::AtomicRMWKind::minnumf:return gpu::AllReduceOperation::MINNUMF;
    case arith::AtomicRMWKind::maxs:   return gpu::AllReduceOperation::MAXSI;
    case arith::AtomicRMWKind::maxu:   return gpu::AllReduceOperation::MAXUI;
    case arith::AtomicRMWKind::mins:   return gpu::AllReduceOperation::MINSI;
    case arith::AtomicRMWKind::minu:   return gpu::AllReduceOperation::MINUI;
    case arith::AtomicRMWKind::ori:    return gpu::AllReduceOperation::OR;
    case arith::AtomicRMWKind::andi:   return gpu::AllReduceOperation::AND;
    default: return std::nullopt;
  }
}

static Value buildReductionIdentity(OpBuilder &builder, Location loc,
                                    arith::AtomicRMWKind kind, Type elemTy) {
  if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
    double val = 0.0;
    switch (kind) {
    case arith::AtomicRMWKind::mulf: val = 1.0; break;
    case arith::AtomicRMWKind::maximumf:
    case arith::AtomicRMWKind::maxnumf: val = -std::numeric_limits<double>::infinity(); break;
    case arith::AtomicRMWKind::minimumf:
    case arith::AtomicRMWKind::minnumf: val = std::numeric_limits<double>::infinity(); break;
    default: val = 0.0; break;
    }
    return builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(floatTy, val));
  }
  auto intTy = cast<IntegerType>(elemTy);
  unsigned w = intTy.getWidth();
  int64_t val = 0;
  switch (kind) {
  case arith::AtomicRMWKind::muli: val = 1; break;
  case arith::AtomicRMWKind::andi:
  case arith::AtomicRMWKind::minu: val = -1; break;
  case arith::AtomicRMWKind::maxs: val = (w >= 64) ? std::numeric_limits<int64_t>::min() : -(int64_t(1) << (w - 1)); break;
  case arith::AtomicRMWKind::mins: val = (w >= 64) ? std::numeric_limits<int64_t>::max() : (int64_t(1) << (w - 1)) - 1; break;
  default: val = 0; break;
  }
  return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(intTy, val));
}

/// Emits nested scf.for loops to fill `target` memref with `identity` value.
static void emitZeroFillLoop(OpBuilder &builder, Location loc, Value target, Value identity) {
  auto memTy = cast<MemRefType>(target.getType());
  int64_t rank = memTy.getRank();

  if (rank == 0) {
    builder.create<memref::StoreOp>(loc, identity, target);
    return;
  }

  SmallVector<Value> ivs;
  SmallVector<scf::ForOp> forOps;

  for (int i = 0; i < rank; ++i) {
    Value lower = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value upper;
    if (memTy.isDynamicDim(i)) {
      upper = builder.create<memref::DimOp>(loc, target, i);
    } else {
      upper = builder.create<arith::ConstantIndexOp>(loc, memTy.getDimSize(i));
    }
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto forOp = builder.create<scf::ForOp>(loc, lower, upper, step);
    ivs.push_back(forOp.getInductionVar());
    builder.setInsertionPointToStart(forOp.getBody());
    forOps.push_back(forOp);
  }

  builder.create<memref::StoreOp>(loc, identity, target, ivs);

  if (!forOps.empty())
    builder.setInsertionPointAfter(forOps[0]);
}

static Value buildReductionCombineOp(OpBuilder &builder, Location loc,
                                     arith::AtomicRMWKind kind, Value lhs, Value rhs) {
  switch (kind) {
  case arith::AtomicRMWKind::addf: return builder.create<arith::AddFOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::mulf: return builder.create<arith::MulFOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::maximumf: return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::minimumf: return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::maxnumf: return builder.create<arith::MaxNumFOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::minnumf: return builder.create<arith::MinNumFOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::addi: return builder.create<arith::AddIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::muli: return builder.create<arith::MulIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::maxs: return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::maxu: return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::mins: return builder.create<arith::MinSIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::minu: return builder.create<arith::MinUIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::ori: return builder.create<arith::OrIOp>(loc, lhs, rhs);
  case arith::AtomicRMWKind::andi: return builder.create<arith::AndIOp>(loc, lhs, rhs);
  default: llvm_unreachable("unsupported reduction kind");
  }
}

/// Holds a matched double-atomic triad inside a gpu.launch body.
struct DoubleAtomicTriad {
  memref::AtomicRMWOp atomicToScratch; // (A)
  memref::LoadOp loadFromScratch;      // (B)
  memref::AtomicRMWOp atomicToOutput;  // (C)
  Value partialValue;                  // register value feeding (A)
  Value scratchMemref;                 // locally-allocated scalar
  Value outputMemref;                  // externally-defined output
  SmallVector<Value> outputIndices;
  arith::AtomicRMWKind kind;
  /// Non-null when (A) is inside nested scf.for loops.  Points to the
  /// outermost scf.for ancestor of (A) that lives in (B)/(C)'s block.
  Operation *outermostContainingLoop = nullptr;
};

/// Trace `val` through the vectorize-introduced round-trip:
///   memref.load %scratch  ←  vector.store (vector.insert %scalar []) %scratch
/// Returns the original f32 scalar that was inserted, or `val` if the chain
/// is not recognized.  Used to recover the true per-thread partial from the
/// store-load idiom the vectorize pass leaves behind.
static Value resolveScalarPartial(Value val) {
  auto loadOp = val.getDefiningOp<memref::LoadOp>();
  if (!loadOp)
    return val;
  Value scratch = loadOp.getMemRef();
  Block *block = loadOp->getBlock();

  // Find the latest vector::StoreOp to the same scratch that precedes the load.
  vector::StoreOp latestVStore;
  for (Operation *user : scratch.getUsers()) {
    auto s = dyn_cast<vector::StoreOp>(user);
    if (!s || s->getBlock() != block || !s->isBeforeInBlock(loadOp))
      continue;
    if (!latestVStore || latestVStore->isBeforeInBlock(s))
      latestVStore = s;
  }
  if (!latestVStore)
    return val;

  // Stored value should be: vector.insert %scalar [] : f32 into vector<f32>
  // (i.e. a scalar wrapped into a 0-d vector — the pattern the vectorize pass emits).
  auto insertOp = latestVStore.getValueToStore().getDefiningOp<vector::InsertOp>();
  if (!insertOp || !insertOp.getStaticPosition().empty())
    return val;

  return insertOp.getOperand(0); // the actual f32 per-thread partial (source operand)
}

/// Returns true if `mem` is defined outside the gpu.launch region.
static bool isExternalMemRef(Value mem, gpu::LaunchOp launchOp) {
  Operation *defOp = mem.getDefiningOp();
  if (!defOp) {
    if (auto arg = dyn_cast<BlockArgument>(mem))
      defOp = arg.getOwner()->getParentOp();
  }
  return defOp && !launchOp->isAncestor(defOp);
}

/// Like isExternalMemRef but also traces through memref.subview ops to
/// find the root source. Used for stride-32 pattern where the atomic target
/// is a subview of an external alloc.
static bool isExternalMemRefThroughSubview(Value mem, gpu::LaunchOp launchOp) {
  Value current = mem;
  while (auto subview = current.getDefiningOp<memref::SubViewOp>())
    current = subview.getSource();
  return isExternalMemRef(current, launchOp);
}

/// Extract constant integer from a Value, or return std::nullopt.
static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return std::nullopt;
}

/// Check if all grid dimensions are constant 1 (single-block launch).
static bool isSingleBlockLaunch(gpu::LaunchOp launch) {
  auto gx = getConstantIndex(launch.getGridSizeX());
  auto gy = getConstantIndex(launch.getGridSizeY());
  auto gz = getConstantIndex(launch.getGridSizeZ());
  return gx && gy && gz && *gx == 1 && *gy == 1 && *gz == 1;
}

/// Check if grid=(1,1,1) AND block=(1,1,1) — exactly 1 thread total.
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

// buildIdentityValue → now buildReductionIdentity (NovaVectorReduction.h)

/// Return the innermost gpu.LaunchOp ancestor of `op`, or null.
static gpu::LaunchOp getEnclosingLaunch(Operation *op) {
  Operation *cursor = op->getParentOp();
  while (cursor) {
    if (auto l = dyn_cast<gpu::LaunchOp>(cursor))
      return l;
    cursor = cursor->getParentOp();
  }
  return nullptr;
}

/// After eliminating a double-atomic triad, clean up the scratch memref that
/// sourceLaunch no longer writes to:
///   1. Erase any gpu.memset emitted by SCFScalarizeAccumulator for this scratch.
///   2. Erase the single-thread init launch (if any) whose sole purpose was
///      writing 0 into scratch before the now-eliminated multi-thread kernel.
///
/// Bails out silently if the scratch has any remaining non-trivial uses.
static void tryCleanupScratch(Value scratchMemref, gpu::LaunchOp sourceLaunch) {
  SmallVector<gpu::MemsetOp> memsets;
  SmallVector<gpu::LaunchOp> initLaunches;

  for (Operation *user : scratchMemref.getUsers()) {
    // Uses inside sourceLaunch were eliminated by the triad erasure.
    if (sourceLaunch->isAncestor(user))
      continue;
    // Deallocs free memory but don't read — not a reason to keep the scratch.
    if (isa<memref::DeallocOp>(user))
      continue;
    // gpu.memset initializes the scratch — candidate for erasure.
    if (auto m = dyn_cast<gpu::MemsetOp>(user)) {
      memsets.push_back(m);
      continue;
    }
    // If this user is inside a single-thread launch that only writes to scratch,
    // it is the SCFScalarize init kernel — queue it for erasure.
    gpu::LaunchOp enclosing = getEnclosingLaunch(user);
    if (!enclosing || !isSingleThreadLaunch(enclosing))
      return; // non-trivial use — bail out conservatively

    bool onlyInitScratch = true;
    enclosing.getBody().walk([&](Operation *op) {
      if (isa<gpu::TerminatorOp>(op)) return;
      if (auto s = dyn_cast<memref::StoreOp>(op)) {
        if (s.getMemRef() != scratchMemref) onlyInitScratch = false;
        return;
      }
      if (auto a = dyn_cast<memref::AtomicRMWOp>(op)) {
        if (a.getMemref() != scratchMemref) onlyInitScratch = false;
        return;
      }
      if (isa<arith::ConstantOp>(op)) return; // harmless inline constant
      onlyInitScratch = false;
    });

    if (!onlyInitScratch)
      return;
    initLaunches.push_back(enclosing);
  }

  // All remaining uses are dead — safe to erase.
  for (auto m : memsets) m->erase();
  for (auto l : initLaunches) l->erase();
}

// buildCombineOp → now buildReductionCombineOp (NovaVectorReduction.h)

/// Find all double-atomic triads in a gpu.launch body.
///
/// Handles both the simple case (A, B, C in the same block) and the nested
/// case where (A) is inside scf.for loops while (B)/(C) are in the launch
/// body block.
static SmallVector<DoubleAtomicTriad>
findTriads(gpu::LaunchOp launchOp) {
  SmallVector<DoubleAtomicTriad> triads;

  // Collect all atomic_rmw ops in the launch body.
  SmallVector<memref::AtomicRMWOp> atomics;
  launchOp.getBody().walk([&](memref::AtomicRMWOp op) {
    atomics.push_back(op);
  });

  for (size_t i = 0; i < atomics.size(); ++i) {
    auto atomicA = atomics[i];
    Value intermediate = atomicA.getMemref();

    auto memType = cast<MemRefType>(intermediate.getType());

    // Only support float types — integer atomic triads (e.g. scatter_add
    // index accumulators) must not be warp-shuffled.
    if (!isa<FloatType>(memType.getElementType()))
      continue;

    // (A) must target an external scalar memref.
    if (!isExternalMemRef(intermediate, launchOp))
      continue;

    // Must be scalar (rank 0).
    if (memType.getRank() != 0)
      continue;

    // Look for a load from the same memref after this atomic.
    for (auto &use : intermediate.getUses()) {
      auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner());
      if (!loadOp)
        continue;

      // (A) and (B) may be in the same block (simple case) or different
      // blocks when (A) is inside nested scf.for loops.
      Operation *ancestorLoop = nullptr;
      if (atomicA->getBlock() == loadOp->getBlock()) {
        // Simple case: same block, verify ordering.
        if (!atomicA->isBeforeInBlock(loadOp))
          continue;
      } else {
        // Nested case: walk up from atomicA to find the outermost
        // ancestor in loadOp's block.
        Operation *cursor = atomicA->getParentOp();
        while (cursor && cursor->getBlock() != loadOp->getBlock())
          cursor = cursor->getParentOp();
        if (!cursor || !isa<scf::ForOp>(cursor))
          continue;
        if (!cursor->isBeforeInBlock(loadOp))
          continue;
        ancestorLoop = cursor;
      }

      // (B) found. Now look for an atomic_rmw that uses the load result
      // and targets a DIFFERENT external memref.
      for (auto &loadUse : loadOp.getResult().getUses()) {
        auto atomicC = dyn_cast<memref::AtomicRMWOp>(loadUse.getOwner());
        if (!atomicC)
          continue;

        Value output = atomicC.getMemref();
        if (!isExternalMemRef(output, launchOp))
          continue;

        // Output must be different from intermediate.
        if (output == intermediate)
          continue;

        // Verify matching reduction kind.
        if (atomicA.getKind() != atomicC.getKind())
          continue;

        // Verify the value fed to atomicC is the load result.
        if (atomicC.getValue() != loadOp.getResult())
          continue;

        DoubleAtomicTriad triad;
        triad.atomicToScratch = atomicA;
        triad.loadFromScratch = loadOp;
        triad.atomicToOutput = atomicC;
        triad.partialValue = atomicA.getValue();
        triad.scratchMemref = intermediate;
        triad.outputMemref = output;
        triad.outputIndices.assign(atomicC.getIndices().begin(),
                                   atomicC.getIndices().end());
        triad.kind = atomicA.getKind();
        triad.outermostContainingLoop = ancestorLoop;
        triads.push_back(triad);
      }
    }
  }
  return triads;
}

/// Build a butterfly shuffle reduction tree.
/// Produces log2(width) rounds of gpu.shuffle xor + combine.
static Value buildShuffleTree(OpBuilder &builder, Location loc,
                              Value partial, int64_t width,
                              arith::AtomicRMWKind kind) {
  Value acc = partial;
  Value widthVal = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(width));

  for (int64_t offset = width / 2; offset >= 1; offset /= 2) {
    Value offsetVal = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(offset));

    // gpu.shuffle xor returns (shuffled_value, valid_bit)
    auto shuffleOp = builder.create<gpu::ShuffleOp>(
        loc, acc, offsetVal, widthVal, gpu::ShuffleMode::XOR);
    Value shuffled = shuffleOp.getShuffleResult();

    acc = buildReductionCombineOp(builder, loc, kind, acc, shuffled);
  }
  return acc;
}

/// Return true if `partial` is the result of a scf.for whose body uses
/// cross-thread atomic_rmw **of the same reduction kind** to external (shared)
/// memory.  In that case every thread already holds the full reduction result,
/// and shuffling would over-count by N×.
///
/// We require the inner atomic kind to match `triadKind` because a mismatch
/// (e.g., maximumf inside the loop with an addf triad) means the inner op is
/// from a fused elementwise op (like relu), not the reduction combiner — those
/// do NOT make the partial pre-reduced.
static bool isPreReduced(Value partial, gpu::LaunchOp launchOp,
                          arith::AtomicRMWKind triadKind) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(partial.getDefiningOp());
  if (!forOp)
    return false;
  bool found = false;
  forOp.getBody()->walk([&](memref::AtomicRMWOp innerAtomic) {
    if (innerAtomic.getKind() == triadKind &&
        isExternalMemRef(innerAtomic.getMemref(), launchOp))
      found = true;
  });
  return found;
}

/// Privatize the shared scratch used inside a pre-reduced scf.for.
///
/// The shared external scratch causes all threads to see each other's
/// atomic_rmw writes, making every thread hold the full sum instead of
/// a per-thread partial.  This function:
///   1. Creates a memref.alloca (per-thread private) inside the launch body
///   2. Replaces atomic_rmw addf %val, %shared[] with:
///        %old = load %private[]; %new = addf %val, %old; store %new, %private[]
///   3. Replaces all store/load of the shared scratch inside the for-body
///      with the private alloca
///
/// After privatization, each thread accumulates only its own partials.
/// The triad's partialValue becomes a true per-thread partial, and
/// isPreReduced() returns false — enabling the shuffle path.
static bool privatizeScratch(scf::ForOp forOp, Value sharedScratch,
                              gpu::LaunchOp launchOp,
                              arith::AtomicRMWKind triadKind) {
  // Collect all ops inside the for-body that use the shared scratch.
  SmallVector<memref::AtomicRMWOp> innerAtomics;
  SmallVector<memref::StoreOp> innerStores;
  SmallVector<memref::LoadOp> innerLoads;

  forOp.getBody()->walk([&](Operation *op) {
    if (auto atomicOp = dyn_cast<memref::AtomicRMWOp>(op)) {
      if (atomicOp.getMemref() == sharedScratch &&
          atomicOp.getKind() == triadKind)
        innerAtomics.push_back(atomicOp);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      if (storeOp.getMemRef() == sharedScratch)
        innerStores.push_back(storeOp);
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      if (loadOp.getMemRef() == sharedScratch)
        innerLoads.push_back(loadOp);
    }
  });

  if (innerAtomics.empty())
    return false;

  // Create a per-thread private scratch at the start of the launch body.
  auto memType = cast<MemRefType>(sharedScratch.getType());
  auto privateType = MemRefType::get(memType.getShape(),
                                      memType.getElementType());
  Block &launchBody = launchOp.getBody().front();
  OpBuilder allocBuilder(&launchBody, launchBody.begin());
  Value privateScratch = allocBuilder.create<memref::AllocaOp>(
      forOp.getLoc(), privateType);

  // Initialize the private scratch with the identity value for the reduction
  // (e.g., 0.0 for addf). Otherwise the first load from local memory/registers
  // will contain garbage values, leading to NaN.
  Value identity = buildReductionIdentity(allocBuilder, forOp.getLoc(),
                                          triadKind, memType.getElementType());
  allocBuilder.create<memref::StoreOp>(forOp.getLoc(), identity,
                                        privateScratch, ValueRange{});

  // Replace all stores to shared scratch → stores to private scratch.
  for (auto storeOp : innerStores)
    storeOp.getMemrefMutable().assign(privateScratch);

  // Replace all loads from shared scratch → loads from private scratch.
  for (auto loadOp : innerLoads)
    loadOp.getMemrefMutable().assign(privateScratch);

  // Replace atomic_rmw addf %val, %shared[] with:
  //   %old = load %private[]; %new = addf %val, %old; store %new, %private[]
  for (auto atomicOp : innerAtomics) {
    OpBuilder b(atomicOp);
    Location loc = atomicOp.getLoc();
    Value old = b.create<memref::LoadOp>(loc, privateScratch, ValueRange{});
    Value combined = buildReductionCombineOp(b, loc, triadKind,
                                     atomicOp.getValue(), old);
    b.create<memref::StoreOp>(loc, combined, privateScratch, ValueRange{});
    // The atomic_rmw result might be used — replace with the old value.
    atomicOp.getResult().replaceAllUsesWith(old);
    atomicOp.erase();
  }

  LLVM_DEBUG(llvm::dbgs() << "[warp-shuffle] Privatized scratch: "
                           << innerAtomics.size() << " atomics, "
                           << innerStores.size() << " stores, "
                           << innerLoads.size() << " loads\n");
  return true;
}

/// Helper to zero-initialize a shared memory memref once in a block.
static void initializeSharedScratch(OpBuilder &builder, Location loc,
                                    Value scratch, arith::AtomicRMWKind kind,
                                    gpu::LaunchOp launch) {
  // Use the entry block of the launch to insert the guard.
  Block &entryBlock = launch.getBody().front();
  OpBuilder init(&entryBlock.front());
  Operation *scratchDef = scratch.getDefiningOp();
  if (scratchDef && scratchDef->getBlock() == &entryBlock)
    init.setInsertionPointAfter(scratchDef);

  Location iloc = launch.getLoc();

  Value tidX = init.create<gpu::ThreadIdOp>(iloc, gpu::Dimension::x);
  Value tidY = init.create<gpu::ThreadIdOp>(iloc, gpu::Dimension::y);
  Value tidZ = init.create<gpu::ThreadIdOp>(iloc, gpu::Dimension::z);
  Value c0 = init.create<arith::ConstantIndexOp>(iloc, 0);
  Value isT0X = init.create<arith::CmpIOp>(iloc, arith::CmpIPredicate::eq, tidX, c0);
  Value isT0Y = init.create<arith::CmpIOp>(iloc, arith::CmpIPredicate::eq, tidY, c0);
  Value isT0Z = init.create<arith::CmpIOp>(iloc, arith::CmpIPredicate::eq, tidZ, c0);
  Value isThread0 = init.create<arith::AndIOp>(iloc, isT0X, isT0Y);
  isThread0 = init.create<arith::AndIOp>(iloc, isThread0, isT0Z);

  auto memTy = cast<MemRefType>(scratch.getType());
  init.create<scf::IfOp>(iloc, isThread0, [&](OpBuilder &thenB, Location thenL) {
    Value zero = buildReductionIdentity(thenB, thenL, kind, memTy.getElementType());
    emitZeroFillLoop(thenB, thenL, scratch, zero);
    thenB.create<scf::YieldOp>(thenL);
  });
  init.create<gpu::BarrierOp>(iloc);
}

/// Emit the thread-0-guarded write (barrier + tid==0 → load scratch → write
static void emitBarrierAndGuardedWrite(OpBuilder &builder, Location loc,
                                       const DoubleAtomicTriad &triad,
                                       bool singleBlock) {
  builder.create<gpu::BarrierOp>(loc);

  Value tidX = builder.create<gpu::ThreadIdOp>(
      loc, builder.getIndexType(), gpu::Dimension::x);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value isThread0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, tidX, zero);

  builder.create<scf::IfOp>(
      loc, isThread0,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        // scratch is always rank-0 (scalar), so indices are empty.
        Value val = thenBuilder.create<memref::LoadOp>(
            thenLoc, triad.scratchMemref, ValueRange{});
        if (singleBlock) {
          thenBuilder.create<memref::StoreOp>(
              thenLoc, val, triad.outputMemref,
              triad.outputIndices);
        } else {
          thenBuilder.create<memref::AtomicRMWOp>(
              thenLoc, triad.kind, val, triad.outputMemref,
              triad.outputIndices);
        }
        thenBuilder.create<scf::YieldOp>(thenLoc);
      });
}

struct NovaWarpShuffleReductionPass
    : public PassWrapper<NovaWarpShuffleReductionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaWarpShuffleReductionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect, scf::SCFDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<gpu::LaunchOp> launches;
    funcOp.walk([&](gpu::LaunchOp op) { launches.push_back(op); });

    for (auto launchOp : launches) {
      // Extract thread count. Must be constant and 1D.
      auto bsx = getConstantIndex(launchOp.getBlockSizeX());
      auto bsy = getConstantIndex(launchOp.getBlockSizeY());
      auto bsz = getConstantIndex(launchOp.getBlockSizeZ());
      if (!bsx || !bsy || !bsz)
        continue;
      if (*bsy != 1 || *bsz != 1)
        continue;
      int64_t numThreads = *bsx;
      if (numThreads <= 1)
        continue;

      LLVM_DEBUG(llvm::dbgs() << "[warp-shuffle] Checking launch with "
                               << numThreads << " threads\n");

      // Determine reduction strategy:
      //   ≤ 32 threads, power of 2 → butterfly shuffle (all in registers)
      //   > 32 threads             → keep first atomic + barrier + tid==0 guard
      bool useShuffle = (numThreads <= 32 && llvm::isPowerOf2_64(numThreads));
      bool singleBlock = isSingleBlockLaunch(launchOp);

      // ── Step 0: Privatize shared scratch in pre-reduced scf.for loops.
      // This converts cross-thread atomics to per-thread register ops,
      // turning pre-reduced partials into true per-thread partials so the
      // shuffle path can fire.
      if (useShuffle) {
        // Collect candidate scf.for ops whose partialValue feeds an
        // external atomic_rmw (the triad's (A)).  We detect these by
        // walking for atomic_rmw ops to external rank-0 memrefs, then
        // checking if the value comes from a scf.for with inner atomics.
        SmallVector<memref::AtomicRMWOp> outerAtomics;
        launchOp.getBody().walk([&](memref::AtomicRMWOp op) {
          if (isExternalMemRef(op.getMemref(), launchOp) &&
              cast<MemRefType>(op.getMemref().getType()).getRank() == 0)
            outerAtomics.push_back(op);
        });
        for (auto outerAtomic : outerAtomics) {
          Value partial = outerAtomic.getValue();
          auto forOp =
              dyn_cast_or_null<scf::ForOp>(partial.getDefiningOp());
          if (!forOp)
            continue;
          // Check if this for-body has inner atomics to a DIFFERENT
          // external scratch (the inner shared scratch pattern).
          SmallVector<Value> innerScratchCandidates;
          forOp.getBody()->walk([&](memref::AtomicRMWOp innerAtomic) {
            Value mem = innerAtomic.getMemref();
            if (mem != outerAtomic.getMemref() &&
                isExternalMemRef(mem, launchOp) &&
                cast<MemRefType>(mem.getType()).getRank() == 0 &&
                innerAtomic.getKind() == outerAtomic.getKind()) {
              innerScratchCandidates.push_back(mem);
            }
          });
          // Privatize each inner shared scratch.
          for (Value scratch : innerScratchCandidates)
            privatizeScratch(forOp, scratch, launchOp,
                             outerAtomic.getKind());
        }
      }

      auto triads = findTriads(launchOp);

      // ── Single-atomic pattern (from tiling / NovaStrideReduction) ──
      // Each thread has a per-thread partial written via atomic_rmw to an
      // external memref — N threads atomic-add their partials → N× overcounting.
      // Replace with: gpu.all_reduce to combine all partials, thread-0 writes.
      // Works for any thread count (not restricted to powers-of-2 ≤ 32).
      if (triads.empty()) {
        SmallVector<memref::AtomicRMWOp> strideAtomics;
        launchOp.getBody().walk([&](memref::AtomicRMWOp atomicOp) {
          if (!isExternalMemRefThroughSubview(atomicOp.getMemref(), launchOp))
            return;
          auto memTy = cast<MemRefType>(atomicOp.getMemref().getType());
          if (!isa<FloatType>(memTy.getElementType()))
            return;
          // Trace through the vectorize-introduced store-load round-trip:
          //   memref.load ← vector.store ← vector.insert %scalar []
          Value partialVal = resolveScalarPartial(atomicOp.getValue());
          if (!partialVal.getDefiningOp<scf::ForOp>() &&
              !partialVal.getDefiningOp<vector::ReductionOp>() &&
              !partialVal.getDefiningOp<vector::ExtractOp>())
            return;
          strideAtomics.push_back(atomicOp);
        });

        for (auto atomicOp : strideAtomics) {
          Location loc = atomicOp.getLoc();
          OpBuilder builder(atomicOp);

          // Recover the true per-thread partial (bypass store-load round-trip).
          Value partial = resolveScalarPartial(atomicOp.getValue());

          // Cross-thread reduction via gpu.all_reduce — simpler than a manual
          // butterfly shuffle and works for any thread count.
          auto gpuReduceKind = mapToGPUAllReduce(atomicOp.getKind());
          auto reduceAttr = gpu::AllReduceOperationAttr::get(
              builder.getContext(), *gpuReduceKind);
          Value result = builder.create<gpu::AllReduceOp>(
              loc, partial.getType(), partial, reduceAttr,
              /*uniform=*/false);

          // Thread-0 guard: only thread 0 writes the final result.
          Value tidX = builder.create<gpu::ThreadIdOp>(
              loc, builder.getIndexType(), gpu::Dimension::x);
          Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value isThread0 = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, tidX, zero);

          bool singleBlock = isSingleBlockLaunch(launchOp);
          SmallVector<Value> indices(atomicOp.getIndices().begin(),
                                     atomicOp.getIndices().end());
          Value outputMem = atomicOp.getMemref();
          auto kind = atomicOp.getKind();

          builder.create<scf::IfOp>(
              loc, isThread0,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                if (singleBlock) {
                  thenBuilder.create<memref::StoreOp>(
                      thenLoc, result, outputMem, indices);
                } else {
                  thenBuilder.create<memref::AtomicRMWOp>(
                      thenLoc, kind, result, outputMem, indices);
                }
                thenBuilder.create<scf::YieldOp>(thenLoc);
              });

          atomicOp.erase();
        }

        if (!strideAtomics.empty())
          continue;
      }

      if (triads.empty())
        continue;

      for (auto &triad : triads) {
        Location loc = triad.atomicToScratch.getLoc();

        // Check if the partial value is already a full cross-thread sum.
        bool preReduced = isPreReduced(triad.partialValue, launchOp,
                                         triad.kind);

        if (triad.outermostContainingLoop && useShuffle && !preReduced) {
          // ── Nested triad, single warp: hoist accumulation into register
          //    + butterfly shuffle ──
          //
          // Before:
          //   scf.for ... {                       // no iter_arg for partial
          //     ...
          //     atomic_rmw addf %partial, %scratch[]   (A)
          //   }
          //   %val = load %scratch[]                   (B)
          //   atomic_rmw addf %val, %output[]          (C)
          //
          // After:
          //   %total = scf.for ... iter_args(%acc = 0.0) {
          //     ...
          //     %new = addf %partial, %acc
          //     yield %new
          //   }
          //   %shuffled = shuffle_tree(%total)
          //   if (tid == 0) store %shuffled, %output[]
          //
          LLVM_DEBUG(llvm::dbgs()
                     << "[warp-shuffle] Transforming nested triad "
                        "(hoist+shuffle): kind="
                     << (int)triad.kind << " threads=" << numThreads << "\n");

          auto outerFor = cast<scf::ForOp>(triad.outermostContainingLoop);
          OpBuilder builder(outerFor);

          // Build the identity init for the new iter_arg.
          auto elemTy = cast<MemRefType>(triad.scratchMemref.getType())
                            .getElementType();
          Value identityInit = buildReductionIdentity(builder, loc, triad.kind,
                                                  elemTy);

          // Rebuild the outermost scf.for with one additional iter_arg.
          // Clone the body and replace atomicA with register accumulation.
          auto newFor = builder.create<scf::ForOp>(
              loc, outerFor.getLowerBound(), outerFor.getUpperBound(),
              outerFor.getStep(),
              /*iterArgs=*/ValueRange{identityInit});

          // Map old induction var → new induction var.
          IRMapping mapping;
          mapping.map(outerFor.getInductionVar(),
                      newFor.getInductionVar());
          // Map any existing iter_args of the old loop (there are none in
          // the typical case, but handle it generically).
          for (auto [oldArg, newArg] :
               llvm::zip(outerFor.getRegionIterArgs(),
                         newFor.getRegionIterArgs().drop_back(1))) {
            mapping.map(oldArg, newArg);
          }

          // The new accumulator iter_arg is the last one.
          Value newAcc = newFor.getRegionIterArgs().back();

          // Clone all ops from the old for-body into the new for-body,
          // except the yield (we'll build a new one).
          builder.setInsertionPointToStart(newFor.getBody());
          for (auto &op : outerFor.getBody()->without_terminator()) {
            if (&op == triad.atomicToScratch.getOperation()) {
              // Replace atomic_rmw with register combine.
              Value partial = mapping.lookupOrDefault(triad.partialValue);
              Value combined =
                  buildReductionCombineOp(builder, loc, triad.kind, partial, newAcc);
              newAcc = combined;
            } else {
              builder.clone(op, mapping);
            }
          }

          // Build yield: forward any original iter results + our accumulator.
          SmallVector<Value> yieldVals;
          auto oldYield =
              cast<scf::YieldOp>(outerFor.getBody()->getTerminator());
          for (Value v : oldYield.getOperands())
            yieldVals.push_back(mapping.lookupOrDefault(v));
          yieldVals.push_back(newAcc);
          builder.create<scf::YieldOp>(loc, yieldVals);

          // The per-thread total is the last result of the new for.
          Value perThreadTotal = newFor.getResults().back();

          // Forward any original results.
          for (auto [oldRes, newRes] :
               llvm::zip(outerFor.getResults(),
                         newFor.getResults().drop_back(1))) {
            oldRes.replaceAllUsesWith(newRes);
          }

          // Shuffle the per-thread totals and thread-0 writes.
          builder.setInsertionPointAfter(newFor);
          Value shuffled = buildShuffleTree(builder, loc, perThreadTotal,
                                            numThreads, triad.kind);

          Value tidX = builder.create<gpu::ThreadIdOp>(
              loc, builder.getIndexType(), gpu::Dimension::x);
          Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value isThread0 = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, tidX, zero);

          builder.create<scf::IfOp>(
              loc, isThread0,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                if (singleBlock) {
                  thenBuilder.create<memref::StoreOp>(
                      thenLoc, shuffled, triad.outputMemref,
                      triad.outputIndices);
                } else {
                  thenBuilder.create<memref::AtomicRMWOp>(
                      thenLoc, triad.kind, shuffled, triad.outputMemref,
                      triad.outputIndices);
                }
                thenBuilder.create<scf::YieldOp>(thenLoc);
              });

          // Erase the old loop and (B)/(C).
          triad.atomicToOutput.erase();
          triad.loadFromScratch.erase();
          outerFor.erase();
          tryCleanupScratch(triad.scratchMemref, launchOp);

          // Ensure the output is zeroed if it's external (and not already zeroed by memset).
          // SCFScalarizeAccumulator should handle global memrefs.
          // For shared scratch, NovaStrideReduction/SCFScalarize might have missed it.
          initializeSharedScratch(builder, loc, triad.scratchMemref, triad.kind, launchOp);

        } else if (triad.outermostContainingLoop && preReduced) {
          // ── Nested triad, pre-reduced: hoist to register, skip shuffle ──
          // The inner scf.for uses shared external scratch, so every
          // thread's partial is already the full sum.  Hoist the outer
          // accumulation into a register (avoiding 32× from outer atomics)
          // and let thread-0 write directly — no shuffle needed.
          LLVM_DEBUG(llvm::dbgs()
                     << "[warp-shuffle] Nested pre-reduced triad "
                        "(hoist, tid0-only): kind="
                     << (int)triad.kind << " threads=" << numThreads << "\n");

          auto outerFor = cast<scf::ForOp>(triad.outermostContainingLoop);
          OpBuilder builder(outerFor);

          auto elemTy = cast<MemRefType>(triad.scratchMemref.getType())
                            .getElementType();
          Value identityInit = buildReductionIdentity(builder, loc, triad.kind,
                                                  elemTy);

          auto newFor = builder.create<scf::ForOp>(
              loc, outerFor.getLowerBound(), outerFor.getUpperBound(),
              outerFor.getStep(), ValueRange{identityInit});

          IRMapping mapping;
          mapping.map(outerFor.getInductionVar(),
                      newFor.getInductionVar());
          for (auto [oldArg, newArg] :
               llvm::zip(outerFor.getRegionIterArgs(),
                         newFor.getRegionIterArgs().drop_back(1)))
            mapping.map(oldArg, newArg);

          Value newAcc = newFor.getRegionIterArgs().back();

          builder.setInsertionPointToStart(newFor.getBody());
          for (auto &op : outerFor.getBody()->without_terminator()) {
            if (&op == triad.atomicToScratch.getOperation()) {
              Value partial = mapping.lookupOrDefault(triad.partialValue);
              newAcc = buildReductionCombineOp(builder, loc, triad.kind,
                                                    partial, newAcc);
            } else {
              builder.clone(op, mapping);
            }
          }

          SmallVector<Value> yieldVals;
          auto oldYield =
              cast<scf::YieldOp>(outerFor.getBody()->getTerminator());
          for (Value v : oldYield.getOperands())
            yieldVals.push_back(mapping.lookupOrDefault(v));
          yieldVals.push_back(newAcc);
          builder.create<scf::YieldOp>(loc, yieldVals);

          Value perThreadTotal = newFor.getResults().back();
          for (auto [oldRes, newRes] :
               llvm::zip(outerFor.getResults(),
                         newFor.getResults().drop_back(1)))
            oldRes.replaceAllUsesWith(newRes);

          // Thread-0 writes directly — no shuffle.
          builder.setInsertionPointAfter(newFor);
          Value tidX = builder.create<gpu::ThreadIdOp>(
              loc, builder.getIndexType(), gpu::Dimension::x);
          Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value isThread0 = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, tidX, zero);

          builder.create<scf::IfOp>(
              loc, isThread0,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                if (singleBlock) {
                  thenBuilder.create<memref::StoreOp>(
                      thenLoc, perThreadTotal, triad.outputMemref,
                      triad.outputIndices);
                } else {
                  thenBuilder.create<memref::AtomicRMWOp>(
                      thenLoc, triad.kind, perThreadTotal,
                      triad.outputMemref, triad.outputIndices);
                }
                thenBuilder.create<scf::YieldOp>(thenLoc);
              });

          triad.atomicToOutput.erase();
          triad.loadFromScratch.erase();
          outerFor.erase();
          tryCleanupScratch(triad.scratchMemref, launchOp);

        } else if (triad.outermostContainingLoop) {
          // ── Nested triad, multi-warp or other: barrier + thread-0 guard ──
          // Keep atomicA inside the loop (correct accumulation to scratch).
          LLVM_DEBUG(llvm::dbgs()
                     << "[warp-shuffle] Transforming nested triad "
                        "(barrier+guard): kind="
                     << (int)triad.kind << " threads=" << numThreads << "\n");

          OpBuilder builder(triad.outermostContainingLoop->getBlock(),
                            std::next(Block::iterator(
                                triad.outermostContainingLoop)));
          emitBarrierAndGuardedWrite(builder, loc, triad, singleBlock);

          // Erase (B) and (C). Keep (A) — it's inside the loop.
          triad.atomicToOutput.erase();
          triad.loadFromScratch.erase();

        } else if (useShuffle && preReduced) {
          // ── Same-block, pre-reduced: the scf.for loop body already uses
          //    cross-thread atomics to a shared scratch, so every thread
          //    holds the FULL sum, not a per-thread partial.  Shuffling
          //    would multiply by N.  Just let thread-0 write directly. ──
          LLVM_DEBUG(llvm::dbgs()
                     << "[warp-shuffle] Pre-reduced triad (tid0-only): kind="
                     << (int)triad.kind << " threads=" << numThreads << "\n");

          OpBuilder builder(triad.atomicToScratch);
          Value tidX = builder.create<gpu::ThreadIdOp>(
              loc, builder.getIndexType(), gpu::Dimension::x);
          Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value isThread0 = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, tidX, zero);

          builder.create<scf::IfOp>(
              loc, isThread0,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                if (singleBlock) {
                  thenBuilder.create<memref::StoreOp>(
                      thenLoc, triad.partialValue, triad.outputMemref,
                      triad.outputIndices);
                } else {
                  thenBuilder.create<memref::AtomicRMWOp>(
                      thenLoc, triad.kind, triad.partialValue,
                      triad.outputMemref, triad.outputIndices);
                }
                thenBuilder.create<scf::YieldOp>(thenLoc);
              });

          // Erase all three ops — scratch no longer needed.
          triad.atomicToOutput.erase();
          triad.loadFromScratch.erase();
          triad.atomicToScratch.erase();
          tryCleanupScratch(triad.scratchMemref, launchOp);

        } else if (useShuffle) {
          // ── Single-warp, same-block path: butterfly shuffle ──
          // Replace all three ops with shuffle + thread-0 write.
          LLVM_DEBUG(llvm::dbgs()
                     << "[warp-shuffle] Transforming triad (shuffle): kind="
                     << (int)triad.kind << " threads=" << numThreads << "\n");

          OpBuilder builder(triad.atomicToScratch);
          Value result = buildShuffleTree(builder, loc, triad.partialValue,
                                          numThreads, triad.kind);

          Value tidX = builder.create<gpu::ThreadIdOp>(
              loc, builder.getIndexType(), gpu::Dimension::x);
          Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value isThread0 = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, tidX, zero);

          builder.create<scf::IfOp>(
              loc, isThread0,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                if (singleBlock) {
                  thenBuilder.create<memref::StoreOp>(
                      thenLoc, result, triad.outputMemref,
                      triad.outputIndices);
                } else {
                  thenBuilder.create<memref::AtomicRMWOp>(
                      thenLoc, triad.kind, result, triad.outputMemref,
                      triad.outputIndices);
                }
                thenBuilder.create<scf::YieldOp>(thenLoc);
              });

          // Erase all three ops.
          triad.atomicToOutput.erase();
          triad.loadFromScratch.erase();
          triad.atomicToScratch.erase();
          tryCleanupScratch(triad.scratchMemref, launchOp);

        } else {
          // ── Multi-warp, same-block path: barrier + tid==0 guard ──
          LLVM_DEBUG(llvm::dbgs()
                     << "[warp-shuffle] Transforming triad "
                        "(barrier+guard): kind="
                     << (int)triad.kind << " threads=" << numThreads << "\n");

          OpBuilder builder(triad.atomicToScratch);
          builder.setInsertionPointAfter(triad.atomicToScratch);
          emitBarrierAndGuardedWrite(builder, loc, triad, singleBlock);

          // Erase only (B) and (C) — keep (A).
          triad.atomicToOutput.erase();
          triad.loadFromScratch.erase();
        }
      }
    }
  }

  StringRef getArgument() const override {
    return "nova-warp-shuffle-reduction";
  }
  StringRef getDescription() const override {
    return "Replace double-atomic reduction patterns with warp butterfly "
           "shuffle, fixing the N× over-counting bug for GPU reductions";
  }
};

} // namespace

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createNovaWarpShuffleReductionPass() {
  return std::make_unique<NovaWarpShuffleReductionPass>();
}

void registerNovaWarpShuffleReductionPass() {
  PassRegistration<NovaWarpShuffleReductionPass>();
}

} // namespace nova
} // namespace mlir
