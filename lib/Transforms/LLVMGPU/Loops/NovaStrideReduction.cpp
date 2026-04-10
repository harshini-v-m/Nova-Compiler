//===- NovaStrideReduction.cpp - Stride-32 parallelization for reductions ===//
//
// Rewrites single-thread gpu.launch ops containing large scf.for reduction
// loops to use 32 threads with stride-32 access.
//
// Before:
//   gpu.launch blocks(...) threads(1,1,1) {
//     %result = scf.for %i = 0 to 50304 step 1 iter_args(%acc = %init) {
//       // reduction body
//       scf.yield %new_acc
//     }
//   }
//
// After:
//   gpu.launch blocks(...) threads(32,1,1) {
//     %tid = gpu.thread_id x
//     %result = scf.for %i = %tid to 50304 step 32 iter_args(%acc = %init) {
//       // reduction body (unchanged)
//       scf.yield %new_acc
//     }
//   }
//
// Detection criteria:
//   1. gpu.launch has blockSizeX=1, blockSizeY=1, blockSizeZ=1
//   2. scf.for has iter_args (loop-carried accumulator = reduction)
//   3. scf.for has constant lb=0, step=1, and ub >= 32
//   4. ub must be divisible by 32
//
// The pass only modifies launches that are truly single-thread (all block
// dims = 1). Multiple matching scf.for loops inside the same launch are
// all rewritten. Nested scf.for loops that don't meet the criteria are
// left untouched.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-stride-reduction"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract constant integer from a Value, or return std::nullopt.
static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return std::nullopt;
}

/// Returns true if blockSizeX=1, blockSizeY=1, blockSizeZ=1.
static bool isSingleThreadBlock(gpu::LaunchOp launch) {
  auto bx = getConstantIndex(launch.getBlockSizeX());
  auto by = getConstantIndex(launch.getBlockSizeY());
  auto bz = getConstantIndex(launch.getBlockSizeZ());
  return bx && by && bz && *bx == 1 && *by == 1 && *bz == 1;
}

/// Check if an scf.for is a candidate for stride-32 rewrite:
///   - Constant lb == 0
///   - Constant step == 1
///   - Constant ub >= 32 and ub % 32 == 0
///
/// Note: iter_args are NOT required. At this pipeline stage, reduction
/// accumulators are in memrefs (load/store pattern), not iter_args.
/// SCFScalarizeAccumulator runs later and converts to iter_args.
static bool isStrideCandidate(scf::ForOp forOp) {
  auto lb = getConstantIndex(forOp.getLowerBound());
  auto ub = getConstantIndex(forOp.getUpperBound());
  auto step = getConstantIndex(forOp.getStep());

  if (!lb || !ub || !step)
    return false;

  if (*lb != 0 || *step != 1)
    return false;

  if (*ub < 32 || (*ub % 32) != 0)
    return false;

  // Reject loops that store to private (per-thread) memory.
  // With 32 threads, each gets its own private accumulator — the partials
  // can't be combined via atomic or shuffle. Only loops that store to
  // shared/global memory (where SCFScalarize can insert atomic_rmw) are
  // safe to stride.
  bool hasPrivateStore = false;
  forOp.getBody()->walk([&](memref::StoreOp store) {
    auto memTy = dyn_cast<MemRefType>(store.getMemRef().getType());
    if (memTy && memTy.getMemorySpace()) {
      if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(memTy.getMemorySpace())) {
        if (gpuSpace.getValue() == gpu::AddressSpace::Private)
          hasPrivateStore = true;
      }
    }
  });
  if (hasPrivateStore)
    return false;

  return true;
}

/// The stride factor: number of threads to use.
static constexpr int64_t kStrideFactor = 32;

//===----------------------------------------------------------------------===//
// Core transformation
//===----------------------------------------------------------------------===//

/// Rewrite a single gpu.launch: change blockSizeX from 1 to 32, and rewrite
/// all matching scf.for loops inside. Returns true if any changes were made.
static bool rewriteLaunch(gpu::LaunchOp launchOp) {

  SmallVector<scf::ForOp> candidates;
  launchOp.getBody().walk([&](scf::ForOp forOp) {
    if (isStrideCandidate(forOp))
      candidates.push_back(forOp);
  });

  if (candidates.empty())
    return false;

  OpBuilder builder(launchOp.getContext());

  // --- Step 1: Change blockSizeX from 1 to 32 ---
  builder.setInsertionPoint(launchOp);
  Location loc = launchOp.getLoc();
  Value newBlockSizeX = builder.create<arith::ConstantIndexOp>(loc, kStrideFactor);

  // Replace the blockSizeX operand on the launch.
  launchOp.getBlockSizeXMutable().assign(newBlockSizeX);

  // --- Step 2: Rewrite each matching scf.for ---
  for (scf::ForOp forOp : candidates) {
    Location forLoc = forOp.getLoc();

    builder.setInsertionPoint(forOp);

    // Create gpu.thread_id x — this gives the thread index within the block.
    Value tid = builder.create<gpu::ThreadIdOp>(forLoc, gpu::Dimension::x);

    // Create constant 32 as the new step.
    Value strideStep = builder.create<arith::ConstantIndexOp>(forLoc, kStrideFactor);

    // Replace lb with tid, step with 32.
    forOp.getLowerBoundMutable().assign(tid);
    forOp.getStepMutable().assign(strideStep);

    LLVM_DEBUG(llvm::dbgs() << "  Rewrote scf.for at " << forLoc
                            << " to stride-" << kStrideFactor << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Rewrote gpu.launch at " << loc
                          << ": blockSizeX 1 -> " << kStrideFactor
                          << ", " << candidates.size() << " loop(s)\n");
  return true;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaStrideReductionPass
    : public PassWrapper<NovaStrideReductionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaStrideReductionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect all single-thread gpu.launch ops.
    SmallVector<gpu::LaunchOp> launches;
    funcOp.walk([&](gpu::LaunchOp launch) {
      if (isSingleThreadBlock(launch))
        launches.push_back(launch);
    });

    bool changed = false;
    for (gpu::LaunchOp launch : launches)
      changed |= rewriteLaunch(launch);

    if (!changed)
      markAllAnalysesPreserved();
  }

  StringRef getArgument() const override { return "nova-stride-reduction"; }

  StringRef getDescription() const override {
    return "Rewrites single-thread gpu.launch reduction loops to use 32 "
           "threads with stride-32 access";
  }
};

std::unique_ptr<Pass> createNovaStrideReductionPass() {
  return std::make_unique<NovaStrideReductionPass>();
}

void registerNovaStrideReductionPass() {
  PassRegistration<NovaStrideReductionPass>();
}

} // namespace mlir::nova
