//===- NovaAccumulationFusion.cpp -----------------------------------------===//
//
// Fuses pairs of adjacent gradient-accumulation linalg.generic ops into a
// single fused generic.
//
// MOTIVATION
// ----------
// After backward-pass lowering, gradient accumulation looks like:
//
//   %out0  = bufferization.to_tensor %argX restrict writable : ... -> tensor<...>
//   %acc0  = linalg.generic {all-parallel, no shared inputs} ins(%grad0) outs(%out0)
//              { %0 = addf %in, %out; yield %0 }
//   bufferization.materialize_in_destination %acc0 in writable %argX
//
//   %out1  = bufferization.to_tensor %argY restrict writable : ... -> tensor<...>
//   %acc1  = linalg.generic {all-parallel, no shared inputs} ins(%grad1) outs(%out1)
//              { %0 = addf %in, %out; yield %0 }
//   bufferization.materialize_in_destination %acc1 in writable %argY
//
// These two generics share the same iteration space and are truly independent.
// Each one alone becomes its own GPU kernel. Fusing them halves the kernel
// launch overhead and the number of kernel slots consumed (relevant when the
// kernel outlining pass caps at ~76 kernels for a 3-layer GPT-2 backward pass).
//
// WHAT "ADJACENT" MEANS
// ---------------------
// Two generics A and B qualify when:
//   1. Both are all-parallel (no reduction iterators).
//   2. They have identical static loop ranges.
//   3. They have identical indexing-map structures (same maps).
//   4. B immediately follows A in program order with only
//      bufferization.to_tensor / bufferization.materialize_in_destination
//      ops permitted between them (these are the write-back bookkeeping ops
//      attached to A — they don't produce values consumed by B).
//   5. B does not consume any result of A (no producer-consumer edge).
//   6. The inputs of B do not overlap with the inputs of A (accumulations
//      always write to different gradient buffers, so this is expected to hold
//      and acts as a sanity guard).
//
// DOMINANCE SAFETY
// ----------------
// The existing NovaLinalgFusion pass had a dominance problem: it moved ops
// across region boundaries. This pass avoids the issue entirely by:
//   - Never moving any op. The fused generic is inserted at the position of A
//     (the first op), and all operands of both A and B are already defined
//     before A (they come from block arguments or earlier ops).
//   - The to_tensor ops for B's outputs are inserted immediately before A so
//     they dominate the fused op. These are side-effect-free reads from
//     block-argument memrefs — safe to hoist.
//   - No result of A is consumed by B (checked explicitly), so no def-use
//     ordering is disturbed.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaKernelConfig.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace nova {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true when every iterator type in \p op is parallel.
static bool isAllParallel(linalg::GenericOp op) {
  return llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::parallel;
  });
}

/// Returns true when \p op is a bufferization.to_tensor or
/// bufferization.materialize_in_destination — the bookkeeping ops that wrap
/// an accumulation generic but don't carry compute.
static bool isBufferizationBookkeeping(Operation *op) {
  return isa<bufferization::ToTensorOp>(op) ||
         isa<bufferization::MaterializeInDestinationOp>(op);
}

/// Returns true when the inputs of \p a and \p b are disjoint (no shared
/// input value).
static bool inputsAreDisjoint(linalg::GenericOp a, linalg::GenericOp b) {
  llvm::DenseSet<Value> aInputs;
  for (Value v : a.getDpsInputs())
    aInputs.insert(v);
  for (Value v : b.getDpsInputs())
    if (aInputs.count(v))
      return false;
  return true;
}

/// Returns true when \p b does not consume any result of \p a.
static bool bDoesNotConsumeA(linalg::GenericOp a, linalg::GenericOp b) {
  llvm::DenseSet<Value> aResults;
  for (Value r : a->getResults())
    aResults.insert(r);
  for (Value v : b->getOperands())
    if (aResults.count(v))
      return false;
  return true;
}

/// Returns true when the two generics have identical indexing maps and
/// static loop ranges — i.e. they tile the exact same iteration space.
static bool sameIterationSpace(linalg::GenericOp a, linalg::GenericOp b) {
  if (a.getStaticLoopRanges() != b.getStaticLoopRanges())
    return false;
  if (a.getIteratorTypesArray() != b.getIteratorTypesArray())
    return false;
  // Check that the indexing maps have the same structure.
  // Both have one input and one output, both with identity maps.
  auto aMaps = a.getIndexingMapsArray();
  auto bMaps = b.getIndexingMapsArray();
  if (aMaps.size() != bMaps.size())
    return false;
  for (auto [ma, mb] : llvm::zip(aMaps, bMaps))
    if (ma != mb)
      return false;
  return true;
}

// ---------------------------------------------------------------------------
// Core fusion logic
// ---------------------------------------------------------------------------

/// Tries to find a fusion partner for \p first by scanning forward past
/// bufferization bookkeeping ops. Returns the candidate generic or nullptr.
static linalg::GenericOp findAdjacentAccumulationOp(linalg::GenericOp first) {
  Block *block = first->getBlock();
  auto it = std::next(first->getIterator());
  while (it != block->end()) {
    Operation *op = &*it;
    if (isBufferizationBookkeeping(op)) {
      ++it;
      continue;
    }
    // The next non-bookkeeping op must be a generic to qualify.
    auto candidate = dyn_cast<linalg::GenericOp>(op);
    if (!candidate)
      return nullptr;
    return candidate;
  }
  return nullptr;
}

/// Fuses \p first and \p second into a single linalg.generic inserted at the
/// position of \p first. After fusion, both originals are erased and the
/// intervening bookkeeping ops for \p first are also removed.
///
/// Layout of the fused op:
///   ins  = inputs_of_first ++ inputs_of_second   (no dedup — accumulations
///                                                  always have disjoint buffers)
///   outs = outs_of_first ++ outs_of_second
///   body = body_of_first ; body_of_second
///   yield = yield_vals_of_first ++ yield_vals_of_second
static LogicalResult fuseAccumulationPair(linalg::GenericOp first,
                                          linalg::GenericOp second,
                                          const NVIDIATargetInfo &target) {
  // ---- Collect operands --------------------------------------------------
  auto firstInputs  = SmallVector<Value>(first.getDpsInputs());
  auto firstInits   = SmallVector<Value>(first.getDpsInits());
  auto secondInputs = SmallVector<Value>(second.getDpsInputs());
  auto secondInits  = SmallVector<Value>(second.getDpsInits());

  // ---- Hoist second's to_tensor outs before first ------------------------
  // Hoist any to_tensor ops that define second's operands (both inputs and
  // inits) to before `first` so they dominate the fused op.
  // These are side-effect-free reads from function-argument memrefs.
  OpBuilder hoistBuilder(first);
  auto hoistIfNeeded = [&](Value v) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !isa<bufferization::ToTensorOp>(defOp))
      return;
    if (!defOp->isBeforeInBlock(first))
      defOp->moveBefore(first);
  };
  for (Value v : secondInputs)
    hoistIfNeeded(v);
  for (Value v : secondInits)
    hoistIfNeeded(v);

  // ---- Build indexing maps for fused op ----------------------------------
  auto firstMaps  = first.getIndexingMapsArray();
  auto secondMaps = second.getIndexingMapsArray();

  // firstMaps  = [inputMaps_first...,  outputMaps_first...]
  // secondMaps = [inputMaps_second..., outputMaps_second...]
  unsigned nFirstIn  = firstInputs.size();
  unsigned nFirstOut = firstInits.size();
  unsigned nSecondIn = secondInputs.size();

  SmallVector<AffineMap> fusedMaps;
  // Input maps: first inputs, then second inputs.
  for (unsigned i = 0; i < nFirstIn; ++i)
    fusedMaps.push_back(firstMaps[i]);
  for (unsigned i = 0; i < nSecondIn; ++i)
    fusedMaps.push_back(secondMaps[i]);
  // Output maps: first outputs, then second outputs.
  for (unsigned i = nFirstIn; i < (unsigned)firstMaps.size(); ++i)
    fusedMaps.push_back(firstMaps[i]);
  for (unsigned i = nSecondIn; i < (unsigned)secondMaps.size(); ++i)
    fusedMaps.push_back(secondMaps[i]);

  // ---- Collect combined operands -----------------------------------------
  SmallVector<Value> fusedInputs;
  fusedInputs.append(firstInputs.begin(), firstInputs.end());
  fusedInputs.append(secondInputs.begin(), secondInputs.end());

  SmallVector<Value> fusedInits;
  fusedInits.append(firstInits.begin(), firstInits.end());
  fusedInits.append(secondInits.begin(), secondInits.end());

  SmallVector<Type> resultTypes;
  for (Value v : fusedInits)
    resultTypes.push_back(v.getType());

  // ---- Create fused GenericOp shell --------------------------------------
  OpBuilder builder(first);
  auto fusedOp = builder.create<linalg::GenericOp>(
      first.getLoc(), resultTypes, fusedInputs, fusedInits, fusedMaps,
      first.getIteratorTypesArray(),
      [](OpBuilder &, Location, ValueRange) {
        // Body filled below.
      });

  // ---- Merge bodies -------------------------------------------------------
  Block *fusedBlock = fusedOp.getBody();
  builder.setInsertionPointToStart(fusedBlock);
  IRMapping mapping;

  // Helper: map block args of `src` generic into the fused block.
  // inputOffset  = position of this op's first input arg in fused block
  // outputOffset = position of this op's first output arg in fused block
  auto mergeBody = [&](linalg::GenericOp op, unsigned inputOffset,
                       unsigned outputOffset) -> SmallVector<Value> {
    Block *srcBlock = op.getBody();
    unsigned nIn  = op.getDpsInputs().size();
    unsigned nOut = op.getDpsInits().size();

    for (unsigned i = 0; i < nIn; ++i)
      mapping.map(srcBlock->getArgument(i),
                  fusedBlock->getArgument(inputOffset + i));
    for (unsigned i = 0; i < nOut; ++i)
      mapping.map(srcBlock->getArgument(nIn + i),
                  fusedBlock->getArgument(outputOffset + i));

    for (auto &bodyOp : srcBlock->without_terminator())
      builder.clone(bodyOp, mapping);

    SmallVector<Value> yieldVals;
    auto yieldOp = cast<linalg::YieldOp>(srcBlock->getTerminator());
    for (Value v : yieldOp.getOperands())
      yieldVals.push_back(mapping.lookup(v));
    return yieldVals;
  };

  // first:  inputs start at 0,                outputs start at nFirstIn+nSecondIn
  // second: inputs start at nFirstIn,          outputs start at nFirstIn+nSecondIn+nFirstOut
  unsigned firstOutputBase  = nFirstIn + nSecondIn;
  unsigned secondOutputBase = firstOutputBase + nFirstOut;

  SmallVector<Value> yieldOperands;
  auto y0 = mergeBody(first,  0,           firstOutputBase);
  auto y1 = mergeBody(second, nFirstIn,    secondOutputBase);
  yieldOperands.append(y0.begin(), y0.end());
  yieldOperands.append(y1.begin(), y1.end());

  builder.create<linalg::YieldOp>(fusedOp.getLoc(), yieldOperands);

  // ---- Stamp lowering_config on the fused op ------------------------------
  // SelectLoweringStrategy skips small all-parallel accumulation ops, so
  // neither `first` nor the fused op gets a config automatically. Without a
  // config the tiling pass falls into Phase 2 (single-result scf.forall),
  // which only creates one shared_out/parallel_insert_slice for result(0).
  // The second result of the fused op is then trapped inside the child region
  // and its uses outside violate dominance.
  //
  // Fix: call setDefaultConfig directly here to stamp a proper elementwise
  // config so the fused op goes through Phase 1 (tileUsingSCF) instead.
  (void)setDefaultConfig(fusedOp, target);

  // ---- Redirect results ---------------------------------------------------
  // first's results → fusedOp.result[0..nFirstOut-1]
  for (unsigned i = 0; i < (unsigned)first->getNumResults(); ++i)
    first->getResult(i).replaceAllUsesWith(fusedOp.getResult(i));
  // second's results → fusedOp.result[nFirstOut..nFirstOut+nSecondOut-1]
  unsigned secondResultBase = first->getNumResults();
  for (unsigned i = 0; i < (unsigned)second->getNumResults(); ++i)
    second->getResult(i).replaceAllUsesWith(
        fusedOp.getResult(secondResultBase + i));

  // ---- Erase originals ----------------------------------------------------
  second->erase();
  first->erase();

  return success();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct NovaAccumulationFusionPass
    : public PassWrapper<NovaAccumulationFusionPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaAccumulationFusionPass)

  NovaAccumulationFusionPass() = default;
  NovaAccumulationFusionPass(const NovaAccumulationFusionPass &pass)
      : PassWrapper(pass) {}
  explicit NovaAccumulationFusionPass(StringRef arch) {
    cudaArch = arch.str();
  }

  Option<std::string> cudaArch{*this, "cuda-arch",
                               llvm::cl::desc("Target CUDA architecture"),
                               llvm::cl::init("sm_86")};

  StringRef getArgument() const final {
    return "nova-accumulation-fusion";
  }
  StringRef getDescription() const final {
    return "Fuse adjacent all-parallel gradient-accumulation linalg.generic "
           "ops that share the same iteration space and have disjoint inputs.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, bufferization::BufferizationDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    NVIDIATargetInfo target = getNVIDIATargetInfo(cudaArch);
    bool changed = true;
    // Iterate to fixpoint — each fusion may expose a new adjacent pair.
    while (changed) {
      changed = false;
      for (Block &block : func.getBlocks()) {
        for (auto it = block.begin(); it != block.end(); ) {
          auto generic = dyn_cast<linalg::GenericOp>(&*it);
          ++it; // advance before potential erasure
          if (!generic)
            continue;
          if (!isAllParallel(generic))
            continue;

          linalg::GenericOp partner = findAdjacentAccumulationOp(generic);
          if (!partner)
            continue;
          if (!isAllParallel(partner))
            continue;
          if (!sameIterationSpace(generic, partner))
            continue;
          if (!bDoesNotConsumeA(generic, partner))
            continue;
          if (!inputsAreDisjoint(generic, partner))
            continue;

          if (succeeded(fuseAccumulationPair(generic, partner, target))) {
            changed = true;
            // Restart iteration from the beginning of this block since we
            // erased ops and iterators are now invalid.
            it = block.begin();
          }
        }
      }
    }
  }
};

std::unique_ptr<Pass> createNovaAccumulationFusionPass() {
  return std::make_unique<NovaAccumulationFusionPass>();
}

std::unique_ptr<Pass> createNovaAccumulationFusionPass(llvm::StringRef cudaArch) {
  return std::make_unique<NovaAccumulationFusionPass>(cudaArch);
}

void registerNovaAccumulationFusionPass() {
  PassRegistration<NovaAccumulationFusionPass>();
}

} // namespace nova
} // namespace mlir
