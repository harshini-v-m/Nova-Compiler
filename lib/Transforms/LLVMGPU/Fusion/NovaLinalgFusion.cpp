#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace nova {

/// A signature representing the fusion potential of a linalg.generic.
struct FusionSignature {
  SmallVector<int64_t> staticLoopRanges;
  SmallVector<utils::IteratorType> iteratorTypes;
  AffineMap pivotMap;

  bool operator==(const FusionSignature &other) const {
    return staticLoopRanges == other.staticLoopRanges &&
           iteratorTypes == other.iteratorTypes &&
           pivotMap == other.pivotMap;
  }
};

} // namespace nova
} // namespace mlir

namespace llvm {
template <> struct DenseMapInfo<mlir::nova::FusionSignature> {
  static mlir::nova::FusionSignature getEmptyKey() {
    return {{}, {}, mlir::AffineMap()};
  }
  static mlir::nova::FusionSignature getTombstoneKey() {
    return {{-1}, {}, mlir::AffineMap()};
  }
  static unsigned getHashValue(const mlir::nova::FusionSignature &sig) {
    return hash_combine(
        hash_combine_range(sig.staticLoopRanges.begin(),
                           sig.staticLoopRanges.end()),
        hash_combine_range(sig.iteratorTypes.begin(), sig.iteratorTypes.end()),
        sig.pivotMap);
  }
  static bool isEqual(const mlir::nova::FusionSignature &lhs,
                      const mlir::nova::FusionSignature &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir {
namespace nova {

/// Maximum number of linalg.generic ops that may be fused into a single group.
/// Fusing too many ops with distinct inputs increases register pressure and
/// memory traffic, so we cap the group size to keep the fused kernel lean.
static constexpr unsigned kMaxFusionGroupSize = 4;

/// Returns true if `a` and `b` are both produced by linalg.fill with the same
/// scalar constant and the same result type, or are the exact same SSA value.
static bool sameConstantFill(Value a, Value b) {
  if (a == b)
    return true;
  auto fillA = a.getDefiningOp<linalg::FillOp>();
  auto fillB = b.getDefiningOp<linalg::FillOp>();
  if (!fillA || !fillB)
    return false;
  auto cstA = fillA.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto cstB = fillB.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  if (!cstA || !cstB)
    return false;
  return cstA.getValue() == cstB.getValue() && a.getType() == b.getType();
}

/// Returns true if the two linalg.generic bodies are structurally identical.
/// Block arguments at the same index are treated as equivalent; op results
/// are tracked pairwise through the body. This detects duplicate reduction
/// accumulators that compute the same expression over the same input args.
static bool bodiesStructurallyEqual(Block *bodyA, Block *bodyB) {
  if (bodyA->getNumArguments() != bodyB->getNumArguments())
    return false;

  // Build an equivalence map: bodyA value → corresponding bodyB value.
  llvm::DenseMap<Value, Value> equiv;
  for (auto [argA, argB] :
       llvm::zip(bodyA->getArguments(), bodyB->getArguments()))
    equiv[argA] = argB;

  auto opsA = bodyA->without_terminator();
  auto opsB = bodyB->without_terminator();
  auto itA = opsA.begin(), endA = opsA.end();
  auto itB = opsB.begin(), endB = opsB.end();

  for (; itA != endA && itB != endB; ++itA, ++itB) {
    Operation &opA = *itA, &opB = *itB;
    if (opA.getName() != opB.getName())
      return false;
    if (opA.getAttrDictionary() != opB.getAttrDictionary())
      return false;
    if (opA.getNumOperands() != opB.getNumOperands())
      return false;
    if (opA.getNumResults() != opB.getNumResults())
      return false;

    for (auto [ovA, ovB] : llvm::zip(opA.getOperands(), opB.getOperands())) {
      auto it = equiv.find(ovA);
      if (it == equiv.end() || it->second != ovB)
        return false;
    }

    for (auto [resA, resB] : llvm::zip(opA.getResults(), opB.getResults()))
      equiv[resA] = resB;
  }

  if (itA != endA || itB != endB)
    return false;

  // Verify the terminator (linalg.yield) operands.
  Operation *termA = bodyA->getTerminator(), *termB = bodyB->getTerminator();
  if (termA->getNumOperands() != termB->getNumOperands())
    return false;
  for (auto [tvA, tvB] :
       llvm::zip(termA->getOperands(), termB->getOperands())) {
    auto it = equiv.find(tvA);
    if (it == equiv.end() || it->second != tvB)
      return false;
  }

  return true;
}

struct NovaLinalgHorizontalFusionPass
    : public PassWrapper<NovaLinalgHorizontalFusionPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaLinalgHorizontalFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  StringRef getArgument() const final {
    return "nova-linalg-horizontal-fusion";
  }
  StringRef getDescription() const final {
    return "Horizontal fusion of independent linalg.generic ops sharing the "
           "same input and domain";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    for (auto &block : func.getBlocks()) {
    while (deduplicateIdenticalGenerics(block)) { /*repeat*/ }
    fuseGenericsInBlock(block);
  }
  }

private:
  /// Hoists the producer of a value (and its dependencies) to before \p target.
  /// Only moves side-effect-free ops (fills, constants, empties) that live in
  /// the same block — anything else is assumed to already dominate.
  void ensureDominance(Value val, Operation *target) {
    Operation *defOp = val.getDefiningOp();
    if (!defOp || defOp->getBlock() != target->getBlock())
      return;
    if (defOp->isBeforeInBlock(target))
      return;
    for (Value operand : defOp->getOperands())
      ensureDominance(operand, target);
    defOp->moveBefore(target);
  }

  /// Returns true if \p op transitively consumes a result of any op in
  /// \p groupOps.  Only walks defs that live in \p block to bound the search.
  bool transitivelyDependsOnGroup(
      Operation *op, Block &block,
      const llvm::DenseSet<Operation *> &groupOps) {
    // Iterative DFS over the def-use chain to avoid stack overflow on deep
    // graphs.
    llvm::DenseSet<Operation *> visited;
    SmallVector<Operation *> worklist;

    for (Value operand : op->getOperands())
      if (Operation *defOp = operand.getDefiningOp())
        worklist.push_back(defOp);

    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      if (!visited.insert(cur).second)
        continue;
      if (groupOps.count(cur))
        return true;
      // Only chase defs inside the same block.
      if (cur->getBlock() != &block)
        continue;
      for (Value operand : cur->getOperands())
        if (Operation *defOp = operand.getDefiningOp())
          worklist.push_back(defOp);
    }
    return false;
  }

  /// Returns true if `a` and `b` are completely equivalent linalg.generic
  /// reduction ops: same inputs (by SSA value and map), same output maps,
  /// same init fill constants, and structurally identical bodies.
  bool genericsIdentical(GenericOp a, GenericOp b) {
    if (a.getNumDpsInputs() != b.getNumDpsInputs())
      return false;
    if (a.getNumDpsInits() != b.getNumDpsInits())
      return false;
    if (a.getIndexingMapsArray() != b.getIndexingMapsArray())
      return false;

    for (auto [ia, ib] : llvm::zip(a.getDpsInputs(), b.getDpsInputs()))
      if (ia != ib)
        return false;

    for (auto [oa, ob] : llvm::zip(a.getDpsInits(), b.getDpsInits()))
      if (!sameConstantFill(oa, ob))
        return false;

    return bodiesStructurallyEqual(a.getBody(), b.getBody());
  }

  /// Scans \p block for pairs of identical linalg.generic reduction ops and
  /// replaces every duplicate with the first (canonical) occurrence, erasing
  /// the redundant op.  This prevents horizontal fusion from forwarding
  /// identical accumulators as separate outputs of a multi-result generic.

  bool deduplicateIdenticalGenerics(Block &block) {
    SmallVector<GenericOp> candidates;
    for (auto &op : block) {
      auto gop = dyn_cast<GenericOp>(op);
      if (!gop)
        continue;
      if (mlir::linalg::isaContractionOpInterface(cast<LinalgOp>(op)))
        continue;
      bool hasReduction =
          llvm::any_of(gop.getIteratorTypesArray(), [](utils::IteratorType t) {
            return t == utils::IteratorType::reduction;
          });
      if (!hasReduction)
        continue;
      candidates.push_back(gop);
    }

    llvm::DenseSet<Operation *> toErase;
    for (unsigned i = 0; i < candidates.size(); ++i) {
      if (toErase.count(candidates[i]))
        continue;
      for (unsigned j = i + 1; j < candidates.size(); ++j) {
        if (toErase.count(candidates[j]))
          continue;
        if (!genericsIdentical(candidates[i], candidates[j]))
          continue;
        for (auto [ri, rj] :
             llvm::zip(candidates[i].getResults(), candidates[j].getResults()))
          rj.replaceAllUsesWith(ri);
        toErase.insert(candidates[j]);
      }
    }

    for (Operation *op : toErase)
      op->erase();

    return !toErase.empty();
  }

  void fuseGenericsInBlock(Block &block) {
    llvm::DenseMap<FusionSignature, SmallVector<LinalgOp>> groups;

    for (auto &op : block) {
      auto linalgOp = dyn_cast<LinalgOp>(op);
      if (!linalgOp || !isa<GenericOp>(op))
        continue;

      if (mlir::linalg::isaContractionOpInterface(linalgOp))
        continue;

      // Only fuse generics that contain a reduction iterator.  Pure-parallel
      // (elementwise) ops are handled by the elementwise fusion pass.
      bool hasReduction =
          llvm::any_of(linalgOp.getIteratorTypesArray(),
                       [](utils::IteratorType t) {
                         return t == utils::IteratorType::reduction;
                       });
      if (!hasReduction)
        continue;

      // Build the pivot map from the first ranked-tensor input.
      AffineMap pivotMap;
      auto maps = linalgOp.getIndexingMapsArray();
      for (auto [val, map] :
           llvm::zip(linalgOp.getDpsInputs(),
                     llvm::ArrayRef<AffineMap>(maps).take_front(
                         linalgOp.getNumDpsInputs()))) {
        if (isa<RankedTensorType>(val.getType())) {
          pivotMap = map;
          break;
        }
      }

      FusionSignature sig{linalgOp.getStaticLoopRanges(),
                          linalgOp.getIteratorTypesArray(), pivotMap};

      auto &group = groups[sig];

      // Build a set of the ops currently in the group for O(1) lookup during
      // the transitive dependency check.
      llvm::DenseSet<Operation *> groupOpSet;
      for (LinalgOp g : group)
        groupOpSet.insert(g.getOperation());

      // Reject ops with a transitive (not just direct) dependency on any
      // already-grouped op — those have a vertical producer-consumer edge and
      // must stay ordered.
      if (!transitivelyDependsOnGroup(linalgOp.getOperation(), block,
                                      groupOpSet))
        group.push_back(linalgOp);
    }

    for (auto &entry : groups) {
      auto &ops = entry.second;
      if (ops.size() < 2)
        continue;

      // Sort by program order so ops[0] is always the earliest definition in
      // the block.  DenseMap iteration is unordered, so without this the fused
      // op could be inserted before one of its own operands.
      llvm::sort(ops, [](LinalgOp a, LinalgOp b) {
        return a->isBeforeInBlock(b.getOperation());
      });

      // Split large groups into consecutive chunks and fuse each independently.
      for (unsigned i = 0; i < ops.size(); i += kMaxFusionGroupSize) {
        unsigned end =
            std::min<unsigned>(i + kMaxFusionGroupSize, ops.size());
        SmallVector<LinalgOp> chunk(ops.begin() + i, ops.begin() + end);
        if (chunk.size() >= 2)
          fuseGroup(chunk);
      }
    }
  }

  void fuseGroup(SmallVectorImpl<LinalgOp> &ops) {
    LinalgOp firstOp = ops[0];
    OpBuilder builder(firstOp);

    // Keyed by (Value, AffineMap) so the same tensor appearing under two
    // different maps is treated as two distinct inputs.
    llvm::DenseMap<std::pair<Value, AffineMap>, unsigned> inputToIdx;
    SmallVector<Value> combinedInputs;
    SmallVector<AffineMap> inputMaps;

    SmallVector<Value> combinedOutputs;
    SmallVector<AffineMap> outputMaps;

    // -------------------------------------------------------------------------
    // 1. Collect operands, deduplicate shared inputs, and hoist producers.
    // -------------------------------------------------------------------------
    for (LinalgOp op : ops) {
      GenericOp genericOp = cast<GenericOp>(op.getOperation());
      auto allMaps = genericOp.getIndexingMapsArray();
      auto opInputs = genericOp.getDpsInputs();
      auto currentInputMaps =
          llvm::ArrayRef<AffineMap>(allMaps).take_front(opInputs.size());

      for (auto [val, map] : llvm::zip(opInputs, currentInputMaps)) {
        ensureDominance(val, firstOp);
        auto key = std::make_pair(val, map);
        if (!inputToIdx.count(key)) {
          inputToIdx[key] = combinedInputs.size();
          combinedInputs.push_back(val);
          inputMaps.push_back(map);
        }
      }

      auto opInits = genericOp.getDpsInits();
      auto currentOutputMaps =
          llvm::ArrayRef<AffineMap>(allMaps).drop_front(opInputs.size());
      for (auto [val, map] : llvm::zip(opInits, currentOutputMaps)) {
        ensureDominance(val, firstOp);
        combinedOutputs.push_back(val);
        outputMaps.push_back(map);
      }
    }

    // -------------------------------------------------------------------------
    // 2. Create the fused GenericOp shell.
    // -------------------------------------------------------------------------
    SmallVector<AffineMap> allMapsCombined = inputMaps;
    allMapsCombined.append(outputMaps.begin(), outputMaps.end());

    SmallVector<Type> resultTypes;
    for (Value out : combinedOutputs)
      resultTypes.push_back(out.getType());

    auto fusedOp = builder.create<GenericOp>(
        firstOp.getLoc(), resultTypes, combinedInputs, combinedOutputs,
        allMapsCombined, firstOp.getIteratorTypesArray(),
        [](OpBuilder &, Location, ValueRange) {
          // Body is filled in step 3.
        });

    // -------------------------------------------------------------------------
    // 3. Merge bodies from each original op into the fused block.
    // -------------------------------------------------------------------------
    Block *fusedBlock = fusedOp.getBody();
    builder.setInsertionPointToStart(fusedBlock);

    IRMapping mapping;
    SmallVector<Value> yieldOperands;
    unsigned currentOutputArgIdx = combinedInputs.size();

    for (LinalgOp op : ops) {
      GenericOp genericOp = cast<GenericOp>(op.getOperation());
      auto allMaps = genericOp.getIndexingMapsArray();
      auto opInputs = genericOp.getDpsInputs();
      auto opInits = genericOp.getDpsInits();
      auto currentInputMaps =
          llvm::ArrayRef<AffineMap>(allMaps).take_front(opInputs.size());

      // Map original input block-args to the deduplicated fused block-args.
      for (unsigned i = 0; i < opInputs.size(); ++i) {
        auto key = std::make_pair(opInputs[i], currentInputMaps[i]);
        mapping.map(genericOp.getBody()->getArgument(i),
                    fusedBlock->getArgument(inputToIdx[key]));
      }

      // Map original output block-args to the next output slots in fused block.
      for (unsigned i = 0; i < opInits.size(); ++i) {
        mapping.map(genericOp.getBody()->getArgument(opInputs.size() + i),
                    fusedBlock->getArgument(currentOutputArgIdx++));
      }

      for (auto &bodyOp : genericOp.getBody()->without_terminator())
        builder.clone(bodyOp, mapping);

      auto yieldOp = cast<YieldOp>(genericOp.getBody()->getTerminator());
      for (Value val : yieldOp.getOperands())
        yieldOperands.push_back(mapping.lookup(val));
    }

    builder.create<YieldOp>(fusedOp.getLoc(), yieldOperands);

    // -------------------------------------------------------------------------
    // 4. Redirect uses of the original results to the fused results.
    // -------------------------------------------------------------------------
    unsigned resultOffset = 0;
    for (LinalgOp op : ops) {
      unsigned numResults = op->getNumResults();
      for (unsigned i = 0; i < numResults; ++i)
        op->getResult(i).replaceAllUsesWith(
            fusedOp.getResult(resultOffset + i));
      resultOffset += numResults;
    }

    // -------------------------------------------------------------------------
    // 5. Erase the now-dead original ops.
    // -------------------------------------------------------------------------
    for (LinalgOp op : ops)
      op->erase();
  }
};

std::unique_ptr<Pass> createNovaLinalgHorizontalFusionPass() {
  return std::make_unique<NovaLinalgHorizontalFusionPass>();
}

void registerNovaLinalgHorizontalFusionPass() {
  PassRegistration<NovaLinalgHorizontalFusionPass>();
}

} // namespace nova
} // namespace mlir