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
    return { {}, {}, mlir::AffineMap() };
  }
  static mlir::nova::FusionSignature getTombstoneKey() {
    return { { -1 }, {}, mlir::AffineMap() };
  }
  static unsigned getHashValue(const mlir::nova::FusionSignature &sig) {
    return hash_combine(
        hash_combine_range(sig.staticLoopRanges.begin(), sig.staticLoopRanges.end()),
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

struct NovaLinalgHorizontalFusionPass
    : public PassWrapper<NovaLinalgHorizontalFusionPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaLinalgHorizontalFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "nova-linalg-horizontal-fusion"; }
  StringRef getDescription() const final {
    return "Horizontal fusion of independent linalg.generic ops sharing the same input and domain";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    for (auto &block : func.getBlocks()) {
      fuseGenericsInBlock(block);
    }
  }

private:
  /// Hoists the producer of a value (and its dependencies) to before the target operation.
  void ensureDominance(Value val, Operation *target) {
    Operation *defOp = val.getDefiningOp();
    if (!defOp || defOp->getBlock() != target->getBlock())
      return;

    // If it already dominates, we are done.
    if (defOp->isBeforeInBlock(target))
      return;

    // Horizontally fused ops are independent. We hoist "simple" side-effect free
    // producers like fills, constants, and empties.
    for (Value operand : defOp->getOperands()) {
      ensureDominance(operand, target);
    }
    defOp->moveBefore(target);
  }

  void fuseGenericsInBlock(Block &block) {
    llvm::DenseMap<FusionSignature, SmallVector<LinalgOp>> groups;

    for (auto &op : block) {
      auto linalgOp = dyn_cast<LinalgOp>(op);
      if (!linalgOp || !isa<GenericOp>(op))
        continue;

      if (mlir::linalg::isaContractionOpInterface(linalgOp))  
         continue ;
      // Constraint: only fuse generics that contain a reduction iterator.
      // Pure-parallel (elementwise) ops are handled by the elementwise fusion
      // pass; admitting them here caused spurious fill-op fusions.
      bool hasReduction = llvm::any_of(
          linalgOp.getIteratorTypesArray(), [](utils::IteratorType t) {
            return t == utils::IteratorType::reduction;
          });
      if (!hasReduction)
        continue;

      SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
      AffineMap pivotMap;
      auto maps = linalgOp.getIndexingMapsArray();

      for (auto [val, map] : llvm::zip(linalgOp.getDpsInputs(),
                                       llvm::ArrayRef<AffineMap>(maps).take_front(
                                           linalgOp.getNumDpsInputs()))) {
        if (isa<RankedTensorType>(val.getType())) {
          pivotMap = map;
          break;
        }
      }
      FusionSignature sig{loopRanges, linalgOp.getIteratorTypesArray(), pivotMap};
      groups[sig].push_back(linalgOp);
    }

    for (auto &entry : groups) {
      auto &ops = entry.second;
      if (ops.size() < 2)
        continue;

      // Constraint: cap each fusion group at kMaxFusionGroupSize ops.
      // Fusing more ops with distinct inputs inflates the fused kernel's
      // operand count, increasing register pressure and memory traffic.
      // We split large groups into consecutive chunks and fuse each chunk
      // independently.
      for (unsigned i = 0; i < ops.size(); i += kMaxFusionGroupSize) {
        unsigned end = std::min<unsigned>(i + kMaxFusionGroupSize, ops.size());
        SmallVector<LinalgOp> chunk(ops.begin() + i, ops.begin() + end);
        if (chunk.size() >= 2)
          fuseGroup(chunk);
      }
    }
  }

  void fuseGroup(SmallVectorImpl<LinalgOp> &ops) {
    LinalgOp firstOp = ops[0];
    OpBuilder builder(firstOp);

    SmallVector<Value> combinedInputs;
    SmallVector<AffineMap> inputMaps;
    llvm::DenseMap<Value, unsigned> inputToIdx;

    SmallVector<Value> combinedOutputs;
    SmallVector<AffineMap> outputMaps;

    // 1. Collect and Hoist Operands. 
    // We hoist all operands to before 'firstOp' to ensure dominance.
    for (LinalgOp op : ops) {
      GenericOp genericOp = cast<GenericOp>(op.getOperation());
      auto allMaps = genericOp.getIndexingMapsArray();
      auto opInputs = genericOp.getDpsInputs();
      auto currentInputMaps = llvm::ArrayRef<AffineMap>(allMaps).take_front(opInputs.size());
      
      for (auto [val, map] : llvm::zip(opInputs, currentInputMaps)) {
        ensureDominance(val, firstOp);
        if (inputToIdx.count(val)) {
          unsigned existingIdx = inputToIdx[val];
          if (inputMaps[existingIdx] == map) continue;
        }
        inputToIdx[val] = combinedInputs.size();
        combinedInputs.push_back(val);
        inputMaps.push_back(map);
      }

      auto opInits = genericOp.getDpsInits();
      auto currentOutputMaps = llvm::ArrayRef<AffineMap>(allMaps).drop_front(opInputs.size());
      for (auto [val, map] : llvm::zip(opInits, currentOutputMaps)) {
        ensureDominance(val, firstOp);
        combinedOutputs.push_back(val);
        outputMaps.push_back(map);
      }
    }

    // 2. Create Fused Op.
    SmallVector<AffineMap> allMapsCombined = inputMaps;
    allMapsCombined.append(outputMaps.begin(), outputMaps.end());

    SmallVector<Type> resultTypes;
    for (Value out : combinedOutputs) {
      resultTypes.push_back(out.getType());
    }

    auto fusedOp = builder.create<GenericOp>(
        firstOp.getLoc(), resultTypes, combinedInputs, combinedOutputs,
        allMapsCombined, firstOp.getIteratorTypesArray(),
        [](OpBuilder &b, Location loc, ValueRange args) {
          // Placeholder for the merging logic.
        });

    // 3. Merge Bodies.
    Block *fusedBlock = fusedOp.getBody();
    builder.setInsertionPointToStart(fusedBlock);
    
    IRMapping mapping;
    SmallVector<Value> yieldOperands;
    unsigned currentOutputArgIdx = combinedInputs.size();
    
    for (LinalgOp op : ops) {
      GenericOp genericOp = cast<GenericOp>(op.getOperation());
      auto opInputs = genericOp.getDpsInputs();
      auto opInits = genericOp.getDpsInits();
      
      for (unsigned i = 0; i < opInputs.size(); ++i) {
          mapping.map(genericOp.getBody()->getArgument(i), 
                      fusedBlock->getArgument(inputToIdx[opInputs[i]]));
      }
      
      for (unsigned i = 0; i < opInits.size(); ++i) {
          mapping.map(genericOp.getBody()->getArgument(opInputs.size() + i), 
                      fusedBlock->getArgument(currentOutputArgIdx++));
      }

      for (auto &bodyOp : genericOp.getBody()->without_terminator()) {
        builder.clone(bodyOp, mapping);
      }

      auto yieldOp = cast<YieldOp>(genericOp.getBody()->getTerminator());
      for (Value val : yieldOp.getOperands()) {
        yieldOperands.push_back(mapping.lookup(val));
      }
    }

    builder.create<YieldOp>(fusedOp.getLoc(), yieldOperands);

    // 4. Update Uses.
    unsigned resultOffset = 0;
    for (LinalgOp op : ops) {
      unsigned numResults = op->getNumResults();
      for (unsigned i = 0; i < numResults; ++i) {
        op->getResult(i).replaceAllUsesWith(fusedOp.getResult(resultOffset + i));
      }
      resultOffset += numResults;
    }

    // 5. Erase Original Ops.
    for (LinalgOp op : ops) {
      op->erase();
    }
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