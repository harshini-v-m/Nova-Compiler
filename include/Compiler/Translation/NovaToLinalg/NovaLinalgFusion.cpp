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
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace nova {

struct FusionSignature {
  SmallVector<int64_t> staticLoopRanges;
  SmallVector<utils::IteratorType> iteratorTypes;
  Value pivotInput;
  AffineMap pivotMap;

  bool operator==(const FusionSignature &other) const {
    return staticLoopRanges == other.staticLoopRanges &&
           iteratorTypes == other.iteratorTypes &&
           pivotInput == other.pivotInput &&
           pivotMap == other.pivotMap;
  }
};

} // namespace nova
} // namespace mlir

namespace llvm {
template <> struct DenseMapInfo<mlir::nova::FusionSignature> {
  static mlir::nova::FusionSignature getEmptyKey() {
    return { {}, {}, mlir::Value(), mlir::AffineMap() };
  }
  static mlir::nova::FusionSignature getTombstoneKey() {
    return { { -1 }, {}, mlir::Value(), mlir::AffineMap() };
  }
  static unsigned getHashValue(const mlir::nova::FusionSignature &sig) {
    unsigned hash = hash_combine_range(sig.staticLoopRanges.begin(), sig.staticLoopRanges.end());
    hash = hash_combine(hash, hash_combine_range(sig.iteratorTypes.begin(), sig.iteratorTypes.end()));
    hash = hash_combine(hash, mlir::hash_value(sig.pivotInput));
    hash = hash_combine(hash, mlir::hash_value(sig.pivotMap));
    return hash;
  }
  static bool isEqual(const mlir::nova::FusionSignature &lhs,
                     const mlir::nova::FusionSignature &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir {
namespace nova {

struct NovaLinalgHorizontalFusionPass
    : public PassWrapper<NovaLinalgHorizontalFusionPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaLinalgHorizontalFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "nova-linalg-horizontal-fusion"; }
  StringRef getDescription() const final {
    return "Horizontal fusion of independent linalg.generic ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    for (auto &block : func.getBlocks()) {
      while (tryFuseOnce(block)) { /* Greedy restart */ }
    }
  }

private:
  bool tryFuseOnce(Block &block) {
    llvm::DenseMap<FusionSignature, SmallVector<LinalgOp>> groups;

    for (auto &op : block) {
      auto linalgOp = dyn_cast<LinalgOp>(op);
      if (!linalgOp || !isa<GenericOp>(op)) continue;

      // Only fuse reduction ops — pure parallel generics are handled by
      // linalg-fuse-elementwise-ops, and fills have no business being fused here.
      bool hasReduction = llvm::any_of(
          linalgOp.getIteratorTypesArray(), [](utils::IteratorType t) {
            return t == utils::IteratorType::reduction;
          });
      if (!hasReduction) continue;

      // Find the pivot: first ranked-tensor input with its indexing map.
      // Ops with no inputs (pure fills) are skipped here.
      auto maps = linalgOp.getIndexingMapsArray();
      Value pivotInput;
      AffineMap pivotMap;
      for (auto [val, map] : llvm::zip(
               linalgOp.getDpsInputs(),
               llvm::ArrayRef<AffineMap>(maps).take_front(linalgOp.getNumDpsInputs()))) {
        if (isa<RankedTensorType>(val.getType())) {
          pivotInput = val;
          pivotMap = map;
          break;
        }
      }
      if (!pivotInput) continue;

      SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
      FusionSignature sig{loopRanges, linalgOp.getIteratorTypesArray(), pivotInput, pivotMap};
      groups[sig].push_back(linalgOp);
    }

    for (auto &entry : groups) {
      auto &ops = entry.second;
      if (ops.size() < 2) continue;
      
      // Collect independent ops from this group using strict topological tracking.
      SmallVector<LinalgOp> independentOps;
      llvm::SmallPtrSet<Operation*, 32> dependsOnGroup;
      
      Operation *firstOp = ops.front().getOperation();
      Operation *lastOp = ops.back().getOperation();
      
      llvm::SmallPtrSet<Operation*, 8> opsSet;
      for (auto op : ops) opsSet.insert(op.getOperation());

      for (Operation *it = firstOp; it != nullptr; it = it->getNextNode()) {
        bool opDependsOnGroup = false;
        for (Value operand : it->getOperands()) {
          if (auto def = operand.getDefiningOp()) {
            if (dependsOnGroup.count(def)) {
              opDependsOnGroup = true;
              break;
            }
          }
        }

        if (opDependsOnGroup || opsSet.count(it)) {
          dependsOnGroup.insert(it); // Taint this op, its results are now dependent on the group
        }
        
        if (opsSet.count(it) && !opDependsOnGroup) {
          independentOps.push_back(cast<LinalgOp>(it)); // It's safe! No dependencies.
        }

        if (it == lastOp) break;
      }

      if (independentOps.size() >= 2) {
        fuseGroup(independentOps);
        return true;
      }
    }
    return false;
  }

  void fuseGroup(SmallVectorImpl<LinalgOp> &ops) {
    LinalgOp lastOp = ops.back();
    OpBuilder builder(lastOp);

    SmallVector<Value> combinedInputs;
    SmallVector<AffineMap> inputMaps;
    llvm::DenseMap<std::pair<Value, AffineMap>, unsigned> inputToIdx;

    SmallVector<Value> combinedOutputs;
    SmallVector<AffineMap> outputMaps;

    for (LinalgOp op : ops) {
      GenericOp genericOp = cast<GenericOp>(op.getOperation());
      auto allMaps = genericOp.getIndexingMapsArray();
      auto opInputs = genericOp.getDpsInputs();
      for (auto [val, map] : llvm::zip(opInputs, llvm::ArrayRef<AffineMap>(allMaps).take_front(opInputs.size()))) {
        auto key = std::make_pair(val, map);
        if (!inputToIdx.count(key)) {
          inputToIdx[key] = combinedInputs.size();
          combinedInputs.push_back(val);
          inputMaps.push_back(map);
        }
      }
      auto opInits = genericOp.getDpsInits();
      for (auto [val, map] : llvm::zip(opInits, llvm::ArrayRef<AffineMap>(allMaps).drop_front(opInputs.size()))) {
        combinedOutputs.push_back(val);
        outputMaps.push_back(map);
      }
    }

    SmallVector<AffineMap> allMapsCombined = inputMaps;
    allMapsCombined.append(outputMaps.begin(), outputMaps.end());
    SmallVector<Type> resultTypes;
    for (Value out : combinedOutputs) resultTypes.push_back(out.getType());

    auto fusedOp = builder.create<GenericOp>(
        lastOp.getLoc(), resultTypes, combinedInputs, combinedOutputs,
        allMapsCombined, lastOp.getIteratorTypesArray(),
        [](OpBuilder &b, Location loc, ValueRange args) {});

    Block *fusedBlock = fusedOp.getBody();
    builder.setInsertionPointToStart(fusedBlock);
    
    IRMapping mapping;
    SmallVector<Value> yieldOperands;
    unsigned currentOutputArgIdx = combinedInputs.size();
    
    for (LinalgOp op : ops) {
      GenericOp genericOp = cast<GenericOp>(op.getOperation());
      auto opInputs = genericOp.getDpsInputs();
      auto opMaps = genericOp.getIndexingMapsArray();
      for (unsigned i = 0; i < opInputs.size(); ++i) {
          auto key = std::make_pair(opInputs[i], opMaps[i]);
          mapping.map(genericOp.getBody()->getArgument(i), fusedBlock->getArgument(inputToIdx[key]));
      }
      for (unsigned i = 0; i < genericOp.getNumDpsInits(); ++i) {
          mapping.map(genericOp.getBody()->getArgument(opInputs.size() + i), fusedBlock->getArgument(currentOutputArgIdx++));
      }
      for (auto &bodyOp : genericOp.getBody()->without_terminator()) {
        builder.clone(bodyOp, mapping);
      }
      auto yieldOp = cast<YieldOp>(genericOp.getBody()->getTerminator());
      for (Value val : yieldOp.getOperands()) yieldOperands.push_back(mapping.lookupOrDefault(val));
    }
    builder.create<YieldOp>(fusedOp.getLoc(), yieldOperands);

    // Fix dominance for interleaved users.
    llvm::SetVector<Operation*> opsToMove;
    Operation* firstOp = ops.front();
    llvm::SmallPtrSet<Operation*, 16> inGroup;
    for(auto op : ops) inGroup.insert(op.getOperation());

    for (Operation *it = firstOp->getNextNode(); it != fusedOp; it = it->getNextNode()) {
       if (inGroup.count(it)) continue;
       bool dependsOnFused = false;
       for (Value operand : it->getOperands()) {
         if (auto dev = operand.getDefiningOp()) {
           if (inGroup.count(dev) || opsToMove.count(dev)) { dependsOnFused = true; break; }
         }
       }
       if (dependsOnFused) opsToMove.insert(it);
    }
    
    // Move violating users down, PRESERVING their relative order.
    Operation* movePoint = fusedOp;
    for (auto* op : opsToMove) {
      op->moveAfter(movePoint);
      movePoint = op;
    }

    unsigned resultOffset = 0;
    for (LinalgOp op : ops) {
      unsigned n = op->getNumResults();
      for (unsigned i = 0; i < n; ++i) op->getResult(i).replaceAllUsesWith(fusedOp.getResult(resultOffset + i));
      resultOffset += n;
    }
    for (LinalgOp op : ops) op->erase();
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
