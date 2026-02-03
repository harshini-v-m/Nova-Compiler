#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::affine;

namespace {

// Pass to scalarize loop-carried accumulation from buffer to registers
struct AffineScalarizeAccumulatorPass
    : public PassWrapper<AffineScalarizeAccumulatorPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineScalarizeAccumulatorPass)

  void runOnOperation() override {
    auto func = getOperation();
    
    // Collect loops to transform (to avoid iterator invalidation)
    SmallVector<AffineForOp> loopsToTransform;
    
    func.walk([&](AffineForOp forOp) {
      if (shouldScalarize(forOp))
        loopsToTransform.push_back(forOp);
    });
    
    // Transform collected loops
    for (auto forOp : loopsToTransform) {
      scalarizeLoop(forOp);
    }
  }

  bool shouldScalarize(AffineForOp forOp) {
    // Pattern: Find load-compute-store where load and store access the same location
    auto &bodyOps = forOp.getBody()->getOperations();
    if (bodyOps.empty()) return false;
    
    AffineLoadOp loadOp = nullptr;
    AffineStoreOp storeOp = nullptr;
    
    for (auto &op : bodyOps) {
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (!loadOp) loadOp = load;  // Take first load
      }
      if (auto store = dyn_cast<AffineStoreOp>(op)) {
        storeOp = store;  // Take last store
      }
    }
    
    if (!loadOp || !storeOp) return false;
    if (loadOp.getMemRef() != storeOp.getMemRef()) return false;
    if (loadOp.getIndices().size() != storeOp.getIndices().size()) return false;
    
    // Check indices are loop-invariant
    Value loopIV = forOp.getInductionVar();
    for (Value idx : loadOp.getIndices()) {
      if (idx == loopIV) return false;
    }
    
    // Check for init store before loop
    Operation *prevOp = forOp->getPrevNode();
    while (prevOp) {
      if (auto store = dyn_cast<AffineStoreOp>(prevOp)) {
        if (store.getMemRef() == loadOp.getMemRef() &&
            store.getIndices().size() == loadOp.getIndices().size()) {
          return true;  // Found the pattern!
        }
      }
      prevOp = prevOp->getPrevNode();
    }
    
    return false;
  }

  void scalarizeLoop(AffineForOp forOp) {
    auto &bodyOps = forOp.getBody()->getOperations();
    
    // Find load, store, and init
    AffineLoadOp loadOp = nullptr;
    AffineStoreOp storeOp = nullptr;
    
    for (auto &op : bodyOps) {
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (!loadOp) loadOp = load;
      }
      if (auto store = dyn_cast<AffineStoreOp>(op)) {
        storeOp = store;
      }
    }
    
    // Find init store
    AffineStoreOp initStore = nullptr;
    Operation *prevOp = forOp->getPrevNode();
    while (prevOp) {
      if (auto store = dyn_cast<AffineStoreOp>(prevOp)) {
        if (store.getMemRef() == loadOp.getMemRef()) {
          initStore = store;
          break;
        }
      }
      prevOp = prevOp->getPrevNode();
    }
    
    if (!loadOp || !storeOp || !initStore) return;
    
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value initVal = initStore.getValueToStore();
    Type elementType = initVal.getType();
    
    // Create new loop with iter_args using the simple integer bounds builder
    auto newForOp = builder.create<AffineForOp>(
        loc,
        forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStepAsInt(),
        ValueRange{initVal},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          // Map old IV and load result to new  values
          IRMapping mapping;
          mapping.map(forOp.getInductionVar(), iv);
          mapping.map(loadOp.getResult(), iterArgs[0]);  // accum from iter_arg
          
          // Clone body ops except load and store
          Value yieldValue;
          for (auto &op : bodyOps) {
            if (&op == loadOp.getOperation()) {
              continue;  // Skip load
            } else if (&op == storeOp.getOperation()) {
              // Capture the value that would be stored
              yieldValue = mapping.lookupOrDefault(storeOp.getValueToStore());
            } else if (!isa<AffineYieldOp>(op)) {
              b.clone(op, mapping);
            }
          }
          
          // Yield the updated accumulator
          b.create<AffineYieldOp>(loc, ValueRange{yieldValue});
        });
    
    // Store final result after loop
    builder.setInsertionPointAfter(newForOp);
    builder.create<AffineStoreOp>(
        loc, newForOp.getResult(0), storeOp.getMemRef(), storeOp.getIndices());
    
    // Erase old operations
    forOp.erase();
    initStore.erase();
  }

  StringRef getArgument() const final { return "affine-scalarize-accumulator"; }

  StringRef getDescription() const final {
    return "Convert buffer-based loop accumulation to register-based using iter_args";
  }
};

} // namespace

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createAffineScalarizeAccumulatorPass() {
  return std::make_unique<AffineScalarizeAccumulatorPass>();
}

} // namespace nova
} // namespace mlir
