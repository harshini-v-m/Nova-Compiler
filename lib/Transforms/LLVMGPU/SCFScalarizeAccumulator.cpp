#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Pass to scalarize loop-carried accumulation from buffer to registers in scf.for loops
struct SCFScalarizeAccumulatorPass
    : public PassWrapper<SCFScalarizeAccumulatorPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFScalarizeAccumulatorPass)

  void runOnOperation() override {
    auto func = getOperation();
    
    // Collect loops to transform (to avoid iterator invalidation)
    SmallVector<scf::ForOp> loopsToTransform;
    
    func.walk([&](scf::ForOp forOp) {
      if (shouldScalarize(forOp))
        loopsToTransform.push_back(forOp);
    });
    
    // Transform collected loops
    for (auto forOp : loopsToTransform) {
      scalarizeLoop(forOp);
    }
  }

  bool shouldScalarize(scf::ForOp forOp) {
    auto &bodyOps = forOp.getBody()->getOperations();
    if (bodyOps.empty()) return false;
    
    // Find the last store
    memref::StoreOp storeOp = nullptr;
    for (auto &op : llvm::reverse(bodyOps)) {
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        storeOp = store;
        break;
      }
    }
    if (!storeOp) return false;
    
    // Find matching load to the same location
    memref::LoadOp loadOp = nullptr;
    for (auto &op : bodyOps) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (load.getMemRef() == storeOp.getMemRef() && 
            load.getIndices().size() == storeOp.getIndices().size()) {
          // Check if indices match exactly
          bool indicesMatch = true;
          for (auto zip : llvm::zip(load.getIndices(), storeOp.getIndices())) {
            if (std::get<0>(zip) != std::get<1>(zip)) {
              indicesMatch = false;
              break;
            }
          }
          if (indicesMatch) {
            loadOp = load;
            break;
          }
        }
      }
    }
    
    if (!loadOp) return false;
    
    // Check indices are loop-invariant
    Value loopIV = forOp.getInductionVar();
    for (Value idx : loadOp.getIndices()) {
      if (idx == loopIV) return false;
    }
    
    return true;
  }

  void scalarizeLoop(scf::ForOp forOp) {
    auto &bodyOps = forOp.getBody()->getOperations();
    
    memref::StoreOp storeOp = nullptr;
    for (auto &op : llvm::reverse(bodyOps)) {
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        storeOp = store;
        break;
      }
    }
    if (!storeOp) return;

    memref::LoadOp loadOp = nullptr;
    for (auto &op : bodyOps) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (load.getMemRef() == storeOp.getMemRef() && 
            load.getIndices().size() == storeOp.getIndices().size()) {
          bool indicesMatch = true;
          for (auto zip : llvm::zip(load.getIndices(), storeOp.getIndices())) {
            if (std::get<0>(zip) != std::get<1>(zip)) {
              indicesMatch = false;
              break;
            }
          }
          if (indicesMatch) {
            loadOp = load;
            break;
          }
        }
      }
    }
    
    if (!loadOp || !storeOp) return;

    bool isShared = false;
    if (auto forallOp = forOp->getParentOfType<scf::ForallOp>()) {
      Operation *defOp = storeOp.getMemRef().getDefiningOp();
      if (!defOp) {
        if (auto arg = llvm::dyn_cast<BlockArgument>(storeOp.getMemRef())) {
          defOp = arg.getOwner()->getParentOp();
        }
      }
      if (defOp && !forallOp->isAncestor(defOp)) {
        isShared = true;
      }
    } else if (auto launchOp = forOp->getParentOfType<gpu::LaunchOp>()) {
      Operation *defOp = storeOp.getMemRef().getDefiningOp();
      if (!defOp) {
        if (auto arg = llvm::dyn_cast<BlockArgument>(storeOp.getMemRef())) {
          defOp = arg.getOwner()->getParentOp();
        }
      }
      if (defOp && !launchOp->isAncestor(defOp)) {
        isShared = true;
      }
    }

    arith::AtomicRMWKind atomicKind = arith::AtomicRMWKind::addf;
    for (auto &op : bodyOps) {
      if (isa<arith::AddFOp>(op)) { atomicKind = arith::AtomicRMWKind::addf; break; }
      else if (isa<arith::MulFOp>(op)) { atomicKind = arith::AtomicRMWKind::mulf; break; }
      else if (isa<arith::MaxNumFOp>(op)) { atomicKind = arith::AtomicRMWKind::maximumf; break; }
      else if (isa<arith::MinNumFOp>(op)) { atomicKind = arith::AtomicRMWKind::minimumf; break; }
    }

    // Find init store (optional)
    memref::StoreOp initStore = nullptr;
    if (!isShared) {
      Operation *prevOp = forOp->getPrevNode();
      while (prevOp) {
        if (auto store = dyn_cast<memref::StoreOp>(prevOp)) {
          if (store.getMemRef() == loadOp.getMemRef() && 
              store.getIndices().size() == loadOp.getIndices().size()) {
            bool indicesMatch = true;
            for (auto zip : llvm::zip(store.getIndices(), loadOp.getIndices())) {
              if (std::get<0>(zip) != std::get<1>(zip)) {
                indicesMatch = false;
                break;
              }
            }
            if (indicesMatch) {
              initStore = store;
              break;
            }
          }
        }
        if (prevOp->getBlock() != forOp->getBlock()) break;
        prevOp = prevOp->getPrevNode();
      }
    }
    
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value initVal;
    
    if (isShared) {
      auto elemTy = llvm::cast<MemRefType>(storeOp.getMemRef().getType()).getElementType();
      if (atomicKind == arith::AtomicRMWKind::mulf) {
        if (elemTy.isF32()) initVal = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(1.0f));
        else if (elemTy.isF16()) initVal = builder.create<arith::ConstantOp>(loc, builder.getF16FloatAttr(1.0f));
        else initVal = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(elemTy, 1.0f));
      } else {
        if (elemTy.isF32()) initVal = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
        else if (elemTy.isF16()) initVal = builder.create<arith::ConstantOp>(loc, builder.getF16FloatAttr(0.0f));
        else initVal = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(elemTy, 0.0f));
      }
    } else if (initStore) {
      initVal = initStore.getValueToStore();
    } else {
      initVal = builder.create<memref::LoadOp>(
          loc, loadOp.getMemRef(), loadOp.getIndices());
    }
    
    SmallVector<Value> iterArgs = forOp.getInitArgs();
    iterArgs.push_back(initVal);
    
    auto newForOp = builder.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        iterArgs);
    
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      mapping.map(forOp.getRegionIterArg(i), newForOp.getRegionIterArg(i));
    }
    
    Value newIterArg = newForOp.getRegionIterArgs().back();
    mapping.map(loadOp.getResult(), newIterArg);
    
    builder.setInsertionPointToStart(newForOp.getBody());
    
    Value yieldValue;
    for (auto &op : bodyOps) {
      if (&op == loadOp.getOperation()) {
        continue;
      } else if (&op == storeOp.getOperation()) {
        yieldValue = mapping.lookupOrDefault(storeOp.getValueToStore());
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        SmallVector<Value> yields;
        for (Value v : yieldOp.getOperands()) {
          yields.push_back(mapping.lookupOrDefault(v));
        }
        if (yieldValue) yields.push_back(yieldValue);
        builder.create<scf::YieldOp>(loc, yields);
      } else {
        builder.clone(op, mapping);
      }
    }
    
    builder.setInsertionPointAfter(newForOp);
    if (isShared) {
      builder.create<memref::AtomicRMWOp>(
          loc, atomicKind, newForOp.getResults().back(),
          storeOp.getMemRef(), storeOp.getIndices());
    } else {
      builder.create<memref::StoreOp>(
          loc, newForOp.getResults().back(), storeOp.getMemRef(), storeOp.getIndices());
    }
    
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    }
    
    forOp.erase();
    if (initStore) initStore.erase();
  }

  StringRef getArgument() const final { return "scf-scalarize-accumulator"; }

  StringRef getDescription() const final {
    return "Convert buffer-based scf.for loop accumulation to register-based using iter_args";
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
