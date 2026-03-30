//===- NovaRepositionStore.cpp ----------------------------------===//
//
// Repositions store operations to prevent register spilling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "nova-reposition-store"

using namespace mlir;

namespace mlir::nova {

// Returns the direct child of `block` that is an ancestor of `op`,
// or nullptr if `op` is not nested under `block`.
static Operation *ancestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

static bool repositionStoresInBlock(Block *block, OpBuilder &builder) {
  bool changed = false;

  // Snapshot to avoid iterator invalidation during erasure.
  SmallVector<memref::StoreOp> stores;
  for (Operation &op : *block)
    if (auto st = dyn_cast<memref::StoreOp>(op))
      stores.push_back(st);

  for (memref::StoreOp storeOp : stores) {
    if (!storeOp->getBlock())
      continue;

    // `insertAfter` tracks the latest op (in block order) that the new store
    // must follow. We advance it for every dependency:
    //   1. def of storedVal
    //   2. last non-store use of storedVal in this block (before the store)
    //   3. defs of the memref and index operands
    // This guarantees all operands dominate the new insertion point.
    Operation *insertAfter = nullptr;
    auto advance = [&](Operation *op) {
      if (!op || op->getBlock() != block)
        return;
      if (!insertAfter || insertAfter->isBeforeInBlock(op))
        insertAfter = op;
    };

    Value storedVal = storeOp.getValueToStore();

    // (1) def of the stored value itself
    if (Operation *def = storedVal.getDefiningOp())
      advance(ancestorInBlock(def, block));

    // (2) last use of storedVal before the store (uses after the store mean
    //     the register must stay live anyway, so we skip those)
    for (Operation *user : storedVal.getUsers()) {
      if (user == storeOp.getOperation())
        continue;
      Operation *anc = ancestorInBlock(user, block);
      if (anc && anc->isBeforeInBlock(storeOp))
        advance(anc);
    }

    // (3) defs of memref and index operands
    for (Value operand : storeOp->getOperands()) {
      if (operand == storedVal)
        continue;
      if (Operation *def = operand.getDefiningOp())
        advance(ancestorInBlock(def, block));
    }

    if (!insertAfter)
      continue;
    // insertAfter must be strictly before the store; if it is not, a use/def
    // of some operand lies after the store — don't move.
    if (!insertAfter->isBeforeInBlock(storeOp))
      continue;
    // Store is already packed: the gap between insertAfter and storeOp
    // contains only other memref.store ops. Moving would cause the stores to
    // leapfrog with no register pressure benefit.
    bool alreadyPacked = true;
    for (Operation *it = insertAfter->getNextNode();
         it != storeOp.getOperation(); it = it->getNextNode()) {
      if (!isa<memref::StoreOp>(it)) {
        alreadyPacked = false;
        break;
      }
    }
    if (alreadyPacked)
      continue;

    // Alias safety: no load from the same memref between the new position
    // and the original store (conservative: walks nested regions too).
    Value buf = storeOp.getMemref();
    bool aliased = false;
    for (Operation *it = insertAfter->getNextNode();
         it != storeOp.getOperation(); it = it->getNextNode()) {
      it->walk([&](memref::LoadOp load) {
        if (load.getMemref() == buf)
          aliased = true;
      });
      if (aliased)
        break;
    }
    if (aliased)
      continue;

    builder.setInsertionPointAfter(insertAfter);
    builder.create<memref::StoreOp>(storeOp.getLoc(), storedVal,
                                    storeOp.getMemref(), storeOp.getIndices());
    storeOp->erase();
    changed = true;
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaRepositionStorePass
    : public PassWrapper<NovaRepositionStorePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaRepositionStorePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Collect unique blocks with store ops before any modifications.
    llvm::SmallPtrSet<Block *, 16> blockSet;
    funcOp.walk([&](memref::StoreOp st) { blockSet.insert(st->getBlock()); });

    for (Block *block : blockSet)
      repositionStoresInBlock(block, builder);
  }

  StringRef getArgument() const override { return "nova-reposition-store"; }

  StringRef getDescription() const override {
    return "Repositions store operations to prevent register spilling";
  }
};

std::unique_ptr<Pass> createNovaRepositionStorePass() {
  return std::make_unique<NovaRepositionStorePass>();
}

void registerNovaRepositionStorePass() {
  PassRegistration<NovaRepositionStorePass>();
}

} // namespace mlir::nova
