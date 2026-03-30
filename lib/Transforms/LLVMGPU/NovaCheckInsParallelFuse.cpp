//===- NovaCheckInsParallelFuse.cpp ---------------------------------------===//
//
// Fuses parallel linalg.generic operations that share the same inputs
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nova-fuse-parallels-with-same-ins"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::nova {

struct FuseParallelGenericsWithSameInputs
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.isAllParallelLoops())
      return failure();

    // collect input values.
    SmallVector<Value> inputs;
    for (OpOperand *opOperand : genericOp.getDpsInputOperands())
      inputs.push_back(opOperand->get());

    if (inputs.empty())
      return failure();

    // scan forward in the block (ordered)
    Block *block = genericOp->getBlock();
    linalg::GenericOp candidate = nullptr;

    for (Operation &op : llvm::make_range(std::next(genericOp->getIterator()), block->end())) {
      auto other = dyn_cast<linalg::GenericOp>(&op);
      if (!other || !other.isAllParallelLoops())
        continue;

      // Must share the same number of inputs  
      if (other.getNumDpsInputs() != inputs.size())
        continue;
      
      // All input values must be identical
      bool inputsMatch = true;
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (other.getDpsInputOperand(i)->get() != inputs[i]) {
          inputsMatch = false;
          break;
        }
      }
      if (!inputsMatch)
        continue;

      // Guard against op dominance: candidate must not consume genericOp (firstOp) results.
      bool hasDep = false;
      for (Value res : genericOp->getResults()) {
        for (size_t i = 0; i < inputs.size(); ++i) {
          if (other.getDpsInputOperand(i)->get() == res) {
            hasDep = true;
            break;
          }
        }
        if (hasDep) break;
      }
      if (hasDep)
        continue;

      // Iterator types must match
      if (genericOp.getIteratorTypesArray() != other.getIteratorTypesArray())
        continue;

      candidate = other;
      break;
    }

    if (!candidate)
      return failure();

    return fuseOperations(genericOp, candidate, rewriter);
  }

private:
  LogicalResult fuseOperations(linalg::GenericOp op1, linalg::GenericOp op2,
                               PatternRewriter &rewriter) const {
    int nIns       = op1.getNumDpsInputs();
    int op1NOuts   = op1.getNumDpsInits();
    int op2NOuts   = op2.getNumDpsInits();

    // result types for both ops.
    SmallVector<Type> resultTypes;
    llvm::append_range(resultTypes, op1->getResultTypes());
    llvm::append_range(resultTypes, op2->getResultTypes());

    // ins maps (shared) + op1 out maps + op2 out maps = 3 maps.
    SmallVector<AffineMap> op1Maps = op1.getIndexingMapsArray();
    SmallVector<AffineMap> op2Maps = op2.getIndexingMapsArray();
    SmallVector<AffineMap> fusedMaps;
    for (int i = 0; i < nIns; ++i)
      fusedMaps.push_back(op1Maps[i]);
    for (int i = nIns; i < nIns + op1NOuts; ++i)
      fusedMaps.push_back(op1Maps[i]);
    for (int i = nIns; i < nIns + op2NOuts; ++i)
      fusedMaps.push_back(op2Maps[i]);

    // Fused operands: shared ins, then op1 inits, then op2 inits.
    SmallVector<Value> fusedIns, fusedOuts;
    for (OpOperand *o : op1.getDpsInputOperands())
      fusedIns.push_back(o->get());
    for (Value v : op1.getDpsInits())
      fusedOuts.push_back(v);
    for (Value v : op2.getDpsInits())
      fusedOuts.push_back(v);

    // op2's init-defining ops may appear after op1 in the block.
    // Move them before op1 so the fused op (inserted at op1's position) can use them.
    for (Value v : op2.getDpsInits()) {
      if (Operation *defOp = v.getDefiningOp())
        rewriter.moveOpBefore(defOp, op1);
    }

    Block *block1 = op1.getBody();
    Block *block2 = op2.getBody();

    // Create the fused generic using a body builder so the block arguments are
    // available while we clone the two original bodies
    auto fusedOp = rewriter.create<linalg::GenericOp>(
        op1.getLoc(), resultTypes, fusedIns,
        fusedOuts, fusedMaps, op1.getIteratorTypesArray(),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args layout: [ins(0..nIns), op1_outs(nIns..nIns+op1NOuts),
          //               op2_outs(nIns+op1NOuts..end)]

          IRMapping map1, map2;

          // Map block1 args: ins + outs → fused args directly.
          for (int i = 0; i < nIns + op1NOuts; ++i)
            map1.map(block1->getArgument(i), args[i]);

          // Map block2 ins args → same shared fused ins args.
          for (int i = 0; i < nIns; ++i)
            map2.map(block2->getArgument(i), args[i]);
          // Map block2 out args → fused args after op1's outs.
          for (int i = 0; i < op2NOuts; ++i)
            map2.map(block2->getArgument(nIns + i), args[nIns + op1NOuts + i]);

          // Clone op1 body, collect its yield values.
          SmallVector<Value> yields1;
          for (Operation &op : *block1) {
            if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
              for (Value v : yield.getValues())
                yields1.push_back(map1.lookupOrDefault(v));
              continue;
            }
            b.clone(op, map1);
          }

          // Clone op2 body, collect its yield values.
          SmallVector<Value> yields2;
          for (Operation &op : *block2) {
            if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
              for (Value v : yield.getValues())
                yields2.push_back(map2.lookupOrDefault(v));
              continue;
            }
            b.clone(op, map2);
          }

          // Single combined yield for all outputs.
          SmallVector<Value> allYields;
          llvm::append_range(allYields, yields1);
          llvm::append_range(allYields, yields2);
          b.create<linalg::YieldOp>(loc, allYields);
        });

    // Wire fused results back to original consumers
    int op1NResults = op1->getNumResults();
    for (auto [oldRes, newRes] :
         llvm::zip(op1->getResults(),
                   fusedOp->getResults().take_front(op1NResults)))
      rewriter.replaceAllUsesWith(oldRes, newRes);

    for (auto [oldRes, newRes] :
         llvm::zip(op2->getResults(),
                   fusedOp->getResults().drop_front(op1NResults)))
      rewriter.replaceAllUsesWith(oldRes, newRes);

    rewriter.eraseOp(op2);
    rewriter.eraseOp(op1);
    return success();
  }
};

struct NovaCheckInsParallelFuse
    : public PassWrapper<NovaCheckInsParallelFuse, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaCheckInsParallelFuse)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<FuseParallelGenericsWithSameInputs>(funcOp.getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      return signalPassFailure();
  }

  StringRef getArgument() const override { return "nova-parallel-ins-fusion"; }
  StringRef getDescription() const override {
    return "Fuses parallel linalg generics with same inputs";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaCheckInsParallelFuse() {
  return std::make_unique<NovaCheckInsParallelFuse>();
}

void registerNovaCheckInsParallelFuse() {
  PassRegistration<NovaCheckInsParallelFuse>();
}

} // namespace mlir::nova
