/// NovaLinalVerticalFusion.cpp
///
/// Vertical fusion: lower a reduction linalg.generic AND its single
/// all-parallel consumer into one fused SCF loop nest.
///
/// The outer loops iterate over the parallel (non-reduced) dimensions,
/// carrying the output tensor.  The inner loops iterate over the reduction
/// dimensions, carrying a scalar accumulator.  After the inner loops produce
/// the reduced scalar, the elementwise consumer body is applied to it
/// in-place before storing the result — no intermediate tensor is needed.
///
/// Three elemental cases are handled uniformly via the affine-map evaluator:
///
///   Case 1 – unary  :  elemOp has one input  → the reduce result
///   Case 2 – binary :  elemOp has two inputs → reduce result + a tensor
///                       whose indexing map may reach into the original input
///   Case 3 – binary :  same as case 2 but with an independent tensor
///
/// Fused shape for reduce-then-sin  (8×8 → 8 → 8):
///
///   %r = scf.for %i0 = 0 to 8 step 1 iter_args(%t = %redInit) {
///     %acc0  = tensor.extract %t[%i0]           // identity (0.0 for sum)
///     %acc1  = scf.for %i1 = 0 to 8 step 1 iter_args(%a = %acc0) {
///                %val  = tensor.extract %input[%i0,%i1]
///                %nacc = arith.addf %val, %a
///                scf.yield %nacc
///              }
///     %sinv  = math.sin %acc1                   // elementwise body inlined
///     %t2    = tensor.insert %sinv into %t[%i0]
///     scf.yield %t2
///   }

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace nova {

// ── Affine-expression evaluator ──────────────────────────────────────────────

static Value evalAffineExpr(OpBuilder &b, Location loc, AffineExpr expr,
                             ArrayRef<Value> ivs) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return ivs[dim.getPosition()];

  if (auto cst = dyn_cast<AffineConstantExpr>(expr))
    return b.create<arith::ConstantIndexOp>(loc, cst.getValue());

  auto bin = cast<AffineBinaryOpExpr>(expr);
  Value lhs = evalAffineExpr(b, loc, bin.getLHS(), ivs);
  Value rhs = evalAffineExpr(b, loc, bin.getRHS(), ivs);
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  case AffineExprKind::Mul:
    return b.create<arith::MulIOp>(loc, lhs, rhs);
  default:
    llvm_unreachable("unsupported affine expr kind in vertical fusion lowering");
  }
}

static SmallVector<Value> evalAffineMap(OpBuilder &b, Location loc,
                                        AffineMap map, ArrayRef<Value> ivs) {
  SmallVector<Value> indices;
  for (AffineExpr expr : map.getResults())
    indices.push_back(evalAffineExpr(b, loc, expr, ivs));
  return indices;
}

// ── Dominance helper ─────────────────────────────────────────────────────────

static void hoistBeforeOp(Operation *op, Operation *target) {
  if (!op || op->getBlock() != target->getBlock())
    return;
  if (op->isBeforeInBlock(target))
    return;
  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      hoistBeforeOp(defOp, target);
  op->moveBefore(target);
}

// ── Predicate helpers ────────────────────────────────────────────────────────

static bool isAllParallelGeneric(linalg::GenericOp op) {
  return llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::parallel;
  });
}

static bool hasReductionIterator(linalg::GenericOp op) {
  return llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::reduction;
  });
}

// ── Core lowering + fusion ───────────────────────────────────────────────────

static LogicalResult lowerFusedReductionElemToSCF(linalg::GenericOp reduceOp,
                                                   linalg::GenericOp elemOp) {
  OpBuilder builder(reduceOp);
  Location loc = reduceOp.getLoc();

  // ── Reduction op metadata ────────────────────────────────────────────────
  auto iterTypes  = reduceOp.getIteratorTypesArray();
  auto loopRanges = reduceOp.getStaticLoopRanges();
  int64_t numLoops = (int64_t)iterTypes.size();

  SmallVector<int64_t> parallelDims, reductionDims;
  for (int64_t i = 0; i < numLoops; i++) {
    if (iterTypes[i] == utils::IteratorType::parallel)
      parallelDims.push_back(i);
    else
      reductionDims.push_back(i);
  }

  for (int64_t r : loopRanges)
    if (r == ShapedType::kDynamic)
      return failure();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> bounds(numLoops);
  for (int64_t i = 0; i < numLoops; i++)
    bounds[i] = builder.create<arith::ConstantIndexOp>(loc, loopRanges[i]);

  auto reduceMaps   = reduceOp.getIndexingMapsArray();
  auto reduceInputs = reduceOp.getDpsInputs();
  int64_t numReduceInputs = (int64_t)reduceInputs.size();
  SmallVector<AffineMap> reduceInputMaps(reduceMaps.begin(),
                                         reduceMaps.begin() + numReduceInputs);
  AffineMap reduceOutputMap = reduceMaps[numReduceInputs];

  Value initTensor  = reduceOp.getDpsInits()[0];
  Block *reduceBody = reduceOp.getBody();

  // ── Elementwise op metadata ──────────────────────────────────────────────
  auto elemMaps      = elemOp.getIndexingMapsArray();
  auto elemInputs    = elemOp.getDpsInputs();
  int64_t numElemInputs = (int64_t)elemInputs.size();
  SmallVector<AffineMap> elemInputMaps(elemMaps.begin(),
                                       elemMaps.begin() + numElemInputs);
  AffineMap elemOutputMap = elemMaps[numElemInputs];
  Block *elemBody = elemOp.getBody();

  // ── Dominance fixup ──────────────────────────────────────────────────────
  for (Value val : elemInputs) {
    if (val == reduceOp.getResult(0))
      continue;
    if (Operation *defOp = val.getDefiningOp())
      hoistBeforeOp(defOp, reduceOp);
  }

  // ── Shared IV array ──────────────────────────────────────────────────────
  // ivs[d] holds the live loop IV for reduction-op logical dimension d.
  // Initialised to c0; entries are overwritten as each loop opens.
  SmallVector<Value> ivs(numLoops, c0);

  auto makeElemIVs = [&]() {
    SmallVector<Value> ev;
    for (int64_t d : parallelDims)
      ev.push_back(ivs[d]);
    return ev;
  };

  // ── Inner: reduction loop nest (scf.for, carries scalar accumulator) ─────
  std::function<Value(OpBuilder &, int64_t, Value)> buildReductionLoops;
  buildReductionLoops = [&](OpBuilder &b, int64_t ridx, Value acc) -> Value {

    if (ridx == (int64_t)reductionDims.size()) {
      IRMapping mapping;
      for (int64_t i = 0; i < numReduceInputs; i++) {
        SmallVector<Value> idx = evalAffineMap(b, loc, reduceInputMaps[i], ivs);
        Value val = b.create<tensor::ExtractOp>(loc, reduceInputs[i], idx);
        mapping.map(reduceBody->getArgument(i), val);
      }
      mapping.map(reduceBody->getArgument(numReduceInputs), acc);

      for (auto &op : reduceBody->without_terminator())
        b.clone(op, mapping);

      auto yield = cast<linalg::YieldOp>(reduceBody->getTerminator());
      return mapping.lookupOrDefault(yield.getOperand(0));
    }

    int64_t dim = reductionDims[ridx];
    auto loop = b.create<scf::ForOp>(
        loc, c0, bounds[dim], c1, ValueRange{acc},
        [&](OpBuilder &ib, Location, Value iv, ValueRange args) {
          ivs[dim] = iv;
          Value newAcc = buildReductionLoops(ib, ridx + 1, args[0]);
          ib.create<scf::YieldOp>(loc, newAcc);
        });
    return loop.getResult(0);
  };

  // ── Outer: parallel loop nest (scf.for, carries output tensor) ───────────
  //
  // We use scf.for rather than scf.parallel because we need to carry the
  // output tensor through iter_args.  scf.parallel only supports scalar
  // reductions via scf.reduce — it cannot thread an evolving tensor value
  // across iterations.  The tensor updates (tensor.insert) are sequential
  // in the SSA value chain; the underlying bufferization or a later
  // vectorisation pass can exploit parallelism if it is safe to do so.
  std::function<Value(OpBuilder &, int64_t, Value)> buildParallelLoops;
  buildParallelLoops = [&](OpBuilder &b, int64_t pidx,
                            Value currentTensor) -> Value {

    if (pidx == (int64_t)parallelDims.size()) {
      // All parallel IVs are live in `ivs`.

      // Step 1: seed the scalar accumulator from the init tensor.
      SmallVector<Value> redOutIdx = evalAffineMap(b, loc, reduceOutputMap, ivs);
      Value initAcc = b.create<tensor::ExtractOp>(loc, currentTensor, redOutIdx);

      // Step 2: run the reduction nest → reduced scalar.
      Value finalAcc = buildReductionLoops(b, 0, initAcc);

      // Step 3: inline the elementwise body.
      SmallVector<Value> elemIVs = makeElemIVs();

      IRMapping elemMapping;
      for (int64_t i = 0; i < numElemInputs; i++) {
        Value argVal;
        if (elemInputs[i] == reduceOp.getResult(0)) {
          argVal = finalAcc;
        } else {
          SmallVector<Value> idx =
              evalAffineMap(b, loc, elemInputMaps[i], elemIVs);
          argVal = b.create<tensor::ExtractOp>(loc, elemInputs[i], idx);
        }
        elemMapping.map(elemBody->getArgument(i), argVal);
      }

      SmallVector<Value> elemOutIdx =
          evalAffineMap(b, loc, elemOutputMap, elemIVs);
      Value outArgVal =
          b.create<tensor::ExtractOp>(loc, currentTensor, elemOutIdx);
      elemMapping.map(elemBody->getArgument(numElemInputs), outArgVal);

      for (auto &op : elemBody->without_terminator())
        b.clone(op, elemMapping);

      auto elemYield = cast<linalg::YieldOp>(elemBody->getTerminator());
      Value fusedResult = elemMapping.lookupOrDefault(elemYield.getOperand(0));

      // Step 4: store fused scalar into the output tensor.
      return b.create<tensor::InsertOp>(loc, fusedResult, currentTensor,
                                        elemOutIdx);
    }

    // Emit one scf.for loop for this parallel dimension.
    // iter_args carries the evolving output tensor; the loop IV is written
    // into ivs[dim] so that evalAffineMap can see it from nested lambdas.
    int64_t dim = parallelDims[pidx];
    auto loop = b.create<scf::ForOp>(
        loc, c0, bounds[dim], c1,
        /*iterArgs=*/ValueRange{currentTensor},
        [&](OpBuilder &ib, Location, Value iv, ValueRange args) {
          ivs[dim] = iv;   // publish IV into the shared ivs array
          Value newTensor = buildParallelLoops(ib, pidx + 1, args[0]);
          ib.create<scf::YieldOp>(loc, newTensor);
        });
    return loop.getResult(0);
  };

  // ── Emit and wire up ──────────────────────────────────────────────────────
  Value fusedResult = buildParallelLoops(builder, 0, initTensor);

  elemOp.getResult(0).replaceAllUsesWith(fusedResult);
  elemOp->erase();
  reduceOp->erase();
  return success();
}

// ── Pass definition ──────────────────────────────────────────────────────────

struct NovaLinalgVerticalFusionPass
    : public PassWrapper<NovaLinalgVerticalFusionPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaLinalgVerticalFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  StringRef getArgument()    const final { return "nova-linalg-vertical-fusion"; }
  StringRef getDescription() const final {
    return "Fuse a reduction linalg.generic with its single parallel consumer "
           "into one SCF loop nest (vertical fusion)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>> candidates;

    func.walk([&](linalg::GenericOp op) {
      if (!hasReductionIterator(op))
        return;
      if (op->getNumResults() != 1)
        return;
      if (!op->getResult(0).hasOneUse())
        return;
      Operation *user = *op->getResult(0).getUsers().begin();
      auto parallelOp = dyn_cast<linalg::GenericOp>(user);
      if (!parallelOp || !isAllParallelGeneric(parallelOp))
        return;

      int64_t numParallelDims = llvm::count_if(
          op.getIteratorTypesArray(), [](utils::IteratorType t) {
            return t == utils::IteratorType::parallel;
          });
      int64_t numElemLoops =
          (int64_t)parallelOp.getIteratorTypesArray().size();
      if (numElemLoops != numParallelDims)
        return;

      candidates.emplace_back(op, parallelOp);
    });

    for (auto &[reduceOp, elemOp] : candidates) {
      if (failed(lowerFusedReductionElemToSCF(reduceOp, elemOp))) {
        reduceOp.emitError("nova-vertical-fusion: failed to fuse and lower");
        signalPassFailure();
        return;
      }
    }
  }
};

// ── Entry points ─────────────────────────────────────────────────────────────

std::unique_ptr<Pass> createNovaLinalgVerticalFusionPass() {
  return std::make_unique<NovaLinalgVerticalFusionPass>();
}

void registerNovaLinalgVerticalFusionPass() {
  PassRegistration<NovaLinalgVerticalFusionPass>();
}

} // namespace nova
} // namespace mlir