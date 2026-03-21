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
// Recursively walks an AffineExpr and emits arith ops to compute its value
// at runtime, using the supplied induction-variable array indexed by dim.

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

// Evaluate every result expression of an AffineMap at the given IVs.
static SmallVector<Value> evalAffineMap(OpBuilder &b, Location loc,
                                        AffineMap map, ArrayRef<Value> ivs) {
  SmallVector<Value> indices;
  for (AffineExpr expr : map.getResults())
    indices.push_back(evalAffineExpr(b, loc, expr, ivs));
  return indices;
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
// Lowers `reduceOp` (reduction linalg.generic) and its single all-parallel
// consumer `elemOp` into one fused SCF loop nest.
//
// The reduction body is cloned into the innermost loop.  Immediately after
// the inner loops return the reduced scalar, the elementwise body is cloned
// at the same insertion point — no intermediate tensor is allocated.
//
// elemOp inputs fall into two categories:
//   • The operand that IS reduceOp's result  → replaced by the reduced scalar.
//   • Every other operand (independent tensor) → extracted at the current
//     parallel-IVs position using that operand's affine map in elemOp.
//
// The outer loop carries the reduction's init tensor (filled with the
// identity value) so that:
//   1. The per-position identity scalar can be extracted to seed the inner
//      accumulator without an extra constant-extraction step.
//   2. The same tensor becomes the fused output: the elementwise result is
//      inserted back, overwriting the identity placeholder.

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

  // ── Shared IV array ──────────────────────────────────────────────────────
  // ivs[d] = the loop IV for reduction-op logical dimension d.
  // Parallel dims are filled as each outer loop opens; initialised to c0
  // so that evalAffineMap on the output map is always safe.
  SmallVector<Value> ivs(numLoops, c0);

  // elemIVs[i] = ivs[parallelDims[i]].  These are the loop IVs as seen by
  // the elementwise op (its iteration space == the reduction's output shape).
  // We return a fresh snapshot whenever needed so the lambda always sees the
  // up-to-date Values even after ivs is mutated.
  auto makeElemIVs = [&]() {
    SmallVector<Value> ev;
    for (int64_t d : parallelDims)
      ev.push_back(ivs[d]);
    return ev;
  };

  // ── Inner: reduction loop nest ───────────────────────────────────────────
  std::function<Value(OpBuilder &, int64_t, Value)> buildReductionLoops;
  buildReductionLoops = [&](OpBuilder &b, int64_t ridx, Value acc) -> Value {

    if (ridx == (int64_t)reductionDims.size()) {
      // Leaf: clone reduction body once at the current (ivs) position.
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

  // ── Outer: parallel loop nest ────────────────────────────────────────────
  std::function<Value(OpBuilder &, int64_t, Value)> buildParallelLoops;
  buildParallelLoops = [&](OpBuilder &b, int64_t pidx,
                            Value currentTensor) -> Value {

    if (pidx == (int64_t)parallelDims.size()) {
      // All parallel IVs are live.

      // ── Step 1: seed the scalar accumulator from the init tensor ─────────
      SmallVector<Value> redOutIdx = evalAffineMap(b, loc, reduceOutputMap, ivs);
      Value initAcc = b.create<tensor::ExtractOp>(loc, currentTensor, redOutIdx);

      // ── Step 2: run the reduction nest → reduced scalar ──────────────────
      Value finalAcc = buildReductionLoops(b, 0, initAcc);

      // ── Step 3: inline the elementwise body ──────────────────────────────
      // Build the IV snapshot for the elementwise op's affine maps.
      SmallVector<Value> elemIVs = makeElemIVs();

      IRMapping elemMapping;
      for (int64_t i = 0; i < numElemInputs; i++) {
        Value argVal;
        if (elemInputs[i] == reduceOp.getResult(0)) {
          // Case 1 / 2: this input IS the reduction result → use the scalar.
          argVal = finalAcc;
        } else {
          // Case 2 / 3: independent tensor → extract at current position.
          SmallVector<Value> idx =
              evalAffineMap(b, loc, elemInputMaps[i], elemIVs);
          argVal = b.create<tensor::ExtractOp>(loc, elemInputs[i], idx);
        }
        elemMapping.map(elemBody->getArgument(i), argVal);
      }

      // Map the output block-arg to the current tensor value at that slot.
      // (Rarely used in pure-elementwise bodies; safe to always provide it.)
      SmallVector<Value> elemOutIdx =
          evalAffineMap(b, loc, elemOutputMap, elemIVs);
      Value outArgVal =
          b.create<tensor::ExtractOp>(loc, currentTensor, elemOutIdx);
      elemMapping.map(elemBody->getArgument(numElemInputs), outArgVal);

      for (auto &op : elemBody->without_terminator())
        b.clone(op, elemMapping);

      auto elemYield = cast<linalg::YieldOp>(elemBody->getTerminator());
      Value fusedResult = elemMapping.lookupOrDefault(elemYield.getOperand(0));

      // ── Step 4: store the fused scalar into the output tensor ─────────────
      return b.create<tensor::InsertOp>(loc, fusedResult, currentTensor,
                                        elemOutIdx);
    }

    // Emit one parallel loop carrying the tensor and recurse inside.
    int64_t dim = parallelDims[pidx];
    auto loop = b.create<scf::ForOp>(
        loc, c0, bounds[dim], c1, ValueRange{currentTensor},
        [&](OpBuilder &ib, Location, Value iv, ValueRange args) {
          ivs[dim] = iv;
          Value newTensor = buildParallelLoops(ib, pidx + 1, args[0]);
          ib.create<scf::YieldOp>(loc, newTensor);
        });
    return loop.getResult(0);
  };

  // ── Emit and wire up ──────────────────────────────────────────────────────
  Value fusedResult = buildParallelLoops(builder, 0, initTensor);

  // The fused loop produces what elemOp used to produce.
  elemOp.getResult(0).replaceAllUsesWith(fusedResult);
  // Erase elemOp first (it holds the only use of reduceOp's result).
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

    // Collect (reduceOp, elemOp) pairs before mutating the IR.
    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>> candidates;

    func.walk([&](linalg::GenericOp op) {
      // Must have at least one reduction iterator.
      if (!hasReductionIterator(op))
        return;
      // Must produce exactly one result.
      if (op->getNumResults() != 1)
        return;
      // That result must have exactly one use.
      if (!op->getResult(0).hasOneUse())
        return;
      // The single user must be an all-parallel linalg.generic.
      Operation *user = *op->getResult(0).getUsers().begin();
      auto parallelOp = dyn_cast<linalg::GenericOp>(user);
      if (!parallelOp || !isAllParallelGeneric(parallelOp))
        return;

      // The consumer's loop count must equal the reduction's parallel-dim
      // count.  If the consumer is higher-rank (e.g. it broadcasts the
      // reduction result across an extra dimension), our loop-nest fusion
      // would need more IVs than are available from the parallel loops —
      // leading to an out-of-bounds IV lookup in evalAffineExpr.
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

// ── Entry points (declared in NovaToLinalg.h) ────────────────────────────────

std::unique_ptr<Pass> createNovaLinalgVerticalFusionPass() {
  return std::make_unique<NovaLinalgVerticalFusionPass>();
}

void registerNovaLinalgVerticalFusionPass() {
  PassRegistration<NovaLinalgVerticalFusionPass>();
}

} // namespace nova
} // namespace mlir
