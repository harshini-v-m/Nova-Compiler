/// NovaLinalgVerticalFusion.cpp
///
/// Vertical fusion: fuse a reduction linalg.generic with ALL of its
/// all-parallel consumers that live in the same gpu.launch body.
///
/// After the reduction loop produces finalAcc, every elected parallel consumer
/// is emitted sequentially inside the same outer parallel loop — each wrapped
/// in its own broadcast scf.for nest if it has extra dimensions (Case 2), or
/// inlined directly (Case 1).  A single memref.store of finalAcc is emitted
/// once, before any consumer, so non-elected users (e.g. a downstream
/// reduce<max>) always see the correct value.
///
/// Fused shape for reduce<min> + gelu (Case 1) + add (Case 2 broadcast):
///
///   scf.for %i0 = 0 to 8 {
///     acc = memref.load reduce_init[%i0]
///     acc = scf.for %i1 = 0 to 10 { minimumf(arg1[%i0,%i1], acc) }
///
///     memref.store acc, reduce_out[%i0]     // for reduce<max> and any
///                                           // other non-fused users
///     // Consumer 0 — gelu (Case 1, no broadcast loop)
///     %g = gelu(acc)
///     memref.store %g, gelu_out[%i0]
///
///     // Consumer 1 — add (Case 2, broadcast over d1)
///     scf.for %i1 = 0 to 10 {
///       %r = addf(acc, arg2[%i0,%i1])
///       memref.store %r, add_out[%i0,%i1]
///     }
///   }

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace nova {

//===----------------------------------------------------------------------===//
// Affine expression evaluator
//===----------------------------------------------------------------------===//

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
  case AffineExprKind::Add: return b.create<arith::AddIOp>(loc, lhs, rhs);
  case AffineExprKind::Mul: return b.create<arith::MulIOp>(loc, lhs, rhs);
  default:
    llvm_unreachable("unsupported affine expr in vertical fusion");
  }
}

static SmallVector<Value> evalAffineMap(OpBuilder &b, Location loc,
                                        AffineMap map, ArrayRef<Value> ivs) {
  SmallVector<Value> indices;
  for (AffineExpr expr : map.getResults())
    indices.push_back(evalAffineExpr(b, loc, expr, ivs));
  return indices;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

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

static bool inSameRegion(Operation *a, Operation *b) {
  return a->getParentRegion() == b->getParentRegion();
}

/// Returns the memref that is the root allocation backing `val`.
/// Walks through memref.subview and memref.expand_shape/collapse_shape chains
/// to find the alloc that all subviews alias.  Returns `val` itself if no
/// view-defining op is found (already a root).
static Value getRootMemref(Value val) {
  while (val) {
    Operation *def = val.getDefiningOp();
    if (!def) break;
    if (auto sv = dyn_cast<memref::SubViewOp>(def)) { val = sv.getSource(); continue; }
    if (auto es = dyn_cast<memref::ExpandShapeOp>(def)) { val = es.getSrc(); continue; }
    if (auto cs = dyn_cast<memref::CollapseShapeOp>(def)) { val = cs.getSrc(); continue; }
    break;
  }
  return val;
}

static bool isReduceResultBroadcastMap(AffineMap map,
                                       int64_t numReduceParallelDims) {
  if ((int64_t)map.getNumResults() != numReduceParallelDims)
    return false;
  for (AffineExpr res : map.getResults()) {
    bool ok = true;
    res.walk([&](AffineExpr e) {
      if (auto d = dyn_cast<AffineDimExpr>(e))
        if ((int64_t)d.getPosition() >= numReduceParallelDims)
          ok = false;
    });
    if (!ok) return false;
  }
  return true;
}

static bool bodyContainsLinalgIndex(linalg::GenericOp op) {
  return op.getBody()->walk([](linalg::IndexOp) {
    return WalkResult::interrupt();
  }).wasInterrupted();
}

static void hoistBeforeOp(Operation *op, Operation *target) {
  if (!op || op->getBlock() != target->getBlock()) return;
  if (op->isBeforeInBlock(target)) return;
  for (Value v : op->getOperands())
    if (Operation *def = v.getDefiningOp())
      hoistBeforeOp(def, target);
  op->moveBefore(target);
}

//===----------------------------------------------------------------------===//
// Per-consumer descriptor
//===----------------------------------------------------------------------===//

struct ParallelConsumer {
  linalg::GenericOp op;

  // Indexing maps for this consumer's inputs and output.
  SmallVector<AffineMap> inMaps;
  AffineMap              outMap;

  // Static loop ranges for dims beyond numReduceParallel (broadcast dims).
  // Empty for Case 1 (same-loop-count) consumers.
  SmallVector<int64_t> broadcastDimRanges;
};

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

struct FusionCandidate {
  linalg::GenericOp             reduceOp;
  SmallVector<ParallelConsumer> consumers; // all elected parallel consumers
};

//===----------------------------------------------------------------------===//
// Candidate collection
//===----------------------------------------------------------------------===//

static std::optional<FusionCandidate>
tryBuildCandidate(linalg::GenericOp reduceOp) {
  if (!hasReductionIterator(reduceOp)) return std::nullopt;

  // The reduce output may be a subview — walk up to the root alloc so we can
  // match consumers that read a *different* subview of the same buffer.
  Value reduceOut     = reduceOp.getDpsInits()[0];
  Value reduceOutRoot = getRootMemref(reduceOut);

  // The enclosing gpu.launch — consumers must live in the same launch.
  auto parentLaunch = reduceOp->getParentOfType<gpu::LaunchOp>();
  if (!parentLaunch) return std::nullopt;

  int64_t numParallel = llvm::count_if(
      reduceOp.getIteratorTypesArray(),
      [](utils::IteratorType t){ return t == utils::IteratorType::parallel; });

  FusionCandidate cand;
  cand.reduceOp = reduceOp;

  // Walk all linalg.generics in the same gpu.launch and check whether any
  // of their inputs alias the reduce output buffer.
  parentLaunch->walk([&](linalg::GenericOp elemOp) {
    if (elemOp == reduceOp)                   return;
    if (!isAllParallelGeneric(elemOp))        return;
    if (elemOp->getNumResults() != 0)         return;
    if (bodyContainsLinalgIndex(elemOp))      return;

    // Check that at least one input aliases the reduce output root.
    bool readsReduceOut = llvm::any_of(elemOp.getDpsInputs(), [&](Value v) {
      return getRootMemref(v) == reduceOutRoot;
    });
    if (!readsReduceOut) return;

    int64_t numElemLoops = (int64_t)elemOp.getIteratorTypesArray().size();
//    if (numElemLoops < numParallel) ;

    bool isBroadcast = (numElemLoops > numParallel);
    if (isBroadcast) {
      // Verify the map used for the reduce-output input is a valid broadcast map.
      // Match via root memref so subviews of the same alloc are accepted.
      auto maps   = elemOp.getIndexingMapsArray();
      auto inputs = elemOp.getDpsInputs();
      AffineMap reduceResultMap;
      bool found = false;
      for (int64_t i = 0; i < (int64_t)inputs.size(); i++) {
        if (getRootMemref(inputs[i]) == reduceOutRoot) {
          reduceResultMap = maps[i];
          found = true;
          break;
        }
      }
      if (!found) return;
      if (!isReduceResultBroadcastMap(reduceResultMap, numParallel)) return;
      bool allStatic = llvm::all_of(elemOp.getStaticLoopRanges(),
          [](int64_t r){ return r != ShapedType::kDynamic; });
      if (!allStatic) return;
    } else {
      // Case 1: parallel-dim ranges of reduce must match elem loop ranges.
      auto reduceRanges = reduceOp.getStaticLoopRanges();
      auto elemRanges   = elemOp.getStaticLoopRanges();
      auto reduceIters  = reduceOp.getIteratorTypesArray();
      SmallVector<int64_t> reduceParallelRanges;
      for (int64_t i = 0; i < (int64_t)reduceIters.size(); i++)
        if (reduceIters[i] == utils::IteratorType::parallel)
          reduceParallelRanges.push_back(reduceRanges[i]);
      if (reduceParallelRanges !=
          SmallVector<int64_t>(elemRanges.begin(), elemRanges.end()))
        return;
    }

    // Build descriptor.
    ParallelConsumer pc;
    pc.op = elemOp;
    auto maps = elemOp.getIndexingMapsArray();
    int64_t numIn = (int64_t)elemOp.getDpsInputs().size();
    pc.inMaps  = SmallVector<AffineMap>(maps.begin(), maps.begin() + numIn);
    pc.outMap  = maps[numIn];
    auto elemRanges = elemOp.getStaticLoopRanges();
    for (int64_t i = numParallel; i < numElemLoops; i++)
      pc.broadcastDimRanges.push_back(elemRanges[i]);

    cand.consumers.push_back(std::move(pc));
  });  // end parentLaunch->walk

  if (cand.consumers.empty()) return std::nullopt;
  return cand;
}

//===----------------------------------------------------------------------===//
// Core lowering
//===----------------------------------------------------------------------===//

static LogicalResult fuseReductionWithConsumers(FusionCandidate &cand) {
  linalg::GenericOp reduceOp = cand.reduceOp;
  OpBuilder b(reduceOp);
  Location loc = reduceOp.getLoc();

  //------------------------------------------------------------------------
  // Reduce op metadata
  //------------------------------------------------------------------------
  auto iterTypes  = reduceOp.getIteratorTypesArray();
  auto loopRanges = reduceOp.getStaticLoopRanges();
  int64_t numLoops = (int64_t)iterTypes.size();

  SmallVector<int64_t> parallelDims, reductionDims;
  for (int64_t i = 0; i < numLoops; i++)
    (iterTypes[i] == utils::IteratorType::parallel
         ? parallelDims : reductionDims).push_back(i);

  for (int64_t r : loopRanges)
    if (r == ShapedType::kDynamic) return failure();

  int64_t numParallel = (int64_t)parallelDims.size();

  auto reduceMaps   = reduceOp.getIndexingMapsArray();
  auto reduceInputs = reduceOp.getDpsInputs();
  int64_t numReduceIn = (int64_t)reduceInputs.size();
  SmallVector<AffineMap> reduceInMaps(reduceMaps.begin(),
                                      reduceMaps.begin() + numReduceIn);
  AffineMap reduceOutMap = reduceMaps[numReduceIn];
  Value reduceOutMem     = reduceOp.getDpsInits()[0];
  Block *reduceBody      = reduceOp.getBody();

  //------------------------------------------------------------------------
  // Hoist all consumer inputs before reduceOp
  //------------------------------------------------------------------------
  for (auto &pc : cand.consumers) {
    for (Value v : pc.op.getDpsInputs()) {
      if (v == reduceOutMem) continue;
      if (Operation *def = v.getDefiningOp())
        hoistBeforeOp(def, reduceOp);
    }
  }

  //------------------------------------------------------------------------
  // Loop bound constants
  //------------------------------------------------------------------------
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> bounds(numLoops);
  for (int64_t i = 0; i < numLoops; i++)
    bounds[i] = b.create<arith::ConstantIndexOp>(loc, loopRanges[i]);

  // ivs[d] — IV for logical dimension d of reduceOp's loop space.
  SmallVector<Value> ivs(numLoops, c0);

  //------------------------------------------------------------------------
  // Inner: reduction loop nest
  //------------------------------------------------------------------------
  std::function<Value(OpBuilder &, int64_t, Value)> buildReductionLoops;
  buildReductionLoops = [&](OpBuilder &rb, int64_t ridx, Value acc) -> Value {
    if (ridx == (int64_t)reductionDims.size()) {
      IRMapping m;
      for (int64_t i = 0; i < numReduceIn; i++) {
        SmallVector<Value> idx = evalAffineMap(rb, loc, reduceInMaps[i], ivs);
        Value val = rb.create<memref::LoadOp>(loc, reduceInputs[i], idx);
        m.map(reduceBody->getArgument(i), val);
      }
      m.map(reduceBody->getArgument(numReduceIn), acc);
      for (auto &op : reduceBody->without_terminator())
        rb.clone(op, m);
      auto yld = cast<linalg::YieldOp>(reduceBody->getTerminator());
      return m.lookupOrDefault(yld.getOperand(0));
    }
    int64_t dim = reductionDims[ridx];
    auto loop = rb.create<scf::ForOp>(
        loc, c0, bounds[dim], c1, ValueRange{acc},
        [&](OpBuilder &ib, Location, Value iv, ValueRange args) {
          ivs[dim] = iv;
          Value newAcc = buildReductionLoops(ib, ridx + 1, args[0]);
          ib.create<scf::YieldOp>(loc, newAcc);
        });
    return loop.getResult(0);
  };

  //------------------------------------------------------------------------
  // Per-consumer emission
  //
  // For each consumer, elemIVs[i] is the IV for the i-th loop of that
  // consumer.  The first numParallel entries come from the outer parallel
  // loops (shared with ivs[parallelDims[p]]).  The remaining entries are
  // filled by broadcast scf.for loops specific to this consumer.
  //------------------------------------------------------------------------
  auto emitOneConsumer = [&](OpBuilder &eb, ParallelConsumer &pc,
                             Value finalAcc,
                             ArrayRef<Value> parallelIVs) {
    // Build elemIVs for this consumer: start with the parallel IVs,
    // broadcast entries will be filled inside the lambda below.
    int64_t numElemLoops =
        numParallel + (int64_t)pc.broadcastDimRanges.size();
    SmallVector<Value> elemIVs(numElemLoops, c0);
    for (int64_t p = 0; p < numParallel; p++)
      elemIVs[p] = parallelIVs[p];

    auto inputs  = pc.op.getDpsInputs();
    auto outMem  = pc.op.getDpsInits()[0];
    auto body    = pc.op.getBody();
    int64_t numIn = (int64_t)inputs.size();

    // Recursive broadcast loop emitter for this consumer.
    std::function<void(OpBuilder &, int64_t)> emitBcast;
    emitBcast = [&](OpBuilder &bb, int64_t bidx) {
      if (bidx == (int64_t)pc.broadcastDimRanges.size()) {
        // Inline the elementwise body.
        IRMapping m;
        for (int64_t i = 0; i < numIn; i++) {
          Value argVal;
          if (inputs[i] == reduceOutMem) {
            argVal = finalAcc; // scalar — no load needed
          } else {
            SmallVector<Value> idx =
                evalAffineMap(bb, loc, pc.inMaps[i], elemIVs);
            argVal = bb.create<memref::LoadOp>(loc, inputs[i], idx);
          }
          m.map(body->getArgument(i), argVal);
        }
        SmallVector<Value> outIdx = evalAffineMap(bb, loc, pc.outMap, elemIVs);
        Value outInit = bb.create<memref::LoadOp>(loc, outMem, outIdx);
        m.map(body->getArgument(numIn), outInit);
        for (auto &op : body->without_terminator())
          bb.clone(op, m);
        auto yld = cast<linalg::YieldOp>(body->getTerminator());
        Value result = m.lookupOrDefault(yld.getOperand(0));
        bb.create<memref::StoreOp>(loc, result, outMem, outIdx);
        return;
      }
      int64_t elemDim = numParallel + bidx;
      Value ub = bb.create<arith::ConstantIndexOp>(
          loc, pc.broadcastDimRanges[bidx]);
      bb.create<scf::ForOp>(
          loc, c0, ub, c1, ValueRange{},
          [&](OpBuilder &ib, Location, Value iv, ValueRange) {
            elemIVs[elemDim] = iv;
            emitBcast(ib, bidx + 1);
            ib.create<scf::YieldOp>(loc);
          });
    };

    emitBcast(eb, 0);
  };

  //------------------------------------------------------------------------
  // Outer: parallel loop nest over reduceOp's parallel dims
  //------------------------------------------------------------------------
  // Collect parallel IVs as we descend — shared across all consumers.
  SmallVector<Value> parallelIVs(numParallel, c0);

  std::function<void(OpBuilder &, int64_t)> buildParallelLoops;
  buildParallelLoops = [&](OpBuilder &pb, int64_t pidx) {
    if (pidx == numParallel) {
      // All parallel IVs set. Run reduction.
      SmallVector<Value> initIdx = evalAffineMap(pb, loc, reduceOutMap, ivs);
      Value initAcc   = pb.create<memref::LoadOp>(loc, reduceOutMem, initIdx);
      Value finalAcc  = buildReductionLoops(pb, 0, initAcc);

      // Store finalAcc once — keeps all non-elected users correct.
      pb.create<memref::StoreOp>(loc, finalAcc, reduceOutMem, initIdx);

      // Emit each consumer sequentially using the same finalAcc.
      for (auto &pc : cand.consumers)
        emitOneConsumer(pb, pc, finalAcc, parallelIVs);

      return;
    }
    int64_t dim = parallelDims[pidx];
    pb.create<scf::ForOp>(
        loc, c0, bounds[dim], c1, ValueRange{},
        [&](OpBuilder &ib, Location, Value iv, ValueRange) {
          ivs[dim]         = iv;
          parallelIVs[pidx] = iv;
          buildParallelLoops(ib, pidx + 1);
          ib.create<scf::YieldOp>(loc);
        });
  };

  buildParallelLoops(b, 0);

  // Erase all elected consumers, then the reduce op.
  for (auto &pc : cand.consumers)
    pc.op->erase();
  reduceOp->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaLinalgVerticalFusionPass
    : public PassWrapper<NovaLinalgVerticalFusionPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaLinalgVerticalFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    scf::SCFDialect,        gpu::GPUDialect,
                    memref::MemRefDialect>();
  }

  StringRef getArgument()    const final { return "nova-linalg-vertical-fusion"; }
  StringRef getDescription() const final {
    return "Fuse a reduction linalg.generic with all its all-parallel consumers "
           "inside the same gpu.launch body (memref level, post-bufferization)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<FusionCandidate> candidates;

    func.walk([&](linalg::GenericOp op) {
      // Accept ops at any nesting depth inside a gpu.launch — the reduction
      // may be inside an scf.if or scf.for within the launch body.
      if (!op->getParentOfType<gpu::LaunchOp>()) return;
      if (auto cand = tryBuildCandidate(op))
        candidates.push_back(std::move(*cand));
    });

    for (auto &cand : candidates) {
      if (failed(fuseReductionWithConsumers(cand))) {
        cand.reduceOp.emitError("nova-vertical-fusion: fusion failed");
        signalPassFailure();
        return;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaLinalgVerticalFusionPass() {
  return std::make_unique<NovaLinalgVerticalFusionPass>();
}

void registerNovaLinalgVerticalFusionPass() {
  PassRegistration<NovaLinalgVerticalFusionPass>();
}

} // namespace nova
} // namespace mlir