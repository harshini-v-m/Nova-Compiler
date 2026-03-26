//===- NovaScfLoopVectorize.cpp - nova-scf-loop-vectorize pass -----------===//
//
// Vectorizes innermost scf.for loops the same way AffineVectorize handles
// affine.for — but working at the scf + memref level that linalg-to-loops
// produces.
//
// For a qualifying innermost loop the pass generates:
//
//   // --- bounds arithmetic ---
//   %range  = arith.subi  ub, lb
//   %vf     = arith.constant VF : index
//   %rem    = arith.remsi %range, %vf
//   %vub    = arith.subi  ub, %rem     // aligned upper bound
//
//   // --- vectorized main loop ---
//   scf.for %i = lb to %vub step VF {
//       // stride-1 loads  → vector.load  (vector<VF x T>)
//       // stride-1 stores → vector.store
//       // loop-invariant scalars used alongside vectors → vector.broadcast
//       // arith compute   → same arith op, operands/result are vector type
//   }
//
//   // --- scalar cleanup loop ---
//   scf.for %i = %vub to ub step 1 {
//       // original body cloned with IV remapped
//   }
//
// The cleanup loop handles the 0 … VF-1 remainder iterations.  CSE and
// canonicalization (run after this pass in the pipeline) fold dead broadcasts
// and constant arithmetic.
//
// Pipeline position:
//   AFTER  LoopInterchange + LoopSplit (clean stride-1 innermost loops,
//          no conditionals to confuse legality)
//   BEFORE ForLoopSpecialization / Peeling / Unroll (we produce our own
//          remainder loop so specialization is not needed for VF alignment)
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaScfLoopVectorize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-scf-loop-vectorize"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Legality analysis
//===----------------------------------------------------------------------===//

/// Returns true if \p type is a scalar numeric type that can be widened into
/// a vector element (integer or floating-point of any width).
static bool isVectorizableElemType(Type type) {
  return type.isIntOrFloat();
}

/// Returns true if \p iv appears anywhere in the SSA def-chain of \p v,
/// stopping at ops defined outside the loop body block \p loopBlock.
/// Used to detect pseudo-stride-1 patterns like addi(offset, iv) that the
/// vectorizer cannot currently handle (it would scalar-clone the load and
/// broadcast a single element instead of loading a contiguous vector).
static bool ivUsedInValue(Value v, Value iv, Block *loopBlock) {
  if (v == iv)
    return true;
  Operation *defOp = v.getDefiningOp();
  // Block args (e.g. loop iter_args) or values defined outside the loop body
  // cannot involve the IV.
  if (!defOp || defOp->getBlock() != loopBlock)
    return false;
  for (Value operand : defOp->getOperands())
    if (ivUsedInValue(operand, iv, loopBlock))
      return true;
  return false;
}

/// Returns true if the induction variable \p iv appears anywhere in the
/// operands \p idxs at a position OTHER than the last one.
/// An IV reference in a non-last index means the access is not stride-1 along
/// the loop dimension and cannot be widened with a simple vector.load.
/// Uses transitive def-chain lookup so patterns like addi(iv, c) are caught.
static bool ivInNonLastIdx(ValueRange idxs, Value iv, Block *loopBlock) {
  for (size_t i = 0; i + 1 < idxs.size(); ++i)
    if (ivUsedInValue(idxs[i], iv, loopBlock))
      return true;
  return false;
}

/// Decide if \p loop qualifies for vectorization.
static bool isVectorizableLoop(scf::ForOp loop) {
  // ── Structural constraints ────────────────────────────────────────────────

  // Step must be the constant 1 so that VF consecutive iterations are a
  // contiguous VF-element slice of each accessed memref.
  auto stepOp = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!stepOp || stepOp.value() != 1)
    return false;

  // No iter_args: the accumulator must live in a memref (R/W via load/store),
  // not in an SSA loop-carried register.  If the loop carries SSA values the
  // iter_arg types would also need widening, which is out of scope here.
  if (!loop.getInitArgs().empty())
    return false;

  // Innermost loop: no nested scf.for.
  bool hasNested = false;
  loop.getBody()->walk([&](scf::ForOp nested) {
    if (nested != loop)
      hasNested = true;
  });
  if (hasNested)
    return false;

  // ── Body constraints ──────────────────────────────────────────────────────

  Value iv = loop.getInductionVar();
  Block *loopBlock = loop.getBody();
  bool hasStrideOneAccess = false; // at least one load/store on the IV dim
  bool bodyOk = true;

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(&op)) {
      auto idxs = ld.getIndices();
      if (ivInNonLastIdx(idxs, iv, loopBlock)) { bodyOk = false; break; }
      if (!isVectorizableElemType(
              cast<MemRefType>(ld.getMemRef().getType()).getElementType())) {
        bodyOk = false; break;
      }
      if (!idxs.empty() && idxs.back() == iv) {
        hasStrideOneAccess = true;
      } else if (!idxs.empty() && ivUsedInValue(idxs.back(), iv, loopBlock)) {
        // Last index involves IV via a computation (e.g. addi(offset, iv)).
        // buildVectorizedBody would scalar-clone this load (using newIV which
        // steps by VF) and broadcast the single loaded value to all VF lanes,
        // producing wrong results. Reject the loop.
        bodyOk = false; break;
      }
      continue;
    }

    if (auto st = dyn_cast<memref::StoreOp>(&op)) {
      auto idxs = st.getIndices();
      if (ivInNonLastIdx(idxs, iv, loopBlock)) { bodyOk = false; break; }
      auto memTy = cast<MemRefType>(st.getMemRef().getType());
      if (!isVectorizableElemType(memTy.getElementType())) {
        bodyOk = false; break;
      }
      // Reject stores to GPU non-default address spaces (workgroup/shared,
      // private): vectorized stores to these spaces are not reliably lowered.
      if (memTy.getMemorySpace() &&
          !isa<gpu::AddressSpaceAttr>(memTy.getMemorySpace())) {
        // non-GPU custom space — reject conservatively
        bodyOk = false; break;
      }
      if (auto as = dyn_cast_or_null<gpu::AddressSpaceAttr>(
              memTy.getMemorySpace())) {
        if (as.getValue() != gpu::AddressSpace::Global) {
          bodyOk = false; break;
        }
      }
      if (!idxs.empty() && idxs.back() == iv) {
        hasStrideOneAccess = true;
      } else if (!idxs.empty() && ivUsedInValue(idxs.back(), iv, loopBlock)) {
        // Same reasoning as for loads: a store to addi(offset, iv) cannot be
        // correctly widened to a contiguous vector.store by this pass.
        bodyOk = false; break;
      }
      continue;
    }

    // All other ops must be pure (memory-effect-free).
    if (!isMemoryEffectFree(&op)) { bodyOk = false; break; }
  }

  return bodyOk && hasStrideOneAccess;
}

//===----------------------------------------------------------------------===//
// Vectorized body builder
//===----------------------------------------------------------------------===//

/// Build the body of the vectorized loop into builder \p b, which is already
/// positioned inside a new scf.for with induction variable \p newIV.
///
/// Two bookkeeping maps are maintained while scanning the original body:
///   vecMap   – original SSA value  →  its vector<VF x T> counterpart
///   scalarMap – original SSA value → scalar remapping (IV + loop-invariant
///               scalar loads that were cloned unchanged)
///
/// For each original op:
///   • memref.load  at last-idx == IV → vector.load, result added to vecMap
///   • memref.load  NOT at IV index   → scalar clone, result in scalarMap
///   • memref.store at last-idx == IV → vector.store (stored value from vecMap
///     or broadcast if scalar)
///   • memref.store NOT at IV index   → scalar clone
///   • arith / pure op with ≥1 vector operand → vectorized via OperationState;
///     scalar operands are broadcast to the same vector type first
///   • arith / pure op with all-scalar operands → scalar clone (IV remapped)
///
/// The function closes with an empty scf.yield (loop has no iter_args).
static void buildVectorizedBody(OpBuilder &b, Location loc,
                                 scf::ForOp orig, int64_t vf,
                                 Value newIV) {
  // ── Bookkeeping ──────────────────────────────────────────────────────────
  // vecMap:    original Value  →  vector Value (vector<VF x T>)
  // scalarMap: original Value  →  remapped scalar Value (identity if unchanged)
  llvm::DenseMap<Value, Value> vecMap;
  IRMapping scalarMap;
  scalarMap.map(orig.getInductionVar(), newIV);

  Value iv = orig.getInductionVar();

  // Return the vector-typed version of `v`.
  // If already in vecMap: return as-is.
  // Otherwise: fetch the scalar through scalarMap, broadcast to `vt`.
  // The broadcast is cached in vecMap to avoid duplicating the broadcast op.
  auto getVec = [&](Value v, VectorType vt) -> Value {
    auto it = vecMap.find(v);
    if (it != vecMap.end() && it->second.getType() == vt)
      return it->second;
    Value scalar = scalarMap.lookupOrDefault(v);
    Value bc = b.create<vector::BroadcastOp>(loc, vt, scalar);
    vecMap[v] = bc; // cache so repeated uses don't re-emit the broadcast
    return bc;
  };

  // ── Scan original body ───────────────────────────────────────────────────
  for (Operation &op : orig.getBody()->without_terminator()) {

    // ── memref.load ─────────────────────────────────────────────────────────
    if (auto ld = dyn_cast<memref::LoadOp>(&op)) {
      SmallVector<Value> newIdxs;
      for (Value idx : ld.getIndices())
        newIdxs.push_back(scalarMap.lookupOrDefault(idx));

      if (!ld.getIndices().empty() && ld.getIndices().back() == iv) {
        // Stride-1 load → widen to vector.load
        Type elemTy =
            cast<MemRefType>(ld.getMemRef().getType()).getElementType();
        VectorType vt = VectorType::get({vf}, elemTy);
        Value vec =
            b.create<vector::LoadOp>(loc, vt, ld.getMemRef(), newIdxs);
        vecMap[ld.getResult()] = vec;
      } else {
        // Loop-invariant load → keep scalar, record in scalarMap
        auto *cloned = b.clone(op, scalarMap);
        scalarMap.map(ld.getResult(), cloned->getResult(0));
      }
      continue;
    }

    // ── memref.store ─────────────────────────────────────────────────────────
    if (auto st = dyn_cast<memref::StoreOp>(&op)) {
      SmallVector<Value> newIdxs;
      for (Value idx : st.getIndices())
        newIdxs.push_back(scalarMap.lookupOrDefault(idx));

      if (!st.getIndices().empty() && st.getIndices().back() == iv) {
        // Stride-1 store → widen to vector.store
        Value stVal = st.getValueToStore();
        Type elemTy =
            cast<MemRefType>(st.getMemRef().getType()).getElementType();
        VectorType vt = VectorType::get({vf}, elemTy);
        // Stored value may already be vectorized or may need broadcast.
        Value vecStVal = vecMap.count(stVal) ? vecMap[stVal]
                                             : getVec(stVal, vt);
        b.create<vector::StoreOp>(loc, vecStVal, st.getMemRef(), newIdxs);
      } else {
        b.clone(op, scalarMap);
      }
      continue;
    }

    // ── Pure compute op ──────────────────────────────────────────────────────
    // Determine whether any operand has already been vectorized.
    VectorType vecTy;
    bool hasVecOperand = false;
    for (Value operand : op.getOperands()) {
      auto it = vecMap.find(operand);
      if (it != vecMap.end() && isa<VectorType>(it->second.getType())) {
        hasVecOperand = true;
        vecTy = cast<VectorType>(it->second.getType());
        break;
      }
    }

    if (hasVecOperand) {
      // Emit the same op with all operands widened to `vecTy`.
      // Scalar operands are broadcast first (cached by getVec).
      SmallVector<Value> newOperands;
      newOperands.reserve(op.getNumOperands());
      for (Value operand : op.getOperands())
        newOperands.push_back(getVec(operand, vecTy));

      // Build the vectorized op via OperationState so we can set the
      // result types explicitly.  All arith compute ops in a matmul body
      // satisfy SameOperandsAndResultType, so setting every result to vecTy
      // is correct.
      OperationState state(op.getLoc(), op.getName());
      state.addOperands(newOperands);
      for (unsigned i = 0; i < op.getNumResults(); ++i)
        state.addTypes(vecTy);
      state.addAttributes(op.getAttrs());

      Operation *newOp = b.create(state);
      for (auto [origRes, newRes] :
           llvm::zip(op.getResults(), newOp->getResults()))
        vecMap[origRes] = newRes;
    } else {
      // All-scalar op: clone with IV remapping, record scalar results.
      auto *cloned = b.clone(op, scalarMap);
      for (auto [origRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults()))
        scalarMap.map(origRes, newRes);
    }
  }

  // Close the vectorized loop body.  No iter_args → empty yield.
  b.create<scf::YieldOp>(loc);
}

//===----------------------------------------------------------------------===//
// Transformation driver
//===----------------------------------------------------------------------===//

/// Replace \p loop with a vectorized main loop + scalar cleanup loop.
static void vectorizeLoop(scf::ForOp loop, int64_t vf) {
  // Insert both new loops before the original (which we will erase at the end).
  OpBuilder b(loop);
  Location loc = loop.getLoc();

  Value lb   = loop.getLowerBound();
  Value ub   = loop.getUpperBound();
  Value step = loop.getStep(); // constant 1 (verified by legality check)

  // ── Compute aligned upper bound ──────────────────────────────────────────
  //   vub = ub - ((ub - lb) % VF)
  Value vfConst = b.create<arith::ConstantIndexOp>(loc, vf);
  Value range   = b.create<arith::SubIOp>(loc, ub, lb);
  Value rem     = b.create<arith::RemSIOp>(loc, range, vfConst);
  Value vub     = b.create<arith::SubIOp>(loc, ub, rem);

  Value vStep = b.create<arith::ConstantIndexOp>(loc, vf);

  LLVM_DEBUG(llvm::dbgs() << "[nova-scf-loop-vectorize] vectorizing loop "
                           << "with VF=" << vf << "\n");

  // ── Vectorized main loop: [lb, vub, VF) ──────────────────────────────────
  b.create<scf::ForOp>(
      loc, lb, vub, vStep,
      /*iterArgs=*/ValueRange{},
      [&](OpBuilder &ib, Location il, Value newIV, ValueRange /*args*/) {
        buildVectorizedBody(ib, il, loop, vf, newIV);
      });

  // ── Scalar cleanup loop: [vub, ub, 1) ────────────────────────────────────
  // Handles the 0 … VF-1 remainder iterations with the unmodified scalar body.
  b.create<scf::ForOp>(
      loc, vub, ub, step,
      /*iterArgs=*/ValueRange{},
      [&](OpBuilder &ib, Location il, Value newIV, ValueRange /*args*/) {
        IRMapping mp;
        mp.map(loop.getInductionVar(), newIV);
        for (Operation &op : loop.getBody()->without_terminator())
          ib.clone(op, mp);
        ib.create<scf::YieldOp>(il);
      });

  // Erase the original scalar loop.  Safe because it has no iter_args and
  // therefore no SSA results that external users depend on.
  loop->erase();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NovaScfLoopVectorizePass
    : public PassWrapper<NovaScfLoopVectorizePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaScfLoopVectorizePass)

  StringRef getArgument()    const final { return "nova-scf-loop-vectorize"; }
  StringRef getDescription() const final {
    return "Vectorize innermost scf.for loops (stride-1 memref accesses) "
           "using vector.load / vector.store / vector.broadcast";
  }

  explicit NovaScfLoopVectorizePass(int64_t vf = 4) : vectorFactor(vf) {}

  void runOnOperation() override {
    if (vectorFactor <= 0)
      return;

    func::FuncOp func = getOperation();

    // Collect candidates in a snapshot — vectorizeLoop erases the original op
    // so we cannot walk while mutating.
    SmallVector<scf::ForOp, 16> candidates;
    func.walk([&](scf::ForOp loop) {
      if (isVectorizableLoop(loop))
        candidates.push_back(loop);
    });

    // Process in post-order (inner loops first): after an innermost loop is
    // vectorized its parent is no longer innermost, which prevents spurious
    // re-vectorization of already-widened loops.
    for (scf::ForOp loop : llvm::reverse(candidates))
      vectorizeLoop(loop, vectorFactor);
  }

  int64_t vectorFactor;
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createNovaScfLoopVectorizePass(int64_t vectorFactor) {
  return std::make_unique<NovaScfLoopVectorizePass>(vectorFactor);
}

} // namespace nova
} // namespace mlir
