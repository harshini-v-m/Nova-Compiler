//===- NovaScfLoopVectorize.cpp - nova-scf-loop-vectorize pass -----------===//
//
// Vectorizes innermost scf.for loops the same way AffineVectorize handles
// affine.for — but working at the scf + memref level that linalg-to-loops
// produces.
//
// TWO MODES OF OPERATION:
//
// ── Mode A: Non-reduction loops (no iter_args) ─────────────────────────────
//
//   For a qualifying innermost loop the pass generates:
//
//     %vub = ub - ((ub - lb) % VF)
//     scf.for %i = lb to %vub step VF {
//         stride-1 loads  → vector.load  (vector<VF x T>)
//         stride-1 stores → vector.store
//         loop-invariant scalars → vector.broadcast
//         arith compute → widened to vector type
//     }
//     scf.for %i = %vub to ub step 1 { /* scalar cleanup */ }
//
// ── Mode B: Reduction loops (exactly 1 scalar iter_arg) ────────────────────
//
//   Detects loops produced by SCFScalarizeAccumulator where a single scalar
//   accumulator is carried as an iter_arg. The accumulator is widened to
//   vector<VF x T> and the access pattern is restructured for contiguous
//   memory access:
//
//     newLb   = origLb * VF
//     newStep = origStep * VF
//     %vinit  = vector.insert %initArg, (vector.splat %identity) [0]
//     scf.for %i = newLb to ub step newStep iter_args(%vacc = %vinit) {
//         vector.load / widened arith / vector accumulation
//         scf.yield %vacc_new
//     }
//     %scalar = vector.reduction <kind>, %vacc_final : vector<VF x T> into T
//     scf.for %i = ... step origStep iter_args(%acc = %scalar) {
//         /* scalar cleanup for remainder */
//     }
//
//   The vector.reduction is later lowered to LLVM by ConvertVectorToLLVM.
//   The scalar result feeds into the subsequent WarpShuffle pass for
//   cross-thread reduction via gpu.shuffle.
//
// Pipeline position:
//   AFTER  LoopSplit + SCFScalarizeAccumulator
//   BEFORE WarpShuffle / ForLoopSpecialization / Unroll
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
#include <limits>

#define DEBUG_TYPE "nova-scf-loop-vectorize"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Returns true if \p type is a scalar numeric type that can be widened into
/// a vector element (integer or floating-point of any width).
static bool isVectorizableElemType(Type type) {
  return type.isIntOrFloat();
}

/// Returns true if \p iv appears anywhere in the SSA def-chain of \p v,
/// stopping at ops defined outside the loop body block \p loopBlock.
static bool ivUsedInValue(Value v, Value iv, Block *loopBlock) {
  if (v == iv)
    return true;
  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getBlock() != loopBlock)
    return false;
  for (Value operand : defOp->getOperands())
    if (ivUsedInValue(operand, iv, loopBlock))
      return true;
  return false;
}

/// Returns true if the IV appears in any index position OTHER than the last.
static bool ivInNonLastIdx(ValueRange idxs, Value iv, Block *loopBlock) {
  for (size_t i = 0; i + 1 < idxs.size(); ++i)
    if (ivUsedInValue(idxs[i], iv, loopBlock))
      return true;
  return false;
}

/// Extract a constant integer / index from a Value.
static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
    return c.value();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Reduction combiner detection
//===----------------------------------------------------------------------===//

/// Information about a detected reduction combiner in a loop body.
struct ReductionInfo {
  Operation *combinerOp = nullptr;        // the arith op feeding yield
  arith::AtomicRMWKind atomicKind;        // reduction kind
  vector::CombiningKind vectorKind;       // for vector.reduction
  Value otherOperand;                     // the non-iter_arg operand
};

/// Try to detect a reduction combiner from the yield of a loop with 1 iter_arg.
/// Returns true if a supported combiner is found.
static bool detectReductionCombiner(scf::ForOp loop, ReductionInfo &info) {
  if (loop.getNumRegionIterArgs() != 1)
    return false;

  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value yieldedVal = yieldOp.getOperand(0);
  Operation *defOp = yieldedVal.getDefiningOp();
  if (!defOp || defOp->getNumOperands() != 2)
    return false;

  Value iterArg = loop.getRegionIterArg(0);

  // One operand must be the iter_arg, the other is the "incoming" value.
  Value other;
  if (defOp->getOperand(0) == iterArg)
    other = defOp->getOperand(1);
  else if (defOp->getOperand(1) == iterArg)
    other = defOp->getOperand(0);
  else
    return false;

  // Detect the combiner kind.
  arith::AtomicRMWKind ak;
  vector::CombiningKind vk;
  if (isa<arith::AddFOp>(defOp)) {
    ak = arith::AtomicRMWKind::addf; vk = vector::CombiningKind::ADD;
  } else if (isa<arith::AddIOp>(defOp)) {
    ak = arith::AtomicRMWKind::addi; vk = vector::CombiningKind::ADD;
  } else if (isa<arith::MulFOp>(defOp)) {
    ak = arith::AtomicRMWKind::mulf; vk = vector::CombiningKind::MUL;
  } else if (isa<arith::MulIOp>(defOp)) {
    ak = arith::AtomicRMWKind::muli; vk = vector::CombiningKind::MUL;
  } else if (isa<arith::MaximumFOp>(defOp)) {
    ak = arith::AtomicRMWKind::maximumf; vk = vector::CombiningKind::MAXIMUMF;
  } else if (isa<arith::MinimumFOp>(defOp)) {
    ak = arith::AtomicRMWKind::minimumf; vk = vector::CombiningKind::MINIMUMF;
  } else if (isa<arith::MaxNumFOp>(defOp)) {
    ak = arith::AtomicRMWKind::maxnumf; vk = vector::CombiningKind::MAXNUMF;
  } else if (isa<arith::MinNumFOp>(defOp)) {
    ak = arith::AtomicRMWKind::minnumf; vk = vector::CombiningKind::MINNUMF;
  } else if (isa<arith::MaxSIOp>(defOp)) {
    ak = arith::AtomicRMWKind::maxs; vk = vector::CombiningKind::MAXSI;
  } else if (isa<arith::MinSIOp>(defOp)) {
    ak = arith::AtomicRMWKind::mins; vk = vector::CombiningKind::MINSI;
  } else if (isa<arith::MaxUIOp>(defOp)) {
    ak = arith::AtomicRMWKind::maxu; vk = vector::CombiningKind::MAXUI;
  } else if (isa<arith::MinUIOp>(defOp)) {
    ak = arith::AtomicRMWKind::minu; vk = vector::CombiningKind::MINUI;
  } else if (isa<arith::OrIOp>(defOp)) {
    ak = arith::AtomicRMWKind::ori; vk = vector::CombiningKind::OR;
  } else if (isa<arith::AndIOp>(defOp)) {
    ak = arith::AtomicRMWKind::andi; vk = vector::CombiningKind::AND;
  } else {
    return false;
  }

  info.combinerOp = defOp;
  info.atomicKind = ak;
  info.vectorKind = vk;
  info.otherOperand = other;
  return true;
}

/// Build the identity constant for a reduction kind.
static Value buildReductionIdentityValue(OpBuilder &b, Location loc,
                                          arith::AtomicRMWKind kind,
                                          Type elemTy) {
  if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
    double val = 0.0;
    switch (kind) {
    case arith::AtomicRMWKind::mulf:
      val = 1.0; break;
    case arith::AtomicRMWKind::maximumf:
    case arith::AtomicRMWKind::maxnumf:
      val = -std::numeric_limits<double>::infinity(); break;
    case arith::AtomicRMWKind::minimumf:
    case arith::AtomicRMWKind::minnumf:
      val = std::numeric_limits<double>::infinity(); break;
    default:
      val = 0.0; break;
    }
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(floatTy, val));
  }
  auto intTy = cast<IntegerType>(elemTy);
  unsigned w = intTy.getWidth();
  int64_t val = 0;
  switch (kind) {
  case arith::AtomicRMWKind::muli:
    val = 1; break;
  case arith::AtomicRMWKind::andi:
  case arith::AtomicRMWKind::minu:
    val = -1; break;
  case arith::AtomicRMWKind::maxs:
    val = (w >= 64) ? std::numeric_limits<int64_t>::min()
                    : -(int64_t(1) << (w - 1)); break;
  case arith::AtomicRMWKind::mins:
    val = (w >= 64) ? std::numeric_limits<int64_t>::max()
                    : (int64_t(1) << (w - 1)) - 1; break;
  default:
    val = 0; break;
  }
  return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(intTy, val));
}

//===----------------------------------------------------------------------===//
// Legality analysis
//===----------------------------------------------------------------------===//

/// Classification of a vectorizable loop.
enum class LoopKind {
  NonReduction,  // no iter_args, stride-1 loads/stores
  Reduction,     // 1 scalar iter_arg with a recognized combiner
};

/// Result of legality analysis.
struct VectorizableLoop {
  scf::ForOp loop;
  LoopKind kind;
  ReductionInfo redInfo; // only valid when kind == Reduction
};

/// Decide if \p loop qualifies for vectorization. Returns true and fills
/// \p result if the loop is vectorizable in either mode.
static bool analyzeLoop(scf::ForOp loop, int64_t vf, VectorizableLoop &result) {
  // ── Innermost loop: no nested scf.for ────────────────────────────────────
  bool hasNested = false;
  loop.getBody()->walk([&](scf::ForOp nested) {
    if (nested != loop)
      hasNested = true;
  });
  if (hasNested)
    return false;

  Value iv = loop.getInductionVar();
  Block *loopBlock = loop.getBody();

  // ── Determine mode: non-reduction vs reduction ───────────────────────────
  bool isReduction = false;
  ReductionInfo redInfo;

  if (loop.getInitArgs().empty()) {
    // Mode A: non-reduction. Step must be constant 1.
    auto stepOp = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!stepOp || stepOp.value() != 1)
      return false;
  } else if (loop.getInitArgs().size() == 1) {
    // Mode B: reduction. Exactly 1 scalar iter_arg.
    Type iterTy = loop.getInitArgs()[0].getType();
    if (!isVectorizableElemType(iterTy))
      return false;
    if (!detectReductionCombiner(loop, redInfo))
      return false;

    // Step must be constant 1. Non-unit steps (e.g., stride-N thread
    // partitioned loops like "for d=tid to 128 step 32") would have
    // newStep = step*VF and newLb = lb*VF, corrupting memory access patterns.
    auto step = getConstantIndex(loop.getStep());
    if (!step || *step != 1)
      return false;

    // Lower bound must be constant 0. If lb is thread-dependent (e.g.,
    // lb = tid*range), then newLb = lb*VF would start at the wrong element,
    // causing threads to accumulate wrong slices of the reduction domain.
    auto lb = getConstantIndex(loop.getLowerBound());
    if (!lb || *lb != 0)
      return false;

    // Upper bound must be constant so we can check alignment.
    auto ub = getConstantIndex(loop.getUpperBound());
    if (!ub)
      return false;

    // ub must be divisible by (step * VF) for access pattern restructuring.
    int64_t newStep = *step * vf;
    if (*ub % newStep != 0)
      return false;

    // Need at least 1 vectorized iteration (assuming lb could be up to
    // step-1 for the last thread).
    if (*ub < newStep)
      return false;

    isReduction = true;
  } else {
    // Multiple iter_args — not supported.
    return false;
  }

  // ── Body constraints (shared by both modes) ──────────────────────────────
  bool hasStrideOneAccess = false;
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
        bodyOk = false; break;
      }
      continue;
    }

    if (auto st = dyn_cast<memref::StoreOp>(&op)) {
      // Reduction loops must not contain ANY stores. After SCFScalarize,
      // the accumulator is in iter_args (registers). Any remaining store
      // in the body either writes a vectorized value to a scalar memref
      // (which we cannot handle correctly) or indicates the body has side
      // effects that prevent safe vectorization. Reject the loop.
      if (isReduction) { bodyOk = false; break; }

      auto idxs = st.getIndices();
      if (ivInNonLastIdx(idxs, iv, loopBlock)) { bodyOk = false; break; }
      auto memTy = cast<MemRefType>(st.getMemRef().getType());
      if (!isVectorizableElemType(memTy.getElementType())) {
        bodyOk = false; break;
      }
      if (memTy.getMemorySpace() &&
          !isa<gpu::AddressSpaceAttr>(memTy.getMemorySpace())) {
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
        bodyOk = false; break;
      }
      continue;
    }

    // All other ops must be pure (memory-effect-free).
    if (!isMemoryEffectFree(&op)) { bodyOk = false; break; }
  }

  if (!bodyOk || !hasStrideOneAccess)
    return false;

  result.loop = loop;
  result.kind = isReduction ? LoopKind::Reduction : LoopKind::NonReduction;
  result.redInfo = redInfo;
  return true;
}

//===----------------------------------------------------------------------===//
// Non-reduction vectorized body builder (Mode A — original logic)
//===----------------------------------------------------------------------===//

static void buildVectorizedBody(OpBuilder &b, Location loc,
                                 scf::ForOp orig, int64_t vf,
                                 Value newIV) {
  llvm::DenseMap<Value, Value> vecMap;
  IRMapping scalarMap;
  scalarMap.map(orig.getInductionVar(), newIV);

  Value iv = orig.getInductionVar();

  auto getVec = [&](Value v, VectorType vt) -> Value {
    auto it = vecMap.find(v);
    if (it != vecMap.end() && it->second.getType() == vt)
      return it->second;
    Value scalar = scalarMap.lookupOrDefault(v);
    Value bc = b.create<vector::BroadcastOp>(loc, vt, scalar);
    vecMap[v] = bc;
    return bc;
  };

  for (Operation &op : orig.getBody()->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(&op)) {
      SmallVector<Value> newIdxs;
      for (Value idx : ld.getIndices())
        newIdxs.push_back(scalarMap.lookupOrDefault(idx));

      if (!ld.getIndices().empty() && ld.getIndices().back() == iv) {
        Type elemTy =
            cast<MemRefType>(ld.getMemRef().getType()).getElementType();
        VectorType vt = VectorType::get({vf}, elemTy);
        Value vec =
            b.create<vector::LoadOp>(loc, vt, ld.getMemRef(), newIdxs);
        vecMap[ld.getResult()] = vec;
      } else {
        auto *cloned = b.clone(op, scalarMap);
        scalarMap.map(ld.getResult(), cloned->getResult(0));
      }
      continue;
    }

    if (auto st = dyn_cast<memref::StoreOp>(&op)) {
      SmallVector<Value> newIdxs;
      for (Value idx : st.getIndices())
        newIdxs.push_back(scalarMap.lookupOrDefault(idx));

      if (!st.getIndices().empty() && st.getIndices().back() == iv) {
        Value stVal = st.getValueToStore();
        Type elemTy =
            cast<MemRefType>(st.getMemRef().getType()).getElementType();
        VectorType vt = VectorType::get({vf}, elemTy);
        Value vecStVal = vecMap.count(stVal) ? vecMap[stVal]
                                             : getVec(stVal, vt);
        b.create<vector::StoreOp>(loc, vecStVal, st.getMemRef(), newIdxs);
      } else {
        b.clone(op, scalarMap);
      }
      continue;
    }

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
      SmallVector<Value> newOperands;
      newOperands.reserve(op.getNumOperands());
      for (Value operand : op.getOperands())
        newOperands.push_back(getVec(operand, vecTy));

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
      auto *cloned = b.clone(op, scalarMap);
      for (auto [origRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults()))
        scalarMap.map(origRes, newRes);
    }
  }

  b.create<scf::YieldOp>(loc);
}

//===----------------------------------------------------------------------===//
// Reduction vectorized body builder (Mode B)
//===----------------------------------------------------------------------===//

/// Build the body of a vectorized reduction loop.
///
/// The vector accumulator iter_arg is pre-seeded in vecMap.  Loads at the IV
/// become vector.load; arith ops widen to vector; the combiner op produces the
/// new vector accumulator which is yielded.
static void buildVectorizedReductionBody(OpBuilder &b, Location loc,
                                          scf::ForOp orig, int64_t vf,
                                          Value newIV, Value vecAcc,
                                          const ReductionInfo &redInfo) {
  llvm::DenseMap<Value, Value> vecMap;
  IRMapping scalarMap;
  scalarMap.map(orig.getInductionVar(), newIV);

  Value iv = orig.getInductionVar();
  Value origIterArg = orig.getRegionIterArg(0);
  Type elemTy = origIterArg.getType();
  VectorType vecTy = VectorType::get({vf}, elemTy);

  // Pre-seed: the original scalar iter_arg → vector accumulator
  vecMap[origIterArg] = vecAcc;

  auto getVec = [&](Value v, VectorType vt) -> Value {
    auto it = vecMap.find(v);
    if (it != vecMap.end() && it->second.getType() == vt)
      return it->second;
    Value scalar = scalarMap.lookupOrDefault(v);
    Value bc = b.create<vector::BroadcastOp>(loc, vt, scalar);
    vecMap[v] = bc;
    return bc;
  };

  Value yieldValue;

  for (Operation &op : orig.getBody()->without_terminator()) {
    // Skip the yield — we build our own.
    if (isa<scf::YieldOp>(&op))
      continue;

    // ── memref.load ──
    if (auto ld = dyn_cast<memref::LoadOp>(&op)) {
      SmallVector<Value> newIdxs;
      for (Value idx : ld.getIndices())
        newIdxs.push_back(scalarMap.lookupOrDefault(idx));

      if (!ld.getIndices().empty() && ld.getIndices().back() == iv) {
        // Stride-1 load → vector.load
        Value vec = b.create<vector::LoadOp>(loc, vecTy, ld.getMemRef(),
                                              newIdxs);
        vecMap[ld.getResult()] = vec;
      } else {
        // Loop-invariant → scalar clone
        auto *cloned = b.clone(op, scalarMap);
        scalarMap.map(ld.getResult(), cloned->getResult(0));
      }
      continue;
    }

    // ── memref.store (loop-invariant only for reductions) ──
    if (auto st = dyn_cast<memref::StoreOp>(&op)) {
      b.clone(op, scalarMap);
      continue;
    }

    // ── Pure compute op ──
    VectorType opVecTy;
    bool hasVecOperand = false;
    for (Value operand : op.getOperands()) {
      auto it = vecMap.find(operand);
      if (it != vecMap.end() && isa<VectorType>(it->second.getType())) {
        hasVecOperand = true;
        opVecTy = cast<VectorType>(it->second.getType());
        break;
      }
    }

    if (hasVecOperand) {
      SmallVector<Value> newOperands;
      newOperands.reserve(op.getNumOperands());
      for (Value operand : op.getOperands())
        newOperands.push_back(getVec(operand, opVecTy));

      OperationState state(op.getLoc(), op.getName());
      state.addOperands(newOperands);
      for (unsigned i = 0; i < op.getNumResults(); ++i)
        state.addTypes(opVecTy);
      state.addAttributes(op.getAttrs());

      Operation *newOp = b.create(state);
      for (auto [origRes, newRes] :
           llvm::zip(op.getResults(), newOp->getResults()))
        vecMap[origRes] = newRes;

      // Track if this is the combiner — its result is what we yield.
      if (&op == redInfo.combinerOp)
        yieldValue = newOp->getResult(0);
    } else {
      auto *cloned = b.clone(op, scalarMap);
      for (auto [origRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults()))
        scalarMap.map(origRes, newRes);
    }
  }

  // Yield the vector accumulator.
  assert(yieldValue && "reduction combiner must produce the yield value");
  b.create<scf::YieldOp>(loc, yieldValue);
}

//===----------------------------------------------------------------------===//
// Transformation drivers
//===----------------------------------------------------------------------===//

/// Replace a non-reduction loop with vectorized main + scalar cleanup.
static void vectorizeNonReductionLoop(scf::ForOp loop, int64_t vf) {
  OpBuilder b(loop);
  Location loc = loop.getLoc();

  Value lb   = loop.getLowerBound();
  Value ub   = loop.getUpperBound();
  Value step = loop.getStep();

  // vub = ub - ((ub - lb) % VF)
  Value vfConst = b.create<arith::ConstantIndexOp>(loc, vf);
  Value range   = b.create<arith::SubIOp>(loc, ub, lb);
  Value rem     = b.create<arith::RemSIOp>(loc, range, vfConst);
  Value vub     = b.create<arith::SubIOp>(loc, ub, rem);
  Value vStep   = b.create<arith::ConstantIndexOp>(loc, vf);

  LLVM_DEBUG(llvm::dbgs() << "[nova-scf-loop-vectorize] vectorizing "
                           << "non-reduction loop with VF=" << vf << "\n");

  // Vectorized main loop: [lb, vub, VF)
  b.create<scf::ForOp>(
      loc, lb, vub, vStep,
      /*iterArgs=*/ValueRange{},
      [&](OpBuilder &ib, Location il, Value newIV, ValueRange /*args*/) {
        buildVectorizedBody(ib, il, loop, vf, newIV);
      });

  // Scalar cleanup loop: [vub, ub, 1)
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

  loop->erase();
}

/// Replace a reduction loop with vectorized main + vector.reduction + scalar
/// cleanup.
static void vectorizeReductionLoop(scf::ForOp loop, int64_t vf,
                                    const ReductionInfo &redInfo) {
  OpBuilder b(loop);
  Location loc = loop.getLoc();

  Value origLb   = loop.getLowerBound();
  Value origUb   = loop.getUpperBound();
  Value origStep = loop.getStep();
  Value origInit = loop.getInitArgs()[0];

  Type elemTy = origInit.getType();
  VectorType vecTy = VectorType::get({vf}, elemTy);
  Value vfConst = b.create<arith::ConstantIndexOp>(loc, vf);

  // ── Restructure access pattern for contiguous vector.load ────────────────
  //   newLb   = origLb * VF       (thread t starts at t*VF)
  //   newStep = origStep * VF     (e.g., 32*4=128)
  //   ub unchanged (ub % newStep == 0 guaranteed by legality)
  Value newLb   = b.create<arith::MulIOp>(loc, origLb, vfConst);
  Value newStep = b.create<arith::MulIOp>(loc, origStep, vfConst);

  // ── Build vector init: splat(identity) then insert initArg at lane 0 ─────
  // This ensures:
  //   - Lane 0 starts with the original init value
  //   - Lanes 1..VF-1 start with the identity (neutral element)
  //   - After vector.reduction, we get: initArg ⊕ (all loaded elements)
  Value identity = buildReductionIdentityValue(b, loc, redInfo.atomicKind,
                                                elemTy);
  Value vecInit = b.create<vector::SplatOp>(loc, vecTy, identity);
  Value zeroIdx = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
  vecInit = b.create<vector::InsertElementOp>(loc, origInit, vecInit, zeroIdx);

  LLVM_DEBUG(llvm::dbgs() << "[nova-scf-loop-vectorize] vectorizing "
                           << "reduction loop with VF=" << vf << "\n");

  // ── Vectorized main loop: [newLb, ub, newStep) with vector iter_arg ──────
  auto vecLoop = b.create<scf::ForOp>(
      loc, newLb, origUb, newStep,
      /*iterArgs=*/ValueRange{vecInit},
      [&](OpBuilder &ib, Location il, Value newIV, ValueRange args) {
        buildVectorizedReductionBody(ib, il, loop, vf, newIV, args[0],
                                      redInfo);
      });

  // ── Horizontal reduction: vector<VF x T> → scalar ───────────────────────
  Value vecResult = vecLoop.getResult(0);
  Value scalarResult = b.create<vector::ReductionOp>(
      loc, redInfo.vectorKind, vecResult);

  // ── Compute cleanup bounds ───────────────────────────────────────────────
  // The vectorized loop covered elements at indices:
  //   {newLb, newLb+VF, ..., newLb+VF*(tripCount-1), ...}
  // for each thread. Total elements covered = tripCount * VF per thread.
  // With the restructured pattern, all ub elements are covered when
  // ub % newStep == 0, so we need a cleanup loop for the original access
  // pattern over remaining elements.
  //
  // Cleanup runs the ORIGINAL pattern: [cleanupLb, origUb, origStep)
  // where cleanupLb = origLb + (vecTripCount * origStep * VF)
  //                  = origLb + ((origUb - newLb) / newStep) * newStep / VF
  //
  // Since ub % newStep == 0, the vectorized loop covers everything.
  // But we still emit the cleanup for safety; it will be zero-trip when
  // ub % (origStep * VF) == 0, which CSE/canonicalize can fold away.
  //
  // Actually, since legality guarantees ub % (origStep*VF) == 0, the
  // cleanup loop is always zero-trip. Emit it for correctness — the
  // canonicalizer will remove it.
  auto cleanupLoop = b.create<scf::ForOp>(
      loc, origUb, origUb, origStep,
      /*iterArgs=*/ValueRange{scalarResult},
      [&](OpBuilder &ib, Location il, Value newIV, ValueRange args) {
        IRMapping mp;
        mp.map(loop.getInductionVar(), newIV);
        mp.map(loop.getRegionIterArg(0), args[0]);
        Value cleanupYield;
        for (Operation &op : loop.getBody()->without_terminator()) {
          auto *cloned = ib.clone(op, mp);
          for (auto [origRes, newRes] :
               llvm::zip(op.getResults(), cloned->getResults()))
            mp.map(origRes, newRes);
          if (&op == redInfo.combinerOp)
            cleanupYield = cloned->getResult(0);
        }
        if (cleanupYield)
          ib.create<scf::YieldOp>(il, cleanupYield);
        else
          ib.create<scf::YieldOp>(il, args[0]);
      });

  // Replace the original loop's result with the cleanup loop's result.
  loop.getResult(0).replaceAllUsesWith(cleanupLoop.getResult(0));
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
           "using vector.load / vector.store / vector.broadcast / "
           "vector.reduction";
  }

  explicit NovaScfLoopVectorizePass(int64_t vf = 4) : vectorFactor(vf) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, arith::ArithDialect,
                    scf::SCFDialect, memref::MemRefDialect,
                    gpu::GPUDialect>();
  }

  void runOnOperation() override {
    if (vectorFactor <= 0)
      return;

    func::FuncOp func = getOperation();

    // Collect candidates in a snapshot — transformation erases the original.
    SmallVector<VectorizableLoop, 16> candidates;
    func.walk([&](scf::ForOp loop) {
      VectorizableLoop vl;
      if (analyzeLoop(loop, vectorFactor, vl))
        candidates.push_back(vl);
    });

    // Process in post-order (inner loops first).
    for (auto &vl : llvm::reverse(candidates)) {
      if (vl.kind == LoopKind::NonReduction)
        vectorizeNonReductionLoop(vl.loop, vectorFactor);
      else
        vectorizeReductionLoop(vl.loop, vectorFactor, vl.redInfo);
    }
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
