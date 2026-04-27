//===- NovaGenericVectorization.cpp ---------------------------------------===//
//
// Vectorize linalg ops to vector.* for the Nova MMA pipeline.
//
// Three categories:
//   A. MMA contraction ops (linalg.batch_matmul with nova.layout_* attrs):
//        Parks nova.layout_* + lowering_config on the enclosing scf.forall
//        before linalg::vectorize erases the op, then transfers them to the
//        vector.contract after Phase 2 canonicalization.
//        (The contract only appears after populateVectorReductionToContractPatterns
//        which runs via an internal rewriter; the forall is the stable anchor.)
//   B. Static-shape non-MMA ops: linalg::vectorize → vector.transfer_read/write.
//        Pure-gather generics (only index→extract→yield) are rewritten to
//        transfer_read directly.  Mixed bodies with a label-driven gather have
//        the tensor.extract hoisted out so no vector.gather is emitted.
//   C. Dynamic-shape ops: inferNovaVectorSizes() → ValueBounds UB analysis →
//        masked vectorize (mask eliminated in Phase 3 if statically all-true).
//
// Phases:
//   1. Vectorize all LinalgOps + tensor.pad (bottom-up).
//   2. Canonicalize to vector.contract.
//   2.5. Transfer parked nova.layout_* from forall → contract.
//   3. Eliminate always-true vector.mask ops.
//   4. Canonicalize mask predicates.
//   5. Lower vector.mask { transfer } to predicated form.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NovaVectorSizeUtils.h"
#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static constexpr int64_t kMaxVectorSize = 4096LL * 4096LL;

static constexpr StringLiteral kLayout0        = "nova.layout_0";
static constexpr StringLiteral kLayout1        = "nova.layout_1";
static constexpr StringLiteral kLayout2        = "nova.layout_2";
static constexpr StringLiteral kLoweringConfig = "lowering_config";

// Temporary attrs parked on the parent scf.forall during Phase 1.
static constexpr StringLiteral kPendingLayout0 = "nova._pl0";
static constexpr StringLiteral kPendingLayout1 = "nova._pl1";
static constexpr StringLiteral kPendingLayout2 = "nova._pl2";
static constexpr StringLiteral kPendingConfig  = "nova._pc";

//===----------------------------------------------------------------------===//
// Pure helpers
//===----------------------------------------------------------------------===//

static bool hasStaticShape(linalg::LinalgOp op) {
  for (OpOperand &operand : op->getOpOperands()) {
    auto shapedTy = dyn_cast<ShapedType>(operand.get().getType());
    if (shapedTy && !shapedTy.hasStaticShape())
      return false;
  }
  return true;
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp op) {
  int64_t maxFlat = 1;
  for (OpOperand &operand : op->getOpOperands()) {
    auto ty = dyn_cast<ShapedType>(operand.get().getType());
    if (!ty || !ty.hasStaticShape())
      return ty ? failure() : success();
    maxFlat = std::max(maxFlat, ty.getNumElements());
  }
  return success(maxFlat < kMaxVectorSize);
}

static bool isMmaContractionOp(Operation *op) {
  return op->hasAttr(kLayout0);
}

// Returns true when the linalg body is only index computation + tensor.extract
// + yield — no value-domain arithmetic.
static bool isPureGatherBody(linalg::LinalgOp linalgOp) {
  Block &body = linalgOp->getRegion(0).front();
  auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());
  if (yieldOp.getNumOperands() != 1)
    return false;
  Value yieldedVal = yieldOp.getOperand(0);
  if (!isa_and_nonnull<tensor::ExtractOp>(yieldedVal.getDefiningOp()))
    return false;

  for (Operation &op : body) {
    if (isa<linalg::YieldOp, linalg::IndexOp, tensor::ExtractOp,
            tensor::ExtractSliceOp, arith::IndexCastOp,
            affine::AffineApplyOp>(&op))
      continue;
    if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
      if (constOp.getType().isIntOrIndex())
        continue;
      return false; // float constant → real computation
    }
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp,
            arith::RemSIOp, arith::DivSIOp, arith::RemUIOp,
            arith::DivUIOp>(&op)) {
      if (llvm::all_of(op.getResultTypes(), [](Type t) { return t.isIntOrIndex(); }))
        continue;
      return false;
    }
    return false;
  }
  return true;
}

// Returns true if `val`'s def-chain mentions any linalg.IndexOp with dim == vecDim.
static bool mentionsVecDim(Value val, unsigned vecDim) {
  SmallVector<Value> worklist = {val};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;
    Operation *def = v.getDefiningOp();
    if (!def) continue;
    if (auto idx = dyn_cast<linalg::IndexOp>(def))
      if (idx.getDim() == vecDim) return true;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

// Returns true when extractOp inside linalgOp is stride-1 along vecDim.
// Fills varyingIdxPos and baseIndices (nullptr at varyingIdxPos).
static bool isContiguousExtract(linalg::LinalgOp linalgOp,
                                tensor::ExtractOp extractOp,
                                unsigned vecDim,
                                unsigned &varyingIdxPos,
                                SmallVectorImpl<Value> &baseIndices) {
  auto indices = extractOp.getIndices();
  unsigned numIdx = indices.size();
  int varyingCount = 0;
  int varying = -1;
  for (unsigned i = 0; i < numIdx; ++i) {
    if (!mentionsVecDim(indices[i], vecDim)) continue;
    ++varyingCount;
    varying = (int)i;
  }
  if (varyingCount != 1) return false;

  // Find the linalg.IndexOp leaf in the varying index chain.
  linalg::IndexOp idxLeaf;
  {
    SmallVector<Value> wl = {indices[varying]};
    llvm::SmallPtrSet<Value, 8> vis;
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!vis.insert(v).second) continue;
      if (auto op = dyn_cast_or_null<linalg::IndexOp>(v.getDefiningOp()))
        if (op.getDim() == vecDim) { idxLeaf = op; break; }
      if (auto *def = v.getDefiningOp())
        for (Value o : def->getOperands()) wl.push_back(o);
    }
  }
  if (!idxLeaf) return false;

  // Verify stride-1: linalg.index must not appear under any MulIOp.
  {
    SmallVector<Value> wl = {indices[varying]};
    llvm::SmallPtrSet<Value, 8> vis;
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!vis.insert(v).second || v == idxLeaf.getResult()) continue;
      auto *def = v.getDefiningOp();
      if (!def) continue;
      if (isa<arith::MulIOp>(def))
        for (Value o : def->getOperands())
          if (mentionsVecDim(o, vecDim)) return false;
      for (Value o : def->getOperands()) wl.push_back(o);
    }
  }

  varyingIdxPos = (unsigned)varying;
  baseIndices.resize(numIdx);
  for (unsigned i = 0; i < numIdx; ++i)
    baseIndices[i] = (i == (unsigned)varying) ? Value{} : indices[i];
  return true;
}

// Rewrites a pure-gather linalg.generic (body = index→extract→yield) to
// vector.transfer_read + broadcast + transfer_write.  Returns true if rewritten.
static bool tryRewriteContiguousExtract(IRRewriter &rewriter,
                                        linalg::LinalgOp linalgOp) {
  auto outTy = dyn_cast<RankedTensorType>(linalgOp->getResultTypes().front());
  if (!outTy || !outTy.hasStaticShape() || outTy.getRank() < 1) return false;
  if (!isPureGatherBody(linalgOp)) return false;

  unsigned vecDim = (unsigned)(outTy.getRank() - 1);
  int64_t vecWidth = outTy.getDimSize(vecDim);

  tensor::ExtractOp extractOp;
  for (Operation &op : linalgOp->getRegion(0).front())
    if (auto e = dyn_cast<tensor::ExtractOp>(&op)) { extractOp = e; break; }
  if (!extractOp) return false;

  unsigned varyingPos = 0;
  SmallVector<Value> baseIndices;
  if (!isContiguousExtract(linalgOp, extractOp, vecDim, varyingPos, baseIndices))
    return false;

  Location loc = linalgOp.getLoc();
  rewriter.setInsertionPoint(linalgOp);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Clone the varying index chain outside the body with linalg.index(vecDim) → c0.
  IRMapping mapping;
  std::function<Value(Value)> cloneOutside = [&](Value v) -> Value {
    if (mapping.contains(v)) return mapping.lookup(v);
    if (!linalgOp->getRegion(0).isAncestor(v.getParentRegion())) return v;
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      unsigned argIdx = ba.getArgNumber();
      if (argIdx < (unsigned)linalgOp.getNumDpsInputs()) {
        Value inputTensor = linalgOp.getDpsInputOperand(argIdx)->get();
        auto inTy = cast<RankedTensorType>(inputTensor.getType());
        SmallVector<Value> zeros(inTy.getRank(), c0);
        Value scalar = rewriter.create<tensor::ExtractOp>(loc, inputTensor, zeros);
        mapping.map(v, scalar);
        return scalar;
      }
      mapping.map(v, c0);
      return c0;
    }
    Operation *def = v.getDefiningOp();
    if (auto idx = dyn_cast<linalg::IndexOp>(def))
      if (idx.getDim() == vecDim) { mapping.map(v, c0); return c0; }
    SmallVector<Value> newOperands;
    for (Value operand : def->getOperands())
      newOperands.push_back(cloneOutside(operand));
    OperationState state(loc, def->getName());
    state.addOperands(newOperands);
    state.addTypes(def->getResultTypes());
    state.addAttributes(llvm::to_vector(def->getAttrs()));
    Operation *cloned = rewriter.create(state);
    for (auto [orig, rep] : llvm::zip(def->getResults(), cloned->getResults()))
      mapping.map(orig, rep);
    return mapping.lookup(v);
  };

  Value baseCol = cloneOutside(extractOp.getIndices()[varyingPos]);

  SmallVector<Value> transferIndices;
  for (unsigned i = 0; i < baseIndices.size(); ++i)
    transferIndices.push_back(i == varyingPos ? baseCol : cloneOutside(baseIndices[i]));

  Value srcTensor = extractOp.getTensor();
  auto elemTy = cast<RankedTensorType>(srcTensor.getType()).getElementType();
  VectorType vecTy = VectorType::get({vecWidth}, elemTy);
  Value pad = rewriter.create<arith::ConstantOp>(loc, elemTy,
                                                  rewriter.getZeroAttr(elemTy));
  Value flat = rewriter.create<vector::TransferReadOp>(
      loc, vecTy, srcTensor, transferIndices, pad, SmallVector<bool>(1, true));

  int64_t rank = outTy.getRank();
  VectorType outVecTy = VectorType::get(outTy.getShape(), elemTy);
  Value broad = rewriter.create<vector::BroadcastOp>(loc, outVecTy, flat);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, outTy.getShape(), elemTy);
  SmallVector<Value> zeros(rank, c0);
  Value result = rewriter.create<vector::TransferWriteOp>(
      loc, broad, empty, zeros, SmallVector<bool>(rank, true)).getResult();

  rewriter.replaceOp(linalgOp, result);
  return true;
}

// Hoists all tensor.extract ops out of a mixed-body linalg.generic
// (index computation + extracts + value arithmetic) so linalg::vectorize
// emits vector.transfer_read instead of vector.gather.
//
// For each tensor.extract in the body: computes the scalar per lane outside
// the body, packs into a new input tensor, rebuilds the generic with the
// extract replaced by the corresponding new block arg.  Handles one or more
// extracts (e.g. the embedding + position lookup in the token embedding kernel
// has two: %4[pos, col] and %3[tok, col]).
static bool tryHoistLabelDrivenExtract(IRRewriter &rewriter,
                                        linalg::LinalgOp linalgOp) {
  auto outTy = dyn_cast<RankedTensorType>(linalgOp->getResultTypes().front());
  if (!outTy || !outTy.hasStaticShape()) return false;
  if (isPureGatherBody(linalgOp)) return false; // handled by tryRewriteContiguousExtract
  // linalg::vectorize mis-assigns write targets for multi-output ops when
  // hoisted inputs shift operandSegmentSizes — skip and let the main loop
  // handle them with vectorizeNDExtract=true.
  if (linalgOp.getNumDpsInits() != 1) return false;

  Block &body = linalgOp->getRegion(0).front();

  // Collect all tensor.extract ops in the body.
  SmallVector<tensor::ExtractOp> extractOps;
  for (Operation &op : body)
    if (auto e = dyn_cast<tensor::ExtractOp>(&op))
      extractOps.push_back(e);
  if (extractOps.empty()) return false;

  // All extracted tensors must be defined outside the linalg body.
  for (tensor::ExtractOp e : extractOps) {
    Value src = e.getTensor();
    if (linalgOp->getRegion(0).isAncestor(src.getParentRegion())) return false;
    if (!dyn_cast<RankedTensorType>(src.getType())) return false;
  }

  int64_t numLanes = outTy.getNumElements();
  if (numLanes > 16) return false;

  Location loc = linalgOp.getLoc();
  rewriter.setInsertionPoint(linalgOp);

  int64_t rank = outTy.getRank();
  unsigned vecDim = (unsigned)(rank - 1);
  unsigned numIns = linalgOp.getNumDpsInputs();

  // Extract scalar from original input tensor at [0,..,lane] for each input.
  auto extractScalarInput = [&](unsigned inputIdx, int64_t lane) -> Value {
    Value inTensor = linalgOp.getDpsInputOperand(inputIdx)->get();
    auto inTy = dyn_cast<RankedTensorType>(inTensor.getType());
    if (!inTy) return {};
    SmallVector<Value> idxVals;
    for (int d = 0; d < (int)inTy.getRank(); ++d)
      idxVals.push_back(rewriter.create<arith::ConstantIndexOp>(
          loc, d == (int)inTy.getRank() - 1 ? lane : 0));
    return rewriter.create<tensor::ExtractOp>(loc, inTensor, idxVals);
  };

  // Clone a value from the body for a concrete lane, substituting block args
  // with per-lane scalars and linalg.index(vecDim) with the lane constant.
  // `extractResults` maps each original extract result to its already-cloned
  // scalar for this lane (so the index chain of one extract can reference
  // results of another op that was already cloned).
  auto cloneForLane = [&](Value v, int64_t lane,
                           const SmallVectorImpl<Value> &inputScalars,
                           IRMapping &mapping) -> Value {
    std::function<Value(Value)> clone = [&](Value val) -> Value {
      if (mapping.contains(val)) return mapping.lookup(val);
      if (!linalgOp->getRegion(0).isAncestor(val.getParentRegion())) return val;
      if (auto ba = dyn_cast<BlockArgument>(val)) {
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        mapping.map(val, zero);
        return zero;
      }
      Operation *def = val.getDefiningOp();
      if (auto idxOp = dyn_cast<linalg::IndexOp>(def)) {
        Value c = rewriter.create<arith::ConstantIndexOp>(
            loc, idxOp.getDim() == vecDim ? lane : 0);
        mapping.map(val, c);
        return c;
      }
      SmallVector<Value> newOps;
      for (Value o : def->getOperands()) newOps.push_back(clone(o));
      OperationState state(loc, def->getName());
      state.addOperands(newOps);
      state.addTypes(def->getResultTypes());
      state.addAttributes(llvm::to_vector(def->getAttrs()));
      Operation *cloned = rewriter.create(state);
      for (auto [orig, rep] : llvm::zip(def->getResults(), cloned->getResults()))
        mapping.map(orig, rep);
      return mapping.lookup(val);
    };
    return clone(v);
  };

  // For each tensor.extract, build a hoisted tensor<numLanes x elemTy> (flat),
  // then reshape to match outTy.
  auto packAndReshape = [&](ArrayRef<Value> scalars, Type elemTy) -> Value {
    Value packed = rewriter.create<tensor::FromElementsOp>(
        loc, RankedTensorType::get({numLanes}, elemTy), scalars);
    if (rank == 1) return packed;
    SmallVector<ReassociationIndices> reassoc(1);
    for (int64_t d = 0; d < rank; ++d) reassoc[0].push_back(d);
    return rewriter.create<tensor::ExpandShapeOp>(
        loc, RankedTensorType::get(outTy.getShape(), elemTy), packed, reassoc);
  };

  // Build the hoisted input tensors — one per tensor.extract.
  SmallVector<Value> hoistedTensors;
  hoistedTensors.reserve(extractOps.size());

  for (tensor::ExtractOp extractOp : extractOps) {
    Value srcTensor = extractOp.getTensor();
    auto elemTy = cast<RankedTensorType>(srcTensor.getType()).getElementType();
    SmallVector<Value> laneScalars;
    for (int64_t lane = 0; lane < numLanes; ++lane) {
      // Build a shared mapping per lane: block args → per-lane input scalars.
      IRMapping laneMapping;
      SmallVector<Value> inputScalars(numIns);
      for (unsigned i = 0; i < numIns; ++i)
        inputScalars[i] = extractScalarInput(i, lane);
      for (unsigned i = 0; i < numIns; ++i)
        laneMapping.map(linalgOp.getRegionInputArgs()[i], inputScalars[i]);

      SmallVector<Value> idxCloned;
      for (Value idx : extractOp.getIndices())
        idxCloned.push_back(cloneForLane(idx, lane, inputScalars, laneMapping));
      laneScalars.push_back(
          rewriter.create<tensor::ExtractOp>(loc, srcTensor, idxCloned));
    }
    hoistedTensors.push_back(packAndReshape(laneScalars, elemTy));
  }

  // Rebuild linalg.generic: append hoisted tensors as new inputs (before outs).
  SmallVector<Value> newInputs(linalgOp.getDpsInputs());
  for (Value h : hoistedTensors) newInputs.push_back(h);

  // Build new map list: [orig_input_maps..., hoisted_input_maps..., output_maps...].
  // Hoisted tensors are packed to out0's shape (packAndReshape uses outTy), so they
  // use out0's affine map (oldMaps[numIns]). Using oldMaps.back() is wrong for
  // multi-output ops because it picks the last output's (possibly lower-rank) map.
  auto oldMaps = linalgOp.getIndexingMapsArray();
  AffineMap hoistedMap = oldMaps[numIns];
  SmallVector<AffineMap> newMaps;
  newMaps.append(oldMaps.begin(), oldMaps.begin() + numIns);
  for (size_t i = 0; i < extractOps.size(); ++i)
    newMaps.push_back(hoistedMap);
  newMaps.append(oldMaps.begin() + numIns, oldMaps.end());

  unsigned numExtracts = (unsigned)extractOps.size();

  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
      loc,
      linalgOp->getResultTypes(),
      newInputs, linalgOp.getDpsInits(),
      newMaps,
      linalgOp.getIteratorTypesArray(),
      [&](OpBuilder &b, Location innerLoc, ValueRange args) {
        // args layout: [original inputs...] [hoisted inputs...] [original outs...]
        // original outs start at: numIns + numExtracts
        IRMapping bodyMap;
        for (unsigned i = 0; i < numIns; ++i)
          bodyMap.map(body.getArgument(i), args[i]);
        unsigned numOuts = linalgOp.getNumDpsInits();
        for (unsigned i = 0; i < numOuts; ++i)
          bodyMap.map(body.getArgument(numIns + i), args[numIns + numExtracts + i]);
        // Map each extract result → its hoisted block arg.
        for (unsigned i = 0; i < numExtracts; ++i)
          bodyMap.map(extractOps[i].getResult(), args[numIns + i]);

        for (Operation &op : body) {
          // Skip all original extract ops — replaced by block args above.
          if (llvm::is_contained(extractOps, &op)) continue;
          if (isa<linalg::YieldOp>(&op)) {
            SmallVector<Value> yieldVals;
            for (Value v : cast<linalg::YieldOp>(op).getValues())
              yieldVals.push_back(bodyMap.lookupOrDefault(v));
            b.create<linalg::YieldOp>(innerLoc, yieldVals);
            continue;
          }
          SmallVector<Value> newOperands;
          for (Value operand : op.getOperands())
            newOperands.push_back(bodyMap.lookupOrDefault(operand));
          OperationState state(innerLoc, op.getName());
          state.addOperands(newOperands);
          state.addTypes(op.getResultTypes());
          state.addAttributes(llvm::to_vector(op.getAttrs()));
          Operation *cloned = b.create(state);
          for (auto [orig, rep] : llvm::zip(op.getResults(), cloned->getResults()))
            bodyMap.map(orig, rep);
        }
      });

  // Copy non-structural attrs from original op.
  MLIRContext *ctx = rewriter.getContext();
  auto mapsAttrName = StringAttr::get(ctx, "indexing_maps");
  auto iterAttrName = StringAttr::get(ctx, "iterator_types");
  for (auto attr : linalgOp->getAttrs())
    if (attr.getName() != mapsAttrName && attr.getName() != iterAttrName)
      newLinalgOp->setAttr(attr.getName(), attr.getValue());

  rewriter.replaceOp(linalgOp, newLinalgOp->getResults());

  // Immediately vectorize — the new generic has no tensor.extract.
  rewriter.setInsertionPoint(newLinalgOp);
  FailureOr<linalg::VectorizationResult> vecResult =
      linalg::vectorize(rewriter, newLinalgOp, {}, {}, /*vectorizeNDExtract=*/false);
  if (succeeded(vecResult))
    rewriter.replaceOp(newLinalgOp, vecResult->replacements);

  return true;
}

static LogicalResult vectorizeMmaOp(IRRewriter &rewriter, Operation *op) {
  // Park layout attrs on the parent scf.forall before linalg::vectorize erases op.
  // The forall is stable across Phase 2 while the linalg op is not.
  auto parentForall = op->getParentOfType<scf::ForallOp>();
  if (!parentForall) {
    op->emitError("nova-generic-vectorization: MMA batch_matmul is not "
                  "enclosed in an scf.forall — cannot park layout attrs");
    return failure();
  }
  if (auto a = op->getAttr(kLayout0))        parentForall->setAttr(kPendingLayout0, a);
  if (auto a = op->getAttr(kLayout1))        parentForall->setAttr(kPendingLayout1, a);
  if (auto a = op->getAttr(kLayout2))        parentForall->setAttr(kPendingLayout2, a);
  if (auto a = op->getAttr(kLoweringConfig)) parentForall->setAttr(kPendingConfig,  a);

  FailureOr<linalg::VectorizationResult> result =
      linalg::vectorize(rewriter, op, {}, {}, /*vectorizeNDExtract=*/true);
  if (failed(result)) return failure();

  rewriter.replaceOp(op, result->replacements);
  return success();
}

// Walks vector.contracts after Phase 2 and transfers pending nova.layout_*
// attrs from the enclosing scf.forall to the contract.
static void transferParkedLayoutAttrs(func::FuncOp funcOp) {
  funcOp.walk([&](vector::ContractionOp contractOp) {
    if (contractOp->hasAttr(kLayout0)) return;
    scf::ForallOp forall = contractOp->getParentOfType<scf::ForallOp>();
    while (forall) {
      if (!forall->hasAttr(kPendingLayout0)) {
        forall = forall->getParentOfType<scf::ForallOp>();
        continue;
      }
      if (auto a = forall->getAttr(kPendingLayout0)) contractOp->setAttr(kLayout0, a);
      if (auto a = forall->getAttr(kPendingLayout1)) contractOp->setAttr(kLayout1, a);
      if (auto a = forall->getAttr(kPendingLayout2)) contractOp->setAttr(kLayout2, a);
      if (auto a = forall->getAttr(kPendingConfig))  contractOp->setAttr(kLoweringConfig, a);
      forall->removeAttr(kPendingLayout0);
      forall->removeAttr(kPendingLayout1);
      forall->removeAttr(kPendingLayout2);
      forall->removeAttr(kPendingConfig);
      return;
    }
  });
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct NovaGenericVectorizationPass
    : public PassWrapper<NovaGenericVectorizationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGenericVectorizationPass)

  StringRef getArgument() const override { return "nova-generic-vectorization"; }
  StringRef getDescription() const override {
    return "Vectorize linalg ops to vector.contract / vector.transfer_*. "
           "MMA ops: park nova.layout_* on parent scf.forall, vectorize, "
           "then transfer to vector.contract after Phase 2 canonicalization. "
           "Static ops: direct vectorization. "
           "Dynamic ops: masked vectorization via ValueBounds inference.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect,
                    func::FuncDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override;
};

void NovaGenericVectorizationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = funcOp.getContext();
  IRRewriter rewriter(ctx);

  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp, tensor::PadOp>(op))
      candidates.push_back(op);
  });
  std::reverse(candidates.begin(), candidates.end()); // bottom-up

  for (Operation *op : candidates) {
    if (!op || op->getBlock() == nullptr) continue;
    rewriter.setInsertionPoint(op);

    // Case D: tensor.pad
    if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      auto ty = padOp.getResultType();
      if (!ty.hasStaticShape()) continue;
      SmallVector<int64_t> vectorSizes(ty.getShape());
      int64_t flat = std::accumulate(vectorSizes.begin(), vectorSizes.end(),
                                     int64_t(1), std::multiplies<int64_t>{});
      if (flat >= kMaxVectorSize) continue;
      SmallVector<bool> scalableDims(vectorSizes.size(), false);
      FailureOr<linalg::VectorizationResult> result =
          linalg::vectorize(rewriter, op, vectorSizes, scalableDims, true);
      if (succeeded(result))
        rewriter.replaceOp(op, result->replacements);
      continue;
    }

    auto linalgOp = cast<linalg::LinalgOp>(op);

    // Case A: MMA contraction op
    if (isMmaContractionOp(op)) {
      if (!hasStaticShape(linalgOp)) {
        op->emitWarning() << "nova-generic-vectorization: MMA op has dynamic shape — skipping";
        continue;
      }
      if (failed(isWithinVectorSizeLimit(linalgOp))) continue;
      if (failed(vectorizeMmaOp(rewriter, op))) {
        op->emitError() << "nova-generic-vectorization: failed to vectorize MMA op";
        return signalPassFailure();
      }
      continue;
    }
    // Case B: Static non-MMA linalg op
    if (hasStaticShape(linalgOp)) {
      if (failed(isWithinVectorSizeLimit(linalgOp))) continue;

      // Skip global→shared promotion copies and ops feeding them.
      // These must stay as linalg.copy so downstream passes can emit cp.async.
      if (op->hasAttr("nova.promote_to_workgroup")) continue;
      if (llvm::any_of(op->getUsers(), [](Operation *user) {
            return user->hasAttr("nova.promote_to_workgroup");
          }))
        continue;

      if (tryRewriteContiguousExtract(rewriter, linalgOp)) continue;
      if (tryHoistLabelDrivenExtract(rewriter, linalgOp)) continue;

      FailureOr<linalg::VectorizationResult> result =
          linalg::vectorize(rewriter, op, {}, {}, /*vectorizeNDExtract=*/true);
      if (succeeded(result))
        rewriter.replaceOp(op, result->replacements);
      continue;
    }

    // Case C: Dynamic linalg op — masked vectorization
    std::optional<nova::NovaVectorizationTileSizes> maybeSizes =
        nova::inferNovaVectorSizes(linalgOp);
    if (!maybeSizes) continue;
    int64_t flat = std::accumulate(maybeSizes->vectorSizes.begin(),
                                   maybeSizes->vectorSizes.end(),
                                   int64_t(1), std::multiplies<int64_t>{});
    if (flat >= kMaxVectorSize) continue;
    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op, maybeSizes->vectorSizes,
                          maybeSizes->vectorScalableFlags, true);
    if (succeeded(result))
      rewriter.replaceOp(op, result->replacements);
  }

  // Phase 2: Canonicalize to vector.contract.
  // IMPORTANT: Run populateVectorReductionToContractPatterns FIRST in its own
  // pass before populateVectorMultiReductionLoweringPatterns(InnerParallel).
  // linalg::vectorize for linalg.matmul produces vector.multi_reduction. If both
  // patterns compete in the same set, InnerParallel fires first and destroys the
  // multi_reduction into extract/insert chains before the contract conversion runs.
  // Phase 2a: multi_reduction → vector.contract (must be alone so it wins)
  {
    RewritePatternSet contractPatterns(ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(contractPatterns);
    vector::populateVectorReductionToContractPatterns(contractPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(contractPatterns))))
      return signalPassFailure();
  }
  // Phase 2b: Lower any remaining multi_reductions (SIMT path that didn't become contracts).
  // Use InnerReduction so full scalar reductions (all dims are reduction, no
  // parallel dim) lower to vector.reduction on 1-D slices instead of
  // scalarizing into N individual vector.extract + arith.addf ops.
  // InnerParallel is wrong here: when there are no parallel dims it has
  // nothing to keep "inner parallel" and falls back to element-by-element extraction.
  {
    RewritePatternSet simdPatterns(ctx);
    vector::populateSinkVectorOpsPatterns(simdPatterns);
    vector::populateVectorMultiReductionLoweringPatterns(
        simdPatterns, vector::VectorMultiReductionLowering::InnerReduction);
    if (failed(applyPatternsGreedily(funcOp, std::move(simdPatterns))))
      return signalPassFailure();
  }
  // Phase 2.5: Transfer parked nova.layout_* from forall → contract.
  transferParkedLayoutAttrs(funcOp);

  // Phase 3: Eliminate always-true vector.mask ops.
  vector::eliminateVectorMasks(rewriter, funcOp, /*vscaleRange=*/std::nullopt);

  // Phase 4: Canonicalize mask predicates.
  {
    RewritePatternSet maskCanonPatterns(ctx);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(maskCanonPatterns);
    tensor::DimOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns, ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(maskCanonPatterns))))
      return signalPassFailure();
  }
  // Phase 5: Lower vector.mask { transfer } to predicated form.
  {
    RewritePatternSet maskLowerPatterns(ctx);
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(maskLowerPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(maskLowerPatterns))))
      return signalPassFailure();
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGenericVectorizationPass() {
  return std::make_unique<NovaGenericVectorizationPass>();
}

void registerNovaGenericVectorizationPass() {
  PassRegistration<NovaGenericVectorizationPass>();
}

} // namespace mlir::nova
