//===- NovaGPUVectorDistribute.cpp - Per-thread vector distribution -------===//
//
// Distributes vector.contract ops (and their feeding transfer_read/write ops)
// so that each GPU thread holds only its own slice of the computation.
//
// Before this pass every thread redundantly holds the full tile:
//   %lhs = vector.transfer_read ... : vector<1x16x64xf32>
//   %rhs = vector.transfer_read ... : vector<1x64x8xf32>
//   %acc = vector.transfer_read ... : vector<1x16x8xf32>
//   %r   = vector.contract %lhs, %rhs, %acc
//
// After this pass each thread holds only its elem_counts slice:
//   %lhs_t = vector.transfer_read ...[0, t_row, t_col] : vector<1x1x4xf32>
//   %rhs_t = vector.transfer_read ...[0, t_k,   t_col] : vector<1x2x1xf32>
//   %acc_t = vector.transfer_read ...[0, t_row,  t_col]: vector<1x2x2xf32>
//   %r_t   = vector.contract %lhs_t, %rhs_t, %acc_t
//             ← one warp-level MMA instruction after UnrollToIntrinsics
//
// Layout information is read from the nova.layout_0/1/2 DictionaryAttr
// that NovaGPUConfigureTensorLayouts attached to the contract op.
//
// Thread offset formula (per dimension d):
//   warp_id         = linearThreadId / warp_size
//   warp_offset_d   = (warp_id   / sg_strides[d])     % sg_counts[d]
//                     * elem_counts[d] * thread_counts[d]
//   thread_offset_d = (tid / thread_strides[d])        % thread_counts[d]
//                     * elem_counts[d]
//   total_offset_d  = warp_offset_d + thread_offset_d
//   (dims with stride 0 are not distributed → offset = 0)
//
// Pipeline position:
//   NovaGPUVectorAllocPass        → stages operands through shared memory
//   NovaGPUCombineValueBarriers   → merges value_barrier ops
//   THIS PASS                     ← distribute to per-thread slices
//   NovaGPUUnrollToIntrinsics     → unroll batch_counts → nvgpu.mma.sync
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-vector-distribute"

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// §1  Layout struct and DictionaryAttr reader
//===----------------------------------------------------------------------===//

/// Per-operand layout extracted from a nova.layout_* DictionaryAttr.
struct OperandLayout {
  SmallVector<int64_t> sgCounts;      // warps per dim within workgroup
  SmallVector<int64_t> sgStrides;     // strides to linearise warp index
  SmallVector<int64_t> batchCounts;   // MMA tiles each warp owns per dim
  SmallVector<int64_t> threadCounts;  // threads per dim within one MMA tile
  SmallVector<int64_t> threadStrides; // strides to linearise thread index
  SmallVector<int64_t> elemCounts;    // elements each thread holds per dim
  int rank = 0;
};

/// Extract int64 array from an ArrayAttr field of a DictionaryAttr.
static SmallVector<int64_t> getI64Array(DictionaryAttr dict, StringRef key) {
  auto attr = dict.getAs<ArrayAttr>(key);
  if (!attr) return {};
  SmallVector<int64_t> result;
  result.reserve(attr.size());
  for (Attribute a : attr)
    result.push_back(cast<IntegerAttr>(a).getInt());
  return result;
}

/// Read nova.layout_<idx> DictionaryAttr from |op| into |layout|.
static FailureOr<OperandLayout>
readOperandLayout(Operation *op, int idx) {
  auto attr = op->getAttrOfType<DictionaryAttr>(
      "nova.layout_" + std::to_string(idx));
  if (!attr) return failure();

  OperandLayout L;
  L.sgCounts      = getI64Array(attr, "sg_counts");
  L.sgStrides     = getI64Array(attr, "sg_strides");
  L.batchCounts   = getI64Array(attr, "batch_counts");
  L.threadCounts  = getI64Array(attr, "thread_counts");
  L.threadStrides = getI64Array(attr, "thread_strides");
  L.elemCounts    = getI64Array(attr, "elem_counts");
  L.rank          = (int)L.elemCounts.size();

  if (L.rank == 0 || L.sgCounts.size() != (size_t)L.rank ||
      L.threadCounts.size() != (size_t)L.rank ||
      L.elemCounts.size() != (size_t)L.rank)
    return failure();

  return L;
}

/// Returns the index of the reduction dimension in the contraction iteration space.
static int getReductionDim(vector::ContractionOp op) {
  auto iteratorTypes = op.getIteratorTypes().getValue();
  for (int i = 0; i < (int)iteratorTypes.size(); ++i) {
    if (cast<vector::IteratorTypeAttr>(iteratorTypes[i]).getValue() ==
        vector::IteratorType::reduction)
      return i;
  }
  return -1;
}

/// Maps a contract iteration dimension to an operand physical dimension index.
static int mapContractDimToOperandDim(AffineMap map, int contractDim) {
  for (int i = 0; i < (int)map.getNumResults(); ++i) {
    if (auto expr = llvm::dyn_cast<AffineDimExpr>(map.getResult(i))) {
      if (expr.getPosition() == (unsigned)contractDim)
        return i;
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// §2  Thread-offset computation
//===----------------------------------------------------------------------===//

/// Warp size = product of thread_counts (total threads inside one MMA tile).
static int64_t computeWarpSize(const OperandLayout &L) {
  int64_t ws = 1;
  for (int64_t tc : L.threadCounts) ws *= tc;
  return ws;
}

/// Emit arith ops to compute the per-thread offset along one dimension.
///
///   warp_id         = linearTid / warpSize
///   warp_offset_d   = (warp_id   / sg_strides[d]) % sg_counts[d]
///                     * elem_counts[d] * thread_counts[d]
///   thread_offset_d = (linearTid / thread_strides[d]) % thread_counts[d]
///                     * elem_counts[d]
///   total_offset_d  = warp_offset_d + thread_offset_d
///
/// If stride == 0 the dimension is not distributed → returns constant 0.
static SmallVector<Value>
computeDistributedOffsets(OpBuilder &b, Location loc,
                          Value linearTid, int64_t warpSize,
                          const OperandLayout &L) {
  auto idx  = b.getIndexType();
  auto cst  = [&](int64_t v) -> Value {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };

  // warp_id = linearTid / warpSize
  Value warpId = b.create<arith::DivUIOp>(loc, linearTid, cst(warpSize));

  SmallVector<Value> offsets(L.rank);
  for (int d = 0; d < L.rank; ++d) {
    Value off = cst(0);

    // ── Thread offset within the warp ────────────────────────────────
    // stride 0 means "not distributed on this dim" → offset stays 0.
    if (L.threadStrides[d] != 0 && L.threadCounts[d] > 1) {
      // thread_index_d = (linearTid / stride) % count
      Value div = b.create<arith::DivUIOp>(loc, linearTid,
                                          cst(L.threadStrides[d]));
      Value rem = b.create<arith::RemUIOp>(loc, div, cst(L.threadCounts[d]));
      // thread_offset_d = thread_index_d * elem_counts[d]
      off = b.create<arith::MulIOp>(loc, rem, cst(L.elemCounts[d]));
    }

    // ── Warp (subgroup) offset ────────────────────────────────────────
    if (L.sgStrides[d] != 0 && L.sgCounts[d] > 1) {
      // warp_index_d = (warp_id / sg_stride) % sg_count
      Value div2 = b.create<arith::DivUIOp>(loc, warpId,
                                           cst(L.sgStrides[d]));
      Value rem2 = b.create<arith::RemUIOp>(loc, div2, cst(L.sgCounts[d]));
      // warp_offset_d = warp_index_d * (elem * thread)
      Value stride = cst(L.elemCounts[d] * L.threadCounts[d]);
      Value warpOff = b.create<arith::MulIOp>(loc, rem2, stride);
      off = b.create<arith::AddIOp>(loc, off, warpOff);
    }

    offsets[d] = off;
  }
  return offsets;
}

//===----------------------------------------------------------------------===//
// §3  Per-thread vector type
//===----------------------------------------------------------------------===//

/// Build the per-thread VectorType from the layout.
///
/// For non-K (parallel) dimensions the shape is batchCounts[d] * elemCounts[d]
/// so that UnrollToIntrinsics can still see and unroll the batch tiles.
///
/// For the K (reduction) dimension the K-batch loop in distributeContractOp
/// already iterates over batchCounts[K], so each contract slice only covers
/// elemCounts[K].  kAxis is the physical axis index for K in this operand
/// (-1 if this operand has no K axis, e.g. ACC).
static VectorType getPerThreadVectorType(VectorType fullType,
                                          const OperandLayout &L,
                                          int kAxis) {
  SmallVector<int64_t> shape(L.rank);
  for (int d = 0; d < L.rank; ++d)
  if(d==kAxis){
    shape[d]=L.elemCounts[d];
  }
  else{
    shape[d]=L.batchCounts[d]*L.elemCounts[d];
  } 
  return VectorType::get(shape, fullType.getElementType());
}

//===----------------------------------------------------------------------===//
// §4  Distribute one vector.transfer_read feeding the contract
//===----------------------------------------------------------------------===//

/// Replace |readOp| with a sliced read at the thread's offset.
/// |offsets| contains one Value per dim of the operand (already projected).
/// Returns the new (smaller) vector value.
static Value distributeTransferRead(OpBuilder &b,
                                     vector::TransferReadOp readOp,
                                     ArrayRef<Value> offsets,
                                     VectorType perThreadType) {
  Location loc = readOp.getLoc();

  // Build new indices = original_indices + thread_offsets.
  // original indices are usually all-zero from VectorAlloc.
  SmallVector<Value> newIndices(readOp.getIndices().begin(),
                                readOp.getIndices().end());
  assert(newIndices.size() == offsets.size() &&
         "index count must match operand rank");
  for (int i = 0; i < (int)newIndices.size(); ++i) {
    if (auto cst = newIndices[i].getDefiningOp<arith::ConstantIndexOp>()) {
      if (cst.value() == 0 && offsets[i]) {
        newIndices[i] = offsets[i];
        continue;
      }
    }
    newIndices[i] = b.create<arith::AddIOp>(loc, newIndices[i], offsets[i]);
  }

  // All dims are in-bounds for the per-thread slice.
  SmallVector<bool> inBounds(perThreadType.getRank(), true);

  return b.create<vector::TransferReadOp>(
              loc, perThreadType, readOp.getBase(), newIndices,
              /*padding=*/std::nullopt, inBounds)
      .getResult();
}

//===----------------------------------------------------------------------===//
// §5  Distribute one vector.transfer_write that writes the result
//===----------------------------------------------------------------------===//

/// Replace |writeOp| with a sliced write at the thread's offset.
static void distributeTransferWrite(OpBuilder &b,
                                     vector::TransferWriteOp writeOp,
                                     Value perThreadResult,
                                     ArrayRef<Value> offsets) {
  Location loc = writeOp.getLoc();

  SmallVector<Value> newIndices(writeOp.getIndices().begin(),
                                writeOp.getIndices().end());
  for (int i = 0; i < (int)newIndices.size(); ++i) {
    if (auto cst = newIndices[i].getDefiningOp<arith::ConstantIndexOp>()) {
      if (cst.value() == 0 && offsets[i]) {
        newIndices[i] = offsets[i];
        continue;
      }
    }
    newIndices[i] = b.create<arith::AddIOp>(loc, newIndices[i], offsets[i]);
  }

int r = llvm::cast<mlir::VectorType>(perThreadResult.getType()).getRank();

  SmallVector<bool> inBounds(r, true);

  // The write result (updated tensor) replaces the original write result.
  auto newWrite = b.create<vector::TransferWriteOp>(
      loc, perThreadResult, writeOp.getBase(), newIndices, inBounds);
  writeOp.replaceAllUsesWith(newWrite->getResults());
}

//===----------------------------------------------------------------------===//
// §6  Main per-contract distribution
//===----------------------------------------------------------------------===//

/// Distribute a single vector.contract and its surrounding transfer ops.
static LogicalResult distributeContractOp(IRRewriter &rewriter,
                                           vector::ContractionOp contractOp) {
  Location loc = contractOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  // Insert ALL new ops just before the contract — this is inside the
  // thread-mapped scf.forall body, which is the correct GPU kernel scope.
  // gpu.thread_id must live inside the thread forall, not at function top,
  // so that ConvertForallToGPU sees it inside the gpu.launch block.
  rewriter.setInsertionPoint(contractOp);

  // ── Compute linearThreadId ──────────────────────────────────────────────
  // thread_id is inserted here (inside the thread forall) so it ends up
  // inside the gpu.launch block after forall lowering.
  // We use a pure 1-D layout: threads are mapped as linear_dim_0 → tidX.
  // tidY and tidZ are unused in the current MMA mapping (all dims collapse
  // to a single linear thread axis).
  Value tidX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value linearTid = tidX;

  // ── Read layouts for all three operands ──────────────────────────────
  FailureOr<OperandLayout> L0 = readOperandLayout(contractOp, 0); // LHS
  FailureOr<OperandLayout> L1 = readOperandLayout(contractOp, 1); // RHS
  FailureOr<OperandLayout> L2 = readOperandLayout(contractOp, 2); // ACC
  if (failed(L0) || failed(L1) || failed(L2)) {
    contractOp->emitError(
        "nova-gpu-vector-distribute: missing or malformed nova.layout_* attrs");
    return failure();
  }

  // ── Find the K-batch axis and count ──────────────────────────────────
  int kDim = getReductionDim(contractOp);
  int lhsKAxis = -1;
  int rhsKAxis = -1;
  int64_t kBatchCount = 1;
  int64_t lhsKStep = 0;
  int64_t rhsKStep = 0;

  if (kDim != -1) {
    auto maps = contractOp.getIndexingMapsArray();
    lhsKAxis = mapContractDimToOperandDim(maps[0], kDim);
    rhsKAxis = mapContractDimToOperandDim(maps[1], kDim);
    if (lhsKAxis != -1 && rhsKAxis != -1) {
      // Validate that both operands agree on K-batch count and K-step size.
      // If they disagree the layout pass produced inconsistent K layouts.
      int64_t lhsKBatch = L0->batchCounts[lhsKAxis];
      int64_t rhsKBatch = L1->batchCounts[rhsKAxis];
      if (lhsKBatch != rhsKBatch) {
        contractOp->emitError(
            "nova-gpu-vector-distribute: LHS K batch_count (")
            << lhsKBatch << ") != RHS K batch_count (" << rhsKBatch
            << "); layout pass produced inconsistent K decomposition";
        return failure();
      }
      lhsKStep = L0->threadCounts[lhsKAxis] * L0->elemCounts[lhsKAxis];
      rhsKStep = L1->threadCounts[rhsKAxis] * L1->elemCounts[rhsKAxis];
      if (lhsKStep != rhsKStep) {
        contractOp->emitError(
            "nova-gpu-vector-distribute: LHS K step (")
            << lhsKStep << ") != RHS K step (" << rhsKStep
            << "); thread_counts*elem_counts must match on K axis";
        return failure();
      }
      kBatchCount = lhsKBatch;
    }
  }

  // Use warp size from LHS (all operands share the same warp).
  int64_t warpSize = computeWarpSize(*L0);

  // ── Compute per-thread offsets for each operand ───────────────────────
  SmallVector<Value> off0 =
      computeDistributedOffsets(rewriter, loc, linearTid, warpSize, *L0);
  SmallVector<Value> off1 =
      computeDistributedOffsets(rewriter, loc, linearTid, warpSize, *L1);
  SmallVector<Value> off2 =
      computeDistributedOffsets(rewriter, loc, linearTid, warpSize, *L2);

  // ── Build per-thread vector types ────────────────────────────────────
  auto lhsType = cast<VectorType>(contractOp.getLhs().getType());
  auto rhsType = cast<VectorType>(contractOp.getRhs().getType());
  auto accType = cast<VectorType>(contractOp.getAcc().getType());

  // ACC has no K axis (-1), so all its dims get batchCounts * elemCounts.
  VectorType lhsThread = getPerThreadVectorType(lhsType, *L0, lhsKAxis);
  VectorType rhsThread = getPerThreadVectorType(rhsType, *L1, rhsKAxis);
  VectorType accThread = getPerThreadVectorType(accType,  *L2, /*kAxis=*/-1);

  // ── Find original reads ──────────────────────────────────────────────
  auto lhsRead = contractOp.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto rhsRead = contractOp.getRhs().getDefiningOp<vector::TransferReadOp>();
  auto accRead = contractOp.getAcc().getDefiningOp<vector::TransferReadOp>();

  if (!lhsRead || !rhsRead || !accRead) {
    contractOp->emitError("nova-gpu-vector-distribute: "
                          "contract operands must be transfer_reads");
    return failure();
  }

  // ── Read initial ACC ─────────────────────────────────────────────────
  // Each thread owns its ACC slice (distributeTransferRead).
  Value acc = distributeTransferRead(rewriter, accRead, off2, accThread);

  // ── K-batch unroll sequence ──────────────────────────────────────────
  for (int64_t kb = 0; kb < kBatchCount; ++kb) {
    SmallVector<Value> lhsOff = off0;
    SmallVector<Value> rhsOff = off1;

    if (kb > 0 && lhsKAxis != -1 && rhsKAxis != -1) {
      // Advance each operand's K offset independently using its own kStep.
      // After Bug 1 is fixed lhsKStep == rhsKStep, but we keep them separate
      // so a future layout mismatch produces an error above, not silent wrong
      // results here.
      Value lhsAdvance =
          rewriter.create<arith::ConstantIndexOp>(loc, kb * lhsKStep);
      Value rhsAdvance =
          rewriter.create<arith::ConstantIndexOp>(loc, kb * rhsKStep);
      lhsOff[lhsKAxis] =
          rewriter.create<arith::AddIOp>(loc, off0[lhsKAxis], lhsAdvance);
      rhsOff[rhsKAxis] =
          rewriter.create<arith::AddIOp>(loc, off1[rhsKAxis], rhsAdvance);
    }

    Value lhsSlice = distributeTransferRead(rewriter, lhsRead, lhsOff, lhsThread);
    Value rhsSlice = distributeTransferRead(rewriter, rhsRead, rhsOff, rhsThread);

    auto partialContract = rewriter.create<vector::ContractionOp>(
        loc, lhsSlice, rhsSlice, acc, contractOp.getIndexingMaps(),
        contractOp.getIteratorTypes());

    // Preserve lowering_config for downstream passes (UnrollToIntrinsics).
    if (auto cfg = contractOp->getAttr("lowering_config"))
      partialContract->setAttr("lowering_config", cfg);

    acc = partialContract.getResult();
  }

  // ── Distribute the result transfer_write ─────────────────────────────
  // The final fully-reduced ACC result feeds directly into a transfer_write.
  Value contractResult = contractOp.getResult();
  vector::TransferWriteOp resultWrite;
  for (Operation *user : contractResult.getUsers()) {
    if (auto w = dyn_cast<vector::TransferWriteOp>(user)) {
      resultWrite = w;
      break;
    }
  }

  if (resultWrite) {
    rewriter.setInsertionPoint(resultWrite);
    distributeTransferWrite(rewriter, resultWrite, acc, off2);
    rewriter.eraseOp(resultWrite);
  }

  // Erase the original contract (reads are already replaced/erased by unroll).
  rewriter.eraseOp(contractOp);
  if (lhsRead->use_empty()) rewriter.eraseOp(lhsRead);
  if (rhsRead->use_empty()) rewriter.eraseOp(rhsRead);
  if (accRead->use_empty()) rewriter.eraseOp(accRead);

  return success();
}

//===----------------------------------------------------------------------===//
// §7  Pass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorDistributePass
    : public PassWrapper<NovaGPUVectorDistributePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorDistributePass)

  StringRef getArgument() const override {
    return "nova-gpu-vector-distribute";
  }
  StringRef getDescription() const override {
    return "Distribute vector.contract and surrounding transfer ops to "
           "per-thread slices using nova.layout_* DictionaryAttrs.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect,
                    vector::VectorDialect, affine::AffineDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // ── Collect all contracts with nova.layout_0 ───────────────────────
    // Collect upfront — modifying the IR during walk invalidates the iterator.
    SmallVector<vector::ContractionOp> contracts;
    funcOp.walk([&](vector::ContractionOp op) {
      if (op->hasAttr("nova.layout_0"))
        contracts.push_back(op);
    });

    // ── Distribute each contract ────────────────────────────────────────
    // gpu.thread_id is created inside distributeContractOp, right before the
    // contract op itself — which is inside the thread-mapped scf.forall.
    // That ensures thread_id ends up inside the gpu.launch block after
    // ConvertForallToGPU, rather than floating at function scope.
    for (vector::ContractionOp contractOp : contracts) {
      if (failed(distributeContractOp(rewriter, contractOp))) {
        signalPassFailure();
        return;
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-vector-distribute] distributed "
               << contracts.size() << " contract(s)\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUVectorDistributePass() {
  return std::make_unique<NovaGPUVectorDistributePass>();
}

void registerNovaGPUVectorDistributePass() {
  PassRegistration<NovaGPUVectorDistributePass>();
}

} // namespace mlir::nova