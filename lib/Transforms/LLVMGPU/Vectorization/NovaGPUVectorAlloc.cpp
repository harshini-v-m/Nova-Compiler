//===-- NovaGPUVectorAlloc.cpp -------------------------------------------===//
//
// Stages vector.contract operands marked shared_mem=true through workgroup
// shared memory.
//
// KEY FIX over original:
//   For operands already in smem (detected by isFromWorkgroupPromotionForall
//   or explicit workgroup address space), we emit ONLY a nova.value_barrier
//   and re-read.  No new alloc, no write, no intermediate bounce tensor.
//   This eliminates the redundant alloc+write+read cycle that doubled smem
//   usage and produced dead self-copies after bufferization.
//
//   For the ACC operand (index 2 of vector.contraction): skip entirely.
//   VectorDistribute reads ACC directly from the global output subview.
//   Staging ACC through smem loses the writeback (the smem tensor is never
//   copied back to global) and doubles register pressure for no benefit.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::nova {
namespace {

//===----------------------------------------------------------------------===//
// §1  Attribute helpers (unchanged)
//===----------------------------------------------------------------------===//

static bool getSharedMem(DictionaryAttr d) {
  if (!d) return false;
  auto a = d.getAs<BoolAttr>("shared_mem");
  return a && a.getValue();
}

static DictionaryAttr clearSharedMem(MLIRContext *ctx, DictionaryAttr d) {
  SmallVector<NamedAttribute> fields(d.begin(), d.end());
  for (auto &f : fields)
    if (f.getName().getValue() == "shared_mem") {
      f = NamedAttribute(f.getName(), BoolAttr::get(ctx, false));
      break;
    }
  return DictionaryAttr::get(ctx, fields);
}

static SmallVector<int64_t> getIntArray(DictionaryAttr d, StringRef k) {
  SmallVector<int64_t> r;
  if (!d) return r;
  auto a = d.getAs<ArrayAttr>(k);
  if (!a) return r;
  for (Attribute x : a)
    if (auto ia = dyn_cast<IntegerAttr>(x)) r.push_back(ia.getInt());
  return r;
}

//===----------------------------------------------------------------------===//
// §2  Forall helpers (unchanged)
//===----------------------------------------------------------------------===//

static bool isThreadMappedForall(scf::ForallOp f) {
  auto m = f.getMappingAttr();
  if (!m || m.getValue().empty()) return false;
  for (Attribute a : m.getValue())
    if (!isa<gpu::GPUThreadMappingAttr>(a)) return false;
  return true;
}

static bool isBlockMappedForall(scf::ForallOp f) {
  auto m = f.getMappingAttr();
  if (!m || m.getValue().empty()) return false;
  for (Attribute a : m.getValue())
    if (!isa<gpu::GPUBlockMappingAttr>(a)) return false;
  return true;
}

static scf::ForallOp findEnclosingSubgroupForall(Operation *op) {
  Operation *parent = op->getParentOp();
  scf::ForallOp innermostBlock = nullptr;
  while (parent) {
    if (auto f = dyn_cast<scf::ForallOp>(parent)) {
      if (isThreadMappedForall(f)) return f;
      if (isBlockMappedForall(f) && !innermostBlock) innermostBlock = f;
    }
    parent = parent->getParentOp();
  }
  return innermostBlock;
}

//===----------------------------------------------------------------------===//
// §3  Smem source detection
//===----------------------------------------------------------------------===//

static bool isFromWorkgroupPromotionForall(Value val) {
  while (true) {
    if (auto e = val.getDefiningOp<tensor::ExtractSliceOp>())
      val = e.getSource();
    else if (auto b = val.getDefiningOp<nova::ValueBarrierOp>())
      val = b.getOperand(0);
    else break;
  }
  auto fa = val.getDefiningOp<scf::ForallOp>();
  if (!fa) return false;

  // Case 1: forall body contains nova.promote_to_workgroup (original check)
  bool found = false;
  fa->walk([&](Operation *op) {
    if (op->hasAttr("nova.promote_to_workgroup")) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (found) return true;

  if (isThreadMappedForall(fa)) {
    Operation *parent = fa->getParentOp();
    while (parent) {
      if (auto pf = dyn_cast<scf::ForallOp>(parent)) {
        if (isBlockMappedForall(pf))
          return true;
      }
      parent = parent->getParentOp();
    }
  }

  return false;
}

static bool isAlreadyInSmem(Value operand, Attribute smemSpace) {
  auto srcRead = operand.getDefiningOp<vector::TransferReadOp>();
  if (!srcRead) return false;
  Value base = srcRead.getBase();
  if (auto tt = dyn_cast<RankedTensorType>(base.getType()))
    if (tt.getEncoding() == smemSpace) return true;
  return isFromWorkgroupPromotionForall(base);
}

//===----------------------------------------------------------------------===//
// §4  Stage one operand through shared memory
//
// Three cases:
//   A) ACC operand (idx == 2 for vector.contraction): skip entirely.
//      VectorDistribute reads it directly from global; staging loses writeback.
//
//   B) Operand already in smem: emit only nova.value_barrier + re-read.
//      No alloc, no write.  Just sync and re-read from existing smem tensor.
//
//   C) Normal operand not yet in smem: alloc WG tensor → write → barrier
//      → re-read.  Same as original PATH A but without the redundant
//      self-copy that ComprehensiveBufferize was generating.
//===----------------------------------------------------------------------===//

static LogicalResult stageOperand(MLIRContext *ctx, Operation *op,
                                   unsigned idx, DictionaryAttr layoutAttr,
                                   scf::ForallOp sgForall,
                                   OpBuilder &wgAllocBuilder,
                                   llvm::SmallPtrSet<Block *, 4> &barrierBlocks) {
  Location loc = op->getLoc();
  Value operand = op->getOperand(idx);

  // ── Case A: ACC operand — skip, clear shared_mem flag ───────────────────
  if (isa<vector::ContractionOp>(op) && idx == 2u) {
    op->setAttr("nova.layout_" + std::to_string(idx),
                clearSharedMem(ctx, layoutAttr));
    return success();
  }

  auto vecType = dyn_cast<VectorType>(operand.getType());
  if (!vecType || vecType.isScalable()) {
    op->emitError("nova-gpu-vector-alloc: operand ")
        << idx << " is not a static vector";
    return failure();
  }

  int rank = vecType.getRank();
  Attribute smemSpace = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

  auto srcReadOp = operand.getDefiningOp<vector::TransferReadOp>();
  bool alreadySmem = isAlreadyInSmem(operand, smemSpace);

  // Insert gpu.barrier at start of sgForall body (once per block).
  Block *sgBody = sgForall.getBody();
  if (barrierBlocks.insert(sgBody).second) {
    OpBuilder bb(sgBody, sgBody->begin());
    gpu::BarrierOp::create(bb, loc);
  }

  OpBuilder b(op);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> zeros(rank, c0);
  SmallVector<bool> inBounds(rank, true);

  Value newVec;

  if (alreadySmem) {
    // ── Case B: already in smem ──────────────────────────────────────────
    // Emit nova.value_barrier on the existing smem source tensor.
    // Re-read from it using the ORIGINAL indices (VectorDistribute will
    // further slice according to layout attributes).
    // Do NOT create any new alloc or write — that is the key fix.
    Value smemBase = srcReadOp.getBase();

    // Only emit barrier if not already guarded by one.
    Value synced = smemBase;
    if (!smemBase.getDefiningOp<nova::ValueBarrierOp>()) {
      auto barrier = nova::ValueBarrierOp::create(b, loc, ValueRange{smemBase});
      synced = barrier.getResults()[0];
    }

    // Re-read with original indices — preserves indexing for VectorDistribute.
    SmallVector<Value> origIdx(srcReadOp.getIndices().begin(),
                               srcReadOp.getIndices().end());
    newVec = vector::TransferReadOp::create(
                 b, loc, vecType, synced, origIdx,
                 /*padding=*/std::nullopt, inBounds)
                 .getResult();

    // Erase original srcReadOp if now dead.
    if (srcReadOp && operand.use_empty())
      srcReadOp->erase();

  } else {
    // ── Case C: normal smem staging ─────────────────────────────────────
    // Layout-derived workgroup shape.
    auto threadCounts = getIntArray(layoutAttr, "thread_counts");
    auto batchCounts  = getIntArray(layoutAttr, "batch_counts");
    auto elemCounts   = getIntArray(layoutAttr, "elem_counts");
    auto sgCounts     = getIntArray(layoutAttr, "sg_counts");
    auto sgStrides    = getIntArray(layoutAttr, "sg_strides");

    bool hasFields = (int)threadCounts.size() == rank &&
                     (int)batchCounts.size()  == rank &&
                     (int)elemCounts.size()   == rank;

    SmallVector<int64_t> sgShape(rank), wgShape(rank);
    for (int d = 0; d < rank; ++d) {
      int64_t perSg = hasFields
          ? threadCounts[d] * batchCounts[d] * elemCounts[d]
          : vecType.getShape()[d];
      sgShape[d] = perSg;
      wgShape[d] = perSg * std::max<int64_t>(
          ((int)sgCounts.size() > d) ? sgCounts[d] : 1, 1);
    }

    // Allocate WG tensor outside sgForall.
    auto wgTy = RankedTensorType::get(wgShape, vecType.getElementType(),
                                       smemSpace);
    auto wgAlloc = bufferization::AllocTensorOp::create(
        wgAllocBuilder, loc, wgTy, ValueRange{}, Value());
    wgAlloc.setMemorySpaceAttr(smemSpace);
    Value wgTensor = wgAlloc.getResult();

    // Per-sg subview offsets.
    auto ivs = llvm::to_vector(sgForall.getInductionVars());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(rank), strides(rank, b.getIndexAttr(1));
    for (int d = 0; d < rank; ++d) {
      sizes[d] = b.getIndexAttr(sgShape[d]);
      bool isSgDim = (int)sgStrides.size() > d && sgStrides[d] != 0;
      if (isSgDim) {
        // Find which IV drives this dimension.
        int ivIdx = 0;
        for (int dd = 0; dd < d; ++dd)
          if ((int)sgStrides.size() > dd && sgStrides[dd] != 0) ++ivIdx;
        if (ivIdx < (int)ivs.size()) {
          Value stride = b.create<arith::ConstantIndexOp>(loc, sgShape[d]);
          offsets[d] = b.create<arith::MulIOp>(loc, ivs[ivIdx], stride)
                           .getResult();
        }
      }
    }

    // Write operand into per-sg slice.
    auto sgTy = RankedTensorType::get(sgShape, vecType.getElementType(),
                                       smemSpace);
    Value writeSlice = tensor::ExtractSliceOp::create(
        b, loc, sgTy, wgTensor, offsets, sizes, strides);
    Value written = vector::TransferWriteOp::create(
                        b, loc, operand, writeSlice, zeros, inBounds)
                        .getResult();

    // Barrier — sync all threads before any read.
    auto barrier = nova::ValueBarrierOp::create(b, loc, ValueRange{written});
    Value synced = barrier.getResults()[0];

    // Re-read the vector from the synced smem slice.
    newVec = vector::TransferReadOp::create(
                 b, loc, vecType, synced, zeros,
                 /*padding=*/std::nullopt, inBounds)
                 .getResult();

    // Sink srcReadOp into the forall body to eliminate cross-boundary
    // live range (avoids register spill of the full pre-smem vector).
    if (srcReadOp && operand.hasOneUse()) {
      Operation *writeOp = *operand.user_begin();
      OpBuilder sinkB(writeOp);
      Operation *cloned = sinkB.clone(*srcReadOp);
      writeOp->replaceUsesOfWith(operand, cloned->getResult(0));
      srcReadOp->erase();
    }
  }

  // Rewire operand and clear shared_mem flag.
  op->setOperand(idx, newVec);
  op->setAttr("nova.layout_" + std::to_string(idx),
              clearSharedMem(ctx, layoutAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// §5  Per-operation dispatch
//===----------------------------------------------------------------------===//

static LogicalResult processOp(MLIRContext *ctx, Operation *op,
                                llvm::SmallPtrSet<Block *, 4> &barrierBlocks) {
  bool anySmem = false;
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto attr = op->getAttrOfType<DictionaryAttr>(
        "nova.layout_" + std::to_string(i));
    if (attr && getSharedMem(attr)) { anySmem = true; break; }
  }
  if (!anySmem) return success();

  scf::ForallOp sgForall = findEnclosingSubgroupForall(op);
  if (!sgForall) {
    op->emitError("nova-gpu-vector-alloc: no enclosing gpu-mapped forall");
    return failure();
  }

  OpBuilder wgAllocBuilder = isBlockMappedForall(sgForall)
      ? OpBuilder(op) : OpBuilder(sgForall);

  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto attr = op->getAttrOfType<DictionaryAttr>(
        "nova.layout_" + std::to_string(i));
    if (attr && getSharedMem(attr))
      if (failed(stageOperand(ctx, op, i, attr, sgForall,
                              wgAllocBuilder, barrierBlocks)))
        return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// §5b  Staticize dynamic tensor.empty ops inside promotion copy foralls
//
// NovaGenericVectorization uses masked vectorization (Case C) for dynamic-shape
// linalg ops inside the B-tile promotion forall. This emits:
//
//   %dyn = tensor.empty(%n) : tensor<1x1x?xf32>    // dynamic staging buffer
//   vector.transfer_write %v, %dyn[...], mask       // write with mask
//   vector.transfer_read  %dyn[...], 0.0, mask      // read back immediately
//   vector.transfer_write %result, %static[...]     // write to static tile
//
// %dyn has a runtime-valued size and is never assigned a GPU address space by
// NovaGPUInferMemorySpacePass (it is not a shared_out of any thread-forall).
// When EmptyTensorToAllocTensorPass converts it, the dynamic alloc_tensor with
// no memory_space annotation is lowered by comprehensive-bufferize to malloc().
//
// Fix: for every tensor.empty with any dynamic dimension inside an
// scf.forall that contains a nova.promote_to_workgroup op, replace it with a
// tensor.empty using the static upper-bound shape inferred from the element
// type and the surrounding static tile sizes. The existing vector.create_mask
// on the read/write pair already handles the tail correctly — we just need the
// backing tensor to be statically shaped so it can live in registers.
//
// The upper-bound shape for each dimension is the product of thread_counts and
// elem_counts from the enclosing workgroup tile, but in practice these staging
// tensors are always small: the B-tile copy uses tensor<1x1x?xf32> where ? ≤ 4
// (two chunks of 4 elements each), so we use 4 as the upper bound.  We derive
// the static size by finding the corresponding static dim from the op's output
// operand chain (the extract_slice target), which is always static after
// padding.
//===----------------------------------------------------------------------===//

// Returns true if op is nested inside a forall that contains a
// nova.promote_to_workgroup copy (i.e., inside a B-tile promotion forall).
static bool isInsidePromotionForall(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forall = dyn_cast<scf::ForallOp>(parent)) {
      bool hasPromotion = false;
      forall->walk([&](Operation *inner) {
        if (inner->hasAttr("nova.promote_to_workgroup")) {
          hasPromotion = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasPromotion) return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

// Walk the function and replace dynamic tensor.empty ops inside promotion
// foralls with static upper-bound equivalents.
static void staticizeDynamicStagingBuffers(func::FuncOp funcOp) {
  SmallVector<tensor::EmptyOp> toFix;
  funcOp.walk([&](tensor::EmptyOp empty) {
    auto ty = empty.getType();
    if (ty.hasStaticShape()) return;
    if (isInsidePromotionForall(empty)) toFix.push_back(empty);
  });

  for (tensor::EmptyOp empty : toFix) {
    auto ty = empty.getType();
    // Find the static upper bound for each dynamic dimension.
    // Strategy: for each dynamic dim, look at the users of this empty —
    // specifically the vector.transfer_write that writes into it.
    // The vector type written is always static (vectorization uses static
    // upper-bound vector sizes), so use that shape.
    SmallVector<int64_t> staticShape(ty.getShape());
    for (Operation *user : empty->getUsers()) {
      auto writeOp = dyn_cast<vector::TransferWriteOp>(user);
      if (!writeOp) continue;
      auto vecTy = dyn_cast<VectorType>(writeOp.getVector().getType());
      if (!vecTy || vecTy.isScalable()) continue;
      // The vector type is the static upper-bound shape of this tensor.
      for (int d = 0; d < (int)staticShape.size(); ++d) {
        if (staticShape[d] == ShapedType::kDynamic)
          staticShape[d] = vecTy.getShape()[d];
      }
      break;
    }
    // If we couldn't resolve all dynamic dims, skip (conservative).
    if (llvm::any_of(staticShape,
                     [](int64_t d) { return d == ShapedType::kDynamic; }))
      continue;

    OpBuilder b(empty);
    auto staticTy = RankedTensorType::get(staticShape, ty.getElementType());
    auto staticEmpty = b.create<tensor::EmptyOp>(empty.getLoc(), staticTy,
                                                 ValueRange{});
    // Replace all uses: cast back to original dynamic type for users that
    // still expect it (transfer_write with dynamic-shaped tensor operand).
    // Since the dynamic dim is now statically bounded, tensor.cast is valid
    // (dynamic → static is a refinement, always safe).
    Value cast = b.create<tensor::CastOp>(empty.getLoc(), ty, staticEmpty);
    empty.replaceAllUsesWith(cast);
    empty.erase();
  }
}

//===----------------------------------------------------------------------===//
// §6  Pass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorAllocPass
    : public PassWrapper<NovaGPUVectorAllocPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorAllocPass)

  StringRef getArgument()    const override { return "nova-gpu-vector-alloc"; }
  StringRef getDescription() const override {
    return "Stage vector.contract smem operands; skip ACC and already-smem ops."
           " Also staticizes dynamic tensor.empty staging buffers inside "
           "promotion foralls to prevent malloc in GPU kernels.";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
             vector::VectorDialect, nova::NovaDialect, arith::ArithDialect,
             scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    // ── Step 1: Staticize dynamic staging buffers inside promotion foralls ──
    // Must run BEFORE the vector.contract staging walk so that the
    // tensor.empty → static replacement is in place before bufferization.
    staticizeDynamicStagingBuffers(fn);

    // ── Step 2: Stage vector.contract smem operands ─────────────────────────
    llvm::SmallPtrSet<Block *, 4> barrierBlocks;
    fn.walk([&](Operation *op) {
      if (failed(processOp(&getContext(), op, barrierBlocks)))
        signalPassFailure();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createNovaGPUVectorAllocPass() {
  return std::make_unique<NovaGPUVectorAllocPass>();
}
void registerNovaGPUVectorAllocPass() {
  PassRegistration<NovaGPUVectorAllocPass>();
}

} // namespace mlir::nova