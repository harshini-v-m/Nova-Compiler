//===- NovaGPUCreateAsyncCopies.cpp - gmem→smem pairs to nvgpu.cp.async --===//
//
// Port of IREE's `createAsyncGroups` (compiler/src/iree/compiler/Codegen/
// LLVMGPU/Utils/LLVMGPUUtils.cpp:199–331), adapted to the IR shape Nova
// produces after `NovaGPUMultiBufferingPass`.
//
// Purpose
// ───────
// Rewrite gmem→smem `vector.transfer_read`/`vector.transfer_write` pairs into
// `nvgpu.device_async_copy` + `nvgpu.device_async_create_group` +
// `nvgpu.device_async_wait(0)`. The wait(0) preserves serial semantics; the
// downstream pipelining pass (`NovaGPUPipeliningPass`) rewrites the
// `numGroups` attribute to `depth-1` to actually overlap.
//
// Pre-condition (post-NovaGPUMultiBuffering, pre-MapForallToGPU)
// ──────────────────────────────────────────────────────────────
// The K-loop body contains, for each tile (A and B):
//
//   scf.forall (thread) {
//     %v = vector.transfer_read %gmemSubview[...], %pad
//                {in_bounds=[true,true]} : memref<...,1>, vector<MRxNRxT>
//     vector.transfer_write %v, %smemStageSubview[...]
//                {in_bounds=[true,true]} : vector<MRxNRxT>, memref<...,#ws>
//     [vector.transfer_read/write removed by earlier pass]
//   }
//
// Post-condition (per matched pair)
// ─────────────────────────────────
// The transfer_read+transfer_write are erased; in their place:
//
//     %t0 = nvgpu.device_async_copy %smem[i+0,j], %gmem[i+0,j], NR
//     %t1 = nvgpu.device_async_copy %smem[i+1,j], %gmem[i+1,j], NR
//     ...
//     %g  = nvgpu.device_async_create_group %t0, %t1, ...
//     nvgpu.device_async_wait %g                        // numGroups=nullptr → wait-all
//
// The 2-D pair is decomposed into MR row-wise async copies; the inner-dim
// vector size NR drives the cp.async granularity (NR*sizeof(T) ∈ {4,8,16}).
//
// Notes
// ─────
//   * `bypassL1` (cp.async.cg, the cache-bypass form) is intentionally NOT
//     set. matmul.cpp:450 documents that cp.async.cg caused memory
//     corruption in this tree; cp.async.ca is the safe default.
//   * 1-D vectors (rank=1) are handled as a single async copy.
//   * Pairs that don't satisfy alignment / in-bounds requirements are
//     skipped — the original vector.transfer pair is left intact, which is
//     functionally correct (just synchronous).
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-create-async-copies"

using namespace mlir;

namespace mlir::nova {

namespace {

// True if the memref lives in GPU workgroup (shared) memory.
static bool hasSharedMemoryAddressSpace(MemRefType t) {
  auto attr = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return attr &&
         attr.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

// True if the memref is *not* in workgroup memory (i.e. global / generic /
// numbered space). cp.async loads from global; we treat anything outside
// workgroup as a valid source.
static bool isNonSharedMemoryAddressSpace(MemRefType t) {
  return !hasSharedMemoryAddressSpace(t);
}

// Mirrors IREE's resultsInSupportedAsyncCopy alignment check (LLVMGPUUtils.cpp
// :163). For a 1-D copy of `numElements` elements of `elemType`, the byte
// width must be one of {4, 8, 16}.
static bool isSupportedCpAsyncSize(int64_t numElements, Type elemType) {
  if (!elemType.isIntOrFloat())
    return false;
  int64_t bits = elemType.getIntOrFloatBitWidth();
  if ((bits % 8) != 0)
    return false;
  int64_t bytes = (numElements * bits) / 8;
  return bytes == 4 || bytes == 8 || bytes == 16;
}

// Returns the inner-most (fastest-varying) dim size of a vector type.
// For vector<8x4xf32> → 4. For vector<4xf32> → 4.
static int64_t innerDimSize(VectorType vt) {
  return vt.getShape().back();
}

// Returns the leading-dim product (everything except inner-most). For
// vector<8x4xf32> → 8. For vector<4xf32> → 1. For vector<2x4x4xf32> → 8.
static int64_t leadingDimProduct(VectorType vt) {
  int64_t p = 1;
  for (int64_t i = 0, e = vt.getRank() - 1; i < e; ++i)
    p *= vt.getShape()[i];
  return p;
}

// Decompose a leading linear index into per-axis indices for a shape.
// Example: shape=[2,4], linear=5 → {1, 1}.
static SmallVector<int64_t> delinearize(int64_t linear,
                                        ArrayRef<int64_t> shape) {
  SmallVector<int64_t> out(shape.size(), 0);
  for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
    out[i] = linear % shape[i];
    linear /= shape[i];
  }
  return out;
}

// Returns true if `tw` is a candidate for cp.async conversion: defined by a
// matching vector.transfer_read whose source is non-workgroup memory.
// Populates `tr` on success.
static bool matchTransferPair(vector::TransferWriteOp tw,
                              vector::TransferReadOp &tr) {
  // Destination must be workgroup memory.
  auto dstTy = dyn_cast<MemRefType>(tw.getBase().getType());
  if (!dstTy || !hasSharedMemoryAddressSpace(dstTy))
    return false;

  // Must be in-bounds on every dim.
  auto inBounds = tw.getInBounds();
  for (Attribute a : inBounds) {
    auto b = dyn_cast<BoolAttr>(a);
    if (!b || !b.getValue())
      return false;
  }
  if (tw.getMask())
    return false;

  // Permutation map must be minor identity (no transpose / broadcast).
  if (!tw.getPermutationMap().isMinorIdentity())
    return false;

  // The stored value must be the result of a vector.transfer_read.
  Value stored = tw.getValue();
  tr = stored.getDefiningOp<vector::TransferReadOp>();
  if (!tr)
    return false;

  // Source must be non-workgroup memory (gmem or generic).
  auto srcTy = dyn_cast<MemRefType>(tr.getBase().getType());
  if (!srcTy || !isNonSharedMemoryAddressSpace(srcTy))
    return false;

  // Reader must be in-bounds, no mask, minor identity.
  for (Attribute a : tr.getInBounds()) {
    auto b = dyn_cast<BoolAttr>(a);
    if (!b || !b.getValue())
      return false;
  }
  if (tr.getMask())
    return false;
  if (!tr.getPermutationMap().isMinorIdentity())
    return false;

  // Vector type must match between read and write.
  auto vt = dyn_cast<VectorType>(stored.getType());
  if (!vt)
    return false;

  // Inner-dim copy size must be a supported cp.async granularity.
  if (!isSupportedCpAsyncSize(innerDimSize(vt), vt.getElementType()))
    return false;

  return true;
}

// Builds an `arith::AddIOp` of `base + cst`. Folds when `cst == 0`.
static Value addConst(OpBuilder &b, Location loc, Value base, int64_t cst) {
  if (cst == 0)
    return base;
  Value c = b.create<arith::ConstantIndexOp>(loc, cst);
  return b.create<arith::AddIOp>(loc, base, c);
}

// Convert one transfer_read/transfer_write pair into a sequence of
// nvgpu.device_async_copy ops (one per leading-dim element). Appends the
// emitted tokens to `tokens`. Returns success iff conversion was performed.
static LogicalResult convertPairToAsync(vector::TransferWriteOp tw,
                                        vector::TransferReadOp tr,
                                        SmallVectorImpl<Value> &tokens,
                                        OpBuilder &b) {
  Location loc = tw.getLoc();
  auto vt = cast<VectorType>(tr.getType());
  int64_t innerN = innerDimSize(vt);
  ArrayRef<int64_t> leadShape = vt.getShape().drop_back();
  int64_t numCopies = leadingDimProduct(vt);

  // Operand bookkeeping.
  Value srcBase = tr.getBase();
  Value dstBase = tw.getBase();
  ValueRange srcIdx = tr.getIndices();
  ValueRange dstIdx = tw.getIndices();

  // Ranks must match indexing.
  if ((int64_t)srcIdx.size() != vt.getRank())
    return failure();
  if ((int64_t)dstIdx.size() != vt.getRank())
    return failure();

  auto tokenTy = nvgpu::DeviceAsyncTokenType::get(b.getContext());

  // Insertion right at the transfer_write so the cp.async lands in the
  // same block / region.
  b.setInsertionPoint(tw);

  for (int64_t k = 0; k < numCopies; ++k) {
    SmallVector<int64_t> deltas =
        leadShape.empty() ? SmallVector<int64_t>{} : delinearize(k, leadShape);

    // Per-row indices: leading dims get +delta[i]; inner dim unchanged.
    SmallVector<Value> rowSrc, rowDst;
    rowSrc.reserve(vt.getRank());
    rowDst.reserve(vt.getRank());
    for (int64_t i = 0, e = vt.getRank() - 1; i < e; ++i) {
      rowSrc.push_back(addConst(b, loc, srcIdx[i], deltas[i]));
      rowDst.push_back(addConst(b, loc, dstIdx[i], deltas[i]));
    }
    rowSrc.push_back(srcIdx.back());
    rowDst.push_back(dstIdx.back());

    auto copyOp = b.create<nvgpu::DeviceAsyncCopyOp>(
        loc, tokenTy,
        /*dst=*/dstBase, /*dstIndices=*/rowDst,
        /*src=*/srcBase, /*srcIndices=*/rowSrc,
        /*dstElements=*/b.getIndexAttr(innerN),
        /*srcElements=*/Value{},
        /*bypassL1=*/UnitAttr{});
    tokens.push_back(copyOp.getResult());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NovaGPUCreateAsyncCopiesPass
//===----------------------------------------------------------------------===//

struct NovaGPUCreateAsyncCopiesPass
    : public PassWrapper<NovaGPUCreateAsyncCopiesPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUCreateAsyncCopiesPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    nvgpu::NVGPUDialect, scf::SCFDialect,
                    vector::VectorDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.getFunctionBody().empty())
      return;

    // ── Step 1: collect candidate writes in program order ──────────────────
    SmallVector<vector::TransferWriteOp> candidates;
    funcOp.walk([&](vector::TransferWriteOp tw) {
      vector::TransferReadOp tr;
      if (matchTransferPair(tw, tr))
        candidates.push_back(tw);
    });
    if (candidates.empty())
      return;

    // ── Step 2: bucket consecutive candidates that share a parent block ────
    // Each bucket becomes one nvgpu.device_async_create_group + wait-all.
    OpBuilder builder(&getContext());
    unsigned numGroups = 0;
    unsigned numCopies = 0;

    size_t i = 0;
    while (i < candidates.size()) {
      scf::ForOp loop = candidates[i]->getParentOfType<scf::ForOp>();

      size_t j = i + 1;
      while (j < candidates.size() && candidates[j]->getParentOfType<scf::ForOp>() == loop)
        ++j;

      SmallVector<Value> tokens;
      SmallVector<vector::TransferWriteOp> erasedWrites;
      SmallVector<vector::TransferReadOp> erasedReads;
      auto tokenTy = nvgpu::DeviceAsyncTokenType::get(&getContext());

      for (size_t k = i; k < j; ++k) {
        vector::TransferWriteOp tw = candidates[k];
        vector::TransferReadOp tr;
        if (!matchTransferPair(tw, tr))
          continue;

        SmallVector<Value> pairTokens;
        if (failed(convertPairToAsync(tw, tr, pairTokens, builder)))
          continue;

        erasedWrites.push_back(tw);
        erasedReads.push_back(tr);
        tokens.append(pairTokens);
      }

      if (!tokens.empty()) {
        Operation *lastOp = erasedWrites.back();
        if (auto ifOp = lastOp->getParentOfType<scf::IfOp>())
          lastOp = ifOp;

        builder.setInsertionPointAfter(lastOp);  // Bug #2 fix

        auto grp = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(
          candidates[i].getLoc(), tokenTy, ValueRange{});

        // Tag the commit as Stage 0 so it moves to prologue with copies.
        grp->setAttr("__pipelining_first_stage__", builder.getUnitAttr());

        builder.create<nvgpu::DeviceAsyncWaitOp>(
            candidates[i].getLoc(), grp.getResult(), /*numGroups=*/nullptr);
        builder.create<gpu::BarrierOp>(candidates[i].getLoc());
        
        numGroups++;
        numCopies += tokens.size();
      }

      for (auto tw : erasedWrites) tw.erase();
      for (auto tr : erasedReads) if (tr.use_empty()) tr.erase();

      i = j;
    }

    LLVM_DEBUG(llvm::dbgs() << "[nova-create-async] emitted " << numCopies
                            << " cp.async ops in " << numGroups << " group(s)\n");
  }

  StringRef getArgument() const override {
    return "nova-gpu-create-async-copies";
  }
  StringRef getDescription() const override {
    return "Rewrite gmem→smem vector.transfer_read/write pairs into "
           "nvgpu.device_async_copy + create_group + wait(0). Sets up the "
           "IR for the software-pipelining pass to overlap.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUCreateAsyncCopiesPass() {
  return std::make_unique<NovaGPUCreateAsyncCopiesPass>();
}

void registerNovaGPUCreateAsyncCopiesPass() {
  PassRegistration<NovaGPUCreateAsyncCopiesPass>();
}

} // namespace mlir::nova
