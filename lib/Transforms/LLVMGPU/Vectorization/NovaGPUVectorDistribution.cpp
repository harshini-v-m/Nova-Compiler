//===-- NovaGPUVectorDistribute.cpp ---------------------------------------===//
//
// Distributes vector.contract (backed by nvgpu.mma.sync) to per-warp slices.
// For mma_kind != 0 the pass emits nvgpu.mma.sync directly on warp-owned
// fragments WITHOUT any per-thread index arithmetic.  For generic
// vector.contract (mma_kind == 0) it falls back to per-thread slicing via
// the layout attributes.
//
// Key invariant: nvgpu.mma.sync is warp-synchronous.  All 32 threads in a
// warp participate collectively; the hardware maps lane IDs to fragments
// internally.  Emitting per-thread index arithmetic (divui/remui chains) on
// top of mma.sync is both wrong and unnecessary.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "nova-gpu-vector-distribute"

using namespace mlir;

namespace mlir::nova {
namespace {

//===----------------------------------------------------------------------===//
// §1  Layout descriptor
//===----------------------------------------------------------------------===//

struct OperandLayout {
  SmallVector<int64_t> sgCounts;
  SmallVector<int64_t> sgStrides;
  SmallVector<int64_t> batchCounts;
  SmallVector<int64_t> threadCounts;
  SmallVector<int64_t> threadStrides;
  SmallVector<int64_t> elemCounts;
  bool sharedMem = false;
  int rank = 0;
};

static SmallVector<int64_t> getI64Array(DictionaryAttr d, StringRef k) {
  auto a = d.getAs<ArrayAttr>(k);
  if (!a) return {};
  SmallVector<int64_t> r;
  for (Attribute x : a)
    r.push_back(cast<IntegerAttr>(x).getInt());
  return r;
}

static FailureOr<OperandLayout> readLayout(Operation *op, int idx) {
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
  if (auto b = attr.getAs<BoolAttr>("shared_mem"))
    L.sharedMem = b.getValue();
  L.rank = (int)L.elemCounts.size();
  if (L.rank == 0 ||
      (int)L.sgCounts.size()     != L.rank ||
      (int)L.threadCounts.size() != L.rank ||
      (int)L.batchCounts.size()  != L.rank)
    return failure();
  return L;
}

//===----------------------------------------------------------------------===//
// §2  MMA fragment shape derivation (generic, no op-specific knowledge)
//
// For mmaShape = [M, N, K] and element type of width W bits:
//   shapeK   = 128 / W        (elements per 128-bit K-tile)
//   numElemA = 32  / W        (A elements per thread)
//   numElemB = 32  / W        (B elements per thread)
//   numElemC = 2              (C elements per thread, always)
//   mTile    = M / 8
//   nTile    = N / 8
//   kTile    = K / shapeK
//   aFrag    = [mTile * kTile, numElemA]
//   bFrag    = [kTile * nTile, numElemB]
//   cFrag    = [mTile * nTile, numElemC]
//===----------------------------------------------------------------------===//

struct MMAFragmentShapes {
  SmallVector<int64_t> a; // 2D
  SmallVector<int64_t> b; // 2D
  SmallVector<int64_t> c; // 2D
  SmallVector<int64_t> mmaShape; // [M, N, K]
};

static FailureOr<MMAFragmentShapes>
deriveMMAFragments(ArrayRef<int64_t> mmaShape, Type elemType) {
  if (mmaShape.size() < 3) return failure();
  int64_t M = mmaShape[0], N = mmaShape[1], K = mmaShape[2];
  int64_t W       = elemType.getIntOrFloatBitWidth();
  int64_t shapeK  = 128 / W;
  int64_t numA    = 32 / W;
  int64_t numB    = 32 / W;
  int64_t numC    = 2;
  int64_t mTile   = M / 8;
  int64_t nTile   = N / 8;
  int64_t kTile   = K / shapeK;
  MMAFragmentShapes s;
  s.a        = {mTile * kTile, numA};
  s.b        = {kTile * nTile, numB};
  s.c        = {mTile * nTile, numC};
  s.mmaShape = SmallVector<int64_t>(mmaShape.begin(), mmaShape.end());
  return s;
}

static SmallVector<int64_t> getMMAShapeFromOp(vector::ContractionOp op) {
  Operation *cur = op.getOperation();
  while (cur) {
    if (auto cfg = getLoweringConfig(cur)) {
      int32_t k = getMmaKindRaw(cfg);
      if (k != 0) return getMMAShape(k);
    }
    cur = cur->getParentOp();
  }
  return {};
}

static int32_t getMmaKindFromOp(vector::ContractionOp op) {
  Operation *cur = op.getOperation();
  while (cur) {
    if (auto cfg = getLoweringConfig(cur))
      if (int32_t k = getMmaKindRaw(cfg)) return k;
    cur = cur->getParentOp();
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// §3  Forall helpers
//===----------------------------------------------------------------------===//

static bool isThreadForall(scf::ForallOp f) {
  auto m = f.getMappingAttr();
  if (!m || m.getValue().empty()) return false;
  for (Attribute a : m.getValue())
    if (!isa<gpu::GPUThreadMappingAttr>(a)) return false;
  return true;
}

static scf::ForallOp findEnclosingThreadForall(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForallOp>(p))
      if (isThreadForall(f)) return f;
  return nullptr;
}

// Returns true if every mapping attr on `f` is a GPUWarpMappingAttr.
static bool isWarpForall(scf::ForallOp f) {
  auto m = f.getMappingAttr();
  if (!m || m.empty()) return false;
  for (Attribute a : m.getValue())
    if (!isa<gpu::GPUWarpMappingAttr>(a)) return false;
  return true;
}

// Walk up parent chain looking for the innermost #gpu.warp-mapped forall.
// After the Subgroup tiling pass, MMA vector.contract ops live inside one
// of these.  Returns nullptr for non-MMA ops (no warp forall present).
static scf::ForallOp findEnclosingSubgroupForall(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForallOp>(p))
      if (isWarpForall(f)) return f;
  return nullptr;
}

static SmallVector<Value>
computeSgOffsets(OpBuilder &b, Location loc,
                 const OperandLayout &L, scf::ForallOp sgForall) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> off(L.rank, zero);
  if (!sgForall) return off;

  auto ivs = llvm::to_vector(sgForall.getInductionVars());
  // Map: which forall IV drives which operand dimension?
  // sg_strides[d] != 0 marks sg-distributed dimensions.
  // We pair them with IVs in order.
  int ivIdx = 0;
  for (int d = 0; d < L.rank && ivIdx < (int)ivs.size(); ++d) {
    if (L.sgStrides[d] == 0) continue;
    int64_t elemsPerSg =
        L.threadCounts[d] * L.batchCounts[d] * L.elemCounts[d];
    Value stride = b.create<arith::ConstantIndexOp>(loc, elemsPerSg);
    off[d] = b.create<arith::MulIOp>(loc, ivs[ivIdx++], stride);
  }
  return off;
}

//===----------------------------------------------------------------------===//
// §5  Generic per-thread offset (for non-MMA vector.contract fallback only)
//
// Only called when mma_kind == 0.  For mma_kind != 0 we emit nvgpu.mma.sync
// which is warp-synchronous — no per-thread offsets are needed or correct.
//===----------------------------------------------------------------------===//

static SmallVector<Value>
computeThreadOffsets(OpBuilder &b, Location loc,
                     Value linearTid, const OperandLayout &L, int kAxis,
                     bool mmaTileOnly = false,
                     ArrayRef<int64_t> elemIdx = {}) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> off(L.rank, zero);
  for (int d = 0; d < L.rank; ++d) {
    if (L.threadStrides[d] == 0 && L.threadCounts[d] <= 1) {
      // Still might have element offsets even if thread count is 1
      if (!elemIdx.empty() && d < (int)elemIdx.size() && elemIdx[d] > 0) {
         off[d] = b.create<arith::ConstantIndexOp>(loc, elemIdx[d] * L.threadCounts[d]);
      }
      continue;
    }
    Value div = b.create<arith::DivUIOp>(
        loc, linearTid,
        b.create<arith::ConstantIndexOp>(loc, L.threadStrides[d]));
    Value rem = b.create<arith::RemUIOp>(
        loc, div,
        b.create<arith::ConstantIndexOp>(loc, L.threadCounts[d]));
    int64_t baseElems = (mmaTileOnly) ? L.elemCounts[d]
                                    : (d == kAxis) ? L.elemCounts[d]
                                                   : L.batchCounts[d] * L.elemCounts[d];
    Value tOff = (baseElems == 1) ? rem
                               : b.create<arith::MulIOp>(
                                     loc, rem,
                                     b.create<arith::ConstantIndexOp>(loc, baseElems));
    if (!elemIdx.empty() && d < (int)elemIdx.size() && elemIdx[d] > 0) {
      Value eIdx = b.create<arith::ConstantIndexOp>(loc, elemIdx[d]);
      // Element stride for this thread = span of all threads in this dim within one subgroup tile
      int64_t eStride = L.threadCounts[d];
      Value eOff = b.create<arith::MulIOp>(loc, eIdx, b.create<arith::ConstantIndexOp>(loc, eStride));
      tOff = b.create<arith::AddIOp>(loc, tOff, eOff);
    }
    off[d] = tOff;
  }
  return off;
}

static SmallVector<Value>
computeElementOffsets(OpBuilder &b, Location loc, const OperandLayout &L,
                       ArrayRef<int64_t> elemIdx) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> off(L.rank, zero);
  for (int d = 0; d < L.rank; ++d) {
    if (!elemIdx.empty() && d < (int)elemIdx.size() && elemIdx[d] > 0) {
      off[d] = b.create<arith::ConstantIndexOp>(loc, elemIdx[d]);
    }
  }
  return off;
}

// interleavedTransferRead / interleavedTransferWrite
//
// These functions handle the ACC operand whose per-thread vector type is
//   shape[d] = batchCounts[d] * elemCounts[d]   (per perThreadType, kAxis=-1)
//
// The memory layout is NOT contiguous across batch tiles: between consecutive
// batch tiles in dimension d there is a gap of (threadCounts[d] - 1) elements
// because the adjacent elements belong to other threads in the warp.
//
// Correct memory offset for vector index vecIdx[d]:
//   batchIdx[d]  = vecIdx[d] / elemCounts[d]
//   tileIdx[d]   = vecIdx[d] % elemCounts[d]
//   memOff[d]    = batchIdx[d] * (threadCounts[d] * elemCounts[d]) + tileIdx[d]
//
// The loop below iterates over (batchIdx, tileIdx) explicitly so the memory
// offset and the insertion index into the result vector are both computed
// correctly without any intermediate flat-element decode.
static Value interleavedTransferRead(OpBuilder &b, vector::TransferReadOp read,
                                     const OperandLayout &L, VectorType vType,
                                     Value laneId, ArrayRef<Value> baseIndices,
                                     ArrayRef<Value> precomputedTOff = {}) {
  Location loc = read.getLoc();
  auto shape = vType.getShape();
  int vRank  = shape.size();
  int mRank  = baseIndices.size();
  Value result = b.create<arith::ConstantOp>(loc, vType, b.getZeroAttr(vType));

  SmallVector<Value> tOff;
  if (!precomputedTOff.empty()) {
    tOff = SmallVector<Value>(precomputedTOff.begin(), precomputedTOff.end());
  } else {
    tOff = computeThreadOffsets(b, loc, laneId, L, /*kAxis=*/-1,
                                /*mmaTileOnly=*/true);
  }

  SmallVector<int64_t> batchShape(vRank), elemShape(vRank);
  int64_t totalBatch = 1;
  for (int d = 0; d < vRank; ++d) {
    elemShape[d]  = L.elemCounts[d];
    batchShape[d] = (elemShape[d] > 0) ? (shape[d] / elemShape[d]) : 1;
    totalBatch   *= batchShape[d];
  }

  // ── Detect contiguous inner dimension for vectorized loads ──
  // If the innermost dimension of elemShape has size > 1 and stride 1 in the
  // source memref, we can load the entire inner row with one vector.transfer_read
  // instead of N scalar memref.load + vector.insert sequences.
  int innerDim = vRank - 1;
  int64_t innerElemCount = elemShape[innerDim];

  // Check if the source memref has stride 1 on the innermost dimension.
  // For subviews of shared memory (stride [64,1] or [8,1]) this is always true.
  bool innerContiguous = false;
  if (innerElemCount > 1) {
    auto memType = cast<MemRefType>(read.getBase().getType());
    // For static strides: check last stride == 1
    // For dynamic strides from subviews: conservatively check layout
    SmallVector<int64_t> strides;
    int64_t offset;
    if (succeeded(memType.getStridesAndOffset(strides, offset))) {
      if (!strides.empty() && strides.back() == 1)
        innerContiguous = true;
    }
    // Also handle the case where memref has identity layout (no explicit strides)
    if (memType.getLayout().isIdentity())
      innerContiguous = true;
  }

  for (int64_t bi = 0; bi < totalBatch; ++bi) {
    SmallVector<int64_t> bIdx(vRank);
    int64_t tmp = bi;
    for (int d = vRank - 1; d >= 0; --d) {
      bIdx[d] = tmp % batchShape[d];
      tmp     /= batchShape[d];
    }

    if (innerContiguous) {
  int64_t outerTotal = 1;
  SmallVector<int64_t> outerShape(vRank, 1);
  for (int d = 0; d < vRank; ++d) {
    if (d != innerDim) {
      outerShape[d] = elemShape[d];
      outerTotal *= elemShape[d];
    }
  }

  for (int64_t oi = 0; oi < outerTotal; ++oi) {
    SmallVector<int64_t> outerIdx(vRank, 0);
    tmp = oi;
    for (int d = vRank - 2; d >= 0; --d) {
      outerIdx[d] = tmp % outerShape[d];
      tmp /= outerShape[d];
    }

    // Compute memory indices for the start of this row
    SmallVector<Value> loadOff;
    for (int d = 0; d < mRank; ++d) {
      Value off = tOff[d];
      int64_t batchStride = L.threadCounts[d] * L.elemCounts[d];
      if (bIdx[d] > 0) {
        Value bs = b.create<arith::ConstantIndexOp>(loc, bIdx[d] * batchStride);
        off = b.create<arith::AddIOp>(loc, off, bs);
      }
      if (d != innerDim && outerIdx[d] > 0) {
        Value es = b.create<arith::ConstantIndexOp>(loc, outerIdx[d]);
        off = b.create<arith::AddIOp>(loc, off, es);
      }
      loadOff.push_back(b.create<arith::AddIOp>(loc, baseIndices[d], off));
    }

    // Emit a single vector load
    auto rowType = VectorType::get({innerElemCount}, vType.getElementType());
    Value padding = b.create<arith::ConstantOp>(
        loc, vType.getElementType(), b.getZeroAttr(vType.getElementType()));
    SmallVector<bool> inBounds(1, true);
    Value row = b.create<vector::TransferReadOp>(
        loc, rowType, read.getBase(), loadOff, padding, inBounds);

    // ── Direct insertion instead of element-by-element ──
    // Compute the insertion point in the result vector
    SmallVector<int64_t> insertOff(vRank, 0);
    SmallVector<int64_t> insertSize(vRank, 1);
    SmallVector<int64_t> insertStride(vRank, 1);
    for (int d = 0; d < vRank; ++d) {
      if (d == innerDim) {
        insertOff[d] = bIdx[d] * elemShape[d];
        insertSize[d] = innerElemCount;
      } else {
        insertOff[d] = bIdx[d] * elemShape[d] + outerIdx[d];
        insertSize[d] = 1;
      }
    }

    // Reshape the 1-D loaded row to match the slice shape for insertion
    auto sliceType = VectorType::get(insertSize, vType.getElementType());
    Value shaped = b.create<vector::ShapeCastOp>(loc, sliceType, row);
    result = b.create<vector::InsertStridedSliceOp>(
        loc, shaped, result, insertOff, insertStride);
  }
} else {
      // ── Scalar fallback: original element-by-element path ──
      int64_t totalElem = 1;
      for (int d = 0; d < vRank; ++d) totalElem *= elemShape[d];

      for (int64_t ei = 0; ei < totalElem; ++ei) {
        SmallVector<int64_t> eIdx(vRank);
        tmp = ei;
        for (int d = vRank - 1; d >= 0; --d) {
          eIdx[d] = tmp % elemShape[d];
          tmp     /= elemShape[d];
        }

        SmallVector<Value> finalOff;
        for (int d = 0; d < mRank; ++d) {
          Value off = tOff[d];
          int64_t batchStride = L.threadCounts[d] * L.elemCounts[d];
          if (bIdx[d] > 0) {
            Value bs = b.create<arith::ConstantIndexOp>(loc, bIdx[d] * batchStride);
            off = b.create<arith::AddIOp>(loc, off, bs);
          }
          if (eIdx[d] > 0) {
            Value es = b.create<arith::ConstantIndexOp>(loc, eIdx[d]);
            off = b.create<arith::AddIOp>(loc, off, es);
          }
          finalOff.push_back(b.create<arith::AddIOp>(loc, baseIndices[d], off));
        }

        SmallVector<int64_t> vecIdx(vRank);
        for (int d = 0; d < vRank; ++d)
          vecIdx[d] = bIdx[d] * elemShape[d] + eIdx[d];

        Value scalar = b.create<memref::LoadOp>(loc, read.getBase(), finalOff);
        result = b.create<vector::InsertOp>(loc, scalar, result, vecIdx);
      }
    }
  }
  return result;
}

static void interleavedTransferWrite(OpBuilder &b, vector::TransferWriteOp write,
                                      Value data, const OperandLayout &L,
                                      Value laneId, ArrayRef<Value> baseIndices,
                                      ArrayRef<Value> precomputedTOff = {}) {
  Location loc = write.getLoc();
  auto vType = cast<VectorType>(data.getType());
  auto shape = vType.getShape();
  int vRank  = shape.size();
  int mRank  = baseIndices.size();

  SmallVector<Value> tOff;
  if (!precomputedTOff.empty()) {
    tOff = SmallVector<Value>(precomputedTOff.begin(), precomputedTOff.end());
  } else {
    tOff = computeThreadOffsets(b, loc, laneId, L, /*kAxis=*/-1,
                                /*mmaTileOnly=*/true);
  }

  SmallVector<int64_t> batchShape(vRank), elemShape(vRank);
  int64_t totalBatch = 1;
  for (int d = 0; d < vRank; ++d) {
    elemShape[d]  = L.elemCounts[d];
    batchShape[d] = (elemShape[d] > 0) ? (shape[d] / elemShape[d]) : 1;
    totalBatch   *= batchShape[d];
  }

  int innerDim = vRank - 1;
  int64_t innerElemCount = elemShape[innerDim];

  bool innerContiguous = false;
  if (innerElemCount > 1) {
    auto memType = cast<MemRefType>(write.getBase().getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (succeeded(memType.getStridesAndOffset(strides, offset))) {
      if (!strides.empty() && strides.back() == 1)
        innerContiguous = true;
    }
    if (memType.getLayout().isIdentity())
      innerContiguous = true;
  }

  for (int64_t bi = 0; bi < totalBatch; ++bi) {
    SmallVector<int64_t> bIdx(vRank);
    int64_t tmp = bi;
    for (int d = vRank - 1; d >= 0; --d) {
      bIdx[d] = tmp % batchShape[d];
      tmp     /= batchShape[d];
    }

    if (innerContiguous) {
      int64_t outerTotal = 1;
      SmallVector<int64_t> outerShape(vRank, 1);
      for (int d = 0; d < vRank; ++d) {
        if (d != innerDim) {
          outerShape[d] = elemShape[d];
          outerTotal *= elemShape[d];
        }
      }

      for (int64_t oi = 0; oi < outerTotal; ++oi) {
        SmallVector<int64_t> outerIdx(vRank, 0);
        tmp = oi;
        for (int d = vRank - 2; d >= 0; --d) {
          outerIdx[d] = tmp % outerShape[d];
          tmp /= outerShape[d];
        }

        // ── Direct extraction using ExtractStridedSliceOp ──
        SmallVector<int64_t> extractOff(vRank, 0);
        SmallVector<int64_t> extractSize(vRank, 1);
        SmallVector<int64_t> extractStride(vRank, 1);
        for (int d = 0; d < vRank; ++d) {
          if (d == innerDim) {
            extractOff[d] = bIdx[d] * elemShape[d];
            extractSize[d] = innerElemCount;
          } else {
            extractOff[d] = bIdx[d] * elemShape[d] + outerIdx[d];
            extractSize[d] = 1;
          }
        }

        Value slice = b.create<vector::ExtractStridedSliceOp>(
            loc, data, extractOff, extractSize, extractStride);
        auto rowType = VectorType::get({innerElemCount}, vType.getElementType());
        Value row = b.create<vector::ShapeCastOp>(loc, rowType, slice);

        // Compute memory indices for the start of this row
        SmallVector<Value> storeOff;
        for (int d = 0; d < mRank; ++d) {
          Value off = tOff[d];
          int64_t batchStride = L.threadCounts[d] * L.elemCounts[d];
          if (bIdx[d] > 0) {
            Value bs = b.create<arith::ConstantIndexOp>(loc, bIdx[d] * batchStride);
            off = b.create<arith::AddIOp>(loc, off, bs);
          }
          if (d != innerDim && outerIdx[d] > 0) {
            Value es = b.create<arith::ConstantIndexOp>(loc, outerIdx[d]);
            off = b.create<arith::AddIOp>(loc, off, es);
          }
          storeOff.push_back(b.create<arith::AddIOp>(loc, baseIndices[d], off));
        }

        SmallVector<bool> inBounds(1, true);
        b.create<vector::TransferWriteOp>(
            loc, row, write.getBase(), storeOff, inBounds);
      }
    } else {
      // ── Scalar fallback ──
      int64_t totalElem = 1;
      for (int d = 0; d < vRank; ++d) totalElem *= elemShape[d];

      for (int64_t ei = 0; ei < totalElem; ++ei) {
        SmallVector<int64_t> eIdx(vRank);
        tmp = ei;
        for (int d = vRank - 1; d >= 0; --d) {
          eIdx[d] = tmp % elemShape[d];
          tmp     /= elemShape[d];
        }

        SmallVector<Value> finalOff;
        for (int d = 0; d < mRank; ++d) {
          Value off = tOff[d];
          int64_t batchStride = L.threadCounts[d] * L.elemCounts[d];
          if (bIdx[d] > 0) {
            Value bs = b.create<arith::ConstantIndexOp>(loc, bIdx[d] * batchStride);
            off = b.create<arith::AddIOp>(loc, off, bs);
          }
          if (eIdx[d] > 0) {
            Value es = b.create<arith::ConstantIndexOp>(loc, eIdx[d]);
            off = b.create<arith::AddIOp>(loc, off, es);
          }
          finalOff.push_back(b.create<arith::AddIOp>(loc, baseIndices[d], off));
        }

        SmallVector<int64_t> vecIdx(vRank);
        for (int d = 0; d < vRank; ++d)
          vecIdx[d] = bIdx[d] * elemShape[d] + eIdx[d];

        Value scalar = b.create<vector::ExtractOp>(loc, data, vecIdx);
        b.create<memref::StoreOp>(loc, scalar, write.getBase(), finalOff);
      }
    }
  }
}
//===----------------------------------------------------------------------===//
// §6  Per-thread vector type (for non-MMA fallback)
//===----------------------------------------------------------------------===//

static VectorType perThreadType(VectorType full,
                                const OperandLayout &L, int kAxis) {
  SmallVector<int64_t> shape(L.rank);
  for (int d = 0; d < L.rank; ++d)
    shape[d] = (d == kAxis) ? L.elemCounts[d]
                            : L.batchCounts[d] * L.elemCounts[d];
  return VectorType::get(shape, full.getElementType());
}

/// Per-thread type for a single MMA tile (no M/N batch factor).
/// Used inside emitMMASync's MN batch loop where batchCounts[M/N] is handled
/// by the loop itself — the read type must cover exactly one tile's elements.
static VectorType perTileThreadType(VectorType full,
                                    const OperandLayout &L, int kAxis) {
  SmallVector<int64_t> shape(L.rank);
  for (int d = 0; d < L.rank; ++d)
    shape[d] = L.elemCounts[d]; // no batchCounts multiplication for any dim
  return VectorType::get(shape, full.getElementType());
}

//===----------------------------------------------------------------------===//
// §7  Transfer read/write helpers (index adjustment)
//===----------------------------------------------------------------------===//

static Value adjustedTransferRead(OpBuilder &b,
                                   vector::TransferReadOp src,
                                   ArrayRef<Value> extraOff,
                                   VectorType resultType) {
  Location loc = src.getLoc();
  SmallVector<Value> idx(src.getIndices().begin(), src.getIndices().end());
  for (int i = 0; i < (int)idx.size(); ++i) {
    bool isZeroOff = false;
    if (auto c = extraOff[i].getDefiningOp<arith::ConstantIndexOp>())
      isZeroOff = (c.value() == 0);
    if (isZeroOff) continue;
    if (auto c = idx[i].getDefiningOp<arith::ConstantIndexOp>())
      if (c.value() == 0) { idx[i] = extraOff[i]; continue; }
    idx[i] = b.create<arith::AddIOp>(loc, idx[i], extraOff[i]);
  }
  SmallVector<bool> inBounds(resultType.getRank(), true);
  return b.create<vector::TransferReadOp>(
      loc, resultType, src.getBase(), idx,
      /*padding=*/std::nullopt, inBounds);
}

static void adjustedTransferWrite(OpBuilder &b,
                                   vector::TransferWriteOp dst,
                                   Value vec,
                                   ArrayRef<Value> extraOff,
                                   Value overrideBase = nullptr) {
  Location loc = dst.getLoc();
  SmallVector<Value> idx(dst.getIndices().begin(), dst.getIndices().end());
  for (int i = 0; i < (int)idx.size(); ++i) {
    bool isZeroOff = false;
    if (auto c = extraOff[i].getDefiningOp<arith::ConstantIndexOp>())
      isZeroOff = (c.value() == 0);
    if (isZeroOff) continue;
    if (auto c = idx[i].getDefiningOp<arith::ConstantIndexOp>())
      if (c.value() == 0) { idx[i] = extraOff[i]; continue; }
    idx[i] = b.create<arith::AddIOp>(loc, idx[i], extraOff[i]);
  }
  Value base = overrideBase ? overrideBase : dst.getBase();
  SmallVector<bool> inBounds(
      cast<VectorType>(vec.getType()).getRank(), true);
  b.create<vector::TransferWriteOp>(loc, vec, base, idx, inBounds);
}

//===----------------------------------------------------------------------===//
// §8  Warp shuffle reduction (for K-parallel threads, generic)
//===----------------------------------------------------------------------===//
static bool hasWorkgroupAddressSpace(MemRefType memType) {
  if (auto addrSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
          memType.getMemorySpace()))
    return addrSpace.getValue() == gpu::AddressSpace::Workgroup;
  return false;
}


struct LdMatrixConfig {
  int32_t numTiles;
  bool transpose;       // PTX ldmatrix.trans flag — only valid for 16-bit types
  bool indexTranspose;  // true when row→K-axis, col→N-axis (B operand memory layout)
  // Row/col computation parameters (all derived from mmaShape + elemWidth)
  int64_t rowMod;   // row = laneId % rowMod
  int64_t colDiv;   // colGroup = laneId / colDiv
  int64_t colMul;   // col = colGroup * colMul
};

// ldmatrix is only defined for 16-bit element types (f16/bf16/i16).
// For wider types (tf32/f32/f64) callers must fall back to scalar LDS loads.
static bool canUseLdMatrix(Type elemType) {
  return elemType.getIntOrFloatBitWidth() <= 16;
}

static LdMatrixConfig computeLdMatrixConfig(
    ArrayRef<int64_t> mmaShape, Type elemType, bool isOperandB) {
  int64_t M = mmaShape[0], N = mmaShape[1], K = mmaShape[2];
  int64_t W = elemType.getIntOrFloatBitWidth();
  int64_t elemsPerLine = 128 / W;  // elements in one 128-bit load

  LdMatrixConfig cfg;
  if (!isOperandB) {
    // A operand: row-major, shape is M × K
    int64_t mTile = M / 8;
    int64_t kTile = K / elemsPerLine;
    cfg.numTiles       = mTile * kTile;
    cfg.transpose      = false;  // ldmatrix.trans not needed for A
    cfg.indexTranspose = false;  // row → M-axis, col → K-axis
    cfg.rowMod         = M;            // row cycles over M rows
    cfg.colDiv         = M;            // new column group every M lanes
    cfg.colMul         = elemsPerLine; // each group shifts by elemsPerLine columns
  } else {
    // B operand: transposed (col-major in MMA terms), shape is K × N.
    // NOTE: computeLdMatrixConfig must only be called when canUseLdMatrix()
    // is true (W <= 16). For wider types callers use interleavedTransferRead.
    int64_t kTile = K / elemsPerLine;
    int64_t nTile = N / 8;
    cfg.numTiles       = kTile * nTile;
    cfg.transpose      = true;
    cfg.indexTranspose = true; // row → K-axis, col → N-axis
    cfg.rowMod         = K;            // row cycles over K rows
    cfg.colDiv         = K;            // new column group every K lanes
    cfg.colMul         = elemsPerLine; // each group shifts by elemsPerLine columns
  }
  return cfg;
}

/// Build the ldmatrix row and col index values from laneId and config.
static std::pair<Value, Value> buildLdMatrixIndices(
    OpBuilder &b, Location loc, Value laneId, const LdMatrixConfig &cfg) {
  Value rowMod = b.create<arith::ConstantIndexOp>(loc, cfg.rowMod);
  Value row = b.create<arith::RemUIOp>(loc, laneId, rowMod);

  Value col;
  if (cfg.colMul == 0 || cfg.colDiv >= 32) {
    // All lanes fit within one group — col is always 0
    col = b.create<arith::ConstantIndexOp>(loc, 0);
  } else {
    Value colDiv = b.create<arith::ConstantIndexOp>(loc, cfg.colDiv);
    Value colMul = b.create<arith::ConstantIndexOp>(loc, cfg.colMul);
    Value group = b.create<arith::DivUIOp>(loc, laneId, colDiv);
    col = b.create<arith::MulIOp>(loc, group, colMul);
  }
  return {row, col};
}


static Value warpShuffleReduce(OpBuilder &b, Location loc, Value vec,
                                int64_t kThreadCount, int64_t kStride) {
  auto vt = cast<VectorType>(vec.getType());
  int64_t n = 1;
  for (int64_t d : vt.getShape()) n *= d;
  auto flatTy = VectorType::get({n}, vt.getElementType());
  Value flat = b.create<vector::ShapeCastOp>(loc, flatTy, vec);
  auto i32 = b.getI32Type();
  Value w32 = b.create<arith::ConstantOp>(loc, IntegerAttr::get(i32, 32));
  for (int64_t step = 1; step < kThreadCount; step *= 2) {
    Value mask = b.create<arith::ConstantOp>(
        loc, IntegerAttr::get(i32, step * kStride));
    Value snap = flat;
    for (int64_t i = 0; i < n; ++i) {
      Value e = b.create<vector::ExtractOp>(loc, snap, ArrayRef<int64_t>{i});
      auto sh = b.create<gpu::ShuffleOp>(loc, e, mask, w32,
                                          gpu::ShuffleMode::XOR);
      Value r = b.create<arith::AddFOp>(loc, e, sh.getShuffleResult());
      flat = b.create<vector::InsertOp>(loc, r, flat, ArrayRef<int64_t>{i});
    }
  }
  return b.create<vector::ShapeCastOp>(loc, vt, flat);
}


struct MMAEmitter {
  OpBuilder &b;
  Location loc;
  vector::ContractionOp contractOp;
  const OperandLayout &L0, &L1, &L2; // LHS, RHS, ACC layouts
  vector::TransferReadOp lhsRead, rhsRead;
  int lhsKAxis, rhsKAxis, accMAxis, accNAxis;
  int64_t kBatchCount, kStep;
  SmallVector<int64_t> mmaShape;
  int32_t mmaKind;

  // Emit one complete K-batch reduction and return the updated accumulator.
  Value emit(Value accIn) {
    if (mmaKind != 0)
      return emitMMASync(accIn);
    return emitGenericContract(accIn);
  }

private:

  Value emitMMASync(Value acc) {
    auto elemTy = cast<VectorType>(contractOp.getLhs().getType())
                      .getElementType();
    auto frags = deriveMMAFragments(mmaShape, elemTy);
    if (failed(frags))
      return emitGenericContract(acc);

    // isTf32: nvgpu.mma.sync requires this flag when inputs are f32 (tf32 precision).
    // Derived from element type — f32 inputs always imply tf32 MMA;
    // f16/bf16/i8 inputs never do. This avoids a hardcoded intrinsic enum check.
    bool isTf32 = elemTy.isF32();

    auto aFrag2D = VectorType::get(frags->a, elemTy);
    auto bFrag2D = VectorType::get(frags->b, elemTy);
    auto cFrag2D = VectorType::get(frags->c, elemTy);

    auto lhsFullTy = cast<VectorType>(lhsRead.getResult().getType());
    auto rhsFullTy = cast<VectorType>(rhsRead.getResult().getType());
    auto lhsThrTy  = perTileThreadType(lhsFullTy, L0, lhsKAxis);
    auto rhsThrTy  = perTileThreadType(rhsFullTy, L1, rhsKAxis);

    int64_t lhsThrElems = 1, rhsThrElems = 1;
    for (int64_t d : lhsThrTy.getShape()) lhsThrElems *= d;
    for (int64_t d : rhsThrTy.getShape()) rhsThrElems *= d;
    int64_t aFragElems = frags->a[0] * frags->a[1];
    int64_t bFragElems = frags->b[0] * frags->b[1];
    if (lhsThrElems != aFragElems || rhsThrElems != bFragElems)
      return emitGenericContract(acc);

    Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value constant32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value laneId = b.create<arith::RemUIOp>(loc, tid, constant32);

    // ── Detect shared memory ──
    auto lhsMemType = cast<MemRefType>(lhsRead.getBase().getType());
    auto rhsMemType = cast<MemRefType>(rhsRead.getBase().getType());
    bool lhsIsShared = false, rhsIsShared = false;
    if (auto as = dyn_cast_or_null<gpu::AddressSpaceAttr>(
            lhsMemType.getMemorySpace()))
      lhsIsShared = (as.getValue() == gpu::AddressSpace::Workgroup);
    if (auto as = dyn_cast_or_null<gpu::AddressSpaceAttr>(
            rhsMemType.getMemorySpace()))
      rhsIsShared = (as.getValue() == gpu::AddressSpace::Workgroup);

    // ldmatrix is only legal for 16-bit element types (f16/bf16).
    // For wider types (tf32/f32) shared-mem operands must use scalar LDS
    // (interleavedTransferRead) even though the data lives in shared memory.
    const bool useLdm = canUseLdMatrix(elemTy);

    // ── Derive ldmatrix configs from mmaShape + element type ──
    LdMatrixConfig ldmCfgA, ldmCfgB;
    Value ldmA_row, ldmA_col, ldmB_row, ldmB_col;

    if (lhsIsShared && useLdm) {
      ldmCfgA = computeLdMatrixConfig(mmaShape, elemTy, /*isOperandB=*/false);
      std::tie(ldmA_row, ldmA_col) =
          buildLdMatrixIndices(b, loc, laneId, ldmCfgA);
    }
    if (rhsIsShared && useLdm) {
      ldmCfgB = computeLdMatrixConfig(mmaShape, elemTy, /*isOperandB=*/true);
      std::tie(ldmB_row, ldmB_col) =
          buildLdMatrixIndices(b, loc, laneId, ldmCfgB);
    }

    // ── Scalar fallback offsets (when NOT using ldmatrix) ──
    // Computed once here, passed as precomputedTOff into interleavedTransferRead
    // to avoid re-emitting divui/remui/muli thread-position arithmetic on every
    // iteration of the kb × bm × bn loop (kAxis irrelevant when mmaTileOnly=true).
    SmallVector<Value> off0, off1;
    if (!lhsIsShared || !useLdm)
      off0 = computeThreadOffsets(b, loc, laneId, L0, /*kAxis=*/-1, /*mmaTileOnly=*/true);
    if (!rhsIsShared || !useLdm)
      off1 = computeThreadOffsets(b, loc, laneId, L1, /*kAxis=*/-1, /*mmaTileOnly=*/true);

    // ── MN batch axis detection ──
    int lhsMAxis = -1, rhsNAxis = -1;
    for (int d = 0; d < L0.rank; ++d)
      if (d != lhsKAxis) { lhsMAxis = d; break; }
    for (int d = 0; d < L1.rank; ++d)
      if (d != rhsKAxis) { rhsNAxis = d; break; }

    int64_t batchM = (lhsMAxis >= 0) ? L0.batchCounts[lhsMAxis] : 1;
    int64_t batchN = (rhsNAxis >= 0) ? L1.batchCounts[rhsNAxis] : 1;

    // FIX: use MMA tile dimensions directly instead of
    // threadCounts * elemCounts which is wrong for non-standard shapes.
    // elemM = number of M rows in one MMA tile = mmaShape[0]
    // elemN = number of N cols in one MMA tile = mmaShape[1]
    int64_t elemM = mmaShape[0];  // e.g. 16 for m16n8k8
    int64_t elemN = mmaShape[1];  // e.g.  8 for m16n8k8

    SmallVector<int64_t> accSliceShape(
        cast<VectorType>(acc.getType()).getShape());
    if (accMAxis >= 0) accSliceShape[accMAxis] = L2.elemCounts[accMAxis];
    if (accNAxis >= 0) accSliceShape[accNAxis] = L2.elemCounts[accNAxis];
    SmallVector<int64_t> accOnes(
        cast<VectorType>(acc.getType()).getRank(), 1);

    Value result = acc;
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);

    for (int64_t kb = 0; kb < kBatchCount; ++kb) {
      int64_t kOff = kb * kStep;

      // Cache A fragments across bn: A depends on (kb, bm), not bn.
      // Cache B fragments across bm: B depends on (kb, bn), not bm.
      SmallVector<Value> aFragCache(batchM), bFragCache(batchN);

      for (int64_t bm = 0; bm < batchM; ++bm) {
        if (lhsIsShared && useLdm) {
          aFragCache[bm] = emitLdMatrix(
              lhsRead, lhsMAxis, lhsKAxis,
              ldmA_row, ldmA_col, ldmCfgA,
              bm * elemM, kOff, aFrag2D, zero);
        } else {
          SmallVector<Value> tileOff(L0.rank, zero);
          if (kOff > 0 && lhsKAxis >= 0)
            tileOff[lhsKAxis] = b.create<arith::ConstantIndexOp>(loc, kOff);
          if (bm > 0 && lhsMAxis >= 0)
            tileOff[lhsMAxis] =
                b.create<arith::ConstantIndexOp>(loc, bm * elemM);
          Value aSlice = interleavedTransferRead(
              b, lhsRead, L0, lhsThrTy, laneId, tileOff, off0);
          aFragCache[bm] = b.create<vector::ShapeCastOp>(loc, aFrag2D, aSlice);
        }
      }

      for (int64_t bn = 0; bn < batchN; ++bn) {
        if (rhsIsShared && useLdm) {
          bFragCache[bn] = emitLdMatrix(
              rhsRead, rhsNAxis, rhsKAxis,
              ldmB_row, ldmB_col, ldmCfgB,
              bn * elemN, kOff, bFrag2D, zero);
        } else {
          SmallVector<Value> tileOff(L1.rank, zero);
          if (kOff > 0 && rhsKAxis >= 0)
            tileOff[rhsKAxis] = b.create<arith::ConstantIndexOp>(loc, kOff);
          if (bn > 0 && rhsNAxis >= 0)
            tileOff[rhsNAxis] =
                b.create<arith::ConstantIndexOp>(loc, bn * elemN);
          Value bSlice = interleavedTransferRead(
              b, rhsRead, L1, rhsThrTy, laneId, tileOff, off1);
          bFragCache[bn] = b.create<vector::ShapeCastOp>(loc, bFrag2D, bSlice);
        }
      }

      for (int64_t bm = 0; bm < batchM; ++bm) {
        for (int64_t bn = 0; bn < batchN; ++bn) {
          Value aFrag = aFragCache[bm];
          Value bFrag = bFragCache[bn];

          // ── ACC slice → C fragment ──
          SmallVector<int64_t> accOff(
              cast<VectorType>(result.getType()).getRank(), 0);
          if (accMAxis >= 0)
            accOff[accMAxis] = bm * L2.elemCounts[accMAxis];
          if (accNAxis >= 0)
            accOff[accNAxis] = bn * L2.elemCounts[accNAxis];
          Value cSlice = b.create<vector::ExtractStridedSliceOp>(
              loc, result, accOff, accSliceShape, accOnes);
          Value cFrag =
              b.create<vector::ShapeCastOp>(loc, cFrag2D, cSlice);

          // ── nvgpu.mma.sync ──
          Value cResult = b.create<nvgpu::MmaSyncOp>(
              loc, aFrag, bFrag, cFrag,
              frags->mmaShape, isTf32);

          Value accResult = b.create<vector::ShapeCastOp>(
              loc,
              VectorType::get(accSliceShape, elemTy),
              cResult);
          result = b.create<vector::InsertStridedSliceOp>(
              loc, accResult, result, accOff, accOnes);
        }
      }
    }
    return result;
  }

  Value emitLdMatrix(vector::TransferReadOp read,
                     int spatialAxis, int kAxis,
                     Value ldmRow, Value ldmCol,
                     const LdMatrixConfig &cfg,
                     int64_t spatialOff, int64_t kOff,
                     VectorType fragType, Value zero) {
    SmallVector<Value> baseIndices(read.getIndices());
    int memRank = baseIndices.size();

    // ldmatrix row maps to the "row" dimension of the operand:
    //   For A (indexTranspose=false): row → M-axis (spatialAxis), col → K-axis
    //   For B (indexTranspose=true):  row → K-axis,               col → N-axis (spatialAxis)
    // NOTE: cfg.indexTranspose controls the memory index mapping.
    //       cfg.transpose is ONLY the PTX ldmatrix.trans flag (valid ≤16-bit only).
    int rowDim, colDim;
    if (!cfg.indexTranspose) {
      // A: row → spatialAxis (M), col → kAxis
      rowDim = spatialAxis;
      colDim = kAxis;
    } else {
      // B: row → kAxis, col → spatialAxis (N)
      rowDim = kAxis;
      colDim = spatialAxis;
    }

    SmallVector<Value> ldmIndices(memRank, zero);
    for (int d = 0; d < memRank; ++d) {
      Value idx = baseIndices[d];
      if (d == rowDim) {
        // Add hardware row + tile offset
        Value off = ldmRow;
        int64_t tileOff = cfg.indexTranspose ? kOff : spatialOff;
        if (tileOff > 0) {
          Value tOff = b.create<arith::ConstantIndexOp>(loc, tileOff);
          off = b.create<arith::AddIOp>(loc, off, tOff);
        }
        ldmIndices[d] = b.create<arith::AddIOp>(loc, idx, off);
      } else if (d == colDim) {
        // Add hardware col + tile offset
        Value off = ldmCol;
        int64_t tileOff = cfg.indexTranspose ? spatialOff : kOff;
        if (tileOff > 0) {
          Value tOff = b.create<arith::ConstantIndexOp>(loc, tileOff);
          off = b.create<arith::AddIOp>(loc, off, tOff);
        }
        ldmIndices[d] = b.create<arith::AddIOp>(loc, idx, off);
      } else {
        // Batch or other dims: just use base
        ldmIndices[d] = idx;
      }
    }

    return b.create<nvgpu::LdMatrixOp>(
        loc, fragType, read.getBase(), ldmIndices,
        cfg.transpose, cfg.numTiles);
  }
  // ── Generic vector.contract fallback ─────────────────────────────────────
  //
  // Uses per-thread offsets (divui/remui) for non-MMA ops only.
  Value emitGenericContract(Value acc) {
    Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value constant32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value laneId = b.create<arith::RemUIOp>(loc, tid, constant32);

    auto lhsFullTy = cast<VectorType>(lhsRead.getResult().getType());
    auto rhsFullTy = cast<VectorType>(rhsRead.getResult().getType());

    auto lhsThr = perThreadType(lhsFullTy, L0, lhsKAxis);
    auto rhsThr = perThreadType(rhsFullTy, L1, rhsKAxis);

    SmallVector<Value> off0 = computeThreadOffsets(b, loc, laneId, L0, lhsKAxis);
    SmallVector<Value> off1 = computeThreadOffsets(b, loc, laneId, L1, rhsKAxis);

    for (int64_t kb = 0; kb < kBatchCount; ++kb) {
      SmallVector<Value> lo0 = off0, lo1 = off1;
      if (kb > 0 && lhsKAxis >= 0) {
        Value adv0 = b.create<arith::ConstantIndexOp>(loc, kb * kStep);
        Value adv1 = b.create<arith::ConstantIndexOp>(loc, kb * kStep);
        lo0[lhsKAxis] = b.create<arith::AddIOp>(loc, off0[lhsKAxis], adv0);
        lo1[rhsKAxis] = b.create<arith::AddIOp>(loc, off1[rhsKAxis], adv1);
      }
      Value lhsSlice = adjustedTransferRead(b, lhsRead, lo0, lhsThr);
      Value rhsSlice = adjustedTransferRead(b, rhsRead, lo1, rhsThr);
      auto partial = b.create<vector::ContractionOp>(
          loc, lhsSlice, rhsSlice, acc,
          contractOp.getIndexingMaps(), contractOp.getIteratorTypes());
      if (auto cfg = contractOp->getAttr("lowering_config"))
        partial->setAttr("lowering_config", cfg);
      acc = partial.getResult();
    }
    return acc;
  }
};

//===----------------------------------------------------------------------===//
// §10  ACC axis detection (generic, layout-driven, no op-specific knowledge)
//===----------------------------------------------------------------------===//

static void findAccAxes(const OperandLayout &L2,
                        int &accMAxis, int &accNAxis) {
  accMAxis = accNAxis = -1;
  // M axis: first dimension distributed across subgroups (sgStrides != 0).
  for (int d = 0; d < L2.rank && accMAxis < 0; ++d)
    if (L2.sgStrides[d] != 0) accMAxis = d;
  // N axis: first dimension with multiple threads but no sg distribution.
  for (int d = 0; d < L2.rank && accNAxis < 0; ++d)
    if (d != accMAxis && L2.threadCounts[d] > 1 && L2.sgStrides[d] == 0)
      accNAxis = d;
  // Positional fallback.
  if (accMAxis < 0) accMAxis = (L2.rank > 1) ? 1 : 0;
  if (accNAxis < 0) accNAxis = (L2.rank > 2) ? 2 : (L2.rank > 1 ? 1 : 0);
}

//===----------------------------------------------------------------------===//
// §11  Contract-to-K-axis mapping
//===----------------------------------------------------------------------===//

static int getReductionDim(vector::ContractionOp op) {
  auto types = op.getIteratorTypes().getValue();
  for (int i = 0; i < (int)types.size(); ++i)
    if (cast<vector::IteratorTypeAttr>(types[i]).getValue() ==
        vector::IteratorType::reduction)
      return i;
  return -1;
}

static int contractDimToOperandDim(AffineMap m, int contractDim) {
  for (int i = 0; i < (int)m.getNumResults(); ++i)
    if (auto e = llvm::dyn_cast<AffineDimExpr>(m.getResult(i)))
      if ((int)e.getPosition() == contractDim) return i;
  return -1;
}

//===----------------------------------------------------------------------===//
// §12  Main distribution function (barrier insertion delegated to
//      NovaGPUInsertWorkgroupBarriersPass — see Memory/NovaGPUBufferize.cpp)
//===----------------------------------------------------------------------===//

static LogicalResult distributeContract(IRRewriter &rw,
                                         vector::ContractionOp op,
                                         int64_t kUnrollLimit) {
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard guard(rw);
  rw.setInsertionPoint(op);

  // Read layouts.
  auto L0 = readLayout(op, 0);
  auto L1 = readLayout(op, 1);
  auto L2 = readLayout(op, 2);
  if (failed(L0) || failed(L1) || failed(L2)) {
    op->emitError("nova-gpu-vector-distribute: missing nova.layout_* attrs");
    return failure();
  }

  // Require transfer_read operands for LHS and RHS.
  auto lhsRead = op.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto rhsRead = op.getRhs().getDefiningOp<vector::TransferReadOp>();
  if (!lhsRead || !rhsRead) {
    op->emitError("nova-gpu-vector-distribute: LHS/RHS must be transfer_reads");
    return failure();
  }

  // K-axis info.
  int kDim = getReductionDim(op);
  auto maps = op.getIndexingMapsArray();
  int lhsKAxis = (kDim >= 0) ? contractDimToOperandDim(maps[0], kDim) : -1;
  int rhsKAxis = (kDim >= 0) ? contractDimToOperandDim(maps[1], kDim) : -1;

  int64_t kBatch = 1, kStep = 1;
  int64_t lhsKThreadStride = 0;
  if (lhsKAxis >= 0 && rhsKAxis >= 0) {
    // K-step from MMA shape (generic: derived purely from mmaShape[2]).
    SmallVector<int64_t> mmaShape = getMMAShapeFromOp(op);
    if (!mmaShape.empty()) {
      kStep = mmaShape[2];
    } else {
      kStep = L0->threadCounts[lhsKAxis] * L0->elemCounts[lhsKAxis];
    }
    kBatch = L0->batchCounts[lhsKAxis];
    lhsKThreadStride = L0->threadStrides[lhsKAxis];
  }

  // ACC handling.
  int accMAxis, accNAxis;
  findAccAxes(*L2, accMAxis, accNAxis);

  auto accRead = op.getAcc().getDefiningOp<vector::TransferReadOp>();
  // After the Subgroup tiling pass, MMA contracts live inside a #gpu.warp
  // forall.  findEnclosingSubgroupForall walks up for that.  For non-MMA
  // contracts (mmaKind==0) it returns nullptr — computeSgOffsets handles
  // nullptr by returning zero offsets, which is correct for that path.
  scf::ForallOp sgForall = findEnclosingSubgroupForall(op);

  // Build ACC vector.
  // For MMA ops: ACC is read at sg-offset from global output (no smem bounce).
  // For generic: ACC is read per-thread.
  Value acc;
  SmallVector<Value> off2;
  auto accType = cast<VectorType>(op.getAcc().getType());
  VectorType accThr = perThreadType(accType, *L2, /*kAxis=*/-1);
  int32_t mmaKind = getMmaKindFromOp(op);

  if (accRead) {
    if (mmaKind != 0) {
      // MMA path: ACC is read from the tiled subview that the subgroup tiling
      // pass already offset by warp_id * elemsPerWarp.  We must still add
      // thread-level offsets within the warp tile.
      Value tid = rw.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      Value constant32 = rw.create<arith::ConstantIndexOp>(loc, 32);
      Value laneId = rw.create<arith::RemUIOp>(loc, tid, constant32);
      SmallVector<Value> accIndices(accRead.getIndices());
      acc = interleavedTransferRead(rw, accRead, *L2, accThr, laneId, accIndices);
    } else {
      // Generic path: per-thread read.
      Value tid = rw.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      Value constant32 = rw.create<arith::ConstantIndexOp>(loc, 32);
      Value laneId = rw.create<arith::RemUIOp>(loc, tid, constant32);
      off2 = computeThreadOffsets(rw, loc, laneId, *L2, /*kAxis=*/-1);
      acc = adjustedTransferRead(rw, accRead, off2, accThr);
    }
  } else {
    off2 = computeSgOffsets(rw, loc, *L2, sgForall);
    if (auto c = op.getAcc().getDefiningOp<arith::ConstantOp>()) {
      if (auto d = dyn_cast<DenseElementsAttr>(c.getValue()))
        if (d.isSplat())
          acc = rw.create<arith::ConstantOp>(
              loc, DenseElementsAttr::get(accThr, d.getSplatValue<Attribute>()));
    }
    if (!acc)
      acc = rw.create<arith::ConstantOp>(loc, rw.getZeroAttr(accThr));
  }

  // Emit MMA/contract.
  SmallVector<int64_t> mmaShape = getMMAShapeFromOp(op);
  MMAEmitter emitter{rw, loc, op, *L0, *L1, *L2,
                     lhsRead, rhsRead,
                     lhsKAxis, rhsKAxis, accMAxis, accNAxis,
                     kBatch, kStep, mmaShape, mmaKind};
  Value result = emitter.emit(acc);

  // Warp-shuffle reduction if K is thread-parallel (generic path only).
  if (mmaKind == 0 && lhsKAxis >= 0 && L0->threadCounts[lhsKAxis] > 1) {
    int64_t kTC = L0->threadCounts[lhsKAxis];
    int64_t stride = (lhsKThreadStride > 0) ? lhsKThreadStride : 1;
    assert(llvm::isPowerOf2_64((uint64_t)kTC) && kTC <= 32);
    result = warpShuffleReduce(rw, loc, result, kTC, stride);
  }

  // Write result.
  Value contractResult = op.getResult();
  vector::TransferWriteOp resultWrite;
  for (Operation *u : contractResult.getUsers())
    if (auto w = dyn_cast<vector::TransferWriteOp>(u)) { resultWrite = w; break; }

  if (resultWrite) {
    rw.setInsertionPoint(resultWrite);
    Location wLoc = resultWrite.getLoc();
    if (mmaKind != 0) {
      // Partition. Fragmented transfers ensure no collision between threads.
      Value tid = rw.create<gpu::ThreadIdOp>(wLoc, gpu::Dimension::x);
      Value constant32 = rw.create<arith::ConstantIndexOp>(wLoc, 32);
      Value laneId = rw.create<arith::RemUIOp>(wLoc, tid, constant32);
      SmallVector<Value> writeIndices(resultWrite.getIndices());
      interleavedTransferWrite(rw, resultWrite, result, *L2, laneId, writeIndices);
    } else {
      adjustedTransferWrite(rw, resultWrite, result, off2,
                            /*overrideBase=*/nullptr);
    }
    rw.eraseOp(resultWrite);
  }

  // Cleanup.
  rw.eraseOp(op);
  if (lhsRead->use_empty()) rw.eraseOp(lhsRead);
  if (rhsRead->use_empty()) rw.eraseOp(rhsRead);
  if (accRead && accRead->use_empty()) rw.eraseOp(accRead);
  return success();
}

//===----------------------------------------------------------------------===//
// §13  Pass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorDistributePass
    : public PassWrapper<NovaGPUVectorDistributePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorDistributePass)

  NovaGPUVectorDistributePass() = default;
  NovaGPUVectorDistributePass(const NovaGPUVectorDistributePass &other)
      : PassWrapper(other) {}

  Option<int64_t> kBatchUnrollLimit{
      *this, "k-batch-unroll-limit",
      llvm::cl::desc("Max K-batch unroll (default 16)"),
      llvm::cl::init(16)};

  StringRef getArgument()    const override { return "nova-gpu-vector-distribute"; }
  StringRef getDescription() const override {
    return "Distribute vector.contract to warp-level nvgpu.mma.sync "
           "(MMA path) or per-thread slices (generic fallback).";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, gpu::GPUDialect, vector::VectorDialect,
             func::FuncDialect, scf::SCFDialect, NVVM::NVVMDialect,
             nvgpu::NVGPUDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    IRRewriter rw(&getContext());

    SmallVector<vector::ContractionOp> ops;
    fn.walk([&](vector::ContractionOp op) {
      if (op->hasAttr("nova.layout_0")) ops.push_back(op);
    });
    for (auto op : ops)
      if (failed(distributeContract(rw, op, kBatchUnrollLimit)))
        return signalPassFailure();
    // Barrier insertion is intentionally omitted here.
    // NovaGPUInsertWorkgroupBarriersPass owns all barrier placement and runs
    // after bufferization, so inserting barriers here would create duplicates.
  }
};

} // namespace

std::unique_ptr<Pass> createNovaGPUVectorDistributePass() {
  return std::make_unique<NovaGPUVectorDistributePass>();
}
void registerNovaGPUVectorDistributePass() {
  PassRegistration<NovaGPUVectorDistributePass>();
}

} // namespace mlir::nova