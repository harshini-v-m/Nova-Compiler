//===- NovaGPUNestedLayoutDistributionPatterns.cpp
//-------------------------===//
//
// Port of IREE's GPUNestedLayoutDistributionPatterns.cpp.
// Patterns for distributing ops annotated with NestedLayoutAttr.
//
// Patterns implemented (P1 for batch_matmul):
//   DistributeTransferRead    — vector.transfer_read with nested layout
//   DistributeTransferWrite   — vector.transfer_write with nested layout
//   DistributeBroadcast       — vector.broadcast
//   DistributeMultiReduction  — vector.multi_reduction (K-thread reduce)
//   DistributeContract        — vector.contract (nested layout, non-MMA path)
//   DistributeShapeCast       — vector.shape_cast
//
// Key differences from IREE:
//   - nova::vec_ext namespace instead of IREE::VectorExt
//   - No MaskedOpDistributionPattern; mask handling omitted for MVP
//   - No IREE LinalgExt scatter/gather patterns
//   - No AMD-specific patterns
//   - basisFromSizesStrides inlined (static in NovaVectorExtAttrs.cpp)
//   - getCombiningIdentityValue implemented inline
//
//===----------------------------------------------------------------------===//

#include "NovaGPUVectorDistribution.h"

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaVectorOpUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <numeric>

#define DEBUG_TYPE "nova-gpu-vector-distribute"

namespace mlir::nova {

using namespace vec_ext;
using VectorValue = TypedValue<VectorType>;

//===----------------------------------------------------------------------===//
// Local helpers
//===----------------------------------------------------------------------===//

static bool isBroadcast(AffineExpr expr) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
    return constExpr.getValue() == 0;
  return false;
}

/// Compute the element-level (innermost) tile shape from a nested layout.
/// Returns a shape with 1s in all batch/outer positions and the element sizes.
static SmallVector<int64_t> getElementVectorTileShape(NestedLayoutAttr layout) {
  int64_t rank = layout.getRank();
  SmallVector<int64_t> tileShape = layout.getDistributedShape();
  for (int i = 0, e = rank * 2; i < e; ++i)
    tileShape[i] = 1;
  return tileShape;
}

/// Compute the flat offsets into the distributed vector for a given
/// batch/outer offset combination.
static SmallVector<int64_t>
getDistributedTransferOffsets(ArrayRef<int64_t> offsets,
                              NestedLayoutAttr layout) {
  int64_t rank = layout.getRank();
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerOffsets(offsets.begin() + rank, rank);
  ArrayRef<int64_t> outerSizes = layout.getOuterTile();
  ArrayRef<int64_t> elementSizes = layout.getElementTile();

  SmallVector<int64_t> result;
  result.reserve(rank);
  for (auto [b, o, os, es] :
       llvm::zip(batchOffsets, outerOffsets, outerSizes, elementSizes))
    result.push_back(b * os * es + o * es);
  return result;
}

/// Cache key for a linearize_index op: SSA values + integer constants + sizes.
/// Allows getTransferIndices to reuse identical linearizations across tile
/// iterations instead of emitting a fresh op each time.
struct LinearizeKey {
  Value warp, thread, base;
  int64_t batchOff, outerOff;
  SmallVector<int64_t, 5> sizes;
  bool operator==(const LinearizeKey &o) const {
    return warp == o.warp && thread == o.thread && base == o.base &&
           batchOff == o.batchOff && outerOff == o.outerOff && sizes == o.sizes;
  }
};
struct LinearizeKeyInfo : llvm::DenseMapInfo<LinearizeKey> {
  static LinearizeKey getEmptyKey() {
    return {llvm::DenseMapInfo<Value>::getEmptyKey(),
            llvm::DenseMapInfo<Value>::getEmptyKey(),
            llvm::DenseMapInfo<Value>::getEmptyKey(),
            -1,
            -1,
            {}};
  }
  static LinearizeKey getTombstoneKey() {
    return {llvm::DenseMapInfo<Value>::getTombstoneKey(),
            llvm::DenseMapInfo<Value>::getTombstoneKey(),
            llvm::DenseMapInfo<Value>::getTombstoneKey(),
            -2,
            -2,
            {}};
  }
  static unsigned getHashValue(const LinearizeKey &k) {
    unsigned h =
        llvm::hash_combine(k.warp, k.thread, k.base, k.batchOff, k.outerOff);
    for (int64_t s : k.sizes)
      h = llvm::hash_combine(h, s);
    return h;
  }
  static bool isEqual(const LinearizeKey &a, const LinearizeKey &b) {
    return a == b;
  }
};
using LinearizeCache = llvm::DenseMap<LinearizeKey, Value, LinearizeKeyInfo>;

/// Compute the memory indices for one tile of a transfer_read/write.
/// \p cache is shared across all tile iterations of the same read/write op so
/// duplicate linearize_index ops (same warp/thread/base/offsets/sizes) are
/// emitted only once and reused.
static SmallVector<Value>
getTransferIndices(OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
                   NestedLayoutAttr layout, AffineMap permMap,
                   ArrayRef<Value> warpIndices, ArrayRef<Value> threadIndices,
                   LinearizeCache &cache) {
  int64_t rank = layout.getRank();
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerOffsets(offsets.begin() + rank, rank);

  SmallVector<Value> sliced(indices);
  for (auto [i, dim] : llvm::enumerate(permMap.getResults())) {
    if (isBroadcast(dim))
      continue;
    unsigned pos = cast<AffineDimExpr>(dim).getPosition();
    Value base = indices[pos];
    int64_t elems = layout.getElementTile()[i];
    Location loc = base.getLoc();
    SmallVector<int64_t, 5> sizes = {
        layout.getSubgroupTile()[i], layout.getBatchTile()[i],
        layout.getOuterTile()[i], layout.getThreadTile()[i], elems};
    LinearizeKey key{warpIndices[i],  threadIndices[i], base,
                     batchOffsets[i], outerOffsets[i],  sizes};
    auto [it, inserted] = cache.try_emplace(key, Value{});
    if (inserted) {
      Value batchConst = b.create<arith::ConstantIndexOp>(loc, batchOffsets[i]);
      Value outerConst = b.create<arith::ConstantIndexOp>(loc, outerOffsets[i]);
      SmallVector<Value> ids = {warpIndices[i], batchConst, outerConst,
                                threadIndices[i], base};
      bool disjoint = false;
      if (auto c = getConstantIntValue(base))
        disjoint = *c < elems;
      it->second =
          affine::AffineLinearizeIndexOp::create(b, loc, ids, sizes, disjoint);
    }
    sliced[pos] = it->second;
  }
  return sliced;
}

/// Extract one element-tile from the distributed vector.
static VectorValue extractSliceAsVector(RewriterBase &rewriter, Location loc,
                                        Value src, ArrayRef<int64_t> offsets) {
  Value slice = vector::ExtractOp::create(rewriter, loc, src, offsets);
  if (!isa<VectorType>(slice.getType())) {
    auto promoted = VectorType::get({}, getElementTypeOrSelf(slice));
    slice = vector::BroadcastOp::create(rewriter, loc, promoted, slice);
  }
  return cast<VectorValue>(slice);
}

/// Extract a sliced mask for one tile.
static VectorValue getSlicedPermutedMask(PatternRewriter &rewriter,
                                         Location loc,
                                         ArrayRef<int64_t> offsets,
                                         NestedLayoutAttr layout,
                                         VectorValue mask) {
  SmallVector<int64_t> maskOffsets =
      getDistributedTransferOffsets(offsets, layout);
  SmallVector<int64_t> strides(layout.getElementTile().size(), 1);
  return vector::ExtractStridedSliceOp::create(
      rewriter, loc, mask, maskOffsets, layout.getElementTile(), strides);
}

/// Compute warp and thread indices from a linear thread ID.
static LogicalResult
populateWarpAndThreadIndices(RewriterBase &rewriter, Value threadId,
                             int64_t subgroupSize, NestedLayoutAttr layout,
                             SmallVector<Value> &warpIndices,
                             SmallVector<Value> &threadIndices) {
  int64_t rank = layout.getRank();
  SmallVector<Value> ids =
      layout.computeThreadIds(threadId, subgroupSize, rewriter);
  if (ids.empty() && rank != 0)
    return failure();
  warpIndices = SmallVector<Value>(ids.begin(), ids.begin() + rank);
  threadIndices =
      SmallVector<Value>(ids.begin() + rank, ids.begin() + 2 * rank);
  return success();
}

/// Broadcast a value to a distributed shape, marking `broadcastedDims`.
static VectorValue broadcastToShape(RewriterBase &rewriter, Value source,
                                    ArrayRef<int64_t> shape,
                                    ArrayRef<bool> broadcastedDims) {
  assert(shape.size() == broadcastedDims.size());
  SmallVector<int64_t> bcastIdx, ubcastIdx;
  for (auto [i, b] : llvm::enumerate(broadcastedDims)) {
    if (b)
      bcastIdx.push_back(i);
    else
      ubcastIdx.push_back(i);
  }
  SmallVector<int64_t> perm =
      llvm::to_vector(llvm::concat<int64_t>(bcastIdx, ubcastIdx));
  SmallVector<int64_t> leadShape = applyPermutation(shape, perm);
  VectorType bcastTy = VectorType::get(leadShape, getElementTypeOrSelf(source));
  VectorValue bcast =
      vector::BroadcastOp::create(rewriter, source.getLoc(), bcastTy, source);
  SmallVector<int64_t> inv = invertPermutationVector(perm);
  if (isIdentityPermutation(inv))
    return bcast;
  return vector::TransposeOp::create(rewriter, source.getLoc(), bcast, inv);
}

/// Return the identity value for a combining kind.
static Value getCombiningIdentityValue(Location loc, OpBuilder &b,
                                       vector::CombiningKind kind,
                                       Type identityType) {
  auto vecTy = dyn_cast<VectorType>(identityType);
  Type elemTy = vecTy ? vecTy.getElementType() : identityType;
  TypedAttr attr;
  switch (kind) {
  case vector::CombiningKind::ADD:
    attr = b.getZeroAttr(elemTy);
    break;
  case vector::CombiningKind::MUL:
    if (elemTy.isIntOrIndex())
      attr = b.getIntegerAttr(elemTy, 1);
    else
      attr = b.getFloatAttr(elemTy, 1.0);
    break;
  case vector::CombiningKind::MINIMUMF:
  case vector::CombiningKind::MINNUMF: {
    auto inf = APFloat::getInf(cast<FloatType>(elemTy).getFloatSemantics());
    attr = b.getFloatAttr(elemTy, inf);
    break;
  }
  case vector::CombiningKind::MAXIMUMF:
  case vector::CombiningKind::MAXNUMF: {
    auto ninf = APFloat::getInf(cast<FloatType>(elemTy).getFloatSemantics(),
                                /*Negative=*/true);
    attr = b.getFloatAttr(elemTy, ninf);
    break;
  }
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXSI:
    attr = b.getIntegerAttr(elemTy, std::numeric_limits<int64_t>::min());
    break;
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINSI:
    attr = b.getIntegerAttr(elemTy, std::numeric_limits<int64_t>::max());
    break;
  default:
    attr = b.getZeroAttr(elemTy);
  }
  if (vecTy)
    attr = DenseElementsAttr::get(vecTy, attr);
  return arith::ConstantOp::create(b, loc, identityType, attr);
}

/// Map CombiningKind → gpu::AllReduceOperation.
static gpu::AllReduceOperation
combiningKindToAllReduce(vector::CombiningKind kind) {
  switch (kind) {
  case vector::CombiningKind::ADD:
    return gpu::AllReduceOperation::ADD;
  case vector::CombiningKind::MUL:
    return gpu::AllReduceOperation::MUL;
  case vector::CombiningKind::MINUI:
    return gpu::AllReduceOperation::MINUI;
  case vector::CombiningKind::MINSI:
    return gpu::AllReduceOperation::MINSI;
  case vector::CombiningKind::MAXUI:
    return gpu::AllReduceOperation::MAXUI;
  case vector::CombiningKind::MAXSI:
    return gpu::AllReduceOperation::MAXSI;
  case vector::CombiningKind::AND:
    return gpu::AllReduceOperation::AND;
  case vector::CombiningKind::OR:
    return gpu::AllReduceOperation::OR;
  case vector::CombiningKind::XOR:
    return gpu::AllReduceOperation::XOR;
  default:
    return gpu::AllReduceOperation::ADD;
  }
}

/// Return the deinterleaved, packed form of a distributed vector:
/// B1xB2xO1xO2xE1xE2 → B1xO1xE1 x B2xO2xE2 (packed).
static VectorValue getDeinterleavedPackedForm(PatternRewriter &rewriter,
                                              VectorValue val,
                                              NestedLayoutAttr layout) {
  Location loc = val.getLoc();
  int64_t rank = layout.getRank();
  SmallVector<int64_t> shape(rank * 3, 0);
  for (int64_t d : llvm::seq<int64_t>(rank)) {
    SmallVector<int64_t> packed = layout.getPackedShapeForUndistributedDim(d);
    shape[rank * 0 + d] = packed[1]; // batch
    shape[rank * 1 + d] = packed[2]; // outer
    shape[rank * 2 + d] = packed[4]; // element
  }
  auto iTy = VectorType::get(shape, val.getType().getElementType());
  auto iVal = vector::ShapeCastOp::create(rewriter, loc, iTy, val);
  // permute 0,1,2,...,2r-1 → 0,r,2r, 1,r+1,2r+1, ...
  SmallVector<int64_t> perm;
  for (int64_t d : llvm::seq<int64_t>(rank))
    for (int64_t t : llvm::seq<int64_t>(3))
      perm.push_back(t * rank + d);
  return vector::TransposeOp::create(rewriter, loc, iVal, perm);
}

/// Convert B1xB2xO1xO2xE1xE2 to [B1xO1xE1]x[B2xO2xE2].
static VectorValue getDeinterleavedUnpackedForm(PatternRewriter &rewriter,
                                                VectorValue val,
                                                NestedLayoutAttr layout) {
  Location loc = val.getLoc();
  auto packed = getDeinterleavedPackedForm(rewriter, val, layout);
  ArrayRef<int64_t> pshape = packed.getType().getShape();
  int64_t rank = layout.getRank();
  SmallVector<int64_t> unshape;
  unshape.reserve(rank);
  for (int64_t d : llvm::seq<int64_t>(rank)) {
    unshape.push_back(pshape[d * 3 + 0] * pshape[d * 3 + 1] *
                      pshape[d * 3 + 2]);
  }
  auto uTy = VectorType::get(unshape, packed.getType().getElementType());
  return vector::ShapeCastOp::create(rewriter, loc, uTy, packed);
}

/// Convert [B1xO1xE1]x[B2xO2xE2] back to B1xB2xO1xO2xE1xE2.
static VectorValue getInterleavedPackedForm(PatternRewriter &rewriter,
                                            VectorValue val,
                                            NestedLayoutAttr layout) {
  Location loc = val.getLoc();
  int64_t rank = layout.getRank();
  SmallVector<int64_t> shape;
  for (int64_t d : llvm::seq<int64_t>(rank)) {
    SmallVector<int64_t> packed = layout.getPackedShapeForUndistributedDim(d);
    shape.push_back(packed[1]); // batch
    shape.push_back(packed[2]); // outer
    shape.push_back(packed[4]); // element
  }
  auto niTy = VectorType::get(shape, val.getType().getElementType());
  auto niVal = vector::ShapeCastOp::create(rewriter, loc, niTy, val);
  // permute 0,1,2,...,2r-1 → 0,3,1,4,2,5,...
  SmallVector<int64_t> perm;
  for (int64_t t : llvm::seq<int64_t>(3))
    for (int64_t d : llvm::seq<int64_t>(rank))
      perm.push_back(t + 3 * d);
  return vector::TransposeOp::create(rewriter, loc, niVal, perm);
}

/// Project a vector: transpose so projected-out dims are leading, then extract.
static VectorValue projectVector(RewriterBase &rewriter, Location loc,
                                 VectorValue val, AffineMap projMap) {
  SmallVector<int64_t> remaining;
  llvm::SmallDenseSet<int64_t> slicedSet;
  for (int64_t d : llvm::seq<int64_t>(projMap.getNumDims()))
    slicedSet.insert(d);
  for (int64_t i : llvm::seq<int64_t>(projMap.getNumResults())) {
    int64_t p = projMap.getDimPosition(i);
    remaining.push_back(p);
    slicedSet.erase(p);
  }
  auto transposePerm = llvm::to_vector_of<int64_t>(slicedSet);
  transposePerm.append(remaining);
  auto transposed =
      vector::TransposeOp::create(rewriter, loc, val, transposePerm);
  SmallVector<int64_t> extractPos(slicedSet.size(), 0);
  auto sliced =
      vector::ExtractOp::create(rewriter, loc, transposed, extractPos);
  return cast<VectorValue>(sliced.getResult());
}

/// Correct port of IREE's basisFromSizesStrides
/// (iree/compiler/Utils/Indexing.cpp). Handles stride gaps by inserting
/// anonymous jump entries, reverses the basis to descending stride order, and
/// returns 1-indexed dimToResult values so that index 0 is always the overflow
/// (outermost) slot inserted by AffineDelinearizeIndexOp when
/// hasOuterBound=true.
static LogicalResult
basisFromSizesStrides(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                      SmallVectorImpl<int64_t> &basis,
                      SmallVectorImpl<size_t> &dimToResult) {
  assert(sizes.size() == strides.size());
  size_t numDims = sizes.size();
  basis.reserve(numDims);

  SmallVector<std::tuple<int64_t, int64_t, size_t>> terms =
      llvm::map_to_vector(llvm::enumerate(strides, sizes), [&](auto tuple) {
        auto [dim, stride, size] = tuple;
        return std::make_tuple(stride, size, dim);
      });
  llvm::sort(terms);

  int64_t previousSizes = 1;
  SmallVector<std::optional<size_t>> basisEntryToDim;
  basisEntryToDim.reserve(numDims);
  for (auto [stride, size, dim] : terms) {
    if (stride == 0) {
      stride = 1;
      size = 1;
    }
    if (stride % previousSizes != 0)
      return failure();

    if (stride != previousSizes) {
      int64_t jumpSize = stride / previousSizes;
      basisEntryToDim.push_back(std::nullopt);
      basis.push_back(jumpSize);
      previousSizes *= jumpSize;
    }

    basisEntryToDim.push_back(dim);
    basis.push_back(size);
    previousSizes *= size;
  }

  std::reverse(basis.begin(), basis.end());
  size_t basisLength = basis.size();
  dimToResult.assign(numDims, ~0u);
  for (auto [reverseBasisPos, dimPos] : llvm::enumerate(basisEntryToDim)) {
    if (!dimPos)
      continue;
    dimToResult[*dimPos] = basisLength - reverseBasisPos;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ldmatrix.x4 emission for the LHS (A-operand) read path.
//
// When a vector.transfer_read with NestedLayoutAttr feeds the LHS of an
// nvgpu.mma.sync (kind=tf32 16x8x8), and the source is workgroup memory and
// the per-warp coverage is exactly the ldmatrix.x4 unit (16 rows × 8 cols
// for f32 TF32), we can collapse the 64 per-element transfer_reads the
// ordinary distribution emits into batch_tile[0]*batch_tile[1] ldmatrix
// calls — one per (mBatch, kBatch) producing a vector<4x1xf32> A-fragment.
//
// The lane→(row,col) math is delegated to upstream's public NVGPU MMAUtils
// helpers (getLdMatrixParams + getLaneIdToLdMatrixMatrixCoord); this avoids
// re-deriving the PTX A-fragment layout. We construct a synthetic
// nvgpu::WarpMatrixInfo from Nova's NestedLayoutAttr instead of calling
// upstream's getWarpMatrixInfo (which requires a vector.contract user that
// Nova has already removed from this op's use-def chain by the time
// DistributeTransferRead runs).
//===----------------------------------------------------------------------===//

// True if the read's element type is f32 *and* its consumer chain ends at a
// vector.contract carrying nova.gpu.mma (kind tf32 16x8x8) with this read
// feeding the LHS. We trace through to_layout / insert / insert_strided_slice
// / shape_cast ops; the typical chain in Nova IR right before vector
// distribution runs is:
//   transfer_read → to_layout → vector.contract
// (Step 5.5 of NovaGPUVectorDistribute moves the mma_kind attribute from the
// to_layout onto the contract before this pattern fires, so the contract
// itself carries `nova.gpu.mma`.)
static bool isTf32MmaLhsRead(vector::TransferReadOp readOp) {
  Type elemTy = readOp.getVectorType().getElementType();
  if (!elemTy.isF32())
    return false;
  llvm::SmallPtrSet<Operation *, 8> seen;
  llvm::SmallVector<Value> stack = {readOp.getResult()};
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (auto contract = dyn_cast<vector::ContractionOp>(user)) {
        auto kindAttr = contract->getAttrOfType<IntegerAttr>("nova.gpu.mma");
        if (!kindAttr)
          continue;
        // tf32 m16n8k8 = NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8 (kind 5);
        // we keep the value-check string-free to avoid pulling in the enum.
        if (kindAttr.getInt() != 5)
          return false;
        if (contract.getLhs() == v)
          return true;
        return false; // feeds RHS or ACC — not what we handle here.
      }
      if (isa<ToLayoutOp, vector::InsertOp, vector::InsertStridedSliceOp,
              vector::ShapeCastOp>(user)) {
        for (Value r : user->getResults())
          stack.push_back(r);
      }
    }
  }
  return false;
}

// Redistribute the 4 f32 elements an ldmatrix.x4.b16 lands per thread so the
// result matches the NestedLayoutAttr-encoded LHS A-fragment for
// mma.sync.m16n8k8.tf32.
//
// Why this is needed:
//   ldmatrix.x4.b16 on a 16×8 f32 source delivers, per thread T = (T_hi=T/4,
//   T_lo=T%4):
//     r0 = A[T_hi,   T_lo  ]   r1 = A[T_hi+8, T_lo  ]
//     r2 = A[T_hi,   T_lo+4]   r3 = A[T_hi+8, T_lo+4]
//   → K-cols held per thread = { T_lo, T_lo+4 }
//
//   The rest of the Nova pipeline (setContractionAnchor → NestedLayout →
//   DistributeContract's {0,2,1,3} per-thread shuffle) is built around K-cols
//   per thread = { 2*T_lo, 2*T_lo+1 }. mma.sync invariance over K-partition
//   means matmul is correct *iff A and B agree on the partition*. The scalar
//   RHS fallback respects the NestedLayout partition, so we must convert
//   ldmatrix output into that same partition here. Otherwise any matmul that
//   mixes ldmatrix-A with scalar-B (e.g. backward passes where one side has a
//   non-identity permutation_map and is rejected from the fast path)
//   contracts mis-aligned K elements.
//
//   After this redistribution, per thread T:
//     n0 = A[T_hi,   2*T_lo  ]   n1 = A[T_hi,   2*T_lo+1]
//     n2 = A[T_hi+8, 2*T_lo  ]   n3 = A[T_hi+8, 2*T_lo+1]
//   stored in linear order [n0, n1, n2, n3] inside the returned vector<4x1xf32>,
//   matching the NestedLayout's (O0, O1, E0, E1) → (M_outer, K_outer, M_elem,
//   K_elem) traversal. The caller's shape_cast to vector<2x1x1x2xf32> then
//   places elements into the slots DistributeContract's {0,2,1,3} shuffle
//   expects, so mma.sync receives the standard PTX A-fragment.
//
// Cost: 8 gpu.shuffle idx + 4 selects per fragment. Each shuffle is 1 SASS
// instruction (shfl.sync.idx); the alternative scalar-load path issues 4
// memref.load per thread per fragment, so the redistribution is still a net
// win for SMEM bandwidth/issue-rate.
//
// Implementation: each row-quad (4 lanes sharing the same T_hi) holds the
// full row of A across its lanes. For an output column 2*T_lo we read from
// lane (quad_base + 2*(T_lo%2)); for 2*T_lo+1 we read from (quad_base +
// 2*(T_lo%2) + 1). On each source lane we issue two shuffles — one carrying
// r0/r1 (cols 0..3 of A), one carrying r2/r3 (cols 4..7 of A) — and the
// receiver picks via (T_lo < 2) which of the two it wanted.
static Value redistributeLdMatrixToNestedLayout(RewriterBase &rewriter,
                                                Location loc, Value ldmFrag) {
  MLIRContext *ctx = rewriter.getContext();
  Type f32 = Float32Type::get(ctx);
  Type i32 = rewriter.getI32Type();

  // Extract r0..r3 from the vector<4x1xf32> ldmatrix result.
  Value r0 = vector::ExtractOp::create(rewriter, loc, ldmFrag,
                                       ArrayRef<int64_t>{0, 0});
  Value r1 = vector::ExtractOp::create(rewriter, loc, ldmFrag,
                                       ArrayRef<int64_t>{1, 0});
  Value r2 = vector::ExtractOp::create(rewriter, loc, ldmFrag,
                                       ArrayRef<int64_t>{2, 0});
  Value r3 = vector::ExtractOp::create(rewriter, loc, ldmFrag,
                                       ArrayRef<int64_t>{3, 0});

  // lane-quad addressing.
  Value laneIdx =
      gpu::LaneIdOp::create(rewriter, loc, /*upperBound=*/IntegerAttr{});
  Value laneI32 =
      arith::IndexCastOp::create(rewriter, loc, i32, laneIdx);
  Value c1 = arith::ConstantOp::create(rewriter, loc, i32,
                                       rewriter.getI32IntegerAttr(1));
  Value c2 = arith::ConstantOp::create(rewriter, loc, i32,
                                       rewriter.getI32IntegerAttr(2));
  Value c3 = arith::ConstantOp::create(rewriter, loc, i32,
                                       rewriter.getI32IntegerAttr(3));
  Value cNeg4 = arith::ConstantOp::create(rewriter, loc, i32,
                                          rewriter.getI32IntegerAttr(~3));
  Value cWidth = arith::ConstantOp::create(rewriter, loc, i32,
                                           rewriter.getI32IntegerAttr(32));

  // quad_base = laneId & ~3
  Value quadBase = arith::AndIOp::create(rewriter, loc, laneI32, cNeg4);
  // T_lo = laneId & 3
  Value tLo = arith::AndIOp::create(rewriter, loc, laneI32, c3);
  // src_lane_for_2T_lo   = quad_base + 2 * (T_lo & 1)
  Value tLoLow = arith::AndIOp::create(rewriter, loc, tLo, c1);
  Value srcOffsetEven = arith::ShLIOp::create(rewriter, loc, tLoLow, c1);
  Value srcLaneEven = arith::AddIOp::create(rewriter, loc, quadBase,
                                            srcOffsetEven);
  // src_lane_for_2T_lo+1 = src_lane_for_2T_lo + 1
  Value srcLaneOdd = arith::AddIOp::create(rewriter, loc, srcLaneEven, c1);

  // is_high = T_lo >= 2  (selects r2/r3 contributions when true).
  Value isHigh = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                                       tLo, c2);

  auto emitShuffle = [&](Value v, Value srcLane) -> Value {
    auto shuf = gpu::ShuffleOp::create(rewriter, loc, v, srcLane, cWidth,
                                       gpu::ShuffleMode::IDX);
    return shuf.getShuffleResult();
  };

  // Row T_hi (cols 2*T_lo, 2*T_lo+1) — choose between r0 and r2 by T_lo<2.
  Value s_r0_even = emitShuffle(r0, srcLaneEven);
  Value s_r2_even = emitShuffle(r2, srcLaneEven);
  Value s_r0_odd  = emitShuffle(r0, srcLaneOdd);
  Value s_r2_odd  = emitShuffle(r2, srcLaneOdd);
  // Row T_hi+8 (cols 2*T_lo, 2*T_lo+1) — choose between r1 and r3 by T_lo<2.
  Value s_r1_even = emitShuffle(r1, srcLaneEven);
  Value s_r3_even = emitShuffle(r3, srcLaneEven);
  Value s_r1_odd  = emitShuffle(r1, srcLaneOdd);
  Value s_r3_odd  = emitShuffle(r3, srcLaneOdd);

  Value n0 = arith::SelectOp::create(rewriter, loc, isHigh, s_r2_even, s_r0_even);
  Value n1 = arith::SelectOp::create(rewriter, loc, isHigh, s_r2_odd,  s_r0_odd);
  Value n2 = arith::SelectOp::create(rewriter, loc, isHigh, s_r3_even, s_r1_even);
  Value n3 = arith::SelectOp::create(rewriter, loc, isHigh, s_r3_odd,  s_r1_odd);

  // Pack back into vector<4x1xf32> in NestedLayout linear order
  // [n0, n1, n2, n3] = [(M=T_hi,K=2T_lo), (M=T_hi,K=2T_lo+1),
  //                    (M=T_hi+8,K=2T_lo), (M=T_hi+8,K=2T_lo+1)].
  auto fragTy = VectorType::get({4, 1}, f32);
  Value out = arith::ConstantOp::create(rewriter, loc, fragTy,
                                        rewriter.getZeroAttr(fragTy));
  out = vector::InsertOp::create(rewriter, loc, n0, out,
                                 ArrayRef<int64_t>{0, 0});
  out = vector::InsertOp::create(rewriter, loc, n1, out,
                                 ArrayRef<int64_t>{1, 0});
  out = vector::InsertOp::create(rewriter, loc, n2, out,
                                 ArrayRef<int64_t>{2, 0});
  out = vector::InsertOp::create(rewriter, loc, n3, out,
                                 ArrayRef<int64_t>{3, 0});
  return out;
}

// Try to emit nvgpu.ldmatrix for one warp-level fragment.
// Returns the loaded vector<4x1xf32> on success, nullptr on failure.
//
// `baseIndices` are the original transfer_read indices, ALREADY translated
// by the (mBatch, kBatch) batch step (caller adjusts before calling).
static Value emitOneLdMatrixForLhsTf32(RewriterBase &rewriter, Location loc,
                                       Value srcMemref, ValueRange baseIndices,
                                       AffineMap permMap) {
  MLIRContext *ctx = rewriter.getContext();
  Type f32 = Float32Type::get(ctx);
  // Synthesise warp-matrix info: 16x8 f32 LHS.
  nvgpu::WarpMatrixInfo info;
  info.vectorType = VectorType::get({16, 8}, f32);
  info.operandRole = nvgpu::MatMulOperandRole::A;
  // Caller has filtered to non-transposed minor-identity reads.
  FailureOr<nvgpu::LdMatrixParams> params =
      nvgpu::getLdMatrixParams(info, /*transpose=*/false);
  if (failed(params))
    return nullptr;
  FailureOr<AffineMap> offsets =
      nvgpu::getLaneIdToLdMatrixMatrixCoord(rewriter, loc, *params);
  if (failed(offsets))
    return nullptr;

  Value laneId =
      gpu::LaneIdOp::create(rewriter, loc, /*upperBound=*/IntegerAttr{});

  // Compose laneId-derived (rowOff, colOff) onto the existing transfer
  // indices using the read's permutation map (same trick upstream uses in
  // getXferIndices).
  SmallVector<Value, 4> indices(baseIndices.begin(), baseIndices.end());
  unsigned offsetIdx = 0;
  for (auto expr : permMap.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim)
      continue;
    Value prev = indices[dim.getPosition()];
    SmallVector<OpFoldResult, 2> dims = {laneId, prev};
    AffineExpr d0 = rewriter.getAffineDimExpr(offsets->getNumDims());
    indices[dim.getPosition()] = affine::makeComposedAffineApply(
        rewriter, loc, d0 + offsets->getResult(offsetIdx++), dims);
  }

  // For TF32 m16n8k8 LHS the per-thread fragment is vector<4x1xf32>.
  auto fragTy = VectorType::get({4, 1}, f32);
  auto ld = nvgpu::LdMatrixOp::create(rewriter, loc, fragTy, srcMemref, indices,
                                      /*transpose=*/false,
                                      /*numTiles=*/4);

  // ldmatrix output is in K-partition {T_lo, T_lo+4} per thread; redistribute
  // to the NestedLayout-encoded {2*T_lo, 2*T_lo+1} partition so it composes
  // with the rest of the pipeline (including the scalar RHS fallback used by
  // backward-pass matmuls with non-identity permutation_maps).
  return redistributeLdMatrixToNestedLayout(rewriter, loc, ld.getResult());
}

// Top-level helper invoked from DistributeTransferRead. Returns a fully
// populated distributed accumulator on success, nullptr to fall through.
//
// `distShape` and `tileShape` are the same shapes the existing per-tile loop
// would iterate over. The distributed type is vector<distShape x f32>; we
// match it by inserting each ldmatrix's vector<4x1xf32> result at the
// (mBatch, kBatch, 0, 0, ...) offset that the existing per-tile loop would
// have used for the same fragment's first element.
static Value
tryEmitLdMatrixForLhs(RewriterBase &rewriter, vector::TransferReadOp readOp,
                      NestedLayoutAttr layout, ArrayRef<Value> warpIdx,
                      ArrayRef<Value> threadIdx, ArrayRef<int64_t> distShape,
                      ArrayRef<int64_t> tileShape) {
  // ldmatrix is warp-collective; per-thread (`threadIdx`) and per-warp
  // (`warpIdx`) lane fan-out is handled below via gpu.lane_id and the
  // upstream lane→fragment affine map. The args are accepted for symmetry
  // with the per-tile path.
  (void)warpIdx;
  (void)threadIdx;
  (void)tileShape;
  // ── eligibility ─────────────────────────────────────────────────────────
  //
  // ldmatrix.x4.b16 on 16×8 f32 delivers per-thread K-cols { T_lo, T_lo+4 },
  // which is NOT the partition the NestedLayout / DistributeContract pipeline
  // is built around ({ 2*T_lo, 2*T_lo+1 }). Both partitions are correct for
  // mma.sync.m16n8k8.tf32 *in isolation* (the row-quad sum is invariant to
  // K-partition), but mixing them across A and B contracts mis-aligned K
  // pairs — which is what made the scalar-RHS fallback (used by backward
  // matmuls whose LHS has a non-identity permutation_map) silently corrupt
  // training in the prior patch.
  //
  // Fix: emitOneLdMatrixForLhsTf32 redistributes ldmatrix output into the
  // NestedLayout partition via 8 gpu.shuffle idx + 4 selects, so this fast
  // path is layout-compatible with the rest of the pipeline. The
  // {0,2,1,3} per-thread shuffle in DistributeContract still fires (we
  // don't set nova.gpu.lhs_from_ldmatrix any more) to convert the
  // NestedLayout linear order into mma.sync's PTX A-fragment register
  // order.
  if (!isTf32MmaLhsRead(readOp)) {
    LLVM_DEBUG(llvm::dbgs() << "[ldmatrix] reject: not tf32 mma LHS read\n");
    return nullptr;
  }

  auto srcTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!srcTy) {
    LLVM_DEBUG(llvm::dbgs() << "[ldmatrix] reject: source not a memref\n");
    return nullptr;
  }
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(srcTy.getMemorySpace());
  if (!space ||
      space.getValue() != gpu::GPUDialect::getWorkgroupAddressSpace()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[ldmatrix] reject: source not workgroup memref\n");
    return nullptr;
  }

  // Innermost stride must be 1 (contiguous columns). Strided memrefs with a
  // non-unit innermost stride aren't ldmatrix-compatible.
  if (auto strided = dyn_cast<StridedLayoutAttr>(srcTy.getLayout())) {
    if (!strided.getStrides().empty() && strided.getStrides().back() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "[ldmatrix] reject: innermost stride != 1\n");
      return nullptr;
    }
  }
  if (!readOp.getPermutationMap().isMinorIdentity()) {
    LLVM_DEBUG(llvm::dbgs() << "[ldmatrix] reject: non-identity permutation\n");
    return nullptr;
  }

  // Per-warp coverage of the read along (M, K) must be exactly 16 × 8 for
  // the TF32 m16n8k8 ldmatrix.x4 unit. (subgroup_tile is per-workgroup and
  // we treat one warp == one subgroup.)  Rank must be exactly 2 so the
  // (B0,B1,O0,O1,E0,E1) distributed-shape layout is well-defined.
  if (layout.getRank() != 2)
    return nullptr;
  ArrayRef<int64_t> outerT = layout.getOuterTile();
  ArrayRef<int64_t> threadT = layout.getThreadTile();
  ArrayRef<int64_t> elementT = layout.getElementTile();
  ArrayRef<int64_t> sgT = layout.getSubgroupTile();
  if (outerT.size() < 2 || threadT.size() < 2 || elementT.size() < 2)
    return nullptr;
  // Subgroup_tile != [1,1] would mean the layout itself splits the read
  // across multiple warps; the warp scf.forall above is then NOT the only
  // source of per-warp offsets, and we'd have to fold warpIdx into the
  // ldmatrix base. Restrict to the common Nova case where one warp==one
  // subgroup and warp partitioning is handled by the enclosing forall.
  if (!sgT.empty() && sgT.size() >= 2 && (sgT[0] != 1 || sgT[1] != 1))
    return nullptr;
  int64_t perWarpM = outerT[0] * threadT[0] * elementT[0];
  int64_t perWarpK = outerT[1] * threadT[1] * elementT[1];
  if (perWarpM != 16 || perWarpK != 8)
    return nullptr;
  if (distShape.size() != 6)
    return nullptr;

  // ── emit ────────────────────────────────────────────────────────────────
  Type f32 = Float32Type::get(rewriter.getContext());
  Location loc = readOp.getLoc();

  // Distributed result vector: same shape the per-tile path would build.
  auto distTy = VectorType::get(distShape, f32);
  Value zero = arith::ConstantOp::create(rewriter, loc, distTy,
                                         rewriter.getZeroAttr(distTy));
  Value acc = zero;

  ArrayRef<int64_t> batchT = layout.getBatchTile();
  if (batchT.size() < 2)
    return nullptr;
  int64_t mBatches = batchT[0];
  int64_t kBatches = batchT[1];

  // ldmatrix is warp-COLLECTIVE: each lane provides an address but only one
  // address per row of the source tile is actually used (8 rows × 4 tiles
  // = 32 lanes for ldmatrix.x4). We must therefore pass the *warp-tile
  // origin* as the base — NOT a per-thread fan-out. The original
  // readOp.getIndices() already point at the warp's sub-tile origin
  // (materialised by the warp scf.forall body before this pattern runs);
  // we just advance them by (mb * 16, kb * 8) per fragment.
  //
  // Using `getTransferIndices` here would fold in the layout's per-thread
  // (thread_tile) offsets — that double-counts the lane and produces
  // misaligned addresses on most lanes (each lane in a 4-thread group of
  // the ldmatrix unit would request a row that is per-thread shifted, but
  // ldmatrix already shifts by lane%16 internally). The result is a
  // CUDA_ERROR_MISALIGNED_ADDRESS at runtime.
  // `vector.insert_strided_slice` strides must match the SOURCE rank, not
  // the layout rank. Our source per fragment is vector<O0xO1xE0xE1xf32>
  // (rank = 2 * layout.getRank() = 4 for rank-2 layouts).
  SmallVector<int64_t> strides(2 * layout.getRank(), 1);
  for (int64_t mb = 0; mb < mBatches; ++mb) {
    for (int64_t kb = 0; kb < kBatches; ++kb) {
      // Build warp-tile origin for this fragment by advancing the read's
      // base indices: row += mb * 16, col += kb * 8.
      SmallVector<Value> baseIndices(readOp.getIndices().begin(),
                                     readOp.getIndices().end());
      // Permutation map tells us which dim is row (M) vs col (K). For a
      // minor-identity 2-D read the last two indices are (row, col).
      AffineMap permMap = readOp.getPermutationMap();
      assert(permMap.getNumResults() == 2 &&
             "expected rank-2 transfer_read for tf32 mma LHS");
      // Position within `indices` of M (row) and K (col):
      auto rowDim = cast<AffineDimExpr>(permMap.getResult(0)).getPosition();
      auto colDim = cast<AffineDimExpr>(permMap.getResult(1)).getPosition();
      if (mb != 0) {
        Value mAdv = rewriter.create<arith::ConstantIndexOp>(loc, mb * 16);
        baseIndices[rowDim] =
            rewriter.create<arith::AddIOp>(loc, baseIndices[rowDim], mAdv);
      }
      if (kb != 0) {
        Value kAdv = rewriter.create<arith::ConstantIndexOp>(loc, kb * 8);
        baseIndices[colDim] =
            rewriter.create<arith::AddIOp>(loc, baseIndices[colDim], kAdv);
      }
      Value frag =
          emitOneLdMatrixForLhsTf32(rewriter, loc, readOp.getBase(),
                                    baseIndices, readOp.getPermutationMap());
      if (!frag)
        return nullptr;

      // Insert the vector<4x1xf32> fragment into `acc` at the position the
      // existing distribution would have placed it — namely
      // [mBatch, kBatch, outer0=0, outer1=0, elem0, elem1] when written in
      // (B0,B1,O0,O1,E0,E1) layout. We populate the (E0,E1) sub-block
      // wholesale: one strided insert per fragment.
      SmallVector<int64_t> insertOffsets;
      insertOffsets.push_back(mb);
      insertOffsets.push_back(kb);
      for (size_t i = 2; i < distShape.size(); ++i)
        insertOffsets.push_back(0);
      // distShape order: B0,B1,O0,O1,E0,E1 (per layout encoding). Insert
      // vector<4x1> at offsets [mb, kb, 0, 0, 0, 0] with shape strides=1.
      // The fragment occupies E0=4, E1=1 — but the distributed shape's
      // last two dims may differ in size from (4,1); we instead reshape
      // the fragment to match the (O0,O1,E0,E1) shape this layout encodes
      // for one batch.
      // Reshape vector<4x1xf32> → vector<O0xO1xE0xE1xf32>.
      // For LHS layout outer=[2,1] element=[1,2] this is vector<2x1x1x2xf32>;
      // the 4 lanes of the fragment are laid out as O0=2, E1=2.
      // We rely on element-count match (4) and a plain shape_cast.
      int64_t outerProd = outerT[0] * outerT[1];
      int64_t elemProd = elementT[0] * elementT[1];
      if (outerProd * elemProd != 4) {
        // Layout doesn't carry exactly 4 per-thread elements per fragment
        // (e.g. an unexpected element_tile). Fall through.
        return nullptr;
      }
      auto subTy = VectorType::get(
          {outerT[0], outerT[1], elementT[0], elementT[1]}, f32);
      Value reshaped = vector::ShapeCastOp::create(rewriter, loc, subTy, frag);

      acc = vector::InsertStridedSliceOp::create(rewriter, loc, reshaped, acc,
                                                 insertOffsets, strides);
    }
  }
  return acc;
}

namespace {

// Check if this transfer_read produces the RHS (B matrix) for a TF32 mma.sync.
static bool isTf32MmaRhsRead(vector::TransferReadOp readOp) {
  llvm::SmallPtrSet<Operation *, 4> seen;
  llvm::SmallVector<Value> stack = {readOp.getResult()};
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (auto contract = dyn_cast<vector::ContractionOp>(user)) {
        auto kindAttr = contract->getAttrOfType<IntegerAttr>("nova.gpu.mma");
        if (!kindAttr || kindAttr.getInt() != 5) // 5 = MMA_SYNC_TF32_16x8x8
          continue;
        if (contract.getRhs() == v)
          return true;
      }
      if (isa<ToLayoutOp, vector::InsertOp, vector::InsertStridedSliceOp,
              vector::ShapeCastOp>(user))
        for (Value r : user->getResults())
          stack.push_back(r);
    }
  }
  return false;
}

// Emits two scalar loads per fragment for the TF32 mma.sync B matrix (8×8).
//
// PTX B-fragment layout for m16n8k8.tf32: thread T holds
//   b0 = B[2*(T%4)  ][T/4]   (K-col = 2*T_lo,   N-row = T_hi)
//   b1 = B[2*(T%4)+1][T/4]   (K-col = 2*T_lo+1, N-row = T_hi)
// This is exactly the NestedLayout K-partition {2*T_lo, 2*T_lo+1}, so
// NVIDIADistributeContract passes the B-slice directly to nvgpu.mma.sync
// WITHOUT any register-level shuffle (cf. the {0,2,1,3} shuffle for A).
//
// Prior code incorrectly used {T%4, T%4+4} — matching the ldmatrix.x4 A
// partition rather than the PTX B-fragment layout — which caused K-element
// mismatch and ~0.4–1% relative error in mma.sync output.
//
// Eligibility: minor-identity permutation map only (B stored K×N, contiguous
// K dimension). Backward reads with non-identity perm fall through to the
// generic scalar fallback (getTransferIndices), which is always correct.
static Value
tryEmitTf32RhsLoad(RewriterBase &rewriter, vector::TransferReadOp readOp,
                   NestedLayoutAttr layout, ArrayRef<Value> warpIdx,
                   ArrayRef<Value> threadIdx, ArrayRef<int64_t> distShape,
                   ArrayRef<int64_t> tileShape) {
  if (!isTf32MmaRhsRead(readOp))
    return nullptr;

  auto srcTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!srcTy)
    return nullptr;
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(srcTy.getMemorySpace());
  if (!space || space.getValue() != gpu::GPUDialect::getWorkgroupAddressSpace())
    return nullptr;

  // Only fire for forward reads (B stored K×N, contiguous K). Backward reads
  // with transposed permutation maps use the generic scalar fallback.
  if (!readOp.getPermutationMap().isMinorIdentity()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[rhs-load] reject: non-identity permutation\n");
    return nullptr;
  }

  if (layout.getRank() != 2)
    return nullptr;
  ArrayRef<int64_t> outerT = layout.getOuterTile();
  ArrayRef<int64_t> threadT = layout.getThreadTile();
  ArrayRef<int64_t> elementT = layout.getElementTile();
  ArrayRef<int64_t> sgT = layout.getSubgroupTile();
  if (outerT.size() < 2 || threadT.size() < 2 || elementT.size() < 2)
    return nullptr;
  if (!sgT.empty() && sgT.size() >= 2 && (sgT[0] != 1 || sgT[1] != 1))
    return nullptr;

  // Verify it is an 8x8 tile
  int64_t perWarpK = outerT[0] * threadT[0] * elementT[0];
  int64_t perWarpN = outerT[1] * threadT[1] * elementT[1];
  if (perWarpK != 8 || perWarpN != 8)
    return nullptr;
  if (distShape.size() != 6)
    return nullptr;

  Location loc = readOp.getLoc();
  Type elemTy = srcTy.getElementType();
  auto vecType = VectorType::get(distShape, elemTy);
  Value zero = arith::ConstantOp::create(rewriter, loc, vecType,
                                         rewriter.getZeroAttr(vecType));
  VectorValue acc = cast<VectorValue>(zero);
  AffineMap permMap = readOp.getPermutationMap();
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, IntegerAttr{});
  AffineExpr d0 = rewriter.getAffineDimExpr(0);
  // K-offsets matching the PTX B-fragment layout: {2*(T%4), 2*(T%4)+1}.
  Value twoLaneMod4 =
      affine::makeComposedAffineApply(rewriter, loc, (d0 % 4) * 2, {laneId});
  Value twoLaneMod4Plus1 =
      affine::makeComposedAffineApply(rewriter, loc, (d0 % 4) * 2 + 1, {laneId});
  Value laneDiv4 =
      affine::makeComposedAffineApply(rewriter, loc, d0.floorDiv(4), {laneId});

  for (auto [idx, offsets] :
       llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {

    SmallVector<Value> loopIndices(readOp.getIndices().begin(),
                                   readOp.getIndices().end());
    unsigned offsetIdx = 0;
    for (auto expr : permMap.getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim) {
        offsetIdx++;
        continue;
      }
      unsigned pos = dim.getPosition();
      // offsets[offsetIdx] is the batch index. Each batch is 8 elements.
      if (offsets[offsetIdx] != 0) {
        Value adv = rewriter.create<arith::ConstantIndexOp>(
            loc, offsets[offsetIdx] * 8);
        loopIndices[pos] =
            rewriter.create<arith::AddIOp>(loc, loopIndices[pos], adv);
      }
      offsetIdx++;
    }

    auto loadElement = [&](Value kLaneOff, Value nLaneOff) -> Value {
      SmallVector<Value, 4> indices(loopIndices.begin(), loopIndices.end());
      unsigned offsetIdx2 = 0;
      for (auto expr : permMap.getResults()) {
        auto dim = dyn_cast<AffineDimExpr>(expr);
        if (!dim) {
          offsetIdx2++;
          continue;
        }
        Value prev = indices[dim.getPosition()];
        Value off = (offsetIdx2 == 0) ? kLaneOff : nLaneOff;

        AffineExpr d1 = rewriter.getAffineDimExpr(0);
        AffineExpr d2 = rewriter.getAffineDimExpr(1);
        indices[dim.getPosition()] = affine::makeComposedAffineApply(
            rewriter, loc, d1 + d2, {prev, off});
        offsetIdx2++;
      }
      return rewriter.create<memref::LoadOp>(loc, readOp.getBase(), indices);
    };

    Value elem0 = loadElement(twoLaneMod4, laneDiv4);
    Value elem1 = loadElement(twoLaneMod4Plus1, laneDiv4);

    auto fragTy = VectorType::get({2, 1}, elemTy);
    Value frag = rewriter.create<arith::ConstantOp>(
        loc, fragTy, rewriter.getZeroAttr(fragTy));
    frag = rewriter.create<vector::InsertOp>(loc, elem0, frag,
                                             ArrayRef<int64_t>{0, 0});
    frag = rewriter.create<vector::InsertOp>(loc, elem1, frag,
                                             ArrayRef<int64_t>{1, 0});

    // Sub-tile type is vector<O0xO1x2x1xf32> (which is what tileShape requires)
    auto subTy = VectorType::get(
        {distShape[2], distShape[3], distShape[4], distShape[5]}, elemTy);
    Value reshaped = vector::ShapeCastOp::create(rewriter, loc, subTy, frag);

    SmallVector<int64_t> insertOffsets = {offsets[0], offsets[1], 0, 0, 0, 0};
    SmallVector<int64_t> stridesVec(distShape.size(), 1);
    acc = vector::InsertStridedSliceOp::create(rewriter, loc, reshaped, acc,
                                               insertOffsets, stridesVec);
  }

  return acc;
}

//===----------------------------------------------------------------------===//
// DistributeTransferRead
//===----------------------------------------------------------------------===//

struct DistributeTransferRead final
    : OpDistributionPattern<vector::TransferReadOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferRead(MLIRContext *ctx,
                         const llvm::DenseMap<Block *, Value> &threadIdMap,
                         int64_t subgroupSize)
      : OpDistributionPattern(ctx), threadIdMap(threadIdMap),
        subgroupSize(subgroupSize) {}

  // Walk up from op to find the enclosing warp forall body block.
  Value getThreadId(Operation *op) const {
    Block *blk = op->getBlock();
    while (blk) {
      auto it = threadIdMap.find(blk);
      if (it != threadIdMap.end())
        return it->second;
      Operation *parent = blk->getParentOp();
      if (!parent)
        break;
      blk = parent->getBlock();
    }
    return {};
  }

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto layout = dyn_cast<NestedLayoutAttr>(signature[readOp.getResult()]);
    if (!layout)
      return rewriter.notifyMatchFailure(readOp, "non-nested layout");
    if (!isa<MemRefType>(readOp.getBase().getType()))
      return rewriter.notifyMatchFailure(readOp,
                                         "distribution expects memrefs");

    Value threadId = getThreadId(readOp);
    if (!threadId)
      return rewriter.notifyMatchFailure(readOp,
                                         "no thread id for this forall");

    VectorValue mask = readOp.getMask();
    NestedLayoutAttr maskLayout;
    if (mask) {
      maskLayout = dyn_cast<NestedLayoutAttr>(signature[mask]);
      if (!maskLayout)
        return rewriter.notifyMatchFailure(readOp, "non-nested mask layout");
      mask = getDistributed(rewriter, mask, maskLayout);
      mask = getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
    }

    SmallVector<int64_t> distShape = layout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(layout);
    int64_t rank = layout.getRank();
    Type elemTy = readOp.getBase().getType().getElementType();

    auto vecType = VectorType::get(distShape, elemTy);
    auto innerTy = VectorType::get(layout.getElementTile(), elemTy);
    Value zero = arith::ConstantOp::create(rewriter, readOp.getLoc(), vecType,
                                           rewriter.getZeroAttr(vecType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIdx, threadIdx;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            layout, warpIdx, threadIdx)))
      return rewriter.notifyMatchFailure(readOp,
                                         "failed to compute thread ids");

    // Try the ldmatrix.x4 fast path for TF32 m16n8k8 LHS reads from SMEM.
    // On success this replaces the entire per-tile transfer_read chain with
    // batch_tile[0]*batch_tile[1] nvgpu.ldmatrix calls.  See
    // tryEmitLdMatrixForLhs above for eligibility.
    if (!mask) {
      if (Value lifted =
              tryEmitLdMatrixForLhs(rewriter, readOp, layout, warpIdx,
                                    threadIdx, distShape, tileShape)) {
        // The ldmatrix path now redistributes output into NestedLayout-
        // encoded linear order (see redistributeLdMatrixToNestedLayout), so
        // the consuming contract's {0,2,1,3} per-thread shuffle MUST still
        // fire — we do NOT set nova.gpu.lhs_from_ldmatrix here. The shuffle
        // converts (M_outer, M_elem, K_outer, K_elem) traversal order into
        // the PTX A-fragment register order mma.sync.m16n8k8.tf32 consumes.
        replaceOpWithDistributedValues(rewriter, readOp, lifted);
        return success();
      }
      // RHS explicit-load path for TF32.
      if (auto res = tryEmitTf32RhsLoad(rewriter, readOp, layout, warpIdx,
                                        threadIdx, distShape, tileShape)) {
        replaceOpWithDistributedValues(rewriter, readOp, res);
        return success();
      }
    }

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    if (mask) {
      SmallVector<int64_t> mds = maskLayout.getDistributedShape();
      SmallVector<int64_t> mts = getElementVectorTileShape(maskLayout);
      allMaskOffsets = llvm::to_vector(StaticTileOffsetRange(mds, mts));
    }

    SmallVector<int64_t> strides(rank, 1);
    LinearizeCache linCache;
    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndices(
          rewriter, readOp.getIndices(), offsets, layout,
          readOp.getPermutationMap(), warpIdx, threadIdx, linCache);

      VectorValue slicedMask = nullptr;
      if (mask) {
        auto mds = maskLayout.getDistributedShape();
        auto mts = getElementVectorTileShape(maskLayout);
        slicedMask = getSlicedPermutedMask(
            rewriter, readOp.getLoc(), allMaskOffsets[idx], maskLayout, mask);
      }

      VectorValue tile = vector::TransferReadOp::create(
          rewriter, readOp.getLoc(), innerTy, readOp.getBase(), slicedIndices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), slicedMask,
          readOp.getInBoundsAttr());

      if (acc.getType().getRank() == 0) {
        acc = tile;
      } else {
        acc = vector::InsertStridedSliceOp::create(rewriter, readOp.getLoc(),
                                                   tile, acc, offsets, strides);
      }
    }
    replaceOpWithDistributedValues(rewriter, readOp, acc);
    return success();
  }

  const llvm::DenseMap<Block *, Value> &threadIdMap;
  int64_t subgroupSize;
};

//===----------------------------------------------------------------------===//
// DistributeTransferWrite
//===----------------------------------------------------------------------===//

struct DistributeTransferWrite final
    : OpDistributionPattern<vector::TransferWriteOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferWrite(MLIRContext *ctx,
                          const llvm::DenseMap<Block *, Value> &threadIdMap,
                          int64_t subgroupSize, ArrayRef<int64_t> workgroupSize)
      : OpDistributionPattern(ctx), threadIdMap(threadIdMap),
        subgroupSize(subgroupSize) {
    if (!workgroupSize.empty())
      numThreads = std::accumulate(workgroupSize.begin(), workgroupSize.end(),
                                   int64_t(1), std::multiplies<int64_t>());
  }

  // Walk up from op to find the enclosing warp forall body block.
  Value getThreadId(Operation *op) const {
    Block *blk = op->getBlock();
    while (blk) {
      auto it = threadIdMap.find(blk);
      if (it != threadIdMap.end())
        return it->second;
      Operation *parent = blk->getParentOp();
      if (!parent)
        break;
      blk = parent->getBlock();
    }
    return {};
  }

  /// Returns a boolean Value that is true only for threads that should write
  /// (avoids duplicate writes when broadcast dims exist).
  FailureOr<Value> getNoOverlapCondition(OpBuilder &b, Location loc,
                                         Value threadId,
                                         NestedLayoutAttr layout) const {
    ArrayRef<int64_t> threadTile = layout.getThreadTile();
    ArrayRef<int64_t> threadStrides = layout.getThreadStrides();
    ArrayRef<int64_t> subgroupTile = layout.getSubgroupTile();
    SmallVector<int64_t> subgroupStrides =
        llvm::map_to_vector(layout.getSubgroupStrides(),
                            [&](int64_t x) { return x * subgroupSize; });
    auto concatTiles =
        llvm::to_vector(llvm::concat<const int64_t>(subgroupTile, threadTile));
    auto concatStrides = llvm::to_vector(
        llvm::concat<const int64_t>(subgroupStrides, threadStrides));

    SmallVector<int64_t> basis;
    SmallVector<size_t> dimToResult;
    if (failed(basisFromSizesStrides(concatTiles, concatStrides, basis,
                                     dimToResult)))
      return failure();
    if (numThreads.has_value()) {
      int64_t outer = numThreads.value() /
                      std::accumulate(basis.begin(), basis.end(), int64_t(1),
                                      std::multiplies<int64_t>());
      basis.insert(basis.begin(), outer);
    }
    SmallVector<Value> delinearized;
    b.createOrFold<affine::AffineDelinearizeIndexOp>(
        delinearized, loc, threadId, basis, numThreads.has_value());
    Value cond = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);
    for (auto [i, v] : llvm::enumerate(delinearized)) {
      if (llvm::is_contained(dimToResult, i))
        continue;
      Value isZero =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, v,
                                  b.create<arith::ConstantIndexOp>(loc, 0));
      cond = b.create<arith::AndIOp>(loc, cond, isZero);
    }
    return cond;
  }

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto layout =
        dyn_cast<NestedLayoutAttr>(signature[writeOp.getValueToStore()]);
    if (!layout)
      return rewriter.notifyMatchFailure(writeOp, "non-nested layout");
    if (!isa<MemRefType>(writeOp.getBase().getType()))
      return rewriter.notifyMatchFailure(writeOp,
                                         "distribution expects memrefs");

    Value threadId = getThreadId(writeOp);
    if (!threadId)
      return rewriter.notifyMatchFailure(writeOp,
                                         "no thread id for this forall");

    VectorValue mask = writeOp.getMask();
    NestedLayoutAttr maskLayout;
    if (mask) {
      maskLayout = dyn_cast<NestedLayoutAttr>(signature[mask]);
      if (!maskLayout)
        return rewriter.notifyMatchFailure(writeOp, "non-nested mask layout");
      mask = getDistributed(rewriter, mask, maskLayout);
      mask = getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
    }

    SmallVector<int64_t> distShape = layout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(layout);
    int64_t rank = layout.getRank();

    SmallVector<Value> warpIdx, threadIdx;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            layout, warpIdx, threadIdx)))
      return rewriter.notifyMatchFailure(writeOp,
                                         "failed to compute thread ids");

    Location loc = writeOp.getLoc();
    FailureOr<Value> doWrite =
        getNoOverlapCondition(rewriter, loc, threadId, layout);
    if (failed(doWrite))
      return rewriter.notifyMatchFailure(writeOp,
                                         "failed no-overlap condition");

    auto ifOp = scf::IfOp::create(rewriter, loc, doWrite.value());
    rewriter.setInsertionPoint(ifOp.thenYield());

    Value distVec = getDistributed(rewriter, writeOp.getValueToStore(), layout);

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    if (mask) {
      auto mds = maskLayout.getDistributedShape();
      auto mts = getElementVectorTileShape(maskLayout);
      allMaskOffsets = llvm::to_vector(StaticTileOffsetRange(mds, mts));
    }

    LinearizeCache linCache;
    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndices(
          rewriter, writeOp.getIndices(), offsets, layout,
          writeOp.getPermutationMap(), warpIdx, threadIdx, linCache);
      ArrayRef<int64_t> oArr(offsets);
      VectorValue tile = extractSliceAsVector(rewriter, loc, distVec,
                                              oArr.take_front(rank * 2));

      VectorValue slicedMask = nullptr;
      if (mask)
        slicedMask = getSlicedPermutedMask(rewriter, loc, allMaskOffsets[idx],
                                           maskLayout, mask);

      vector::TransferWriteOp::create(rewriter, loc, tile, writeOp.getBase(),
                                      slicedIndices,
                                      writeOp.getPermutationMapAttr(),
                                      slicedMask, writeOp.getInBoundsAttr());
    }

    rewriter.eraseOp(writeOp);
    return success();
  }

  const llvm::DenseMap<Block *, Value> &threadIdMap;
  int64_t subgroupSize;
  std::optional<int64_t> numThreads;
};

//===----------------------------------------------------------------------===//
// DistributeBroadcast
//===----------------------------------------------------------------------===//

struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto dstVec = broadcastOp.getVector();
    auto layout = dyn_cast<NestedLayoutAttr>(signature[dstVec]);
    if (!layout)
      return rewriter.notifyMatchFailure(broadcastOp, "non-nested layout");

    SmallVector<int64_t> distShape = layout.getDistributedShape();
    SmallVector<bool> broadcastedDims(distShape.size(), false);

    VectorValue srcVec = dyn_cast<VectorValue>(broadcastOp.getSource());
    int64_t bcastRank = layout.getRank();
    if (srcVec)
      bcastRank -= srcVec.getType().getRank();

    int64_t rank = layout.getRank();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < bcastRank; ++j)
        broadcastedDims[j + i * rank] = true;

    Value distSrc = broadcastOp.getSource();
    if (srcVec) {
      auto srcLayout = dyn_cast<NestedLayoutAttr>(signature[srcVec]);
      if (!srcLayout)
        return rewriter.notifyMatchFailure(broadcastOp,
                                           "non-nested src layout");
      distSrc = getDistributed(rewriter, srcVec, srcLayout);
    }

    VectorValue result =
        broadcastToShape(rewriter, distSrc, distShape, broadcastedDims);
    replaceOpWithDistributedValues(rewriter, broadcastOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeMultiReduction
//===----------------------------------------------------------------------===//

struct DistributeMultiReduction final
    : OpDistributionPattern<vector::MultiDimReductionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeMultiReduction(MLIRContext *ctx, int64_t subgroupSize,
                           int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : OpDistributionPattern(ctx, benefit), subgroupSize(subgroupSize),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  /// Perform inter-thread butterfly reduction on a flat vector.
  FailureOr<VectorValue> doThreadReduction(RewriterBase &rewriter,
                                           NestedLayoutAttr layout,
                                           VectorValue flat,
                                           vector::CombiningKind kind,
                                           ArrayRef<bool> reductionMask) const {
    VectorType flatTy = flat.getType();
    int64_t n = flatTy.getNumElements();
    Location loc = flat.getLoc();
    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(flatTy));
    auto res = cast<VectorValue>(zero.getResult());
    for (int64_t i = 0; i < n; ++i) {
      Value elem = vector::ExtractOp::create(rewriter, loc, flat, i);
      for (auto [d, isReduced] : llvm::enumerate(reductionMask)) {
        if (!isReduced)
          continue;
        int64_t offset = layout.getThreadStrides()[d];
        int64_t width = layout.getThreadTile()[d];
        if (offset == 0 || width <= 1)
          continue;
        elem = gpu::SubgroupReduceOp::create(
            rewriter, loc, elem, combiningKindToAllReduce(kind),
            /*uniform=*/false, static_cast<uint32_t>(width),
            static_cast<uint32_t>(offset));
      }
      res = vector::InsertOp::create(rewriter, loc, elem, res, i);
    }
    return res;
  }

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorValue srcVec = op.getSource();
    Value acc = op.getAcc();
    Value res = op.getResult();
    auto accVec = dyn_cast<VectorValue>(acc);
    auto resVec = dyn_cast<VectorValue>(res);

    auto srcLayout = dyn_cast_if_present<NestedLayoutAttr>(signature[srcVec]);
    if (!srcLayout)
      return rewriter.notifyMatchFailure(op, "non-nested layout");

    Type elemTy = srcVec.getType().getElementType();
    if (elemTy.getIntOrFloatBitWidth() > maxBitsPerShuffle)
      return rewriter.notifyMatchFailure(op, "element too wide for shuffle");

    VectorValue disSrc = getDistributed(rewriter, srcVec, signature[srcVec]);
    Value disAcc;
    if (accVec)
      disAcc = getDistributed(rewriter, accVec, signature[accVec]);
    else
      disAcc = acc;

    SmallVector<bool> redMask = op.getReductionMask();
    int64_t rank = srcVec.getType().getRank();

    // Distributed reduction mask: replicate 3× (batch, outer, element tiers).
    SmallVector<bool> distRedMask;
    distRedMask.reserve(3 * rank);
    for (int i = 0; i < 3; ++i)
      distRedMask.append(redMask.begin(), redMask.end());

    Value localInit = getCombiningIdentityValue(loc, rewriter, op.getKind(),
                                                disAcc.getType());
    Value localRed = vector::MultiDimReductionOp::create(
        rewriter, loc, disSrc, localInit, distRedMask, op.getKind());

    VectorValue locallyReduced;
    if (accVec) {
      locallyReduced = dyn_cast<VectorValue>(localRed);
    } else {
      auto vTy = VectorType::get({1}, elemTy);
      locallyReduced =
          vector::BroadcastOp::create(rewriter, loc, vTy, localRed);
    }
    assert(locallyReduced);

    VectorType shaped = locallyReduced.getType();
    VectorValue threadRed = locallyReduced;
    bool hasThreadRed = llvm::any_of(op.getReductionDims(), [&](int64_t d) {
      return srcLayout.getThreadTile()[d] > 1;
    });
    if (hasThreadRed) {
      int64_t numElems = shaped.getNumElements();
      auto flatTy = VectorType::get({numElems}, elemTy);
      VectorValue flat =
          vector::ShapeCastOp::create(rewriter, loc, flatTy, locallyReduced);
      auto reduced =
          doThreadReduction(rewriter, srcLayout, flat, op.getKind(), redMask);
      if (failed(reduced))
        return failure();
      threadRed = vector::ShapeCastOp::create(rewriter, loc, shaped, *reduced);
    }

    if (!accVec)
      disAcc = vector::BroadcastOp::create(rewriter, loc, shaped, disAcc);

    // For our batch_matmul case there are no inter-subgroup reductions.
    bool hasSubgroupRed = llvm::any_of(op.getReductionDims(), [&](int64_t d) {
      return srcLayout.getSubgroupTile()[d] > 1;
    });
    if (hasSubgroupRed)
      return rewriter.notifyMatchFailure(op,
                                         "subgroup reduction not implemented");

    Value accRed = vector::makeArithReduction(rewriter, loc, op.getKind(),
                                              threadRed, disAcc);
    auto accReduced = dyn_cast<VectorValue>(accRed);
    if (!accReduced)
      return failure();

    if (resVec) {
      replaceOpWithDistributedValues(rewriter, op, accReduced);
    } else {
      Value scalar = vector::ExtractOp::create(rewriter, loc, accRed,
                                               ArrayRef<int64_t>{0});
      replaceOpWithDistributedValues(rewriter, op, scalar);
    }
    return success();
  }

  int64_t subgroupSize;
  int64_t maxBitsPerShuffle;
};

//===----------------------------------------------------------------------===//
// DistributeContract
// Non-MMA path: local contraction + K-thread reduction via MultiDimReduction.
//===----------------------------------------------------------------------===//

struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  static vector::ContractionOp
  doLocalContraction(RewriterBase &rewriter, Location loc, MLIRContext *ctx,
                     vector::ContractionOp op, Value lhs, Value rhs,
                     Value acc) {
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    ArrayRef<Attribute> iters = op.getIteratorTypes().getValue();

    SmallVector<Attribute> newIters;
    for (int i = 0; i < 3; ++i)
      newIters.append(iters.begin(), iters.end());

    SmallVector<AffineMap> newMaps;
    for (AffineMap m : maps) {
      int64_t nd = m.getNumDims(), nr = m.getNumResults();
      SmallVector<AffineExpr> exprs;
      for (int i = 0; i < 3; ++i) {
        AffineMap shifted = m.shiftDims(i * nd);
        for (int j = 0; j < nr; ++j)
          exprs.push_back(shifted.getResult(j));
      }
      newMaps.push_back(AffineMap::get(3 * nd, m.getNumSymbols(), exprs, ctx));
    }
    // Mirror IREE: use acc.getType() as the init so MLIR infers the result
    // type directly from the acc operand (result type == acc type).
    // Do NOT compute the result type explicitly from LHS/RHS dims — that
    // diverges from the acc shape when element tiles are asymmetric (e.g.
    // TF32 ACC elem_N=2 vs RHS elem_N=1) and causes a shape_cast mismatch
    // in the K-reduction step.
    Value localInit =
        getCombiningIdentityValue(loc, rewriter, op.getKind(), acc.getType());
    auto localOp = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, localInit,
        rewriter.getAffineMapArrayAttr(newMaps),
        rewriter.getArrayAttr(newIters), op.getKind());
    localOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    return localOp;
  }

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    // Skip if MMA attr present — defer to MMA-specific patterns.
    if (op->hasAttr("nova.gpu.mma") || op->hasAttr("iree.gpu.mma"))
      return rewriter.notifyMatchFailure(op, "MMA intrinsic attr exists");

    FailureOr<VectorContractOpInfo> maybeInfo =
        VectorContractOpInfo::inferFromIndexingMaps(op.getIndexingMapsArray());
    if (failed(maybeInfo))
      return rewriter.notifyMatchFailure(op, "not a contraction");

    auto lhsLayout = dyn_cast<NestedLayoutAttr>(signature[op.getLhs()]);
    if (!lhsLayout)
      return rewriter.notifyMatchFailure(op, "missing lhs nested layout");
    auto rhsLayout = dyn_cast<NestedLayoutAttr>(signature[op.getRhs()]);
    if (!rhsLayout)
      return rewriter.notifyMatchFailure(op, "missing rhs nested layout");

    NestedLayoutAttr resLayout;
    if (auto resVec = dyn_cast<VectorValue>(op.getResult()))
      resLayout = dyn_cast<NestedLayoutAttr>(signature[resVec]);

    if (!resLayout)
      resLayout = NestedLayoutAttr::get(ctx, ArrayRef<int64_t>{}, {}, {}, {},
                                        {}, {}, {});

    Value disLhs = getDistributed(rewriter, op.getLhs(), lhsLayout);
    Value disRhs = getDistributed(rewriter, op.getRhs(), rhsLayout);

    Value acc = op.getAcc();
    auto accVec = dyn_cast<VectorValue>(acc);
    auto resVec = dyn_cast<VectorValue>(op.getResult());
    Value disAcc =
        accVec ? getDistributed(rewriter, accVec, signature[accVec]) : acc;
    Type elemTy = getElementTypeOrSelf(acc.getType());

    // Step 1: local contraction.
    Value localInit = getCombiningIdentityValue(loc, rewriter, op.getKind(),
                                                disAcc.getType());
    Value localContract =
        doLocalContraction(rewriter, loc, ctx, op, disLhs, disRhs, localInit);

    VectorValue lcVec;
    if (accVec) {
      lcVec = dyn_cast<VectorValue>(localContract);
    } else {
      auto vTy = VectorType::get({1}, elemTy);
      lcVec = vector::BroadcastOp::create(rewriter, loc, vTy, localContract);
    }
    assert(lcVec);

    // Step 2: identify reduction dims and compute reductionLayout.
    auto lhsMap = op.getIndexingMapsArray()[0];
    SmallVector<int64_t> redSGTile, redSGStrides, redThTile, redThStrides;
    SmallVector<int64_t> partialDims;
    for (auto [idx, iterAttr] : llvm::enumerate(op.getIteratorTypes())) {
      if (!vector::isReductionIterator(iterAttr))
        continue;
      auto dimExpr = getAffineDimExpr(idx, ctx);
      std::optional<unsigned> lhsPos = lhsMap.getResultPosition(dimExpr);
      if (!lhsPos.has_value())
        continue;
      unsigned ri = *lhsPos;
      partialDims.push_back(resLayout.getRank() + (int64_t)redSGTile.size());
      redSGTile.push_back(lhsLayout.getSubgroupTile()[ri]);
      redSGStrides.push_back(lhsLayout.getSubgroupStrides()[ri]);
      redThTile.push_back(lhsLayout.getThreadTile()[ri]);
      redThStrides.push_back(lhsLayout.getThreadStrides()[ri]);
    }
    SmallVector<int64_t> unitTile(redThTile.size(), 1);
    auto reductionLayout =
        NestedLayoutAttr::get(ctx, resLayout, redSGTile, unitTile, unitTile,
                              redThTile, unitTile, redSGStrides, redThStrides);

    // Step 3: shape-cast + wrap in to_simd + multi_reduction.
    VectorType partialType =
        VectorType::get(reductionLayout.getDistributedShape(), elemTy);
    Value shapeCasted =
        vector::ShapeCastOp::create(rewriter, loc, partialType, lcVec);
    VectorType undistrType =
        VectorType::get(reductionLayout.getUndistributedShape(), elemTy);
    Value undistrLC =
        vec_ext::ToSIMDOp::create(rewriter, loc, undistrType, shapeCasted);
    auto partialRed = vector::MultiDimReductionOp::create(
        rewriter, loc, op.getKind(), undistrLC, acc, partialDims);
    if (resVec) {
      setSignatureForRedistribution(rewriter, partialRed,
                                    {reductionLayout, signature[resVec]},
                                    {signature[resVec]});
    } else {
      setSignatureForRedistribution(rewriter, partialRed, {reductionLayout},
                                    {});
    }
    rewriter.replaceOp(op, partialRed);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeShapeCast
//===----------------------------------------------------------------------===//

struct DistributeShapeCast final : OpDistributionPattern<vector::ShapeCastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = shapeCastOp.getResult();
    auto resLayout = dyn_cast<NestedLayoutAttr>(signature[result]);
    if (!resLayout)
      return rewriter.notifyMatchFailure(shapeCastOp,
                                         "non-nested result layout");

    VectorValue src = shapeCastOp.getSource();
    auto srcLayout = dyn_cast<NestedLayoutAttr>(signature[src]);
    if (!srcLayout)
      return rewriter.notifyMatchFailure(shapeCastOp, "non-nested src layout");

    VectorValue disSrc = getDistributed(rewriter, src, srcLayout);
    VectorType distResType = VectorType::get(resLayout.getDistributedShape(),
                                             result.getType().getElementType());
    Value distRes = vector::ShapeCastOp::create(rewriter, shapeCastOp.getLoc(),
                                                distResType, disSrc);
    replaceOpWithDistributedValues(rewriter, shapeCastOp, distRes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NVIDIADistributeContract
//
// MMA-path distribution for vector.contract ops annotated with "nova.gpu.mma".
// Emits nvgpu.mma.sync for mma.sync intrinsics (Ampere+) and nvgpu.wmma.mma
// is not yet supported (Volta/Turing). For now only mma.sync is handled.
//
// Algorithm (mirrors AMDGPUDistributeContract.cpp):
//   For each (result_batch, k_batch) tile:
//     lhsSlice = extract(distLhs, [resultBatch..., kBatch])
//     rhsSlice = extract(distRhs, [kBatch, resultBatch...])
//     accSlice = extract(distAcc, [resultBatch...])
//     accSlice = nvgpu.mma.sync(lhsSlice, rhsSlice, accSlice)
//     finalTile = insert(accSlice, finalTile, [resultBatch...])
//   replaceWithDistributed(finalTile)
//===----------------------------------------------------------------------===//

struct NovaMNKShape {
  int64_t m, n, k;
};

/// Return the (M, N, K) shape for a nova.gpu.mma i32 attribute value.
static NovaMNKShape getMNKForMmaAttr(int32_t mmaKind) {
  SmallVector<int64_t, 3> s = getMMAShape(mmaKind);
  if (s.size() < 3)
    return {0, 0, 0};
  return {s[0], s[1], s[2]};
}

/// Return the PTX-mandated per-thread vector shapes for a single nvgpu.mma.sync
/// call.  These are hardware constants from the PTX ISA — not derivable from
/// the NestedLayoutAttr alone.
///
///  operandIndex: 0=A (LHS), 1=B (RHS), 2=C/D (ACC/result)
///
///  mma.sync m16n8k16 f16/bf16:  A=(4,2) B=(2,2) C=(2,2)
///  mma.sync m16n8k8  tf32:      A=(4,1) B=(2,1) C=(2,2)
static VectorType getNvgpuSliceType(int32_t mmaKind, int operandIndex,
                                    Type elemTy) {
  using V = NVMMAIntrinsicValues;
  auto kind = static_cast<V>(mmaKind);
  SmallVector<int64_t, 2> shape;
  switch (kind) {
  case V::MMA_SYNC_F16_16x8x16:
  case V::MMA_SYNC_BF16_16x8x16:
    shape = (operandIndex == 0)   ? SmallVector<int64_t, 2>{4, 2}
            : (operandIndex == 1) ? SmallVector<int64_t, 2>{2, 2}
                                  : SmallVector<int64_t, 2>{2, 2};
    break;
  case V::MMA_SYNC_TF32_16x8x8:
    shape = (operandIndex == 0)   ? SmallVector<int64_t, 2>{4, 1}
            : (operandIndex == 1) ? SmallVector<int64_t, 2>{2, 1}
                                  : SmallVector<int64_t, 2>{2, 2};
    break;
  default:
    // Fallback: compute outer*element per dim (may be wrong for unusual
    // layouts)
    llvm_unreachable("getNvgpuSliceType called for unsupported mma kind");
  }
  return VectorType::get(shape, elemTy);
}

struct NVIDIADistributeContract final
    : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Only handle ops with the nova.gpu.mma marker.
    auto mmaAttr = op->getAttrOfType<IntegerAttr>("nova.gpu.mma");
    if (!mmaAttr)
      return rewriter.notifyMatchFailure(op, "no nova.gpu.mma attr");

    int32_t mmaKind = static_cast<int32_t>(mmaAttr.getInt());

    // Only support mma.sync intrinsics for now (kind 3, 4, 5).
    using V = NVMMAIntrinsicValues;
    auto kind = static_cast<V>(mmaKind);
    bool isMmaSync =
        (kind == V::MMA_SYNC_F16_16x8x16 || kind == V::MMA_SYNC_BF16_16x8x16 ||
         kind == V::MMA_SYNC_TF32_16x8x8);
    if (!isMmaSync)
      return rewriter.notifyMatchFailure(
          op, "unsupported MMA kind (WMMA not yet lowered here)");

    NovaMNKShape mnk = getMNKForMmaAttr(mmaKind);

    auto lhsLayout = dyn_cast<NestedLayoutAttr>(signature[op.getLhs()]);
    auto rhsLayout = dyn_cast<NestedLayoutAttr>(signature[op.getRhs()]);
    if (!lhsLayout || !rhsLayout)
      return rewriter.notifyMatchFailure(op,
                                         "missing nested layout on lhs/rhs");

    auto accVec = dyn_cast<VectorValue>(op.getAcc());
    auto resVec = dyn_cast<VectorValue>(op.getResult());
    if (!accVec || !resVec)
      return rewriter.notifyMatchFailure(op, "acc/result must be vectors");

    auto accLayout = dyn_cast<NestedLayoutAttr>(signature[accVec]);
    if (!accLayout)
      return rewriter.notifyMatchFailure(op, "missing nested layout on acc");

    Location loc = op.getLoc();
    Type elemTy =
        mlir::cast<VectorType>(op.getLhs().getType()).getElementType();
    Type accElemTy = accVec.getType().getElementType();

    // Get distributed (per-thread batch-tiled) operands.
    Value distLhs = getDistributed(rewriter, op.getLhs(), lhsLayout);
    Value distRhs = getDistributed(rewriter, op.getRhs(), rhsLayout);
    Value distAcc = getDistributed(rewriter, accVec, accLayout);

    // Batch counts: from acc layout (result dims) and lhs layout (K dim).
    SmallVector<int64_t> resultBatches(accLayout.getBatchTile());

    // K batch count: look up the batch count for the K iter-dim(s) in the LHS
    // layout, NOT simply .back(). The LHS indexing map may permute dimensions
    // (e.g. `(d1, d0)` for a transposed LHS), so .back() would return the
    // M-batch count instead of the K-batch count.
    //
    // We derive kIterDims below from the maps, so we first do a quick scan to
    // find which LHS result-dim position corresponds to the K iter dim.
    // For now, identify K-result-dims as those NOT appearing in the ACC map.
    // We compute the final kBatchCount after kIterDims is built (see below).
    // Initialise to 1; corrected after kIterDims is populated.
    int64_t kBatchCount = 1;

    // Indexing maps for projecting result/K batch offsets → operand offsets.
    // The contract indexing maps tell us which iteration dims each operand
    // uses.
    auto indexingMaps = op.getIndexingMapsArray(); // [lhs, rhs, acc]
    int64_t numIterDims = indexingMaps[0].getNumDims();

    // PTX-mandated per-call slice shapes for nvgpu.mma.sync.
    // These are hardware constants — NOT derivable from the layout alone.
    VectorType lhsSliceTy = getNvgpuSliceType(mmaKind, 0, elemTy);
    VectorType rhsSliceTy = getNvgpuSliceType(mmaKind, 1, elemTy);
    VectorType accSliceTy = getNvgpuSliceType(mmaKind, 2, accElemTy);

    // mmaShape attribute: [M, N, K] from the intrinsic.
    SmallVector<int64_t> mmaShape = {mnk.m, mnk.n, mnk.k};

    // Use the existing distributed accumulator as the reassembly base.
    // This makes the loop-carried SSA dependency on distAcc explicit in the
    // insert chain: the first vector.insert targets %arg8 (the iter_arg) rather
    // than a zero constant. LLVM then sees %result = f(%arg8, mma_outputs)
    // directly, forming the correct phi node for the K-loop accumulator without
    // relying on the transitive MMA C-input dependency to prove the recurrence.
    Value finalTile = distAcc;

    // Build a mapping from iteration-space dims to batch-index values.
    // The acc indexing map tells us which iter dims are used by the
    // accumulator. We invert it: for each result dim position i,
    // iterDimForAccDim[i] = iter dim.
    AffineMap accMap = indexingMaps[2]; // acc is operand index 2
    AffineMap lhsMap = indexingMaps[0];
    AffineMap rhsMap = indexingMaps[1];

    // Build iteration-space → batch offset lookup. For acc dims, we know the
    // batch offsets directly from resultBatchOffsets. K is filled in per k
    // loop. iterBatch[d] = batch offset for iteration dim d, or -1 if unknown.
    // We detect which iter dim is the K-reduction dim by finding the iter dim
    // that appears in lhsMap/rhsMap but NOT in accMap.
    SmallVector<int64_t> accIterDims; // which iter dims acc uses, in order
    for (auto expr : accMap.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
        accIterDims.push_back(dimExpr.getPosition());
    }
    // Find K iter dims: dims in lhsMap that are NOT in accMap.
    DenseSet<int64_t> accIterDimsSet(accIterDims.begin(), accIterDims.end());
    SmallVector<int64_t> kIterDims;
    for (auto expr : lhsMap.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        int64_t d = dimExpr.getPosition();
        if (!accIterDimsSet.count(d))
          kIterDims.push_back(d);
      }
    }
    // kIterDims should have exactly 1 entry for a standard matmul contraction.

    // Now compute the correct kBatchCount: the batch count for the K-reduction
    // dim(s) in the LHS layout. We find which LHS result-dim positions carry K
    // iter dims, then read those from lhsLayout.getBatchTile().
    //
    // Example: LHS map `(d1, d0) → [K=d1, M=d0]` with batch_tile=[1, 2]:
    //   result-dim 0 = d1 (K iter dim) → K-batch = lhsBatchTile[0] = 1 ✓
    //   result-dim 1 = d0 (M iter dim) → M-batch = lhsBatchTile[1] = 2
    // Using .back() would have given M-batch=2 as kBatchCount, causing OOB.
    {
      DenseSet<int64_t> kIterDimSet(kIterDims.begin(), kIterDims.end());
      SmallVector<int64_t> lhsBatch(lhsLayout.getBatchTile());
      for (auto [resultDimIdx, expr] : llvm::enumerate(lhsMap.getResults())) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          if (kIterDimSet.count(dimExpr.getPosition()) &&
              resultDimIdx < lhsBatch.size()) {
            // Take the max in case of multiple K dims (e.g. depth-wise convs).
            kBatchCount = std::max(kBatchCount, lhsBatch[resultDimIdx]);
          }
        }
      }
    }

    // Helper: given an indexing map, build the operand batch offset vector
    // using the current result batch offsets and the k index.
    auto buildOperandBatchOff = [&](AffineMap map,
                                    ArrayRef<int64_t> resultBatchOffsets,
                                    int64_t k) -> SmallVector<int64_t> {
      // Build iter-dim → offset map.
      SmallVector<int64_t> iterOff(numIterDims, 0);
      for (auto [i, iterDim] : llvm::enumerate(accIterDims))
        iterOff[iterDim] = resultBatchOffsets[i];
      for (int64_t kd : kIterDims)
        iterOff[kd] = k;
      // Project through the operand map.
      SmallVector<int64_t> opOff;
      for (auto expr : map.getResults()) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
          opOff.push_back(iterOff[dimExpr.getPosition()]);
        else
          opOff.push_back(
              0); // constant/symbol — shouldn't appear in batch maps
      }
      return opOff;
    };

    // Iterate over result batches (all acc batch dims, e.g. [B, M, N] for
    // rank-3).
    SmallVector<int64_t> resultBatchShape(resultBatches);
    SmallVector<int64_t> resultBatchTile(resultBatches.size(), 1);

    // Helper: validate that a batch-offset vector is in-bounds for a given
    // distributed vector. Returns failure() with a diagnostic on violation.
    // This guards against layout/batchCounts mismatches that would otherwise
    // produce illegal vector.extract ops (e.g. position [1,0] into
    // vector<1x...>).
    auto validateExtractPos = [&](Value vec, ArrayRef<int64_t> pos,
                                  StringRef label) -> LogicalResult {
      auto vecTy = dyn_cast<VectorType>(vec.getType());
      if (!vecTy)
        return success(); // scalar — no dimension to check
      ArrayRef<int64_t> shape = vecTy.getShape();
      for (auto [i, p] : llvm::enumerate(pos)) {
        if (i >= shape.size())
          break; // extract leaves trailing dims
        if (p < 0 || p >= shape[i]) {
          return op.emitOpError()
                 << "NVIDIADistributeContract: " << label
                 << " extract position[" << i << "]= " << p
                 << " is out-of-bounds for dim size " << shape[i]
                 << " (distributed vector " << vecTy << ")."
                 << " Check batchCounts in setContractionAnchor.";
        }
      }
      return success();
    };

    // Precompute LHS slices (extract + shuffle) keyed on lhsBatchOff.
    //
    // lhsBatchOff = [k, m_batch, ...] — it depends on k and the M-result dims
    // but NOT on the N-result dims. For a standard 4×4 result tile with 4
    // K-batches, each (k, m) pair is reused across all 4 N-columns, so the
    // extract+shuffle chain would otherwise be emitted 4× per (k,m).
    // Precomputing here reduces 64 chains → 16 chains per kernel (4×
    // reduction).
    std::map<SmallVector<int64_t>, Value> lhsSliceCache;
    for (auto resultBatchOffsets :
         StaticTileOffsetRange(resultBatchShape, resultBatchTile)) {
      for (int64_t k = 0; k < kBatchCount; ++k) {
        SmallVector<int64_t> lhsBatchOff =
            buildOperandBatchOff(lhsMap, resultBatchOffsets, k);
        if (lhsSliceCache.count(lhsBatchOff))
          continue;
        Value lhsSlice =
            vector::ExtractOp::create(rewriter, loc, distLhs, lhsBatchOff);
        if (lhsSlice.getType() != lhsSliceTy)
          lhsSlice =
              vector::ShapeCastOp::create(rewriter, loc, lhsSliceTy, lhsSlice);
        // Skip the lane-permute shuffle when the LHS came from nvgpu.ldmatrix
        // — its output is already in PTX A-fragment lane order, so re-shuffling
        // would corrupt the fragment.
        bool lhsFromLdmatrix = op->hasAttr("nova.gpu.lhs_from_ldmatrix");
        if (!lhsFromLdmatrix && (kind == V::MMA_SYNC_F16_16x8x16 ||
                                 kind == V::MMA_SYNC_BF16_16x8x16)) {
          auto flat1D = VectorType::get({8}, lhsSliceTy.getElementType());
          Value flat =
              vector::ShapeCastOp::create(rewriter, loc, flat1D, lhsSlice);
          Value shuffled = vector::ShuffleOp::create(
              rewriter, loc, flat, flat,
              ArrayRef<int64_t>{2, 3, 0, 1, 6, 7, 4, 5});
          lhsSlice =
              vector::ShapeCastOp::create(rewriter, loc, lhsSliceTy, shuffled);
        } else if (!lhsFromLdmatrix && kind == V::MMA_SYNC_TF32_16x8x8) {
          auto flat1D = VectorType::get({4}, lhsSliceTy.getElementType());
          Value flat =
              vector::ShapeCastOp::create(rewriter, loc, flat1D, lhsSlice);
          Value shuffled = vector::ShuffleOp::create(
              rewriter, loc, flat, flat, ArrayRef<int64_t>{0, 2, 1, 3});
          lhsSlice =
              vector::ShapeCastOp::create(rewriter, loc, lhsSliceTy, shuffled);
        }
        lhsSliceCache[lhsBatchOff] = lhsSlice;
      }
    }

    for (auto [batchIdx, resultBatchOffsets] : llvm::enumerate(
             StaticTileOffsetRange(resultBatchShape, resultBatchTile))) {

      // Validate before creating the extract to catch layout mismatches early.
      if (failed(validateExtractPos(distAcc, resultBatchOffsets, "acc")))
        return rewriter.notifyMatchFailure(
            op, "acc extract position out-of-bounds (batchCounts mismatch)");

      // Start accumulator slice from distAcc at this batch offset.
      Value accSlice =
          vector::ExtractOp::create(rewriter, loc, distAcc, resultBatchOffsets);

      // batchSliceTy: the shape that vector.insert expects at these batch
      // offsets. Computed once per result-batch, used after all K steps.
      SmallVector<int64_t> batchSliceShape;
      for (int64_t v : accLayout.getOuterTile())
        batchSliceShape.push_back(v);
      for (int64_t v : accLayout.getElementTile())
        batchSliceShape.push_back(v);
      VectorType batchSliceTy = VectorType::get(batchSliceShape, accElemTy);

      // Cast accSlice to accSliceTy once before the K loop so we stay in the
      // mma fragment shape across all K steps (avoids a cast-in + cast-out per
      // K iteration that otherwise round-trips through batchSliceTy every
      // step).
      if (accSlice.getType() != accSliceTy)
        accSlice =
            vector::ShapeCastOp::create(rewriter, loc, accSliceTy, accSlice);

      // Accumulate over K batches.
      for (int64_t k = 0; k < kBatchCount; ++k) {
        // Project iteration-space offsets through each operand's indexing map.
        SmallVector<int64_t> lhsBatchOff =
            buildOperandBatchOff(lhsMap, resultBatchOffsets, k);
        SmallVector<int64_t> rhsBatchOff =
            buildOperandBatchOff(rhsMap, resultBatchOffsets, k);

        // Validate operand extract positions before creation.
        if (failed(validateExtractPos(distLhs, lhsBatchOff, "lhs")))
          return rewriter.notifyMatchFailure(
              op, "lhs extract position out-of-bounds (batchCounts mismatch)");
        if (failed(validateExtractPos(distRhs, rhsBatchOff, "rhs")))
          return rewriter.notifyMatchFailure(
              op, "rhs extract position out-of-bounds (batchCounts mismatch)");

        // Reuse precomputed LHS slice (extract + shuffle done above).
        Value lhsSlice = lhsSliceCache[lhsBatchOff];

        Value rhsSlice =
            vector::ExtractOp::create(rewriter, loc, distRhs, rhsBatchOff);

        if (rhsSlice.getType() != rhsSliceTy)
          rhsSlice =
              vector::ShapeCastOp::create(rewriter, loc, rhsSliceTy, rhsSlice);

        bool tf32 = (kind == V::MMA_SYNC_TF32_16x8x8);
        accSlice = nvgpu::MmaSyncOp::create(rewriter, loc, lhsSlice, rhsSlice,
                                            accSlice, mmaShape, tf32);
      }

      // Cast mma result from accSliceTy back to batchSliceTy for vector.insert.
      if (accSlice.getType() != batchSliceTy)
        accSlice =
            vector::ShapeCastOp::create(rewriter, loc, batchSliceTy, accSlice);

      // Insert accumulated slice back into the result tile.
      finalTile = vector::InsertOp::create(rewriter, loc, accSlice, finalTile,
                                           resultBatchOffsets);
    }

    replaceOpWithDistributedValues(rewriter, op, finalTile);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void populateNovaGPUDistributeNestedLayoutAttrPatterns(
    RewritePatternSet &patterns,
    const llvm::DenseMap<Block *, Value> &threadIdMap, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize, int64_t maxBitsPerShuffle) {
  patterns.add<DistributeTransferRead>(patterns.getContext(), threadIdMap,
                                       subgroupSize);
  patterns.add<DistributeTransferWrite>(patterns.getContext(), threadIdMap,
                                        subgroupSize, workgroupSize);
  patterns.add<DistributeBroadcast, DistributeShapeCast>(patterns.getContext());
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                         maxBitsPerShuffle);
  patterns.add<DistributeContract>(patterns.getContext());
  patterns.add<NVIDIADistributeContract>(patterns.getContext());
}

} // namespace mlir::nova