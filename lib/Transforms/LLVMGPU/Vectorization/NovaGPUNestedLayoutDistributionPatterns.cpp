//===- NovaGPUNestedLayoutDistributionPatterns.cpp -------------------------===//
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
#include "Compiler/Transforms/LLVMGPU/NovaVectorOpUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

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
static SmallVector<int64_t>
getElementVectorTileShape(NestedLayoutAttr layout) {
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
  ArrayRef<int64_t> outerSizes   = layout.getOuterTile();
  ArrayRef<int64_t> elementSizes = layout.getElementTile();

  SmallVector<int64_t> result;
  result.reserve(rank);
  for (auto [b, o, os, es] :
       llvm::zip(batchOffsets, outerOffsets, outerSizes, elementSizes))
    result.push_back(b * os * es + o * es);
  return result;
}

/// Compute the memory indices for one tile of a transfer_read/write.
static SmallVector<Value>
getTransferIndices(OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
                   NestedLayoutAttr layout, AffineMap permMap,
                   ArrayRef<Value> warpIndices,
                   ArrayRef<Value> threadIndices) {
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
    Value batchConst = b.create<arith::ConstantIndexOp>(loc, batchOffsets[i]);
    Value outerConst = b.create<arith::ConstantIndexOp>(loc, outerOffsets[i]);
    SmallVector<Value> ids = {warpIndices[i], batchConst, outerConst,
                               threadIndices[i], base};
    SmallVector<int64_t> sizes = {layout.getSubgroupTile()[i],
                                   layout.getBatchTile()[i],
                                   layout.getOuterTile()[i],
                                   layout.getThreadTile()[i], elems};
    bool disjoint = false;
    if (auto c = getConstantIntValue(base))
      disjoint = *c < elems;
    sliced[pos] = affine::AffineLinearizeIndexOp::create(b, loc, ids, sizes,
                                                          disjoint);
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
  SmallVector<int64_t> maskOffsets = getDistributedTransferOffsets(offsets, layout);
  SmallVector<int64_t> strides(layout.getElementTile().size(), 1);
  return vector::ExtractStridedSliceOp::create(rewriter, loc, mask, maskOffsets,
                                               layout.getElementTile(), strides);
}

/// Compute warp and thread indices from a linear thread ID.
static LogicalResult
populateWarpAndThreadIndices(RewriterBase &rewriter, Value threadId,
                              int64_t subgroupSize, NestedLayoutAttr layout,
                              SmallVector<Value> &warpIndices,
                              SmallVector<Value> &threadIndices) {
  int64_t rank = layout.getRank();
  SmallVector<Value> ids = layout.computeThreadIds(threadId, subgroupSize,
                                                    rewriter);
  if (ids.empty() && rank != 0)
    return failure();
  warpIndices   = SmallVector<Value>(ids.begin(), ids.begin() + rank);
  threadIndices = SmallVector<Value>(ids.begin() + rank, ids.begin() + 2*rank);
  return success();
}

/// Broadcast a value to a distributed shape, marking `broadcastedDims`.
static VectorValue broadcastToShape(RewriterBase &rewriter, Value source,
                                     ArrayRef<int64_t> shape,
                                     ArrayRef<bool> broadcastedDims) {
  assert(shape.size() == broadcastedDims.size());
  SmallVector<int64_t> bcastIdx, ubcastIdx;
  for (auto [i, b] : llvm::enumerate(broadcastedDims)) {
    if (b) bcastIdx.push_back(i);
    else    ubcastIdx.push_back(i);
  }
  SmallVector<int64_t> perm = llvm::to_vector(
      llvm::concat<int64_t>(bcastIdx, ubcastIdx));
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
    attr = b.getZeroAttr(elemTy); break;
  case vector::CombiningKind::MUL:
    if (elemTy.isIntOrIndex())
      attr = b.getIntegerAttr(elemTy, 1);
    else
      attr = b.getFloatAttr(elemTy, 1.0);
    break;
  case vector::CombiningKind::MINIMUMF:
  case vector::CombiningKind::MINNUMF: {
    auto inf = APFloat::getInf(cast<FloatType>(elemTy).getFloatSemantics());
    attr = b.getFloatAttr(elemTy, inf); break;
  }
  case vector::CombiningKind::MAXIMUMF:
  case vector::CombiningKind::MAXNUMF: {
    auto ninf = APFloat::getInf(cast<FloatType>(elemTy).getFloatSemantics(),
                                 /*Negative=*/true);
    attr = b.getFloatAttr(elemTy, ninf); break;
  }
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXSI:
    attr = b.getIntegerAttr(elemTy, std::numeric_limits<int64_t>::min()); break;
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINSI:
    attr = b.getIntegerAttr(elemTy, std::numeric_limits<int64_t>::max()); break;
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
  case vector::CombiningKind::ADD:  return gpu::AllReduceOperation::ADD;
  case vector::CombiningKind::MUL:  return gpu::AllReduceOperation::MUL;
  case vector::CombiningKind::MINUI: return gpu::AllReduceOperation::MINUI;
  case vector::CombiningKind::MINSI: return gpu::AllReduceOperation::MINSI;
  case vector::CombiningKind::MAXUI: return gpu::AllReduceOperation::MAXUI;
  case vector::CombiningKind::MAXSI: return gpu::AllReduceOperation::MAXSI;
  case vector::CombiningKind::AND:  return gpu::AllReduceOperation::AND;
  case vector::CombiningKind::OR:   return gpu::AllReduceOperation::OR;
  case vector::CombiningKind::XOR:  return gpu::AllReduceOperation::XOR;
  default: return gpu::AllReduceOperation::ADD;
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
    shape[rank*0 + d] = packed[1]; // batch
    shape[rank*1 + d] = packed[2]; // outer
    shape[rank*2 + d] = packed[4]; // element
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
    unshape.push_back(pshape[d*3+0] * pshape[d*3+1] * pshape[d*3+2]);
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
      perm.push_back(t + 3*d);
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
  auto transposed = vector::TransposeOp::create(rewriter, loc, val, transposePerm);
  SmallVector<int64_t> extractPos(slicedSet.size(), 0);
  auto sliced = vector::ExtractOp::create(rewriter, loc, transposed, extractPos);
  return cast<VectorValue>(sliced.getResult());
}

/// Correct port of IREE's basisFromSizesStrides (iree/compiler/Utils/Indexing.cpp).
/// Handles stride gaps by inserting anonymous jump entries, reverses the basis
/// to descending stride order, and returns 1-indexed dimToResult values so that
/// index 0 is always the overflow (outermost) slot inserted by
/// AffineDelinearizeIndexOp when hasOuterBound=true.
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

namespace {

//===----------------------------------------------------------------------===//
// DistributeTransferRead
//===----------------------------------------------------------------------===//

struct DistributeTransferRead final
    : OpDistributionPattern<vector::TransferReadOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferRead(MLIRContext *ctx, Value threadId, int64_t subgroupSize)
      : OpDistributionPattern(ctx), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto layout = dyn_cast<NestedLayoutAttr>(signature[readOp.getResult()]);
    if (!layout)
      return rewriter.notifyMatchFailure(readOp, "non-nested layout");
    if (!isa<MemRefType>(readOp.getBase().getType()))
      return rewriter.notifyMatchFailure(readOp, "distribution expects memrefs");

    VectorValue mask    = readOp.getMask();
    NestedLayoutAttr maskLayout;
    if (mask) {
      maskLayout = dyn_cast<NestedLayoutAttr>(signature[mask]);
      if (!maskLayout)
        return rewriter.notifyMatchFailure(readOp, "non-nested mask layout");
      mask = getDistributed(rewriter, mask, maskLayout);
      mask = getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
    }

    SmallVector<int64_t> distShape  = layout.getDistributedShape();
    SmallVector<int64_t> tileShape  = getElementVectorTileShape(layout);
    int64_t rank = layout.getRank();
    Type elemTy  = readOp.getBase().getType().getElementType();

    auto vecType = VectorType::get(distShape, elemTy);
    auto innerTy = VectorType::get(layout.getElementTile(), elemTy);
    Value zero = arith::ConstantOp::create(rewriter, readOp.getLoc(), vecType,
                                            rewriter.getZeroAttr(vecType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIdx, threadIdx;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                             layout, warpIdx, threadIdx)))
      return rewriter.notifyMatchFailure(readOp, "failed to compute thread ids");

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    if (mask) {
      SmallVector<int64_t> mds  = maskLayout.getDistributedShape();
      SmallVector<int64_t> mts  = getElementVectorTileShape(maskLayout);
      allMaskOffsets = llvm::to_vector(StaticTileOffsetRange(mds, mts));
    }

    SmallVector<int64_t> strides(rank, 1);
    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndices(
          rewriter, readOp.getIndices(), offsets, layout,
          readOp.getPermutationMap(), warpIdx, threadIdx);

      VectorValue slicedMask = nullptr;
      if (mask) {
        auto mds = maskLayout.getDistributedShape();
        auto mts = getElementVectorTileShape(maskLayout);
        slicedMask = getSlicedPermutedMask(rewriter, readOp.getLoc(),
                                            allMaskOffsets[idx], maskLayout, mask);
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

  Value   threadId;
  int64_t subgroupSize;
};

//===----------------------------------------------------------------------===//
// DistributeTransferWrite
//===----------------------------------------------------------------------===//

struct DistributeTransferWrite final
    : OpDistributionPattern<vector::TransferWriteOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferWrite(MLIRContext *ctx, Value threadId, int64_t subgroupSize,
                           ArrayRef<int64_t> workgroupSize)
      : OpDistributionPattern(ctx), threadId(threadId),
        subgroupSize(subgroupSize) {
    if (!workgroupSize.empty())
      numThreads = std::accumulate(workgroupSize.begin(), workgroupSize.end(),
                                    int64_t(1), std::multiplies<int64_t>());
  }

  /// Returns a boolean Value that is true only for threads that should write
  /// (avoids duplicate writes when broadcast dims exist).
  FailureOr<Value> getNoOverlapCondition(OpBuilder &b, Location loc,
                                          NestedLayoutAttr layout) const {
    ArrayRef<int64_t> threadTile    = layout.getThreadTile();
    ArrayRef<int64_t> threadStrides = layout.getThreadStrides();
    ArrayRef<int64_t> subgroupTile  = layout.getSubgroupTile();
    SmallVector<int64_t> subgroupStrides =
        llvm::map_to_vector(layout.getSubgroupStrides(),
                            [&](int64_t x) { return x * subgroupSize; });
    auto concatTiles   = llvm::to_vector(llvm::concat<const int64_t>(subgroupTile, threadTile));
    auto concatStrides = llvm::to_vector(llvm::concat<const int64_t>(subgroupStrides, threadStrides));

    SmallVector<int64_t> basis;
    SmallVector<size_t> dimToResult;
    if (failed(basisFromSizesStrides(concatTiles, concatStrides, basis, dimToResult)))
      return failure();
    // basisFromSizesStrides produces 1-indexed dimToResult values: index 0 is
    // always the overflow slot that AffineDelinearizeIndexOp inserts as the
    // outermost (hasOuterBound=true) term. No manual index adjustment needed
    // regardless of whether numThreads is prepended.
    if (numThreads.has_value()) {
      int64_t outer = numThreads.value() /
                      std::accumulate(basis.begin(), basis.end(),
                                      int64_t(1), std::multiplies<int64_t>());
      basis.insert(basis.begin(), outer);
    }
    SmallVector<Value> delinearized;
    b.createOrFold<affine::AffineDelinearizeIndexOp>(
        delinearized, loc, threadId, basis, numThreads.has_value());
    // arith::ConstantIntOp(value=1, width=1) → i1 true; avoids TypedAttr overload.
    Value cond = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);
    for (auto [i, v] : llvm::enumerate(delinearized)) {
      if (llvm::is_contained(dimToResult, i))
        continue;
      Value isZero = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, v,
          b.create<arith::ConstantIndexOp>(loc, 0));
      cond = b.create<arith::AndIOp>(loc, cond, isZero);
    }
    return cond;
  }

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto layout = dyn_cast<NestedLayoutAttr>(signature[writeOp.getValueToStore()]);
    if (!layout)
      return rewriter.notifyMatchFailure(writeOp, "non-nested layout");
    if (!isa<MemRefType>(writeOp.getBase().getType()))
      return rewriter.notifyMatchFailure(writeOp, "distribution expects memrefs");

    VectorValue mask    = writeOp.getMask();
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
      return rewriter.notifyMatchFailure(writeOp, "failed to compute thread ids");

    Location loc = writeOp.getLoc();
    FailureOr<Value> doWrite = getNoOverlapCondition(rewriter, loc, layout);
    if (failed(doWrite))
      return rewriter.notifyMatchFailure(writeOp, "failed no-overlap condition");

    auto ifOp = scf::IfOp::create(rewriter, loc, doWrite.value());
    rewriter.setInsertionPoint(ifOp.thenYield());

    Value distVec = getDistributed(rewriter, writeOp.getValueToStore(), layout);

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    if (mask) {
      auto mds = maskLayout.getDistributedShape();
      auto mts = getElementVectorTileShape(maskLayout);
      allMaskOffsets = llvm::to_vector(StaticTileOffsetRange(mds, mts));
    }

    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndices(
          rewriter, writeOp.getIndices(), offsets, layout,
          writeOp.getPermutationMap(), warpIdx, threadIdx);
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

  Value   threadId;
  int64_t subgroupSize;
  std::optional<int64_t> numThreads;
};

//===----------------------------------------------------------------------===//
// DistributeBroadcast
//===----------------------------------------------------------------------===//

struct DistributeBroadcast final
    : OpDistributionPattern<vector::BroadcastOp> {
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
    int64_t bcastRank  = layout.getRank();
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
        return rewriter.notifyMatchFailure(broadcastOp, "non-nested src layout");
      distSrc = getDistributed(rewriter, srcVec, srcLayout);
    }

    VectorValue result = broadcastToShape(rewriter, distSrc, distShape, broadcastedDims);
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
      : OpDistributionPattern(ctx, benefit),
        subgroupSize(subgroupSize), maxBitsPerShuffle(maxBitsPerShuffle) {}

  /// Perform inter-thread butterfly reduction on a flat vector.
  FailureOr<VectorValue>
  doThreadReduction(RewriterBase &rewriter, NestedLayoutAttr layout,
                    VectorValue flat, vector::CombiningKind kind,
                    ArrayRef<bool> reductionMask) const {
    VectorType flatTy = flat.getType();
    int64_t n = flatTy.getNumElements();
    Location loc = flat.getLoc();
    auto zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(flatTy));
    auto res = cast<VectorValue>(zero.getResult());
    for (int64_t i = 0; i < n; ++i) {
      Value elem = vector::ExtractOp::create(rewriter, loc, flat, i);
      for (auto [d, isReduced] : llvm::enumerate(reductionMask)) {
        if (!isReduced)
          continue;
        int64_t offset = layout.getThreadStrides()[d];
        int64_t width  = layout.getThreadTile()[d];
        if (offset == 0 || width <= 1)
          continue;
        elem = gpu::SubgroupReduceOp::create(
            rewriter, loc, elem,
            combiningKindToAllReduce(kind), /*uniform=*/false,
            static_cast<uint32_t>(width),
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
    Value localRed  = vector::MultiDimReductionOp::create(
        rewriter, loc, disSrc, localInit, distRedMask, op.getKind());

    VectorValue locallyReduced;
    if (accVec) {
      locallyReduced = dyn_cast<VectorValue>(localRed);
    } else {
      auto vTy = VectorType::get({1}, elemTy);
      locallyReduced = vector::BroadcastOp::create(rewriter, loc, vTy, localRed);
    }
    assert(locallyReduced);

    VectorType shaped   = locallyReduced.getType();
    VectorValue threadRed = locallyReduced;
    bool hasThreadRed = llvm::any_of(op.getReductionDims(), [&](int64_t d) {
      return srcLayout.getThreadTile()[d] > 1;
    });
    if (hasThreadRed) {
      int64_t numElems = shaped.getNumElements();
      auto flatTy = VectorType::get({numElems}, elemTy);
      VectorValue flat = vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                      locallyReduced);
      auto reduced = doThreadReduction(rewriter, srcLayout, flat, op.getKind(),
                                        redMask);
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
      return rewriter.notifyMatchFailure(op, "subgroup reduction not implemented");

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

struct DistributeContract final
    : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  static vector::ContractionOp
  doLocalContraction(RewriterBase &rewriter, Location loc, MLIRContext *ctx,
                     vector::ContractionOp op, Value lhs, Value rhs,
                     Value acc) {
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    ArrayRef<Attribute> iters  = op.getIteratorTypes().getValue();

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
      newMaps.push_back(
          AffineMap::get(3*nd, m.getNumSymbols(), exprs, ctx));
    }
    // Mirror IREE: use acc.getType() as the init so MLIR infers the result
    // type directly from the acc operand (result type == acc type).
    // Do NOT compute the result type explicitly from LHS/RHS dims — that
    // diverges from the acc shape when element tiles are asymmetric (e.g.
    // TF32 ACC elem_N=2 vs RHS elem_N=1) and causes a shape_cast mismatch
    // in the K-reduction step.
    Value localInit = getCombiningIdentityValue(loc, rewriter, op.getKind(),
                                                 acc.getType());
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
      resLayout = NestedLayoutAttr::get(
          ctx, ArrayRef<int64_t>{}, {}, {}, {}, {}, {}, {});

    Value disLhs = getDistributed(rewriter, op.getLhs(), lhsLayout);
    Value disRhs = getDistributed(rewriter, op.getRhs(), rhsLayout);

    Value acc    = op.getAcc();
    auto accVec  = dyn_cast<VectorValue>(acc);
    auto resVec  = dyn_cast<VectorValue>(op.getResult());
    Value disAcc = accVec ? getDistributed(rewriter, accVec, signature[accVec])
                           : acc;
    Type elemTy  = getElementTypeOrSelf(acc.getType());

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
    auto reductionLayout = NestedLayoutAttr::get(
        ctx, resLayout, redSGTile, unitTile, unitTile,
        redThTile, unitTile, redSGStrides, redThStrides);

    // Step 3: shape-cast + wrap in to_simd + multi_reduction.
    VectorType partialType = VectorType::get(reductionLayout.getDistributedShape(),
                                              elemTy);
    Value shapeCasted = vector::ShapeCastOp::create(rewriter, loc, partialType,
                                                     lcVec);
    VectorType undistrType = VectorType::get(reductionLayout.getUndistributedShape(),
                                              elemTy);
    Value undistrLC = vec_ext::ToSIMDOp::create(rewriter, loc, undistrType,
                                                  shapeCasted);
    auto partialRed = vector::MultiDimReductionOp::create(
        rewriter, loc, op.getKind(), undistrLC, acc, partialDims);
    if (resVec) {
      setSignatureForRedistribution(rewriter, partialRed,
                                     {reductionLayout, signature[resVec]},
                                     {signature[resVec]});
    } else {
      setSignatureForRedistribution(rewriter, partialRed,
                                     {reductionLayout}, {});
    }
    rewriter.replaceOp(op, partialRed);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeShapeCast
//===----------------------------------------------------------------------===//

struct DistributeShapeCast final
    : OpDistributionPattern<vector::ShapeCastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = shapeCastOp.getResult();
    auto resLayout = dyn_cast<NestedLayoutAttr>(signature[result]);
    if (!resLayout)
      return rewriter.notifyMatchFailure(shapeCastOp, "non-nested result layout");

    VectorValue src = shapeCastOp.getSource();
    auto srcLayout = dyn_cast<NestedLayoutAttr>(signature[src]);
    if (!srcLayout)
      return rewriter.notifyMatchFailure(shapeCastOp, "non-nested src layout");

    VectorValue disSrc = getDistributed(rewriter, src, srcLayout);
    VectorType distResType =
        VectorType::get(resLayout.getDistributedShape(),
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

struct NovaMNKShape { int64_t m, n, k; };

/// Return the (M, N, K) shape for a nova.gpu.mma i32 attribute value.
static NovaMNKShape getMNKForMmaAttr(int32_t mmaKind) {
  SmallVector<int64_t, 3> s = getMMAShape(mmaKind);
  if (s.size() < 3) return {0, 0, 0};
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
    shape = (operandIndex == 0) ? SmallVector<int64_t,2>{4, 2}
          : (operandIndex == 1) ? SmallVector<int64_t,2>{2, 2}
                                : SmallVector<int64_t,2>{2, 2};
    break;
  case V::MMA_SYNC_TF32_16x8x8:
    shape = (operandIndex == 0) ? SmallVector<int64_t,2>{4, 1}
          : (operandIndex == 1) ? SmallVector<int64_t,2>{2, 1}
                                : SmallVector<int64_t,2>{2, 2};
    break;
  default:
    // Fallback: compute outer*element per dim (may be wrong for unusual layouts)
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
    bool isMmaSync = (kind == V::MMA_SYNC_F16_16x8x16 ||
                      kind == V::MMA_SYNC_BF16_16x8x16 ||
                      kind == V::MMA_SYNC_TF32_16x8x8);
    if (!isMmaSync)
      return rewriter.notifyMatchFailure(op, "unsupported MMA kind (WMMA not yet lowered here)");

    NovaMNKShape mnk = getMNKForMmaAttr(mmaKind);

    auto lhsLayout = dyn_cast<NestedLayoutAttr>(signature[op.getLhs()]);
    auto rhsLayout = dyn_cast<NestedLayoutAttr>(signature[op.getRhs()]);
    if (!lhsLayout || !rhsLayout)
      return rewriter.notifyMatchFailure(op, "missing nested layout on lhs/rhs");

    auto accVec  = dyn_cast<VectorValue>(op.getAcc());
    auto resVec  = dyn_cast<VectorValue>(op.getResult());
    if (!accVec || !resVec)
      return rewriter.notifyMatchFailure(op, "acc/result must be vectors");

    auto accLayout = dyn_cast<NestedLayoutAttr>(signature[accVec]);
    if (!accLayout)
      return rewriter.notifyMatchFailure(op, "missing nested layout on acc");

    Location loc = op.getLoc();
    Type elemTy  = mlir::cast<VectorType>(op.getLhs().getType()).getElementType();
    Type accElemTy = accVec.getType().getElementType();

    // Get distributed (per-thread batch-tiled) operands.
    Value distLhs = getDistributed(rewriter, op.getLhs(), lhsLayout);
    Value distRhs = getDistributed(rewriter, op.getRhs(), rhsLayout);
    Value distAcc = getDistributed(rewriter, accVec, accLayout);

    // Batch counts: from acc layout (result dims) and lhs layout (K dim).
    SmallVector<int64_t> resultBatches(accLayout.getBatchTile());
    // K batch count: last non-1 batch entry in lhsLayout (the reduction dim).
    // For rank-3 batch_matmul lhsLayout batch=[B,M,K], K is the last dim.
    int64_t kBatchCount = lhsLayout.getBatchTile().back();

    // Indexing maps for projecting result/K batch offsets → operand offsets.
    // The contract indexing maps tell us which iteration dims each operand uses.
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
    // The acc indexing map tells us which iter dims are used by the accumulator.
    // We invert it: for each result dim position i, iterDimForAccDim[i] = iter dim.
    AffineMap accMap = indexingMaps[2]; // acc is operand index 2
    AffineMap lhsMap = indexingMaps[0];
    AffineMap rhsMap = indexingMaps[1];

    // Build iteration-space → batch offset lookup. For acc dims, we know the
    // batch offsets directly from resultBatchOffsets. K is filled in per k loop.
    // iterBatch[d] = batch offset for iteration dim d, or -1 if unknown.
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

    // Helper: given an indexing map, build the operand batch offset vector
    // using the current result batch offsets and the k index.
    auto buildOperandBatchOff =
        [&](AffineMap map, ArrayRef<int64_t> resultBatchOffsets,
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
          opOff.push_back(0); // constant/symbol — shouldn't appear in batch maps
      }
      return opOff;
    };

    // Iterate over result batches (all acc batch dims, e.g. [B, M, N] for rank-3).
    SmallVector<int64_t> resultBatchShape(resultBatches);
    SmallVector<int64_t> resultBatchTile(resultBatches.size(), 1);
    for (auto [batchIdx, resultBatchOffsets] :
         llvm::enumerate(StaticTileOffsetRange(resultBatchShape, resultBatchTile))) {

      // Start accumulator slice from distAcc at this batch offset.
      Value accSlice = vector::ExtractOp::create(
          rewriter, loc, distAcc, resultBatchOffsets);

      // Accumulate over K batches.
      for (int64_t k = 0; k < kBatchCount; ++k) {
        // Project iteration-space offsets through each operand's indexing map.
        SmallVector<int64_t> lhsBatchOff =
            buildOperandBatchOff(lhsMap, resultBatchOffsets, k);
        SmallVector<int64_t> rhsBatchOff =
            buildOperandBatchOff(rhsMap, resultBatchOffsets, k);

        Value lhsSlice = vector::ExtractOp::create(
            rewriter, loc, distLhs, lhsBatchOff);
        Value rhsSlice = vector::ExtractOp::create(
            rewriter, loc, distRhs, rhsBatchOff);

        // Fix A-operand register order for nvgpu.mma.sync. PTX expects lane
        // registers {a0,a1,a2,a3} at rows {gID, gID+8, gID, gID+8}, but the
        // NestedLayout's row-major flatten of [outer_M, outer_K, elem_M,
        // elem_K] produces {a0,a2,a1,a3}. Transpose outer_M ↔ outer_K (or for
        // TF32 where outer_K=1, outer_M ↔ elem_K). Mirrors IREE's fix in
        // MMAAttr::buildMmaOperation (IREEGPUAttrs.cpp).
        Type lhsElemTy =
            mlir::cast<VectorType>(lhsSlice.getType()).getElementType();
        if (kind == V::MMA_SYNC_F16_16x8x16 ||
            kind == V::MMA_SYNC_BF16_16x8x16) {
          auto nonUnitTy = VectorType::get({2, 2, 2}, lhsElemTy);
          Value reshaped = vector::ShapeCastOp::create(
              rewriter, loc, nonUnitTy, lhsSlice);
          Value transposed = vector::TransposeOp::create(
              rewriter, loc, reshaped, ArrayRef<int64_t>{1, 0, 2});
          lhsSlice = vector::ShapeCastOp::create(
              rewriter, loc, lhsSlice.getType(), transposed);
        } else if (kind == V::MMA_SYNC_TF32_16x8x8) {
          auto nonUnitTy = VectorType::get({2, 2}, lhsElemTy);
          Value reshaped = vector::ShapeCastOp::create(
              rewriter, loc, nonUnitTy, lhsSlice);
          Value transposed = vector::TransposeOp::create(
              rewriter, loc, reshaped, ArrayRef<int64_t>{1, 0});
          lhsSlice = vector::ShapeCastOp::create(
              rewriter, loc, lhsSlice.getType(), transposed);
        }

        // Cast slices to expected vector types if needed.
        if (lhsSlice.getType() != lhsSliceTy)
          lhsSlice = vector::ShapeCastOp::create(
              rewriter, loc, lhsSliceTy, lhsSlice);
        if (rhsSlice.getType() != rhsSliceTy)
          rhsSlice = vector::ShapeCastOp::create(
              rewriter, loc, rhsSliceTy, rhsSlice);
        if (accSlice.getType() != accSliceTy)
          accSlice = vector::ShapeCastOp::create(
              rewriter, loc, accSliceTy, accSlice);

        bool tf32 = (kind == V::MMA_SYNC_TF32_16x8x8);
        accSlice = nvgpu::MmaSyncOp::create(
            rewriter, loc, lhsSlice, rhsSlice, accSlice,
            mmaShape, tf32);
      }

      // After nvgpu.mma.sync the result has the flat nvgpu shape (e.g. <2x2>).
      // vector.insert requires position_rank + source_rank == dest_rank, so
      // shape-cast back to the batch-extract slice shape [outer..., element...]
      // before inserting at the batch offsets.
      SmallVector<int64_t> batchSliceShape;
      for (int64_t v : accLayout.getOuterTile())   batchSliceShape.push_back(v);
      for (int64_t v : accLayout.getElementTile()) batchSliceShape.push_back(v);
      VectorType batchSliceTy = VectorType::get(batchSliceShape, accElemTy);
      if (accSlice.getType() != batchSliceTy)
        accSlice = vector::ShapeCastOp::create(rewriter, loc, batchSliceTy, accSlice);

      // Insert accumulated slice back into the result tile.
      finalTile = vector::InsertOp::create(
          rewriter, loc, accSlice, finalTile, resultBatchOffsets);
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
    RewritePatternSet &patterns, Value threadId, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize, int64_t maxBitsPerShuffle) {
  patterns.add<DistributeTransferRead>(patterns.getContext(), threadId,
                                        subgroupSize);
  patterns.add<DistributeTransferWrite>(patterns.getContext(), threadId,
                                         subgroupSize, workgroupSize);
  patterns.add<DistributeBroadcast, DistributeShapeCast>(patterns.getContext());
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                          maxBitsPerShuffle);
  patterns.add<DistributeContract>(patterns.getContext());
  patterns.add<NVIDIADistributeContract>(patterns.getContext());
}

} // namespace mlir::nova
