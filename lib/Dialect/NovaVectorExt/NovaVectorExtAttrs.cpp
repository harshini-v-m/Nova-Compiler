#include <numeric>

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "nova-vector-ext-layout-attr"

using namespace mlir;

namespace mlir::nova::vec_ext {

using VectorValue = TypedValue<VectorType>;

//===----------------------------------------------------------------------===//
// basisFromSizesStrides helper (ported from iree/compiler/Utils/Indexing.cpp)
//
// Computes a delinearization basis from per-dimension sizes and strides.
// Dimensions with stride 0 are treated as size-1 dims placed at the end.
// Returns failure() if a consistent basis cannot be constructed.
//===----------------------------------------------------------------------===//
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
// NestedLayoutAttr
//===----------------------------------------------------------------------===//

VectorLayoutInterface
NestedLayoutAttr::project(ArrayRef<bool> droppedDims) const {
  assert(droppedDims.size() == (size_t)getRank() &&
         "droppedDims size must match layout rank");

  SmallVector<int64_t> subgroupCount;
  SmallVector<int64_t> batchCount;
  SmallVector<int64_t> outerCount;
  SmallVector<int64_t> threadCount;
  SmallVector<int64_t> elementCount;
  SmallVector<int64_t> subgroupStrides;
  SmallVector<int64_t> threadStrides;

  llvm::DenseMap<int64_t, int64_t> indexToRankReducedIndexMap;
  int64_t count = 0;
  for (auto [idx, isProjected] : llvm::enumerate(droppedDims)) {
    if (!isProjected) {
      subgroupCount.push_back(getSubgroupTile()[idx]);
      batchCount.push_back(getBatchTile()[idx]);
      outerCount.push_back(getOuterTile()[idx]);
      threadCount.push_back(getThreadTile()[idx]);
      elementCount.push_back(getElementTile()[idx]);
      subgroupStrides.push_back(getSubgroupStrides()[idx]);
      threadStrides.push_back(getThreadStrides()[idx]);
      indexToRankReducedIndexMap[idx] = count++;
    }
  }
  assert(count >= 0 && "unimplemented rank-0 vector");

  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

VectorLayoutInterface NestedLayoutAttr::apply(AffineMap map) const {
  assert(map.getNumDims() == (unsigned)getRank() &&
         "map domain size must match layout rank");

  SmallVector<int64_t> subgroupCount(map.getNumResults(), 1);
  SmallVector<int64_t> batchCount(map.getNumResults(), 1);
  SmallVector<int64_t> outerCount(map.getNumResults(), 1);
  SmallVector<int64_t> threadCount(map.getNumResults(), 1);
  SmallVector<int64_t> elementCount(map.getNumResults(), 1);
  SmallVector<int64_t> subgroupStrides(map.getNumResults(), 0);
  SmallVector<int64_t> threadStrides(map.getNumResults(), 0);

  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      int64_t pos = dim.getPosition();
      subgroupCount[idx] = getSubgroupTile()[pos];
      batchCount[idx] = getBatchTile()[pos];
      outerCount[idx] = getOuterTile()[pos];
      threadCount[idx] = getThreadTile()[pos];
      elementCount[idx] = getElementTile()[pos];
      subgroupStrides[idx] = getSubgroupStrides()[pos];
      threadStrides[idx] = getThreadStrides()[pos];
    }
  }

  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

VectorLayoutInterface
NestedLayoutAttr::reshape(ArrayRef<int64_t> newShape) const {
  SmallVector<int64_t> subgroupCount;
  SmallVector<int64_t> batchCount;
  SmallVector<int64_t> outerCount;
  SmallVector<int64_t> threadCount;
  SmallVector<int64_t> elementCount;
  SmallVector<int64_t> subgroupStrides;
  SmallVector<int64_t> threadStrides;

  // Process all shapes and layouts from the inside out (reverse order).
  SmallVector<int64_t> remainingLevels = {1, 1, 1, 1, 1};
  SmallVector<int64_t> remainingStrides = {0, 0};
  SmallVector<bool> unitDims = {true, true, true, true, true};
  int64_t currDim = getRank();
  int64_t currLevel = 5; // triggers init on first iteration
  int64_t minLevel = 0;

  for (int64_t dim : llvm::reverse(newShape)) {
    // levels: element(0), thread(1), outer(2), batch(3), subgroup(4)
    SmallVector<int64_t> levels(5, 1);
    SmallVector<int64_t> strides(2, 0);

    int64_t dimRemaining = dim;
    do {
      if (dimRemaining == 1)
        break;

      if (currLevel == 5) {
        if (currDim == 0)
          break;
        --currDim;
        currLevel = 0;
        minLevel = 0;
        remainingLevels = llvm::to_vector(
            llvm::reverse(getPackedShapeForUndistributedDim(currDim)));
        remainingStrides = {getThreadStrides()[currDim],
                            getSubgroupStrides()[currDim]};
        unitDims = llvm::map_to_vector(remainingLevels,
                                       [](int64_t v) { return v == 1; });
      }

      if (unitDims[currLevel]) {
        ++currLevel;
        continue;
      }
      if (remainingLevels[currLevel] == 1) {
        ++currLevel;
        ++minLevel;
        continue;
      }

      if (currLevel < minLevel) {
        LLVM_DEBUG(llvm::dbgs()
                   << "nova-vec-ext: invariant violated, trying to move below "
                      "minimum level, aborting layout reshaping\n");
        return VectorLayoutInterface();
      }

      int64_t consume = std::min(dimRemaining, remainingLevels[currLevel]);
      if (currLevel == 1) { // thread level
        if (strides[0] != 0 &&
            strides[0] * levels[currLevel] != remainingStrides[0]) {
          LLVM_DEBUG(llvm::dbgs()
                     << "nova-vec-ext: cannot consume stride, aborting "
                        "layout reshaping\n");
          return VectorLayoutInterface();
        }
        strides[0] = (strides[0] == 0 ? remainingStrides[0] : strides[0]);
        remainingStrides[0] *= consume;
      }
      if (currLevel == 4) { // subgroup level
        if (strides[1] != 0 &&
            strides[1] * levels[currLevel] != remainingStrides[1]) {
          return VectorLayoutInterface();
        }
        strides[1] = (strides[1] == 0 ? remainingStrides[1] : strides[1]);
        remainingStrides[1] *= consume;
      }

      levels[currLevel] *= consume;
      dimRemaining /= consume;
      remainingLevels[currLevel] /= consume;
    } while (true);

    assert(dimRemaining == 1 && "cannot reshape, remaining dim not consumed");
    elementCount.push_back(levels[0]);
    threadCount.push_back(levels[1]);
    outerCount.push_back(levels[2]);
    batchCount.push_back(levels[3]);
    subgroupCount.push_back(levels[4]);
    threadStrides.push_back(strides[0]);
    subgroupStrides.push_back(strides[1]);

    if (llvm::all_of(remainingLevels, [](int64_t v) { return v == 1; })) {
      minLevel = 0;
      currLevel = 5;
    }
  }

  std::reverse(subgroupCount.begin(), subgroupCount.end());
  std::reverse(batchCount.begin(), batchCount.end());
  std::reverse(outerCount.begin(), outerCount.end());
  std::reverse(threadCount.begin(), threadCount.end());
  std::reverse(elementCount.begin(), elementCount.end());
  std::reverse(threadStrides.begin(), threadStrides.end());
  std::reverse(subgroupStrides.begin(), subgroupStrides.end());

  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

VectorLayoutInterface
NestedLayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  SmallVector<int64_t> subgroupCount =
      applyPermutation(getSubgroupTile(), permutation);
  SmallVector<int64_t> batchCount =
      applyPermutation(getBatchTile(), permutation);
  SmallVector<int64_t> outerCount =
      applyPermutation(getOuterTile(), permutation);
  SmallVector<int64_t> threadCount =
      applyPermutation(getThreadTile(), permutation);
  SmallVector<int64_t> elementCount =
      applyPermutation(getElementTile(), permutation);
  SmallVector<int64_t> subgroupStrides =
      applyPermutation(getSubgroupStrides(), permutation);
  SmallVector<int64_t> threadStrides =
      applyPermutation(getThreadStrides(), permutation);
  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

SmallVector<int64_t> NestedLayoutAttr::getDistributedShape() const {
  SmallVector<int64_t> shape;
  shape.append(getBatchTile().begin(), getBatchTile().end());
  shape.append(getOuterTile().begin(), getOuterTile().end());
  shape.append(getElementTile().begin(), getElementTile().end());
  return shape;
}

SmallVector<int64_t> NestedLayoutAttr::getUndistributedPackedShape() const {
  SmallVector<int64_t> shape;
  int64_t rank = getRank();
  shape.reserve(rank * 5);
  shape.append(getSubgroupTile().begin(), getSubgroupTile().end());
  shape.append(getBatchTile().begin(), getBatchTile().end());
  shape.append(getOuterTile().begin(), getOuterTile().end());
  shape.append(getThreadTile().begin(), getThreadTile().end());
  shape.append(getElementTile().begin(), getElementTile().end());
  return shape;
}

SmallVector<int64_t> NestedLayoutAttr::getUndistributedShape() const {
  int64_t rank = getRank();
  SmallVector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t i : llvm::seq<int64_t>(rank)) {
    int64_t expectedDimLen = getSubgroupTile()[i] * getBatchTile()[i] *
                             getOuterTile()[i] * getThreadTile()[i] *
                             getElementTile()[i];
    shape.push_back(expectedDimLen);
  }
  return shape;
}

SmallVector<int64_t>
NestedLayoutAttr::getPackedShapeForUndistributedDim(int64_t dim) const {
  return {getSubgroupTile()[dim], getBatchTile()[dim], getOuterTile()[dim],
          getThreadTile()[dim],   getElementTile()[dim]};
}

SmallVector<int64_t> NestedLayoutAttr::getDistributedUnpackedShape() const {
  SmallVector<int64_t> shape;
  shape.reserve(getRank());
  for (auto [batch, outer, element] :
       llvm::zip(getBatchTile(), getOuterTile(), getElementTile())) {
    shape.push_back(batch * outer * element);
  }
  return shape;
}

int64_t NestedLayoutAttr::getRank() const {
  return getBatchTile().size();
}

LogicalResult NestedLayoutAttr::isValidLayout(ShapedType shapeTy,
                                              Location loc) const {
  int64_t rank = getRank();
  ArrayRef<int64_t> shape = shapeTy.getShape();
  if ((int64_t)shape.size() != rank) {
    return emitError(loc, "Rank of vector (")
           << shape.size() << ") does not match rank of layout (" << rank
           << ").";
  }
  for (int i = 0; i < rank; ++i) {
    int64_t expectedShape = getSubgroupTile()[i] * getBatchTile()[i] *
                            getOuterTile()[i] * getThreadTile()[i] *
                            getElementTile()[i];
    if (ShapedType::isStatic(shape[i]) && expectedShape != shape[i]) {
      std::string layoutStr;
      llvm::raw_string_ostream layoutOs(layoutStr);
      printStripped(layoutOs);
      return emitError(loc, "Vector shape: [")
             << llvm::interleaved(shape) << "] does not match the layout ("
             << layoutStr + ") at dim " << i
             << ". Dimension expected by layout: " << expectedShape
             << " actual: " << shape[i];
    }
  }
  return success();
}

NestedLayoutAttr NestedLayoutAttr::getChecked(
    llvm::function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<int64_t> subgroupTile, ArrayRef<int64_t> batchTile,
    ArrayRef<int64_t> outerTile, ArrayRef<int64_t> threadTile,
    ArrayRef<int64_t> elementTile, ArrayRef<int64_t> subgroupStrides,
    ArrayRef<int64_t> threadStrides) {
  if (failed(NestedLayoutAttr::verify(emitError, subgroupTile, batchTile,
                                      outerTile, threadTile, elementTile,
                                      subgroupStrides, threadStrides))) {
    return NestedLayoutAttr();
  }
  return NestedLayoutAttr::get(context, subgroupTile, batchTile, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

NestedLayoutAttr NestedLayoutAttr::get(
    MLIRContext *context, ArrayRef<int64_t> subgroupTile,
    ArrayRef<int64_t> batchTile, ArrayRef<int64_t> outerTile,
    ArrayRef<int64_t> threadTile, ArrayRef<int64_t> elementTile,
    ArrayRef<int64_t> subgroupStrides, ArrayRef<int64_t> threadStrides) {

  SmallVector<int64_t> normalizedSubgroupStrides(subgroupStrides);
  SmallVector<int64_t> normalizedThreadStrides(threadStrides);

  // Normalize strides for size-1 dims to 0 for consistency.
  for (auto [stride, size] :
       llvm::zip_equal(normalizedSubgroupStrides, subgroupTile)) {
    if (size == 1)
      stride = 0;
  }
  for (auto [stride, size] :
       llvm::zip_equal(normalizedThreadStrides, threadTile)) {
    if (size == 1)
      stride = 0;
  }

  return Base::get(context, subgroupTile, batchTile, outerTile, threadTile,
                   elementTile, normalizedSubgroupStrides,
                   normalizedThreadStrides);
}

static SmallVector<int64_t> appendDims(ArrayRef<int64_t> tileLens,
                                       ArrayRef<int64_t> appendLens) {
  SmallVector<int64_t> result = llvm::to_vector(tileLens);
  result.insert(result.end(), appendLens.begin(), appendLens.end());
  return result;
}

NestedLayoutAttr NestedLayoutAttr::get(MLIRContext *context,
                                       NestedLayoutAttr source,
                                       ArrayRef<int64_t> appendSubGroupLens,
                                       ArrayRef<int64_t> appendBatchLens,
                                       ArrayRef<int64_t> appendOuterLens,
                                       ArrayRef<int64_t> appendThreadLens,
                                       ArrayRef<int64_t> appendElementLens,
                                       ArrayRef<int64_t> appendSubgroupStrides,
                                       ArrayRef<int64_t> appendThreadStrides) {
  return NestedLayoutAttr::get(
      context,
      appendDims(source.getSubgroupTile(), appendSubGroupLens),
      appendDims(source.getBatchTile(), appendBatchLens),
      appendDims(source.getOuterTile(), appendOuterLens),
      appendDims(source.getThreadTile(), appendThreadLens),
      appendDims(source.getElementTile(), appendElementLens),
      appendDims(source.getSubgroupStrides(), appendSubgroupStrides),
      appendDims(source.getThreadStrides(), appendThreadStrides));
}

VectorLayoutInterface
NestedLayoutAttr::getRecombinedLayout(ArrayRef<VectorLayoutInterface> layouts,
                                      ArrayRef<AffineMap> maps,
                                      AffineMap resultMap) {
  constexpr int64_t kInvalid = -1;
  if (!llvm::all_of(layouts, llvm::IsaPred<NestedLayoutAttr>))
    return NestedLayoutAttr();

  MLIRContext *context = resultMap.getContext();
  SmallVector<NestedLayoutAttr> nestedLayouts;
  llvm::transform(layouts, std::back_inserter(nestedLayouts),
                  [](VectorLayoutInterface l) {
                    return cast<NestedLayoutAttr>(l);
                  });

  int64_t resRank = resultMap.getNumResults();
  SmallVector<int64_t> subgroupTile(resRank, kInvalid);
  SmallVector<int64_t> batchTile(resRank, kInvalid);
  SmallVector<int64_t> outerTile(resRank, kInvalid);
  SmallVector<int64_t> threadTile(resRank, kInvalid);
  SmallVector<int64_t> elementTile(resRank, kInvalid);
  SmallVector<int64_t> subgroupStrides(resRank, kInvalid);
  SmallVector<int64_t> threadStrides(resRank, kInvalid);

  auto checkedUpdate = [&](int64_t &data, int64_t v) -> bool {
    if (data != kInvalid && data != v)
      return false;
    data = v;
    return true;
  };

  for (auto [layout, indexingMap] : llvm::zip(nestedLayouts, maps)) {
    for (int64_t resultIdx : llvm::seq<int64_t>(indexingMap.getNumResults())) {
      int64_t iterSpacePos = indexingMap.getDimPosition(resultIdx);
      std::optional<unsigned> mayBeResultPos =
          resultMap.getResultPosition(getAffineDimExpr(iterSpacePos, context));
      if (!mayBeResultPos)
        continue;
      int64_t resultPos = *mayBeResultPos;

      if (!checkedUpdate(subgroupTile[resultPos], layout.getSubgroupTile()[resultIdx]) ||
          !checkedUpdate(batchTile[resultPos], layout.getBatchTile()[resultIdx]) ||
          !checkedUpdate(outerTile[resultPos], layout.getOuterTile()[resultIdx]) ||
          !checkedUpdate(threadTile[resultPos], layout.getThreadTile()[resultIdx]) ||
          !checkedUpdate(elementTile[resultPos], layout.getElementTile()[resultIdx]) ||
          !checkedUpdate(subgroupStrides[resultPos], layout.getSubgroupStrides()[resultIdx]) ||
          !checkedUpdate(threadStrides[resultPos], layout.getThreadStrides()[resultIdx]))
        return NestedLayoutAttr();
    }
  }

  for (const auto &tile :
       {subgroupTile, batchTile, outerTile, threadTile, subgroupStrides,
        threadStrides}) {
    if (llvm::any_of(tile, [](int64_t v) { return v == kInvalid; }))
      return NestedLayoutAttr();
  }

  return NestedLayoutAttr::get(context, subgroupTile, batchTile, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

LogicalResult NestedLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> subgroupTile, ArrayRef<int64_t> batchTile,
    ArrayRef<int64_t> outerTile, ArrayRef<int64_t> threadTile,
    ArrayRef<int64_t> elementTile, ArrayRef<int64_t> subgroupStrides,
    ArrayRef<int64_t> threadStrides) {
  size_t rank = subgroupTile.size();
  auto checkTile = [&](ArrayRef<int64_t> tileShape) -> LogicalResult {
    if (tileShape.size() != rank) {
      emitError() << "all fields must have the same rank as the layout";
      return failure();
    }
    return success();
  };
  if (failed(checkTile(subgroupTile)) || failed(checkTile(batchTile)) ||
      failed(checkTile(outerTile)) || failed(checkTile(threadTile)) ||
      failed(checkTile(elementTile)) || failed(checkTile(subgroupStrides)) ||
      failed(checkTile(threadStrides)))
    return failure();
  return success();
}

SmallVector<Value>
NestedLayoutAttr::computeThreadIds(Value threadId, int64_t subgroupSize,
                                   RewriterBase &rewriter) const {
  SmallVector<Value> virtualTids;
  Location loc = threadId.getLoc();

  SmallVector<int64_t> subgroupBasis, threadBasis;
  SmallVector<size_t> subgroupDimToResult, threadDimToResult;

  if (failed(basisFromSizesStrides(getSubgroupTile(), getSubgroupStrides(),
                                   subgroupBasis, subgroupDimToResult)))
    return {};
  if (failed(basisFromSizesStrides(getThreadTile(), getThreadStrides(),
                                   threadBasis, threadDimToResult)))
    return {};

  subgroupBasis.push_back(subgroupSize);

  auto subgroupSplit = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, threadId, subgroupBasis, /*hasOuterBound=*/false);
  auto threadSplit = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, threadId, threadBasis, /*hasOuterBound=*/false);

  llvm::transform(subgroupDimToResult, std::back_inserter(virtualTids),
                  [&](size_t idx) { return subgroupSplit.getResult(idx); });
  llvm::transform(threadDimToResult, std::back_inserter(virtualTids),
                  [&](size_t idx) { return threadSplit.getResult(idx); });

  return virtualTids;
}

} // namespace mlir::nova::vec_ext

using namespace mlir::nova::vec_ext;

#define GET_ATTRDEF_CLASSES
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtAttrs.cpp.inc"

void NovaVectorExtDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtAttrs.cpp.inc"
      >();
}
