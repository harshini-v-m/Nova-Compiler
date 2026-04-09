#include "Compiler/Dialect/nova/NovaVectorLayoutAttr.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::nova;

//===----------------------------------------------------------------------===//
// Serialization helpers
//===----------------------------------------------------------------------===//

static ArrayAttr makeI64Array(MLIRContext *ctx,
                               llvm::ArrayRef<int64_t> vals) {
  llvm::SmallVector<Attribute> attrs;
  attrs.reserve(vals.size());
  auto i64 = IntegerType::get(ctx, 64);
  for (int64_t v : vals)
    attrs.push_back(IntegerAttr::get(i64, v));
  return ArrayAttr::get(ctx, attrs);
}

static LogicalResult extractI64Array(DictionaryAttr dict, StringRef key,
                                      llvm::SmallVector<int64_t> &out) {
  Attribute rawAttr = dict.get(key);
  if (!rawAttr)
    return failure();
  auto arr = dyn_cast<ArrayAttr>(rawAttr);
  if (!arr)
    return failure();
  out.reserve(arr.size());
  for (Attribute elem : arr) {
    auto intAttr = dyn_cast<IntegerAttr>(elem);
    if (!intAttr)
      return failure();
    out.push_back(intAttr.getInt());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NovaVectorLayout::toAttr
//===----------------------------------------------------------------------===//

DictionaryAttr NovaVectorLayout::toAttr(MLIRContext *ctx) const {
  llvm::SmallVector<NamedAttribute> fields;
  fields.emplace_back(StringAttr::get(ctx, "sg_counts"),
                      makeI64Array(ctx, subgroupCounts));
  fields.emplace_back(StringAttr::get(ctx, "batch_counts"),
                      makeI64Array(ctx, batchCounts));
  fields.emplace_back(StringAttr::get(ctx, "outer_counts"),
                      makeI64Array(ctx, outerCounts));
  fields.emplace_back(StringAttr::get(ctx, "thread_counts"),
                      makeI64Array(ctx, threadCounts));
  fields.emplace_back(StringAttr::get(ctx, "elem_counts"),
                      makeI64Array(ctx, elementCounts));
  fields.emplace_back(StringAttr::get(ctx, "sg_strides"),
                      makeI64Array(ctx, subgroupStrides));
  fields.emplace_back(StringAttr::get(ctx, "thread_strides"),
                      makeI64Array(ctx, threadStrides));
  fields.emplace_back(StringAttr::get(ctx, "shared_mem"),
                      BoolAttr::get(ctx, sharedMemory));
  return DictionaryAttr::get(ctx, fields);
}

//===----------------------------------------------------------------------===//
// NovaVectorLayout::fromAttr
//===----------------------------------------------------------------------===//

FailureOr<NovaVectorLayout> NovaVectorLayout::fromAttr(DictionaryAttr attr) {
  if (!attr)
    return failure();

  NovaVectorLayout layout;

  if (failed(extractI64Array(attr, "sg_counts",     layout.subgroupCounts))  ||
      failed(extractI64Array(attr, "batch_counts",  layout.batchCounts))     ||
      failed(extractI64Array(attr, "outer_counts",  layout.outerCounts))     ||
      failed(extractI64Array(attr, "thread_counts", layout.threadCounts))    ||
      failed(extractI64Array(attr, "elem_counts",   layout.elementCounts))   ||
      failed(extractI64Array(attr, "sg_strides",    layout.subgroupStrides)) ||
      failed(extractI64Array(attr, "thread_strides",layout.threadStrides)))
    return failure();

  if (Attribute raw = attr.get("shared_mem")) {
    auto boolAttr = dyn_cast<BoolAttr>(raw);
    if (!boolAttr)
      return failure();
    layout.sharedMemory = boolAttr.getValue();
  }

  return layout;
}

//===----------------------------------------------------------------------===//
// NovaVectorLayout::apply
//===----------------------------------------------------------------------===//

NovaVectorLayout NovaVectorLayout::apply(AffineMap map, int /*fullRank*/) const {
  // Project the full-rank layout through an indexing map.
  //
  // The map must be composed entirely of AffineDimExpr results (simple
  // projections / permutations).  For each result dim i, we look up the
  // source dimension index j and copy the j-th layout entry to position i.
  //
  // Example
  //   Full rank (B, M, N, K): subgroupCounts = [8, 64, 192, 1]
  //   LHS map: (B, M, N, K) -> (B, M, K)     i.e. dims [0, 1, 3]
  //   Result:  subgroupCounts                = [8, 64,   1]   (N dropped)

  int resultRank = static_cast<int>(map.getNumResults());

  NovaVectorLayout result;
  result.subgroupCounts.assign(resultRank, 1);
  result.batchCounts.assign(resultRank, 1);
  result.outerCounts.assign(resultRank, 1);
  result.threadCounts.assign(resultRank, 1);
  result.elementCounts.assign(resultRank, 1);
  result.subgroupStrides.assign(resultRank, 0);
  result.threadStrides.assign(resultRank, 0);
  result.sharedMemory = sharedMemory;

  for (int i = 0; i < resultRank; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
    if (!dimExpr)
      continue; // non-dim expression: leave defaults (count=1, stride=0)

    int src = static_cast<int>(dimExpr.getPosition());
    if (src >= static_cast<int>(subgroupCounts.size()))
      continue;

    result.subgroupCounts[i]  = subgroupCounts[src];
    result.batchCounts[i]     = batchCounts[src];
    result.outerCounts[i]     = outerCounts[src];
    result.threadCounts[i]    = threadCounts[src];
    result.elementCounts[i]   = elementCounts[src];
    result.subgroupStrides[i] = subgroupStrides[src];
    result.threadStrides[i]   = threadStrides[src];
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Op-level helpers
//===----------------------------------------------------------------------===//

NovaVectorLayout mlir::nova::getVectorLayout(Operation *op) {
  auto dictAttr = op->getAttrOfType<DictionaryAttr>(kNovaLayoutAttr);
  if (!dictAttr)
    return {};
  auto result = NovaVectorLayout::fromAttr(dictAttr);
  if (failed(result))
    return {};
  return *result;
}

void mlir::nova::setVectorLayout(Operation *op,
                                  const NovaVectorLayout &layout) {
  op->setAttr(kNovaLayoutAttr, layout.toAttr(op->getContext()));
}

bool mlir::nova::hasVectorLayout(Operation *op) {
  return op->hasAttr(kNovaLayoutAttr);
}
