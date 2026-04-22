#include "Compiler/Transforms/LLVMGPU/NovaVectorOpUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::nova {

std::pair<int, int> VectorContractOpInfo::getOperandMNIndex() const {
  return {(int)lhsMDims.back(), (int)rhsNDims.back()};
}

std::pair<int, int> VectorContractOpInfo::getOperandKIndex() const {
  return {(int)lhsKDim.back(), (int)rhsKDim.back()};
}

std::pair<int, int> VectorContractOpInfo::getResultMNIndex() const {
  return {(int)outMDims.back(), (int)outNDims.back()};
}

FailureOr<VectorContractOpInfo>
VectorContractOpInfo::inferFromIndexingMaps(ArrayRef<AffineMap> maps) {
  if (!llvm::all_of(maps, [](AffineMap map) {
        return map.isProjectedPermutation(/*allowZeroInResults=*/true);
      }))
    return failure();

  auto maybeContractionDims = linalg::inferContractionDims(maps);
  if (failed(maybeContractionDims))
    return failure();

  auto contractionDims = *maybeContractionDims;
  MLIRContext *ctx = maps[0].getContext();
  VectorContractOpInfo opInfo;

  for (auto m : contractionDims.m) {
    opInfo.lhsMDims.push_back(
        *maps[0].getResultPosition(getAffineDimExpr(m, ctx)));
    opInfo.outMDims.push_back(
        *maps[2].getResultPosition(getAffineDimExpr(m, ctx)));
  }
  for (auto n : contractionDims.n) {
    opInfo.rhsNDims.push_back(
        *maps[1].getResultPosition(getAffineDimExpr(n, ctx)));
    opInfo.outNDims.push_back(
        *maps[2].getResultPosition(getAffineDimExpr(n, ctx)));
  }
  for (auto k : contractionDims.k) {
    opInfo.lhsKDim.push_back(
        *maps[0].getResultPosition(getAffineDimExpr(k, ctx)));
    opInfo.rhsKDim.push_back(
        *maps[1].getResultPosition(getAffineDimExpr(k, ctx)));
  }

  opInfo.lhsUnitDims = maps[0].getBroadcastDims();
  opInfo.rhsUnitDims = maps[1].getBroadcastDims();
  opInfo.accUnitDims = maps[2].getBroadcastDims();
  opInfo.contractionDims = contractionDims;

  return opInfo;
}

} // namespace mlir::nova
