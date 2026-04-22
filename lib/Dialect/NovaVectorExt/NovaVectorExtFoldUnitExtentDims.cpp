//===- NovaVectorExtFoldUnitExtentDims.cpp - Drop unit dims from to_layout -===//
//
// Ports IREE's VectorExtFoldUnitExtentDims.cpp.
//
// Problem: after ConfigureTensorLayouts, every to_layout op carries a
// NestedLayoutAttr whose rank matches the tensor rank including the batch=1
// dim (e.g. subgroup_tile=[1,2,1] for tensor<1x128x16xf32>).
//
// When the standard LinalgFoldUnitExtentDimsPass later folds the linalg.generic
// (formerly batch_matmul), it inserts a tensor.collapse_shape between the
// to_layout output and the generic input:
//
//   %tl = to_layout %x {subgroup=[1,2,1]} : tensor<1x128x16xf32>
//   %c  = tensor.collapse_shape %tl ...   : tensor<128x16xf32>
//   %mm = linalg.generic ins(%c ...)
//
// VectorDistribute expects to_layout directly on the generic's operands.
// With a collapse_shape in between it misses the anchor → wrong codegen.
//
// Fix: for every to_layout whose tensor has a unit dim, replace it with:
//   1. tensor.extract_slice (rank-reducing) to drop the unit dim
//   2. a new to_layout with the rank-reduced tensor and projected layout
//   3. tensor.insert_slice to re-expand the result, preserving the original
//      type for downstream users
//
// After this pass the tensor.collapse_shape injected by LinalgFoldUnitExtent
// folds with the extract_slice, leaving to_layout directly on the 2D operand.
//
// Mirrors: iree/compiler/Codegen/Dialect/VectorExt/Transforms/
//          VectorExtFoldUnitExtentDims.cpp
//===----------------------------------------------------------------------===//

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nova::vec_ext;

namespace {

//===----------------------------------------------------------------------===//
// Pattern: drop unit dims from a to_layout op
//===----------------------------------------------------------------------===//

struct DropToLayoutUnitDims final : OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    // Only tensor-semantic to_layout (not vector form).
    auto inputTy = dyn_cast<RankedTensorType>(toLayoutOp.getInput().getType());
    if (!inputTy)
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "requires tensor semantics");

    ArrayRef<int64_t> shape = inputTy.getShape();

    // Collect unit dims and compute the target (rank-reduced) shape.
    SmallVector<bool> unitDims(shape.size(), false);
    SmallVector<int64_t> targetShape;
    bool hasUnitDims = false;
    for (auto [idx, size] : llvm::enumerate(shape)) {
      if (size == 1) {
        unitDims[idx] = true;
        hasUnitDims = true;
      } else {
        targetShape.push_back(size);
      }
    }

    if (!hasUnitDims)
      return rewriter.notifyMatchFailure(toLayoutOp, "no unit dims present");

    Location loc = toLayoutOp.getLoc();

    // 1. Rank-reducing extract_slice: tensor<1x128x16> → tensor<128x16>.
    FailureOr<Value> rankReducedExtract =
        tensor::ExtractSliceOp::rankReduceIfNeeded(
            rewriter, loc, toLayoutOp.getInput(), targetShape);
    if (failed(rankReducedExtract))
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "rankReduceIfNeeded failed");

    // 2. Project the NestedLayoutAttr: drop the same unit dims.
    //    layout.project(unitDims) removes each dim[i] where unitDims[i]=true.
    VectorLayoutInterface newLayout =
        cast<NestedLayoutAttr>(toLayoutOp.getLayout()).project(unitDims);

    // 3. New to_layout on the rank-reduced tensor with the projected layout.
    Value reduced = *rankReducedExtract;
    auto newToLayout = ToLayoutOp::create(
        rewriter, loc, reduced.getType(), reduced, newLayout,
        toLayoutOp.getSharedMemoryConversionAttr(),
        toLayoutOp.getMmaKindAttr());

    // 4. Re-expand back to the original shape via insert_slice so that
    //    existing users of the original to_layout result keep working.
    //    The destination is a fresh tensor.empty of the original size.
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, toLayoutOp.getInput());
    Value dest = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, inputTy.getElementType());

    int64_t rank = inputTy.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, dest);
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        toLayoutOp, newToLayout.getResult(), dest, offsets, sizes, strides);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaVectorExtFoldUnitExtentDimsPass
    : public PassWrapper<NovaVectorExtFoldUnitExtentDimsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaVectorExtFoldUnitExtentDimsPass)

  StringRef getArgument() const override {
    return "nova-vector-ext-fold-unit-extent-dims";
  }
  StringRef getDescription() const override {
    return "Drop unit extent dims from nova_vector_ext.to_layout ops and "
           "project their NestedLayoutAttr accordingly";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, NovaVectorExtDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DropToLayoutUnitDims>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::nova {

std::unique_ptr<Pass> createNovaVectorExtFoldUnitExtentDimsPass() {
  return std::make_unique<NovaVectorExtFoldUnitExtentDimsPass>();
}

void registerNovaVectorExtFoldUnitExtentDimsPass() {
  PassRegistration<NovaVectorExtFoldUnitExtentDimsPass>();
}

} // namespace mlir::nova
