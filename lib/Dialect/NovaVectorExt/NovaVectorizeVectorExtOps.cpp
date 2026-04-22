//===- NovaVectorizeVectorExtOps.cpp - Vectorize tensor to_layout ops -----===//
//
// Ports IREE's VectorizeIREEVectorExtOpsPass.
//
// After OptimizeTensorInsertExtractSlices the K-loop iter_arg is 2D static and
// all nova_vector_ext.to_layout ops are tensor-semantic
// (input : tensor<MxKxf32>, output : tensor<MxKxf32>).
// VectorDistribute requires them to be vector-semantic instead.
//
// Pattern VectorizeToLayoutOpPattern converts each tensor to_layout:
//
//   %out = nova_vector_ext.to_layout %in to layout(...) : tensor<MxKxf32>
//
// into:
//
//   %vec  = vector.transfer_read %in[0,0] : tensor<MxKxf32>, vector<MxKxf32>
//   %vout = nova_vector_ext.to_layout %vec to layout(...) : vector<MxKxf32>
//   %out  = vector.transfer_write %vout, %in[0,0] : vector<MxKxf32>, tensor<MxKxf32>
//
// Mirrors IREE's VectorizeIREEVectorExtOpsPass.
//===----------------------------------------------------------------------===//

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nova::vec_ext;

namespace {

//===----------------------------------------------------------------------===//
// VectorizeToLayoutOpPattern
//===----------------------------------------------------------------------===//

struct VectorizeToLayoutOpPattern final : OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    // Only fire on tensor-semantic to_layout.
    if (!toLayoutOp.hasTensorSemantics())
      return failure();

    Location loc = toLayoutOp.getLoc();
    auto inputTy = cast<RankedTensorType>(toLayoutOp.getInput().getType());

    // Undistributed shape = full vector shape before distribution.
    SmallVector<int64_t> readShape =
        toLayoutOp.getLayout().getUndistributedShape();
    auto vectorTy = VectorType::get(readShape, inputTy.getElementType());

    // Build zero indices — one per input rank.
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(inputTy.getRank(), zero);
    SmallVector<bool> inBoundsFlags(vectorTy.getRank(), true);

    // tensor → vector
    //   TransferReadOp::build(vectorType, source, indices, padding=nullopt, inBounds)
    //   uses getTransferMinorIdentityMap internally.
    auto readOp = rewriter.create<vector::TransferReadOp>(
        loc, vectorTy, toLayoutOp.getInput(), indices,
        /*padding=*/std::optional<Value>{}, inBoundsFlags);

    // vector to_layout (vector-semantic, same layout/mma_kind/shared_memory attrs)
    Attribute mmaKind = toLayoutOp.getMmaKind().value_or(Attribute{});
    auto newLayoutOp = rewriter.create<ToLayoutOp>(
        loc, readOp.getResult(), toLayoutOp.getLayout(),
        mmaKind,
        static_cast<bool>(toLayoutOp.getSharedMemoryConversion()));

    // vector → tensor (write result back; uses dest = original input tensor)
    //   TransferWriteOp::build(vector, dest, indices, inBounds)
    //   uses getTransferMinorIdentityMap internally.
    auto writeOp = rewriter.create<vector::TransferWriteOp>(
        loc, newLayoutOp.getOutput(), toLayoutOp.getInput(), indices,
        inBoundsFlags);

    rewriter.replaceOp(toLayoutOp, writeOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaVectorizeVectorExtOpsPass
    : public PassWrapper<NovaVectorizeVectorExtOpsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaVectorizeVectorExtOpsPass)

  StringRef getArgument() const override {
    return "nova-vectorize-vector-ext-ops";
  }
  StringRef getDescription() const override {
    return "Convert tensor-semantic nova_vector_ext.to_layout ops to "
           "vector-semantic via vector.transfer_read/write wrapping. "
           "Must run before GenericVectorization.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, arith::ArithDialect,
                    NovaVectorExtDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<VectorizeToLayoutOpPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::nova {

std::unique_ptr<Pass> createNovaVectorizeVectorExtOpsPass() {
  return std::make_unique<NovaVectorizeVectorExtOpsPass>();
}

void registerNovaVectorizeVectorExtOpsPass() {
  PassRegistration<NovaVectorizeVectorExtOpsPass>();
}

} // namespace mlir::nova
