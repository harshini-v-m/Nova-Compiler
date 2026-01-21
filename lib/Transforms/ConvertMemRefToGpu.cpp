#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
#define GEN_PASS_DEF_CONVERTMEMREFTOGPU
#include "Compiler/Transforms/Passes.h.inc"

namespace mlir {
namespace nova {

static bool isMemorySpaceOne(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
    return intAttr.getInt() == 1;
  }
  return false;
}

class ConvertAllocOp : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (!isMemorySpaceOne(type.getMemorySpace()))
      return failure();

    rewriter.replaceOpWithNewOp<gpu::AllocOp>(
        op, type, /*asyncToken=*/Type(), /*asyncDependencies=*/ValueRange{},
        op.getDynamicSizes(), op.getSymbolOperands(), /*hostShared=*/false);
    return success();
  }
};

class ConvertDeallocOp : public OpRewritePattern<memref::DeallocOp> {
public:
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();
    MemRefType type = llvm::dyn_cast<MemRefType>(memref.getType());
    if (!type || !isMemorySpaceOne(type.getMemorySpace()))
      return failure();

    rewriter.replaceOpWithNewOp<gpu::DeallocOp>(op, TypeRange{}, ValueRange{},
                                                memref);
    return success();
  }
};
class ConvertMemrefOp : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = llvm::dyn_cast<MemRefType>(op.getSource().getType());
    auto dstType = llvm::dyn_cast<MemRefType>(op.getTarget().getType());

    if (!srcType || !dstType)
      return failure();

    // Convert to gpu.memcpy if either source or destination is in memory space
    // 1
    if (isMemorySpaceOne(srcType.getMemorySpace()) ||
        isMemorySpaceOne(dstType.getMemorySpace())) {
      // Synchronous memcpy (no async token)
      rewriter.replaceOpWithNewOp<gpu::MemcpyOp>(
          op,TypeRange{}, ValueRange{}, op.getTarget(), op.getSource());
      return success();
    }
    return failure();
  }
};
struct ConvertMemRefToGpuPass
    : public ::impl::ConvertMemRefToGpuBase<ConvertMemRefToGpuPass> {
  using ::impl::ConvertMemRefToGpuBase<
      ConvertMemRefToGpuPass>::ConvertMemRefToGpuBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<gpu::GPUDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    Operation *module = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Replace #nova.device attributes in all types and attributes
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](nova::NovaDeviceAttr attr) -> std::optional<Attribute> {
          if (attr.getValue().getValue() == "1")
            return IntegerAttr::get(IntegerType::get(ctx, 32), 1);
          return IntegerAttr::get(IntegerType::get(ctx, 32), 0);
        });

    replacer.addReplacement([&](MemRefType type) -> std::optional<Type> {
      Attribute space = type.getMemorySpace();
      if (!space)
        return std::nullopt;
      Attribute newSpace = replacer.replace(space);
      if (newSpace == space)
        return std::nullopt;
      return MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout(), newSpace);
    });

    replacer.addReplacement([&](RankedTensorType type) -> std::optional<Type> {
      Attribute encoding = type.getEncoding();
      if (!encoding)
        return std::nullopt;
      Attribute newEncoding = replacer.replace(encoding);
      if (newEncoding == encoding)
        return std::nullopt;
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newEncoding);
    });

    replacer.addReplacement(
        [&](DenseElementsAttr attr) -> std::optional<Attribute> {
          Type newType = replacer.replace(attr.getType());
          if (newType == attr.getType())
            return std::nullopt;
          return attr.reshape(llvm::cast<ShapedType>(newType));
        });

    replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    // 2. Explicitly handle function signatures which might be missed by
    // recursive replacement
    module->walk([&](func::FuncOp func) {
      SmallVector<Type> argTypes;
      for (auto type : func.getArgumentTypes()) {
        Type newType = replacer.replace(type);
        argTypes.push_back(newType);
      }

      SmallVector<Type> resultTypes;
      for (auto type : func.getResultTypes()) {
        Type newType = replacer.replace(type);
        resultTypes.push_back(newType);
      }

      func.setType(FunctionType::get(ctx, argTypes, resultTypes));
    });

    // 3. Convert memref.alloc/dealloc with memory space 1 to gpu.alloc/dealloc
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAllocOp, ConvertDeallocOp, ConvertMemrefOp>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertMemRefToGpuPass() {
  return std::make_unique<ConvertMemRefToGpuPass>();
}

} // namespace nova
} // namespace mlir
