#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
#define GEN_PASS_DEF_CONVERTMEMREFTOGPU
#include "Compiler/Transforms/Passes.h.inc"

namespace mlir {
namespace nova {

// static bool isMemorySpaceOne(Attribute memorySpace) {
//   if (!memorySpace)
//     return false;
//   if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
//     return intAttr.getInt() == 1;
//   }
//   if (auto novaAttr = llvm::dyn_cast<nova::NovaDeviceAttr>(memorySpace)) {
//     return novaAttr.getValue().getValue() == "1";
//   }
//   return false;
// }

class ConvertAllocOp : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (mlir::isa_and_present<gpu::AddressSpaceAttr>(type.getMemorySpace()))
      return failure();

    // Inside GPU kernels: skip — these are handled by
    // NovaConvertSharedMemAllocs which converts them to memref.global +
    // memref.get_global (device global memory). Converting to alloca here
    // would create per-thread stack allocations that are catastrophic for
    // large workspace buffers (e.g. 1x1024x384xf32 = 1.5 MB per thread).
    if (op->getParentOfType<gpu::GPUFuncOp>()) {
      rewriter.replaceOpWithNewOp<memref::AllocaOp>(
          op, type, op.getDynamicSizes(), op.getSymbolOperands());
      return success();
    }

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
    if (type && mlir::isa_and_present<gpu::AddressSpaceAttr>(type.getMemorySpace()))
      return failure();

    // Inside GPU kernels: erase deallocs (alloca has automatic lifetime).
    if (op->getParentOfType<gpu::GPUFuncOp>()) {
      rewriter.eraseOp(op);
      return success();
    }

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

    // Guard: do NOT convert copies that involve a GPU-specific address space
    // (private or workgroup). Those are intra-kernel copies (e.g. padding
    // copies from a plain memref into a thread-private register tile, or
    // shared-memory fills). After GpuKernelOutliningPass these end up inside
    // gpu.func where gpu.memcpy has no PTX equivalent and is marked illegal
    // by createConvertGpuOpsToNVVMOps.
    // Plain memref.copy ops in GPU address spaces are correctly lowered by
    // FinalizeMemRefToLLVM via load/store sequences.
    auto isGpuAddrSpace = [](MemRefType t) {
      return mlir::isa_and_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
    };
    if (isGpuAddrSpace(srcType) || isGpuAddrSpace(dstType))
      return failure();

    // Only convert plain (no GPU address space) host-side copies to
    // gpu.memcpy — these are cross-kernel host↔device transfers produced
    // after ConvertMemRefToGpu promotes memref.alloc → gpu.alloc.
    rewriter.replaceOpWithNewOp<gpu::MemcpyOp>(
        op, TypeRange{}, ValueRange{}, op.getTarget(), op.getSource());
    return success();
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
            return IntegerAttr::get(IntegerType::get(ctx, 64), 1);
          return IntegerAttr::get(IntegerType::get(ctx, 64), 0);
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

    // replacer.addReplacement([&](RankedTensorType type) -> std::optional<Type> {
    //   Attribute encoding = type.getEncoding();
    //   if (!encoding)
    //     return std::nullopt;
    //   Attribute newEncoding = replacer.replace(encoding);
    //   if (newEncoding == encoding)
    //     return std::nullopt;
    //   return RankedTensorType::get(type.getShape(), type.getElementType(),
    //                                newEncoding);
    // });

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

    // 3. Explicitly handle function signatures which might be missed by
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

    // 4. Convert memref.alloc/dealloc with memory space 1 to gpu.alloc/dealloc
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAllocOp, ConvertDeallocOp, ConvertMemrefOp>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    // 5. Fix host-side memref.store ops that now target gpu.alloc'd buffers.
    //    After pattern rewriting, memref.alloc → gpu.alloc, but any host-side
    //    memref.store (e.g. scalar reduction accumulator zero-init) still writes
    //    to the device pointer from the host, causing a segfault.
    //    Convert: memref.store %val, %gpu_buf[indices...]
    //    Into:    %host = memref.alloca; memref.store %val, %host[...];
    //             gpu.memcpy %gpu_buf, %host
    SmallVector<memref::StoreOp> storesToFix;
    module->walk([&](memref::StoreOp store) {
      // Only fix stores on the host side (not inside gpu.func).
      if (store->getParentOfType<gpu::GPUFuncOp>())
        return;
      Value target = store.getMemref();
      if (target.getDefiningOp<gpu::AllocOp>())
        storesToFix.push_back(store);
    });

    for (auto store : storesToFix) {
      Location loc = store.getLoc();
      Value gpuBuf = store.getMemref();
      auto memrefType = cast<MemRefType>(gpuBuf.getType());

      // Find the outermost loop ancestor (if any) so we can hoist the alloca
      // above it.  Without hoisting, an alloca inside a loop body causes a
      // new stack allocation per iteration → stack overflow for large trip
      // counts (e.g. 50304-element reductions lowered to host-side loops).
      Operation *allocaInsertPt = store.getOperation();
      Operation *parent = store->getParentOp();
      while (parent && !isa<func::FuncOp>(parent)) {
        if (isa<scf::ForOp, scf::ForallOp, scf::WhileOp>(parent))
          allocaInsertPt = parent;
        parent = parent->getParentOp();
      }

      // Create alloca at the hoisted point (before outermost loop, or at the
      // store location if not inside any loop).
      OpBuilder builder(allocaInsertPt);
      auto hostBuf = builder.create<memref::AllocaOp>(loc, memrefType);

      // Store and memcpy stay at the original store location.
      builder.setInsertionPoint(store);
      builder.create<memref::StoreOp>(loc, store.getValue(), hostBuf,
                                      store.getIndices());

      // Copy from host to device.
      builder.create<gpu::MemcpyOp>(loc, /*asyncToken=*/Type(),
                                    /*asyncDeps=*/ValueRange{},
                                    /*dst=*/gpuBuf, /*src=*/hostBuf);

      // Erase the original store.
      store.erase();
    }

    // 6. Copy host-side memref.get_global constants to GPU memory before
    //    they are passed to gpu.launch_func arguments.
    //    After kernel outlining (Step 12), constants that were used inside
    //    gpu.launch become gpu.launch_func arguments, but their data
    //    remains in host memory — the GPU kernel cannot dereference host
    //    pointers via ld.global, causing CUDA_ERROR_ILLEGAL_ADDRESS.
    module->walk([&](memref::GetGlobalOp getGlobal) {
      // Only handle host-side get_globals (not inside gpu.func).
      if (getGlobal->getParentOfType<gpu::GPUFuncOp>())
        return;

      // Check if any user is a gpu.launch_func.
      bool usedByKernel = false;
      for (Operation *user : getGlobal->getUsers()) {
        if (isa<gpu::LaunchFuncOp>(user)) {
          usedByKernel = true;
          break;
        }
      }
      if (!usedByKernel)
        return;

      auto memrefType = cast<MemRefType>(getGlobal.getType());
      Location loc = getGlobal.getLoc();

      // Create gpu.alloc + gpu.memcpy right after the get_global.
      OpBuilder builder(getGlobal->getNextNode());
      auto gpuBuf = builder.create<gpu::AllocOp>(
          loc, memrefType, /*asyncToken=*/Type(),
          /*asyncDependencies=*/ValueRange{},
          /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{},
          /*hostShared=*/false);
      builder.create<gpu::MemcpyOp>(loc, /*asyncToken=*/Type(),
                                    /*asyncDeps=*/ValueRange{},
                                    /*dst=*/gpuBuf.getMemref(),
                                    /*src=*/getGlobal.getResult());

      // Replace only the uses inside gpu.launch_func ops.
      getGlobal.getResult().replaceUsesWithIf(
          gpuBuf.getMemref(), [](OpOperand &use) {
            return isa<gpu::LaunchFuncOp>(use.getOwner());
          });
    });
  }
};

std::unique_ptr<Pass> createConvertMemRefToGpuPass() {
  return std::make_unique<ConvertMemRefToGpuPass>();
}

} // namespace nova
} // namespace mlir
