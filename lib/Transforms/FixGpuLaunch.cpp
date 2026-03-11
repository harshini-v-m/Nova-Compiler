#include "Compiler/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace nova {

// Helper to declare runtime functions
static LLVM::LLVMFuncOp getOrDeclareFunc(ModuleOp module, OpBuilder &rewriter,
                                         StringRef name, Type resultTy,
                                         ArrayRef<Type> argTypes) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;

  // Check for collision
  if (auto existing = SymbolTable::lookupSymbolIn(module, name)) {
    if (auto llvmFunc = llvm::dyn_cast<LLVM::LLVMFuncOp>(existing))
      return llvmFunc;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType = LLVM::LLVMFunctionType::get(resultTy, argTypes);
  return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType);
}

class ConvertGpuAllocToCall : public OpRewritePattern<gpu::AllocOp> {
public:
  using OpRewritePattern<gpu::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();

    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMalloc = getOrDeclareFunc(module, rewriter, "cudaMallocAsync", int32Ty,
                                       {genericPtrTy, int64Ty, genericPtrTy});

    MemRefType memRefType = op.getType();
    unsigned addressSpace = 0;
    if (auto intAttr =
            llvm::dyn_cast_or_null<IntegerAttr>(memRefType.getMemorySpace())) {
      addressSpace = intAttr.getInt();
    } else if (auto addrSpaceAttr = llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
                   memRefType.getMemorySpace())) {
      addressSpace = (unsigned)addrSpaceAttr.getValue();
    }
    auto devicePtrTy = LLVM::LLVMPointerType::get(ctx, addressSpace);

    Type elementTy = memRefType.getElementType();
    int64_t elementSize = 0;
    if (elementTy.isIntOrFloat())
      elementSize = elementTy.getIntOrFloatBitWidth() / 8;
    else if (elementTy.isIndex())
      elementSize = 8; // Treat index as 64-bit

    if (elementSize == 0)
      elementSize = 1;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));

    unsigned dynamicIdx = 0;
    auto dynamicSizes = op.getDynamicSizes();

    for (int i = 0; i < memRefType.getRank(); ++i) {
      Value dimSize;
      if (memRefType.isDynamicDim(i)) {
        Value dynSize = dynamicSizes[dynamicIdx++];
        dimSize =
            rewriter.create<UnrealizedConversionCastOp>(loc, int64Ty, dynSize)
                .getResult(0);
      } else {
        dimSize = rewriter.create<LLVM::ConstantOp>(
            loc, int64Ty, rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
      }
      sizeBytes = rewriter.create<LLVM::MulOp>(loc, sizeBytes, dimSize);
    }

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));
    // Allocate space on stack for the device pointer (always space 0)
    Value ptrVar =
        rewriter.create<LLVM::AllocaOp>(loc, genericPtrTy, devicePtrTy, one, 8);

    Value stream = rewriter.create<LLVM::ZeroOp>(loc, genericPtrTy);
    if (!op.getAsyncDependencies().empty()) {
       Value token = op.getAsyncDependencies().front();
       auto ptrTy = LLVM::LLVMPointerType::get(ctx);
       stream = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, token).getResult(0);
    } 

    rewriter.create<LLVM::CallOp>(loc, cudaMalloc,
                                  ValueRange{ptrVar, sizeBytes, stream});
    Value devicePtr = rewriter.create<LLVM::LoadOp>(loc, devicePtrTy, ptrVar);

    SmallVector<Type> elemTypes;
    elemTypes.push_back(devicePtrTy);
    elemTypes.push_back(devicePtrTy);
    elemTypes.push_back(int64Ty);
    if (memRefType.getRank() > 0) {
      elemTypes.push_back(
          LLVM::LLVMArrayType::get(int64Ty, memRefType.getRank()));
      elemTypes.push_back(
          LLVM::LLVMArrayType::get(int64Ty, memRefType.getRank()));
    }
    Type descType = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);

    Value desc = rewriter.create<LLVM::UndefOp>(loc, descType);
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, devicePtr,
                                                ArrayRef<int64_t>{0});
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, devicePtr,
                                                ArrayRef<int64_t>{1});
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(0));
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, zero,
                                                ArrayRef<int64_t>{2});

    if (memRefType.getRank() > 0) {
      Value sizesArray = rewriter.create<LLVM::UndefOp>(loc, elemTypes[3]);
      unsigned dIdx = 0;
      for (int i = 0; i < memRefType.getRank(); ++i) {
        Value dimSize;
        if (memRefType.isDynamicDim(i)) {
          Value dynSize = dynamicSizes[dIdx++];
          dimSize =
              rewriter.create<UnrealizedConversionCastOp>(loc, int64Ty, dynSize)
                  .getResult(0);
        } else {
          dimSize = rewriter.create<LLVM::ConstantOp>(
              loc, int64Ty,
              rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
        }
        sizesArray = rewriter.create<LLVM::InsertValueOp>(
            loc, sizesArray, dimSize, ArrayRef<int64_t>{i});
      }
      desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, sizesArray,
                                                  ArrayRef<int64_t>{3});
      Value stridesArray = rewriter.create<LLVM::UndefOp>(loc, elemTypes[4]);

      // Compute row-major strides: stride[i] = product(dim[j] for j > i)
      Value currentStride = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(1));
      for (int i = memRefType.getRank() - 1; i >= 0; --i) {
        stridesArray = rewriter.create<LLVM::InsertValueOp>(
            loc, stridesArray, currentStride, ArrayRef<int64_t>{i});

        Value dimSize;
        if (memRefType.isDynamicDim(i)) {
          unsigned dIdx2 = 0;
          for (int j = 0; j < i; ++j)
            if (memRefType.isDynamicDim(j))
              dIdx2++;
          Value dynSize = dynamicSizes[dIdx2];
          dimSize =
              rewriter.create<UnrealizedConversionCastOp>(loc, int64Ty, dynSize)
                  .getResult(0);
        } else {
          dimSize = rewriter.create<LLVM::ConstantOp>(
              loc, int64Ty,
              rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
        }
        currentStride =
            rewriter.create<LLVM::MulOp>(loc, currentStride, dimSize);
      }
      desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, stridesArray,
                                                  ArrayRef<int64_t>{4});
    }

    SmallVector<Value, 2> results;
    Value memrefVal = rewriter.create<UnrealizedConversionCastOp>(loc, memRefType, desc).getResult(0);
    results.push_back(memrefVal);

    if (op.getAsyncToken()) {
      auto tokenTy = gpu::AsyncTokenType::get(ctx);
      results.push_back(
          rewriter.create<UnrealizedConversionCastOp>(loc, tokenTy, stream).getResult(0));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

class ConvertGpuMemcpyToCall : public OpRewritePattern<gpu::MemcpyOp> {
public:
  using OpRewritePattern<gpu::MemcpyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::MemcpyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();

    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemcpy =
        getOrDeclareFunc(module, rewriter, "cudaMemcpyAsync", int32Ty,
                         {genericPtrTy, genericPtrTy, int64Ty, int32Ty, genericPtrTy});

    Value dst = op.getDst();
    Value src = op.getSrc();

    auto getPtrFromMemRef = [&](Value val) -> Value {
      auto memRefTy = llvm::cast<MemRefType>(val.getType());
      unsigned addressSpace = 0;
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(memRefTy.getMemorySpace())) {
        addressSpace = intAttr.getInt();
      } else if (auto addrSpaceAttr =
                     llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
                         memRefTy.getMemorySpace())) {
        addressSpace = (unsigned)addrSpaceAttr.getValue();
      }
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, addressSpace);
      SmallVector<Type> elemTypes;
      elemTypes.push_back(ptrTy);
      elemTypes.push_back(ptrTy);
      elemTypes.push_back(int64Ty);
      if (memRefTy.getRank() > 0) {
        elemTypes.push_back(
            LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
        elemTypes.push_back(
            LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
      }
      Type descTy = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);
      Value desc = rewriter.create<UnrealizedConversionCastOp>(loc, descTy, val)
                       .getResult(0);
      Value rawPtr = rewriter.create<LLVM::ExtractValueOp>(
          loc, desc, ArrayRef<int64_t>{1});
      // Cast to generic pointer for cudaMemcpy call if necessary
      if (addressSpace != 0) {
        return rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy,
                                                      rawPtr);
      }
      return rawPtr;
    };

    Value dstPtr = getPtrFromMemRef(dst);
    Value srcPtr = getPtrFromMemRef(src);

    MemRefType memRefType = llvm::cast<MemRefType>(dst.getType());
    Type elementTy = memRefType.getElementType();
    int64_t elementSize = 0;
    if (elementTy.isIntOrFloat())
      elementSize = elementTy.getIntOrFloatBitWidth() / 8;
    else if (elementTy.isIndex())
      elementSize = 8;

    if (elementSize == 0)
      elementSize = 1;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));

    // Dynamic dim handling for memcpy (fixing the previously identified bug)
    auto getDescriptor = [&](Value val) -> Value {
      auto memRefTy = llvm::cast<MemRefType>(val.getType());
      unsigned addressSpace = 0;
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(memRefTy.getMemorySpace())) {
        addressSpace = intAttr.getInt();
      } else if (auto addrSpaceAttr =
                     llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
                         memRefTy.getMemorySpace())) {
        addressSpace = (unsigned)addrSpaceAttr.getValue();
      }
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, addressSpace);
      SmallVector<Type> elemTypes;
      elemTypes.push_back(ptrTy);
      elemTypes.push_back(ptrTy);
      elemTypes.push_back(int64Ty);
      if (memRefTy.getRank() > 0) {
        elemTypes.push_back(
            LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
        elemTypes.push_back(
            LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
      }
      Type descTy = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);
      return rewriter.create<UnrealizedConversionCastOp>(loc, descTy, val)
          .getResult(0);
    };

    Value desc = getDescriptor(dst);
    for (int i = 0; i < memRefType.getRank(); ++i) {
      Value dimSize;
      if (memRefType.isDynamicDim(i)) {
        // Extract size from descriptor
        dimSize = rewriter.create<LLVM::ExtractValueOp>(
            loc, desc, ArrayRef<int64_t>{3, i});
      } else {
        dimSize = rewriter.create<LLVM::ConstantOp>(
            loc, int64Ty, rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
      }
      sizeBytes = rewriter.create<LLVM::MulOp>(loc, sizeBytes, dimSize);
    }

    auto isDeviceMemRef = [&](Value val) -> bool {
      auto memRefTy = llvm::cast<MemRefType>(val.getType());
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(memRefTy.getMemorySpace())) {
        if (intAttr.getInt() != 0) return true;
      } else if (auto addrSpaceAttr =
                     llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
                         memRefTy.getMemorySpace())) {
        if ((unsigned)addrSpaceAttr.getValue() != 0) return true;
      }
      
      // Trace back the value to see if it comes from `gpu.alloc`
      Value current = val;
      while (current) {
        if (auto expandOp = current.getDefiningOp<memref::ExpandShapeOp>()) {
          current = expandOp.getSrc();
        } else if (auto collapseOp = current.getDefiningOp<memref::CollapseShapeOp>()) {
          current = collapseOp.getSrc();
        } else if (auto castOp = current.getDefiningOp<memref::CastOp>()) {
          current = castOp.getSource();
        } else if (auto subviewOp = current.getDefiningOp<memref::SubViewOp>()) {
          current = subviewOp.getSource();
        } else if (auto allocTensor = current.getDefiningOp<bufferization::ToTensorOp>()) {
          current = allocTensor.getOperand();
        } else if (auto allocOp = current.getDefiningOp<gpu::AllocOp>()) {
          return true; // Discovered it originally came from gpu.alloc
        } else {
          break;
        }
      }
      return false;
    };

    bool dstIsDev = isDeviceMemRef(dst);
    bool srcIsDev = isDeviceMemRef(src);
    int kindVal = 4; // Default
    if (!dstIsDev && srcIsDev) kindVal = 2; // DeviceToHost
    else if (dstIsDev && !srcIsDev) kindVal = 1; // HostToDevice
    else if (dstIsDev && srcIsDev) kindVal = 3; // DeviceToDevice
    else kindVal = 0; // HostToHost

    Value kind = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(kindVal));
    Value stream = rewriter.create<LLVM::ZeroOp>(loc, genericPtrTy);
    if (!op.getAsyncDependencies().empty()) {
       Value token = op.getAsyncDependencies().front();
       auto ptrTy = LLVM::LLVMPointerType::get(ctx);
       stream = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, token).getResult(0);
    } 

    rewriter.create<LLVM::CallOp>(loc, cudaMemcpy,
                                  ValueRange{dstPtr, srcPtr, sizeBytes, kind, stream});

    if (op.getAsyncToken()) {
      auto tokenTy = gpu::AsyncTokenType::get(ctx);
      Value dummyToken =
          rewriter.create<UnrealizedConversionCastOp>(loc, tokenTy, stream)
              .getResult(0);
      rewriter.replaceOp(op, {dummyToken});
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class ConvertGpuMemsetToCall : public OpRewritePattern<gpu::MemsetOp> {
public:
  using OpRewritePattern<gpu::MemsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::MemsetOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();

    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemset = getOrDeclareFunc(module, rewriter, "cudaMemsetAsync", int32Ty,
                                       {genericPtrTy, int32Ty, int64Ty, genericPtrTy});

    Value memref = op.getDst();

    auto getDescriptor = [&](Value val) -> Value {
      auto memRefTy = llvm::cast<MemRefType>(val.getType());
      unsigned addressSpace = 0;
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(memRefTy.getMemorySpace())) {
        addressSpace = intAttr.getInt();
      } else if (auto addrSpaceAttr =
                     llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
                         memRefTy.getMemorySpace())) {
        addressSpace = (unsigned)addrSpaceAttr.getValue();
      }
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, addressSpace);
      SmallVector<Type> elemTypes;
      elemTypes.push_back(ptrTy);
      elemTypes.push_back(ptrTy);
      elemTypes.push_back(int64Ty);
      if (memRefTy.getRank() > 0) {
        elemTypes.push_back(
            LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
        elemTypes.push_back(
            LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
      }
      Type descTy = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);
      return rewriter.create<UnrealizedConversionCastOp>(loc, descTy, val)
          .getResult(0);
    };

    Value desc = getDescriptor(memref);
    Value rawPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, desc, ArrayRef<int64_t>{1});

    auto memRefType = llvm::cast<MemRefType>(memref.getType());
    unsigned addressSpace = 0;
    if (auto intAttr =
            llvm::dyn_cast_or_null<IntegerAttr>(memRefType.getMemorySpace())) {
      addressSpace = intAttr.getInt();
    }
    Value ptr = rawPtr;
    if (addressSpace != 0) {
      ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, rawPtr);
    }

    Type elementTy = memRefType.getElementType();
    int64_t elementSize = 0;
    if (elementTy.isIntOrFloat())
      elementSize = elementTy.getIntOrFloatBitWidth() / 8;
    else if (elementTy.isIndex())
      elementSize = 8;

    if (elementSize == 0)
      elementSize = 1;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));

    for (int i = 0; i < memRefType.getRank(); ++i) {
      Value dimSize;
      if (memRefType.isDynamicDim(i)) {
        dimSize = rewriter.create<LLVM::ExtractValueOp>(
            loc, desc, ArrayRef<int64_t>{3, i});
      } else {
        dimSize = rewriter.create<LLVM::ConstantOp>(
            loc, int64Ty, rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
      }
      sizeBytes = rewriter.create<LLVM::MulOp>(loc, sizeBytes, dimSize);
    }

    // Convert value to i32 (cudaMemset expects a byte or i32 depending on API,
    // usually byte-wise)
    // For float 0.0, we can just use 0.
    Value fillValue = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(0));

    Value stream = rewriter.create<LLVM::ZeroOp>(loc, genericPtrTy);
    if (!op.getAsyncDependencies().empty()) {
       Value token = op.getAsyncDependencies().front();
       auto ptrTy = LLVM::LLVMPointerType::get(ctx);
       stream = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, token).getResult(0);
    } 

    rewriter.create<LLVM::CallOp>(loc, cudaMemset,
                                  ValueRange{ptr, fillValue, sizeBytes, stream});

    if (op.getAsyncToken()) {
      auto tokenTy = gpu::AsyncTokenType::get(ctx);
      Value dummyToken =
          rewriter.create<UnrealizedConversionCastOp>(loc, tokenTy, stream)
              .getResult(0);
      rewriter.replaceOp(op, {dummyToken});
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class ConvertGpuDeallocToCall : public OpRewritePattern<gpu::DeallocOp> {
public:
  using OpRewritePattern<gpu::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int32Ty = IntegerType::get(ctx, 32);
    auto int64Ty = IntegerType::get(ctx, 64);

    auto cudaFree =
        getOrDeclareFunc(module, rewriter, "cudaFreeAsync", int32Ty, {genericPtrTy, genericPtrTy});
    auto memRef = op.getMemref();

    auto memRefTy = llvm::cast<MemRefType>(memRef.getType());
    unsigned addressSpace = 0;
    if (auto intAttr =
            llvm::dyn_cast_or_null<IntegerAttr>(memRefTy.getMemorySpace())) {
      addressSpace = intAttr.getInt();
    } else if (auto addrSpaceAttr = llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
                   memRefTy.getMemorySpace())) {
      addressSpace = (unsigned)addrSpaceAttr.getValue();
    }
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, addressSpace);
    SmallVector<Type> elemTypes;
    elemTypes.push_back(ptrTy);
    elemTypes.push_back(ptrTy);
    elemTypes.push_back(int64Ty);
    if (memRefTy.getRank() > 0) {
      elemTypes.push_back(
          LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
      elemTypes.push_back(
          LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
    }
    Type descTy = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);

    Value desc =
        rewriter.create<UnrealizedConversionCastOp>(loc, descTy, memRef)
            .getResult(0);
    Value rawPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, desc, ArrayRef<int64_t>{1});

    // Cast to generic pointer for cudaFree call if necessary
    Value ptr = rawPtr;
    if (addressSpace != 0) {
      ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, rawPtr);
    }

    Value stream = rewriter.create<LLVM::ZeroOp>(loc, genericPtrTy);
    if (!op.getAsyncDependencies().empty()) {
       Value token = op.getAsyncDependencies().front();
       auto ptrTy = LLVM::LLVMPointerType::get(ctx);
       stream = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, token).getResult(0);
    } 

    rewriter.create<LLVM::CallOp>(loc, cudaFree, ValueRange{ptr, stream});

    if (op.getAsyncToken()) {
      auto tokenTy = gpu::AsyncTokenType::get(ctx);
      Value dummyToken =
          rewriter.create<UnrealizedConversionCastOp>(loc, tokenTy, stream)
              .getResult(0);
      rewriter.replaceOp(op, {dummyToken});
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class FixHostGpuAccess : public OpRewritePattern<LLVM::LoadOp> {
public:
  using OpRewritePattern<LLVM::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ptr = op.getAddr();
    auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
    if (!ptrType || ptrType.getAddressSpace() != 1)
      return failure();

    // Found a host-side load from GPU memory!
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemcpySync =
        getOrDeclareFunc(module, rewriter, "cudaMemcpy", int32Ty,
                         {genericPtrTy, genericPtrTy, int64Ty, int32Ty});

    // Create a temporary host buffer
    Type elemTy = op.getType();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));
    Value hostPtrVar = rewriter.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(ctx), elemTy, one, 8);

    // Prepare pointers for cudaMemcpy
    Value dstPtr =
        rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, hostPtrVar);
    Value srcPtr =
        rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, ptr);

    int64_t elementSize = 0;
    if (elemTy.isIntOrFloat())
      elementSize = elemTy.getIntOrFloatBitWidth() / 8;
    else if (elemTy.isIndex())
      elementSize = 8;

    if (elementSize == 0)
      elementSize = 4; // Default to 4 for float/int32 if unknown
    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));

    Value kind = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(2)); // DeviceToHost = 2

    rewriter.create<LLVM::CallOp>(loc, cudaMemcpySync,
                                  ValueRange{dstPtr, srcPtr, sizeBytes, kind});

    // Load from host buffer instead
    Value hostVal = rewriter.create<LLVM::LoadOp>(loc, elemTy, hostPtrVar);
    rewriter.replaceOp(op, hostVal);

    return success();
  }
};

class FixHostGpuStore : public OpRewritePattern<LLVM::StoreOp> {
public:
  using OpRewritePattern<LLVM::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto ptr = op.getAddr();
    auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
    if (!ptrType || ptrType.getAddressSpace() != 1)
      return failure();

    // Found a host-side store to GPU memory!
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemcpySync =
        getOrDeclareFunc(module, rewriter, "cudaMemcpy", int32Ty,
                         {genericPtrTy, genericPtrTy, int64Ty, int32Ty});

    Value value = op.getValue();
    Type elemTy = value.getType();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));
    Value hostPtrVar = rewriter.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(ctx), elemTy, one, 8);

    // Store value to host buffer
    rewriter.create<LLVM::StoreOp>(loc, value, hostPtrVar);

    // Prepare pointers for cudaMemcpy
    Value dstPtr =
        rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, ptr);
    Value srcPtr =
        rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, hostPtrVar);

    int64_t elementSize = 0;
    if (elemTy.isIntOrFloat())
      elementSize = elemTy.getIntOrFloatBitWidth() / 8;
    else if (elemTy.isIndex())
      elementSize = 8;

    if (elementSize == 0)
      elementSize = 4;
    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));

    Value kind = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1)); // HostToDevice = 1

    rewriter.create<LLVM::CallOp>(loc, cudaMemcpySync,
                                  ValueRange{dstPtr, srcPtr, sizeBytes, kind});

    rewriter.eraseOp(op);
    return success();
  }
};

// Returns true when `val` was produced by a call to "mgpuMemAlloc" or
// "cudaMallocAsync" — i.e. a plain ptr (addrspace 0) that points into CUDA
// device memory, but whose address space was erased during gpu-to-llvm
// lowering.
static bool isGpuAllocPtr(Value val) {
  auto callOp = val.getDefiningOp<LLVM::CallOp>();
  if (!callOp)
    return false;
  auto callee = callOp.getCallee();
  return callee &&
         (*callee == "mgpuMemAlloc" || *callee == "cudaMallocAsync");
}

// Host-side load from a plain GPU pointer (addrspace 0) returned by
// mgpuMemAlloc/cudaMallocAsync.  Replaces the load with a synchronous D2H
// cudaMemcpy into a host stack buffer, then a load from that buffer.
class FixHostMgpuPtrLoad : public OpRewritePattern<LLVM::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value ptr = op.getAddr();
    auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
    if (!ptrType || ptrType.getAddressSpace() != 0)
      return failure();
    if (!isGpuAllocPtr(ptr))
      return failure();

    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemcpySync =
        getOrDeclareFunc(module, rewriter, "cudaMemcpy", int32Ty,
                         {genericPtrTy, genericPtrTy, int64Ty, int32Ty});

    Type elemTy = op.getType();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));
    Value hostBuf =
        rewriter.create<LLVM::AllocaOp>(loc, genericPtrTy, elemTy, one, 8);

    int64_t elementSize = 0;
    if (elemTy.isIntOrFloat())
      elementSize = elemTy.getIntOrFloatBitWidth() / 8;
    else if (elemTy.isIndex())
      elementSize = 8;
    if (elementSize == 0)
      elementSize = 4;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));
    Value kind = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(2)); // DeviceToHost

    rewriter.create<LLVM::CallOp>(loc, cudaMemcpySync,
                                  ValueRange{hostBuf, ptr, sizeBytes, kind});
    Value hostVal = rewriter.create<LLVM::LoadOp>(loc, elemTy, hostBuf);
    rewriter.replaceOp(op, hostVal);
    return success();
  }
};

// Host-side store to a plain GPU pointer (addrspace 0) returned by
// mgpuMemAlloc/cudaMallocAsync.  Replaces the store with a store to a host
// stack buffer followed by a synchronous H2D cudaMemcpy.
class FixHostMgpuPtrStore : public OpRewritePattern<LLVM::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value ptr = op.getAddr();
    auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
    if (!ptrType || ptrType.getAddressSpace() != 0)
      return failure();
    if (!isGpuAllocPtr(ptr))
      return failure();

    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    auto genericPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemcpySync =
        getOrDeclareFunc(module, rewriter, "cudaMemcpy", int32Ty,
                         {genericPtrTy, genericPtrTy, int64Ty, int32Ty});

    Value value = op.getValue();
    Type elemTy = value.getType();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));
    Value hostBuf =
        rewriter.create<LLVM::AllocaOp>(loc, genericPtrTy, elemTy, one, 8);

    rewriter.create<LLVM::StoreOp>(loc, value, hostBuf);

    int64_t elementSize = 0;
    if (elemTy.isIntOrFloat())
      elementSize = elemTy.getIntOrFloatBitWidth() / 8;
    else if (elemTy.isIndex())
      elementSize = 8;
    if (elementSize == 0)
      elementSize = 4;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));
    Value kind = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1)); // HostToDevice

    rewriter.create<LLVM::CallOp>(loc, cudaMemcpySync,
                                  ValueRange{ptr, hostBuf, sizeBytes, kind});
    rewriter.eraseOp(op);
    return success();
  }
};

class GpuRuntimeLoweringPass
    : public PassWrapper<GpuRuntimeLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuRuntimeLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](LLVM::LoadOp op) {
      auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(op.getAddr().getType());
      if (ptrType) {
        llvm::errs() << "  [load] addrspace=" << ptrType.getAddressSpace()
                     << " addr_defop=" << (op.getAddr().getDefiningOp()
                        ? op.getAddr().getDefiningOp()->getName().getStringRef()
                        : "blockarg")
                     << "\n";
      }
    });

    RewritePatternSet patterns(module.getContext());
    patterns.add<ConvertGpuAllocToCall>(module.getContext());
    patterns.add<ConvertGpuMemcpyToCall>(module.getContext());
    patterns.add<ConvertGpuDeallocToCall>(module.getContext());
    patterns.add<ConvertGpuMemsetToCall>(module.getContext());
    patterns.add<FixHostGpuAccess>(module.getContext());
    patterns.add<FixHostGpuStore>(module.getContext());
    patterns.add<FixHostMgpuPtrLoad>(module.getContext());
    patterns.add<FixHostMgpuPtrStore>(module.getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
    llvm::errs() << "[GpuRuntimeLowering] Done.\n";
  }

  StringRef getArgument() const final { return "gpu-runtime-lowering"; }
  StringRef getDescription() const final {
    return "Lower abstract GPU memory operations to CUDA runtime calls";
  }
};

std::unique_ptr<Pass> createGpuRuntimeLoweringPass() {
  return std::make_unique<GpuRuntimeLoweringPass>();
}

} // namespace nova
} // namespace mlir