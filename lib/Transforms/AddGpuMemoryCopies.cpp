#include "Compiler/Transforms/AddGpuMemoryCopies.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace mlir {
namespace nova {

struct AddGpuMemoryCopiesPass
    : public PassWrapper<AddGpuMemoryCopiesPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddGpuMemoryCopiesPass)

  // Helper to recursively update memory space of index-aliasing operations
  void updateMemorySpaceRecursively(Value val, Attribute newSpace) {
    auto oldType = llvm::dyn_cast<MemRefType>(val.getType());
    if (!oldType)
      return;

    auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                   oldType.getLayout(), newSpace);
    val.setType(newType);

    for (auto &use : val.getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
              memref::CastOp, memref::ReshapeOp, memref::TransposeOp,
              memref::ReinterpretCastOp, bufferization::ToBufferOp>(user)) {
        for (Value result : user->getResults()) {
          updateMemorySpaceRecursively(result, newSpace);
        }
      }
    }
  }

  // Check if a value originates from a host constant (memref.get_global or similar)
  bool isHostConstant(Value val) {
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return false;

    // memref.get_global is always host memory
    if (isa<memref::GetGlobalOp>(defOp))
      return true;

    // Trace through view-like ops
    if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
            memref::CastOp, memref::ReshapeOp>(defOp)) {
      for (Value operand : defOp->getOperands()) {
        if (llvm::isa<MemRefType>(operand.getType()) && isHostConstant(operand))
          return true;
      }
    }

    return false;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()))
      return;

    MLIRContext *ctx = func.getContext();

    // Track host constants that need to be copied to device
    llvm::MapVector<Value, Value> hostToDeviceMap;
    SmallVector<Value, 4> deviceAllocsToDealloc;

    // Walk all gpu.launch operations and find host constants used inside
    func.walk([&](gpu::LaunchOp launchOp) {
      launchOp.getRegion().walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value val = operand.get();
          auto memRefType = llvm::dyn_cast<MemRefType>(val.getType());
          if (!memRefType)
            continue;

          // Skip values defined inside the launch region
          if (launchOp.getRegion().isAncestor(val.getParentRegion()))
            continue;

          // Skip if already mapped
          if (hostToDeviceMap.count(val)) {
            operand.set(hostToDeviceMap[val]);
            continue;
          }

          // Only process host constants (memref.get_global and derived views)
          if (!isHostConstant(val))
            continue;

          OpBuilder builder(func.getBody().front().getTerminator());
          if (auto defOp = val.getDefiningOp())
            builder.setInsertionPointAfter(defOp);
          else
            builder.setInsertionPointToStart(&func.getBody().front());

          Location loc = val.getLoc();

          // Create device buffer with address space 1
          MemRefType deviceType = MemRefType::get(
              memRefType.getShape(), memRefType.getElementType(),
              memRefType.getLayout(), builder.getI64IntegerAttr(1));

          SmallVector<Value> dynamicSizes;
          for (int i = 0; i < memRefType.getRank(); ++i) {
            if (memRefType.isDynamicDim(i)) {
              Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
              Value dim = builder.create<memref::DimOp>(loc, val, idx);
              dynamicSizes.push_back(dim);
            }
          }

          auto allocOp = builder.create<gpu::AllocOp>(
              loc, deviceType, ValueRange{}, dynamicSizes, ValueRange{});
          Value deviceMem = allocOp.getResult(0);
          hostToDeviceMap[val] = deviceMem;
          deviceAllocsToDealloc.push_back(deviceMem);

          // Copy host constant to device
          builder.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{},
                                        deviceMem, val);

          // Replace the operand with device buffer
          operand.set(deviceMem);
        }
      });
    });

    // Deallocate device buffers for constants at function return
    SmallVector<func::ReturnOp, 2> returnOps;
    func.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

    for (Value deviceMem : deviceAllocsToDealloc) {
      for (auto returnOp : returnOps) {
        OpBuilder builder(returnOp);
        builder.create<gpu::DeallocOp>(returnOp.getLoc(), ValueRange{},
                                       deviceMem);
      }
    }
  }

  StringRef getArgument() const final { return "add-gpu-memory-copies"; }
  StringRef getDescription() const final {
    return "Copy host constants to GPU memory for kernel access";
  }
};

std::unique_ptr<Pass> createAddGpuMemoryCopiesPass() {
  return std::make_unique<AddGpuMemoryCopiesPass>();
}

} // namespace nova
} // namespace mlir