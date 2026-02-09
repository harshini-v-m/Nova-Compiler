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

  // Helper to check if a memory space is device (1)
  bool isDeviceMemorySpace(Attribute memorySpace) {
    if (!memorySpace)
      return false;
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
      return intAttr.getInt() != 0;
    }
    if (llvm::isa<nova::NovaDeviceAttr>(memorySpace)) {
      return true;
    }
    return false;
  }

  // Helper to decide if we should register/pin host memory for faster transfers
  bool shouldRegisterHostMemory(Value val) {
    auto memRefType = llvm::dyn_cast<MemRefType>(val.getType());
    if (!memRefType)
      return false;

    // NEVER register memory that is already on a device
    if (isDeviceMemorySpace(memRefType.getMemorySpace()))
      return false;

    // Static size check
    if (memRefType.hasStaticShape()) {
      int64_t numElements = memRefType.getNumElements();
      int64_t eltSizeInBits = memRefType.getElementTypeBitWidth();
      int64_t totalSizeInBytes = (numElements * eltSizeInBits) / 8;

      // Heuristic:
      // Minimum: 4KB (4096 bytes) to justify overhead
      // Maximum: 12GB (12 * 1024^3 bytes) safety limit
      const int64_t minSize = 4096;
      const int64_t maxSize = 12LL * 1024 * 1024 * 1024;

      if (totalSizeInBytes > minSize && totalSizeInBytes < maxSize) {
        return true;
      }
      return false;
    }

    // Dynamic shapes: Be conservative
    return false;
  }

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
      // List of index-aliasing operations that preserve memory space properties
      if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
              memref::CastOp, bufferization::ToBufferOp>(user)) {
        for (Value result : user->getResults()) {
          updateMemorySpaceRecursively(result, newSpace);
        }
      }
    }
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()))
      return;

    ModuleOp module = func->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(module);

    // Collect all ReturnOps and returned values up front to avoid redundant
    // walks
    SmallVector<func::ReturnOp, 2> returnOps;
    SmallPtrSet<Value, 4> returnedValues;
    func.walk([&](func::ReturnOp returnOp) {
      returnOps.push_back(returnOp);
      for (Value val : returnOp.getOperands())
        returnedValues.insert(val);
    });

    // 1. Identify all "device sinks" and propagate backwards to identify which
    // allocations must be on device
    SmallPtrSet<Value, 8> mustBeOnDevice;

    // Initial sinks: function results that expect device memrefs
    for (auto returnOp : returnOps) {
      for (unsigned i = 0; i < returnOp->getNumOperands(); ++i) {
        Value val = returnOp.getOperand(i);
        auto expectedType = func.getResultTypes()[i];
        if (auto memrefType = llvm::dyn_cast<MemRefType>(expectedType)) {
          if (isDeviceMemorySpace(memrefType.getMemorySpace())) {
            mustBeOnDevice.insert(val);
          }
        }
      }
    }

    // Initial sinks: Function arguments that already have device memory space
    // (important for DPS)
    for (auto arg : func.getArguments()) {
      if (auto memrefType = llvm::dyn_cast<MemRefType>(arg.getType())) {
        if (isDeviceMemorySpace(memrefType.getMemorySpace())) {
          mustBeOnDevice.insert(arg);
        }
      }
    }

    // Worklist for propagation
    SmallVector<Value, 16> worklist;
    for (Value v : mustBeOnDevice) {
      worklist.push_back(v);
    }

    // Initial sinks: GPU launches and HostRegister (these are consumers)
    func.walk([&](Operation *op) {
      // llvm::errs() << "[AGMC] Walker visiting: " << *op << "\n";
      if (isa<gpu::LaunchOp>(op) || op->getParentOfType<gpu::LaunchOp>() ||
          isa<gpu::HostRegisterOp>(op)) {
        for (Value operand : op->getOperands()) {
          if (llvm::isa<MemRefType>(operand.getType())) {
            if (mustBeOnDevice.insert(operand).second) {
              worklist.push_back(operand);
            }
          }
        }
      }
      // Also direct copies to device memory (both memref and gpu dialects)
      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        auto destType =
            llvm::dyn_cast<MemRefType>(copyOp.getTarget().getType());
        if (destType && isDeviceMemorySpace(destType.getMemorySpace())) {
          if (mustBeOnDevice.insert(copyOp.getSource()).second) {
            worklist.push_back(copyOp.getSource());
          }
        }
      }
      if (auto linalgCopyOp = dyn_cast<linalg::CopyOp>(op)) {
        auto destType =
            llvm::dyn_cast<MemRefType>(linalgCopyOp.getOutputs()[0].getType());
        if (destType && isDeviceMemorySpace(destType.getMemorySpace())) {
          if (mustBeOnDevice.insert(linalgCopyOp.getInputs()[0]).second) {
            worklist.push_back(linalgCopyOp.getInputs()[0]);
          }
        }
      }
      if (auto gpuCopyOp = dyn_cast<gpu::MemcpyOp>(op)) {
        auto destType =
            llvm::dyn_cast<MemRefType>(gpuCopyOp.getDst().getType());
        if (destType && isDeviceMemorySpace(destType.getMemorySpace())) {
          if (mustBeOnDevice.insert(gpuCopyOp.getSrc()).second) {
            worklist.push_back(gpuCopyOp.getSrc());
          }
        }
      }
      // Also captures inside gpu.launch regions
      if (auto launchOp = dyn_cast<gpu::LaunchOp>(op)) {
        launchOp.getRegion().walk([&](Operation *innerOp) {
          for (Value operand : innerOp->getOperands()) {
            if (llvm::isa<MemRefType>(operand.getType())) {
              if (mustBeOnDevice.insert(operand).second)
                worklist.push_back(operand);
            }
          }
        });
      }
    });

    while (!worklist.empty()) {
      Value curr = worklist.pop_back_val();
      Operation *defOp = curr.getDefiningOp();
      if (!defOp)
        continue;

      // Propagate backwards through copies
      if (auto copyOp = dyn_cast<memref::CopyOp>(defOp)) {
        if (mustBeOnDevice.insert(copyOp.getSource()).second)
          worklist.push_back(copyOp.getSource());
      } else if (auto gpuCopyOp = dyn_cast<gpu::MemcpyOp>(defOp)) {
        if (mustBeOnDevice.insert(gpuCopyOp.getSrc()).second)
          worklist.push_back(gpuCopyOp.getSrc());
      }
      // Propagate through subviews, casts, etc.
      else if (isa<memref::SubViewOp, memref::CastOp, memref::CollapseShapeOp,
                   memref::ExpandShapeOp, bufferization::ToBufferOp>(defOp)) {
        for (Value operand : defOp->getOperands()) {
          if (llvm::isa<MemRefType>(operand.getType())) {
            if (mustBeOnDevice.insert(operand).second)
              worklist.push_back(operand);
          }
        }
      }
      // Propagate through linalg ops
      else if (auto linalgOp = dyn_cast<mlir::linalg::LinalgOp>(defOp)) {
        for (Value operand : linalgOp->getOperands()) {
          if (llvm::isa<MemRefType>(operand.getType())) {
            if (mustBeOnDevice.insert(operand).second)
              worklist.push_back(operand);
          }
        }
      }
      // Propagate through SCF loops (Parallel and For)
      else if (isa<scf::ParallelOp, scf::ForOp>(defOp)) {
        defOp->walk([&](Operation *innerOp) {
          for (Value operand : innerOp->getOperands()) {
            if (llvm::isa<MemRefType>(operand.getType())) {
              if (mustBeOnDevice.insert(operand).second)
                worklist.push_back(operand);
            }
          }
        });
      }
    }

    // Now promote all allocations that are marked as "mustBeOnDevice"
    SmallVector<Value, 4> promotedAllocsToDealloc;

    // Lambda to promote an allocation
    auto promoteAlloc = [&](Value res, Operation *op) {
      if (!mustBeOnDevice.count(res))
        return;

      Attribute currentSpace;
      if (auto allocOp = dyn_cast<memref::AllocOp>(op))
        currentSpace = allocOp.getType().getMemorySpace();
      else if (auto gpuAllocOp = dyn_cast<gpu::AllocOp>(op))
        currentSpace = gpuAllocOp.getType().getMemorySpace();
      else
        return;

      if (!isDeviceMemorySpace(currentSpace)) {
        Attribute newSpace =
            IntegerAttr::get(IntegerType::get(func.getContext(), 64), 1);
        updateMemorySpaceRecursively(res, newSpace);

        // Track for deallocation (if it doesn't have one)
        bool hasDealloc = false;
        for (auto &use : res.getUses()) {
          if (isa<memref::DeallocOp, gpu::DeallocOp>(use.getOwner())) {
            hasDealloc = true;
            break;
          }
        }
        if (!hasDealloc)
          promotedAllocsToDealloc.push_back(res);
      }
    };

    func.walk([&](Operation *op) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(op))
        promoteAlloc(allocOp.getResult(), op);
      else if (auto gpuAllocOp = dyn_cast<gpu::AllocOp>(op))
        promoteAlloc(gpuAllocOp.getResult(0), op);
    });
    llvm::MapVector<Value, Value> hostToDeviceMap;
    llvm::MapVector<Value, Value> registeredHostMem;
    func.walk([&](gpu::LaunchOp launchOp) {
      launchOp.getRegion().walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value val = operand.get();
          auto memRefType = llvm::dyn_cast<MemRefType>(val.getType());
          if (!memRefType || isDeviceMemorySpace(memRefType.getMemorySpace()))
            continue;

          if (launchOp.getRegion().isAncestor(val.getParentRegion()))
            continue;

          if (hostToDeviceMap.count(val)) {
            operand.set(hostToDeviceMap[val]);
            continue;
          }

          OpBuilder builder(func.getBody().front().getTerminator());
          if (auto defOp = val.getDefiningOp())
            builder.setInsertionPointAfter(defOp);
          else
            builder.setInsertionPointToStart(&func.getBody().front());

          Location loc = val.getLoc();

          if (shouldRegisterHostMemory(val)) {
            auto unrankedType =
                UnrankedMemRefType::get(memRefType.getElementType(), 0);
            auto castOp =
                builder.create<memref::CastOp>(loc, unrankedType, val);
            builder.create<gpu::HostRegisterOp>(loc, castOp);
            registeredHostMem[val] = castOp;
          }

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

          builder.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{},
                                        deviceMem, val);
          operand.set(deviceMem);
        }
      });
    });

    // 3. Update ReturnOps based on function signature
    for (auto returnOp : returnOps) {
      OpBuilder builder(returnOp);
      Location loc = returnOp.getLoc();

      for (unsigned i = 0; i < returnOp->getNumOperands(); ++i) {
        OpOperand &operand = returnOp->getOpOperand(i);
        Value val = operand.get();
        auto expectedType = func.getResultTypes()[i];
        auto expectedMemRef = llvm::dyn_cast<MemRefType>(expectedType);
        if (!expectedMemRef)
          continue;

        auto currentMemRef = llvm::cast<MemRefType>(val.getType());
        bool expectedDevice =
            isDeviceMemorySpace(expectedMemRef.getMemorySpace());
        bool currentDevice =
            isDeviceMemorySpace(currentMemRef.getMemorySpace());

        if (expectedDevice && !currentDevice) {
          Value deviceMem =
              hostToDeviceMap.count(val) ? hostToDeviceMap[val] : nullptr;
          if (!deviceMem) {
            MemRefType deviceType = MemRefType::get(
                currentMemRef.getShape(), currentMemRef.getElementType(),
                currentMemRef.getLayout(), builder.getI64IntegerAttr(1));

            SmallVector<Value> dynamicSizes;
            for (int j = 0; j < currentMemRef.getRank(); ++j) {
              if (currentMemRef.isDynamicDim(j)) {
                Value idx = builder.create<arith::ConstantIndexOp>(loc, j);
                Value dim = builder.create<memref::DimOp>(loc, val, idx);
                dynamicSizes.push_back(dim);
              }
            }
            auto allocOp = builder.create<gpu::AllocOp>(
                loc, deviceType, ValueRange{}, dynamicSizes, ValueRange{});
            deviceMem = allocOp.getResult(0);
            hostToDeviceMap[val] = deviceMem;
            builder.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{},
                                          deviceMem, val);
          }
          operand.set(deviceMem);
        } else if (!expectedDevice && currentDevice) {
          MemRefType hostType = MemRefType::get(
              currentMemRef.getShape(), currentMemRef.getElementType(),
              currentMemRef.getLayout(), Attribute());
          auto hostAlloc = builder.create<memref::AllocOp>(loc, hostType);
          builder.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{},
                                        hostAlloc, val);
          operand.set(hostAlloc);
        }
      }
    }

    // 4. Deallocate shadows and promoted allocs (unless returned)
    auto deallocBuffer = [&](Value buffer) {
      if (!buffer || returnedValues.count(buffer))
        return;

      Block *allocBlock = buffer.getParentBlock();
      Operation *lastUser = nullptr;
      bool safeToDeallocEarly = true;

      for (auto &use : buffer.getUses()) {
        Operation *ancestor = use.getOwner();
        while (ancestor && ancestor->getBlock() != allocBlock)
          ancestor = ancestor->getParentOp();

        if (!ancestor) {
          safeToDeallocEarly = false;
          break;
        }
        if (!lastUser || lastUser->isBeforeInBlock(ancestor))
          lastUser = ancestor;
      }

      if (safeToDeallocEarly && lastUser) {
        OpBuilder builder(lastUser->getBlock(),
                          std::next(Block::iterator(lastUser)));
        builder.create<gpu::DeallocOp>(lastUser->getLoc(), ValueRange{},
                                       buffer);
      } else {
        for (auto returnOp : returnOps) {
          OpBuilder builder(returnOp);
          builder.create<gpu::DeallocOp>(returnOp.getLoc(), ValueRange{},
                                         buffer);
        }
      }
    };

    for (auto pair : hostToDeviceMap)
      deallocBuffer(pair.second);
    for (auto val : promotedAllocsToDealloc)
      deallocBuffer(val);

    // 5. Unregister host memory
    if (!registeredHostMem.empty()) {
      for (auto returnOp : returnOps) {
        OpBuilder builder(returnOp);
        for (auto pair : registeredHostMem)
          builder.create<gpu::HostUnregisterOp>(returnOp.getLoc(), pair.second);
      }
    }
  }

  StringRef getArgument() const final { return "add-gpu-memory-copies"; }
  StringRef getDescription() const final {
    return "Insert explicit gpu.alloc and memcpy for kernel arguments";
  }
};

std::unique_ptr<Pass> createAddGpuMemoryCopiesPass() {
  return std::make_unique<AddGpuMemoryCopiesPass>();
}

} // namespace nova
} // namespace mlir