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
#include "mlir/IR/IRMapping.h"
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
  void updateMemorySpaceRecursively(Value val, Attribute newSpace, SmallVectorImpl<Operation*>& redundantCopies) {
    auto oldType = llvm::dyn_cast<MemRefType>(val.getType());
    if (!oldType)
      return;

    auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                   oldType.getLayout(), newSpace);
    val.setType(newType);

    for (auto &use : val.getUses()) {
      Operation *user = use.getOwner();
      
      // If the user is a memcpy and both sides are now the same space, mark it redundant
      if (auto memcpyOp = llvm::dyn_cast<gpu::MemcpyOp>(user)) {
        // We'll check this later in the main loop to avoid iterator invalidation
        redundantCopies.push_back(user);
        continue;
      }

      if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
              memref::CastOp, memref::ReshapeOp, memref::TransposeOp,
              memref::ReinterpretCastOp, bufferization::ToBufferOp>(user)) {
        for (Value result : user->getResults()) {
          updateMemorySpaceRecursively(result, newSpace, redundantCopies);
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

    ModuleOp module = func->getParentOfType<ModuleOp>();
    // MLIRContext *ctx = func.getContext(); // Removed unused variable
    OpBuilder moduleBuilder(module.getBodyRegion());

    // Phase 1: Promote host memref arguments to address space 1
    SmallVector<Operation*, 8> redundantCopies;
    SmallVector<Type, 8> newArgTypes;
    bool signatureChanged = false;
    for (auto arg : func.getArguments()) {
      auto memRefType = llvm::dyn_cast<MemRefType>(arg.getType());
      if (memRefType && memRefType.getMemorySpaceAsInt() == 0) {
        auto newType = MemRefType::get(
            memRefType.getShape(), memRefType.getElementType(),
            memRefType.getLayout(), moduleBuilder.getI64IntegerAttr(1));
        arg.setType(newType);
        updateMemorySpaceRecursively(arg, moduleBuilder.getI64IntegerAttr(1), redundantCopies);
        newArgTypes.push_back(newType);
        signatureChanged = true;
      } else {
        newArgTypes.push_back(arg.getType());
      }
    }

    if (signatureChanged) {
      auto newFuncType = FunctionType::get(
          func.getContext(), newArgTypes, func.getFunctionType().getResults());
      func.setType(newFuncType);
    }

    // Erase redundant copies (where src and dst are now in the same space)
    for (Operation* op : redundantCopies) {
      if (auto memcpyOp = llvm::dyn_cast<gpu::MemcpyOp>(op)) {
         Value src = memcpyOp.getSrc();
         Value dst = memcpyOp.getDst();
         if (src.getType() == dst.getType()) {
            dst.replaceAllUsesWith(src);
            op->erase();
         }
      }
    }

    // Phase 2: persistent constants (Already implemented)
    func.walk([&](gpu::LaunchOp launchOp) {
      launchOp.getRegion().walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value val = operand.get();
          auto memRefType = llvm::dyn_cast<MemRefType>(val.getType());
          if (!memRefType || memRefType.getMemorySpaceAsInt() != 0)
            continue;

          // Skip values defined inside the launch region
          if (launchOp.getRegion().isAncestor(val.getParentRegion()))
            continue;

          // Only process host constants (memref.get_global and derived views)
          if (!isHostConstant(val))
            continue;

          // Find the original memref.get_global and collect views
          memref::GetGlobalOp getGlobal;
          SmallVector<Operation*, 4> views;
          Operation *curr = val.getDefiningOp();
          while (curr && !llvm::isa<memref::GetGlobalOp>(curr)) {
            views.push_back(curr);
            if (curr->getNumOperands() > 0 && llvm::isa<MemRefType>(curr->getOperand(0).getType()))
              curr = curr->getOperand(0).getDefiningOp();
            else
              curr = nullptr;
          }
          getGlobal = dyn_cast_or_null<memref::GetGlobalOp>(curr);
          if (!getGlobal) continue;

          StringRef hostSymbol = getGlobal.getName();
          std::string deviceSymbol = (hostSymbol + "_device").str();

          // Ensure the device global exists
          if (!module.lookupSymbol(deviceSymbol)) {
            auto hostGlobal = module.lookupSymbol<memref::GlobalOp>(hostSymbol);
            if (!hostGlobal) continue;

            auto hostGlobalType = llvm::cast<MemRefType>(hostGlobal.getType());
            auto deviceGlobalType = MemRefType::get(
                hostGlobalType.getShape(), hostGlobalType.getElementType(),
                hostGlobalType.getLayout(), moduleBuilder.getI64IntegerAttr(1));

            moduleBuilder.setInsertionPointToStart(module.getBody());
            Attribute initVal = hostGlobal.getInitialValue() ? *hostGlobal.getInitialValue() : Attribute();
            moduleBuilder.create<memref::GlobalOp>(
                getGlobal.getLoc(), deviceSymbol,
                moduleBuilder.getStringAttr("public"), deviceGlobalType,
                initVal, /*constant=*/true, /*alignment=*/nullptr);
          }

          // Create get_global and re-apply views at the start of the function
          OpBuilder funcBuilder(&func.getBody().front().front());
          auto deviceGlobalType = llvm::cast<MemRefType>(module.lookupSymbol<memref::GlobalOp>(deviceSymbol).getType());
          Value deviceVal = funcBuilder.create<memref::GetGlobalOp>(
              getGlobal.getLoc(), deviceGlobalType, deviceSymbol);
          
          // Reverse order of views as we collected them from leaf to root
          for (int i = views.size() - 1; i >= 0; --i) {
             Operation* oldView = views[i];
             IRMapping mapping;
             mapping.map(oldView->getOperand(0), deviceVal);
             Operation* newView = funcBuilder.clone(*oldView, mapping);
             
             // Update return type of cloned view to address space 1
             auto oldType = llvm::cast<MemRefType>(oldView->getResult(0).getType());
             auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                            oldType.getLayout(), funcBuilder.getI64IntegerAttr(1));
             newView->getResult(0).setType(newType);
             deviceVal = newView->getResult(0);
          }
          
          operand.set(deviceVal);
        }
      });
    });
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