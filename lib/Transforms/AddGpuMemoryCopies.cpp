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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace mlir {
namespace nova {

struct AddGpuMemoryCopiesPass
    : public PassWrapper<AddGpuMemoryCopiesPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddGpuMemoryCopiesPass)

  // Helper to recursively update memory space of index-aliasing operations.
  // Also follows memref::CopyOp chains to promote clone buffers created by
  // the buffer deallocation pipeline, marking redundant same-space copies.
  void updateMemorySpaceRecursively(Value val, Attribute newSpace,
                                    SmallVectorImpl<Operation*>& redundantCopies,
                                    DenseSet<Value>& visited) {
    if (!visited.insert(val).second)
      return;

    auto oldType = llvm::dyn_cast<MemRefType>(val.getType());
    if (!oldType)
      return;

    auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                   oldType.getLayout(), newSpace);
    val.setType(newType);

    for (auto &use : val.getUses()) {
      Operation *user = use.getOwner();

      // If the user is a gpu.memcpy, mark it redundant
      if (llvm::isa<gpu::MemcpyOp>(user)) {
        redundantCopies.push_back(user);
        continue;
      }

      // If the user is a memref.copy, propagate space change through the copy
      // chain — but ONLY for clone buffers (local allocs). Copies to/from
      // function arguments or globals are meaningful data transfers, not clones.
      if (auto copyOp = llvm::dyn_cast<memref::CopyOp>(user)) {
        Value src = copyOp.getSource();
        Value dst = copyOp.getTarget();
        Value otherSide = (val == src) ? dst : src;
        // Only propagate and mark redundant if the other side is a local alloc
        // (clone pattern). Don't touch function args or globals.
        if (otherSide.getDefiningOp<memref::AllocOp>()) {
          updateMemorySpaceRecursively(otherSide, newSpace, redundantCopies, visited);
          if (val == src)
            redundantCopies.push_back(user);
        }
        continue;
      }

      if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
              memref::CastOp, memref::ReshapeOp, memref::TransposeOp,
              memref::ReinterpretCastOp, bufferization::ToBufferOp>(user)) {
        for (Value result : user->getResults()) {
          updateMemorySpaceRecursively(result, newSpace, redundantCopies, visited);
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

  // Find the last operation in `block` that uses `val` or any of its aliases
  // (through view-like ops). Returns nullptr if no users found in the block.
  Operation *findLastUseInBlock(Value val, Block *block) {
    Operation *lastUser = nullptr;
    DenseSet<Value> visited;

    std::function<void(Value)> chase = [&](Value v) {
      if (!visited.insert(v).second)
        return;
      for (Operation *user : v.getUsers()) {
        // Chase through alias-producing ops
        if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp,
                memref::SubViewOp, memref::CastOp, memref::ReshapeOp,
                memref::TransposeOp, memref::ReinterpretCastOp>(user)) {
          for (Value result : user->getResults()) {
            if (llvm::isa<MemRefType>(result.getType()))
              chase(result);
          }
        }

        // Find ancestor of this user that lives in `block`
        Operation *ancestor = user;
        while (ancestor && ancestor->getBlock() != block)
          ancestor = ancestor->getParentOp();

        if (!ancestor)
          continue;

        if (!lastUser || lastUser->isBeforeInBlock(ancestor))
          lastUser = ancestor;
      }
    };

    chase(val);
    return lastUser;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()))
      return;

    ModuleOp module = func->getParentOfType<ModuleOp>();
    OpBuilder moduleBuilder(module.getBodyRegion());
    
    // Map to track Host -> Device value mappings
    // Key: Original Host Value (Argument or Global)
    // Value: Device Allocation (Space 1)
    DenseMap<Value, Value> hostToDeviceMap;
    // Map to track Host Symbol -> Device Allocation (for globals)
    DenseMap<StringRef, Value> symbolToDeviceAlloc;

    // Phase 1: Handle Function Arguments
    // Don't change signature. Just register and copy if used in GPU.
    // For now, we eagerly copy all Space 0 memref args to ensure availability.
    // This is safer for dynamic/mixed usage.
    
    OpBuilder entryBuilder(&func.getBody().front().front());
    
    // List to keep track of arguments we need to copy back
    SmallVector<std::pair<Value, Value>, 4> argsToCopyBack;

    for (auto arg : func.getArguments()) {
        auto memRefType = llvm::dyn_cast<MemRefType>(arg.getType());
        if (memRefType && memRefType.getMemorySpaceAsInt() == 0) {
            // 1. Register host memory (Pinned Memory)
            Value unrankedArg = entryBuilder.create<memref::CastOp>(
                func.getLoc(),
                UnrankedMemRefType::get(memRefType.getElementType(), 0),
                arg);
            entryBuilder.create<gpu::HostRegisterOp>(func.getLoc(), unrankedArg);
            
            // 2. Alloc Device Memory
            auto deviceType = MemRefType::get(
                memRefType.getShape(), memRefType.getElementType(),
                memRefType.getLayout(), moduleBuilder.getI64IntegerAttr(1));
            
            Value deviceAlloc = entryBuilder.create<memref::AllocOp>(func.getLoc(), deviceType);
            
            // 3. Copy Host -> Device
            entryBuilder.create<memref::CopyOp>(func.getLoc(), arg, deviceAlloc);
            
            // 4. Map it
            hostToDeviceMap[arg] = deviceAlloc;
            argsToCopyBack.push_back({arg, deviceAlloc});
            
            // 5. Dealloc at return (optional but good for hygiene, though usually arena)
            // Skipping dealloc for now as it complicates JIT return value handling if returned.
        }
    }

    // Insert CopyBack (Device -> Host) and Dealloc before all returns
    func.walk([&](func::ReturnOp returnOp) {
        OpBuilder returnBuilder(returnOp);
        for (auto &pair : argsToCopyBack) {
            // Copy Device -> Host
            // pair.first is Host (dst), pair.second is Device (src)
            returnBuilder.create<memref::CopyOp>(returnOp.getLoc(), pair.second, pair.first);
        }
        // Dealloc all argument device allocs after copy-back, before return
        for (auto &pair : argsToCopyBack) {
            returnBuilder.create<memref::DeallocOp>(returnOp.getLoc(), pair.second);
        }
    });

    // Ops to erase AFTER the walk completes (erasing during walk causes
    // iterator invalidation since these ops are in the same block being walked).
    SmallPtrSet<Operation*, 16> deferredErase;

    // Phase 2: Process GPU Launches to swap operands
    func.walk([&](gpu::LaunchOp launchOp) {
      launchOp.getRegion().walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value val = operand.get();
          auto memRefType = llvm::dyn_cast<MemRefType>(val.getType());
          if (!memRefType) continue;

          Attribute space = memRefType.getMemorySpace();
          // Skip non-integer address spaces (e.g., #gpu.address_space<workgroup>)
          if (space && !llvm::isa<IntegerAttr>(space))
            continue;

          // Now safely check integer value for host space (0)
          if (memRefType.getMemorySpaceAsInt() != 0)
            continue;

          // Skip values defined inside (allocs in kernel)
          if (launchOp.getRegion().isAncestor(val.getParentRegion()))
            continue;

          // Trace origin to find Host Root (Arg or Global)
          SmallVector<Operation*, 4> viewChain;
          Operation *curr = val.getDefiningOp();
          Value hostRoot = val;

          while (true) {
              if (!curr) {
                  // It's a BlockArgument
                  if (auto blockArg = llvm::dyn_cast<BlockArgument>(hostRoot)) {
                      if (blockArg.getOwner() == &func.getBody().front()) {
                          // Function Argument
                          break;
                      }
                  }
                  // Unknown block arg (e.g. loop iterator?), abort trace
                  hostRoot = nullptr;
                  break;
              }

              if (llvm::isa<memref::GetGlobalOp>(curr)) {
                  // Found global
                  break;
              }

              if (llvm::isa<memref::AllocOp>(curr)) {
                  // Found local allocation — will promote to device space
                  break;
              }

              // View-like ops
              if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
                      memref::CastOp, memref::ReshapeOp, memref::TransposeOp,
                      memref::ReinterpretCastOp>(curr)) {
                  viewChain.push_back(curr);
                  hostRoot = curr->getOperand(0);
                  curr = hostRoot.getDefiningOp();
              } else {
                  // Unknown op in chain (maybe load result?), abort
                  hostRoot = nullptr;
                  break;
              }
          }

          if (!hostRoot) continue;

          Value deviceRoot = nullptr;

          // Case A: It's an Argument
          if (hostToDeviceMap.count(hostRoot)) {
              deviceRoot = hostToDeviceMap[hostRoot];
          }
          // Case B: It's a Global
          else if (auto getGlobal = val.getDefiningOp<memref::GetGlobalOp>()) {
              // Direct global usage
              StringRef name = getGlobal.getName();
              if (symbolToDeviceAlloc.count(name)) {
                  deviceRoot = symbolToDeviceAlloc[name];
              } else {
                  // Create global alloc/copy logic (similar to original code)
                  auto hostGlobal = module.lookupSymbol<memref::GlobalOp>(name);
                  if (hostGlobal) {
                      auto hostType = llvm::cast<MemRefType>(hostGlobal.getType());
                      auto devType = MemRefType::get(
                          hostType.getShape(), hostType.getElementType(),
                          hostType.getLayout(), moduleBuilder.getI64IntegerAttr(1));

                      OpBuilder funcTop(&func.getBody().front().front());
                      deviceRoot = funcTop.create<memref::AllocOp>(getGlobal.getLoc(), devType);

                      // Using a fresh get_global at top to avoid dominance issues or moving original
                      Value hVal = funcTop.create<memref::GetGlobalOp>(getGlobal.getLoc(), hostType, name);
                      funcTop.create<memref::CopyOp>(getGlobal.getLoc(), hVal, deviceRoot);

                      symbolToDeviceAlloc[name] = deviceRoot;
                  }
              }
          }
          // Case C: View chain from Global
          else if (auto defOp = hostRoot.getDefiningOp<memref::GetGlobalOp>()) {
               StringRef name = defOp.getName();
               // ... (Handling similar to Direct Global, sharing logic)
               if (symbolToDeviceAlloc.count(name)) {
                  deviceRoot = symbolToDeviceAlloc[name];
               } else {
                  // Copy-paste creation logic for robustness
                  auto hostGlobal = module.lookupSymbol<memref::GlobalOp>(name);
                  if (hostGlobal) {
                      auto hostType = llvm::cast<MemRefType>(hostGlobal.getType());
                      auto devType = MemRefType::get(
                          hostType.getShape(), hostType.getElementType(),
                          hostType.getLayout(), moduleBuilder.getI64IntegerAttr(1));

                      OpBuilder funcTop(&func.getBody().front().front());
                      deviceRoot = funcTop.create<memref::AllocOp>(defOp.getLoc(), devType);
                      Value hVal = funcTop.create<memref::GetGlobalOp>(defOp.getLoc(), hostType, name);
                      funcTop.create<memref::CopyOp>(defOp.getLoc(), hVal, deviceRoot);
                      symbolToDeviceAlloc[name] = deviceRoot;
                  }
               }
          }

          // Case D: Local allocation — promote to device space and eliminate clones
          if (!deviceRoot && hostRoot && hostRoot.getDefiningOp<memref::AllocOp>()) {
              SmallVector<Operation*, 4> redundantCopies;
              DenseSet<Value> visited;
              updateMemorySpaceRecursively(hostRoot, moduleBuilder.getI64IntegerAttr(1),
                                           redundantCopies, visited);

              // Eliminate redundant same-space copies and their clone buffers.
              // The dealloc pipeline creates: clone = alloc; copy src, clone; use clone; dealloc clone
              // After promoting both sides to space 1, we replace clone with src and remove the copy.
              // Use-replacements are safe during the walk (they don't invalidate iterators),
              // but actual erasures are deferred to avoid corrupting the outer func.walk.
              for (auto *redundant : redundantCopies) {
                if (auto copyOp = llvm::dyn_cast<memref::CopyOp>(redundant)) {
                  Value src = copyOp.getSource();
                  Value dst = copyOp.getTarget();

                  // Collect deallocs of dst BEFORE replacing uses
                  for (auto *user : dst.getUsers()) {
                    if (isa<memref::DeallocOp>(user))
                      deferredErase.insert(user);
                  }

                  // Replace all uses of clone (dst) with original (src)
                  dst.replaceAllUsesWith(src);

                  // Mark the copy for deferred erasure
                  deferredErase.insert(copyOp);

                  // Mark clone's alloc for deferred erasure if now unused
                  if (Operation *allocOp = dst.getDefiningOp()) {
                    if (allocOp->use_empty())
                      deferredErase.insert(allocOp);
                  }
                } else {
                  deferredErase.insert(redundant);
                }
              }

              // The alloc is now space 1 — operand already points to the right value
              continue;
          }

          if (!deviceRoot) continue;

          // Rebuild View Chain on Device Root
          OpBuilder launchBuilder(launchOp);
          Value currentDeviceVal = deviceRoot;

          // viewChain is Op* list. Iterate reverse (Root -> Leaf)
          for (int i = viewChain.size() - 1; i >= 0; --i) {
              Operation* oldView = viewChain[i];
              IRMapping mapping;
              mapping.map(oldView->getOperand(0), currentDeviceVal);
              Operation* newView = launchBuilder.clone(*oldView, mapping);

              auto oldType = llvm::cast<MemRefType>(oldView->getResult(0).getType());
              auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                             oldType.getLayout(), moduleBuilder.getI64IntegerAttr(1));
              newView->getResult(0).setType(newType);
              currentDeviceVal = newView->getResult(0);
          }

          // Replace usage
          operand.set(currentDeviceVal);
        }
      });
    });

    // Phase 3: Erase redundant ops collected during Phase 2 (deferred to
    // avoid iterator invalidation in the walk).
    for (auto *op : deferredErase)
      op->erase();

    // Phase 4: Insert DeallocOps for global device copies after their last use.
    // These allocs were created in Phase 2 (Cases B/C) but the buffer
    // deallocation pipeline ran before this pass, so it never saw them.
    Block *entryBlock = &func.getBody().front();
    for (auto &[symbol, deviceAlloc] : symbolToDeviceAlloc) {
      Operation *lastUse = findLastUseInBlock(deviceAlloc, entryBlock);
      if (lastUse) {
        OpBuilder deallocBuilder(lastUse->getBlock(),
                                 std::next(lastUse->getIterator()));
        deallocBuilder.create<memref::DeallocOp>(deviceAlloc.getLoc(),
                                                  deviceAlloc);
      }
    }

    // Debug: Dump the module to see changes
    // llvm::errs() << "[[[ IR after AddGpuMemoryCopies ]]]\n";
    // module.dump();
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