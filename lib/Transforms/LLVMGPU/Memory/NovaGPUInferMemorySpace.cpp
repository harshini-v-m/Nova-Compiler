//===- NovaGPUInferMemorySpace.cpp - Structural memory-space annotation ====//
//
// Pre-bufferization pass.  Annotates bufferization.alloc_tensor ops with
// the appropriate GPU address space before OneShotBufferize runs.
//
// Decision logic (mirrors IREE's GPUInferMemorySpacePass):
//
//   Shared (workgroup)  (#gpu.address_space<workgroup>)
//     The alloc_tensor is used as shared_outs of a thread-distributed
//     scf.forall that is nested inside a block-mapped forall.
//
//   Private  (#gpu.address_space<private>)
//     Everything else.  Thread-local by default.
//
// CSE de-aliasing:
//   Nova's pipeline runs CSE between PromoteMatmulOperands and this pass.
//   CSE merges all tensor.empty() ops with the same type into one SSA value.
//   After EmptyTensorToAllocTensor, this becomes a single alloc_tensor with
//   mixed users — some thread-forall (need workgroup) and some block-forall
//   or other ops (need private).
//
//   To handle this, we first split such mixed-user alloc_tensors: each
//   thread-forall-inside-block-forall use gets its own cloned alloc_tensor.
//   After splitting, each alloc_tensor has homogeneous memory-space needs.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-infer-memory-space"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// forall mapping helpers
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUThreadMappingAttr>(a);
  });
}

static bool hasWarpMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUWarpMappingAttr>(a);
  });
}

static bool hasBlockMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUBlockMappingAttr>(a);
  });
}

//===----------------------------------------------------------------------===//
// isInsideWorkgroupForall
//===----------------------------------------------------------------------===//

static bool isInsideWorkgroupForall(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(parent)) {
      if (hasBlockMapping(forallOp))
        return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// isThreadForallInsideBlockForall
//
// Returns true if `op` is a thread-mapped scf.forall nested inside a
// block-mapped scf.forall.  This is the structural signature of a promoted
// operand cooperative copy: threads cooperatively fill a shared buffer.
//===----------------------------------------------------------------------===//

static bool isThreadForallInsideBlockForall(Operation *op) {
  auto forallOp = dyn_cast<scf::ForallOp>(op);
  if (!forallOp)
    return false;
  // Thread-mapped OR warp-mapped foralls inside a block forall need workgroup
  // (shared) memory for their promoted operand buffers.  Warp-mapped foralls
  // are created by subgroup tiling for the MMA (tensor core) path — the
  // promoted A/B copies must be in shared memory so all warps can read them
  // after a barrier.
  if (!hasThreadMapping(forallOp) && !hasWarpMapping(forallOp))
    return false;
  return isInsideWorkgroupForall(forallOp);
}

//===----------------------------------------------------------------------===//
// splitMixedUserAllocTensors
//
// CSE merges all tensor.empty() with the same type into one SSA value.
// After EmptyTensorToAllocTensor, one alloc_tensor may feed both:
//   - thread-mapped foralls inside block foralls (need workgroup memory)
//   - block-mapped foralls or other ops (need private memory)
//
// This function clones the alloc_tensor for each workgroup-bound use so
// that every alloc_tensor has a single memory-space need.
//===----------------------------------------------------------------------===//

static void splitMixedUserAllocTensors(func::FuncOp funcOp) {
  SmallVector<bufferization::AllocTensorOp> allocs;
  funcOp.walk([&](bufferization::AllocTensorOp alloc) {
    if (!alloc.getMemorySpace().has_value())
      allocs.push_back(alloc);
  });

  for (auto alloc : allocs) {
    // Classify each use of this alloc_tensor.
    SmallVector<OpOperand *> sharedUses, otherUses;
    for (OpOperand &use : alloc->getUses()) {
      if (isThreadForallInsideBlockForall(use.getOwner()))
        sharedUses.push_back(&use);
      else
        otherUses.push_back(&use);
    }

    // Only split if there are BOTH shared and non-shared uses.
    if (sharedUses.empty() || otherUses.empty())
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]  splitting alloc_tensor: "
               << sharedUses.size() << " shared + "
               << otherUses.size() << " other uses\n");

    // Clone the alloc_tensor for each shared (thread-forall) use.
    OpBuilder builder(alloc);
    for (OpOperand *use : sharedUses) {
      builder.setInsertionPoint(use->getOwner());
      auto clone = cast<bufferization::AllocTensorOp>(builder.clone(*alloc));
      use->set(clone.getResult());
    }
    // The original alloc_tensor now only has non-shared (private) uses.
  }
}

//===----------------------------------------------------------------------===//
// isDefinitelyShared
//
// After splitting, each alloc_tensor has homogeneous users.
// Mirrors IREE's isDefinitelyShared: an alloc is shared if every user is
// a thread/warp-mapped scf.forall.  We additionally require that user to
// be inside a block-mapped forall (workgroup scope).
//===----------------------------------------------------------------------===//

static bool isDefinitelyShared(bufferization::AllocTensorOp alloc) {
  // An allocation can only be in workgroup (shared) memory if it is
  // defined inside a block-mapped forall (the workgroup scope).
  // If it's outside (at the host/func level), it must be private or global.
  if (!isInsideWorkgroupForall(alloc))
    return false;

  bool hasThreadForallUser = false;
  for (Operation *user : alloc->getUsers()) {
    if (!isThreadForallInsideBlockForall(user))
      return false;
    hasThreadForallUser = true;
  }
  return hasThreadForallUser;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUInferMemorySpacePass
    : public PassWrapper<NovaGPUInferMemorySpacePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUInferMemorySpacePass)

  NovaGPUInferMemorySpacePass() = default;
  NovaGPUInferMemorySpacePass(const NovaGPUInferMemorySpacePass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();

    // Step 1: Split alloc_tensors that CSE merged into mixed-user values.
    splitMixedUserAllocTensors(funcOp);

    auto privateSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getPrivateAddressSpace());
    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    // Step 2: Tag each alloc_tensor with its inferred memory space.
    WalkResult res = funcOp.walk([&](bufferization::AllocTensorOp alloc) {
      std::optional<Attribute> existingSpace = alloc.getMemorySpace();
      if (existingSpace.has_value()) {
        if (*existingSpace == workgroupSpace ||
            *existingSpace == privateSpace) {
          return WalkResult::advance();
        }
        alloc.emitOpError(
            "unexpected gpu memory space — must be private or workgroup");
        return WalkResult::interrupt();
      }

      if (isDefinitelyShared(alloc)) {
        alloc.setMemorySpaceAttr(workgroupSpace);
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  workgroup (thread-forall user)\n");
      } else if (isInsideWorkgroupForall(alloc)) {
        alloc.setMemorySpaceAttr(privateSpace);
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  private (default)\n");
      }
      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      funcOp->emitOpError("failed to set gpu memory space for all "
                          "bufferization.alloc_tensor ops");
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-infer-memory-space";
  }
  StringRef getDescription() const override {
    return "Structural GPU memory space inference with CSE de-aliasing: "
           "shared (thread-forall inside block forall) or private";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUInferMemorySpacePass() {
  return std::make_unique<NovaGPUInferMemorySpacePass>();
}

void registerNovaGPUInferMemorySpacePass() {
  PassRegistration<NovaGPUInferMemorySpacePass>();
}

} // namespace mlir::nova
