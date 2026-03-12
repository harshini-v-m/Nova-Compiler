//===- NovaGPUInferMemorySpace.cpp - Infer GPU memory spaces pre-bufferize ===//
//
// Assigns gpu memory-space attributes to unattributed
// `bufferization.alloc_tensor` ops just before bufferization.
//
// IMPORTANT: This pass must run immediately before OneShotBufferize so that
// the GPU-aware allocation function (gpuRequireMemSpaceAllocationFn) receives
// the correct MemRefType and can emit memref.alloc (workgroup/shared SRAM)
// vs memref.alloca (private/register) correctly.
//
// IREE equivalent: GPUInferMemorySpacePass
//   (iree/compiler/src/iree/compiler/Codegen/Common/GPU/GPUInferMemorySpace.cpp)
//
// Decision rules (same as IREE):
//   1. If the alloc already has memory_space = private or workgroup → keep it.
//   2. If all users of the alloc are thread-mapped scf.forall ops (i.e., the
//      alloc is a shared_outs init) → workgroup (shared) memory.
//   3. If the alloc is defined at function body level AND a user is a
//      workgroup-level scf.forall → leave without space (global memory).
//   4. Otherwise → private (register) memory.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::nova {

/// Returns true if the given scf.forall has a thread-level mapping attribute
/// (i.e., any dimension is mapped to #gpu.thread<...>).
static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value())
    return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute attr) {
    return isa<gpu::GPUThreadMappingAttr>(attr);
  });
}

/// Returns true if the given scf.forall has a block-level mapping attribute
/// (i.e., a workgroup-level distribution loop).
static bool isWorkgroupForall(scf::ForallOp forall) {
  if (!forall.getMapping().has_value())
    return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute attr) {
    return isa<gpu::GPUBlockMappingAttr>(attr);
  });
}

// CORE LOGIC — Returns true when `alloc` is definitely shared memory.
//
// An alloc is shared if ALL of its users are thread-mapped scf.forall ops.
// This mirrors IREE's isDefinitelyShared() heuristic — the alloc becomes a
// shared workgroup buffer when it is passed as shared_outs to a thread forall.
// Any non-forall user (e.g. an extract_slice going into a scalar op) means we
// cannot guarantee the buffer is only accessed from thread-collective code.
static bool isDefinitelyShared(bufferization::AllocTensorOp alloc) {
  for (auto *user : alloc->getUsers()) {
    // Thread-mapped forall → this alloc becomes a shared output.
    if (auto forallOp = dyn_cast<scf::ForallOp>(user)) {
      if (hasThreadMapping(forallOp))
        continue; // OK – this use makes it shared
    }
    // Any other use means we cannot be certain it's shared.
    return false;
  }
  return true;
}

// CORE LOGIC — Returns true when `alloc` is used across workgroup boundaries.
//
// Such buffers must live in global memory (no address space tag), so that the
// GPU runtime can pass them between host allocations and kernels.
// A buffer is cross-workgroup if:
//   - It is defined at the function body level (not inside any forall/launch).
//   - At least one user is a workgroup-level scf.forall (it is passed as
//     shared_outs init for workgroup distribution).
static bool isCrossWorkgroupUsed(bufferization::AllocTensorOp alloc) {
  // If defined inside a forall, it's local to that scope.
  auto parentForall = alloc->getParentOfType<scf::ForallOp>();
  if (parentForall)
    return false;

  // Check if any user is a workgroup-level forall.
  for (auto *user : alloc->getUsers()) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(user)) {
      if (isWorkgroupForall(forallOp))
        return true;
    }
  }
  return false;
}

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

    auto privateSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getPrivateAddressSpace());
    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    bool failed = false;
    funcOp.walk([&](bufferization::AllocTensorOp alloc) {
      // Skip allocs that already carry a valid gpu memory space.
      std::optional<Attribute> existingSpace = alloc.getMemorySpace();
      if (existingSpace.has_value()) {
        if (*existingSpace == privateSpace || *existingSpace == workgroupSpace)
          return; // Already correctly attributed.
        alloc.emitOpError("unexpected memory space – must be private or "
                          "workgroup (got ") << *existingSpace << ")";
        failed = true;
        return;
      }

      // Infer based on usage pattern.
      if (isDefinitelyShared(alloc)) {
        alloc.setMemorySpaceAttr(workgroupSpace);
      } else if (isCrossWorkgroupUsed(alloc)) {
        // Cross-workgroup buffer: leave without memory space (global memory).
        // The allocation function will emit a plain memref.alloc.
      } else {
        alloc.setMemorySpaceAttr(privateSpace);
      }
    });

    if (failed)
      signalPassFailure();
  }

  StringRef getArgument() const override {
    return "nova-gpu-infer-memory-space";
  }
  StringRef getDescription() const override {
    return "Infers GPU memory spaces for bufferization.alloc_tensor ops "
           "before bufferization (shared vs private)";
  }
};

std::unique_ptr<Pass> createNovaGPUInferMemorySpacePass() {
  return std::make_unique<NovaGPUInferMemorySpacePass>();
}

void registerNovaGPUInferMemorySpacePass() {
  PassRegistration<NovaGPUInferMemorySpacePass>();
}

} // namespace mlir::nova
