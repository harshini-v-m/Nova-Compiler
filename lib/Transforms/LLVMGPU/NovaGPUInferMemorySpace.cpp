// Nova GPU Infer Memory Space Pass
//
// This pass assigns gpu memory-space attributes to unattributed
// `bufferization.alloc_tensor` ops just before bufferization.
//
// IREE equivalent: GPUInferMemorySpacePass
// (iree/compiler/src/iree/compiler/Codegen/Common/GPU/GPUInferMemorySpace.cpp)
//
// Decision rules (same as IREE):
//   1. If the alloc already has memory_space = private or workgroup → keep it.
//   2. If all I/O-like users of the alloc are thread-mapped scf.forall ops
//      (i.e., the alloc is a shared_outs init) → workgroup (shared) memory.
//   3. Otherwise → private (register) memory.
//
// Run this pass immediately before OneShotBufferize so that the allocation
// function receives the right MemRefType and can emit memref.alloc (workgroup)
// vs memref.alloca (private) correctly.

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

/// Returns true when `alloc` is *definitely* shared memory.
/// This mirrors IREE's isDefinitelyShared():
///   - The alloc is used as the initial value for the shared_outs of ≥1
///     thread-mapped scf.forall ops AND no other non-forall users exist.
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
