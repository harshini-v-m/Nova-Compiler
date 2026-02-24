// Nova GPU Comprehensive Bufferize Pass
//
// This is Nova's equivalent of IREE's IREEComprehensiveBufferizePass.
// It performs three jobs in one pass:
//
//  1. GAP 2 — Erase nova.fusion_barrier ops (they are pure identities that
//             only block fusion analysis during tiling; they have no
//             bufferized form and must be removed before bufferization).
//
//  2. GAP 3 — Run OneShotBufferize with GPU-aware allocation + copy functions:
//
//     allocationFn (mirrors IREE's gpuRequireMemSpaceAllocationFn):
//       * #gpu.address_space<workgroup>  → memref.alloc  (shared SRAM)
//       * #gpu.address_space<private>   → memref.alloca (per-thread register)
//       * no memory space specified     → memref.alloca (default, no space tag)
//
//     memCpyFn:
//       * emits memref.copy
//       * wraps the copy with gpu.barrier when either operand is in
//         workgroup (shared) memory.
//
// IREE reference:
//   IREEComprehensiveBufferizePass::runOnOperation()
//   (iree/compiler/src/iree/compiler/Codegen/Common/IREEComprehensiveBufferizePass.cpp)

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::nova {

// ---------------------------------------------------------------------------
// GPU allocation function — mirrors IREE's gpuRequireMemSpaceAllocationFn
// ---------------------------------------------------------------------------
static FailureOr<Value> gpuRequireMemSpaceAllocationFn(OpBuilder &builder,
                                                        Location loc,
                                                        MemRefType memRefType,
                                                        ValueRange dynamicSizes,
                                                        unsigned alignment) {
  Attribute memSpace = memRefType.getMemorySpace();
  if (memSpace && !isa<gpu::AddressSpaceAttr>(memSpace))
    return failure();

  auto privSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  auto wkgpSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  // No explicit GPU address space → default to private (per-thread).
  // Note: This may create large private allocas for the output buffer.
  // This is resolved later when scf.forall → gpu.launch converts
  // the function into a kernel where output is in global memory.
  if (!memSpace) {
    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), privSpace);
    return memref::AllocaOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  // Explicit private → per-thread register/stack.
  if (memSpace == privSpace)
    return memref::AllocaOp::create(builder, loc, memRefType, dynamicSizes)
        .getResult();

  // Workgroup (shared) memory → heap allocated for the whole workgroup.
  auto allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), wkgpSpace);
  return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
      .getResult();
}

// Helper: true when the memref is in GPU workgroup (shared) address space.
static bool isWorkgroupMemref(MemRefType t) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

// GPU copy function — inserts gpu.barrier before/after workgroup copies.
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool needsBarrier = isWorkgroupMemref(cast<MemRefType>(from.getType())) ||
                      isWorkgroupMemref(cast<MemRefType>(to.getType()));
  if (needsBarrier)
    gpu::BarrierOp::create(builder, loc);
  memref::CopyOp::create(builder, loc, from, to);
  if (needsBarrier)
    gpu::BarrierOp::create(builder, loc);
  return success();
}

// ---------------------------------------------------------------------------
// Pass definition — operates on func::FuncOp
// ---------------------------------------------------------------------------
struct NovaGPUComprehensiveBufferizePass
    : public PassWrapper<NovaGPUComprehensiveBufferizePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUComprehensiveBufferizePass)

  NovaGPUComprehensiveBufferizePass() = default;
  NovaGPUComprehensiveBufferizePass(
      const NovaGPUComprehensiveBufferizePass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                    gpu::GPUDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    nova::NovaDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // --- GAP 2: Erase nova.fusion_barrier ops. ----------------------------
    IRRewriter rewriter(funcOp.getContext());
    SmallVector<FusionBarrierOp> barriers;
    funcOp.walk([&](FusionBarrierOp b) { barriers.push_back(b); });
    for (FusionBarrierOp b : barriers)
      rewriter.replaceOp(b, b.getSource());

    // --- GAP 3: GPU-aware OneShotBufferize. --------------------------------
    bufferization::OneShotBufferizationOptions opts;
    opts.allocationFn = gpuRequireMemSpaceAllocationFn;
    opts.memCpyFn = gpuCopyFn;
    opts.bufferizeFunctionBoundaries = false;
    opts.checkParallelRegions = false;

    bufferization::BufferizationState bufState;
    if (failed(
            bufferization::runOneShotBufferize(funcOp, opts, bufState))) {
      funcOp.emitOpError("GPU-aware bufferization failed");
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-comprehensive-bufferize";
  }
  StringRef getDescription() const override {
    return "Erases nova.fusion_barrier ops then runs OneShotBufferize with "
           "GPU-aware allocation (workgroup=memref.alloc, "
           "private=memref.alloca) and barrier-fenced copies";
  }
};

std::unique_ptr<Pass> createNovaGPUComprehensiveBufferizePass() {
  return std::make_unique<NovaGPUComprehensiveBufferizePass>();
}

void registerNovaGPUComprehensiveBufferizePass() {
  PassRegistration<NovaGPUComprehensiveBufferizePass>();
}

} // namespace mlir::nova
