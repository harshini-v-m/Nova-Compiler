//===- NovaConvertSharedMemAllocs.cpp - Shared memory alloc lowering ------===//
//
// Two rewrite patterns that handle GPU shared (workgroup) memory:
//
//   ConvertSharedMemAllocOp:
//     Converts memref.alloc with a workgroup address space into a module-level
//     memref.global declaration + memref.get_global.
//
//     In CUDA, workgroup (shared) memory is represented as a global variable
//     in address space 3. It cannot be dynamically allocated inside kernels
//     (address-space 3 has no malloc). Bufferization emits memref.alloc;
//     this pass converts those to the static declaration form that PTX expects.
//
//   DropGPUMemoryDeallocOp:
//     Erases all memref.dealloc ops inside GPU modules. GPU shared memory is
//     static and freed automatically when the kernel terminates. Without this,
//     bufferization-generated deallocs lower to `llvm.call @free` which does
//     not exist in GPU device code.
//
// Also contains NovaGPULowerMemorySpacePass, which converts
// #gpu.address_space<private> on memref types to generic address space 0
// before finalizeMemRefToLLVM. This avoids an assertion in
// LLVM's ScalarEvolutionExpander (via LoopStrengthReduce) that fires on
// memref ops in non-default address spaces on NVPTX.
//
// Run order:
//   This pass must run AFTER bufferization (which produces the alloc/dealloc
//   ops) and BEFORE any LLVM lowering (which would incorrectly lower
//   a workgroup alloc to a malloc call).
//
// Ported from IREE:
//   iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.cpp   lines 152–203
//   iree/compiler/Codegen/Common/GPU/GPUPatterns.cpp   lines 211–223
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

/// Returns true if the memref type has a workgroup (shared) memory address
/// space.  Mirrors IREE's hasSharedMemoryAddressSpace (GPUUtils.cpp:1220).
static bool hasSharedMemoryAddressSpace(MemRefType memrefType) {
  auto addrSpace =
      dyn_cast_if_present<gpu::AddressSpaceAttr>(memrefType.getMemorySpace());
  return addrSpace &&
         addrSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertSharedMemAllocOp
//
// CORE LOGIC — converts memref.alloc with workgroup address space into a
// module-level memref.global declaration + memref.get_global.
//
// Ported from IREE's ConvertSharedMemAllocOp (ConvertToLLVM.cpp:152–203).
//===----------------------------------------------------------------------===//
struct ConvertSharedMemAllocOp : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Only handle workgroup (shared) memory allocations.
    if (!hasSharedMemoryAddressSpace(allocOp.getType()))
      return failure();

    // Shared memory must be statically shaped — dynamic shared memory requires
    // a different lowering path (extern __shared__ arrays) not yet implemented.
    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (ShapedType::isDynamicShape(shape))
      return failure();

    // GPU / MEMORY SENSITIVE — Alignment computation.
    // The PTX ISA requires that shared memory buffers are aligned to at least
    // the size of one element.  Under-alignment causes memory access exceptions
    // in the SM hardware. We prefer the user-supplied alignment; fall back to
    // element-size alignment otherwise.
    uint64_t alignment;
    if (std::optional<uint64_t> alignmentInfo = allocOp.getAlignment()) {
      alignment = alignmentInfo.value();
    } else {
      Type elType = allocOp.getType().getElementType();
      if (auto shapeType = dyn_cast<ShapedType>(elType)) {
        alignment =
            shapeType.getNumElements() * shapeType.getElementTypeBitWidth() / 8;
      } else if (elType.isIndex()) {
        alignment = 8; // 64-bit index type
      } else {
        // Use 128-byte alignment to enable v2.b64 vectorized SMEM stores
        // (ld.global.v2.b64 / st.shared.v2.b64). The PTX ISA only emits
        // 128-bit vector memory ops when the buffer is 128-byte aligned.
        // Minimum 1 byte, power-of-2, clamped up to 128 for f32/f16 tiles.
        uint64_t elemAlign = std::max<uint64_t>(
            llvm::PowerOf2Ceil(elType.getIntOrFloatBitWidth() / 8), 1);
        alignment = std::max<uint64_t>(elemAlign, 128);
      }
    }

    // IMPORTANT: Use SymbolTable::getNearestSymbolTable instead of
    // getParentOfType<ModuleOp>. This pattern runs nested inside a
    // gpu::GPUModuleOp (which implements SymbolTable but is NOT a ModuleOp
    // subclass), so walking up to the parent plain ModuleOp would escape the
    // gpu.module and place the global in the wrong scope.
    MemRefType allocType = allocOp.getType();
    auto funcOp = allocOp->getParentOfType<mlir::FunctionOpInterface>();
    Operation *symbolTableOp =
        SymbolTable::getNearestSymbolTable(funcOp->getParentOp());
    SymbolTable symbolTable(symbolTableOp);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&symbolTableOp->getRegion(0).front().front());
    auto global = memref::GlobalOp::create(
        rewriter, funcOp.getLoc(), "__shared_memory__",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/allocType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false,
        /*alignment=*/rewriter.getI64IntegerAttr(alignment));
    symbolTable.insert(global);

    // Replace the alloc with a get_global at the kernel function entry so the
    // shared buffer is visible before any intra-kernel use.
    rewriter.setInsertionPointToStart(&(*funcOp.getFunctionBody().begin()));
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, global.getType(),
                                                     global.getName());
    return success();
  }
};

// ConvertGlobalMemAllocOp: DELETED
//
// This pattern previously handled untagged memref.alloc ops inside gpu.func
// using a three-tier size heuristic (≤1KB→alloca, ≤48KB→global, >48KB→global).
// After the InferMemorySpace + bufferizer refactor, every allocation is tagged
// as either workgroup or private before bufferization. There are no untagged
// memref.alloc ops inside gpu.func, so this pattern is no longer needed.

//===----------------------------------------------------------------------===//
// DropGPUMemoryDeallocOp
//
// Erases ALL memref.dealloc ops inside GPU modules.  GPU kernels do not have
// explicit deallocation semantics — shared memory is freed when the kernel
// terminates and register/stack memory is managed by the SM hardware.
// Bufferization-generated deallocs would incorrectly lower to `llvm.call @free`
// which does not exist in GPU device code (no libc inside kernels).
//===----------------------------------------------------------------------===//
struct DropGPUMemoryDeallocOp : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NovaConvertSharedMemAllocsPass
//
// Pass that applies ConvertSharedMemAllocOp + DropGPUMemoryDeallocOp on any
// module-like op (builtin.module or gpu.module). Nested inside
// gpu::GPUModuleOp after kernel outlining.
//===----------------------------------------------------------------------===//
struct NovaConvertSharedMemAllocsPass
    : public PassWrapper<NovaConvertSharedMemAllocsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaConvertSharedMemAllocsPass)

  NovaConvertSharedMemAllocsPass() = default;
  NovaConvertSharedMemAllocsPass(const NovaConvertSharedMemAllocsPass &) =
      default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertSharedMemAllocOp>(&getContext());
    patterns.add<DropGPUMemoryDeallocOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

  StringRef getArgument() const override {
    return "nova-convert-shared-mem-allocs";
  }
  StringRef getDescription() const override {
    return "Converts workgroup memref.alloc to memref.global and drops "
           "workgroup memref.dealloc ops for GPU shared memory lowering";
  }
};

//===----------------------------------------------------------------------===//
// NovaGPULowerMemorySpacePass
//
// Converts #gpu.address_space<private> on memref types to generic integer
// address space 0.
//
// IMPORTANT: This mapping (private → AS 0) is intentional.  On NVPTX, alloca
// in AS 0 still lands in local (register/stack) memory, but uses generic
// pointers that LLVM's LoopStrengthReduce (via ScalarEvolutionExpander) can
// optimise without hitting the non-default-AS assertion that fires when private
// address spaces are still present during that LLVM pass.
//
// Workgroup and global address spaces are left unchanged — they are handled
// by ConvertSharedMemAllocs and gpu-to-nvvm respectively.
//===----------------------------------------------------------------------===//
struct NovaGPULowerMemorySpacePass
    : public PassWrapper<NovaGPULowerMemorySpacePass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPULowerMemorySpacePass)

  NovaGPULowerMemorySpacePass() = default;
  NovaGPULowerMemorySpacePass(const NovaGPULowerMemorySpacePass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    AttrTypeReplacer replacer;

    // Map all #gpu.address_space tokens to integer address spaces. Workgroup
    // memrefs MUST be rewritten to AS 3 here so that finalize-memref-to-llvm
    // can lower them — leaving them as symbolic gpu::AddressSpaceAttr makes
    // that pass emit "conversion of memref memory space ... failed".
    replacer.addReplacement(
        [&](gpu::AddressSpaceAttr attr) -> std::optional<Attribute> {
          unsigned as = 0;
          switch (attr.getValue()) {
          case gpu::AddressSpace::Private:   as = 0; break;
          case gpu::AddressSpace::Global:    as = 1; break;
          case gpu::AddressSpace::Workgroup: as = 3; break;
          default: return std::nullopt;
          }
          return IntegerAttr::get(IntegerType::get(ctx, 64), as);
        });

    // Also remap the MemRefType itself so structural type equality is maintained
    // after replacing the address-space attribute inside it.
    replacer.addReplacement([&](MemRefType type) -> std::optional<Type> {
      auto space =
          dyn_cast_if_present<gpu::AddressSpaceAttr>(type.getMemorySpace());
      if (!space)
        return std::nullopt;

      unsigned as = 0;
      switch (space.getValue()) {
      case gpu::AddressSpace::Private:   as = 0; break;
      case gpu::AddressSpace::Global:    as = 1; break;
      case gpu::AddressSpace::Workgroup: as = 3; break;
      default: return std::nullopt;
      }

      return MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout(),
                             IntegerAttr::get(IntegerType::get(ctx, 64), as));
    });

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    // ── Fix verifier for gpu.dynamic_shared_memory ─────────────────────────
    // After the bulk remap above, gpu.dynamic_shared_memory's result type
    // carries integer AS 3 — but the op's ODS verifier requires the symbolic
    // gpu::AddressSpaceAttr<workgroup>. Restore the symbolic type on the
    // result and bridge to its AS-3 consumers via unrealized_conversion_cast.
    // gpu-to-nvvm later folds the cast away when lowering.
    op->walk([&](gpu::DynamicSharedMemoryOp smemOp) {
      Value result = smemOp.getResult();
      auto currentTy = dyn_cast<MemRefType>(result.getType());
      if (!currentTy)
        return;
      // Only fix if the result is currently in integer AS 3.
      auto space = currentTy.getMemorySpace();
      auto intSpace = dyn_cast_if_present<IntegerAttr>(space);
      if (!intSpace || intSpace.getInt() != 3)
        return;

      auto workgroupAS = gpu::AddressSpaceAttr::get(
          ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
      auto symbolicTy = MemRefType::get(
          currentTy.getShape(), currentTy.getElementType(),
          currentTy.getLayout(), workgroupAS);

      // CRITICAL: insert the cast and replace consumer uses BEFORE changing
      // the producer's type, otherwise the producer/consumer types disagree
      // mid-rewrite and the verifier fires inside the walk.
      OpBuilder b(smemOp);
      b.setInsertionPointAfter(smemOp);
      auto castOp = b.create<UnrealizedConversionCastOp>(
          smemOp.getLoc(), currentTy, result);
      result.replaceAllUsesExcept(castOp.getResult(0), castOp);
      result.setType(symbolicTy);
    });
  }

  StringRef getArgument() const override {
    return "nova-gpu-lower-memory-space";
  }
  StringRef getDescription() const override {
    return "Converts #gpu.address_space<private> to generic address space 0 "
           "to avoid LLVM LSR issues with non-default address spaces on NVPTX";
  }
};

// NovaGPUPromoteGlobalsToSharedPass: DELETED
//
// This pass previously promoted __global_memory__ LLVM globals from device
// global (AS 0) to shared memory (AS 3). It existed solely to fix up the
// output of ConvertGlobalMemAllocOp, which has also been deleted.
// After the InferMemorySpace refactor, promoted operands are tagged as
// workgroup before bufferization and become memref.global in AS 3 directly
// via ConvertSharedMemAllocOp. No post-hoc promotion is needed.

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaConvertSharedMemAllocsPass() {
  return std::make_unique<NovaConvertSharedMemAllocsPass>();
}

std::unique_ptr<Pass> createNovaGPULowerMemorySpacePass() {
  return std::make_unique<NovaGPULowerMemorySpacePass>();
}

void registerNovaConvertSharedMemAllocsPass() {
  PassRegistration<NovaConvertSharedMemAllocsPass>();
}

void registerNovaGPULowerMemorySpacePass() {
  PassRegistration<NovaGPULowerMemorySpacePass>();
}

} // namespace mlir::nova
