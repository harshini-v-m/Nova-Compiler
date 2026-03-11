// Nova Convert Shared Memory Allocs Pass
//
// Ported from IREE's ConvertSharedMemAllocOp and DropSharedMemoryDeallocOp:
//   iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.cpp  (lines 152-203)
//   iree/compiler/Codegen/Common/GPU/GPUPatterns.cpp  (lines 211-223)
//
// Two rewrite patterns:
//   1. ConvertSharedMemAllocOp: converts memref.alloc with workgroup address
//      space into a memref.global + memref.get_global pair. GPU shared memory
//      must be statically declared at module level, not dynamically allocated.
//   2. DropSharedMemoryDeallocOp: erases memref.dealloc for workgroup memory
//      since shared memory is static and freed when the kernel ends.
//
// This pass must run AFTER bufferization (which produces memref.alloc/dealloc)
// and BEFORE any LLVM lowering (which would try to lower alloc → malloc).

#include "Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::nova {

/// Returns true if the memref type has workgroup (shared) memory address space.
/// Mirrors IREE's hasSharedMemoryAddressSpace (GPUUtils.cpp:1220-1225).
static bool hasSharedMemoryAddressSpace(MemRefType memrefType) {
  auto addrSpace =
      dyn_cast_if_present<gpu::AddressSpaceAttr>(memrefType.getMemorySpace());
  return addrSpace &&
         addrSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

namespace {

/// Converts memref.alloc with workgroup address space into a module-level
/// memref.global declaration + memref.get_global.
///
/// Ported from IREE's ConvertSharedMemAllocOp (ConvertToLLVM.cpp:152-203).
///
/// In CUDA, workgroup (shared) memory is represented by a global variable
/// in address space 3. It cannot be dynamically allocated inside kernels.
struct ConvertSharedMemAllocOp : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Only handle workgroup (shared) memory allocations.
    if (!hasSharedMemoryAddressSpace(allocOp.getType())) {
      return failure();
    }

    // Shared memory must be statically shaped.
    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (ShapedType::isDynamicShape(shape)) {
      return failure();
    }

    // Compute alignment.
    uint64_t alignment;
    if (std::optional<uint64_t> alignmentInfo = allocOp.getAlignment()) {
      alignment = alignmentInfo.value();
    } else {
      // If no alignment specified, align at least to the size of an element.
      Type elType = allocOp.getType().getElementType();
      if (auto shapeType = dyn_cast<ShapedType>(elType)) {
        alignment =
            shapeType.getNumElements() * shapeType.getElementTypeBitWidth() / 8;
      } else if (elType.isIndex()) {
        // Default to 8 bytes for index types (64-bit).
        alignment = 8;
      } else {
        // Alignment must be at least 1 byte and a power of 2.
        // For sub-byte types (e.g. i1), ceil to 1.
        alignment = std::max<uint64_t>(
            llvm::PowerOf2Ceil(elType.getIntOrFloatBitWidth() / 8), 1);
      }
    }

    // Create a memref.global at the nearest symbol-table scope.
    // Using SymbolTable::getNearestSymbolTable instead of getParentOfType<ModuleOp>
    // so this pattern works inside BOTH builtin.module AND gpu.module (which also
    // implements the SymbolTable trait but is not a ModuleOp subclass).
    MemRefType allocType = allocOp.getType();
    auto funcOp = allocOp->getParentOfType<mlir::FunctionOpInterface>();
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(funcOp->getParentOp());
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

    // Replace alloc with get_global at the kernel function entry.
    rewriter.setInsertionPointToStart(&(*funcOp.getFunctionBody().begin()));
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, global.getType(),
                                                     global.getName());
    return success();
  }
};

/// Erases ALL memref.dealloc ops inside GPU modules. GPU kernels don't need
/// explicit deallocation — shared memory is static and private memory is freed
/// when the kernel terminates. Without this, bufferization-generated deallocs
/// lower to llvm.call @free which doesn't exist in GPU device code.
struct DropGPUMemoryDeallocOp : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pass that converts shared memory allocs to globals and drops deallocs.
/// Runs on any module-like op (builtin.module or gpu.module) so it can be
/// nested inside a gpu::GPUModuleOp pass manager after kernel outlining.
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
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-convert-shared-mem-allocs";
  }
  StringRef getDescription() const override {
    return "Converts workgroup memref.alloc to memref.global and drops "
           "workgroup memref.dealloc ops for GPU shared memory lowering";
  }
};

/// Pass that converts #gpu.address_space<private> on memref types to NVVM
/// integer address space 5 (local memory). Workgroup and global address
/// spaces are left as-is because they are handled by the existing
/// ConvertSharedMemAllocs and gpu-to-nvvm passes respectively.
///
/// Must run inside gpu.module BEFORE finalizeMemRefToLLVMConversionPass,
/// which requires integer address spaces on all memrefs.
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
    // Only convert private address space; leave workgroup/global for
    // downstream passes (ConvertSharedMemAllocs, gpu-to-nvvm).
    // Map private to address space 0 (generic).  On NVPTX, alloca in AS 0
    // still lands in local memory, but uses generic pointers that LLVM's
    // LoopStrengthReduce can optimise without hitting the non-default-AS
    // assertion in ScalarEvolutionExpander.
    replacer.addReplacement(
        [&](gpu::AddressSpaceAttr attr) -> std::optional<Attribute> {
          if (attr.getValue() == gpu::AddressSpace::Private)
            return IntegerAttr::get(IntegerType::get(ctx, 64), /*generic=*/0);
          return std::nullopt; // keep workgroup/global as-is
        });
    replacer.addReplacement([&](MemRefType type) -> std::optional<Type> {
      auto space =
          dyn_cast_if_present<gpu::AddressSpaceAttr>(type.getMemorySpace());
      if (!space || space.getValue() != gpu::AddressSpace::Private)
        return std::nullopt;
      return MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout(),
                             IntegerAttr::get(IntegerType::get(ctx, 64), 0));
    });
    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }

  StringRef getArgument() const override {
    return "nova-gpu-lower-memory-space";
  }
  StringRef getDescription() const override {
    return "Converts #gpu.address_space<private> to generic address space 0 "
           "to avoid LLVM LSR issues with non-default address spaces on NVPTX";
  }
};

} // namespace

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
