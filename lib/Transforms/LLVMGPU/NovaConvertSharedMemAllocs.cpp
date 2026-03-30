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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
        // Alignment must be at least 1 byte and a power of 2.
        alignment = std::max<uint64_t>(
            llvm::PowerOf2Ceil(elType.getIntOrFloatBitWidth() / 8), 1);
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

//===----------------------------------------------------------------------===//
// ConvertGlobalMemAllocOp
//
// Converts memref.alloc with NO address space inside gpu.func into a
// module-level memref.global declaration + memref.get_global.
//
// After bufferization, workspace tensors (shared_outs init for thread foralls)
// become memref.alloc() with no address space inside block-level foralls.
// After kernel outlining these end up inside gpu.func. Without this pattern,
// ConvertMemRefToGpu would turn them into memref.alloca (per-thread stack),
// which is catastrophic for large buffers (e.g. 1x1024x384xf32 = 1.5 MB per
// thread × 1024 threads = 1.5 GB).
//
// Instead, we promote them to device global memory (memref.global in the
// gpu.module). In PTX this becomes a .global variable — one copy per module,
// accessible by all threads. This is safe because:
//   - The buffer is written by all threads via parallel_insert_slice (each
//     thread writes to a disjoint slice)
//   - The buffer is not needed across kernel launches (workspace, not output)
//===----------------------------------------------------------------------===//
struct ConvertGlobalMemAllocOp : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Only handle allocs with NO address space (default/global memory).
    // Workgroup allocs are handled by ConvertSharedMemAllocOp.
    if (allocOp.getType().getMemorySpace())
      return failure();

    // Only inside GPU functions (after kernel outlining).
    if (!allocOp->getParentOfType<gpu::GPUFuncOp>())
      return failure();

    // Dynamic allocs inside GPU kernels: convert to a STATIC memref.alloca
    // placed at the gpu.func entry block. This is critical because:
    //   - memref.alloc with dynamic sizes inside gpu.func becomes device-side
    //     malloc() which crashes with CUDA_ERROR_ILLEGAL_ADDRESS.
    //   - Placing alloca in-place (inside a loop body) causes PTX alloca to
    //     grow the stack each iteration → stack overflow when the K-reduction
    //     loop runs many iterations (e.g. 12 × 2KB = 24KB per thread).
    //   - The dynamic dimension is bounded by the tile config (typically ≤ 16
    //     or ≤ 128) and does NOT change across loop iterations — it depends
    //     only on blockIdx and threadIdx.
    //
    // Strategy: replace each dynamic dim with the maximum possible value
    // (inferred from the tile configuration or a conservative upper bound),
    // then place the alloca at the function entry so it's allocated once.
    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (ShapedType::isDynamicShape(shape)) {
      auto funcOp = allocOp->getParentOfType<gpu::GPUFuncOp>();
      if (!funcOp)
        return failure();

      // Build a static shape by replacing dynamic dims with upper bounds.
      // For tile-based GPU kernels, dynamic dims come from boundary handling
      // (min(remaining, tileSize)) so the tile size IS the upper bound.
      SmallVector<int64_t> staticShape;
      unsigned dynIdx = 0;
      for (int64_t dim : shape) {
        if (ShapedType::isDynamic(dim)) {
          // Try to infer upper bound from the dynamic size operand.
          // Common pattern: affine.min or arith.minsi/minui with a constant
          // tile size → use that constant as the upper bound.
          int64_t upperBound = 128; // conservative default for tile dims
          if (dynIdx < allocOp.getDynamicSizes().size()) {
            Value dynSize = allocOp.getDynamicSizes()[dynIdx];
            // Walk through min operations to find the constant bound
            if (auto minOp = dynSize.getDefiningOp<arith::MinSIOp>()) {
              for (Value operand : {minOp.getLhs(), minOp.getRhs()}) {
                if (auto cst = operand.getDefiningOp<arith::ConstantIndexOp>())
                  upperBound = std::min(upperBound, cst.value());
              }
            } else if (auto minOp = dynSize.getDefiningOp<arith::MinUIOp>()) {
              for (Value operand : {minOp.getLhs(), minOp.getRhs()}) {
                if (auto cst = operand.getDefiningOp<arith::ConstantIndexOp>())
                  upperBound = std::min(upperBound, cst.value());
              }
            } else if (auto cst = dynSize.getDefiningOp<arith::ConstantIndexOp>()) {
              upperBound = cst.value();
            }
          }
          staticShape.push_back(upperBound);
          ++dynIdx;
        } else {
          staticShape.push_back(dim);
        }
      }

      auto staticType = MemRefType::get(
          staticShape, allocOp.getType().getElementType(),
          AffineMap(), allocOp.getType().getMemorySpace());

      // Place alloca at function entry — allocated once per kernel invocation.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      auto alloca = memref::AllocaOp::create(rewriter, allocOp.getLoc(),
                                              staticType);

      // Create a subview with the original dynamic sizes so downstream code
      // sees the correct (possibly smaller) shape.
      SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      dynIdx = 0;
      for (unsigned i = 0; i < shape.size(); ++i) {
        if (ShapedType::isDynamic(shape[i])) {
          sizes.push_back(allocOp.getDynamicSizes()[dynIdx++]);
        } else {
          sizes.push_back(rewriter.getIndexAttr(shape[i]));
        }
      }
      SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));

      // Insert subview at the original alloc location (where dynamic sizes
      // are available).
      rewriter.setInsertionPoint(allocOp);
      auto subview = memref::SubViewOp::create(
          rewriter, allocOp.getLoc(), alloca.getResult(), offsets, sizes,
          strides);
      rewriter.replaceOp(allocOp, subview.getResult());
      return success();
    }

    // ---- Three-tier allocation strategy for static allocs ----
    //
    // Compute buffer size to decide the right allocation strategy.
    // Different buffer sizes have fundamentally different semantics:
    //   - Small (<=1 KB): per-thread accumulators (K-loop iter_args, bias scratch)
    //   - Medium (1 KB - 48 KB): per-block tile buffers (matmul output tiles)
    //   - Large (>48 KB): workspace buffers (LN backward full [B,T,D] tensors)
    int64_t numElements = 1;
    for (int64_t dim : shape)
      numElements *= dim;
    int64_t elemBits = allocOp.getType().getElementTypeBitWidth();
    int64_t sizeBytes = numElements * (elemBits / 8);

    // Tier 1: SMALL (<=1 KB) → memref.alloca (per-thread stack).
    // Each thread gets its own copy. Correct for K-reduction accumulators
    // (e.g. memref<1x1x16xf32> = 64 bytes) where each thread accumulates
    // independently. Using memref.global here would create a single copy
    // shared by all threads → race condition on the accumulator.
    constexpr int64_t kAllocaThreshold = 1024; // 1 KB
    if (sizeBytes <= kAllocaThreshold) {
      rewriter.replaceOpWithNewOp<memref::AllocaOp>(
          allocOp, allocOp.getType(), allocOp.getDynamicSizes(),
          allocOp.getSymbolOperands());
      return success();
    }

    // Tier 2: MEDIUM (1 KB - 48 KB) → memref.global (module-level).
    // These are per-block tile buffers where all threads in the block
    // write to disjoint slices. The later NovaGPUPromoteGlobalsToSharedPass
    // promotes them from .global (cross-block shared) to .shared (per-block)
    // when they fit in the 48 KB shared memory budget.
    constexpr int64_t kSharedMemLimit = 48 * 1024; // 48 KB
    if (sizeBytes <= kSharedMemLimit) {
      uint64_t alignment;
      if (std::optional<uint64_t> alignmentInfo = allocOp.getAlignment()) {
        alignment = alignmentInfo.value();
      } else {
        alignment = std::max<uint64_t>(
            llvm::PowerOf2Ceil(elemBits / 8), 1);
      }

      MemRefType allocType = allocOp.getType();
      auto funcOp = allocOp->getParentOfType<mlir::FunctionOpInterface>();
      Operation *symbolTableOp =
          SymbolTable::getNearestSymbolTable(funcOp->getParentOp());
      SymbolTable symbolTable(symbolTableOp);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(
          &symbolTableOp->getRegion(0).front().front());
      auto global = memref::GlobalOp::create(
          rewriter, funcOp.getLoc(), "__global_memory__",
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/allocType,
          /*initial_value=*/ElementsAttr(),
          /*constant=*/false,
          /*alignment=*/rewriter.getI64IntegerAttr(alignment));
      symbolTable.insert(global);

      rewriter.setInsertionPointToStart(
          &(*funcOp.getFunctionBody().begin()));
      rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
          allocOp, global.getType(), global.getName());
      return success();
    }

    // Tier 3: LARGE (>48 KB) → memref.global (device global memory).
    //
    // Too large for per-thread stack (would overflow) and too large for
    // shared memory (48 KB limit on sm_86). We convert to memref.global
    // in the gpu.module, which becomes a .global PTX variable — one copy
    // in device global memory shared by all thread blocks.
    //
    // This is correct when the buffer is used for:
    //   - Constant fills (e.g. dOutput = dense<1.0>) — all blocks write
    //     the same value, benign write-write race.
    //   - Workspace where blocks access disjoint regions (indexed by blockIdx).
    //
    // Without this, the alloc survives to LLVM lowering and becomes a
    // device-side malloc() call, which crashes with CUDA_ERROR_ILLEGAL_ADDRESS
    // because device malloc has a tiny default heap (8 MB) and each of the
    // thousands of threads calls it independently.
    {
      uint64_t alignment;
      if (std::optional<uint64_t> alignmentInfo = allocOp.getAlignment()) {
        alignment = alignmentInfo.value();
      } else {
        alignment = std::max<uint64_t>(
            llvm::PowerOf2Ceil(elemBits / 8), 1);
      }

      MemRefType allocType = allocOp.getType();
      auto funcOp = allocOp->getParentOfType<mlir::FunctionOpInterface>();
      Operation *symbolTableOp =
          SymbolTable::getNearestSymbolTable(funcOp->getParentOp());
      SymbolTable symbolTable(symbolTableOp);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(
          &symbolTableOp->getRegion(0).front().front());
      auto global = memref::GlobalOp::create(
          rewriter, funcOp.getLoc(), "__global_memory_large__",
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/allocType,
          /*initial_value=*/ElementsAttr(),
          /*constant=*/false,
          /*alignment=*/rewriter.getI64IntegerAttr(alignment));
      symbolTable.insert(global);

      rewriter.setInsertionPointToStart(
          &(*funcOp.getFunctionBody().begin()));
      rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
          allocOp, global.getType(), global.getName());
      return success();
    }
  }
};

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
    patterns.add<ConvertGlobalMemAllocOp>(&getContext());
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

    // Map #gpu.address_space<private> → IntegerAttr(64, 0) (generic AS).
    // Workgroup and global are left as-is for downstream passes.
    replacer.addReplacement(
        [&](gpu::AddressSpaceAttr attr) -> std::optional<Attribute> {
          if (attr.getValue() == gpu::AddressSpace::Private)
            return IntegerAttr::get(IntegerType::get(ctx, 64), 0);
          if (attr.getValue() == gpu::AddressSpace::Workgroup)
            return IntegerAttr::get(IntegerType::get(ctx, 64), 3);
          if (attr.getValue() == gpu::AddressSpace::Global)
            return IntegerAttr::get(IntegerType::get(ctx, 64), 1);
          return std::nullopt;
        });

    // Also remap the MemRefType itself so structural type equality is maintained
    // after replacing the address-space attribute inside it.
    replacer.addReplacement([&](MemRefType type) -> std::optional<Type> {
      auto space =
          dyn_cast_if_present<gpu::AddressSpaceAttr>(type.getMemorySpace());
      if (!space)
        return std::nullopt;

      unsigned as = 0;
      if (space.getValue() == gpu::AddressSpace::Private)
        as = 0;
      else if (space.getValue() == gpu::AddressSpace::Workgroup)
        as = 3;
      else if (space.getValue() == gpu::AddressSpace::Global)
        as = 1;
      else
        return std::nullopt;

      return MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout(),
                             IntegerAttr::get(IntegerType::get(ctx, 64), as));
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

//===----------------------------------------------------------------------===//
// NovaGPUPromoteGlobalsToSharedPass
//
// Runs AFTER full LLVM lowering inside gpu.module.  Finds llvm.mlir.global
// ops named "__global_memory__*" (created by ConvertGlobalMemAllocOp) and
// promotes them from address space 0 (device global) to address space 3
// (shared / per-block).
//
// In PTX this turns `.global` variables into `.shared` variables, giving
// each thread block its own copy — fixing the cross-block race condition
// that occurs when multiple blocks in a grid write to the same `.global`.
//
// At the LLVM dialect level, pointers are opaque.  We change the global's
// addr_space and insert llvm.addrspacecast (ptr<3> -> ptr) at each
// llvm.mlir.addressof use so downstream GEPs/loads/stores continue to
// use generic pointers.
//===----------------------------------------------------------------------===//
struct NovaGPUPromoteGlobalsToSharedPass
    : public PassWrapper<NovaGPUPromoteGlobalsToSharedPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUPromoteGlobalsToSharedPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  // Returns the size in bytes of an LLVM global's type, or 0 if unknown.
  static uint64_t getGlobalSizeBytes(LLVM::GlobalOp global) {
    Type ty = global.getGlobalType();
    // Flatten nested arrays: e.g. !llvm.array<4096 x i8> → 4096 bytes.
    uint64_t numElements = 1;
    while (auto arrTy = dyn_cast<LLVM::LLVMArrayType>(ty)) {
      numElements *= arrTy.getNumElements();
      ty = arrTy.getElementType();
    }
    // Now ty should be the scalar element type (i8, f32, etc.).
    if (ty.isIntOrFloat())
      return numElements * (ty.getIntOrFloatBitWidth() / 8);
    // Unknown element type — conservatively return a large size to
    // prevent promotion.
    return UINT64_MAX;
  }

  void runOnOperation() override {
    Operation *moduleOp = getOperation();

    // sm_86 default static shared memory limit.  Conservative: use 48 KB
    // (the guaranteed minimum without cudaFuncSetAttribute).
    constexpr uint64_t kMaxSharedBytes = 48 * 1024;
    constexpr unsigned kSharedAS = 3;

    // Group globals by the gpu.func that uses them.  A global may be used
    // by multiple functions (unlikely after outlining, but be safe).
    // For each function, compute total shared bytes if we promoted all its
    // __global_memory__* globals + any existing __shared_memory__* globals.
    //
    // Strategy: per function, sort candidate globals by size (smallest
    // first) and greedily promote until the budget is exhausted.

    // Step 1: collect all __global_memory__* globals in this module.
    SmallVector<LLVM::GlobalOp> allCandidates;
    moduleOp->walk([&](LLVM::GlobalOp global) {
      if (global.getSymName().starts_with("__global_memory__"))
        allCandidates.push_back(global);
    });

    if (allCandidates.empty())
      return;

    // Step 2: for each function, find which globals it references and
    // compute existing shared memory usage.
    // Since globals are module-scoped and typically used by one function,
    // we just compute a global budget across all functions in the module.
    uint64_t existingSharedBytes = 0;
    moduleOp->walk([&](LLVM::GlobalOp global) {
      if (global.getSymName().starts_with("__shared_memory__") &&
          global.getAddrSpace() == kSharedAS) {
        existingSharedBytes += getGlobalSizeBytes(global);
      }
    });

    // Step 3: sort candidates by size (smallest first) for greedy packing.
    llvm::sort(allCandidates, [](LLVM::GlobalOp a, LLVM::GlobalOp b) {
      return getGlobalSizeBytes(a) < getGlobalSizeBytes(b);
    });

    // Step 4: greedily promote globals that fit in the shared budget.
    uint64_t usedSharedBytes = existingSharedBytes;
    SmallVector<LLVM::GlobalOp> toPromote;
    for (auto global : allCandidates) {
      uint64_t size = getGlobalSizeBytes(global);
      if (usedSharedBytes + size <= kMaxSharedBytes) {
        toPromote.push_back(global);
        usedSharedBytes += size;
      }
      // else: leave as .global (cross-block shared, but at least no crash)
    }

    // Step 5: promote selected globals to shared memory.
    auto ptrShared = LLVM::LLVMPointerType::get(&getContext(), kSharedAS);
    auto ptrGeneric = LLVM::LLVMPointerType::get(&getContext(), 0);

    for (auto global : toPromote) {
      global.setAddrSpace(kSharedAS);

      StringRef symName = global.getSymName();
      auto *symbolTableOp = SymbolTable::getNearestSymbolTable(global);
      if (!symbolTableOp)
        continue;

      symbolTableOp->walk([&](LLVM::AddressOfOp addressOf) {
        if (addressOf.getGlobalName() != symName)
          return;

        addressOf.getResult().setType(ptrShared);

        OpBuilder builder(addressOf);
        builder.setInsertionPointAfter(addressOf);
        auto cast = builder.create<LLVM::AddrSpaceCastOp>(
            addressOf.getLoc(), ptrGeneric, addressOf.getResult());

        addressOf.getResult().replaceAllUsesExcept(cast.getResult(), cast);
      });
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-promote-globals-to-shared";
  }
  StringRef getDescription() const override {
    return "Promotes __global_memory__ LLVM globals from device global (AS 0) "
           "to shared memory (AS 3) for per-block isolation";
  }
};

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

std::unique_ptr<Pass> createNovaGPUPromoteGlobalsToSharedPass() {
  return std::make_unique<NovaGPUPromoteGlobalsToSharedPass>();
}

void registerNovaConvertSharedMemAllocsPass() {
  PassRegistration<NovaConvertSharedMemAllocsPass>();
}

void registerNovaGPULowerMemorySpacePass() {
  PassRegistration<NovaGPULowerMemorySpacePass>();
}

void registerNovaGPUPromoteGlobalsToSharedPass() {
  PassRegistration<NovaGPUPromoteGlobalsToSharedPass>();
}

} // namespace mlir::nova
