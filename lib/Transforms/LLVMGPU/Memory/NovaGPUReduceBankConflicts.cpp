//===- NovaGPUReduceBankConflicts.cpp - Pad shared mem to cut bank conflicts ===//
//
// For every memref.alloc with #gpu.address_space<workgroup> and rank >= 2,
// pads the innermost dimension by (64 / elementBitWidth) elements (= 8 bytes).
// This shifts the bank alignment of consecutive rows, eliminating the most
// common pattern of 32-way bank conflicts in 2D shared memory access.
//
// Background — GPU shared memory bank conflicts:
//   Ampere has 32 shared memory banks, each 4 bytes wide (128-byte bank span).
//   If N threads in a warp all access the same bank, the hardware serializes
//   those accesses into N sequential cycles.  The worst case is 32-way
//   serialization (32x slowdown on shared memory bandwidth).
//
//   For a 2D tile with N columns, consecutive rows start at byte offsets
//   0, N*elemBytes, 2*N*elemBytes, ...  When N*elemBytes is a multiple of
//   128, column 0 of every row maps to the same bank — a 32-way conflict for
//   any warp that reads column 0 of all 32 rows simultaneously.
//
// Padding amount: 16 bytes / elemBytes elements (= always 16 bytes):
//   f32 (4 B):   pad 4 elements per row  → stride += 16 bytes
//   f16/bf16 (2 B): pad 8 elements       → stride += 16 bytes
//   f64 (8 B):   pad 2 elements          → stride += 16 bytes
//
// Example (f32, 128-column tile):
//   Before: memref<128x128xf32, workgroup>  → row stride = 512 bytes = 4×128
//           → 32-way bank conflict on col-0 access pattern
//   After:  %padded = memref.alloc() : memref<128x132xf32, workgroup>
//           %view   = memref.subview %padded[0,0][128,128][1,1]
//           → row stride = 528 bytes (not a multiple of 128, 16-byte aligned)
//           → bank conflicts eliminated; ldmatrix 16-byte alignment satisfied
//
// Pre-flight check: the pass skips any allocation where the padded footprint
// would itself exceed maxWorkgroupMemBytes — GPUCheckResourceUsagePass will
// then flag the unpadded allocation as already too large.
//
// Mirrors IREE's GPUReduceSharedMemoryBankConflictsPass
//   (iree/compiler/Codegen/Common/GPU/GPUReduceSharedMemoryBankConflicts.cpp)
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-reduce-bank-conflicts"

using namespace mlir;

namespace mlir::nova {

struct NovaGPUReduceBankConflictsPass
    : public PassWrapper<NovaGPUReduceBankConflictsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUReduceBankConflictsPass)

  std::string archName;

  NovaGPUReduceBankConflictsPass() = default;
  explicit NovaGPUReduceBankConflictsPass(StringRef arch)
      : archName(arch.str()) {}
  NovaGPUReduceBankConflictsPass(const NovaGPUReduceBankConflictsPass &p)
      : PassWrapper(p), archName(p.archName) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    // Resolve the SRAM limit so the pre-flight check is target-accurate.
    int64_t maxSRAM = 48 * 1024;
    if (!archName.empty()) {
      NVIDIATargetInfo info = getNVIDIATargetInfo(archName);
      if (info.isValid())
        maxSRAM = static_cast<int64_t>(info.maxWorkgroupMemBytes);
    }

    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    // Collect all workgroup allocs first to avoid iterator invalidation while
    // we insert new ops and erase old ones.
    SmallVector<memref::AllocOp> wgAllocs;
    funcOp.walk([&](memref::AllocOp alloc) {
      auto memrefType = alloc.getType();
      // Only rank-2+ allocations with a static innermost dimension need
      // padding.  1D accesses don't have the cross-row conflict pattern.
      if (memrefType.getMemorySpace() != workgroupSpace)
        return;
      if (memrefType.getRank() < 2)
        return;
      if (ShapedType::isDynamic(memrefType.getShape().back()))
        return;
      if (alloc->hasAttr("nova.swizzled")) {
        llvm::errs() << "[nova-reduce-bank-conflicts] skipping swizzled alloc: " << alloc << "\n";
        return;
      }
      wgAllocs.push_back(alloc);
    });

    if (wgAllocs.empty())
      return;

    OpBuilder builder(ctx);

    for (memref::AllocOp alloc : wgAllocs) {
      auto memrefType = alloc.getType();
      ArrayRef<int64_t> shape = memrefType.getShape();
      Type elemType = memrefType.getElementType();

      // Compute padding such that the row stride stays a multiple of 16 bytes.
      //
      // Why 16 bytes?
      //   nvgpu.ldmatrix (Ampere mma.sync path) requires the shared memory
      //   address of each matrix fragment row to be 16-byte aligned.  If the
      //   row stride is not a multiple of 16 bytes the hardware raises
      //   CUDA_ERROR_MISALIGNED_ADDRESS.  With the previous fixed 8-byte
      //   padding (64 bits / elemBits), f32 tiles went from stride 256 B
      //   (16-byte aligned) to 264 B (not aligned) — exactly the crash seen.
      //
      //   paddingElems = 16 / elemBytes  →  paddingBytes = 16 (always).
      //   f32 (4 B):  pad 4 elements → stride += 16 B  (64→68, 256→272 B)
      //   f16 (2 B):  pad 8 elements → stride += 16 B  (128→136, 256→272 B)
      //   f64 (8 B):  pad 2 elements → stride += 16 B
      //   Minimum 1 element for exotic / wide types.
      //
      //   272 bytes = 34 × 8 B: not a multiple of 128 B → bank conflicts gone.
      //   272 bytes = 17 × 16 B: 16-byte aligned → ldmatrix safe.
      unsigned elemBits = elemType.getIntOrFloatBitWidth();
      unsigned elemBytes = elemBits / 8;
      // 16 / elemBytes, clamped to [1, …].
      int64_t paddingElems = std::max<int64_t>(1, 16 / static_cast<int64_t>(elemBytes));
      int64_t newInnerDim = shape.back() + paddingElems;

      // Build the padded shape: same rank, only innermost dim grows.
      SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
      paddedShape.back() = newInnerDim;

      // Pre-flight: verify the padded allocation still fits in SRAM.
      // If it doesn't fit, skip — GPUCheckResourceUsagePass will flag the
      // underlying size problem.
      int64_t paddedBytes = 1;
      for (int64_t d : paddedShape)
        paddedBytes *= d;
      paddedBytes *= static_cast<int64_t>(elemBytes);
      if (paddedBytes > maxSRAM) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-reduce-bank-conflicts] skipping alloc at "
                   << alloc.getLoc()
                   << ": padded size " << paddedBytes
                   << " exceeds SRAM budget " << maxSRAM << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-reduce-bank-conflicts] padding innermost dim "
                 << shape.back() << " → " << newInnerDim
                 << " (" << paddingElems << " elems = "
                 << (paddingElems * elemBytes) << " bytes)\n");

      auto paddedMemrefType =
          MemRefType::get(paddedShape, elemType,
                          /*layout=*/MemRefLayoutAttrInterface{},
                          workgroupSpace);

      builder.setInsertionPoint(alloc);
      Location loc = alloc.getLoc();

      // Create the padded allocation, preserving any alignment attribute.
      auto paddedAlloc = memref::AllocOp::create(
          builder, loc, paddedMemrefType, alloc.getDynamicSizes(),
          alloc.getAlignmentAttr());

      // Build a subview presenting the original (unpadded) shape so all
      // downstream ops see the correct sizes without any code changes.
      // offsets: all zeros, sizes: original shape, strides: all ones.
      SmallVector<OpFoldResult> offsets(shape.size(), builder.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      for (int64_t d : shape)
        sizes.push_back(builder.getIndexAttr(d));
      SmallVector<OpFoldResult> strides(shape.size(), builder.getIndexAttr(1));

      auto subview = memref::SubViewOp::create(builder, loc,
                                               paddedAlloc.getResult(),
                                               offsets, sizes, strides);

      // Replace all uses of the original unpadded alloc with the subview,
      // EXCEPT dealloc ops — those must target the padded base alloc, not a
      // subview.  Deallocing a subview is undefined behaviour (it's not an
      // allocation root); we redirect them to paddedAlloc before the blanket
      // replaceAllUsesWith so the dealloc keeps pointing at valid memory.
      SmallVector<memref::DeallocOp> deallocsToFix;
      for (Operation *user : llvm::make_early_inc_range(alloc.getResult().getUsers())) {
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocsToFix.push_back(dealloc);
      }
      for (memref::DeallocOp dealloc : deallocsToFix)
        dealloc.getMemrefMutable().assign(paddedAlloc.getResult());

      alloc.getResult().replaceAllUsesWith(subview.getResult());
      alloc.erase();

      // -----------------------------------------------------------------------
      // Propagate the updated stride through downstream memref.subview ops.
      //
      // replaceAllUsesWith above rewired operands, but any pre-existing
      // SubViewOp that sliced the original alloc has a STALE result type:
      // its outer stride was inferred from the old unpadded stride (e.g., 128)
      // whereas the source now carries stride 130.  The verifier catches this
      // as a layout mismatch.
      //
      // Fix: walk the use-def chain from our new subview and recompute each
      // downstream SubViewOp's result type by multiplying the source strides
      // by the subview's own static strides.  The offset is always left as
      // dynamic (ShapedType::kDynamic) because subview offsets are runtime.
      // -----------------------------------------------------------------------
      SmallVector<Value> worklist = {subview.getResult()};
      SmallPtrSet<Operation *, 16> visited;
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (!visited.insert(user).second)
            continue;
          auto sv = dyn_cast<memref::SubViewOp>(user);
          if (!sv)
            continue;

          // Get the source memref's strides from its StridedLayoutAttr.
          // SubViewOp results always carry an explicit StridedLayoutAttr, so
          // this cast is safe for any subview chained from our padding subview.
          auto srcType = cast<MemRefType>(sv.getSource().getType());
          auto srcLayout = dyn_cast<StridedLayoutAttr>(srcType.getLayout());
          if (!srcLayout)
            continue;

          ArrayRef<int64_t> srcStrides = srcLayout.getStrides();
          SmallVector<int64_t> svStaticStrides =
              llvm::to_vector(sv.getStaticStrides());
          SmallVector<int64_t> svStaticOffsets =
              llvm::to_vector(sv.getStaticOffsets());

          // result_stride[i] = src_stride[i] * sv_static_stride[i]
          // Keep the full per-source-dim stride list; we'll filter out the
          // dropped dims below for rank-reducing subviews.
          SmallVector<int64_t> allStrides;
          allStrides.reserve(srcStrides.size());
          for (size_t i = 0; i < srcStrides.size(); ++i) {
            if (ShapedType::isDynamic(srcStrides[i]) ||
                ShapedType::isDynamic(svStaticStrides[i]))
              allStrides.push_back(ShapedType::kDynamic);
            else
              allStrides.push_back(srcStrides[i] * svStaticStrides[i]);
          }

          // result_base_offset = src_base_offset + Σ(sv_offset[i] * src_stride[i])
          // If any term is dynamic the whole offset is dynamic.
          int64_t newOffset = srcLayout.getOffset();
          for (size_t i = 0; i < srcStrides.size(); ++i) {
            if (ShapedType::isDynamic(newOffset) ||
                ShapedType::isDynamic(svStaticOffsets[i]) ||
                ShapedType::isDynamic(srcStrides[i])) {
              newOffset = ShapedType::kDynamic;
              break;
            }
            newOffset += svStaticOffsets[i] * srcStrides[i];
          }

          // Phase 5 fix: strip strides for dropped dims on rank-reducing
          // subviews.  The multi-buffer pass (Phase 1 of software pipelining)
          // produces rank-reducing subviews of the form
          //   memref.subview %ring[%i, 0, 0][1, M, K][1,1,1] :
          //     memref<N×M×K> to memref<M×K, strided<[K, 1], offset: ?>>
          // where the leading unit dim is dropped.  Without this filter we
          // would pass 3 strides to a rank-2 MemRefType::get and crash the
          // verifier.
          llvm::SmallBitVector droppedDims = sv.getDroppedDims();
          SmallVector<int64_t> newStrides;
          newStrides.reserve(allStrides.size() - droppedDims.count());
          for (size_t i = 0; i < allStrides.size(); ++i) {
            if (!droppedDims.test(i))
              newStrides.push_back(allStrides[i]);
          }

          auto oldResult = sv.getType();
          // Safety: if stride filtering disagrees with the result rank (e.g.,
          // pre-existing malformed subview), skip this op rather than crash.
          if (static_cast<int64_t>(newStrides.size()) != oldResult.getRank())
            continue;

          auto newResultType = MemRefType::get(
              oldResult.getShape(), oldResult.getElementType(),
              StridedLayoutAttr::get(ctx, newOffset, newStrides),
              oldResult.getMemorySpace());

          sv.getResult().setType(newResultType);
          // Continue propagating down any further subview chains.
          worklist.push_back(sv.getResult());
        }
      }
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-reduce-bank-conflicts";
  }
  StringRef getDescription() const override {
    return "Pads the innermost dimension of workgroup shared memory allocs "
           "by 16 bytes (4 f32 / 8 f16 elems) to shift row bank alignment "
           "and satisfy ldmatrix 16-byte alignment on Ampere+";
  }
};

std::unique_ptr<Pass> createNovaGPUReduceBankConflictsPass(StringRef arch) {
  return std::make_unique<NovaGPUReduceBankConflictsPass>(arch);
}

void registerNovaGPUReduceBankConflictsPass() {
  PassRegistration<NovaGPUReduceBankConflictsPass>();
}

} // namespace mlir::nova