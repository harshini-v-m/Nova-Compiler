//===- NovaGPUReduceBankConflicts.cpp - Pad shared mem to cut bank conflicts ===//
//
// For every memref.alloc with #gpu.address_space<workgroup> and rank >= 2,
// pads the innermost dimension by (16 / elemBytes) elements (= always 16 bytes).
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
//   NOTE: We deliberately use 16 bytes rather than IREE's 8-byte (64-bit)
//   default.  nvgpu.ldmatrix requires the shared-memory address of each
//   matrix-fragment row to be 16-byte aligned.  With 8-byte padding, a
//   128-f32 tile goes from stride 512 B to 520 B (not 16-byte aligned),
//   causing CUDA_ERROR_MISALIGNED_ADDRESS at runtime.  16-byte padding
//   always yields a 16-byte-aligned stride (e.g., 512 → 528 = 33×16 B).
//
// Example (f32, 128-column tile):
//   Before: memref<128x128xf32, workgroup>  → row stride = 512 bytes = 4×128
//           → 32-way bank conflict on col-0 access pattern
//   After:  %padded = memref.alloc() : memref<128x132xf32, workgroup>
//           %view   = memref.subview %padded[0,0][128,128][1,1]
//           → row stride = 528 bytes (not a multiple of 128, 16-byte aligned)
//           → bank conflicts eliminated; ldmatrix 16-byte alignment satisfied
//
// Pre-flight checks (mirrors IREE's canReplaceMemrefUsesAndPropagateType):
//   1. ALL dimensions must be static (hasStaticShape) — dynamic outer dims
//      make the budget computation incorrect.
//   2. No CollapseShapeOp, ExpandShapeOp, or CastOp reachable through the
//      view-like user chain — those require either full type re-inference
//      (ExpandShape) or explicit re-creation (Cast), and CollapseShape is
//      incompatible with an added innermost-dim stride.
//   3. Global budget: total current + total extra bytes across ALL allocs
//      must not exceed maxSRAM before any alloc is padded.
//   4. Any dynamic-shape alloc causes the entire pass to abort (makes the
//      global budget check impossible).
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
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-reduce-bank-conflicts"

using namespace mlir;

namespace mlir::nova {

// Walk the transitive view-like user chain of 'val'.
// Returns true if any CollapseShapeOp, ExpandShapeOp, or CastOp is reachable.
// - CollapseShapeOp: the padded innermost-dim stride is not collapsible.
// - ExpandShapeOp: result type requires ExpandShapeOp::computeExpandedType
//                  re-inference which we don't implement here.
// - CastOp: has an explicit result type that would need manual reconstruction.
// Any of these means we cannot safely propagate the stride change.
static bool hasUnsafeViewUser(Value val) {
  SmallVector<Value> worklist = {val};
  SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::CastOp>(user))
        return true;
      if (isa<ViewLikeOpInterface>(user)) {
        for (Value result : user->getResults())
          worklist.push_back(result);
      }
    }
  }
  return false;
}

// Get element bitwidth via DataLayout, which correctly handles index type and
// any type with a data-layout entry (unlike getIntOrFloatBitWidth which
// asserts on non-integer/float types).
static unsigned getElemBitWidth(Type elemType, const DataLayout &dl) {
  if (elemType.isIntOrFloat())
    return elemType.getIntOrFloatBitWidth();
  return static_cast<unsigned>(dl.getTypeSizeInBits(elemType));
}

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
    DataLayout dl = DataLayout::closest(funcOp);

    int64_t maxSRAM = 48 * 1024;
    if (!archName.empty()) {
      NVIDIATargetInfo info = getNVIDIATargetInfo(archName);
      if (info.isValid())
        maxSRAM = static_cast<int64_t>(info.maxWorkgroupMemBytes);
    }

    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    // Collect candidate allocs. Require hasStaticShape() across ALL dims —
    // checking only the innermost dim (as before) would miscalculate paddedBytes
    // when any outer dim is dynamic (treated as 0 or 1 in the product).
    SmallVector<memref::AllocOp> wgAllocs;
    bool anyDynamic = false;
    funcOp.walk([&](memref::AllocOp alloc) {
      auto memrefType = alloc.getType();
      if (memrefType.getMemorySpace() != workgroupSpace)
        return;
      if (memrefType.getRank() < 2)
        return;
      if (alloc->hasAttr("nova.swizzled")) {
        LLVM_DEBUG(llvm::dbgs() << "[nova-reduce-bank-conflicts] skipping "
                                   "swizzled alloc\n");
        return;
      }
      if (!memrefType.hasStaticShape()) {
        // A dynamic alloc makes the global budget check impossible.
        anyDynamic = true;
        return;
      }
      wgAllocs.push_back(alloc);
    });

    // If any workgroup alloc has a dynamic shape, budget computation is
    // impossible — skip all padding for this function (IREE's behaviour).
    if (anyDynamic)
      return;
    if (wgAllocs.empty())
      return;

    // Global budget check: compute current total and projected extra bytes
    // across ALL candidate allocs before modifying any of them.
    // Checking per-alloc (as before) allows the cumulative total to exceed
    // the limit even when each individual padded alloc passes the check.
    int64_t currentTotal = 0;
    int64_t totalExtra = 0;
    for (memref::AllocOp alloc : wgAllocs) {
      auto memrefType = alloc.getType();
      Type elemType = memrefType.getElementType();
      unsigned elemBits = getElemBitWidth(elemType, dl);
      if (elemBits == 0)
        continue;
      unsigned elemBytes = elemBits / 8;

      int64_t nElems = 1;
      for (int64_t d : memrefType.getShape())
        nElems *= d;
      currentTotal += nElems * static_cast<int64_t>(elemBytes);

      // Extra bytes = (product of all dims except innermost) * paddingElems * elemBytes.
      int64_t paddingElems =
          std::max<int64_t>(1, 16 / static_cast<int64_t>(elemBytes));
      int64_t outerProduct = nElems / memrefType.getShape().back();
      totalExtra +=
          outerProduct * paddingElems * static_cast<int64_t>(elemBytes);
    }

    if (currentTotal + totalExtra > maxSRAM) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-reduce-bank-conflicts] skipping all: "
                 << currentTotal << " current + " << totalExtra
                 << " extra = " << (currentTotal + totalExtra)
                 << " exceeds SRAM budget " << maxSRAM << "\n");
      return;
    }

    OpBuilder builder(ctx);

    for (memref::AllocOp alloc : wgAllocs) {
      auto memrefType = alloc.getType();
      ArrayRef<int64_t> shape = memrefType.getShape();
      Type elemType = memrefType.getElementType();

      unsigned elemBits = getElemBitWidth(elemType, dl);
      if (elemBits == 0)
        continue;
      unsigned elemBytes = elemBits / 8;

      // Pre-flight: skip if any user in the view chain cannot safely receive
      // the updated stride (CollapseShapeOp, ExpandShapeOp, CastOp).
      if (hasUnsafeViewUser(alloc.getResult())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-reduce-bank-conflicts] skipping alloc: "
                      "unsafe view user (collapse/expand/cast)\n");
        continue;
      }

      int64_t paddingElems =
          std::max<int64_t>(1, 16 / static_cast<int64_t>(elemBytes));
      int64_t newInnerDim = shape.back() + paddingElems;

      SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
      paddedShape.back() = newInnerDim;

      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-reduce-bank-conflicts] padding innermost dim "
                 << shape.back() << " → " << newInnerDim << " ("
                 << paddingElems << " elems = " << (paddingElems * elemBytes)
                 << " bytes)\n");

      auto paddedMemrefType =
          MemRefType::get(paddedShape, elemType,
                          /*layout=*/MemRefLayoutAttrInterface{},
                          workgroupSpace);

      builder.setInsertionPoint(alloc);
      Location loc = alloc.getLoc();

      auto paddedAlloc = memref::AllocOp::create(
          builder, loc, paddedMemrefType, alloc.getDynamicSizes(),
          alloc.getAlignmentAttr());

      SmallVector<OpFoldResult> offsets(shape.size(), builder.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      for (int64_t d : shape)
        sizes.push_back(builder.getIndexAttr(d));
      SmallVector<OpFoldResult> strides(shape.size(), builder.getIndexAttr(1));

      auto subview = memref::SubViewOp::create(builder, loc,
                                               paddedAlloc.getResult(),
                                               offsets, sizes, strides);

      // Redirect deallocs to the padded base alloc. Deallocing a subview is
      // UB (it is not an allocation root); redirect before replaceAllUsesWith
      // so the dealloc keeps pointing at a valid allocation.
      SmallVector<memref::DeallocOp> deallocsToFix;
      for (Operation *user :
           llvm::make_early_inc_range(alloc.getResult().getUsers())) {
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocsToFix.push_back(dealloc);
      }
      for (memref::DeallocOp dealloc : deallocsToFix)
        dealloc.getMemrefMutable().assign(paddedAlloc.getResult());

      alloc.getResult().replaceAllUsesWith(subview.getResult());
      alloc.erase();

      // -----------------------------------------------------------------------
      // Propagate the updated stride through downstream memref.subview chains.
      //
      // replaceAllUsesWith rewired operands, but pre-existing SubViewOps that
      // sliced the original alloc carry stale result types: the outer stride
      // was inferred from the old unpadded stride (e.g., 128) whereas the
      // source now carries stride 132.  The verifier catches this as a layout
      // mismatch.
      //
      // Fix: walk the use-def chain from the new subview and recompute each
      // downstream SubViewOp's result type.
      //
      // CollapseShapeOp / ExpandShapeOp / CastOp users are already ruled out
      // by the hasUnsafeViewUser pre-flight above, so the only view-like ops
      // we encounter here are SubViewOps.
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

          // Strip strides for dropped dims on rank-reducing subviews.
          // The multi-buffer pass produces rank-reducing subviews of the form
          //   subview %ring[%i, 0, 0][1, M, K][1,1,1] : memref<N×M×K> to
          //   memref<M×K, strided<[K, 1], offset: ?>>
          // where the leading unit dim is dropped. Without this filter we pass
          // 3 strides to a rank-2 MemRefType::get and crash the verifier.
          llvm::SmallBitVector droppedDims = sv.getDroppedDims();
          SmallVector<int64_t> newStrides;
          newStrides.reserve(allStrides.size() - droppedDims.count());
          for (size_t i = 0; i < allStrides.size(); ++i) {
            if (!droppedDims.test(i))
              newStrides.push_back(allStrides[i]);
          }

          auto oldResult = sv.getType();
          if (static_cast<int64_t>(newStrides.size()) != oldResult.getRank())
            continue;

          auto newResultType = MemRefType::get(
              oldResult.getShape(), oldResult.getElementType(),
              StridedLayoutAttr::get(ctx, newOffset, newStrides),
              oldResult.getMemorySpace());

          sv.getResult().setType(newResultType);
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
