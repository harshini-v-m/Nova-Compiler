//===- NovaGPUMaterializeDynamicSharedMem.cpp ---------------------------===//
//
// For each gpu.launch op in the function, replaces every static-shape
// workgroup `memref.alloc` with a `memref.view` over a single
// `gpu.dynamic_shared_memory` buffer and stamps the launch's
// `dynamicSharedMemorySize` operand with the cumulative byte size.
//
// Must run AFTER NovaGPUCreateAsyncCopies and NovaGPUPipelining: those passes
// pattern-match on plain workgroup `memref.alloc` operands and bail on the
// `memref.view`-rooted IR shape this pass produces. Keeping the alloc form
// alive until after async-copy/pipelining preserves the cp.async + pipelined
// K-loop fast path.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "nova-gpu-materialize-dynamic-smem"

using namespace mlir;

namespace mlir::nova {

namespace {

static bool hasWorkgroupAddressSpace(MemRefType t) {
  auto as = dyn_cast_if_present<gpu::AddressSpaceAttr>(t.getMemorySpace());
  return as && as.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

static void materializeDynamicSharedMemory(gpu::LaunchOp launch) {
  MLIRContext *ctx = launch.getContext();
  Block &body = launch.getBody().front();
  Location loc = launch.getLoc();

  SmallVector<memref::AllocOp> wgAllocs;
  body.walk([&](memref::AllocOp alloc) {
    auto mrTy = alloc.getType();
    if (hasWorkgroupAddressSpace(mrTy) && mrTy.hasStaticShape())
      wgAllocs.push_back(alloc);
  });
  if (wgAllocs.empty())
    return;

  SmallVector<int64_t> byteOffsets(wgAllocs.size(), 0);
  int64_t running = 0;
  for (auto [i, alloc] : llvm::enumerate(wgAllocs)) {
    MemRefType ty = alloc.getType();
    int64_t elemBits = ty.getElementTypeBitWidth();
    int64_t elems = ty.getNumElements();
    int64_t bytes = (elems * elemBits + 7) / 8;
    bytes = llvm::alignTo(bytes, (int64_t)16);
    byteOffsets[i] = running;
    running += bytes;
  }
  int64_t totalBytes = running;

  OpBuilder hostBuilder(launch);
  Value dynSmemSize = hostBuilder.create<arith::ConstantIntOp>(
      loc, /*value=*/totalBytes, /*bitwidth=*/32);
  launch.getDynamicSharedMemorySizeMutable().assign(dynSmemSize);

  OpBuilder inBody(ctx);
  inBody.setInsertionPointToStart(&body);
  auto workgroupAS = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto i8DynTy = MemRefType::get({ShapedType::kDynamic},
                                 inBody.getIntegerType(8),
                                 MemRefLayoutAttrInterface{}, workgroupAS);
  Value dynBase =
      inBody.create<gpu::DynamicSharedMemoryOp>(loc, i8DynTy).getResult();

  for (auto [i, alloc] : llvm::enumerate(wgAllocs)) {
    OpBuilder b(alloc);
    Value offset = b.create<arith::ConstantIndexOp>(loc, byteOffsets[i]);
    MemRefType viewTy = alloc.getType();
    auto view = b.create<memref::ViewOp>(loc, viewTy, dynBase, offset,
                                         /*sizes=*/ValueRange{});
    alloc.getResult().replaceAllUsesWith(view.getResult());
    alloc.erase();
  }

  SmallVector<memref::DeallocOp> staleDeallocs;
  body.walk([&](memref::DeallocOp d) {
    auto mrTy = dyn_cast<MemRefType>(d.getMemref().getType());
    if (mrTy && hasWorkgroupAddressSpace(mrTy))
      staleDeallocs.push_back(d);
  });
  for (memref::DeallocOp d : staleDeallocs)
    d.erase();

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << wgAllocs.size()
                          << " alloc(s), " << totalBytes << " bytes total\n");
}

struct NovaGPUMaterializeDynamicSharedMemPass
    : public PassWrapper<NovaGPUMaterializeDynamicSharedMemPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUMaterializeDynamicSharedMemPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<gpu::LaunchOp> launches;
    funcOp.walk([&](gpu::LaunchOp launch) { launches.push_back(launch); });
    for (gpu::LaunchOp launch : launches)
      materializeDynamicSharedMemory(launch);
  }

  StringRef getArgument() const override {
    return "nova-gpu-materialize-dynamic-smem";
  }
  StringRef getDescription() const override {
    return "Replace workgroup memref.alloc inside gpu.launch with "
           "memref.view over a single gpu.dynamic_shared_memory buffer "
           "and stamp the launch's dynamicSharedMemorySize.";
  }
};

} // namespace

std::unique_ptr<Pass> createNovaGPUMaterializeDynamicSharedMemPass() {
  return std::make_unique<NovaGPUMaterializeDynamicSharedMemPass>();
}

void registerNovaGPUMaterializeDynamicSharedMemPass() {
  PassRegistration<NovaGPUMaterializeDynamicSharedMemPass>();
}

} // namespace mlir::nova
