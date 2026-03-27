//===- NovaGPUPromoteMatmulOperands.cpp - Promote operands to shared mem ===//
//
// Config-driven promotion of operands to GPU shared memory.
// Mirrors IREE's GPUPromoteMatmulOperands.cpp.
//
// The pass walks ALL ops with a "promoted_operands" list in their
// lowering_config attribute, not just contractions. This handles matmuls,
// reductions, and elementwise ops uniformly.
//
// For each promoted operand index:
//   - index < numDpsInputs  → promote input operand (shared copy)
//   - index >= numDpsInputs → promote result (DPS init → shared copy)
//
// Input promotion (two-phase IREE approach):
//   %empty = tensor.empty(sizes) : tensor<...>
//   %copy  = linalg.copy(%operand -> %empty)
//            {lowering_config = {thread = [1, 1, vectorSize], ...}}
//   The original operand use is replaced with %copy result.
//
//   NO memory space annotation at this stage. The copy has a thread-only
//   lowering_config (zero workgroup/reduction tiles, non-zero thread tiles):
//     - K-tiling (Step 4) fuses the copy as a producer into the K-loop,
//       shrinking it from [wgM × K] to [wgM × kStep].
//     - Thread tiling (Step 5) sees non-zero thread tiles and creates a
//       separate cooperative-loading scf.forall for the copy.
//     - InferMemorySpace (Step 8) detects the tensor.empty is used as
//       shared_outs of a thread-mapped forall → tags as workgroup memory.
//     - InsertWorkgroupBarriers (Step 11) adds gpu.barrier between the
//       cooperative store and the matmul read.
//
// Result promotion:
//   Uses alloc_tensor(workgroup) + copy + nova.fusion_barrier + per-thread
//   copy, so the result is written to shared and then consumed per-thread.
//
// Optimizations:
//   - Fill producers are skipped (no benefit from promoting constants)
//   - Contraction producers are skipped (they manage their own shared memory)
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::nova {

/// Builds a per-thread promoted copy of `v`:
///   %empty  = tensor.empty(sizes) : same type as v
///   %result = linalg.copy(v -> %empty)
/// Returns the result of the copy.
/// Mirrors IREE's promoteValue().
static Value buildPerThreadCopy(OpBuilder &builder, Location loc, Value v) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);
  Value empty = tensor::EmptyOp::create(builder, loc, mixedSizes,
                                        tensorType.getElementType());
  auto copy = linalg::CopyOp::create(builder, loc, v, empty);
  return copy.getResult(0);
}

/// Returns true if the operand's producer is a fill op (linalg.fill or a
/// generic that implements fill semantics). Fill producers should not be
/// promoted to shared memory — they are just constants.
/// Mirrors IREE's promotionImpl() fill-skip logic.
static bool isFillProducer(Value operand) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<linalg::FillOp>(defOp))
    return true;
  if (auto generic = dyn_cast<linalg::GenericOp>(defOp))
    return linalg::isaFillOpInterface(generic).has_value();
  return false;
}

/// Promote a single input operand to workgroup (shared) memory.
///
/// Inserts: alloc_tensor(workgroup) + linalg.copy(global → shared).
/// The operand is replaced with the copy result (in workgroup space).
///
/// NO fusion_barrier or per-thread copy is inserted for inputs.  The copy
/// is left as a plain producer so that:
///   - K-reduction tiling (Step 4) fuses it into the K-loop, giving each
///     iteration a [wgM × kStep]-sized shared tile instead of [wgM × K].
///   - Thread tiling (Step 5) distributes the copy across threads for
///     cooperative loading.
///   - InsertWorkgroupBarriers (Step 11) adds gpu.barrier between the
///     cooperative store and the matmul read after bufferization.
///
/// This matches IREE's input promotion strategy where the copy tiles
/// naturally with the consumer — no opaque barrier to block tiling.
static void promoteOperandToShared(OpBuilder &builder,
                                   Operation *op,
                                   unsigned inputIdx) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp || inputIdx >= dpsOp.getNumDpsInputs())
    return;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  Location loc = op->getLoc();
  Value operand = dpsOp.getDpsInputOperand(inputIdx)->get();
  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType)
    return;

  // Skip fill producers (no benefit from promoting constants to shared).
  if (isFillProducer(operand))
    return;

  // Skip contraction producers — they manage their own shared memory via
  // their own lowering_config and promotion. All other linalg producers
  // (broadcasts, LN normalize, elementwise) benefit from promotion because
  // K-tiling creates a loop and without promotion each K-iteration reads
  // from global memory independently. With promotion, cooperative loading
  // into shared memory enables reuse across threads.
  if (auto producer = dyn_cast_if_present<linalg::LinalgOp>(
          operand.getDefiningOp())) {
    if (linalg::isaContractionOpInterface(producer))
      return;
  }

  // Phase 1: tensor.empty() + linalg.copy with thread-only lowering_config.
  //
  // The copy has NO workgroup or reduction tiles, so:
  //   - K-tiling (Step 4) fuses it as a producer into the K-loop, naturally
  //     shrinking the copy to [wgM × kStep] or [kStep × wgN].
  //   - Thread tiling (Step 5) sees the non-zero thread tiles and creates a
  //     separate cooperative-loading scf.forall for the copy. This makes the
  //     tensor.empty() a shared_outs arg of the forall.
  //   - InferMemorySpace (Step 8) detects the alloc_tensor is used as
  //     shared_outs of a thread-mapped forall → tags as workgroup memory.
  //
  // This mirrors IREE's DerivedThreadConfigAttr approach.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, loc, operand);
  Value empty = tensor::EmptyOp::create(builder, loc, mixedSizes,
                                        tensorType.getElementType());
  auto copyOp = linalg::CopyOp::create(builder, loc, operand, empty);

  // Attach a derived-thread lowering_config: placeholder thread tiles [1,..,1]
  // plus a "derived_thread = true" marker and the contraction's target thread
  // count.  At thread-tiling time (Step 5), the tiling pass detects the marker
  // and recomputes tile sizes from the copy's actual (K-tiled) loop ranges so
  // that trip_count == targetThreads.  This guarantees the copy forall and
  // matmul forall have matching bounds, enabling FuseForalls (Step 6).
  //
  // Mirrors IREE's DerivedThreadConfigAttr pattern.
  unsigned numLoops = copyOp.getNumLoops();
  SmallVector<int64_t> threadTiles(numLoops, 1); // placeholder (non-zero)

  // Compute the contraction's thread count from its lowering_config.
  int64_t targetThreads = 0;
  DictionaryAttr parentConfig = getLoweringConfig(op);
  if (parentConfig) {
    auto wgTiles = getLoweringConfigTileSizes(parentConfig, kWorkgroupKey);
    auto thTiles = getLoweringConfigTileSizes(parentConfig, kThreadKey);
    if (wgTiles.size() == thTiles.size()) {
      targetThreads = 1;
      for (size_t i = 0; i < wgTiles.size(); ++i) {
        if (thTiles[i] > 0 && wgTiles[i] > 0)
          targetThreads *= (wgTiles[i] / thTiles[i]);
      }
    }
  }

  SmallVector<int64_t> zeros(numLoops, 0);
  // Build the base config with placeholder thread tiles.
  SmallVector<NamedAttribute> attrs;
  MLIRContext *ctx = builder.getContext();
  setLoweringConfigTileSizes(ctx, attrs, kWorkgroupKey, zeros);
  setLoweringConfigTileSizes(ctx, attrs, kReductionKey, zeros);
  setLoweringConfigTileSizes(ctx, attrs, kThreadKey, threadTiles);
  setLoweringConfigTileSizes(ctx, attrs, kSubgroupKey, zeros);
  setMmaKindRaw(ctx, attrs, 0);
  appendPromotedOperandsList(ctx, attrs, {});
  // Mark as derived-thread config with the target thread count.
  if (targetThreads > 0) {
    Builder b(ctx);
    attrs.emplace_back(StringAttr::get(ctx, kDerivedThreadKey),
                       b.getBoolAttr(true));
    attrs.emplace_back(StringAttr::get(ctx, kTargetThreadsKey),
                       b.getI64IntegerAttr(targetThreads));
  }
  setLoweringConfig(copyOp, DictionaryAttr::get(ctx, attrs));

  // Mark this copy as a promoted-input copy so InferMemorySpace can
  // unconditionally classify its destination as workgroup memory.
  // The marker survives K-tiling and Thread-tiling because MLIR clones
  // preserve all op attributes on tiled successors.
  copyOp->setAttr(StringAttr::get(ctx, kPromoteToWorkgroupAttr),
                  UnitAttr::get(ctx));

  // Replace the operand with the copy result.
  op->setOperand(dpsOp.getDpsInputOperand(inputIdx)->getOperandNumber(),
                 copyOp.getResult(0));
}

/// Result promotion: promote a DPS init result to shared memory.
/// When operand index >= numDpsInputs, the index refers to a result.
/// Mirrors IREE's promoteResult():
///   1. Alloc in workgroup shared memory
///   2. Copy result → shared
///   3. Insert fusion barrier
///   4. Per-thread copy from shared
///   5. Replace uses in consumers
static void promoteResultToShared(OpBuilder &builder,
                                  Operation *op,
                                  unsigned resultIdx) {
  if (resultIdx >= op->getNumResults())
    return;

  Value result = op->getResult(resultIdx);
  auto tensorType = dyn_cast<RankedTensorType>(result.getType());
  if (!tensorType)
    return;

  IRRewriter rewriter(builder);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(result);
  Location loc = op->getLoc();

  // Allocate in workgroup shared memory.
  SmallVector<Value> dynamicSizes;
  for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size))
      dynamicSizes.push_back(tensor::DimOp::create(rewriter, loc, result, idx));
  }
  Attribute addressSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto alloc = bufferization::AllocTensorOp::create(rewriter, loc, tensorType,
                                                    dynamicSizes);
  alloc.setMemorySpaceAttr(addressSpace);

  // Copy result → shared memory.
  auto copyToShared =
      linalg::CopyOp::create(rewriter, loc, result, alloc.getResult());
  Value replacement = copyToShared.getResult(0);

  // Insert fusion barrier.
  replacement =
      FusionBarrierOp::create(rewriter, loc, replacement).getResult();

  // Per-thread copy from shared.
  rewriter.setInsertionPointAfterValue(replacement);
  replacement = buildPerThreadCopy(rewriter, loc, replacement);

  // Replace uses of the original result in downstream consumers.
  result.replaceAllUsesExcept(replacement, copyToShared);
}

struct NovaGPUPromoteMatmulOperandsPass
    : public PassWrapper<NovaGPUPromoteMatmulOperandsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUPromoteMatmulOperandsPass)

  NovaGPUPromoteMatmulOperandsPass() = default;
  NovaGPUPromoteMatmulOperandsPass(const NovaGPUPromoteMatmulOperandsPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect, nova::NovaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Config-driven walk: promote operands of ANY op that has a
    // "promoted_operands" list in its lowering_config. This handles
    // contractions, elementwise, and reduction ops uniformly.
    // Mirrors IREE's GPUPromoteMatmulOperandsPass::runOnOperation().
    funcOp.walk([&](Operation *op) {
      DictionaryAttr config = getLoweringConfig(op);
      if (!config)
        return WalkResult::advance();

      std::optional<SmallVector<int64_t>> promotedOperands =
          getPromotedOperandList(config);
      if (!promotedOperands || promotedOperands->empty())
        return WalkResult::advance();

      auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
      if (!dpsOp)
        return WalkResult::advance();

      builder.setInsertionPoint(op);
      unsigned numInputs = dpsOp.getNumDpsInputs();

      for (int64_t idx : *promotedOperands) {
        unsigned i = static_cast<unsigned>(idx);
        if (i < numInputs) {
          // Input operand promotion: shared memory copy.
          promoteOperandToShared(builder, op, i);
        } else {
          // Result promotion: index beyond inputs refers to DPS init result.
          unsigned resultIdx = i - numInputs;
          promoteResultToShared(builder, op, resultIdx);
        }
      }

      return WalkResult::advance();
    });
  }

  StringRef getArgument() const override {
    return "nova-gpu-promote-matmul-operands";
  }
  StringRef getDescription() const override {
    return "Promotes operands to GPU shared memory using "
           "two-stage copy with nova.fusion_barrier";
  }
};

std::unique_ptr<Pass> createNovaGPUPromoteMatmulOperandsPass() {
  return std::make_unique<NovaGPUPromoteMatmulOperandsPass>();
}

void registerNovaGPUPromoteMatmulOperandsPass() {
  PassRegistration<NovaGPUPromoteMatmulOperandsPass>();
}

} // namespace mlir::nova
