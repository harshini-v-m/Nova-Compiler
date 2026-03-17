//===- NovaGPUPromoteMatmulOperands.cpp - Promote operands to shared mem ===//
//
// Config-driven promotion of operands to GPU shared memory using a two-stage
// copy pattern. Mirrors IREE's GPUPromoteMatmulOperands.cpp.
//
// The pass walks ALL ops with a "promoted_operands" list in their
// lowering_config attribute, not just contractions. This handles matmuls,
// reductions, and elementwise ops uniformly.
//
// For each promoted operand index:
//   - index < numDpsInputs  → promote input operand (two-stage shared copy)
//   - index >= numDpsInputs → promote result (DPS init → shared copy)
//
// Two-stage promotion for input operands:
//
//   Stage 1 — Cooperative global→shared copy:
//     %alloc  = bufferization.alloc_tensor {memory_space = workgroup}
//     %shared = linalg.copy(%operand -> %alloc)
//
//   Fence — prevent Stage 1 from fusing into Stage 2's loop:
//     %fence  = nova.fusion_barrier %shared
//
//   Stage 2 — Per-thread promoted local copy:
//     %empty  = tensor.empty(sizes)
//     %local  = linalg.copy(%fence -> %empty)
//
//   The original operand use is replaced with %local.
//
// Optimizations (ported from IREE):
//   - Fill producers are skipped (no benefit from promoting constants)
//   - Existing linalg producers are left as-is (thread tiling handles them)
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

/// Full two-stage promotion for a single input operand.
///
///  Stage 1: Allocate in workgroup shared memory, cooperative copy global→shared.
///  Fence:   Insert nova.fusion_barrier to prevent Stage 1 fusing into Stage 2.
///  Stage 2: Per-thread copy from shared into private (register) tensor.
///           Replace the original operand use with the per-thread copy.
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

  // Optimization: skip fill producers (no benefit from shared memory).
  if (isFillProducer(operand))
    return;

  // Optimization: if the producer is already a linalg op (not a fill),
  // don't insert a copy — thread tiling will handle it naturally.
  // This mirrors IREE's promotionImpl() producer annotation logic.
  if (auto producer = dyn_cast_if_present<linalg::LinalgOp>(
          operand.getDefiningOp())) {
    if (!isa<linalg::FillOp>(producer.getOperation()))
      return;
  }

  // Stage 1: Allocate in workgroup shared memory + cooperative copy.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, loc, operand);
  SmallVector<Value> dynSizes;
  for (auto ofr : mixedSizes)
    if (auto val = dyn_cast<Value>(ofr))
      dynSizes.push_back(val);

  Attribute workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocOp =
      bufferization::AllocTensorOp::create(builder, loc, tensorType, dynSizes);
  allocOp.setMemorySpaceAttr(workgroupSpace);

  auto stage1Copy =
      linalg::CopyOp::create(builder, loc, operand, allocOp.getResult());
  Value stage1Result = stage1Copy.getResult(0);

  // Fence: Insert nova.fusion_barrier to prevent Stage 1 from fusing
  // into Stage 2's per-thread loop.
  Value fenced = FusionBarrierOp::create(builder, loc, stage1Result).getResult();

  // Stage 2: Per-thread copy from shared memory into private registers.
  Value promoted = buildPerThreadCopy(builder, loc, fenced);

  // Replace the operand with the promoted per-thread copy.
  op->setOperand(dpsOp.getDpsInputOperand(inputIdx)->getOperandNumber(),
                 promoted);
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
          // Input operand promotion: two-stage shared memory copy.
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
