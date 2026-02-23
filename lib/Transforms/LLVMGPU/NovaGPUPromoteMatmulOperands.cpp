// Nova GPU Promote Matmul Operands Pass
//
// Promotes A and B operands of linalg.matmul / linalg.batch_matmul inside
// scf.forall (workgroup tiling) to shared memory using a two-stage copy pattern
// that matches IREE's GPUPromoteMatmulOperands.cpp:
//
//  Stage 1 — fill shared memory:
//    %alloc  = bufferization.alloc_tensor {memory_space = #gpu.address_space<workgroup>}
//    %shared = linalg.copy(%operand -> %alloc)
//
//  Fence — prevent Stage 1 from fusing into Stage 2's loop:
//    %fence  = nova.fusion_barrier %shared
//
//  Stage 2 — per-thread promoted local copy (tiling/scheduling copy):
//    %empty  = tensor.empty(sizes)
//    %local  = linalg.copy(%fence -> %empty)
//
//  The original operand use is then replaced with %local.
//
// The nova.fusion_barrier op is a pure identity on the tensor that acts as an
// opaque fence: the fusion analysis cannot see through it, so Stage 1 and
// Stage 2 always end up in separate loops (the K-loop iteration only reloads
// from shared memory, not from global memory again).
//
// The nova.fusion_barrier is erased at the start of bufferization once all
// tiling and fusion decisions have been made.
//
// Directly mirrors IREE: GPUPromoteMatmulOperands.cpp (promoteOperand / promoteResult)

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
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::nova {

// Returns true if the given scf.forall has gpu.block_id mapping attributes
// (i.e., it is a workgroup-level forall, not a thread-level one).
static bool isWorkgroupForall(scf::ForallOp forallOp) {
  auto mappingAttr = forallOp.getMappingAttr();
  if (!mappingAttr)
    return false;
  return llvm::any_of(mappingAttr.getValue(), [](Attribute attr) {
    return isa<gpu::GPUBlockMappingAttr>(attr);
  });
}

// Builds a per-thread promoted copy of `v`:
//   %empty  = tensor.empty(dynamic_sizes) : same type as v
//   %result = linalg.copy(v -> %empty)
// Returns the result of the copy.
// Mirrors IREE's promoteValue().
static Value buildPerThreadCopy(OpBuilder &builder, Location loc, Value v) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);

  // Collect only the dynamic sizes for tensor.empty's operand list.
  SmallVector<Value> dynSizes;
  for (auto ofr : mixedSizes)
    if (auto val = dyn_cast<Value>(ofr))
      dynSizes.push_back(val);

  Value empty = tensor::EmptyOp::create(builder, loc, mixedSizes,
                                        tensorType.getElementType());
  auto copy = linalg::CopyOp::create(builder, loc, v, empty);
  return copy.getResult(0);
}

// Performs the full two-stage promotion for a single input operand at `index`
// of `linalgOp`:
//
//  1. Allocate in workgroup shared memory.
//  2. Copy global → shared  (linalg.copy).
//  3. Insert nova.fusion_barrier to block Stage-1 from fusing into Stage-2.
//  4. Make a per-thread copy of the fenced value (linalg.copy).
//  5. Replace the original operand use with the per-thread copy.
//
// Mirrors IREE's promoteOperand() + inline promoteResult() logic.
static void promoteOperandToShared(OpBuilder &builder,
                                   linalg::LinalgOp linalgOp,
                                   unsigned inputIdx) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(linalgOp);

  Location loc = linalgOp.getLoc();
  Value operand = linalgOp.getDpsInputOperand(inputIdx)->get();
  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType)
    return;

  // -----------------------------------------------------------------------
  // Stage 1: allocate a tensor in workgroup shared memory and copy into it.
  // -----------------------------------------------------------------------
  // Build dynamic sizes list.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, loc, operand);
  SmallVector<Value> dynSizes;
  for (auto ofr : mixedSizes)
    if (auto val = dyn_cast<Value>(ofr))
      dynSizes.push_back(val);

  // bufferization.alloc_tensor with workgroup address space.
  Attribute workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocOp =
      bufferization::AllocTensorOp::create(builder, loc, tensorType, dynSizes);
  allocOp.setMemorySpaceAttr(workgroupSpace);

  // Stage-1 copy: global operand → workgroup alloc.
  auto stage1Copy =
      linalg::CopyOp::create(builder, loc, operand, allocOp.getResult());
  Value stage1Result = stage1Copy.getResult(0);

  // -----------------------------------------------------------------------
  // Fusion barrier: prevents Stage 1 from being fused into Stage 2's loop.
  // Mirrors: IREE::Codegen::FusionBarrierOp::create(rewriter, loc, ...)
  // -----------------------------------------------------------------------
  Value fenced =
      FusionBarrierOp::create(builder, loc, stage1Result.getType(), stage1Result)
          .getResult();

  // -----------------------------------------------------------------------
  // Stage 2: per-thread copy from shared memory.
  // The fusion analysis sees %fenced (not %stage1Result) as a producer, so it
  // will never fuse Stage 1 into Stage 2's tiled loop.
  // -----------------------------------------------------------------------
  Value promoted = buildPerThreadCopy(builder, loc, fenced);

  // Replace this operand of the linalg op with the promoted per-thread copy.
  linalgOp->setOperand(
      linalgOp.getDpsInputOperand(inputIdx)->getOperandNumber(), promoted);
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

    // Walk all scf.forall ops that are workgroup-level.
    funcOp.walk([&](scf::ForallOp forallOp) {
      if (!isWorkgroupForall(forallOp))
        return WalkResult::advance();

      // Collect contraction ops first to avoid iterator invalidation.
      SmallVector<linalg::LinalgOp> contractionOps;
      forallOp.walk([&](linalg::LinalgOp op) {
        if (linalg::isaContractionOpInterface(op))
          contractionOps.push_back(op);
      });

      for (linalg::LinalgOp linalgOp : contractionOps) {
        // Determine which operand indices to promote.
        // Prefer the list from the LoweringConfig attribute (set by the
        // strategy pass).  Fall back to {0, 1} (A and B) when absent.
        SmallVector<int64_t> toPromote;
        if (DictionaryAttr config = getLoweringConfig(linalgOp.getOperation())) {
          auto maybeList = getPromotedOperandList(config);
          if (maybeList && !maybeList->empty()) {
            toPromote = *maybeList;
          }
        }
        if (toPromote.empty()) {
          // Hardcoded fallback: promote A (0) and B (1) only.
          unsigned numInputs = linalgOp.getNumDpsInputs();
          for (unsigned i = 0; i < std::min(numInputs, 2u); ++i)
            toPromote.push_back(static_cast<int64_t>(i));
        }

        for (int64_t idx : toPromote) {
          unsigned i = static_cast<unsigned>(idx);
          if (i >= linalgOp.getNumDpsInputs())
            continue;
          Value operand = linalgOp.getDpsInputOperand(i)->get();
          if (isa<RankedTensorType>(operand.getType()))
            promoteOperandToShared(builder, linalgOp, i);
        }
      }

      return WalkResult::advance();
    });
  }

  StringRef getArgument() const override {
    return "nova-gpu-promote-matmul-operands";
  }
  StringRef getDescription() const override {
    return "Promotes matmul A/B operands to GPU shared memory using "
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
