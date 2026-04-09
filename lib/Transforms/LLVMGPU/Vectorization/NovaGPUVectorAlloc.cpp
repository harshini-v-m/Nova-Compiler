//===- NovaGPUVectorAlloc.cpp - Stage operands through shared memory ------===//
//
// Nova equivalent of IREE's GPUVectorAllocPass.
//
// IREE finds shared-memory conflicts by:
//   1. Running layout analysis (propagateVectorLayoutInfo)
//   2. Walking to_layout ops and checking needsSharedMemoryForConversion
//   3. Materializing: barrier → alloc_tensor → transfer_write →
//                     value_barrier → transfer_read → rewire operand
//
// Nova replaces steps 1+2 with reading the pre-computed shared_mem flag
// from nova.layout_* DictionaryAttrs (set by NovaGPUConfigureTensorLayouts).
// Step 3 is identical to IREE: vector-level staging via transfer_write/read.
//
// For each vector.contract operand i with shared_mem = true:
//
//   (A) gpu.barrier at the START of the op's immediate block
//       — same conservative "HACK" placement as IREE, guards against the
//         previous loop iteration's shared-memory readers.
//
//   (B) bufferization.alloc_tensor() {memory_space = workgroup}
//       — empty alloc, same as IREE's allocateTensorForVector.
//
//   (C) vector.transfer_write %vec, %alloc[0, 0, ...]
//       — write the vector into shared memory, same as IREE.
//
//   (D) nova.value_barrier %written
//       — post-write sync, equivalent to IREE's iree_gpu.value_barrier.
//
//   (E) vector.transfer_read %barriered[0, 0, ...]
//       — read the vector back from shared memory in the new layout.
//
//   (F) Replace the contract operand with the read-back vector.
//       nova.layout_i's shared_mem flag is cleared to false.
//
// Downstream pipeline:
//   ComprehensiveBufferize       → lowers alloc_tensor to __shared__ alloca
//   NovaGPUInsertWorkgroupBarriers → replaces nova.value_barrier with
//                                    gpu.barrier after bufferization
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::nova {

namespace {

//===----------------------------------------------------------------------===//
// §1  Attribute helpers
//     (same logic as before — read/clear the shared_mem field)
//===----------------------------------------------------------------------===//

/// Returns true when |dict| has shared_mem = true.
static bool getSharedMem(DictionaryAttr dict) {
  if (!dict)
    return false;
  auto attr = dict.getAs<BoolAttr>("shared_mem");
  return attr && attr.getValue();
}

/// Returns a copy of |dict| with the shared_mem field forced to false.
/// Preserves alphabetical field order — required by MLIR's DictionaryAttr.
static DictionaryAttr clearSharedMem(MLIRContext *ctx, DictionaryAttr dict) {
  SmallVector<NamedAttribute> fields(dict.begin(), dict.end());
  for (auto &f : fields) {
    if (f.getName().getValue() == "shared_mem") {
      f = NamedAttribute(f.getName(), BoolAttr::get(ctx, false));
      break;
    }
  }
  return DictionaryAttr::get(ctx, fields);
}

//===----------------------------------------------------------------------===//
// §2  Vector-level staging helpers
//     Direct port of IREE's allocateTensorForVector + readVectorFromTensor.
//     These operate on vectors (not tensors) because by the time this pass
//     runs, NovaGPUGenericVectorization has already converted linalg ops to
//     vector.contract with vector operands.
//===----------------------------------------------------------------------===//

/// Allocate a workgroup-memory tensor and write |vector| into it.
/// Returns the tensor value AFTER the transfer_write (SSA updated tensor).
///
/// Mirrors IREE's allocateTensorForVector exactly:
///   alloc_tensor (empty, workgroup) → transfer_write → return written tensor
///
/// Returns failure() if the vector type is scalable (can't statically
/// allocate shared memory for a size unknown at compile time).
static FailureOr<Value> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector) {
  // Cast to VectorType — callers guarantee this is a vector operand.
  auto vectorType = cast<VectorType>(vector.getType());

  // Scalable vectors (SVE-style [N] dims) cannot be statically allocated.
  // Same check as IREE's allocateTensorForVector.
  if (vectorType.isScalable())
    return failure();

  // Build the workgroup address-space attribute.
  Attribute smemSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  // Build the tensor type: same shape + element type as the vector,
  // tagged with the workgroup address space.
  // Vectors are always statically shaped at this point in the pipeline.
  RankedTensorType tensorType = RankedTensorType::get(
      vectorType.getShape(), vectorType.getElementType(), smemSpace);

  // ── Step B: empty alloc_tensor ────────────────────────────────────────
  // Allocate uninitialised. No copy= operand — the transfer_write below
  // fills it. This matches IREE exactly (vs. our old copy= approach which
  // was tensor-level and only worked before vectorization).
  auto allocOp = bufferization::AllocTensorOp::create(
      b, loc, tensorType,
      /*dynamicSizes=*/ValueRange{},
      /*copy=*/Value()); // empty — no copy operand
  allocOp.setMemorySpaceAttr(smemSpace);

  // ── Step C: transfer_write the vector into the alloc ─────────────────
  // All indices are zero: we always write/read the whole vector at [0,0,...].
  // inBounds=true on all dims: no out-of-bounds masking needed.
  //
  // NOTE: do NOT use arith::ConstantIndexOp::create(b, loc, 0).
  // In MLIR 21.x that static create() overload does not exist; C++ silently
  // resolves 0 as a null TypedAttr and calls ConstantOp::create(b,loc,null),
  // producing arith.constant {} : ()->??? and a fatal type-inference crash.
  // b.create<> dispatches through ConstantIndexOp::build(builder,state,int64_t)
  // which is the only safe path.
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);

  Value written =
      vector::TransferWriteOp::create(b, loc, vector, allocOp, indices,
                                      inBounds)
          .getResult();
  return written;
}

/// Read a vector of |vectorType| back from |tensor| at index [0, 0, ...].
/// Mirrors IREE's readVectorFromTensor exactly.
static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  // Same fix as allocateTensorForVector: use b.create<> not the static create().
  Value c0 = b.create<arith::ConstantIndexOp>(tensor.getLoc(), 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);

  // padding=nullopt: no padding value needed because inBounds is all-true.
  return vector::TransferReadOp::create(b, tensor.getLoc(), vectorType, tensor,
                                        indices, /*padding=*/std::nullopt,
                                        inBounds)
      .getResult();
}

//===----------------------------------------------------------------------===//
// §3  Per-op materialization
//     Direct port of IREE's materializeSharedMemoryConversions, adapted to
//     read shared_mem from nova.layout_* attrs instead of ToLayoutOp attrs.
//===----------------------------------------------------------------------===//

/// For one vector.contract op, stage every operand whose nova.layout_i has
/// shared_mem=true through workgroup memory.
///
/// Emits for each such operand:
///   (A) gpu.barrier  at block start   (pre-write sync — IREE HACK strategy)
///   (B) alloc_tensor (empty, workgroup)
///   (C) transfer_write vector → alloc
///   (D) nova.value_barrier alloc result  (post-write sync)
///   (E) transfer_read  → new vector
///   (F) replace contract operand + clear shared_mem flag
static LogicalResult materializeForContractOp(MLIRContext *ctx,
                                              vector::ContractionOp contractOp,
                                              llvm::SmallPtrSet<Block *, 4> &barrierBlocks) {
  Location loc = contractOp.getLoc();
  unsigned layoutCount =
      std::min(contractOp->getNumOperands(), 3u);

  // Check whether this op has ANY operand needing staging.
  bool anySharedMem = false;
  for (unsigned i = 0; i < layoutCount; i++) {
    auto attr = contractOp->getAttrOfType<DictionaryAttr>(
        "nova.layout_" + std::to_string(i));
    if (getSharedMem(attr)) {
      anySharedMem = true;
      break;
    }
  }
  if (!anySharedMem)
    return success(); // nothing to do

  // ── Step A: pre-write gpu.barrier at start of op's block ─────────────
  // Same conservative placement as IREE: set at the start of the block that
  // contains the op. This prevents threads from overwriting shared memory
  // while other threads are still reading it from the previous loop iteration.
  //
  // IREE's comment: "HACK: Until proper barrier placement is handled later
  // we have to synchronize explicitly in this pass."
  Block *block = contractOp->getBlock();
  if (barrierBlocks.insert(block).second) {
    OpBuilder barrierBuilder(block, block->begin());
    gpu::BarrierOp::create(barrierBuilder, loc);
  }

  // Insertion point for alloc + write + barrier + read: just before the op.
  OpBuilder b(contractOp);

  for (unsigned i = 0; i < layoutCount; i++) {
    auto attr = contractOp->getAttrOfType<DictionaryAttr>(
        "nova.layout_" + std::to_string(i));
    if (!getSharedMem(attr))
      continue;

    Value operand = contractOp->getOperand(i);

    // Operand must be a vector — guaranteed by this point in the pipeline.
    auto vecType = dyn_cast<VectorType>(operand.getType());
    if (!vecType) {
      contractOp->emitError(
          "nova-gpu-vector-alloc: operand ")
          << i << " is not a vector — cannot stage through shared memory";
      return failure();
    }

    // ── Steps B + C: alloc + transfer_write ──────────────────────────────
    auto written = allocateTensorForVector(b, loc, operand);
    if (failed(written)) {
      contractOp->emitError(
          "nova-gpu-vector-alloc: failed to allocate shared memory "
          "for operand ")
          << i << " (scalable vector type?)";
      return failure();
    }

    // ── Step D: post-write value_barrier ─────────────────────────────────
    // Equivalent to IREE's iree_gpu.value_barrier: all threads must finish
    // writing before any thread reads. Value-semantic so downstream passes
    // can reason about ordering without inspecting control flow.
    auto postBarrier =
        nova::ValueBarrierOp::create(b, loc, ValueRange{*written});
    Value syncedTensor = postBarrier.getResults()[0];

    // ── Step E: transfer_read back to vector ─────────────────────────────
    // Read the vector back from shared memory. Because all threads wrote to
    // a shared tensor, each thread now reads the data in the layout it needs
    // (the new distribution is handled by downstream passes that use the
    // thread/sg layout info from nova.layout_i).
    Value newVec = readVectorFromTensor(b, vecType, syncedTensor);

    // ── Step F: rewire operand + clear flag ──────────────────────────────
    contractOp->setOperand(i, newVec);
    contractOp->setAttr("nova.layout_" + std::to_string(i),
                        clearSharedMem(ctx, attr));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// §4  Pass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorAllocPass
    : public PassWrapper<NovaGPUVectorAllocPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorAllocPass)

  StringRef getArgument() const override { return "nova-gpu-vector-alloc"; }
  StringRef getDescription() const override {
    return "Stage vector.contract operands marked shared_mem=true in "
           "nova.layout_* attributes through workgroup memory, using "
           "vector.transfer_write/read pairs with barriers.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
                    vector::VectorDialect, nova::NovaDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ── Collect all vector.contract ops upfront ───────────────────────────
    // Same reason as IREE's collect-then-process pattern: modifying the IR
    // during a walk invalidates the walk's internal iterator.
    SmallVector<vector::ContractionOp> contractOps;
    funcOp.walk([&](vector::ContractionOp op) {
      // Only collect ops that have at least one nova.layout_* with
      // shared_mem=true. This avoids paying the per-op overhead for ops
      // that don't need staging (e.g. reduction-only contracts).
      unsigned layoutCount = std::min(op->getNumOperands(), 3u);
      for (unsigned i = 0; i < layoutCount; i++) {
        auto attr = op->getAttrOfType<DictionaryAttr>(
            "nova.layout_" + std::to_string(i));
        if (getSharedMem(attr)) {
          contractOps.push_back(op);
          return; // one flagged operand is enough to include the op
        }
      }
    });

    // Track blocks that already received a pre-write gpu.barrier so we
    // don't insert duplicates when multiple contracts share a block.
    llvm::SmallPtrSet<Block *, 4> barrierInsertedBlocks;

    for (vector::ContractionOp contractOp : contractOps) {
      if (failed(materializeForContractOp(ctx, contractOp,
                                         barrierInsertedBlocks))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUVectorAllocPass() {
  return std::make_unique<NovaGPUVectorAllocPass>();
}

void registerNovaGPUVectorAllocPass() {
  PassRegistration<NovaGPUVectorAllocPass>();
}

} // namespace mlir::nova