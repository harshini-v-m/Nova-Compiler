//===- NovaGPUVectorAlloc.cpp - Stage operands through shared memory ------===//
//
// Nova equivalent of IREE's GPUVectorAllocPass.
//
// For each vector.contract operand i with shared_mem = true in nova.layout_i:
//
//   The contract lives inside a subgroup scf.forall (e.g. (1,4,1)) whose trip
//   counts equal wg_subgroup from the lowering_config.  Each subgroup owns a
//   per-subgroup slice of the workgroup tile.
//
//   OLD (buggy): alloc inside sg_forall body, sized for per-sg vector, written
//   at [0,0,0].  All 4 subgroups map to the same __shared__ address → race.
//
//   NEW (correct):
//     (A) gpu.barrier at the START of the sg_forall body
//         — guards previous K-iteration's readers from the upcoming write.
//     (B) bufferization.alloc_tensor OUTSIDE the sg_forall, sized for the
//         full workgroup tile: wgShape[d] = sgShape[d] * sg_counts[d].
//     (C) Inside the sg_forall body (just before contractOp):
//           compute per-subgroup write offset from induction vars:
//             offset[d] = iv[d] * sgShape[d]  if sg_strides[d] != 0
//                       = 0                   otherwise
//           tensor.extract_slice wgAlloc → per-sg slice
//           vector.transfer_write %vec into the slice at [0,0,...]
//     (D) nova.value_barrier on the written per-sg tensor (inside body)
//         — post-write sync before the read.
//     (E) vector.transfer_read from the synced tensor back to vector
//     (F) Replace contract operand with the new vector.
//         Clear shared_mem flag in nova.layout_i.
//
// After bufferization the wg-sized alloc_tensor becomes a single __shared__
// alloca.  Each subgroup writes to its non-overlapping slice (no race).
// nova.value_barrier → gpu.barrier is inserted by NovaGPUInsertWorkgroupBarriers.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
//===----------------------------------------------------------------------===//

static bool getSharedMem(DictionaryAttr dict) {
  if (!dict)
    return false;
  auto attr = dict.getAs<BoolAttr>("shared_mem");
  return attr && attr.getValue();
}

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

/// Read an integer array field from a nova.layout_* DictionaryAttr.
static SmallVector<int64_t> getIntArray(DictionaryAttr dict, StringRef field) {
  SmallVector<int64_t> result;
  if (!dict)
    return result;
  auto arr = dict.getAs<ArrayAttr>(field);
  if (!arr)
    return result;
  for (Attribute a : arr)
    if (auto ia = dyn_cast<IntegerAttr>(a))
      result.push_back(ia.getInt());
  return result;
}

//===----------------------------------------------------------------------===//
// §2  Subgroup forall detection
//
// The subgroup forall is the innermost scf.forall ancestor of the contractOp
// whose mapping is all gpu.thread attributes (as opposed to the workgroup
// forall which uses gpu.block attributes).
//===----------------------------------------------------------------------===//

static bool isThreadMappedForall(scf::ForallOp forall) {
  auto mapping = forall.getMappingAttr();
  if (!mapping || mapping.getValue().empty())
    return false;
  for (Attribute attr : mapping.getValue())
    if (!isa<gpu::GPUThreadMappingAttr>(attr))
      return false;
  return true;
}

/// Return the innermost thread-mapped scf.forall enclosing |op|, or nullptr.
static scf::ForallOp findEnclosingSubgroupForall(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forall = dyn_cast<scf::ForallOp>(parent))
      if (isThreadMappedForall(forall))
        return forall;
    parent = parent->getParentOp();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// §3  Per-operand staging
//===----------------------------------------------------------------------===//

/// Stage one vector.contract operand through workgroup-sized shared memory.
///
/// |operandIdx|      which operand (0=A/LHS, 1=B/RHS, 2=ACC)
/// |layoutAttr|      nova.layout_{operandIdx} DictionaryAttr
/// |sgForall|        enclosing subgroup scf.forall
/// |wgAllocBuilder|  OpBuilder positioned just before sgForall (outside)
/// |barrierBlocks|   tracks blocks that already have a pre-write gpu.barrier
static LogicalResult stageOperandThroughSmem(
    MLIRContext *ctx, vector::ContractionOp contractOp, unsigned operandIdx,
    DictionaryAttr layoutAttr, scf::ForallOp sgForall,
    OpBuilder &wgAllocBuilder,
    llvm::SmallPtrSet<Block *, 4> &barrierBlocks) {

  Location loc = contractOp.getLoc();
  Value operand = contractOp->getOperand(operandIdx);

  auto vecType = dyn_cast<VectorType>(operand.getType());
  if (!vecType) {
    contractOp->emitError("nova-gpu-vector-alloc: operand ")
        << operandIdx << " is not a vector — cannot stage through shared memory";
    return failure();
  }
  if (vecType.isScalable()) {
    contractOp->emitError("nova-gpu-vector-alloc: scalable vector operand ")
        << operandIdx << " cannot be statically allocated in shared memory";
    return failure();
  }

  int rank = vecType.getRank();
  SmallVector<int64_t> sgShape(vecType.getShape().begin(),
                               vecType.getShape().end());

  // Read sg_counts and sg_strides from the layout attribute.
  SmallVector<int64_t> sgCounts  = getIntArray(layoutAttr, "sg_counts");
  SmallVector<int64_t> sgStrides = getIntArray(layoutAttr, "sg_strides");

  if ((int)sgCounts.size() != rank || (int)sgStrides.size() != rank) {
    contractOp->emitError("nova-gpu-vector-alloc: sg_counts/sg_strides rank "
                          "mismatch for operand ")
        << operandIdx;
    return failure();
  }

  // ── Compute workgroup-tile shape ────────────────────────────────────────
  // wgShape[d] = sgShape[d] * sg_counts[d]
  // For dims where sg_counts[d] == 1 (not distributed) this is a no-op.
  SmallVector<int64_t> wgShape(rank);
  for (int d = 0; d < rank; ++d)
    wgShape[d] = sgShape[d] * std::max<int64_t>(sgCounts[d], 1);

  Attribute smemSpace = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

  // ── Step A: gpu.barrier at start of sg_forall body ─────────────────────
  Block *sgBody = sgForall.getBody();
  if (barrierBlocks.insert(sgBody).second) {
    OpBuilder barrierBuilder(sgBody, sgBody->begin());
    gpu::BarrierOp::create(barrierBuilder, loc);
  }

  // ── Step B: alloc_tensor OUTSIDE sg_forall, WG-sized ───────────────────
  RankedTensorType wgTensorType =
      RankedTensorType::get(wgShape, vecType.getElementType(), smemSpace);
  auto wgAllocOp = bufferization::AllocTensorOp::create(
      wgAllocBuilder, loc, wgTensorType,
      /*dynamicSizes=*/ValueRange{}, /*copy=*/Value());
  wgAllocOp.setMemorySpaceAttr(smemSpace);
  Value wgAlloc = wgAllocOp.getResult();

  // ── Steps C–E: inside the sg_forall body, just before contractOp ───────
  OpBuilder b(contractOp);

  // Compute per-subgroup slice offsets using the forall induction variables.
  //
  // TileAndDistribute tiles dim d of the workgroup tile using forall dim d,
  // so sgForall.getInductionVars()[d] corresponds to tensor dim d directly.
  // sg_strides[d] != 0 marks dims that are distributed across subgroups;
  // for those dims: offset[d] = iv[d] * sgShape[d].
  // For non-distributed dims (sg_strides[d] == 0): offset[d] = 0.
  SmallVector<Value> ivs = llvm::to_vector(sgForall.getInductionVars());

  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(rank);
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

  for (int d = 0; d < rank; ++d) {
    sizes[d] = b.getIndexAttr(sgShape[d]);
    if (sgStrides[d] != 0 && d < (int)ivs.size()) {
      // offset[d] = iv[d] * sgShape[d]
      Value mulStride = b.create<arith::ConstantIndexOp>(loc, sgShape[d]);
      Value off = b.create<arith::MulIOp>(loc, ivs[d], mulStride);
      offsets[d] = off;
    }
  }

  // Step C: extract the per-sg write slice from the wg alloc.
  RankedTensorType sgTensorType =
      RankedTensorType::get(sgShape, vecType.getElementType(), smemSpace);
  Value writeSlice = tensor::ExtractSliceOp::create(
      b, loc, sgTensorType, wgAlloc, offsets, sizes, strides);

  // Step C: transfer_write the per-sg vector into the slice at [0,0,...].
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> zeroIndices(rank, c0);
  SmallVector<bool> inBounds(rank, true);
  Value written = vector::TransferWriteOp::create(
                      b, loc, operand, writeSlice, zeroIndices, inBounds)
                      .getResult();

  // Step D: nova.value_barrier on the written per-sg tensor (inside body).
  // Ensures all threads finish writing their slice before any thread reads.
  auto barrier = nova::ValueBarrierOp::create(b, loc, ValueRange{written});
  Value syncedTensor = barrier.getResults()[0];

  // Step E: transfer_read the vector back from the synced slice.
  Value newVec = vector::TransferReadOp::create(
                     b, loc, vecType, syncedTensor, zeroIndices,
                     /*padding=*/std::nullopt, inBounds)
                     .getResult();

  // Step F: rewire the contract operand and clear the shared_mem flag.
  contractOp->setOperand(operandIdx, newVec);
  contractOp->setAttr("nova.layout_" + std::to_string(operandIdx),
                      clearSharedMem(ctx, layoutAttr));

  return success();
}

//===----------------------------------------------------------------------===//
// §4  Per-contract dispatch
//===----------------------------------------------------------------------===//

static LogicalResult materializeForContractOp(
    MLIRContext *ctx, vector::ContractionOp contractOp,
    llvm::SmallPtrSet<Block *, 4> &barrierBlocks) {

  unsigned layoutCount = std::min(contractOp->getNumOperands(), 3u);

  bool anySharedMem = false;
  for (unsigned i = 0; i < layoutCount; i++) {
    if (getSharedMem(contractOp->getAttrOfType<DictionaryAttr>(
            "nova.layout_" + std::to_string(i)))) {
      anySharedMem = true;
      break;
    }
  }
  if (!anySharedMem)
    return success();

  scf::ForallOp sgForall = findEnclosingSubgroupForall(contractOp);
  if (!sgForall) {
    contractOp->emitError(
        "nova-gpu-vector-alloc: no enclosing thread-mapped scf.forall found "
        "for vector.contract with shared_mem=true operand");
    return failure();
  }

  // All wg-sized allocs are inserted just before the sg_forall (outside it).
  OpBuilder wgAllocBuilder(sgForall);

  for (unsigned i = 0; i < layoutCount; i++) {
    auto attr = contractOp->getAttrOfType<DictionaryAttr>(
        "nova.layout_" + std::to_string(i));
    if (!getSharedMem(attr))
      continue;
    if (failed(stageOperandThroughSmem(ctx, contractOp, i, attr, sgForall,
                                       wgAllocBuilder, barrierBlocks)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// §5  Pass
//===----------------------------------------------------------------------===//

struct NovaGPUVectorAllocPass
    : public PassWrapper<NovaGPUVectorAllocPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUVectorAllocPass)

  StringRef getArgument() const override { return "nova-gpu-vector-alloc"; }
  StringRef getDescription() const override {
    return "Stage vector.contract operands marked shared_mem=true through "
           "workgroup-sized shared memory (race-condition-free).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
                    vector::VectorDialect, nova::NovaDialect,
                    arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<vector::ContractionOp> contractOps;
    funcOp.walk([&](vector::ContractionOp op) {
      unsigned layoutCount = std::min(op->getNumOperands(), 3u);
      for (unsigned i = 0; i < layoutCount; i++) {
        if (getSharedMem(op->getAttrOfType<DictionaryAttr>(
                "nova.layout_" + std::to_string(i)))) {
          contractOps.push_back(op);
          return;
        }
      }
    });

    llvm::SmallPtrSet<Block *, 4> barrierInsertedBlocks;
    for (vector::ContractionOp contractOp : contractOps) {
      if (failed(materializeForContractOp(ctx, contractOp,
                                         barrierInsertedBlocks)))
        return signalPassFailure();
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
