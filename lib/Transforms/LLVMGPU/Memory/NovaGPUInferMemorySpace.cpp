//===- NovaGPUInferMemorySpace.cpp - Structural memory-space annotation ====//
//
// Pre-bufferization pass.  Annotates bufferization.alloc_tensor ops with
// the appropriate GPU address space before OneShotBufferize runs.
//
// Decision logic:
//
//   Shared (workgroup)  (#gpu.address_space<workgroup>)
//     The alloc_tensor is used as shared_outs of a thread-distributed
//     scf.forall that is nested inside a block-mapped forall.
//
//     ALSO: the alloc_tensor's value flows (through scf.for iter_args,
//     vector.transfer_write tensor results, or scf.forall results) into
//     a thread-forall inside a block-forall.  This handles the ACC
//     accumulator pattern:
//
//       %acc = bufferization.alloc_tensor() : tensor<1x64x8xf32>
//       %init = vector.transfer_write %bias, %acc    ← direct user: write op
//       %result = scf.for iter_args(%arg = %init)    ← direct user: scf.for
//         scf.forall (1,4,1) shared_outs(%s = %arg)  ← thread-forall writes
//           vector.contract ... → writes into %s
//         scf.yield %s
//       scf.forall (1,64,4) shared_outs(%dst = %wg)  ← thread-forall reads
//         tensor.extract_slice %result ...
//
//     Without chasing through iter_args, isDefinitelyShared sees only
//     vector.transfer_write and scf.for as direct users — neither is a
//     thread-forall — so %acc is left unannotated and bufferizes to a
//     private per-thread stack alloca.  When the (1,4,1) sg_forall writes
//     into it and the (1,64,4) copy forall reads it, the 252 non-writer
//     threads read garbage → CUDA_ERROR_ILLEGAL_ADDRESS.
//
//   Unannotated (register / no explicit address space)
//     Everything else is left without a memory-space annotation so that
//     downstream vectorization and scalarization passes can keep values in
//     registers.  We explicitly avoid #gpu.address_space<private> because
//     despite the name "private", CUDA private memory maps to thread-local
//     storage in *global* (off-chip DRAM) memory — not registers.
//
// CSE de-aliasing:
//   Nova's pipeline runs CSE between PromoteMatmulOperands and this pass.
//   CSE merges all tensor.empty() ops with the same type into one SSA value.
//   After EmptyTensorToAllocTensor, this becomes a single alloc_tensor with
//   mixed users — some thread-forall (need workgroup) and some block-forall
//   or other ops (should remain unannotated).
//
//   To handle this, we first split such mixed-user alloc_tensors: each
//   thread-forall-inside-block-forall use gets its own cloned alloc_tensor
//   tagged workgroup.  The original retains the non-shared uses and stays
//   unannotated so downstream passes handle it at register level.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-infer-memory-space"

using namespace mlir;

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// forall mapping helpers
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUThreadMappingAttr>(a);
  });
}

static bool hasWarpMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUWarpMappingAttr>(a);
  });
}

static bool hasBlockMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) return false;
  return llvm::any_of(*forall.getMapping(), [](Attribute a) {
    return isa<gpu::GPUBlockMappingAttr>(a);
  });
}

//===----------------------------------------------------------------------===//
// isInsideWorkgroupForall
//===----------------------------------------------------------------------===//

static bool isInsideWorkgroupForall(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(parent)) {
      if (hasBlockMapping(forallOp))
        return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// isThreadForallInsideBlockForall
//===----------------------------------------------------------------------===//

static bool isThreadForallInsideBlockForall(Operation *op) {
  auto forallOp = dyn_cast<scf::ForallOp>(op);
  if (!forallOp)
    return false;
  // Thread-mapped OR warp-mapped foralls inside a block forall need workgroup
  // (shared) memory for their promoted operand buffers.  Warp-mapped foralls
  // are created by subgroup tiling for the MMA (tensor core) path — the
  // promoted A/B copies must be in shared memory so all warps can read them
  // after a barrier.
  if (!hasThreadMapping(forallOp) && !hasWarpMapping(forallOp))
    return false;
  return isInsideWorkgroupForall(forallOp);
}

//===----------------------------------------------------------------------===//
// valueReachesThreadForall
//
// Chase SSA uses of |val| transitively through:
//   - scf.for iter_args  (tensor accumulator passed loop-carried)
//   - vector.transfer_write tensor results  (write returns updated tensor)
//   - scf.forall results  (shared_outs result value)
//   - scf.for results  (post-loop result value)
//
// Returns true if any reachable use is a thread-mapped scf.forall inside
// a block-mapped scf.forall.
//
// This handles the ACC accumulator pattern where:
//   alloc_tensor → transfer_write → scf.for iter_arg
//                                 → (1,4,1) sg_forall (writes)
//                                 → (1,64,4) copy forall (reads)
//
// Without this chase, the pass only sees transfer_write and scf.for as
// direct users — neither is a thread-forall — and leaves the alloc_tensor
// unannotated, causing it to bufferize to a private per-thread stack alloca.
//
// |visited| guards against cycles in the SSA graph (shouldn't occur in
// well-formed MLIR but is cheap to check).
//===----------------------------------------------------------------------===//

static bool valueReachesThreadForall(Value val,
                                     DenseSet<Value> &visited,
                                     int depth = 0) {
  // Cycle / depth guard — the SSA graph is a DAG so cycles shouldn't
  // occur, but cap depth at 16 to be safe.
  if (depth > 16) return false;
  if (!visited.insert(val).second) return false;

  for (Operation *user : val.getUsers()) {
    // ── Direct hit: user is a thread-forall inside a block-forall ────────
    if (isThreadForallInsideBlockForall(user))
      return true;

    // ── Chase through scf.for iter_args ──────────────────────────────────
    // Pattern:
    //   %result = scf.for iter_args(%arg = %val) -> tensor<...> {
    //     scf.forall (thread-mapped) shared_outs(%s = %arg) { ... }
    //   }
    // We chase both:
    //   (a) the block argument (%arg) — used inside the loop body
    //   (b) the for op's results — used after the loop
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      // (a) block args corresponding to iter_args
      for (auto [operand, blockArg] :
           llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
        if (operand == val) {
          if (valueReachesThreadForall(blockArg, visited, depth + 1))
            return true;
        }
      }
      // (b) results of the for op (post-loop)
      for (Value result : forOp.getResults()) {
        if (valueReachesThreadForall(result, visited, depth + 1))
          return true;
      }
    }

    // ── Chase through vector.transfer_write (tensor SSA result) ──────────
    // Pattern:
    //   %written = vector.transfer_write %vec, %val[...] : tensor<...>
    //   %result  = scf.for iter_args(%arg = %written) -> tensor<...> { ... }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(user)) {
      for (Value result : writeOp->getResults()) {
        if (valueReachesThreadForall(result, visited, depth + 1))
          return true;
      }
    }

    // ── Chase through scf.forall results (shared_outs) ───────────────────
    // Pattern:
    //   %out = scf.forall shared_outs(%arg = %val) -> tensor<...> { ... }
    //   next use of %out ...
    if (auto forallOp = dyn_cast<scf::ForallOp>(user)) {
      for (Value result : forallOp.getResults()) {
        if (valueReachesThreadForall(result, visited, depth + 1))
          return true;
      }
    }

    // ── Chase through tensor.parallel_insert_slice ───────────────────────
    // Pattern:
    //   scf.forall shared_outs(%arg = %val) -> tensor<...> {
    //     tensor.parallel_insert_slice %slice into %arg[...] : tensor<...>
    //   }
    // %val is used as the shared_outs initializer of a forall. The forall
    // body inserts slices via parallel_insert_slice. The forall itself is
    // what we want to detect as a thread-forall, but %val's direct user is
    // the scf.forall op via its shared_outs operand list — so this is
    // actually already handled by the scf::ForallOp branch above.
    //
    // However, %val also reaches parallel_insert_slice indirectly when it
    // flows through a vector.transfer_write whose result is then inserted.
    // In that case we need to check whether the parallel_insert_slice's
    // parent scf.forall is a thread-forall.
    if (isa<tensor::ParallelInsertSliceOp>(user)) {
      // The parent of parallel_insert_slice is scf.forall.in_parallel,
      // whose parent is scf.forall.
      Operation *inParallel = user->getParentOp();
      if (inParallel) {
        Operation *parentForall = inParallel->getParentOp();
        if (parentForall && isThreadForallInsideBlockForall(parentForall))
          return true;
      }
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// splitMixedUserAllocTensors
//===----------------------------------------------------------------------===//

static void splitMixedUserAllocTensors(func::FuncOp funcOp) {
  SmallVector<bufferization::AllocTensorOp> allocs;
  funcOp.walk([&](bufferization::AllocTensorOp alloc) {
    if (!alloc.getMemorySpace().has_value())
      allocs.push_back(alloc);
  });

  for (auto alloc : allocs) {
    SmallVector<OpOperand *> sharedUses, otherUses;
    for (OpOperand &use : alloc->getUses()) {
      if (isThreadForallInsideBlockForall(use.getOwner()))
        sharedUses.push_back(&use);
      else
        otherUses.push_back(&use);
    }

    if (sharedUses.empty() || otherUses.empty())
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]  splitting alloc_tensor: "
               << sharedUses.size() << " shared + "
               << otherUses.size() << " other uses\n");

    OpBuilder builder(alloc);
    for (OpOperand *use : sharedUses) {
      builder.setInsertionPoint(use->getOwner());
      auto clone = cast<bufferization::AllocTensorOp>(builder.clone(*alloc));
      use->set(clone.getResult());
    }
  }
}

//===----------------------------------------------------------------------===//
// isDefinitelyShared
//
// An alloc_tensor should be placed in workgroup (shared) memory when:
//
//   (A) All direct users are thread-foralls inside block-foralls, OR
//
//   (B) Its value reaches a thread-forall inside a block-forall through
//       a chain of: scf.for iter_args, vector.transfer_write results,
//       scf.forall results.
//
// Case (B) catches the ACC accumulator pattern where the alloc's direct
// users are vector.transfer_write and scf.for — not thread-foralls — but
// the value eventually feeds into a thread-forall that writes/reads it.
//===----------------------------------------------------------------------===//

static bool isDefinitelyShared(bufferization::AllocTensorOp alloc) {
  // Must be inside a block-mapped forall (workgroup scope) to be shareable.
  if (!isInsideWorkgroupForall(alloc))
    return false;

  // ── Case A: all direct users are thread-foralls ──────────────────────
  bool hasThreadForallUser = false;
  bool allUsersAreThreadForall = true;

  for (Operation *user : alloc->getUsers()) {
    if (isThreadForallInsideBlockForall(user)) {
      hasThreadForallUser = true;
    } else {
      allUsersAreThreadForall = false;
    }
  }

  if (hasThreadForallUser && allUsersAreThreadForall) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]  workgroup (all direct users are "
                  "thread-foralls)\n");
    return true;
  }

  // ── Case B: value reaches a thread-forall transitively ───────────────
  // Handles the ACC accumulator:
  //   alloc_tensor → transfer_write → scf.for iter_arg
  //                                 → (1,4,1) thread-forall writes into it
  //                                 → (1,64,4) thread-forall reads from it
  DenseSet<Value> visited;
  if (valueReachesThreadForall(alloc.getResult(), visited)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]  workgroup (reaches thread-forall "
                  "transitively through iter_args/write chain)\n");
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUInferMemorySpacePass
    : public PassWrapper<NovaGPUInferMemorySpacePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGPUInferMemorySpacePass)

  NovaGPUInferMemorySpacePass() = default;
  NovaGPUInferMemorySpacePass(const NovaGPUInferMemorySpacePass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, gpu::GPUDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();

    // Step 1: Split alloc_tensors that CSE merged into mixed-user values.
    splitMixedUserAllocTensors(funcOp);

    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    // Step 2: Tag each alloc_tensor with its inferred memory space.
    WalkResult res = funcOp.walk([&](bufferization::AllocTensorOp alloc) {
      std::optional<Attribute> existingSpace = alloc.getMemorySpace();
      if (existingSpace.has_value()) {
        if (*existingSpace == workgroupSpace)
          return WalkResult::advance();
        alloc.emitOpError("unexpected gpu memory space — must be workgroup");
        return WalkResult::interrupt();
      }

      if (isDefinitelyShared(alloc)) {
        alloc.setMemorySpaceAttr(workgroupSpace);
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  workgroup: " << alloc << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]  unannotated (register path): "
                   << alloc << "\n");
      }
      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      funcOp->emitOpError("failed to set gpu memory space for all "
                          "bufferization.alloc_tensor ops");
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "nova-gpu-infer-memory-space";
  }
  StringRef getDescription() const override {
    return "Structural GPU memory space inference: workgroup for alloc_tensors "
           "that reach thread-foralls (directly or through iter_args/write "
           "chains), unannotated for register-level values.";
  }
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNovaGPUInferMemorySpacePass() {
  return std::make_unique<NovaGPUInferMemorySpacePass>();
}

void registerNovaGPUInferMemorySpacePass() {
  PassRegistration<NovaGPUInferMemorySpacePass>();
}

} // namespace mlir::nova