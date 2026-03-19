//===- NovaGPUTileAndFuseUtils.cpp - Tile-and-fuse loop utilities ---------===//
//
// Utility helpers shared by the GPU tiling passes:
//
//   fuseProducersOfSlices   — fuses producer ops into a tile-and-fuse loop
//                            by processing a worklist of extract_slice ops.
//   collectTiledAndFusedOps — BFS collection of all tilable ops reachable
//                            from a root op (producers + consumers).
//   fuseConsumersIntoForall — fuses tilable consumer ops into a scf.forall
//                            via tileAndFuseConsumerOfSlices. Uses dominance
//                            ordering to fuse consumers in safe order.
//
// Mirrors IREE's mlir::iree_compiler::tileAndFuseProductors / consumers
// from Codegen/Common/TileAndFuseUtils.cpp.
//
//===----------------------------------------------------------------------===//

#include "NovaGPUTileAndFuseUtils.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-tile-and-fuse-utils"

namespace mlir::nova {

// Helper for isa_and_present check
template <typename T>
static bool isa_and_present(Operation *op) {
  return op && isa<T>(op);
}

// Processes a worklist of extract_slice ops and attempts to fuse the producer
// of each slice into the surrounding tiled loop. Newly generated slices from
// each successful fusion are added to the worklist for transitive fusion.
void fuseProducersOfSlices(RewriterBase &rewriter,
                            std::queue<Operation *> &worklist,
                            scf::SCFTileAndFuseOptions &options,
                            MutableArrayRef<LoopLikeOpInterface> loops) {
  while (!worklist.empty()) {
    auto candidateSlice = cast<tensor::ExtractSliceOp>(worklist.front());
    worklist.pop();

    auto fusableProducer =
        candidateSlice.getSource().getDefiningOp<TilingInterface>();
    if (!fusableProducer) {
      continue;
    }

    std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> controlFnResult =
        options.fusionControlFn(candidateSlice,
                                cast<OpResult>(candidateSlice.getSource()),
                                /*destinationInitArg=*/false);
    if (!controlFnResult) {
      continue;
    }

    // The operands of the fused producer might themselves be slices of
    // values produced by operations that implement the `TilingInterface`.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSlice, loops);
    if (!fusedResult) {
      continue;
    }

    for (auto newSlice : fusedResult->generatedSlices) {
      worklist.push(newSlice);
    }
  }
}

// BFS from `rootOp` collecting all tilable ops reachable via producer
// (operand-defining TilingInterface) and consumer (result-using TilingInterface)
// edges. Used to scope the fusion analysis to the compute cluster around
// the root matmul/reduction op.
//
// IMPORTANT: When following consumer edges, stop at contraction ops
// (matmul, batch_matmul). Without this, sibling contractions that share a
// producer (e.g. grad_input and grad_weight matmuls sharing a grad_output
// fill) would be merged into the same cluster, causing both to be fused
// into one forall. The second matmul's full-size output then materializes
// in every block's local memory (30MB+), crashing CUDA.
void collectTiledAndFusedOps(Operation *rootOp,
                              llvm::SmallDenseSet<Operation *> &result) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  result.insert(rootOp);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    // Collect all tilable producers (traverse backward unconditionally).
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          result.contains(producer)) {
        continue;
      }
      worklist.push_back(producer);
      result.insert(producer);
    }
    // Collect tilable consumers, but stop at contraction boundaries.
    // Contraction ops are independent roots that should get their own forall.
    for (auto user : current->getUsers()) {
      if (result.count(user))
        continue;
      if (!isa<TilingInterface>(user))
        continue;
      // Don't traverse into contractions — they are independent roots.
      if (auto linalgUser = dyn_cast<linalg::LinalgOp>(user)) {
        if (linalg::isaContractionOpInterface(linalgUser))
          continue;
      }
      worklist.push_back(user);
      result.insert(user);
    }
  }
}

namespace {
struct ConsumerFusionQueueEntry {
  ConsumerFusionQueueEntry(SmallVector<Operation *> &&slices,
                           Operation *fusableUser)
      : slices(std::move(slices)), fusableUser(fusableUser) {}

  SmallVector<Operation *> slices;
  Operation *fusableUser;
};
} // namespace

// PERFORMANCE CRITICAL — Fuses tilable consumer ops into a scf.forall loop.
//
// Finds all parallel-insert-slice-producing consumers of the tiled ops,
// sorts them in dominance order so that the closest consumer is fused last
// (this preserves use-def correctness after each replacement), then calls
// scf::tileAndFuseConsumerOfSlices for each.
//
// IMPORTANT: Uses a `replacedOps` set to skip stale candidates.
// MLIR bump-allocates ops; after replaceOp() the memory is freed but the
// pointer value still uniquely identifies the erased op for the lifetime of
// this function (no new op is allocated at the same address within one pass
// run). Checking the erased set avoids accessing destroyed op memory.
FailureOr<std::queue<Operation *>> fuseConsumersIntoForall(
    RewriterBase &rewriter, ArrayRef<Operation *> tiledOps,
    MutableArrayRef<LoopLikeOpInterface> loops,
    std::function<bool(Operation *)> filterFn) {
  
  SmallVector<ConsumerFusionQueueEntry> candidates;
  llvm::SmallDenseSet<tensor::ParallelInsertSliceOp> allCandidates;
  // Track ops erased by replaceOp so we can skip stale candidates.
  // MLIR bump-allocates ops, so pointer comparison remains valid after
  // erasure even though the op's contents are destroyed.
  llvm::SmallPtrSet<Operation *, 8> replacedOps;

  auto addCandidateSlices = [&candidates, &allCandidates, &replacedOps,
                             &filterFn](Operation *fusedOp,
                                        DominanceInfo &dominanceInfo) {
    for (auto *userOp : fusedOp->getResults().getUsers()) {
      auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(userOp);
      if (!sliceOp || allCandidates.contains(sliceOp)) {
        continue;
      }

      auto currLoop =
          cast<scf::ForallOp>(sliceOp->getParentOp()->getParentOp());
      OpResult loopResult = currLoop.getTiedOpResult(
          currLoop.getTiedOpOperand(cast<BlockArgument>(sliceOp.getDest())));
      SmallVector<Operation *> users = llvm::to_vector(
          llvm::make_filter_range(loopResult.getUsers(), filterFn));
      if (users.empty()) {
        continue;
      }
      mlir::computeTopologicalSorting(users);
      unsigned forallDims = currLoop.getInductionVars().size();
      // Get forall upper bounds for dimension-wise overlap check.
      auto forallUBs = currLoop.getStaticUpperBound();
      for (Operation *fusableUser : users) {
        // Skip consumers whose results can't be uniquely addressed by all
        // forall induction variables. Two cases:
        //   1. result rank < forall dims (e.g., 2D output in 3D forall)
        //   2. result dim[i] < forall upper_bound[i] for some i
        //      (e.g., 8x1024x384 result in (8,1024,1536) forall — the
        //       N IV has no matching dim, so 12 N-iterations write to the
        //       same positions → race condition)
        {
          bool wouldOverlap = false;
          for (Value result : fusableUser->getResults()) {
            auto tensorType = dyn_cast<RankedTensorType>(result.getType());
            if (!tensorType)
              continue;
            unsigned rank = tensorType.getRank();
            if (rank < forallDims) {
              wouldOverlap = true;
              break;
            }
            // Check dimension-wise: result shape must cover forall bounds.
            auto shape = tensorType.getShape();
            for (unsigned i = 0; i < forallDims && i < rank; ++i) {
              if (!ShapedType::isDynamic(shape[i]) &&
                  !ShapedType::isDynamic(forallUBs[i]) &&
                  shape[i] < forallUBs[i]) {
                wouldOverlap = true;
                break;
              }
            }
            if (wouldOverlap)
              break;
          }
          if (wouldOverlap) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[nova-tile-fuse-utils] Skipping consumer fusion: "
                       << "result shape doesn't cover forall bounds ("
                       << forallDims << " dims), would create overlapping "
                       << "writes\n");
            continue;
          }
        }

        // Don't fuse consumers whose OTHER operands (not from the forall)
        // transitively depend on any result of the forall.  This prevents
        // circular data dependencies where the fused consumer reads a value
        // that can only be computed AFTER the forall finishes.
        //
        // Example: LayerNorm after residual add:
        //   residual_add (forall result) → reduce → mean
        //   x_minus_mean = sub(residual_add, mean)
        // If x_minus_mean is fused, it reads 'mean' which needs the FULL
        // residual_add — creating a circular dependency.
        {
          bool hasCircularDep = false;
          // Collect all forall results for fast lookup.
          SmallPtrSet<Value, 4> forallResults;
          for (OpResult r : currLoop->getResults())
            forallResults.insert(r);

          for (OpOperand &opOperand : fusableUser->getOpOperands()) {
            Value operand = opOperand.get();
            // Skip operands that come directly from the forall (these are
            // the ones being fused — they're fine).
            if (forallResults.contains(operand))
              continue;
            // Walk the def chain of this operand to check if it
            // transitively depends on any forall result.
            SmallVector<Value, 8> worklist;
            SmallPtrSet<Value, 16> visited;
            worklist.push_back(operand);
            while (!worklist.empty() && !hasCircularDep) {
              Value v = worklist.pop_back_val();
              if (!visited.insert(v).second)
                continue;
              if (forallResults.contains(v)) {
                hasCircularDep = true;
                break;
              }
              Operation *defOp = v.getDefiningOp();
              if (!defOp)
                continue;
              for (Value inp : defOp->getOperands())
                worklist.push_back(inp);
            }
          }
          if (hasCircularDep) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[nova-tile-fuse-utils] Skipping consumer fusion: "
                       << "operand transitively depends on forall result "
                       << "(circular dependency)\n");
            continue;
          }
        }

        // Check all operands from the `scf.forall`
        SmallVector<OpResult> loopResults;
        for (OpOperand &opOperand : fusableUser->getOpOperands()) {
          if (opOperand.get().getDefiningOp() == currLoop.getOperation()) {
            loopResults.push_back(cast<OpResult>(opOperand.get()));
          }
        }

        SmallVector<Operation *> fusedSlices;
        for (OpResult result : loopResults) {
          BlockArgument tiedBlockArg =
              currLoop.getTiedBlockArgument(currLoop.getTiedOpOperand(result));
          SmallVector<tensor::ParallelInsertSliceOp> slices =
              llvm::map_to_vector(
                  currLoop.getCombiningOps(tiedBlockArg), [](Operation *op) {
                    return cast<tensor::ParallelInsertSliceOp>(op);
                  });
          llvm::append_range(fusedSlices, slices);
          allCandidates.insert_range(slices);
        }
        if (!fusedSlices.empty()) {
          ConsumerFusionQueueEntry entry(std::move(fusedSlices), fusableUser);

          // Comparator that puts the dominating user last.
          // Guard against replaced/erased ops to avoid accessing
          // destroyed memory in the dominance check.
          auto comp = [&](const ConsumerFusionQueueEntry &lhs,
                          const ConsumerFusionQueueEntry &rhs) {
            if (replacedOps.contains(lhs.fusableUser) ||
                replacedOps.contains(rhs.fusableUser))
              return false;
            return dominanceInfo.properlyDominates(rhs.fusableUser,
                                                   lhs.fusableUser);
          };

          auto *it = llvm::lower_bound(candidates, entry, comp);
          if (it != candidates.end() && it->fusableUser == fusableUser) {
            *it = std::move(entry);
          } else {
            candidates.insert(it, std::move(entry));
          }
        }
      }
    }
  };

  // Add slices from all tiled ops, not only the "main" one.
  DominanceInfo dominanceInfo;
  for (Operation *tiledOp : tiledOps) {
    addCandidateSlices(tiledOp, dominanceInfo);
  }

  std::queue<Operation *> newFusionOpportunities;
  while (!candidates.empty()) {
    ConsumerFusionQueueEntry entry = candidates.pop_back_val();

    // Skip candidates whose fusableUser was already erased by a prior fusion.
    if (replacedOps.contains(entry.fusableUser))
      continue;

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlices(rewriter, entry.slices, loops);
    if (failed(fusedResult)) {
      return failure();
    }

    // Replace the original consumer operation with the tiled implementation.
    if (!fusedResult->origConsumerOperands.empty() && !fusedResult->tiledOps.empty()) {
      Operation *origConsumer =
          fusedResult->origConsumerOperands.front()->getOwner();
      replacedOps.insert(origConsumer);
      rewriter.replaceOp(origConsumer, fusedResult->tiledOps.front());
    }

    DominanceInfo dominanceInfo;
    if (!fusedResult->tiledAndFusedConsumerOperands.empty()) {
      addCandidateSlices(
          fusedResult->tiledAndFusedConsumerOperands.front()->getOwner(),
          dominanceInfo);
    }

    // Add the list of new producer fusion opportunities.
    for (auto tiledOp : fusedResult.value().tiledOps) {
      for (auto operand : tiledOp->getOperands()) {
        if (auto sliceProducer =
                operand.getDefiningOp<tensor::ExtractSliceOp>()) {
          if (isa_and_present<TilingInterface>(
                  sliceProducer.getSource().getDefiningOp())) {
            newFusionOpportunities.push(sliceProducer);
          }
        }
      }
    }
  }
  return newFusionOpportunities;
}

} // namespace mlir::nova
