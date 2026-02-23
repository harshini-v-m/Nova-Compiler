#ifndef NOVA_COMPILER_TRANSFORMS_LLVMGPU_TILEANDFUSEUTILS_H_
#define NOVA_COMPILER_TRANSFORMS_LLVMGPU_TILEANDFUSEUTILS_H_

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <queue>

namespace mlir::nova {

/// Tile and fuse producers of extract slice operations from the worklist into
/// the given loops.
void fuseProducersOfSlices(RewriterBase &rewriter,
                           std::queue<Operation *> &worklist,
                           scf::SCFTileAndFuseOptions &options,
                           MutableArrayRef<LoopLikeOpInterface> loops);

/// Starting from `op` walk all operands backwards to find all fusible operations.
void collectTiledAndFusedOps(Operation *rootOp,
                             llvm::SmallDenseSet<Operation *> &result);

/// Fuse all consumers of the given `tiledOps` into the surrounding `scf.forall`.
FailureOr<std::queue<Operation *>> fuseConsumersIntoForall(
    RewriterBase &rewriter, ArrayRef<Operation *> tiledOps,
    MutableArrayRef<LoopLikeOpInterface> loops,
    std::function<bool(Operation *)> filterFn = [](Operation *) {
      return true;
    });

} // namespace mlir::nova

#endif // NOVA_COMPILER_TRANSFORMS_LLVMGPU_TILEANDFUSEUTILS_H_
