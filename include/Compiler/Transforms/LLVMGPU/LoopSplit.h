#ifndef COMPILER_TRANSFORMS_LLVMGPU_LOOP_SPLIT_H_
#define COMPILER_TRANSFORMS_LLVMGPU_LOOP_SPLIT_H_

#include <memory>

namespace mlir {
class Pass;
namespace nova {

/// Creates a pass that splits scf.for loops at condition boundaries.
///
/// When a loop body contains an scf.if whose condition is a signed comparison
/// (slt / sle / sgt / sge) between the induction variable and a loop-invariant
/// value, the loop is split at that boundary into two loops.  The scf.if is
/// removed from each half — replaced by the always-true or always-false branch
/// — letting downstream canonicalization fold dead code away.
///
/// Only loops with step == 1 are split (guarantees split-point alignment).
/// Loops with iter_args and scf.if with results are fully supported.
///
/// Registered as --nova-scf-loop-split for standalone nova-opt use.
std::unique_ptr<Pass> createNovaScfLoopSplitPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_LLVMGPU_LOOP_SPLIT_H_
