#ifndef COMPILER_TRANSFORMS_LLVMGPU_LOOP_FUSION_H_
#define COMPILER_TRANSFORMS_LLVMGPU_LOOP_FUSION_H_

#include <memory>

namespace mlir {
class Pass;
namespace nova {

/// Fuses adjacent scf.for loops that have identical (lb, ub, step) bounds and
/// disjoint memory footprints into a single loop.  The pass is conservative:
/// it only fuses two sibling loops when the memrefs written by one loop are
/// provably not accessed (read or written) by the other, ruling out any
/// loop-carried memory dependence across the boundary.
std::unique_ptr<Pass> createNovaScfLoopFusionPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_LLVMGPU_LOOP_FUSION_H_
