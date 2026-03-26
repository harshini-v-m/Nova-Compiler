#ifndef COMPILER_TRANSFORMS_LLVMGPU_NOVA_SCF_LOOP_VECTORIZE_H_
#define COMPILER_TRANSFORMS_LLVMGPU_NOVA_SCF_LOOP_VECTORIZE_H_

#include <memory>

namespace mlir {
class Pass;
namespace nova {

/// Creates a pass that vectorizes innermost scf.for loops.
///
/// Mirrors what affine loop vectorization does for affine.for loops, but
/// operates directly on scf.for / memref.load / memref.store IR.
///
/// For each eligible innermost loop the pass:
///   1. Emits a vectorized main loop [lb, vub, VF) that replaces stride-1
///      memref.load/store with vector.load/vector.store, broadcasts loop-
///      invariant scalars to vector via vector.broadcast, and widens all
///      element-wise arith ops to operate on the vector type.
///   2. Appends a scalar cleanup loop [vub, ub, 1) for the remainder
///      iterations (identical to the original body, IV remapped).
///
/// Eligibility requirements:
///   - Innermost loop: no nested scf.for in body.
///   - Step must be the constant 1.
///   - No iter_args (accumulator lives in memory, not SSA).
///   - Every memref.load / memref.store that references the IV must use it
///     as the *last* (innermost / stride-1) index.
///   - All non-load/store body ops must be memory-effect-free.
///   - At least one load or store accesses the IV in the last dimension.
///
/// The default vector factor is 4 (128-bit for f32).
/// Registered as --nova-scf-loop-vectorize for standalone nova-opt use.
std::unique_ptr<Pass> createNovaScfLoopVectorizePass(int64_t vectorFactor = 4);

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_LLVMGPU_NOVA_SCF_LOOP_VECTORIZE_H_
