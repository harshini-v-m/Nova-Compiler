#ifndef COMPILER_TRANSFORMS_NOVA_SCF_LOOP_UNROLL_H_
#define COMPILER_TRANSFORMS_NOVA_SCF_LOOP_UNROLL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

/// Creates a pass that unrolls every scf.for in the function body by
/// `unrollFactor` using mlir::loopUnrollByFactor.
/// Registered as --nova-scf-loop-unroll for use with nova-opt.
std::unique_ptr<Pass> createNovaScfLoopUnrollPass(uint64_t unrollFactor = 4);

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_NOVA_SCF_LOOP_UNROLL_H_
