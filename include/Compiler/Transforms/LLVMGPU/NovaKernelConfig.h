//===- NovaKernelConfig.h - GPU kernel config heuristic ----------*- C++ -*-===//
//
// Provides initNovaGPULaunchConfig() which walks a function, finds root
// compute operations, and attaches #nova.lowering_config attributes.
//
// Mirrors IREE's KernelConfig.cpp role (initGPULaunchConfig / setRootConfig)
// but simplified for Nova's current pipeline (no convolutions, no attention).
//
// Config dispatch order (single-root per dispatch cluster):
//   1. Contractions (matmul, batch_matmul) → MMA or SIMT table config
//   2. Generic with reduction → default config (vectorized reduction)
//   3. Generic all-parallel → default config (vectorized elementwise)
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_NOVAKERNELCONFIG_H_
#define NOVA_TRANSFORMS_LLVMGPU_NOVAKERNELCONFIG_H_

#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::nova {

/// Analyzes a contraction op, selects a tile schedule and MMA intrinsic from
/// `target`, and attaches a "lowering_config" DictionaryAttr.
///
/// Returns failure() if:
///   - The op is not a contraction (can't infer M/N/K dims).
///   - Any relevant loop dimension is dynamic.
///   - The op is a matvec (one of M,N == 1).
LogicalResult setContractConfig(linalg::LinalgOp op,
                                const NVIDIATargetInfo &target);

/// Sets a lowering config for a generic linalg op (reduction or elementwise).
/// Mirrors IREE's setRootDefaultConfig: distributes parallel dims across
/// workgroup threads with vectorization, tiles reduction dims to 4.
///
/// Handles both:
///   - Ops with reduction iterators (softmax, layer norm, sum)
///   - All-parallel ops (standalone elementwise)
LogicalResult setDefaultConfig(linalg::LinalgOp op,
                               const NVIDIATargetInfo &target);

/// Walk all linalg ops inside `funcOp` and attach lowering configs.
/// Uses a single-root model per compute cluster: finds root ops, stamps
/// configs, and lets the TileDispatch pass handle fusion.
void initNovaGPULaunchConfig(mlir::func::FuncOp funcOp,
                              const NVIDIATargetInfo &target);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_NOVAKERNELCONFIG_H_
