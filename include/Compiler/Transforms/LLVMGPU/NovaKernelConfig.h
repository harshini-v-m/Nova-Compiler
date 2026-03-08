//===- NovaKernelConfig.h - GPU matmul config heuristic ---------*- C++ -*-===//
//
// Provides setMatmulLoweringConfig() which analyzes a linalg.matmul problem
// shape, selects tile sizes and MMA intrinsic from the given NVIDIA target,
// and attaches a #nova.lowering_config dict attribute to the op.
//
// Mirrors IREE's KernelConfig.cpp role (setMatmulVectorDistributionConfig /
// setRootConfig) but simplified for Nova's current pipeline.
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_NOVAKERNELCONFIG_H_
#define NOVA_TRANSFORMS_LLVMGPU_NOVAKERNELCONFIG_H_

#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::nova {

/// Analyzes `matmul`, selects a tile schedule and MMA intrinsic from `target`,
/// and attaches a "lowering_config" DictionaryAttr to the op.
///
/// The schedule follows a 4-subgroup-per-workgroup strategy:
///   workgroupM = mmaM * subgroupTilesM * numSubgroupsM
///   workgroupN = mmaN * subgroupTilesN * numSubgroupsN
///   reductionK = mmaK * kTilesPerStep     (the K-loop step)
///
/// Falls back to a SIMT tile table when no MMA is available.
///
/// Returns failure() if:
///   - The op is not a contraction (can't infer M/N/K dims).
///   - Any relevant loop dimension is dynamic.
LogicalResult setMatmulLoweringConfig(linalg::LinalgOp matmul,
                                      const NVIDIATargetInfo &target);

/// Sets a lowering config for a linalg.generic op with reduction iterators
/// (softmax, layer norm, sum, etc.). Tiles parallel dims to workgroups/threads
/// and reduction dims with a small factor for vectorization.
/// Mirrors IREE's setTileAndFuseLoweringConfig for non-contraction reductions.
LogicalResult setReductionLoweringConfig(linalg::LinalgOp op,
                                         const NVIDIATargetInfo &target);

/// Sets a lowering config for an all-parallel linalg.generic (elementwise ops).
/// Tiles parallel dims to workgroups and threads.
LogicalResult setElementwiseLoweringConfig(linalg::LinalgOp op,
                                           const NVIDIATargetInfo &target);

/// Walk all linalg ops inside `funcOp` and attach lowering configs.
/// Handles contractions (matmul), reductions (softmax, sum), and elementwise
/// ops in priority order. Ops that already have a "lowering_config" attribute
/// are skipped.
void initNovaGPULaunchConfig(mlir::func::FuncOp funcOp,
                              const NVIDIATargetInfo &target);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_NOVAKERNELCONFIG_H_
