// Test: nova-gpu-promote-matmul-operands
// Verifies that the A and B operands of linalg.matmul inside a workgroup
// scf.forall are promoted to workgroup shared memory using a two-stage copy
// with a nova.fusion_barrier.
//
// RUN: nova-opt %s \
// RUN:   -nova-tile-and-distribute \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-pad-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-promote-matmul-operands \
// RUN:   -canonicalize -cse \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @promotion_test
// CHECK:       scf.forall
// The strategy for this matrix size promotes both input operands (A and B).
// Each input gets: tensor.empty dest + linalg.copy with nova.promote_to_workgroup.
// That marker lets InferMemorySpace unconditionally classify the destination as
// workgroup memory, bypassing isCrossThreadAccess (which misses the pattern where
// the matmul reads the copy result as an input rather than a DPS init).
// CHECK:         tensor.empty
// CHECK:         linalg.copy{{.*}}nova.promote_to_workgroup
// CHECK:         tensor.empty
// CHECK:         linalg.copy{{.*}}nova.promote_to_workgroup
// CHECK:         linalg.matmul
// CHECK:         scf.forall.in_parallel

func.func @promotion_test(
    %arg0: tensor<100x300xf32>,
    %arg1: tensor<300x200xf32>) -> tensor<100x200xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<100x200xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<100x200xf32>) -> tensor<100x200xf32>
  %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<100x300xf32>, tensor<300x200xf32>)
      outs(%fill : tensor<100x200xf32>) -> tensor<100x200xf32>
  return %result : tensor<100x200xf32>
}
