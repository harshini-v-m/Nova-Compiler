// Test: nova-gpu-apply-tiling-level-subgroup
// Verifies that the M/N parallel dimensions of linalg.matmul are tiled into
// an scf.for loop with step 16.
//
// RUN: nova-opt %s \
// RUN:   -nova-tile-and-distribute \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-pad-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-promote-matmul-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-reduction \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-subgroup \
// RUN:   -canonicalize -cse \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @subgroup_tiling_test
// CHECK:       scf.forall
// K-tiling loop
// CHECK:         scf.for {{.*}} step %c32
// Subgroup-tiling loop
// CHECK:           scf.for {{.*}} step %c16
// CHECK:             scf.for {{.*}} step %c16
// CHECK:               linalg.matmul

func.func @subgroup_tiling_test(
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
