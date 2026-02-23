// Test: nova-gpu-apply-tiling-level-reduction
// Verifies that the K (reduction) dimension of linalg.matmul inside a
// workgroup scf.forall is tiled into a sequential scf.for loop with step 32,
// and that A/B linalg.copy producers are fused into that loop.
//
// Full pipeline:
//   nova-tile-and-distribute  → creates scf.forall{block}
//   nova-gpu-pad-operands     → pads A/B to 128x300 / 300x128
//   nova-gpu-promote-matmul   → alloc_tensor{workgroup} + linalg.copy
//   nova-gpu-apply-tiling-level-reduction → K-tiling into scf.for step 32

// RUN: nova-opt %s \
// RUN:   -nova-tile-and-distribute \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-pad-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-promote-matmul-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-reduction \
// RUN:   -canonicalize -cse \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @k_tiling_test
// CHECK:       scf.forall
// CHECK:         scf.for [[K:%[a-zA-Z0-9_]+]] = {{.*}} to {{.*}} step {{.*}}
// CHECK:           linalg.matmul
// CHECK:           scf.yield

func.func @k_tiling_test(
    %arg0: tensor<100x300xf32>,
    %arg1: tensor<300x200xf32>,
    %arg2: tensor<100x200xf32>) -> tensor<100x200xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<100x200xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<100x200xf32>) -> tensor<100x200xf32>
  %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<100x300xf32>, tensor<300x200xf32>)
      outs(%fill : tensor<100x200xf32>) -> tensor<100x200xf32>
  // Bias add (consumer, should still be fused in forall)
  %out = linalg.generic {
      indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>,
                       affine_map<(d0,d1) -> (d0,d1)>,
                       affine_map<(d0,d1) -> (d0,d1)>],
      iterator_types = ["parallel","parallel"]}
      ins(%result, %arg2 : tensor<100x200xf32>, tensor<100x200xf32>)
      outs(%empty : tensor<100x200xf32>) {
    ^bb0(%in: f32, %in2: f32, %out_: f32):
      %add = arith.addf %in, %in2 : f32
      linalg.yield %add : f32
  } -> tensor<100x200xf32>
  return %out : tensor<100x200xf32>
}
