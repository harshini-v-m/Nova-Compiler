// Test: Matmul + elementwise bias add (consumer fusion).
// Verifies:
//   - Matmul tiled and padded
//   - Bias add fused as consumer into the forall loop
//   - ExtractSliceOfPadTensorSwapPattern: pad pushed inward on A
//   - FoldFillIntoPad: zero-fill output becomes fill(0, empty(128x128))
//   - Both matmul and bias add inside single scf.forall

// RUN: %nova-opt %s -nova-tile-and-distribute -nova-gpu-pad-operands \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @matmul_plus_bias
// CHECK:         scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (100, 200) step (128, 128)
// CHECK:           tensor.pad
// CHECK:           linalg.fill
// CHECK:           linalg.matmul
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK:           scf.forall.in_parallel
// CHECK:         mapping = [#gpu.block<y>, #gpu.block<x>]

func.func @matmul_plus_bias(%A: tensor<100x300xf32>,
                             %B: tensor<300x200xf32>,
                             %bias: tensor<100x200xf32>) -> tensor<100x200xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<100x200xf32>
  %C = linalg.fill ins(%cst : f32) outs(%empty : tensor<100x200xf32>) -> tensor<100x200xf32>
  %matmul = linalg.matmul ins(%A, %B : tensor<100x300xf32>, tensor<300x200xf32>)
                          outs(%C : tensor<100x200xf32>) -> tensor<100x200xf32>
  %out = tensor.empty() : tensor<100x200xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<100x200xf32>, tensor<100x200xf32>)
    outs(%out : tensor<100x200xf32>) {
  ^bb0(%m: f32, %b: f32, %o: f32):
    %add = arith.addf %m, %b : f32
    linalg.yield %add : f32
  } -> tensor<100x200xf32>
  return %result : tensor<100x200xf32>
}
