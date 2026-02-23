// Test: Matmul + bias add + ReLU (two consumers fused).
// Verifies:
//   - Both bias add and ReLU fused into the same forall loop
//   - Correct dynamic slice sizes for boundary tiles
//   - All three ops (matmul, bias, relu) inside single scf.forall

// RUN: %nova-opt %s -nova-tile-and-distribute -nova-gpu-pad-operands \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK:         scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (100, 200) step (128, 128)
// CHECK:           linalg.matmul
// CHECK:           linalg.generic
// CHECK-SAME:        addf
// CHECK:           linalg.generic
// CHECK-SAME:        maximumf
// CHECK:           scf.forall.in_parallel
// CHECK:         mapping = [#gpu.block<y>, #gpu.block<x>]

func.func @matmul_bias_relu(%A: tensor<100x300xf32>,
                             %B: tensor<300x200xf32>,
                             %bias: tensor<100x200xf32>) -> tensor<100x200xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<100x200xf32>
  %C = linalg.fill ins(%cst : f32) outs(%empty : tensor<100x200xf32>) -> tensor<100x200xf32>
  %matmul = linalg.matmul ins(%A, %B : tensor<100x300xf32>, tensor<300x200xf32>)
                          outs(%C : tensor<100x200xf32>) -> tensor<100x200xf32>
  %bias_out = tensor.empty() : tensor<100x200xf32>
  %biased = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<100x200xf32>, tensor<100x200xf32>)
    outs(%bias_out : tensor<100x200xf32>) {
  ^bb0(%m: f32, %b: f32, %o: f32):
    %add = arith.addf %m, %b : f32
    linalg.yield %add : f32
  } -> tensor<100x200xf32>
  %relu_out = tensor.empty() : tensor<100x200xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%biased : tensor<100x200xf32>)
    outs(%relu_out : tensor<100x200xf32>) {
  ^bb0(%x: f32, %o: f32):
    %relu = arith.maximumf %x, %cst : f32
    linalg.yield %relu : f32
  } -> tensor<100x200xf32>
  return %result : tensor<100x200xf32>
}
