// Test: Matmul with exact tile size (128x128 x 128x128).
// Verifies full-tile optimization:
//   - Both M=128 and N=128 equal the tile size (128)
//   - Full-tile opt zeros one dim to avoid single-trip loops
//   - forall becomes 1D (only one non-zero tile dim kept)
//   - No padding needed (sizes already match tile)

// RUN: %nova-opt %s -nova-tile-and-distribute -nova-gpu-pad-operands \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @matmul_exact_tile
// CHECK:         scf.forall (%{{.*}}) = (0) to (128) step (128)
// CHECK-NOT:     tensor.pad
// CHECK:           linalg.matmul
// CHECK:         mapping = [#gpu.block<x>]

func.func @matmul_exact_tile(%A: tensor<128x128xf32>,
                              %B: tensor<128x128xf32>,
                              %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                          outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
