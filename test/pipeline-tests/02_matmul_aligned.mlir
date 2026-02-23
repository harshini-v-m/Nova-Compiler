// Test: Matmul with already-aligned sizes (256x128 x 128x512).
// Verifies:
//   - Tiled to forall (0,0) to (256,512) step (128,128)
//   - No padding ops generated (sizes are multiples of 128)
//   - Static tile shapes in extract_slice

// RUN: %nova-opt %s -nova-tile-and-distribute -nova-gpu-pad-operands \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @matmul_aligned
// CHECK:         scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (256, 512) step (128, 128)
// CHECK-NOT:     tensor.pad
// CHECK:           linalg.matmul
// CHECK:         mapping = [#gpu.block<y>, #gpu.block<x>]

func.func @matmul_aligned(%A: tensor<256x128xf32>,
                           %B: tensor<128x512xf32>,
                           %C: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %result = linalg.matmul ins(%A, %B : tensor<256x128xf32>, tensor<128x512xf32>)
                          outs(%C : tensor<256x512xf32>) -> tensor<256x512xf32>
  return %result : tensor<256x512xf32>
}
