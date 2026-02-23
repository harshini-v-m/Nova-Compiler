// Test: Batch matmul (4x64x256 x 4x256x64).
// Verifies:
//   - 3D forall: batch tiled to 1, M/N tiled to 128
//   - K=256 stays 0 (reduction dim, not tiled at workgroup level)
//   - Mapped to gpu.block<z>, gpu.block<y>, gpu.block<x>
//   - M and N padded: 64->128 (+64 each)

// RUN: %nova-opt %s -nova-tile-and-distribute -nova-gpu-pad-operands \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul
// CHECK:         scf.forall (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (4, 64, 64) step (1, 128, 128)
// CHECK:           tensor.pad
// CHECK-SAME:        to tensor<1x128x256xf32>
// CHECK:           tensor.pad
// CHECK-SAME:        to tensor<1x256x128xf32>
// CHECK:           linalg.batch_matmul
// CHECK:         mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>]

func.func @batch_matmul(%A: tensor<4x64x256xf32>,
                         %B: tensor<4x256x64xf32>,
                         %C: tensor<4x64x64xf32>) -> tensor<4x64x64xf32> {
  %result = linalg.batch_matmul ins(%A, %B : tensor<4x64x256xf32>, tensor<4x256x64xf32>)
                                 outs(%C : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  return %result : tensor<4x64x64xf32>
}
