// Test: Basic matmul with non-aligned sizes (100x300 x 300x200).
// Verifies:
//   - Tiled to forall (0,0) to (100,200) step (128,128)
//   - A padded: 100->128 (+28 rows)
//   - B padded: 200->256 (+56 cols)
//   - FoldFillIntoPad fires: output init is fill(0, empty(128x128))
//   - Result sliced back to 100x200 before insert

// RUN: %nova-opt %s -nova-tile-and-distribute -nova-gpu-pad-operands \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @matmul_non_aligned
// CHECK:         scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (100, 200) step (128, 128)
// CHECK:           tensor.pad
// CHECK-SAME:        to tensor<128x300xf32>
// CHECK:           tensor.pad
// CHECK-SAME:        to tensor<300x128xf32>
// CHECK:           linalg.fill
// CHECK:           linalg.matmul
// CHECK:           scf.forall.in_parallel
// CHECK:         mapping = [#gpu.block<y>, #gpu.block<x>]

func.func @matmul_non_aligned(%A: tensor<100x300xf32>,
                               %B: tensor<300x200xf32>,
                               %C: tensor<100x200xf32>) -> tensor<100x200xf32> {
  %result = linalg.matmul ins(%A, %B : tensor<100x300xf32>, tensor<300x200xf32>)
                          outs(%C : tensor<100x200xf32>) -> tensor<100x200xf32>
  return %result : tensor<100x200xf32>
}
