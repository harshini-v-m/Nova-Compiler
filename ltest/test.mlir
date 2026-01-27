module {
  // 2D Matmul
  func.func @matmul_2d(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32>, tensor<16x32xf32>
    return %0 : tensor<8x32xf32>
  }

  // 3D Matmul (Batch Matmul)
  func.func @matmul_3d(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x16x32xf32>) -> tensor<4x8x32xf32> {
    %0 = nova.matmul %arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x16x32xf32>
    return %0 : tensor<4x8x32xf32>
  }

  // 4D Matmul (Batch Matmul with 2 batch dims)
  func.func @matmul_4d(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x16x32xf32>) -> tensor<2x4x8x32xf32> {
    %0 = nova.matmul %arg0, %arg1 : tensor<2x4x8x16xf32>, tensor<2x4x16x32xf32>
    return %0 : tensor<2x4x8x32xf32>
  }
}
