module {
  // 2D Matmul
  func.func @matmul_2d(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %1=nova.sqrt %arg0 : tensor <8x16xf32>
    %0 = nova.div %arg0, %1 : tensor<8x16xf32>, tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}
