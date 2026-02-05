module {
  func.func @test_gather_3d(%arg0: tensor<10x20xf32>, %arg1: tensor<5x8xi32>) -> tensor<5x8x20xf32> {
    %0 = "nova.gather"(%arg0, %arg1) {axis = 0 : i64} : (tensor<10x20xf32>, tensor<5x8xi32>) -> tensor<5x8x20xf32>
    return %0 : tensor<5x8x20xf32>
  }
}
