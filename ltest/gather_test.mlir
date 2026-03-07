func.func @test_gather(%arg0: tensor<4x4xf32>, %arg1: tensor<2xi64>) -> tensor<2x4xf32> {
  %0 = nova.gather %arg0 [%arg1] axis = 0 : tensor<4x4xf32>, tensor<2xi64> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
