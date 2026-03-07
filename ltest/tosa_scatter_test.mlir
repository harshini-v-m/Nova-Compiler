func.func @test_tosa_scatter(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x2xi32>, %arg2: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<1x2x3xf32>, tensor<1x2xi32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}
