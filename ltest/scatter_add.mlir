

func.func @test_scatter_add_custom_format(%init: tensor<4x4xf32>, %indices: tensor<2xi64>, %src: tensor<2x4xf32>) -> tensor<4x4xf32> {
  %0 = nova.scatter_add %init, %indices, %src axis = 0 : tensor<4x4xf32>, tensor<2xi64>, tensor<2x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
