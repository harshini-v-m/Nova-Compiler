
func.func @test_sce(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<1xf32> {
  %0 = nova.sce %arg0, %arg1 : tensor<4x4xf32>, tensor<4xi32>
  return %0 : tensor<1xf32>
}
