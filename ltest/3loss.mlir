
func.func @test_bce(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<1xf32> {
  %0 = nova.bce %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
  return %0 : tensor<1xf32>
}

