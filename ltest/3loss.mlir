func.func @test_mae(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<1xf32> {
  %0 = nova.mae %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>
  return %0 : tensor<1xf32>
}

func.func @test_mse(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<1xf32> {
  %0 = nova.mse %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>
  return %0 : tensor<1xf32>
}

func.func @test_cce(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<1xf32> {
  %0 = nova.cce %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>
  return %0 : tensor<1xf32>
}

func.func @test_bce(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<1xf32> {
  %0 = nova.bce %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>
  return %0 : tensor<1xf32>
}

func.func @test_sce(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<1xf32> {
  %0 = nova.sce %arg0, %arg1 : tensor<4x4xf32>, tensor<4xi32>
  return %0 : tensor<1xf32>
}
