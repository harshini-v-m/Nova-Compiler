func.func @test_mae(%arg0: tensor<4x4xf32, #nova.device<"0">>, %arg1: tensor<4x4xf32, #nova.device<"0">>) -> tensor<1xf32, #nova.device<"0">> {
  %0 = nova.mae %arg0, %arg1 : tensor<4x4xf32, #nova.device<"0">>, tensor<4x4xf32, #nova.device<"0">>
  return %0 : tensor<1xf32, #nova.device<"0">>
}

func.func @test_mse(%arg0: tensor<4x4xf32, #nova.device<"0">>, %arg1: tensor<4x4xf32, #nova.device<"0">>) -> tensor<1xf32, #nova.device<"0">> {
  %0 = nova.mse %arg0, %arg1 : tensor<4x4xf32, #nova.device<"0">>, tensor<4x4xf32, #nova.device<"0">>
  return %0 : tensor<1xf32, #nova.device<"0">>
}

func.func @test_cce(%arg0: tensor<4x4xf32, #nova.device<"0">>, %arg1: tensor<4x4xf32, #nova.device<"0">>) -> tensor<1xf32, #nova.device<"0">> {
  %0 = nova.cce %arg0, %arg1 : tensor<4x4xf32, #nova.device<"0">>, tensor<4x4xf32, #nova.device<"0">>
  return %0 : tensor<1xf32, #nova.device<"0">>
}

func.func @test_bce(%arg0: tensor<4x4xf32, #nova.device<"0">>, %arg1: tensor<4x4xf32, #nova.device<"0">>) -> tensor<1xf32, #nova.device<"0">> {
  %0 = nova.bce %arg0, %arg1 : tensor<4x4xf32, #nova.device<"0">>, tensor<4x4xf32, #nova.device<"0">>
  return %0 : tensor<1xf32, #nova.device<"0">>
}

func.func @test_sce(%arg0: tensor<4x4xf32, #nova.device<"0">>, %arg1: tensor<4xi32, #nova.device<"0">>) -> tensor<1xf32, #nova.device<"0">> {
  %0 = nova.sce %arg0, %arg1 : tensor<4x4xf32, #nova.device<"0">>, tensor<4xi32, #nova.device<"0">>
  return %0 : tensor<1xf32, #nova.device<"0">>
}
