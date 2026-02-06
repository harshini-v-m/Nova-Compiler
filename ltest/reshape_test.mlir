func.func @test_reshape(%arg0: tensor<4x4xf32, #nova.device<"1">>) -> tensor<2x8xf32, #nova.device<"1">> {
  %0 = nova.reshape %arg0 : tensor<4x4xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
