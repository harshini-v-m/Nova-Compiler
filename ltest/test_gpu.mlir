module {
  func.func @test_cpu(%arg0: tensor<4x8xf32, #nova.device<"1">>) 
  -> tensor<f32, #nova.device<"1">> {
    %min = nova.reduce<mean> %arg0 dimension = [0,1]
      : tensor<4x8xf32, #nova.device<"1">>
    return %min : tensor<f32, #nova.device<"1">>
  }
}
