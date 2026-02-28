module {
  func.func @test_gelu(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = nova.gelu %input : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func @test_gelu_backward(%grad_out: tensor<2x3xf32>, %input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = nova.gelu_backward %grad_out, %input : tensor<2x3xf32>, tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
