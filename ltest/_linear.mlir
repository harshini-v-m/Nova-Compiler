module {
  func.func @test_linear_backward(%grad_out: tensor<512x1024xf32>, %input: tensor<512x1024xf32>, %weight: tensor<1024x1024xf32>) -> (tensor<512x1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>) {
    %grad_input, %grad_weight, %grad_bias = nova.linear_backward %grad_out, %input, %weight : tensor<512x1024xf32>, tensor<512x1024xf32>, tensor<1024x1024xf32>
    return %grad_input, %grad_weight, %grad_bias : tensor<512x1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>
  }
}
