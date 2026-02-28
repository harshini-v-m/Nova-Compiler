module {
  func.func @test_layernorm(%input: tensor<2x3xf32>, %gamma: tensor<3xf32>, %beta: tensor<3xf32>) -> tensor<2x3xf32> {
    %0 = nova.layer_norm %input, %gamma, %beta : tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>
    return %0 : tensor<2x3xf32>
  }

  func.func @test_layernorm_backward(%grad_y: tensor<2x3xf32>, %x: tensor<2x3xf32>, %mean: tensor<2x1xf32>, %rstd: tensor<2x1xf32>, %gamma: tensor<3xf32>) -> (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>) {
    %dx, %dgamma, %dbeta = nova.layer_norm_backward %grad_y, %x, %mean, %rstd, %gamma : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<3xf32>
    return %dx, %dgamma, %dbeta : tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>
  }
}
