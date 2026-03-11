module {
  func.func @test_layernorm(%input: tensor<2x3xf32>,%grad_y:tensor<2x3xf32>, %gamma: tensor<3xf32>, %beta: tensor<3xf32>) 
  -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>) {

    %0 = nova.layer_norm %input, %gamma, %beta : tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>
    %dx, %dgamma, %dbeta = nova.layer_norm_backward %grad_y, %input, %gamma : tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>
    return %0 ,%dx, %dgamma, %dbeta: tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>
  }
}