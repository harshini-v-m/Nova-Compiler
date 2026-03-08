module {
  func.func @test_layernorm(%input: tensor<128x256xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<128x256xf32> {
    %0 = nova.layer_norm %input, %gamma, %beta : tensor<128x256xf32>, tensor<256xf32>, tensor<256xf32>
    return %0 : tensor<128x256xf32>
  }

  func.func @test_layernorm_backward(%grad_y: tensor<128x256xf32>, %x: tensor<128x256xf32>, %mean: tensor<128x1xf32>, %rstd: tensor<128x1xf32>, %gamma: tensor<256xf32>) -> (tensor<128x256xf32>, tensor<256xf32>, tensor<256xf32>) {
    %dx, %dgamma, %dbeta = nova.layer_norm_backward %grad_y, %x, %mean, %rstd, %gamma : tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<256xf32>
    return %dx, %dgamma, %dbeta : tensor<128x256xf32>, tensor<256xf32>, tensor<256xf32>
  }
}
