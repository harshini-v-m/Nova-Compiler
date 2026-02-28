module {
  func.func @test_linear(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 =  nova.linear %arg0, %arg1, %arg2 : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func @test_linear_backward(%grad_out: tensor<2x4xf32>, %input: tensor<2x3xf32>, %weight: tensor<3x4xf32>) -> (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) {
    %grad_input, %grad_weight, %grad_bias = nova.linear_backward %grad_out, %input, %weight : tensor<2x4xf32>, tensor<2x3xf32>, tensor<3x4xf32>
    return %grad_input, %grad_weight, %grad_bias : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
  }
}
