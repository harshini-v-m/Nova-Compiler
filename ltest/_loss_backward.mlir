func.func @test_cce_backward(%grad_y:tensor<4x1xf32>,%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = nova.cce_backward %grad_y,%arg0, %arg1 :tensor<4x1xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}