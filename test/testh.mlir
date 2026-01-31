
func.func @test_nova_fusion(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {

  %c0 = nova.transpose %arg0 axes1 = 0 axes2 = 1  : tensor<4x8xf32>


  %result = nova.add %c0, %arg1 : tensor<8x4xf32>, tensor<8x4xf32> 

  return %result : tensor<8x4xf32>
}