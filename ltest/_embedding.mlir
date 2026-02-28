module {
  func.func @test_embedding_forward(%weight: tensor<10x32xf32>, %indices: tensor<4xi64>) -> tensor<4x32xf32> {
    %0 = nova.gather %weight [%indices] axis = 0 : tensor<10x32xf32>, tensor<4xi64> -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  func.func @test_embedding_backward(%grad_out: tensor<4x32xf32>, %indices: tensor<4xi64>, %weight: tensor<10x32xf32>) -> tensor<10x32xf32> {
    %0 = nova.scatter_add %weight, %indices, %grad_out axis = 0 : tensor<10x32xf32>, tensor<4xi64>, tensor<4x32xf32> -> tensor<10x32xf32>
    return %0 : tensor<10x32xf32>
  }
}
