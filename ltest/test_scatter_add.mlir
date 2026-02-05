module {
  func.func @test_scatter_add(%input: tensor<10x20xf32>, %indices: tensor<5xi32>, %src: tensor<5x20xf32>) -> tensor<10x20xf32> {
    %0 = "nova.scatter_add"(%input, %indices, %src) {axis = 0 : i64} : (tensor<10x20xf32>, tensor<5xi32>, tensor<5x20xf32>) -> tensor<10x20xf32>
    return %0 : tensor<10x20xf32>
  }
}
