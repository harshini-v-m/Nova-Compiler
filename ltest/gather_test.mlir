module {
  // Test gather with axis 0
  func.func @test_gather_axis0(%arg0: tensor<10x20xf32>, %arg1: tensor<5xi32>) -> tensor<5x20xf32> {
    %0 = nova.gather %arg0[%arg1] axis = 0 : tensor<10x20xf32>, tensor<5xi32> -> tensor<5x20xf32>
    return %0 : tensor<5x20xf32>
  }

  // Test gather with axis 1
  func.func @test_gather_axis1(%arg0: tensor<10x20xf32>, %arg1: tensor<5xi32>) -> tensor<10x5xf32> {
    %0 = nova.gather %arg0[%arg1] axis = 1 : tensor<10x20xf32>, tensor<5xi32> -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
  }

  // Test gather with multidimensional indices
  func.func @test_gather_multi_indices(%arg0: tensor<10x20x30xf32>, %arg1: tensor<2x4xi32>) -> tensor<10x2x4x30xf32> {
    %0 = nova.gather %arg0[%arg1] axis = 1 : tensor<10x20x30xf32>, tensor<2x4xi32> -> tensor<10x2x4x30xf32>
    return %0 : tensor<10x2x4x30xf32>
  }

  // Test gather with default axis (axis 0)
  func.func @test_gather_default_axis(%arg0: tensor<10x20xf32>, %arg1: tensor<5xi32>) -> tensor<5x20xf32> {
    %0 = nova.gather %arg0[%arg1] : tensor<10x20xf32>, tensor<5xi32> -> tensor<5x20xf32>
    return %0 : tensor<5x20xf32>
  }

  // Test scatter_add with axis 0
  func.func @test_scatter_add_axis0(%arg0: tensor<10x20xf32>, %arg1: tensor<5xi32>, %arg2: tensor<5x20xf32>) -> tensor<10x20xf32> {
    %0 = nova.scatter_add %arg0, %arg1, %arg2 axis = 0 : tensor<10x20xf32>, tensor<5xi32>, tensor<5x20xf32> -> tensor<10x20xf32>
    return %0 : tensor<10x20xf32>
  }

  // Test scatter_add with axis 1
  func.func @test_scatter_add_axis1(%arg0: tensor<10x20xf32>, %arg1: tensor<5xi32>, %arg2: tensor<10x5xf32>) -> tensor<10x20xf32> {
    %0 = nova.scatter_add %arg0, %arg1, %arg2 axis = 1 : tensor<10x20xf32>, tensor<5xi32>, tensor<10x5xf32> -> tensor<10x20xf32>
    return %0 : tensor<10x20xf32>
  }
}
