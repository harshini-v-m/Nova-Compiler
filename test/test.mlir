module {
  func.func @main(%arg0: tensor<80x16xf32>, %arg1: tensor<16x1000xf32>, %arg2: tensor<1x1000xf32>, %arg3: tensor<80xi32>) -> tensor<f32> {
    %0 = nova.matmul %arg0, %arg1 : tensor<80x16xf32>, tensor<16x1000xf32>
    %1 = nova.sub %0, %arg2 : tensor<80x1000xf32>, tensor<1x1000xf32>
    %2 = nova.sce %1, %arg3 : tensor<80x1000xf32>, tensor<80xi32>
    return %2 : tensor<f32>
  }
}