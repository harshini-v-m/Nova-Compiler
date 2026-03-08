module {
  func.func @binary_arithmetic_3d(
    %arg0: tensor<2x4x8xf32>,
    %arg1: tensor<2x4x8xf32>
  ) -> tensor<2x4x8xf32> {
    %add = nova.add %arg0, %arg1 : tensor<2x4x8xf32>, tensor<2x4x8xf32>
    %sub = nova.sub %add, %arg1 : tensor<2x4x8xf32>, tensor<2x4x8xf32>
    %mul = nova.mul %sub, %arg1 : tensor<2x4x8xf32>, tensor<2x4x8xf32>
    %div = nova.div %mul, %arg1 : tensor<2x4x8xf32>, tensor<2x4x8xf32>
    return %div : tensor<2x4x8xf32>
  }
}
