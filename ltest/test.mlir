module {
  func.func @partial_reduction_sum(%arg0: tensor<4x8x20xf32>) 
  -> tensor<20xf32> {
    %sum = nova.reduce<mean> %arg0 dimension = [0,1] 
      : tensor<4x8x20xf32>
    return %sum : tensor<20xf32>
  }
}