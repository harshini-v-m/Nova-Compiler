module {
  func.func @reduce_sum_f16(%arg0: tensor<4x8xf16>) -> tensor<8xf16> {
    %sum = nova.reduce<sum> %arg0 dimension = [0] : tensor<4x8xf16>
    return %sum : tensor<8xf16>
  }

  func.func @reduce_sum_bf16(%arg0: tensor<4x8xbf16>) -> tensor<8xbf16> {
    %sum = nova.reduce<sum> %arg0 dimension = [0] : tensor<4x8xbf16>
    return %sum : tensor<8xbf16>
  }

  func.func @reduce_mean_f16(%arg0: tensor<4x8xf16>) -> tensor<8xf16> {
    %mean = nova.reduce<mean> %arg0 dimension = [0] : tensor<4x8xf16>
    return %mean : tensor<8xf16>
  }
}
