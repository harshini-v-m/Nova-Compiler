module {
  // Sum reduction along dimension 0
  func.func @reduce_sum(
    %arg0: tensor<4x8xf32>
  ) -> tensor<8xf32> {
    %sum_d0 = nova.reduce<sum> %arg0 dimension = [0]
      : tensor<4x8xf32>
    return %sum_d0 : tensor<8xf32>
  }

  // Max reduction along dimension -1 with keepdims
  func.func @reduce_max_keepdims(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4x1xf32> {
    %max_d1_keep = nova.reduce<max> %arg0 dimension = [-1] keepdims = true
      : tensor<4x8xf32>
    return %max_d1_keep : tensor<4x1xf32>
  }

  // Mean reduction over all dimensions (COMMENTED OUT DUE TO LEGALIZATION ISSUE)
   func.func @reduce_mean_all(
     %arg0: tensor<4x8xf32>
   ) -> tensor<1xf32> {
     %mean_all = nova.reduce<mean> %arg0 dimension = [0, 1]
       : tensor<4x8xf32>
     return %mean_all : tensor<1xf32>
   }

  // Min reduction along dimension 0
  func.func @reduce_min(
    %arg0: tensor<4x8xf32>
  ) -> tensor<8xf32> {
    %min_d0 = nova.reduce<min> %arg0 dimension = [0]
      : tensor<4x8xf32>
    return %min_d0 : tensor<8xf32>
  }

  // Product reduction
  func.func @reduce_product(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4xf32> {
    %product = nova.reduce<product> %arg0 dimension=[1]
      : tensor<4x8xf32>
    return %product : tensor<4xf32>
  }

  // All reduction
  func.func @reduce_all(
    %arg0: tensor<4x8xf32>
  ) -> tensor<1xi1> {
    %all = nova.reduce<all> %arg0
      : tensor<4x8xf32>
    return %all : tensor<1xi1>
  }

  // Any reduction
  func.func @reduce_any(
    %arg0: tensor<4x8xf32>
  ) -> tensor<1xi1> {
    %any = nova.reduce<any> %arg0
      : tensor<4x8xf32>
    return %any : tensor<1xi1>
  }

  // Argmax along dimension 1
  func.func @argmax_op(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4xi32> {
    %argmax_d1 = nova.argmax %arg0 dimension = 1
      : tensor<4x8xf32>
    return %argmax_d1 : tensor<4xi32>
  }

  // Argmin along dimension 0
  func.func @argmin_op(
    %arg0: tensor<4x8xf32>
  ) -> tensor<8xi32> {
    %argmin_d0 = nova.argmin %arg0 dimension = 0
      : tensor<4x8xf32>
    return %argmin_d0 : tensor<8xi32>
  }
}
