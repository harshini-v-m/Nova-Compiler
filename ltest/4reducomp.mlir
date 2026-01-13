// Test file for reduction and comparison operations
// This file contains 2 functions demonstrating reduction and comparison operations

module {
  // Function 1: Reduction Operations
  func.func @reduction_ops(
    %arg0: tensor<4x8xf32, #nova.device<"1">>
  ) -> (tensor<8xf32, #nova.device<"1">>, tensor<4x1xf32, #nova.device<"1">>, tensor<f32, #nova.device<"1">>, tensor<4xi32, #nova.device<"1">>, tensor<8xi32, #nova.device<"1">>) {
    
    // Sum reduction along dimension 0 → shape: 8
    %sum_d0 = nova.reduce<sum> %arg0 dimension = [0]
      : tensor<4x8xf32, #nova.device<"1">>
    
    // Max reduction along dimension 1 with keepdims → shape: 4x1
    %max_d1_keep = nova.reduce<max> %arg0 dimension = [-1] keepdims = true
      : tensor<4x8xf32, #nova.device<"1">>
    
    // Mean reduction over all dimensions → scalar (1xf32)
    %mean_all = nova.reduce<mean> %arg0
      : tensor<4x8xf32, #nova.device<"1">>
    
    // Min reduction along dimension 0
    %min_d0 = nova.reduce<min> %arg0 dimension = [0]
      : tensor<4x8xf32, #nova.device<"1">>
    
    %product = nova.reduce<product> %arg0
      : tensor<4x8xf32, #nova.device<"1">>

    %all = nova.reduce<all> %arg0
      : tensor<4x8xf32, #nova.device<"1">>
    
    %any = nova.reduce<any> %arg0
      : tensor<4x8xf32, #nova.device<"1">>

    
    // Argmax along dimension 1 → returns indices
    %argmax_d1 = nova.argmax %arg0 dimension = 1
      : tensor<4x8xf32, #nova.device<"1">>
    
    // Argmin along dimension 0 → returns indices
    %argmin_d0 = nova.argmin %arg0 dimension = 0
      : tensor<4x8xf32, #nova.device<"1">>
    
    return %sum_d0, %max_d1_keep, %mean_all, %argmax_d1, %argmin_d0
      : tensor<8xf32, #nova.device<"1">>, tensor<4x1xf32, #nova.device<"1">>, tensor<f32, #nova.device<"1">>, tensor<4xi32, #nova.device<"1">>, tensor<8xi32, #nova.device<"1">>
  }

  // Function 2: Comparison Operations
  func.func @comparison_ops(
    %arg0: tensor<4x8xf32, #nova.device<"1">>,
    %arg1: tensor<4x8xf32, #nova.device<"1">>
  ) -> (tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>) {
    
    // Equal comparison
    %cmp_eq = nova.compare<eq> %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    // Greater than comparison
    %cmp_gt = nova.compare<gt> %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    // Less than or equal comparison
    %cmp_le = nova.compare<le> %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    // Less than comparison
    %cmp_lt = nova.compare<lt> %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    // Greater than or equal comparison
    %cmp_ge = nova.compare<ge> %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    // Not equal comparison
    %cmp_ne = nova.compare<neq> %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    return %cmp_eq, %cmp_gt, %cmp_le, %cmp_lt, %cmp_ge, %cmp_ne
      : tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xi1, #nova.device<"1">>
  }
}