module{

  // Comparison Operations
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