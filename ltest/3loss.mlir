// Test file for loss functions
// This file demonstrates all loss operations in nova dialect

module {
  func.func @loss_functions(
    %predictions: tensor<8x8xf32, #nova.device<"1">>,
    %targets: tensor<8x8xf32, #nova.device<"1">>,
    %target_indices: tensor<8xi32, #nova.device<"1">>
  ) -> (tensor<f32, #nova.device<"1">>) {
    
    // Mean Absolute Error
    %mae = nova.mae %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    
    // Mean Squared Error
    %mse = nova.mse %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    
    // Binary Cross Entropy
    %bce = nova.bce %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    
    // Categorical Cross Entropy
    %cce = nova.cce %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    //Sparse Cross Entropy
    %sce = nova.sce %predictions, %target_indices 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8xi32, #nova.device<"1">>
    return  %sce
      : tensor<f32, #nova.device<"1">>
  }
}

