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
module {
  // Case 1: Logits 3D, Target 2D
  func.func @sce_3d_2d(%logits: tensor<2x4x10xf32>, %target: tensor<2x4xi32>) -> tensor<f32> {
    %0 = nova.sce %logits, %target : tensor<2x4x10xf32>, tensor<2x4xi32>
    return %0 : tensor<f32>
  }

  // Case 2: Logits 2D, Target 1D
  func.func @sce_2d_1d(%logits: tensor<4x10xf32>, %target: tensor<4xi32>) -> tensor<f32> {
    %0 = nova.sce %logits, %target : tensor<4x10xf32>, tensor<4xi32>
    return %0 : tensor<f32>
  }
}
