// Test file for loss functions
// This file demonstrates all loss operations in nova dialect

module {
  func.func @test_mae(
    %predictions: tensor<8x8xf32, #nova.device<"1">>,
    %targets: tensor<8x8xf32, #nova.device<"1">>
  ) -> (tensor<f32, #nova.device<"1">>) {
    %mae = nova.mae %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    return %mae : tensor<f32, #nova.device<"1">>
  }

  func.func @test_mse(
    %predictions: tensor<8x8xf32, #nova.device<"1">>,
    %targets: tensor<8x8xf32, #nova.device<"1">>
  ) -> (tensor<f32, #nova.device<"1">>) {
    %mse = nova.mse %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    return %mse : tensor<f32, #nova.device<"1">>
  }

  func.func @test_bce(
    %predictions: tensor<8x8xf32, #nova.device<"1">>,
    %targets: tensor<8x8xf32, #nova.device<"1">>
  ) -> (tensor<f32, #nova.device<"1">>) {
    %bce = nova.bce %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    return %bce : tensor<f32, #nova.device<"1">>
  }

  func.func @test_cce(
    %predictions: tensor<8x8xf32, #nova.device<"1">>,
    %targets: tensor<8x8xf32, #nova.device<"1">>
  ) -> (tensor<f32, #nova.device<"1">>) {
    %cce = nova.cce %predictions, %targets 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    return %cce : tensor<f32, #nova.device<"1">>
  }

  func.func @test_sce(
    %predictions: tensor<8x8xf32, #nova.device<"1">>,
    %target_indices: tensor<8xi32, #nova.device<"1">>
  ) -> (tensor<f32, #nova.device<"1">>) {
    %sce = nova.sce %predictions, %target_indices 
      : tensor<8x8xf32, #nova.device<"1">>, tensor<8xi32, #nova.device<"1">>
    return %sce : tensor<f32, #nova.device<"1">>
  }
}

