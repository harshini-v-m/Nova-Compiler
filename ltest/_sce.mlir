module {
  func.func @test_sce(%logits: tensor<4x10xf32>, %targets: tensor<4xi32>) -> tensor<1xf32> {
    %0 = nova.sce %logits, %targets : tensor<4x10xf32>, tensor<4xi32>
    return %0 : tensor<1xf32>
  }

  func.func @test_sce_backward(%logits: tensor<4x10xf32>, %targets: tensor<4xi32>) -> tensor<4x10xf32> {
    %0 = nova.sce_backward %logits, %targets {dim = -1 : i64} : tensor<4x10xf32>, tensor<4xi32>
    return %0 : tensor<4x10xf32>
  }
}
