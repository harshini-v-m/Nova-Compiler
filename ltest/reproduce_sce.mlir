module {
  func.func @test_sce(%logits: tensor<4x10xf32>, %targets: tensor<4xi32>) -> tensor<1xf32> {
    %loss = nova.sce %logits, %targets : tensor<4x10xf32>, tensor<4xi32>
    return %loss : tensor<1xf32>
  }
}
