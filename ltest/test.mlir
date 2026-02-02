module {

  func.func @sce_3d_2d(%logits: tensor<2x4x10xf32,#nova.device<"1">>, %target: tensor<2x4xi32,#nova.device<"1">>) -> tensor<f32,#nova.device<"1">> {
    %0 = nova.sce %logits, %target : tensor<2x4x10xf32,#nova.device<"1">>, tensor<2x4xi32,#nova.device<"1">>
    return %0 : tensor<f32,#nova.device<"1">>
  }

}
