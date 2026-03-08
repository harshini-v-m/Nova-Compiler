module {
  func.func @unary_exponent_4d(
    %arg0: tensor<2x3x4x8xf32>
  ) -> tensor<2x3x4x8xf32> {
    %exp   = nova.exp   %arg0 : tensor<2x3x4x8xf32>
    %exp2  = nova.exp2  %exp : tensor<2x3x4x8xf32>
    %log   = nova.log   %exp2 : tensor<2x3x4x8xf32>
    %log2  = nova.log2  %log : tensor<2x3x4x8xf32>
    %log10 = nova.log10 %log2 : tensor<2x3x4x8xf32>
    return %log10 : tensor<2x3x4x8xf32>
  }
}
