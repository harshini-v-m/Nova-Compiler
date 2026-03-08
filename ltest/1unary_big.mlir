module {
  func.func @unary_exponent(
    %arg0: tensor<128x256xf32>
  ) -> tensor<128x256xf32> {
    %exp   = nova.exp   %arg0 : tensor<128x256xf32>
    %exp2  = nova.exp2  %exp : tensor<128x256xf32>
    %log   = nova.log   %exp2 : tensor<128x256xf32>
    %log2  = nova.log2  %log : tensor<128x256xf32>
    %log10 = nova.log10 %log2 : tensor<128x256xf32>
    return %log10 : tensor<128x256xf32>
  }
}
