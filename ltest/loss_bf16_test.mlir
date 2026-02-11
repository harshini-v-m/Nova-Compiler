module {
  func.func @test_mae_bf16(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<1xbf16> {
    %0 = nova.mae %arg0, %arg1 : tensor<2x2xbf16>, tensor<2x2xbf16>
    return %0 : tensor<1xbf16>
  }

  func.func @test_mse_bf16(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<1xbf16> {
    %0 = nova.mse %arg0, %arg1 : tensor<2x2xbf16>, tensor<2x2xbf16>
    return %0 : tensor<1xbf16>
  }

  func.func @test_bce_bf16(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<1xbf16> {
    %0 = nova.bce %arg0, %arg1 : tensor<2x2xbf16>, tensor<2x2xbf16>
    return %0 : tensor<1xbf16>
  }

  func.func @test_cce_bf16(%arg0: tensor<2x10xbf16>, %arg1: tensor<2x10xbf16>) -> tensor<1xbf16> {
    %0 = nova.cce %arg0, %arg1 : tensor<2x10xbf16>, tensor<2x10xbf16>
    return %0 : tensor<1xbf16>
  }

}
