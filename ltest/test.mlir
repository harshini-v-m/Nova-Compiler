module {
  func.func @main(%arg0: tensor<128x128xf32, #nova.device<"0">>) -> tensor<128xf32> {
    %sum = nova.reduce<sum> %arg0 dimension = [0] : tensor<128x128xf32, #nova.device<"0">>
    %1 = nova.sin %sum : tensor<128xf32>
    return %1 : tensor<128xf32>
  }
}