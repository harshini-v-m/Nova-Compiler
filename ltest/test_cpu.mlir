module {
  func.func @main(%arg0: tensor<8x16xf32, #nova.device<"1">>, %arg1: tensor<16x10xf32, #nova.device<"1">>, %arg2: tensor<1x10xf32, #nova.device<"1">>, %arg3: tensor<8x10xf32, #nova.device<"1">>) -> tensor<8x10xf32, #nova.device<"1">> {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32, #nova.device<"1">>, tensor<16x10xf32, #nova.device<"1">>
    %1 = nova.add %0, %arg2 : tensor<8x10xf32, #nova.device<"1">>, tensor<1x10xf32, #nova.device<"1">>
    return %1 : tensor<8x10xf32, #nova.device<"1">>
  }
}