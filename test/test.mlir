module {
  func.func @main1(%arg0: tensor<16x16xf32, #nova.device<"1">>, %arg1: tensor<16x16xf32, #nova.device<"1">>) -> tensor<f32, #nova.device<"0">> attributes {llvm.emit_c_interface} {
    %0 = nova.add %arg0, %arg1 : tensor<16x16xf32, #nova.device<"1">>, tensor<16x16xf32, #nova.device<"1">>
    %1 = nova.matmul %0, %arg0 : tensor<16x16xf32, #nova.device<"1">>, tensor<16x16xf32, #nova.device<"1">>
    %2 = nova.mse %1, %1 : tensor<16x16xf32, #nova.device<"1">>, tensor<16x16xf32, #nova.device<"1">>
    %3=nova.to_device %2:tensor<f32, #nova.device<"1">> ->tensor<f32, #nova.device<"0">>
    return %3 : tensor<f32, #nova.device<"0">>
  }
}
