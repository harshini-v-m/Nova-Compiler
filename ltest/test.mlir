 module {
   func.func @main(%arg0: tensor<1x4xf32, #nova.device<"1">>) -> tensor<f32, #nova.device<"1">> attributes {llvm.emit_c_interface} {
     %0 = nova.mse %arg0, %arg0 : tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>
    return %0 : tensor<f32, #nova.device<"1">>
  }
 }