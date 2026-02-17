module {
  func.func @main(%arg0: memref<128x128xf32, 1>, %arg1: memref<128x128xf32, 1>, %arg2: memref<1xf32, 1>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<128x128xf32, 1> to tensor<128x128xf32, #nova.device<"1">>
    %1 = bufferization.to_tensor %arg1 restrict : memref<128x128xf32, 1> to tensor<128x128xf32, #nova.device<"1">>
    %2 = nova.mse %0, %1 : tensor<128x128xf32, #nova.device<"1">>, tensor<128x128xf32, #nova.device<"1">>
    bufferization.materialize_in_destination %2 in writable %arg2 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    return
  }
}