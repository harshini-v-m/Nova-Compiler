module {
  func.func @main(%arg0: memref<8x1024x384xf32, 1>, %arg1: memref<384xf32, 1>, %arg2: memref<384xf32, 1>, %arg3: memref<8x1024x384xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x1024x384xf32, 1> to tensor<8x1024x384xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<384xf32, 1> to tensor<384xf32>
    %2 = bufferization.to_tensor %arg2 restrict : memref<384xf32, 1> to tensor<384xf32>
    %3 = nova.layer_norm %0, %1, %2 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %3 in writable %arg3 : (tensor<8x1024x384xf32>, memref<8x1024x384xf32, 1>) -> ()
    return
  }
}
