module {
  func.func @main(%arg0: memref<8x1024x384xf32, 1>, %arg1: memref<384xf32, 1>, %arg2: memref<384xf32, 1>, %arg3: memref<8x1024x384xf32, 1> {bufferization.writable = true}, %arg4: memref<8x1024x384xf32, 1> {bufferization.writable = true}, %arg5: memref<384xf32, 1> {bufferization.writable = true}, %arg6: memref<384xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x1024x384xf32, 1> to tensor<8x1024x384xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<384xf32, 1> to tensor<384xf32>
    %2 = bufferization.to_tensor %arg2 restrict : memref<384xf32, 1> to tensor<384xf32>
    %3 = nova.layer_norm %0, %1, %2 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %4 = nova.constant {value = dense<1.000000e+00> : tensor<8x1024x384xf32>} : tensor<8x1024x384xf32>
    %grad_x, %grad_gamma, %grad_beta = nova.layer_norm_backward %4, %0, %1 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %3 in writable %arg3 : (tensor<8x1024x384xf32>, memref<8x1024x384xf32, 1>) -> ()
    %5 = bufferization.to_tensor %arg4 restrict writable : memref<8x1024x384xf32, 1> to tensor<8x1024x384xf32>
    %6 = nova.add %5, %grad_x {in_place = true} : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    bufferization.materialize_in_destination %6 in writable %arg4 : (tensor<8x1024x384xf32>, memref<8x1024x384xf32, 1>) -> ()
    %7 = bufferization.to_tensor %arg5 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %8 = nova.add %7, %grad_gamma {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %8 in writable %arg5 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %9 = bufferization.to_tensor %arg6 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %10 = nova.add %9, %grad_beta {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %10 in writable %arg6 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    return
  }
}
