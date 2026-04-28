module {
  func.func @main(%arg0: memref<8x384x16xf32, 1>, %arg1: memref<1xf32, 1> {bufferization.writable = true}, %arg2: memref<8x384x16xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x384x16xf32, 1> to tensor<8x384x16xf32>
    %1 = nova.reduce<sum> %0 : tensor<8x384x16xf32>
    %2 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32>} : tensor<1xf32>
    %3 = nova.constant {value = dense<1.000000e+00> : tensor<8x384x16xf32>} : tensor<8x384x16xf32>
    %4 = nova.mul %2, %3 : tensor<1xf32>, tensor<8x384x16xf32>
    %5 = bufferization.to_tensor %arg2 restrict writable : memref<8x384x16xf32, 1> to tensor<8x384x16xf32>
    %6 = nova.add %5, %4 {in_place = true} : tensor<8x384x16xf32>, tensor<8x384x16xf32>
    bufferization.materialize_in_destination %6 in writable %arg2 : (tensor<8x384x16xf32>, memref<8x384x16xf32, 1>) -> ()
    bufferization.materialize_in_destination %1 in writable %arg1 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    return
  }
}
