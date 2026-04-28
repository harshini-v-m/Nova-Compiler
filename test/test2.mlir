module {
  func.func @main(%arg0: memref<5120x5120xf32, 1>, %arg1: memref<5120x5120xf32, 1>, %arg2: memref<5120x5120xf32, 1> {bufferization.writable = true}, %arg3: memref<5120x5120xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<5120x5120xf32, 1> to tensor<5120x5120xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<5120x5120xf32, 1> to tensor<5120x5120xf32>
    %2 = nova.matmul %0, %1 : tensor<5120x5120xf32>, tensor<5120x5120xf32>
    %3 = nova.constant {value = dense<1.000000e+00> : tensor<5120x5120xf32>} : tensor<5120x5120xf32>
    %4 = nova.transpose %0 : tensor<5120x5120xf32>
    %5 = nova.transpose %1 : tensor<5120x5120xf32>
    %6 = nova.matmul %3, %5 : tensor<5120x5120xf32>, tensor<5120x5120xf32>
    %7 = nova.matmul %4, %3 : tensor<5120x5120xf32>, tensor<5120x5120xf32>
    %8 = bufferization.to_tensor %arg3 restrict writable : memref<5120x5120xf32, 1> to tensor<5120x5120xf32>
    %9 = nova.add %8, %7 {in_place = true} : tensor<5120x5120xf32>, tensor<5120x5120xf32>
    bufferization.materialize_in_destination %9 in writable %arg3 : (tensor<5120x5120xf32>, memref<5120x5120xf32, 1>) -> ()
    bufferization.materialize_in_destination %2 in writable %arg2 : (tensor<5120x5120xf32>, memref<5120x5120xf32, 1>) -> ()
    return
  }
}
