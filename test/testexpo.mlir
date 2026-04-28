module {
  func.func @main(%arg0: memref<5120x5120xf32, 1>, %arg1: memref<5120x5120xf32, 1>, %arg2: memref<5120x5120xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<5120x5120xf32, 1> to tensor<5120x5120xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<5120x5120xf32, 1> to tensor<5120x5120xf32>
    %2 = nova.matmul %0, %1 : tensor<5120x5120xf32>, tensor<5120x5120xf32>
    bufferization.materialize_in_destination %2 in writable %arg2 : (tensor<5120x5120xf32>, memref<5120x5120xf32, 1>) -> ()
    return
  }
}