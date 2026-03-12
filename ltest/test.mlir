module {
  func.func @main(%arg0: memref<8x1024x1536xf32, 1>, %arg1: memref<8x1024xi32, 1>, %arg2: memref<1xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x1024x1536xf32, 1> to tensor<8x1024x1536xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<8x1024xi32, 1> to tensor<8x1024xi32>
    %2 = nova.sce %0, %1 : tensor<8x1024x1536xf32>, tensor<8x1024xi32>
    bufferization.materialize_in_destination %2 in writable %arg2 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    return
  }
}
