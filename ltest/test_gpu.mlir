module {
  func.func @main(%arg0: memref<1024x1536xf32, 1>, %arg1: memref<1536x5034xf32, 1>, %arg2: memref<5034xf32, 1>, %arg3: memref<1024x5034xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1536xf32, 1> to tensor<1024x1536xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1536x5034xf32, 1> to tensor<1536x5034xf32>
    %2 = bufferization.to_tensor %arg2 restrict : memref<5034xf32, 1> to tensor<5034xf32>
    %3 = nova.linear %0, %1, %2 : tensor<1024x1536xf32>, tensor<1536x5034xf32>, tensor<5034xf32>
    bufferization.materialize_in_destination %3 in writable %arg3 : (tensor<1024x5034xf32>, memref<1024x5034xf32, 1>) -> ()
    return
  }
}