#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1024x2048xf32>, %arg1: tensor<2048x512xf32>, %arg2: tensor<1024x512xf32>) -> (tensor<1024x512xf32>) attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg1 : tensor<1024x2048xf32>, tensor<2048x512xf32>
    %1 = nova.add %0, %arg2 : tensor<1024x512xf32>, tensor<1024x512xf32>
    %2 = nova.relu %1 : tensor<1024x512xf32>
    return %2: tensor<1024x512xf32>
  }
}