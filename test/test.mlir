module {
  func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x64xf32>, %arg2: tensor<1x64xf32>) -> (tensor<8x64xf32>) attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32>, tensor<16x64xf32>
    %1 = nova.add %0, %arg2 : tensor<8x64xf32>, tensor<1x64xf32>
    %2 = nova.relu %1 : tensor<8x64xf32>
    return %2: tensor<8x64xf32>
  }
}