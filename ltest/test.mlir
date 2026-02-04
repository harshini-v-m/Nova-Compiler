module {
  func.func @main(%arg0: tensor<1x4xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1x2xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<1x8xf32>, %arg5: tensor<8x4xf32>, %arg6: tensor<1x4xf32>, %arg7: tensor<4x2xf32>, %arg8: tensor<1x2xf32>) -> tensor<1xf32> attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg3 : tensor<1x4xf32>, tensor<4x8xf32>
    %1 = nova.add %0, %arg4 : tensor<1x8xf32>, tensor<1x8xf32>
    %2 = nova.gelu %1 : tensor<1x8xf32>
    %3 = nova.matmul %2, %arg5 : tensor<1x8xf32>, tensor<8x4xf32>
    %4 = nova.add %3, %arg6 : tensor<1x4xf32>, tensor<1x4xf32>
    %5 = nova.relu %4 : tensor<1x4xf32>
    %6 = nova.matmul %5, %arg7 : tensor<1x4xf32>, tensor<4x2xf32>
    %7 = nova.add %6, %arg8 : tensor<1x2xf32>, tensor<1x2xf32>
    %8 = nova.sigmoid %7 : tensor<1x2xf32>
    %9 = nova.sce %8, %arg1 : tensor<1x2xf32>, tensor<1xi32>
    return %9 : tensor<1xf32>
  }
}