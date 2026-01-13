module {
  func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<8x10xf32>, %arg2: tensor<16x10xf32>, %arg3: tensor<1x10xf32>) -> (tensor<f32>, tensor<16x10xf32>, tensor<1x10xf32>) attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg2 : tensor<8x16xf32>, tensor<16x10xf32>
    %1 = nova.add %0, %arg3 : tensor<8x10xf32>, tensor<1x10xf32>
    %2 = nova.mse %1, %arg1 : tensor<8x10xf32>, tensor<8x10xf32>
    %3 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32>} : tensor<1xf32>
    %4 = nova.constant {value = dense<2.500000e-02> : tensor<1xf32>} : tensor<1xf32>
    %5 = nova.sub %1, %arg1 : tensor<8x10xf32>, tensor<8x10xf32>
    %6 = nova.mul %5, %4 : tensor<8x10xf32>, tensor<1xf32>
    %7 = nova.mul %6, %3 : tensor<8x10xf32>, tensor<1xf32>
    %8 = nova.transpose %7 : tensor<8x10xf32>
    %9 = nova.reduce<sum> %8 dimension = [1] keepdims = true : tensor<10x8xf32>
    %10 = nova.transpose %9 : tensor<10x1xf32>
    %11 = nova.transpose %arg0 : tensor<8x16xf32>
    %12 = nova.matmul %11, %7 : tensor<16x8xf32>, tensor<8x10xf32>
    return %2, %12, %10 : tensor<f32>, tensor<16x10xf32>, tensor<1x10xf32>
  }
}