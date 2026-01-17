module {
  func.func @main(%arg0: tensor<8x16xf32>) -> (tensor<f32>, tensor<8x16xf32>) attributes {llvm.emit_c_interface} {
    %0 = nova.reduce<sum> %arg0 dimension = [1] keepdims = true : tensor<8x16xf32>
    %1 = nova.reduce<mean> %0 : tensor<8x1xf32>
    %2 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32>} : tensor<1xf32>
    %3 = nova.constant {value = dense<1.250000e-01> : tensor<1xf32>} : tensor<1xf32>
    %4 = nova.mul %2, %3 : tensor<1xf32>, tensor<1xf32>
    %5 = nova.constant {value = dense<1.000000e+00> : tensor<8x1xf32>} : tensor<8x1xf32>
    %6 = nova.mul %4, %5 : tensor<1xf32>, tensor<8x1xf32>
    %7 = nova.constant {value = dense<1.000000e+00> : tensor<8x16xf32>} : tensor<8x16xf32>
    %8 = nova.mul %6, %7 : tensor<8x1xf32>, tensor<8x16xf32>
    return %1, %8 : tensor<f32>, tensor<8x16xf32>
  }
}