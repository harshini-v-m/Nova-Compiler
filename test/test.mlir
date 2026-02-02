module {
  func.func @main(%arg0: tensor<1x4xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1x2xf32>, %arg3: tensor<1x1xf32>, %arg4: tensor<1x2xf32>, %arg5: tensor<1x4xf32>, %arg6: tensor<1x4xf32>, %arg7: tensor<1x8xf32>, %arg8: tensor<1x8xf32>, %arg9: tensor<1x8xf32>, %arg10: tensor<1x8xf32>, %arg11: tensor<1x8xf32>, %arg12: tensor<1x8xf32>, %arg13: tensor<1x8xf32>, %arg14: tensor<4x8xf32>, %arg15: tensor<1x8xf32>, %arg16: tensor<8x4xf32>, %arg17: tensor<1x4xf32>, %arg18: tensor<4x2xf32>, %arg19: tensor<1x2xf32>) -> (tensor<f32>, tensor<1x1xf32>, tensor<4x8xf32>, tensor<1x8xf32>, tensor<8x4xf32>, tensor<1x4xf32>, tensor<4x2xf32>, tensor<1x2xf32>) attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg14 : tensor<1x4xf32>, tensor<4x8xf32>
    %1 = nova.add %0, %arg15 : tensor<1x8xf32>, tensor<1x8xf32>
    %2 = nova.gelu %1 : tensor<1x8xf32>
    %3 = nova.matmul %2, %arg16 : tensor<1x8xf32>, tensor<8x4xf32>
    %4 = nova.add %3, %arg17 : tensor<1x4xf32>, tensor<1x4xf32>
    %5 = nova.relu %4 : tensor<1x4xf32>
    %6 = nova.matmul %5, %arg18 : tensor<1x4xf32>, tensor<4x2xf32>
    %7 = nova.add %6, %arg19 : tensor<1x2xf32>, tensor<1x2xf32>
    %8 = nova.sigmoid %7 : tensor<1x2xf32>
    %9 = nova.sce %8, %arg1 : tensor<1x2xf32>, tensor<1xi32>
    %10 = nova.softmax %8 dimension = 1 : tensor<1x2xf32>
    %11 = nova.constant {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
    %cst = arith.constant dense<[1, 2]> : tensor<2xindex>
    %reshape = tensor.reshape %11(%cst) : (tensor<2xf32>, tensor<2xindex>) -> tensor<1x2xf32>
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
    %12 = nova.broadcast_in_dim %expanded, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x2xi32>
    %13 = nova.broadcast_in_dim %reshape, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %14 = nova.compare<eq> %12, %13 : tensor<1x2xi32>, tensor<1x2xf32>
    %15 = tosa.cast %14 : (tensor<1x2xi1>) -> tensor<1x2xf32>
    %16 = nova.sub %10, %15 : tensor<1x2xf32>, tensor<1x2xf32>
    %17 = nova.mul %16, %arg3 : tensor<1x2xf32>, tensor<1x1xf32>
    %18 = nova.sub %arg4, %8 : tensor<1x2xf32>, tensor<1x2xf32>
    %19 = nova.mul %8, %18 : tensor<1x2xf32>, tensor<1x2xf32>
    %20 = nova.mul %17, %19 : tensor<1x2xf32>, tensor<1x2xf32>
    %21 = nova.transpose %arg18 axes1 = 0 axes2 = 1 : tensor<4x2xf32>
    %22 = nova.matmul %20, %21 : tensor<1x2xf32>, tensor<2x4xf32>
    %23 = nova.transpose %5 axes1 = 0 axes2 = 1 : tensor<1x4xf32>
    %24 = nova.matmul %23, %20 : tensor<4x1xf32>, tensor<1x2xf32>
    %25 = nova.mul %4, %arg5 : tensor<1x4xf32>, tensor<1x4xf32>
    %26 = nova.relu %25 : tensor<1x4xf32>
    %27 = nova.min %26, %arg6 : tensor<1x4xf32>, tensor<1x4xf32>
    %28 = nova.mul %22, %27 : tensor<1x4xf32>, tensor<1x4xf32>
    %29 = nova.transpose %arg16 axes1 = 0 axes2 = 1 : tensor<8x4xf32>
    %30 = nova.matmul %28, %29 : tensor<1x4xf32>, tensor<4x8xf32>
    %31 = nova.transpose %2 axes1 = 0 axes2 = 1 : tensor<1x8xf32>
    %32 = nova.matmul %31, %28 : tensor<8x1xf32>, tensor<1x4xf32>
    %33 = nova.mul %1, %1 : tensor<1x8xf32>, tensor<1x8xf32>
    %34 = nova.mul %33, %1 : tensor<1x8xf32>, tensor<1x8xf32>
    %35 = nova.mul %34, %arg7 : tensor<1x8xf32>, tensor<1x8xf32>
    %36 = nova.add %1, %35 : tensor<1x8xf32>, tensor<1x8xf32>
    %37 = nova.mul %36, %arg8 : tensor<1x8xf32>, tensor<1x8xf32>
    %38 = nova.tanh %37 : tensor<1x8xf32>
    %39 = nova.add %arg9, %38 : tensor<1x8xf32>, tensor<1x8xf32>
    %40 = nova.mul %39, %arg10 : tensor<1x8xf32>, tensor<1x8xf32>
    %41 = nova.mul %38, %38 : tensor<1x8xf32>, tensor<1x8xf32>
    %42 = nova.sub %arg9, %41 : tensor<1x8xf32>, tensor<1x8xf32>
    %43 = nova.mul %33, %arg11 : tensor<1x8xf32>, tensor<1x8xf32>
    %44 = nova.add %arg9, %43 : tensor<1x8xf32>, tensor<1x8xf32>
    %45 = nova.mul %44, %arg12 : tensor<1x8xf32>, tensor<1x8xf32>
    %46 = nova.mul %1, %42 : tensor<1x8xf32>, tensor<1x8xf32>
    %47 = nova.mul %46, %45 : tensor<1x8xf32>, tensor<1x8xf32>
    %48 = nova.mul %47, %arg13 : tensor<1x8xf32>, tensor<1x8xf32>
    %49 = nova.add %40, %48 : tensor<1x8xf32>, tensor<1x8xf32>
    %50 = nova.mul %30, %49 : tensor<1x8xf32>, tensor<1x8xf32>
    %51 = nova.transpose %arg0 axes1 = 0 axes2 = 1 : tensor<1x4xf32>
    %52 = nova.matmul %51, %50 : tensor<4x1xf32>, tensor<1x8xf32>
    return %9, %arg3, %52, %50, %32, %28, %24, %20 : tensor<f32>, tensor<1x1xf32>, tensor<4x8xf32>, tensor<1x8xf32>, tensor<8x4xf32>, tensor<1x4xf32>, tensor<4x2xf32>, tensor<1x2xf32>
  }
}