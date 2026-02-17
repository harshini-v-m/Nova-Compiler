module {
  func.func @main(%arg0: memref<8x32x32xf32, 1>, %arg1: memref<8x32x32xf32, 1>, %arg2: memref<1x1xf32, 1>, %arg3: memref<8x32x32xf32, 1>, %arg4: memref<8x32x384xf32, 1>, %arg5: memref<32x384xf32, 1>, %arg6: memref<384xf32, 1>, %arg7: memref<384x32xf32, 1>, %arg8: memref<32xf32, 1>, %arg9: memref<1xf32, 1>, %arg10: memref<32x384xf32, 1>, %arg11: memref<384xf32, 1>, %arg12: memref<384x32xf32, 1>, %arg13: memref<32xf32, 1>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x32x32xf32, 1> to tensor<8x32x32xf32, #nova.device<"1">>
    %1 = bufferization.to_tensor %arg1 restrict : memref<8x32x32xf32, 1> to tensor<8x32x32xf32, #nova.device<"1">>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1x1xf32, 1> to tensor<1x1xf32, #nova.device<"1">>
    %3 = bufferization.to_tensor %arg3 restrict : memref<8x32x32xf32, 1> to tensor<8x32x32xf32, #nova.device<"1">>
    %4 = bufferization.to_tensor %arg4 restrict : memref<8x32x384xf32, 1> to tensor<8x32x384xf32, #nova.device<"1">>
    %5 = bufferization.to_tensor %arg5 restrict : memref<32x384xf32, 1> to tensor<32x384xf32, #nova.device<"1">>
    %6 = bufferization.to_tensor %arg6 restrict : memref<384xf32, 1> to tensor<384xf32, #nova.device<"1">>
    %7 = bufferization.to_tensor %arg7 restrict : memref<384x32xf32, 1> to tensor<384x32xf32, #nova.device<"1">>
    %8 = bufferization.to_tensor %arg8 restrict : memref<32xf32, 1> to tensor<32xf32, #nova.device<"1">>
    %9 = nova.matmul %0, %5 : tensor<8x32x32xf32, #nova.device<"1">>, tensor<32x384xf32, #nova.device<"1">>
    %10 = nova.add %9, %6 : tensor<8x32x384xf32>, tensor<384xf32, #nova.device<"1">>
    %11 = nova.relu %10 : tensor<8x32x384xf32>
    %12 = nova.matmul %11, %7 : tensor<8x32x384xf32>, tensor<384x32xf32, #nova.device<"1">>
    %13 = nova.add %12, %8 : tensor<8x32x32xf32>, tensor<32xf32, #nova.device<"1">>
    %14 = nova.mse %13, %1 : tensor<8x32x32xf32>, tensor<8x32x32xf32, #nova.device<"1">>
    %15 = nova.sub %13, %1 : tensor<8x32x32xf32>, tensor<8x32x32xf32, #nova.device<"1">>
    %16 = nova.mul %15, %3 : tensor<8x32x32xf32>, tensor<8x32x32xf32, #nova.device<"1">>
    %17 = nova.mul %16, %2 : tensor<8x32x32xf32>, tensor<1x1xf32, #nova.device<"1">>
    %18 = nova.reduce<sum> %17 dimension = [0, 1] : tensor<8x32x32xf32>
    %19 = nova.transpose %7 axes1 = 1 axes2 = 0 : tensor<384x32xf32, #nova.device<"1">>
    %20 = nova.matmul %17, %19 : tensor<8x32x32xf32>, tensor<32x384xf32>
    %21 = nova.transpose %11 axes1 = 2 axes2 = 1 : tensor<8x32x384xf32>
    %22 = nova.matmul %21, %17 : tensor<8x384x32xf32>, tensor<8x32x32xf32>
    %23 = nova.reduce<sum> %22 dimension = [0] : tensor<8x384x32xf32>
    %24 = nova.compare<gt> %10, %4 : tensor<8x32x384xf32>, tensor<8x32x384xf32, #nova.device<"1">>
    %25 = nova.cast %24 : tensor<8x32x384xi1> -> tensor<8x32x384xf32>
    %26 = nova.mul %20, %25 : tensor<8x32x384xf32>, tensor<8x32x384xf32>
    %27 = nova.reduce<sum> %26 dimension = [0, 1] : tensor<8x32x384xf32>
    %28 = nova.transpose %0 axes1 = 2 axes2 = 1 : tensor<8x32x32xf32, #nova.device<"1">>
    %29 = nova.matmul %28, %26 : tensor<8x32x32xf32>, tensor<8x32x384xf32>
    %30 = nova.reduce<sum> %29 dimension = [0] : tensor<8x32x384xf32>
    bufferization.materialize_in_destination %14 in writable %arg9 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    bufferization.materialize_in_destination %30 in writable %arg10 : (tensor<32x384xf32>, memref<32x384xf32, 1>) -> ()
    bufferization.materialize_in_destination %27 in writable %arg11 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    bufferization.materialize_in_destination %23 in writable %arg12 : (tensor<384x32xf32>, memref<384x32xf32, 1>) -> ()
    bufferization.materialize_in_destination %18 in writable %arg13 : (tensor<32xf32>, memref<32xf32, 1>) -> ()
    return
  }
}
