module {
  func.func @main(%arg0: memref<1x10xf32, 1>, %arg1: memref<1x4xf32, 1>, %arg2: memref<1x6xf32, 1>,
   %arg3: memref<1x1xf32, 1>, %arg4: memref<1x4xf32, 1>,
    %arg5: memref<1x4xf32, 1>, %arg6: memref<1x6xf32, 1>,
     %arg7: memref<1x8xf32, 1>, %arg8: memref<1x8xf32, 1>,
      %arg9: memref<10x8xf32>, %arg10: memref<1x8xf32, 1>,
       %arg11: memref<8x6xf32, 1>, %arg12: memref<1x6xf32, 1>,
        %arg13: memref<6x4xf32, 1>, %arg14: memref<1x4xf32, 1>,
         %arg15: memref<1xf32, 1>, %arg16: memref<1x1xf32, 1>, 
         %arg17: memref<10x8xf32, 1>, %arg18: memref<1x8xf32, 1>,
          %arg19: memref<8x6xf32, 1>, %arg20: memref<1x6xf32, 1>, 
          %arg21: memref<6x4xf32, 1>, %arg22: memref<1x4xf32, 1>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1x10xf32, 1> to tensor<1x10xf32, #nova.device<"1">>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1x4xf32, 1> to tensor<1x4xf32, #nova.device<"1">>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1x6xf32, 1> to tensor<1x6xf32, #nova.device<"1">>
    %3 = bufferization.to_tensor %arg3 restrict : memref<1x1xf32, 1> to tensor<1x1xf32, #nova.device<"1">>
    %4 = bufferization.to_tensor %arg4 restrict : memref<1x4xf32, 1> to tensor<1x4xf32, #nova.device<"1">>
    %5 = bufferization.to_tensor %arg5 restrict : memref<1x4xf32, 1> to tensor<1x4xf32, #nova.device<"1">>
    %6 = bufferization.to_tensor %arg6 restrict : memref<1x6xf32, 1> to tensor<1x6xf32, #nova.device<"1">>
    %7 = bufferization.to_tensor %arg7 restrict : memref<1x8xf32, 1> to tensor<1x8xf32, #nova.device<"1">>
    %8 = bufferization.to_tensor %arg8 restrict : memref<1x8xf32, 1> to tensor<1x8xf32, #nova.device<"1">>
    %9 = bufferization.to_tensor %arg9 restrict : memref<10x8xf32> to tensor<10x8xf32>
    %10 = bufferization.to_tensor %arg10 restrict : memref<1x8xf32, 1> to tensor<1x8xf32, #nova.device<"1">>
    %11 = bufferization.to_tensor %arg11 restrict : memref<8x6xf32, 1> to tensor<8x6xf32, #nova.device<"1">>
    %12 = bufferization.to_tensor %arg12 restrict : memref<1x6xf32, 1> to tensor<1x6xf32, #nova.device<"1">>
    %13 = bufferization.to_tensor %arg13 restrict : memref<6x4xf32, 1> to tensor<6x4xf32, #nova.device<"1">>
    %14 = bufferization.to_tensor %arg14 restrict : memref<1x4xf32, 1> to tensor<1x4xf32, #nova.device<"1">>
    %15 = nova.matmul %0, %9 : tensor<1x10xf32, #nova.device<"1">>, tensor<10x8xf32>
    %16 = nova.add %15, %10 : tensor<1x8xf32>, tensor<1x8xf32, #nova.device<"1">>
    %17 = nova.relu %16 : tensor<1x8xf32>
    %18 = nova.matmul %17, %11 : tensor<1x8xf32>, tensor<8x6xf32, #nova.device<"1">>
    %19 = nova.add %18, %12 : tensor<1x6xf32>, tensor<1x6xf32, #nova.device<"1">>
    %20 = nova.sigmoid %19 : tensor<1x6xf32>
    %21 = nova.matmul %20, %13 : tensor<1x6xf32>, tensor<6x4xf32, #nova.device<"1">>
    %22 = nova.add %21, %14 : tensor<1x4xf32>, tensor<1x4xf32, #nova.device<"1">>
    %23 = nova.tanh %22 : tensor<1x4xf32>
    %24 = nova.mse %23, %1 : tensor<1x4xf32>, tensor<1x4xf32, #nova.device<"1">>
    %25 = nova.sub %23, %1 : tensor<1x4xf32>, tensor<1x4xf32, #nova.device<"1">>
    %26 = nova.mul %25, %4 : tensor<1x4xf32>, tensor<1x4xf32, #nova.device<"1">>
    %27 = nova.mul %26, %3 : tensor<1x4xf32>, tensor<1x1xf32, #nova.device<"1">>
    %28 = nova.tanh %22 : tensor<1x4xf32>
    %29 = nova.mul %28, %28 : tensor<1x4xf32>, tensor<1x4xf32>
    %30 = nova.sub %5, %29 : tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32>
    %31 = nova.mul %27, %30 : tensor<1x4xf32>, tensor<1x4xf32>
    %32 = nova.transpose %13 axes1 = 0 axes2 = 1 : tensor<6x4xf32, #nova.device<"1">>
    %33 = nova.matmul %31, %32 : tensor<1x4xf32>, tensor<4x6xf32>
    %34 = nova.transpose %20 axes1 = 0 axes2 = 1 : tensor<1x6xf32>
    %35 = nova.matmul %34, %31 : tensor<6x1xf32>, tensor<1x4xf32>
    %36 = nova.sub %6, %20 : tensor<1x6xf32, #nova.device<"1">>, tensor<1x6xf32>
    %37 = nova.mul %20, %36 : tensor<1x6xf32>, tensor<1x6xf32>
    %38 = nova.mul %33, %37 : tensor<1x6xf32>, tensor<1x6xf32>
    %39 = nova.transpose %11 axes1 = 0 axes2 = 1 : tensor<8x6xf32, #nova.device<"1">>
    %40 = nova.matmul %38, %39 : tensor<1x6xf32>, tensor<6x8xf32>
    %41 = nova.transpose %17 axes1 = 0 axes2 = 1 : tensor<1x8xf32>
    %42 = nova.matmul %41, %38 : tensor<8x1xf32>, tensor<1x6xf32>
    %43 = nova.mul %16, %7 : tensor<1x8xf32>, tensor<1x8xf32, #nova.device<"1">>
    %44 = nova.relu %43 : tensor<1x8xf32>
    %45 = nova.min %44, %8 : tensor<1x8xf32>, tensor<1x8xf32,#nova.device<"1">>
    %46 = nova.mul %40, %45 : tensor<1x8xf32>, tensor<1x8xf32>
    %47 = nova.transpose %0 axes1 = 0 axes2 = 1 : tensor<1x10xf32,#nova.device<"1">>
    %48 = nova.matmul %47, %46 : tensor<10x1xf32>, tensor<1x8xf32>
    bufferization.materialize_in_destination %24 in writable %arg15 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    bufferization.materialize_in_destination %3 in writable %arg16 : (tensor<1x1xf32,#nova.device<"1">>, memref<1x1xf32, 1>) -> ()
    bufferization.materialize_in_destination %48 in writable %arg17 : (tensor<10x8xf32>, memref<10x8xf32, 1>) -> ()
    bufferization.materialize_in_destination %46 in writable %arg18 : (tensor<1x8xf32>, memref<1x8xf32, 1>) -> ()
    bufferization.materialize_in_destination %42 in writable %arg19 : (tensor<8x6xf32>, memref<8x6xf32, 1>) -> ()
    bufferization.materialize_in_destination %38 in writable %arg20 : (tensor<1x6xf32>, memref<1x6xf32, 1>) -> ()
    bufferization.materialize_in_destination %35 in writable %arg21 : (tensor<6x4xf32>, memref<6x4xf32, 1>) -> ()
    bufferization.materialize_in_destination %31 in writable %arg22 : (tensor<1x4xf32>, memref<1x4xf32, 1>) -> ()
    return
  }
}