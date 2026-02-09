module {
  func.func @main(%arg0: memref<2x1x4xf32, 1>, %arg1: memref<2xi32, 1>, %arg2: memref<2x1x3xf32, 1>, %arg3: memref<2x3xf32, 1>, %arg4: memref<1x1xf32, 1>, %arg5: memref<2x1x3xf32, 1>, %arg6: memref<2x1x4xf32, 1>, %arg7: memref<2x1x6xf32, 1>, %arg8: memref<2x1x6xf32, 1>, %arg9: memref<2x1x6xf32, 1>, %arg10: memref<2x1x6xf32, 1>, %arg11: memref<2x1x6xf32, 1>, %arg12: memref<2x1x6xf32, 1>, %arg13: memref<2x1x6xf32, 1>, %arg14: memref<2x4x6xf32, 1>, %arg15: memref<2x1x6xf32, 1>, %arg16: memref<2x6x4xf32, 1>, %arg17: memref<2x1x4xf32, 1>, %arg18: memref<2x4x3xf32, 1>, %arg19: memref<2x1x3xf32, 1>, %arg20: memref<1xf32, 1>, %arg21: memref<2x4x6xf32, 1>, %arg22: memref<2x1x6xf32, 1>, %arg23: memref<2x6x4xf32, 1>, %arg24: memref<2x1x4xf32, 1>, %arg25: memref<2x4x3xf32, 1>, %arg26: memref<2x1x3xf32, 1>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<2x1x4xf32, 1> to tensor<2x1x4xf32, #nova.device<"1">>
    %1 = bufferization.to_tensor %arg1 restrict : memref<2xi32, 1> to tensor<2xi32, #nova.device<"1">>
    %2 = bufferization.to_tensor %arg2 restrict : memref<2x1x3xf32, 1> to tensor<2x1x3xf32, #nova.device<"1">>
    %3 = bufferization.to_tensor %arg3 restrict : memref<2x3xf32, 1> to tensor<2x3xf32, #nova.device<"1">>
    %4 = bufferization.to_tensor %arg4 restrict : memref<1x1xf32, 1> to tensor<1x1xf32, #nova.device<"1">>
    %5 = bufferization.to_tensor %arg5 restrict : memref<2x1x3xf32, 1> to tensor<2x1x3xf32, #nova.device<"1">>
    %6 = bufferization.to_tensor %arg6 restrict : memref<2x1x4xf32, 1> to tensor<2x1x4xf32, #nova.device<"1">>
    %7 = bufferization.to_tensor %arg7 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %8 = bufferization.to_tensor %arg8 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %9 = bufferization.to_tensor %arg9 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %10 = bufferization.to_tensor %arg10 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %11 = bufferization.to_tensor %arg11 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %12 = bufferization.to_tensor %arg12 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %13 = bufferization.to_tensor %arg13 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %14 = bufferization.to_tensor %arg14 restrict : memref<2x4x6xf32, 1> to tensor<2x4x6xf32, #nova.device<"1">>
    %15 = bufferization.to_tensor %arg15 restrict : memref<2x1x6xf32, 1> to tensor<2x1x6xf32, #nova.device<"1">>
    %16 = bufferization.to_tensor %arg16 restrict : memref<2x6x4xf32, 1> to tensor<2x6x4xf32, #nova.device<"1">>
    %17 = bufferization.to_tensor %arg17 restrict : memref<2x1x4xf32, 1> to tensor<2x1x4xf32, #nova.device<"1">>
    %18 = bufferization.to_tensor %arg18 restrict : memref<2x4x3xf32, 1> to tensor<2x4x3xf32, #nova.device<"1">>
    %19 = bufferization.to_tensor %arg19 restrict : memref<2x1x3xf32, 1> to tensor<2x1x3xf32, #nova.device<"1">>
    %20 = nova.matmul %0, %14 : tensor<2x1x4xf32, #nova.device<"1">>, tensor<2x4x6xf32, #nova.device<"1">>
    %21 = nova.add %20, %15 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %22 = nova.gelu %21 : tensor<2x1x6xf32>
    %23 = nova.matmul %22, %16 : tensor<2x1x6xf32>, tensor<2x6x4xf32, #nova.device<"1">>
    %24 = nova.add %23, %17 : tensor<2x1x4xf32>, tensor<2x1x4xf32, #nova.device<"1">>
    %25 = nova.relu %24 : tensor<2x1x4xf32>
    %26 = nova.matmul %25, %18 : tensor<2x1x4xf32>, tensor<2x4x3xf32, #nova.device<"1">>
    %27 = nova.add %26, %19 : tensor<2x1x3xf32>, tensor<2x1x3xf32, #nova.device<"1">>
    %28 = nova.sigmoid %27 : tensor<2x1x3xf32>
    %29 = nova.sce %3, %1 : tensor<2x3xf32, #nova.device<"1">>, tensor<2xi32, #nova.device<"1">>
    %30 = nova.softmax %3 dimension = 1 : tensor<2x3xf32, #nova.device<"1">>
    %31 = nova.constant {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : tensor<3xf32>
    %cst = arith.constant dense<[1, 3]> : tensor<2xindex>
    %reshape = tensor.reshape %31(%cst) : (tensor<3xf32>, tensor<2xindex>) -> tensor<1x3xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [2, 1] : tensor<2xi32, #nova.device<"1">> into tensor<2x1xi32>
    %32 = nova.broadcast_in_dim %expanded, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x3xi32>
    %33 = nova.broadcast_in_dim %reshape, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<2x3xf32>
    %34 = nova.compare<eq> %32, %33 : tensor<2x3xi32>, tensor<2x3xf32>
    %35 = tosa.cast %34 : (tensor<2x3xi1>) -> tensor<2x3xf32>
    %36 = nova.sub %30, %35 : tensor<2x3xf32>, tensor<2x3xf32>
    %37 = nova.constant {value = dense<5.000000e-01> : tensor<2x3xf32>} : tensor<2x3xf32>
    %38 = nova.mul %4, %37 : tensor<1x1xf32, #nova.device<"1">>, tensor<2x3xf32>
    %39 = nova.mul %36, %38 : tensor<2x3xf32>, tensor<2x3xf32>
    %cst_0 = arith.constant dense<[2, 1, 3]> : tensor<3xindex>
    %reshape_1 = tensor.reshape %39(%cst_0) : (tensor<2x3xf32>, tensor<3xindex>) -> tensor<2x1x3xf32>
    %40 = nova.sub %5, %28 : tensor<2x1x3xf32, #nova.device<"1">>, tensor<2x1x3xf32>
    %41 = nova.mul %28, %40 : tensor<2x1x3xf32>, tensor<2x1x3xf32>
    %42 = nova.mul %reshape_1, %41 : tensor<2x1x3xf32>, tensor<2x1x3xf32>
    %43 = nova.transpose %18 axes1 = 2 axes2 = 1 : tensor<2x4x3xf32, #nova.device<"1">>
    %44 = nova.matmul %42, %43 : tensor<2x1x3xf32>, tensor<2x3x4xf32>
    %45 = nova.transpose %25 axes1 = 2 axes2 = 1 : tensor<2x1x4xf32>
    %46 = nova.matmul %45, %42 : tensor<2x4x1xf32>, tensor<2x1x3xf32>
    %47 = nova.compare<gt> %24, %6 : tensor<2x1x4xf32>, tensor<2x1x4xf32, #nova.device<"1">>
    %48 = tosa.cast %47 : (tensor<2x1x4xi1>) -> tensor<2x1x4xf32>
    %49 = nova.mul %44, %48 : tensor<2x1x4xf32>, tensor<2x1x4xf32>
    %50 = nova.transpose %16 axes1 = 2 axes2 = 1 : tensor<2x6x4xf32, #nova.device<"1">>
    %51 = nova.matmul %49, %50 : tensor<2x1x4xf32>, tensor<2x4x6xf32>
    %52 = nova.transpose %22 axes1 = 2 axes2 = 1 : tensor<2x1x6xf32>
    %53 = nova.matmul %52, %49 : tensor<2x6x1xf32>, tensor<2x1x4xf32>
    %54 = nova.mul %21, %21 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %55 = nova.mul %54, %21 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %56 = nova.mul %55, %7 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %57 = nova.add %21, %56 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %58 = nova.mul %57, %8 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %59 = nova.tanh %58 : tensor<2x1x6xf32>
    %60 = nova.add %9, %59 : tensor<2x1x6xf32, #nova.device<"1">>, tensor<2x1x6xf32>
    %61 = nova.mul %60, %10 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %62 = nova.mul %59, %59 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %63 = nova.sub %9, %62 : tensor<2x1x6xf32, #nova.device<"1">>, tensor<2x1x6xf32>
    %64 = nova.mul %54, %11 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %65 = nova.add %9, %64 : tensor<2x1x6xf32, #nova.device<"1">>, tensor<2x1x6xf32>
    %66 = nova.mul %65, %12 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %67 = nova.mul %21, %63 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %68 = nova.mul %67, %66 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %69 = nova.mul %68, %13 : tensor<2x1x6xf32>, tensor<2x1x6xf32, #nova.device<"1">>
    %70 = nova.add %61, %69 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %71 = nova.mul %51, %70 : tensor<2x1x6xf32>, tensor<2x1x6xf32>
    %72 = nova.transpose %0 axes1 = 2 axes2 = 1 : tensor<2x1x4xf32, #nova.device<"1">>
    %73 = nova.matmul %72, %71 : tensor<2x4x1xf32>, tensor<2x1x6xf32>
    bufferization.materialize_in_destination %29 in writable %arg20 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    bufferization.materialize_in_destination %73 in writable %arg21 : (tensor<2x4x6xf32>, memref<2x4x6xf32, 1>) -> ()
    bufferization.materialize_in_destination %71 in writable %arg22 : (tensor<2x1x6xf32>, memref<2x1x6xf32, 1>) -> ()
    bufferization.materialize_in_destination %53 in writable %arg23 : (tensor<2x6x4xf32>, memref<2x6x4xf32, 1>) -> ()
    bufferization.materialize_in_destination %49 in writable %arg24 : (tensor<2x1x4xf32>, memref<2x1x4xf32, 1>) -> ()
    bufferization.materialize_in_destination %46 in writable %arg25 : (tensor<2x4x3xf32>, memref<2x4x3xf32, 1>) -> ()
    bufferization.materialize_in_destination %42 in writable %arg26 : (tensor<2x1x3xf32>, memref<2x1x3xf32, 1>) -> ()
    return
  }
}