module {
  func.func @main(%arg0: memref<8x1024xi16, 1>, %arg1: memref<8x1024xi16, 1>, %arg2: memref<1x1024xi64, 1>, %arg3: memref<50304x384xf32, 1>, %arg4: memref<1024x384xf32, 1>, %arg5: memref<384xf32, 1>, %arg6: memref<384xf32, 1>, %arg7: memref<384x1536xf32, 1>, %arg8: memref<1536xf32, 1>, %arg9: memref<1536x384xf32, 1>, %arg10: memref<384xf32, 1>, %arg11: memref<384xf32, 1>, %arg12: memref<384xf32, 1>, %arg13: memref<384x50304xf32, 1>, %arg14: memref<1xf32, 1> {bufferization.writable = true}, %arg15: memref<1x1024xi64, 1> {bufferization.writable = true}, %arg16: memref<50304x384xf32, 1> {bufferization.writable = true}, %arg17: memref<1024x384xf32, 1> {bufferization.writable = true}, %arg18: memref<384xf32, 1> {bufferization.writable = true}, %arg19: memref<384xf32, 1> {bufferization.writable = true}, %arg20: memref<384x1536xf32, 1> {bufferization.writable = true}, %arg21: memref<1536xf32, 1> {bufferization.writable = true}, %arg22: memref<1536x384xf32, 1> {bufferization.writable = true}, %arg23: memref<384xf32, 1> {bufferization.writable = true}, %arg24: memref<384xf32, 1> {bufferization.writable = true}, %arg25: memref<384xf32, 1> {bufferization.writable = true}, %arg26: memref<384x50304xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x1024xi16, 1> to tensor<8x1024xi16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<8x1024xi16, 1> to tensor<8x1024xi16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1x1024xi64, 1> to tensor<1x1024xi64>
    %3 = bufferization.to_tensor %arg3 restrict : memref<50304x384xf32, 1> to tensor<50304x384xf32>
    %4 = bufferization.to_tensor %arg4 restrict : memref<1024x384xf32, 1> to tensor<1024x384xf32>
    %5 = bufferization.to_tensor %arg5 restrict : memref<384xf32, 1> to tensor<384xf32>
    %6 = bufferization.to_tensor %arg6 restrict : memref<384xf32, 1> to tensor<384xf32>
    %7 = bufferization.to_tensor %arg7 restrict : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %8 = bufferization.to_tensor %arg8 restrict : memref<1536xf32, 1> to tensor<1536xf32>
    %9 = bufferization.to_tensor %arg9 restrict : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %10 = bufferization.to_tensor %arg10 restrict : memref<384xf32, 1> to tensor<384xf32>
    %11 = bufferization.to_tensor %arg11 restrict : memref<384xf32, 1> to tensor<384xf32>
    %12 = bufferization.to_tensor %arg12 restrict : memref<384xf32, 1> to tensor<384xf32>
    %13 = bufferization.to_tensor %arg13 restrict : memref<384x50304xf32, 1> to tensor<384x50304xf32>
    %14 = nova.gather %3 [%0] : tensor<50304x384xf32>, tensor<8x1024xi16> -> tensor<8x1024x384xf32>
    %15 = nova.gather %4 [%2] : tensor<1024x384xf32>, tensor<1x1024xi64> -> tensor<1x1024x384xf32>
    %16 = nova.add %14, %15 : tensor<8x1024x384xf32>, tensor<1x1024x384xf32>
    %17 = nova.layer_norm %16, %5, %6 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %18 = nova.linear %17, %7, %8 : tensor<8x1024x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>
    %19 = nova.gelu %18 : tensor<8x1024x1536xf32>
    %20 = nova.linear %19, %9, %10 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>, tensor<384xf32>
    %21 = nova.add %16, %20 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %22 = nova.layer_norm %21, %11, %12 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %23 = nova.reshape %22 : tensor<8x1024x384xf32> -> tensor<8192x384xf32>
    %24 = nova.matmul %23, %13 : tensor<8192x384xf32>, tensor<384x50304xf32>
    %25 = nova.reshape %24 : tensor<8192x50304xf32> -> tensor<8x1024x50304xf32>
    %26 = nova.sce %25, %1 : tensor<8x1024x50304xf32>, tensor<8x1024xi16>
    %27 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32>} : tensor<1xf32>
    %28 = nova.sce_backward %25, %1 : tensor<8x1024x50304xf32>, tensor<8x1024xi16>
    %29 = nova.mul %28, %27 : tensor<8x1024x50304xf32>, tensor<1xf32>
    %30 = nova.reshape %29 : tensor<8x1024x50304xf32> -> tensor<8192x50304xf32>
    %31 = nova.transpose %23 : tensor<8192x384xf32>
    %32 = nova.transpose %13 : tensor<384x50304xf32>
    %33 = nova.matmul %30, %32 : tensor<8192x50304xf32>, tensor<50304x384xf32>
    %34 = nova.matmul %31, %30 : tensor<384x8192xf32>, tensor<8192x50304xf32>
    %35 = nova.reshape %33 : tensor<8192x384xf32> -> tensor<8x1024x384xf32>
    %grad_x, %grad_gamma, %grad_beta = nova.layer_norm_backward %35, %21, %11 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %grad_input, %grad_weight, %grad_bias = nova.linear_backward %grad_x, %19, %9 : tensor<8x1024x384xf32>, tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %36 = nova.gelu_backward %grad_input, %18 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %grad_input_0, %grad_weight_1, %grad_bias_2 = nova.linear_backward %36, %17, %7 : tensor<8x1024x1536xf32>, tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %grad_x_3, %grad_gamma_4, %grad_beta_5 = nova.layer_norm_backward %grad_input_0, %16, %5 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %37 = nova.add %grad_x, %grad_x_3 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %38 = nova.reduce<sum> %37 dimension = [0] : tensor<8x1024x384xf32>
    %39 = nova.reshape %38 : tensor<1024x384xf32> -> tensor<1x1024x384xf32>
    %40 = nova.reshape %2 : tensor<1x1024xi64> -> tensor<1024xi64>
    %41 = nova.reshape %39 : tensor<1x1024x384xf32> -> tensor<1024x384xf32>
    %42 = nova.constant {value = dense<0.000000e+00> : tensor<1024x384xf32>} : tensor<1024x384xf32>
    %43 = nova.scatter_add %42, %40, %41 : tensor<1024x384xf32>, tensor<1024xi64>, tensor<1024x384xf32> -> tensor<1024x384xf32>
    %44 = nova.reshape %0 : tensor<8x1024xi16> -> tensor<8192xi16>
    %45 = nova.reshape %37 : tensor<8x1024x384xf32> -> tensor<8192x384xf32>
    %46 = nova.constant {value = dense<0.000000e+00> : tensor<50304x384xf32>} : tensor<50304x384xf32>
    %47 = nova.scatter_add %46, %44, %45 : tensor<50304x384xf32>, tensor<8192xi16>, tensor<8192x384xf32> -> tensor<50304x384xf32>
    bufferization.materialize_in_destination %26 in writable %arg14 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    %48 = bufferization.to_tensor %arg16 restrict writable : memref<50304x384xf32, 1> to tensor<50304x384xf32>
    %49 = nova.add %48, %47 {in_place = true} : tensor<50304x384xf32>, tensor<50304x384xf32>
    bufferization.materialize_in_destination %49 in writable %arg16 : (tensor<50304x384xf32>, memref<50304x384xf32, 1>) -> ()
    %50 = bufferization.to_tensor %arg17 restrict writable : memref<1024x384xf32, 1> to tensor<1024x384xf32>
    %51 = nova.add %50, %43 {in_place = true} : tensor<1024x384xf32>, tensor<1024x384xf32>
    bufferization.materialize_in_destination %51 in writable %arg17 : (tensor<1024x384xf32>, memref<1024x384xf32, 1>) -> ()
    %52 = bufferization.to_tensor %arg18 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %53 = nova.add %52, %grad_gamma_4 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %53 in writable %arg18 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %54 = bufferization.to_tensor %arg19 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %55 = nova.add %54, %grad_beta_5 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %55 in writable %arg19 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %56 = bufferization.to_tensor %arg20 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %57 = nova.add %56, %grad_weight_1 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %57 in writable %arg20 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %58 = bufferization.to_tensor %arg21 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %59 = nova.add %58, %grad_bias_2 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %59 in writable %arg21 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %60 = bufferization.to_tensor %arg22 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %61 = nova.add %60, %grad_weight {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %61 in writable %arg22 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %62 = bufferization.to_tensor %arg23 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %63 = nova.add %62, %grad_bias {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %63 in writable %arg23 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %64 = bufferization.to_tensor %arg24 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %65 = nova.add %64, %grad_gamma {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %65 in writable %arg24 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %66 = bufferization.to_tensor %arg25 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %67 = nova.add %66, %grad_beta {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %67 in writable %arg25 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %68 = bufferization.to_tensor %arg26 restrict writable : memref<384x50304xf32, 1> to tensor<384x50304xf32>
    %69 = nova.add %68, %34 {in_place = true} : tensor<384x50304xf32>, tensor<384x50304xf32>
    bufferization.materialize_in_destination %69 in writable %arg26 : (tensor<384x50304xf32>, memref<384x50304xf32, 1>) -> ()
    return
  }
}
