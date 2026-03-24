module {
  func.func @main(%arg0: memref<8x1024xi16, 1>, %arg1: memref<8x1024xi16, 1>, %arg2: memref<1x1024xi64, 1>, %arg3: memref<50304x384xf32, 1>, %arg4: memref<1024x384xf32, 1>, %arg5: memref<384xf32, 1>, %arg6: memref<384xf32, 1>, %arg7: memref<384x1536xf32, 1>, %arg8: memref<1536xf32, 1>, %arg9: memref<1536x384xf32, 1>, %arg10: memref<384xf32, 1>, %arg11: memref<384xf32, 1>, %arg12: memref<384xf32, 1>, %arg13: memref<384x1536xf32, 1>, %arg14: memref<1536xf32, 1>, %arg15: memref<1536x384xf32, 1>, %arg16: memref<384xf32, 1>, %arg17: memref<384xf32, 1>, %arg18: memref<384xf32, 1>, %arg19: memref<384x1536xf32, 1>, %arg20: memref<1536xf32, 1>, %arg21: memref<1536x384xf32, 1>, %arg22: memref<384xf32, 1>, %arg23: memref<384xf32, 1>, %arg24: memref<384xf32, 1>, %arg25: memref<384x50304xf32, 1>, %arg26: memref<1xf32, 1> {bufferization.writable = true}, %arg27: memref<1x1024xi64, 1> {bufferization.writable = true}, %arg28: memref<50304x384xf32, 1> {bufferization.writable = true}, %arg29: memref<1024x384xf32, 1> {bufferization.writable = true}, %arg30: memref<384xf32, 1> {bufferization.writable = true}, %arg31: memref<384xf32, 1> {bufferization.writable = true}, %arg32: memref<384x1536xf32, 1> {bufferization.writable = true}, %arg33: memref<1536xf32, 1> {bufferization.writable = true}, %arg34: memref<1536x384xf32, 1> {bufferization.writable = true}, %arg35: memref<384xf32, 1> {bufferization.writable = true}, %arg36: memref<384xf32, 1> {bufferization.writable = true}, %arg37: memref<384xf32, 1> {bufferization.writable = true}, %arg38: memref<384x1536xf32, 1> {bufferization.writable = true}, %arg39: memref<1536xf32, 1> {bufferization.writable = true}, %arg40: memref<1536x384xf32, 1> {bufferization.writable = true}, %arg41: memref<384xf32, 1> {bufferization.writable = true}, %arg42: memref<384xf32, 1> {bufferization.writable = true}, %arg43: memref<384xf32, 1> {bufferization.writable = true}, %arg44: memref<384x1536xf32, 1> {bufferization.writable = true}, %arg45: memref<1536xf32, 1> {bufferization.writable = true}, %arg46: memref<1536x384xf32, 1> {bufferization.writable = true}, %arg47: memref<384xf32, 1> {bufferization.writable = true}, %arg48: memref<384xf32, 1> {bufferization.writable = true}, %arg49: memref<384xf32, 1> {bufferization.writable = true}, %arg50: memref<384x50304xf32, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
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
    %13 = bufferization.to_tensor %arg13 restrict : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %14 = bufferization.to_tensor %arg14 restrict : memref<1536xf32, 1> to tensor<1536xf32>
    %15 = bufferization.to_tensor %arg15 restrict : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %16 = bufferization.to_tensor %arg16 restrict : memref<384xf32, 1> to tensor<384xf32>
    %17 = bufferization.to_tensor %arg17 restrict : memref<384xf32, 1> to tensor<384xf32>
    %18 = bufferization.to_tensor %arg18 restrict : memref<384xf32, 1> to tensor<384xf32>
    %19 = bufferization.to_tensor %arg19 restrict : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %20 = bufferization.to_tensor %arg20 restrict : memref<1536xf32, 1> to tensor<1536xf32>
    %21 = bufferization.to_tensor %arg21 restrict : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %22 = bufferization.to_tensor %arg22 restrict : memref<384xf32, 1> to tensor<384xf32>
    %23 = bufferization.to_tensor %arg23 restrict : memref<384xf32, 1> to tensor<384xf32>
    %24 = bufferization.to_tensor %arg24 restrict : memref<384xf32, 1> to tensor<384xf32>
    %25 = bufferization.to_tensor %arg25 restrict : memref<384x50304xf32, 1> to tensor<384x50304xf32>
    %26 = nova.gather %3 [%0] : tensor<50304x384xf32>, tensor<8x1024xi16> -> tensor<8x1024x384xf32>
    %27 = nova.gather %4 [%2] : tensor<1024x384xf32>, tensor<1x1024xi64> -> tensor<1x1024x384xf32>
    %28 = nova.add %26, %27 : tensor<8x1024x384xf32>, tensor<1x1024x384xf32>
    %29 = nova.layer_norm %28, %5, %6 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %30 = nova.linear %29, %7, %8 : tensor<8x1024x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>
    %31 = nova.gelu %30 : tensor<8x1024x1536xf32>
    %32 = nova.linear %31, %9, %10 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>, tensor<384xf32>
    %33 = nova.add %28, %32 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %34 = nova.layer_norm %33, %11, %12 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %35 = nova.linear %34, %13, %14 : tensor<8x1024x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>
    %36 = nova.gelu %35 : tensor<8x1024x1536xf32>
    %37 = nova.linear %36, %15, %16 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>, tensor<384xf32>
    %38 = nova.add %33, %37 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %39 = nova.layer_norm %38, %17, %18 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %40 = nova.linear %39, %19, %20 : tensor<8x1024x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>
    %41 = nova.gelu %40 : tensor<8x1024x1536xf32>
    %42 = nova.linear %41, %21, %22 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>, tensor<384xf32>
    %43 = nova.add %38, %42 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %44 = nova.layer_norm %43, %23, %24 : tensor<8x1024x384xf32>, tensor<384xf32>, tensor<384xf32>
    %45 = nova.reshape %44 : tensor<8x1024x384xf32> -> tensor<8192x384xf32>
    %46 = nova.matmul %45, %25 : tensor<8192x384xf32>, tensor<384x50304xf32>
    %47 = nova.reshape %46 : tensor<8192x50304xf32> -> tensor<8x1024x50304xf32>
    %48 = nova.sce %47, %1 : tensor<8x1024x50304xf32>, tensor<8x1024xi16>
    %49 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32>} : tensor<1xf32>
    %50 = nova.sce_backward %47, %1 : tensor<8x1024x50304xf32>, tensor<8x1024xi16>
    %51 = nova.mul %50, %49 : tensor<8x1024x50304xf32>, tensor<1xf32>
    %52 = nova.reshape %51 : tensor<8x1024x50304xf32> -> tensor<8192x50304xf32>
    %53 = nova.transpose %45 : tensor<8192x384xf32>
    %54 = nova.transpose %25 : tensor<384x50304xf32>
    %55 = nova.matmul %52, %54 : tensor<8192x50304xf32>, tensor<50304x384xf32>
    %56 = nova.matmul %53, %52 : tensor<384x8192xf32>, tensor<8192x50304xf32>
    %57 = nova.reshape %55 : tensor<8192x384xf32> -> tensor<8x1024x384xf32>
    %grad_x, %grad_gamma, %grad_beta = nova.layer_norm_backward %57, %43, %23 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %grad_input, %grad_weight, %grad_bias = nova.linear_backward %grad_x, %41, %21 : tensor<8x1024x384xf32>, tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %58 = nova.gelu_backward %grad_input, %40 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %grad_input_0, %grad_weight_1, %grad_bias_2 = nova.linear_backward %58, %39, %19 : tensor<8x1024x1536xf32>, tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %grad_x_3, %grad_gamma_4, %grad_beta_5 = nova.layer_norm_backward %grad_input_0, %38, %17 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %59 = nova.add %grad_x, %grad_x_3 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %grad_input_6, %grad_weight_7, %grad_bias_8 = nova.linear_backward %59, %36, %15 : tensor<8x1024x384xf32>, tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %60 = nova.gelu_backward %grad_input_6, %35 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %grad_input_9, %grad_weight_10, %grad_bias_11 = nova.linear_backward %60, %34, %13 : tensor<8x1024x1536xf32>, tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %grad_x_12, %grad_gamma_13, %grad_beta_14 = nova.layer_norm_backward %grad_input_9, %33, %11 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %61 = nova.add %59, %grad_x_12 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %grad_input_15, %grad_weight_16, %grad_bias_17 = nova.linear_backward %61, %31, %9 : tensor<8x1024x384xf32>, tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %62 = nova.gelu_backward %grad_input_15, %30 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %grad_input_18, %grad_weight_19, %grad_bias_20 = nova.linear_backward %62, %29, %7 : tensor<8x1024x1536xf32>, tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %grad_x_21, %grad_gamma_22, %grad_beta_23 = nova.layer_norm_backward %grad_input_18, %28, %5 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %63 = nova.add %61, %grad_x_21 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %64 = nova.reduce<sum> %63 dimension = [0] : tensor<8x1024x384xf32>
    %65 = nova.reshape %64 : tensor<1024x384xf32> -> tensor<1x1024x384xf32>
    %66 = nova.reshape %2 : tensor<1x1024xi64> -> tensor<1024xi64>
    %67 = nova.reshape %65 : tensor<1x1024x384xf32> -> tensor<1024x384xf32>
    %68 = nova.constant {value = dense<0.000000e+00> : tensor<1024x384xf32>} : tensor<1024x384xf32>
    %69 = nova.scatter_add %68, %66, %67 : tensor<1024x384xf32>, tensor<1024xi64>, tensor<1024x384xf32> -> tensor<1024x384xf32>
    %70 = nova.reshape %0 : tensor<8x1024xi16> -> tensor<8192xi16>
    %71 = nova.reshape %63 : tensor<8x1024x384xf32> -> tensor<8192x384xf32>
    %72 = nova.constant {value = dense<0.000000e+00> : tensor<50304x384xf32>} : tensor<50304x384xf32>
    %73 = nova.scatter_add %72, %70, %71 : tensor<50304x384xf32>, tensor<8192xi16>, tensor<8192x384xf32> -> tensor<50304x384xf32>
    bufferization.materialize_in_destination %48 in writable %arg26 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    %74 = bufferization.to_tensor %arg28 restrict writable : memref<50304x384xf32, 1> to tensor<50304x384xf32>
    %75 = nova.add %74, %73 {in_place = true} : tensor<50304x384xf32>, tensor<50304x384xf32>
    bufferization.materialize_in_destination %75 in writable %arg28 : (tensor<50304x384xf32>, memref<50304x384xf32, 1>) -> ()
    %76 = bufferization.to_tensor %arg29 restrict writable : memref<1024x384xf32, 1> to tensor<1024x384xf32>
    %77 = nova.add %76, %69 {in_place = true} : tensor<1024x384xf32>, tensor<1024x384xf32>
    bufferization.materialize_in_destination %77 in writable %arg29 : (tensor<1024x384xf32>, memref<1024x384xf32, 1>) -> ()
    %78 = bufferization.to_tensor %arg30 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %79 = nova.add %78, %grad_gamma_22 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %79 in writable %arg30 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %80 = bufferization.to_tensor %arg31 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %81 = nova.add %80, %grad_beta_23 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %81 in writable %arg31 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %82 = bufferization.to_tensor %arg32 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %83 = nova.add %82, %grad_weight_19 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %83 in writable %arg32 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %84 = bufferization.to_tensor %arg33 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %85 = nova.add %84, %grad_bias_20 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %85 in writable %arg33 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %86 = bufferization.to_tensor %arg34 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %87 = nova.add %86, %grad_weight_16 {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %87 in writable %arg34 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %88 = bufferization.to_tensor %arg35 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %89 = nova.add %88, %grad_bias_17 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %89 in writable %arg35 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %90 = bufferization.to_tensor %arg36 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %91 = nova.add %90, %grad_gamma_13 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %91 in writable %arg36 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %92 = bufferization.to_tensor %arg37 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %93 = nova.add %92, %grad_beta_14 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %93 in writable %arg37 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %94 = bufferization.to_tensor %arg38 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %95 = nova.add %94, %grad_weight_10 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %95 in writable %arg38 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %96 = bufferization.to_tensor %arg39 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %97 = nova.add %96, %grad_bias_11 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %97 in writable %arg39 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %98 = bufferization.to_tensor %arg40 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %99 = nova.add %98, %grad_weight_7 {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %99 in writable %arg40 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %100 = bufferization.to_tensor %arg41 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %101 = nova.add %100, %grad_bias_8 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %101 in writable %arg41 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %102 = bufferization.to_tensor %arg42 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %103 = nova.add %102, %grad_gamma_4 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %103 in writable %arg42 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %104 = bufferization.to_tensor %arg43 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %105 = nova.add %104, %grad_beta_5 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %105 in writable %arg43 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %106 = bufferization.to_tensor %arg44 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %107 = nova.add %106, %grad_weight_1 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %107 in writable %arg44 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %108 = bufferization.to_tensor %arg45 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %109 = nova.add %108, %grad_bias_2 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %109 in writable %arg45 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %110 = bufferization.to_tensor %arg46 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %111 = nova.add %110, %grad_weight {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %111 in writable %arg46 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %112 = bufferization.to_tensor %arg47 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %113 = nova.add %112, %grad_bias {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %113 in writable %arg47 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %114 = bufferization.to_tensor %arg48 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %115 = nova.add %114, %grad_gamma {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %115 in writable %arg48 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %116 = bufferization.to_tensor %arg49 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %117 = nova.add %116, %grad_beta {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %117 in writable %arg49 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %118 = bufferization.to_tensor %arg50 restrict writable : memref<384x50304xf32, 1> to tensor<384x50304xf32>
    %119 = nova.add %118, %56 {in_place = true} : tensor<384x50304xf32>, tensor<384x50304xf32>
    bufferization.materialize_in_destination %119 in writable %arg50 : (tensor<384x50304xf32>, memref<384x50304xf32, 1>) -> ()
    return
  }
}
