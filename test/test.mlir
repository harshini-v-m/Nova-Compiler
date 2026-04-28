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
    %softmax, %result = nova.sce %47, %1 : tensor<8x1024x50304xf32>, tensor<8x1024xi16>
    %48 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32>} : tensor<1xf32>
    %49 = nova.sce_backward %softmax, %1 : tensor<8x1024x50304xf32>, tensor<8x1024xi16>
    %50 = nova.mul %49, %48 : tensor<8x1024x50304xf32>, tensor<1xf32>
    %51 = nova.reshape %50 : tensor<8x1024x50304xf32> -> tensor<8192x50304xf32>
    %52 = nova.transpose %45 : tensor<8192x384xf32>
    %53 = nova.transpose %25 : tensor<384x50304xf32>
    %54 = nova.matmul %51, %53 : tensor<8192x50304xf32>, tensor<50304x384xf32>
    %55 = nova.matmul %52, %51 : tensor<384x8192xf32>, tensor<8192x50304xf32>
    %56 = bufferization.to_tensor %arg50 restrict writable : memref<384x50304xf32, 1> to tensor<384x50304xf32>
    %57 = nova.add %56, %55 {in_place = true} : tensor<384x50304xf32>, tensor<384x50304xf32>
    bufferization.materialize_in_destination %57 in writable %arg50 : (tensor<384x50304xf32>, memref<384x50304xf32, 1>) -> ()
    %58 = nova.reshape %54 : tensor<8192x384xf32> -> tensor<8x1024x384xf32>
    %grad_x, %grad_gamma, %grad_beta = nova.layer_norm_backward %58, %43, %23 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %59 = bufferization.to_tensor %arg48 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %60 = nova.add %59, %grad_gamma {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %60 in writable %arg48 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %61 = bufferization.to_tensor %arg49 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %62 = nova.add %61, %grad_beta {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %62 in writable %arg49 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %63 = nova.transpose %21 : tensor<1536x384xf32>
    %64 = nova.matmul %grad_x, %63 : tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %65 = nova.transpose %41 : tensor<8x1024x1536xf32>
    %66 = nova.matmul %65, %grad_x : tensor<8x1536x1024xf32>, tensor<8x1024x384xf32>
    %67 = nova.reduce<sum> %66 dimension = [0] : tensor<8x1536x384xf32>
    %68 = nova.reshape %67 : tensor<1536x384xf32> -> tensor<1536x384xf32>
    %69 = nova.reduce<sum> %grad_x dimension = [0, 1] : tensor<8x1024x384xf32>
    %70 = nova.reshape %69 : tensor<384xf32> -> tensor<384xf32>
    %71 = bufferization.to_tensor %arg47 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %72 = nova.add %71, %70 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %72 in writable %arg47 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %73 = bufferization.to_tensor %arg46 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %74 = nova.add %73, %68 {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %74 in writable %arg46 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %75 = nova.gelu_backward %64, %40 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %76 = nova.transpose %19 : tensor<384x1536xf32>
    %77 = nova.matmul %75, %76 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %78 = nova.transpose %39 : tensor<8x1024x384xf32>
    %79 = nova.matmul %78, %75 : tensor<8x384x1024xf32>, tensor<8x1024x1536xf32>
    %80 = nova.reduce<sum> %79 dimension = [0] : tensor<8x384x1536xf32>
    %81 = nova.reshape %80 : tensor<384x1536xf32> -> tensor<384x1536xf32>
    %82 = nova.reduce<sum> %75 dimension = [0, 1] : tensor<8x1024x1536xf32>
    %83 = nova.reshape %82 : tensor<1536xf32> -> tensor<1536xf32>
    %84 = bufferization.to_tensor %arg45 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %85 = nova.add %84, %83 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %85 in writable %arg45 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %86 = bufferization.to_tensor %arg44 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %87 = nova.add %86, %81 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %87 in writable %arg44 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %grad_x_0, %grad_gamma_1, %grad_beta_2 = nova.layer_norm_backward %77, %38, %17 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %88 = nova.add %grad_x, %grad_x_0 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %89 = bufferization.to_tensor %arg42 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %90 = nova.add %89, %grad_gamma_1 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %90 in writable %arg42 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %91 = bufferization.to_tensor %arg43 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %92 = nova.add %91, %grad_beta_2 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %92 in writable %arg43 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %93 = nova.transpose %15 : tensor<1536x384xf32>
    %94 = nova.matmul %88, %93 : tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %95 = nova.transpose %36 : tensor<8x1024x1536xf32>
    %96 = nova.matmul %95, %88 : tensor<8x1536x1024xf32>, tensor<8x1024x384xf32>
    %97 = nova.reduce<sum> %96 dimension = [0] : tensor<8x1536x384xf32>
    %98 = nova.reshape %97 : tensor<1536x384xf32> -> tensor<1536x384xf32>
    %99 = nova.reduce<sum> %88 dimension = [0, 1] : tensor<8x1024x384xf32>
    %100 = nova.reshape %99 : tensor<384xf32> -> tensor<384xf32>
    %101 = bufferization.to_tensor %arg41 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %102 = nova.add %101, %100 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %102 in writable %arg41 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %103 = bufferization.to_tensor %arg40 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %104 = nova.add %103, %98 {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %104 in writable %arg40 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %105 = nova.gelu_backward %94, %35 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %106 = nova.transpose %13 : tensor<384x1536xf32>
    %107 = nova.matmul %105, %106 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %108 = nova.transpose %34 : tensor<8x1024x384xf32>
    %109 = nova.matmul %108, %105 : tensor<8x384x1024xf32>, tensor<8x1024x1536xf32>
    %110 = nova.reduce<sum> %109 dimension = [0] : tensor<8x384x1536xf32>
    %111 = nova.reshape %110 : tensor<384x1536xf32> -> tensor<384x1536xf32>
    %112 = nova.reduce<sum> %105 dimension = [0, 1] : tensor<8x1024x1536xf32>
    %113 = nova.reshape %112 : tensor<1536xf32> -> tensor<1536xf32>
    %114 = bufferization.to_tensor %arg39 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %115 = nova.add %114, %113 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %115 in writable %arg39 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %116 = bufferization.to_tensor %arg38 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %117 = nova.add %116, %111 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %117 in writable %arg38 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %grad_x_3, %grad_gamma_4, %grad_beta_5 = nova.layer_norm_backward %107, %33, %11 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %118 = nova.add %88, %grad_x_3 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %119 = bufferization.to_tensor %arg36 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %120 = nova.add %119, %grad_gamma_4 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %120 in writable %arg36 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %121 = bufferization.to_tensor %arg37 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %122 = nova.add %121, %grad_beta_5 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %122 in writable %arg37 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %123 = nova.transpose %9 : tensor<1536x384xf32>
    %124 = nova.matmul %118, %123 : tensor<8x1024x384xf32>, tensor<384x1536xf32>
    %125 = nova.transpose %31 : tensor<8x1024x1536xf32>
    %126 = nova.matmul %125, %118 : tensor<8x1536x1024xf32>, tensor<8x1024x384xf32>
    %127 = nova.reduce<sum> %126 dimension = [0] : tensor<8x1536x384xf32>
    %128 = nova.reshape %127 : tensor<1536x384xf32> -> tensor<1536x384xf32>
    %129 = nova.reduce<sum> %118 dimension = [0, 1] : tensor<8x1024x384xf32>
    %130 = nova.reshape %129 : tensor<384xf32> -> tensor<384xf32>
    %131 = bufferization.to_tensor %arg35 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %132 = nova.add %131, %130 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %132 in writable %arg35 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %133 = bufferization.to_tensor %arg34 restrict writable : memref<1536x384xf32, 1> to tensor<1536x384xf32>
    %134 = nova.add %133, %128 {in_place = true} : tensor<1536x384xf32>, tensor<1536x384xf32>
    bufferization.materialize_in_destination %134 in writable %arg34 : (tensor<1536x384xf32>, memref<1536x384xf32, 1>) -> ()
    %135 = nova.gelu_backward %124, %30 : tensor<8x1024x1536xf32>, tensor<8x1024x1536xf32>
    %136 = nova.transpose %7 : tensor<384x1536xf32>
    %137 = nova.matmul %135, %136 : tensor<8x1024x1536xf32>, tensor<1536x384xf32>
    %138 = nova.transpose %29 : tensor<8x1024x384xf32>
    %139 = nova.matmul %138, %135 : tensor<8x384x1024xf32>, tensor<8x1024x1536xf32>
    %140 = nova.reduce<sum> %139 dimension = [0] : tensor<8x384x1536xf32>
    %141 = nova.reshape %140 : tensor<384x1536xf32> -> tensor<384x1536xf32>
    %142 = nova.reduce<sum> %135 dimension = [0, 1] : tensor<8x1024x1536xf32>
    %143 = nova.reshape %142 : tensor<1536xf32> -> tensor<1536xf32>
    %144 = bufferization.to_tensor %arg33 restrict writable : memref<1536xf32, 1> to tensor<1536xf32>
    %145 = nova.add %144, %143 {in_place = true} : tensor<1536xf32>, tensor<1536xf32>
    bufferization.materialize_in_destination %145 in writable %arg33 : (tensor<1536xf32>, memref<1536xf32, 1>) -> ()
    %146 = bufferization.to_tensor %arg32 restrict writable : memref<384x1536xf32, 1> to tensor<384x1536xf32>
    %147 = nova.add %146, %141 {in_place = true} : tensor<384x1536xf32>, tensor<384x1536xf32>
    bufferization.materialize_in_destination %147 in writable %arg32 : (tensor<384x1536xf32>, memref<384x1536xf32, 1>) -> ()
    %grad_x_6, %grad_gamma_7, %grad_beta_8 = nova.layer_norm_backward %137, %28, %5 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>, tensor<384xf32>
    %148 = nova.add %118, %grad_x_6 : tensor<8x1024x384xf32>, tensor<8x1024x384xf32>
    %149 = bufferization.to_tensor %arg30 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %150 = nova.add %149, %grad_gamma_7 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %150 in writable %arg30 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %151 = bufferization.to_tensor %arg31 restrict writable : memref<384xf32, 1> to tensor<384xf32>
    %152 = nova.add %151, %grad_beta_8 {in_place = true} : tensor<384xf32>, tensor<384xf32>
    bufferization.materialize_in_destination %152 in writable %arg31 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    %153 = nova.reduce<sum> %148 dimension = [0] : tensor<8x1024x384xf32>
    %154 = nova.reshape %153 : tensor<1024x384xf32> -> tensor<1x1024x384xf32>
    %155 = nova.reshape %2 : tensor<1x1024xi64> -> tensor<1024xi64>
    %156 = nova.reshape %154 : tensor<1x1024x384xf32> -> tensor<1024x384xf32>
    %157 = nova.constant {value = dense<0.000000e+00> : tensor<1024x384xf32>} : tensor<1024x384xf32>
    %158 = nova.scatter_add %157, %155, %156 : tensor<1024x384xf32>, tensor<1024xi64>, tensor<1024x384xf32> -> tensor<1024x384xf32>
    %159 = bufferization.to_tensor %arg29 restrict writable : memref<1024x384xf32, 1> to tensor<1024x384xf32>
    %160 = nova.add %159, %158 {in_place = true} : tensor<1024x384xf32>, tensor<1024x384xf32>
    bufferization.materialize_in_destination %160 in writable %arg29 : (tensor<1024x384xf32>, memref<1024x384xf32, 1>) -> ()
    %161 = nova.reshape %0 : tensor<8x1024xi16> -> tensor<8192xi16>
    %162 = nova.reshape %148 : tensor<8x1024x384xf32> -> tensor<8192x384xf32>
    %163 = nova.constant {value = dense<0.000000e+00> : tensor<50304x384xf32>} : tensor<50304x384xf32>
    %164 = nova.scatter_add %163, %161, %162 : tensor<50304x384xf32>, tensor<8192xi16>, tensor<8192x384xf32> -> tensor<50304x384xf32>
    %165 = bufferization.to_tensor %arg28 restrict writable : memref<50304x384xf32, 1> to tensor<50304x384xf32>
    %166 = nova.add %165, %164 {in_place = true} : tensor<50304x384xf32>, tensor<50304x384xf32>
    bufferization.materialize_in_destination %166 in writable %arg28 : (tensor<50304x384xf32>, memref<50304x384xf32, 1>) -> ()
    bufferization.materialize_in_destination %result in writable %arg26 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    return
  }
}
