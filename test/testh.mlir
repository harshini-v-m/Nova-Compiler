module {
  func.func @main1(%arg0: tensor<1x10xf32, #nova.device<"1">>, %arg1: tensor<1xi64, #nova.device<"1">>, %arg2: tensor<10x8xf32, #nova.device<"1">>, %arg3: tensor<1x8xf32, #nova.device<"1">>, %arg4: tensor<8x6xf32, #nova.device<"1">>, %arg5: tensor<1x6xf32, #nova.device<"1">>, %arg6: tensor<6x4xf32, #nova.device<"1">>, %arg7: tensor<1x4xf32, #nova.device<"1">>) -> (tensor<1xf32, #nova.device<"1">>, tensor<1xf32, #nova.device<"1">>, tensor<10x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>, tensor<8x6xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>, tensor<6x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1x10xf32, #nova.device<"1">>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<1x8xf32, #nova.device<"1">>
    %1 = linalg.matmul ins(%arg0, %arg2 : tensor<1x10xf32, #nova.device<"1">>, tensor<10x8xf32, #nova.device<"1">>) outs(%0 : tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">>
    %2 = tosa.add %1, %arg3 : (tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">>
    %3 = tosa.clamp %2 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">>
    %4 = tensor.empty() : tensor<1x6xf32, #nova.device<"1">>
    %5 = linalg.matmul ins(%3, %arg4 : tensor<1x8xf32, #nova.device<"1">>, tensor<8x6xf32, #nova.device<"1">>) outs(%4 : tensor<1x6xf32, #nova.device<"1">>) -> tensor<1x6xf32, #nova.device<"1">>
    %6 = tosa.add %5, %arg5 : (tensor<1x6xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>) -> tensor<1x6xf32, #nova.device<"1">>
    %7 = tosa.tanh %6 : (tensor<1x6xf32, #nova.device<"1">>) -> tensor<1x6xf32, #nova.device<"1">>
    %8 = tensor.empty() : tensor<1x4xf32, #nova.device<"1">>
    %9 = linalg.matmul ins(%7, %arg6 : tensor<1x6xf32, #nova.device<"1">>, tensor<6x4xf32, #nova.device<"1">>) outs(%8 : tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %10 = tosa.add %9, %arg7 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %11 = nova.gelu %10 : tensor<1x4xf32, #nova.device<"1">>
    %12 = tosa.exp %11 : (tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %cst = arith.constant dense<1.000000e+00> : tensor<4x1xf32, #nova.device<"1">>
    %13 = tensor.empty() : tensor<1x1xf32, #nova.device<"1">>
    %14 = linalg.matmul ins(%12, %cst : tensor<1x4xf32, #nova.device<"1">>, tensor<4x1xf32, #nova.device<"1">>) outs(%13 : tensor<1x1xf32, #nova.device<"1">>) -> tensor<1x1xf32, #nova.device<"1">>
    %15 = tosa.log %14 : (tensor<1x1xf32, #nova.device<"1">>) -> tensor<1x1xf32, #nova.device<"1">>
    %16 = tosa.sub %11, %15 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x1xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %17 = tensor.empty() : tensor<1x1xf32, #nova.device<"1">>
    %18 = linalg.matmul ins(%16, %cst : tensor<1x4xf32, #nova.device<"1">>, tensor<4x1xf32, #nova.device<"1">>) outs(%17 : tensor<1x1xf32, #nova.device<"1">>) -> tensor<1x1xf32, #nova.device<"1">>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32, #nova.device<"1">>
    %c0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %extracted = tensor.extract %18[%c0, %c0_1] : tensor<1x1xf32, #nova.device<"1">>
    %from_elements = tensor.from_elements %extracted : tensor<1xf32, #nova.device<"1">>
    %19 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"1">>} : tensor<1xf32, #nova.device<"1">>
    %20 = nova.softmax %11 dimension = 1 : tensor<1x4xf32, #nova.device<"1">>
    %21 = nova.constant {value = dense<-1.000000e+00> : tensor<1xf32, #nova.device<"1">>} : tensor<1xf32, #nova.device<"1">>
    %22 = nova.scatter_add %20, %arg1, %21 {axis = 1 : i64} : tensor<1x4xf32, #nova.device<"1">>, tensor<1xi64, #nova.device<"1">>, tensor<1xf32, #nova.device<"1">>
    %23 = nova.constant {value = dense<1.000000e+00> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %24 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %25 = tosa.mul %22, %23, %24 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %expanded = tensor.expand_shape %19 [[0, 1]] output_shape [1, 1] : tensor<1xf32, #nova.device<"1">> into tensor<1x1xf32, #nova.device<"1">>
    %26 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %27 = tosa.mul %25, %expanded, %26 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x1xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %28 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %29 = tosa.mul %10, %10, %28 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %30 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %31 = tosa.mul %29, %10, %30 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %32 = nova.constant {value = dense<4.471500e-02> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %33 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %34 = tosa.mul %31, %32, %33 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %35 = tosa.add %10, %34 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %36 = nova.constant {value = dense<0.797884583> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %37 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %38 = tosa.mul %35, %36, %37 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %39 = tosa.tanh %38 : (tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %40 = nova.constant {value = dense<1.000000e+00> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %41 = tosa.add %40, %39 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %42 = nova.constant {value = dense<5.000000e-01> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %43 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %44 = tosa.mul %41, %42, %43 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %45 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %46 = tosa.mul %39, %39, %45 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %47 = tosa.sub %40, %46 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %48 = nova.constant {value = dense<1.341450e-01> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %49 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %50 = tosa.mul %29, %48, %49 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %51 = tosa.add %40, %50 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %52 = nova.constant {value = dense<0.797884583> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %53 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %54 = tosa.mul %51, %52, %53 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %55 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %56 = tosa.mul %10, %47, %55 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %57 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %58 = tosa.mul %56, %54, %57 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %59 = nova.constant {value = dense<5.000000e-01> : tensor<1x4xf32, #nova.device<"1">>} : tensor<1x4xf32, #nova.device<"1">>
    %60 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %61 = tosa.mul %58, %59, %60 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %62 = tosa.add %44, %61 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) -> tensor<1x4xf32, #nova.device<"1">>
    %63 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %64 = tosa.mul %27, %62, %63 : (tensor<1x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x4xf32, #nova.device<"1">>
    %65 = nova.transpose %arg6 axes1 = 0 axes2 = 1 : tensor<6x4xf32, #nova.device<"1">>
    %66 = tensor.empty() : tensor<1x6xf32, #nova.device<"1">>
    %67 = linalg.matmul ins(%64, %65 : tensor<1x4xf32, #nova.device<"1">>, tensor<4x6xf32, #nova.device<"1">>) outs(%66 : tensor<1x6xf32, #nova.device<"1">>) -> tensor<1x6xf32, #nova.device<"1">>
    %68 = nova.transpose %7 axes1 = 0 axes2 = 1 : tensor<1x6xf32, #nova.device<"1">>
    %69 = tensor.empty() : tensor<6x4xf32, #nova.device<"1">>
    %70 = linalg.matmul ins(%68, %64 : tensor<6x1xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>) outs(%69 : tensor<6x4xf32, #nova.device<"1">>) -> tensor<6x4xf32, #nova.device<"1">>
    %71 = tosa.tanh %6 : (tensor<1x6xf32, #nova.device<"1">>) -> tensor<1x6xf32, #nova.device<"1">>
    %72 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %73 = tosa.mul %71, %71, %72 : (tensor<1x6xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x6xf32, #nova.device<"1">>
    %74 = nova.constant {value = dense<1.000000e+00> : tensor<1x6xf32, #nova.device<"1">>} : tensor<1x6xf32, #nova.device<"1">>
    %75 = tosa.sub %74, %73 : (tensor<1x6xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>) -> tensor<1x6xf32, #nova.device<"1">>
    %76 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %77 = tosa.mul %67, %75, %76 : (tensor<1x6xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x6xf32, #nova.device<"1">>
    %78 = nova.transpose %arg4 axes1 = 0 axes2 = 1 : tensor<8x6xf32, #nova.device<"1">>
    %79 = tensor.empty() : tensor<1x8xf32, #nova.device<"1">>
    %80 = linalg.matmul ins(%77, %78 : tensor<1x6xf32, #nova.device<"1">>, tensor<6x8xf32, #nova.device<"1">>) outs(%79 : tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">>
    %81 = nova.transpose %3 axes1 = 0 axes2 = 1 : tensor<1x8xf32, #nova.device<"1">>
    %82 = tensor.empty() : tensor<8x6xf32, #nova.device<"1">>
    %83 = linalg.matmul ins(%81, %77 : tensor<8x1xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>) outs(%82 : tensor<8x6xf32, #nova.device<"1">>) -> tensor<8x6xf32, #nova.device<"1">>
    %84 = nova.sign %2 : tensor<1x8xf32, #nova.device<"1">>
    %85 = tosa.clamp %84 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">>
    %86 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %87 = tosa.mul %80, %85, %86 : (tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>, tensor<1xi8>) -> tensor<1x8xf32, #nova.device<"1">>
    %88 = nova.transpose %arg2 axes1 = 0 axes2 = 1 : tensor<10x8xf32, #nova.device<"1">>
    %89 = tensor.empty() : tensor<1x10xf32, #nova.device<"1">>
    %90 = linalg.matmul ins(%87, %88 : tensor<1x8xf32, #nova.device<"1">>, tensor<8x10xf32, #nova.device<"1">>) outs(%89 : tensor<1x10xf32, #nova.device<"1">>) -> tensor<1x10xf32, #nova.device<"1">>
    %91 = nova.transpose %arg0 axes1 = 0 axes2 = 1 : tensor<1x10xf32, #nova.device<"1">>
    %92 = tensor.empty() : tensor<10x8xf32, #nova.device<"1">>
    %93 = linalg.matmul ins(%91, %87 : tensor<10x1xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>) outs(%92 : tensor<10x8xf32, #nova.device<"1">>) -> tensor<10x8xf32, #nova.device<"1">>
    return %from_elements, %19, %93, %87, %83, %77, %70, %64, %90 : tensor<1xf32, #nova.device<"1">>, tensor<1xf32, #nova.device<"1">>, tensor<10x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>, tensor<8x6xf32, #nova.device<"1">>, tensor<1x6xf32, #nova.device<"1">>, tensor<6x4xf32, #nova.device<"1">>, tensor<1x4xf32, #nova.device<"1">>, tensor<1x10xf32, #nova.device<"1">>
  }
}
