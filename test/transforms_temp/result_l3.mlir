#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1x1024xf32>) -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = tensor.empty() : tensor<1024x1024xf32>
    %2 = tensor.empty() : tensor<1024x1024xf32>
    %3 = tensor.empty() : tensor<1024x1024xf32>
    %4 = scf.for %arg3 = %c0 to %c1024 step %c256 iter_args(%arg4 = %3) -> (tensor<1024x1024xf32>) {
      %5 = scf.for %arg5 = %c0 to %c1024 step %c256 iter_args(%arg6 = %arg4) -> (tensor<1024x1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [256, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<256x1024xf32>
        %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg5] [1024, 256] [1, 1] : tensor<1024x1024xf32> to tensor<1024x256xf32>
        %extracted_slice_2 = tensor.extract_slice %0[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %6 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice_2 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %7 = linalg.matmul ins(%extracted_slice, %extracted_slice_1 : tensor<256x1024xf32>, tensor<1024x256xf32>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %extracted_slice_3 = tensor.extract_slice %arg2[0, %arg5] [1, 256] [1, 1] : tensor<1x1024xf32> to tensor<1x256xf32>
        %extracted_slice_4 = tensor.extract_slice %1[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_3 : tensor<1x256xf32>) outs(%extracted_slice_4 : tensor<256x256xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<256x256xf32>
        %extracted_slice_5 = tensor.extract_slice %2[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%extracted_slice_5 : tensor<256x256xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %11 = arith.addf %in, %in_7 : f32
          linalg.yield %11 : f32
        } -> tensor<256x256xf32>
        %extracted_slice_6 = tensor.extract_slice %arg6[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst : tensor<256x256xf32>, tensor<256x256xf32>) outs(%extracted_slice_6 : tensor<256x256xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %11 = arith.maximumf %in, %in_7 : f32
          linalg.yield %11 : f32
        } -> tensor<256x256xf32>
        %inserted_slice = tensor.insert_slice %10 into %arg6[%arg3, %arg5] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<1024x1024xf32>
        scf.yield %inserted_slice : tensor<1024x1024xf32>
      }
      scf.yield %5 : tensor<1024x1024xf32>
    }
    return %4 : tensor<1024x1024xf32>
  }
}

