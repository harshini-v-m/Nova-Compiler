#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x64xf32>, %arg2: tensor<1x64xf32>) -> tensor<8x64xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<8x64xf32>) -> tensor<8x64xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x64xf32>) outs(%1 : tensor<8x64xf32>) -> tensor<8x64xf32>
    %3 = tensor.empty() : tensor<8x64xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x64xf32>) outs(%3 : tensor<8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x64xf32>
    %5 = tensor.empty() : tensor<8x64xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<8x64xf32>, tensor<8x64xf32>) outs(%5 : tensor<8x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.addf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<8x64xf32>
    %7 = tensor.empty() : tensor<8x64xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %cst : tensor<8x64xf32>, tensor<8x64xf32>) outs(%7 : tensor<8x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.maximumf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<8x64xf32>
    return %8 : tensor<8x64xf32>
  }
}