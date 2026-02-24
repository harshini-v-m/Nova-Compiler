#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1024x2048xf32>, %arg1: tensor<2048x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024x512xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x512xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x2048xf32>, tensor<2048x512xf32>) outs(%1 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %3 = tensor.empty() : tensor<1024x512xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%3 : tensor<1024x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.addf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1024x512xf32>
    %5 = tensor.empty() : tensor<1024x512xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%5 : tensor<1024x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.maximumf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<1024x512xf32>
    return %6 : tensor<1024x512xf32>
  }
}