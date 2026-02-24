#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @matmul_bias_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x512xf32>) -> tensor<128x512xf32> {
    %0 = linalg.matmul {lowering_config = {mma_kind = 0 : i32, promoted_operands = [0, 1], reduction = [0, 0, 8], subgroup = [16, 16, 0], thread = [4, 4, 0], workgroup = [128, 64, 0]}} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%arg3 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %arg2 : tensor<128x512xf32>, tensor<512xf32>) outs(%arg3 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    } -> tensor<128x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x512xf32>) outs(%arg3 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maximumf %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<128x512xf32>
    return %2 : tensor<128x512xf32>
  }
}

