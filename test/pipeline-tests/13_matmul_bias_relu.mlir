// RUN: %nova-opt %s \
// RUN:   -nova-gpu-select-lowering-strategy="cuda-arch=sm_80" \
// RUN:   -nova-tile-and-distribute -nova-config-tracking-canonicalize -cse \
// RUN:   -nova-gpu-pad-operands -nova-config-tracking-canonicalize -cse \
// RUN:   -nova-gpu-promote-matmul-operands -nova-config-tracking-canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-reduction -nova-config-tracking-canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-thread -nova-config-tracking-canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-subgroup -nova-config-tracking-canonicalize -cse

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @matmul_bias_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x512xf32>) -> tensor<128x512xf32> {
  // 1. Matmul
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
                     outs(%arg3 : tensor<128x512xf32>) -> tensor<128x512xf32>
                     
  // 2. Bias Add (broadcast 1D bias to 2D)
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1, #map0],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %arg2 : tensor<128x512xf32>, tensor<512xf32>)
    outs(%arg3 : tensor<128x512xf32>) {
  ^bb0(%in: f32, %bias: f32, %out: f32):
    %add = arith.addf %in, %bias : f32
    linalg.yield %add : f32
  } -> tensor<128x512xf32>
  
  // 3. ReLU (max(0, x))
  %c0 = arith.constant 0.0 : f32
  %2 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"]
  } ins(%1 : tensor<128x512xf32>)
    outs(%arg3 : tensor<128x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %relu = arith.maximumf %in, %c0 : f32
    linalg.yield %relu : f32
  } -> tensor<128x512xf32>

  return %2 : tensor<128x512xf32>
}
