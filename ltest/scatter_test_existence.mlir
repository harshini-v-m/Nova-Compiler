func.func @test_linalg_scatter(%arg0: tensor<2xf32>, %arg1: tensor<2x1xi64>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.scatter ins(%arg0, %arg1 : tensor<2xf32>, tensor<2x1xi64>) outs(%arg2 : tensor<4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %1 = arith.addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
