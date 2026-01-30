func.func @matmul_bias_relu(%A: tensor<512x512xf32>, %B: tensor<512x512xf32>, %C: tensor<512x512xf32>, %Bias: tensor<1x512xf32>) -> tensor<512x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %cst = arith.constant dense<0.0> : tensor<512x512xf32>
  
  // 1. Fill (Producer 1)
  %init = linalg.fill ins(%c0 : f32) outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  // 2. Matmul (Producer 2)
  %matmul = linalg.matmul ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                     outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  // 3. Bias Add (Producer 3 / Intermediate)
  // Broadcasting broadcast bias (1x512) to (512x512)
  // Simulating the tiledresult.mlir structure with linalg.generic
  %bias_res = tensor.empty() : tensor<512x512xf32>
  %add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul, %Bias : tensor<512x512xf32>, tensor<1x512xf32>) 
      outs(%bias_res : tensor<512x512xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<512x512xf32>
  
  // 4. ReLU (Consumer)
  %relu_res = tensor.empty() : tensor<512x512xf32>
  %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add : tensor<512x512xf32>) 
      outs(%relu_res : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<512x512xf32>
  
  return %relu : tensor<512x512xf32>
}
