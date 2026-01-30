func.func @mlp_fusion(%A: tensor<128x128xf32>, %B1: tensor<128x128xf32>, %B2: tensor<128x128xf32>, %Bias1: tensor<128x128xf32>, %Bias2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  
  // ==========================================
  // Layer 1
  // ==========================================
  
  // 1. Matmul
  %empty1 = tensor.empty() : tensor<128x128xf32>
  %matmul1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%empty1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  
  // 2. Bias Add
  %bias_res1 = tensor.empty() : tensor<128x128xf32>
  %add1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul1, %Bias1 : tensor<128x128xf32>, tensor<128x128xf32>) 
      outs(%bias_res1 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<128x128xf32>
  
  // 3. ReLU
  %relu_res1 = tensor.empty() : tensor<128x128xf32>
  %relu1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add1 : tensor<128x128xf32>) 
      outs(%relu_res1 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<128x128xf32>

  // ==========================================
  // Layer 2
  // ==========================================
  
  // 4. Matmul
  %empty2 = tensor.empty() : tensor<128x128xf32>
  %matmul2 = linalg.matmul ins(%relu1, %B2 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%empty2 : tensor<128x128xf32>) -> tensor<128x128xf32>
                     
  // 5. Bias Add
  %bias_res2 = tensor.empty() : tensor<128x128xf32>
  %add2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul2, %Bias2 : tensor<128x128xf32>, tensor<128x128xf32>) 
      outs(%bias_res2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<128x128xf32>
  
  // 6. ReLU
  %relu_res2 = tensor.empty() : tensor<128x128xf32>
  %relu2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add2 : tensor<128x128xf32>) 
      outs(%relu_res2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<128x128xf32>
  
  return %relu2 : tensor<128x128xf32>
}
