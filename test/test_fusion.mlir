func.func @matmul_fusion(%A: tensor<512x512xf32>, %B: tensor<512x512xf32>, %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %c0 = arith.constant 0.0 : f32
  // Producer: Fill op
  %init = linalg.fill ins(%c0 : f32) outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  // Consumer: Matmul op
  %D = linalg.matmul ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                     outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  return %D : tensor<512x512xf32>
}
