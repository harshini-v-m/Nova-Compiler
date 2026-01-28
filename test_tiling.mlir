func.func @matmul(%A: tensor<512x512xf32>, %B: tensor<512x512xf32>, %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %D = linalg.matmul ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                     outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %D : tensor<512x512xf32>
}
