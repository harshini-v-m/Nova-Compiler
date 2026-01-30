module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root 
        : (!transform.any_op) -> !transform.any_op
    
    // Use tile and fuse
    %tiled_op, %loops:3 = transform.structured.fuse %matmuls { tile_sizes = [256, 256, 256] }
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Note: transform.structured.fuse might perform tiling on the consumer and fuse producers.
    // It returns the tiled consumer and the loops.
    // We can then continue tiling the inner op if needed.
    
    transform.yield
  }
}

func.func @matmul_fusion(%A: tensor<512x512xf32>, %B: tensor<512x512xf32>, %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = linalg.fill ins(%c0 : f32) outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
  %D = linalg.matmul ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
                     outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %D : tensor<512x512xf32>
}
