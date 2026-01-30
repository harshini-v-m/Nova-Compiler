module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root 
        : (!transform.any_op) -> !transform.any_op
    
    %fills = transform.structured.match ops{["linalg.fill"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %tiled_l3, %loops_l3:3 = transform.structured.tile_using_for %matmuls 
        tile_sizes [256, 256, 256]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    %fused_fill, %new_loop = transform.structured.fuse_into_containing_op %fills into %loops_l3#0
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    %tiled_l2, %loops_l2:3 = transform.structured.tile_using_for %tiled_l3 
        tile_sizes [128, 128, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    %tiled_l1, %loops_l1:3 = transform.structured.tile_using_for %tiled_l2 
        tile_sizes [32, 32, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    %tiled_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_l1 
        tile_sizes [8, 8, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
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
