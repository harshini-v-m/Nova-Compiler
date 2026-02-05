module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    
    // 1. Identify Ops
    %fill = transform.structured.match ops{["linalg.fill"]} in %root : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %bias = transform.get_consumers_of_result %matmul[0] : (!transform.any_op) -> !transform.any_op
    %relu = transform.get_consumers_of_result %bias[0] : (!transform.any_op) -> !transform.any_op

    // 2. Block-Level Tiling (128x128)
    %block_relu, %block_loop = transform.structured.tile_using_forall %relu 
      tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. Fuse Producers into Block Loop (Corrected Syntax)
    %f_bias_b, %loop_b1 = transform.structured.fuse_into_containing_op %bias into %block_loop 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    %f_matmul_b, %loop_b2 = transform.structured.fuse_into_containing_op %matmul into %loop_b1 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    %f_fill_b, %loop_b3 = transform.structured.fuse_into_containing_op %fill into %loop_b2 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. Thread-Level Tiling (16x16)
    %thread_relu, %thread_loop = transform.structured.tile_using_forall %block_relu
      tile_sizes [16, 16]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 5. INNER FUSION: Pull producers into the Thread Loop (Corrected Syntax)
    %f_bias_t, %loop_t1 = transform.structured.fuse_into_containing_op %f_bias_b into %thread_loop 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    %f_matmul_t, %loop_t2 = transform.structured.fuse_into_containing_op %f_matmul_b into %loop_t1 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      
    %f_fill_t, %loop_t3 = transform.structured.fuse_into_containing_op %f_fill_b into %loop_t2 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 6. K-Dimension Tiling (Reduction)
    %k_tiled_matmul, %k_loop = transform.structured.tile_using_for %f_matmul_t
      tile_sizes [0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 7. Cleanup
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.yield
  }
}
