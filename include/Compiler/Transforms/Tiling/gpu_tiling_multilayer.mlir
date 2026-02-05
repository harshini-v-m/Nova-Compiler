module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    
    // ===================================================================
    // Strategy: Match all matmul ops, then use foreach to process each
    // ===================================================================
    
    %all_matmul = transform.structured.match ops{["linalg.matmul"]} in %root 
      : (!transform.any_op) -> !transform.any_op
    
    // Process each matmul separately
    transform.foreach %all_matmul : !transform.any_op {
      ^bb0(%single_matmul: !transform.any_op):
      
      // Get consumers: matmul -> bias -> relu
      %bias = transform.get_consumers_of_result %single_matmul[0] 
        : (!transform.any_op) -> !transform.any_op
      %relu = transform.get_consumers_of_result %bias[0] 
        : (!transform.any_op) -> !transform.any_op
      
      // Get fill (producer of matmul's output operand)
      // matmul has 3 operands: [lhs, rhs, out]
      // We need to find the fill that produces the 'out' operand (operand 2)
      %fill = transform.get_producer_of_operand %single_matmul[2]
        : (!transform.any_op) -> !transform.any_op
      
      // ===================================================================
      // Block-Level Tiling (128x128)
      // ===================================================================
      %block_relu, %block_loop = transform.structured.tile_using_forall %relu 
        tile_sizes [128, 128]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // ===================================================================
      // Fuse Producers into Block Loop
      // ===================================================================
      %f_bias_b, %loop_b1 = transform.structured.fuse_into_containing_op %bias into %block_loop 
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      %f_matmul_b, %loop_b2 = transform.structured.fuse_into_containing_op %single_matmul into %loop_b1 
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      // Fuse the specific fill for this matmul
      %f_fill_b, %loop_b3 = transform.structured.fuse_into_containing_op %fill into %loop_b2 
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      // ===================================================================
      // Thread-Level Tiling (16x16)
      // ===================================================================
      %thread_relu, %thread_loop = transform.structured.tile_using_forall %block_relu
        tile_sizes [16, 16]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // ===================================================================
      // Fuse Producers into Thread Loop
      // ===================================================================
      %f_bias_t, %loop_t1 = transform.structured.fuse_into_containing_op %f_bias_b into %thread_loop 
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      %f_matmul_t, %loop_t2 = transform.structured.fuse_into_containing_op %f_matmul_b into %loop_t1 
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      %f_fill_t, %loop_t3 = transform.structured.fuse_into_containing_op %f_fill_b into %loop_t2 
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      // ===================================================================
      // K-Dimension Tiling (Reduction)
      // ===================================================================
      //%k_tiled_matmul, %k_loop = transform.structured.tile_using_for %f_matmul_t
      //  tile_sizes [0, 0, 64]
      //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // ===================================================================
      // Register-Level Tiling
      // ===================================================================
      // Tile matmul to 8x8x8 for register-level optimization
      // Returns 3 loops (M, N, K) since we're tiling all 3 dimensions
      //%reg_matmul, %reg_loops_matmul:3 = transform.structured.tile_using_for %k_tiled_matmul
      //  tile_sizes [8, 8, 8]
      //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

      // Tile fused elementwise operations to 8x8 for register-level optimization
      // Returns 2 loops (M, N) since we're tiling 2 dimensions
      //%reg_fused, %reg_loops_fused:2 = transform.structured.tile_using_for %f_bias_t
      //  tile_sizes [8, 8]
      //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    }

    // ===================================================================
    // Cleanup
    // ===================================================================
    %func = transform.structured.match ops{["func.func"]} in %root 
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}
