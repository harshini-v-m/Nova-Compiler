module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    
    // ===================================================================
    // STEP 1: Match Operations
    // ===================================================================
    
    %fill = transform.structured.match ops{["linalg.fill"]} in %root 
      : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root 
      : (!transform.any_op) -> !transform.any_op
    %bias = transform.get_consumers_of_result %matmul[0] 
      : (!transform.any_op) -> !transform.any_op
    %relu = transform.get_consumers_of_result %bias[0] 
      : (!transform.any_op) -> !transform.any_op

    // ===================================================================
    // STEP 2: Block-Level Tiling (128x128) with GPU Block Mapping
    // ===================================================================

    %block_tiled_relu, %block_loop = transform.structured.tile_using_forall %relu 
      tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ===================================================================
    // STEP 3: Fuse Producers into Block Loop
    // ===================================================================

    // Fuse Bias into block loop
    %block_fused_bias, %block_loop_1 = transform.structured.fuse_into_containing_op %bias into %block_loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // Fuse MatMul into block loop
    %block_fused_matmul, %block_loop_2 = transform.structured.fuse_into_containing_op %matmul into %block_loop_1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // Fuse Fill into block loop
    %block_fused_fill, %block_loop_3 = transform.structured.fuse_into_containing_op %fill into %block_loop_2
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ===================================================================
    // STEP 4: Thread-Level Tiling (16x16) with GPU Thread Mapping
    // ===================================================================

    // Match the fused matmul inside the block loop
    %fused_matmul_in_block = transform.structured.match ops{["linalg.matmul"]} in %block_loop_3
      : (!transform.any_op) -> !transform.any_op

    // Tile the matmul at thread level
    %thread_tiled_matmul, %thread_loop = transform.structured.tile_using_forall %fused_matmul_in_block
      tile_sizes [16, 16]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ===================================================================
    // STEP 5: K-Dimension Tiling (Reduction)
    // ===================================================================

    // Match the thread-tiled matmul
    %thread_tiled_matmul_op = transform.structured.match ops{["linalg.matmul"]} in %thread_loop
      : (!transform.any_op) -> !transform.any_op

    // Tile the reduction dimension
    %k_tiled_matmul, %k_loop = transform.structured.tile_using_for %thread_tiled_matmul_op
      tile_sizes [0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ===================================================================
    // STEP 6: Cleanup
    // ===================================================================

    %func = transform.structured.match ops{["func.func"]} in %root 
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}
