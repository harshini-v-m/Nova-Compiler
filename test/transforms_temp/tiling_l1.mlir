module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root 
        : (!transform.any_op) -> !transform.any_op
    
    transform.foreach %matmuls : !transform.any_op {
      ^bb1(%matmul_op: !transform.any_op):
        %bias = transform.get_consumers_of_result %matmul_op[0]
            : (!transform.any_op) -> !transform.any_op
        %relu = transform.get_consumers_of_result %bias[0]
            : (!transform.any_op) -> !transform.any_op

        // L3
        %tiled_l3, %loops_l3:2 = transform.structured.fuse %relu { tile_sizes = [256, 256] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        
        // L2
        %tiled_l2, %loops_l2:2 = transform.structured.fuse %tiled_l3 { tile_sizes = [128, 128] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // L1
        %tiled_l1, %loops_l1:2 = transform.structured.fuse %tiled_l2 { tile_sizes = [32, 32] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    }
    transform.yield
  }
}
