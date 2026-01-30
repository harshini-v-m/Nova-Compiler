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
    }
    transform.yield
  }
}
