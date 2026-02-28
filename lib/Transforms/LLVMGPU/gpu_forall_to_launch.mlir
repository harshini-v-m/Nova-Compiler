// =========================================================================
// GPU Distribute Transform Script (Multi-Dispatch Support)
// =========================================================================

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %module : !transform.any_op {transform.readonly}) {

    // Step 1: Find all functions.
    %funcs = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op

    // Step 2: Match all foralls within each function.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func : !transform.any_op):
        %foralls = transform.structured.match ops{["scf.forall"]} in %func
          : (!transform.any_op) -> !transform.any_op

        // Step 3: Map each forall loop to GPU resources individually.
        // We use failures(suppress) so that inner/thread-mapped foralls
        // which fail block mapping do not abort the script.
        transform.foreach %foralls : !transform.any_op {
          ^bb2(%forall : !transform.any_op):
            transform.sequence %forall : !transform.any_op failures(suppress) {
            ^bb3(%f_inner : !transform.any_op):
              // Map the block forall to a newly generated gpu.launch.
              %gpu_launch = transform.gpu.map_forall_to_blocks %f_inner
                { generate_gpu_launch }
                : (!transform.any_op) -> !transform.any_op

              // Step 4: Map nested thread-mapped scf.forall -> gpu.thread_id.
              transform.gpu.map_nested_forall_to_threads %gpu_launch
                block_dims = [32, 32, 1]
                : (!transform.any_op) -> !transform.any_op
            }
        }
    }

    transform.yield
  }
}
