module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    
    // ============================================================
    // CPU-OPTIMIZED TILING for Intel i7-14700K
    // L1d: 38KB/core, L2: 2.5MB/cluster, L3: 33MB shared
    // ============================================================
    
    // Match linalg.matmul (named op, BEFORE generalization)
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root 
        : (!transform.any_op) -> !transform.any_op
    
    // Tile with [M=128, N=128, K=16]
    // Working set: ~80KB (fits in L2 cache)
    // K=16 matches AVX-512 vector width for optimal SIMD
    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %matmuls 
        tile_sizes [128, 128, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
