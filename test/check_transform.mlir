module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %root : (!transform.any_op) -> !transform.any_op
    
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [10, 10, 10] 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %fill into %loops#0 
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
func.func @test(%A: tensor<100x100xf32>, %B: tensor<100x100xf32>, %C: tensor<100x100xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%C : tensor<100x100xf32>) -> tensor<100x100xf32>
  %D = linalg.matmul ins(%A, %B : tensor<100x100xf32>, tensor<100x100xf32>) outs(%init : tensor<100x100xf32>) -> tensor<100x100xf32>
  return
}
