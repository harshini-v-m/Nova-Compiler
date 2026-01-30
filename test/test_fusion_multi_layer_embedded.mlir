module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // 1. Match Matmuls
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root 
        : (!transform.any_op) -> !transform.any_op
    
    // 2. Iterate over each matmul
    transform.foreach %matmuls : !transform.any_op {
      ^bb1(%matmul_op: !transform.any_op):
      
        // 3. Find Bias (User of Matmul result)
        %bias = transform.get_consumers_of_result %matmul_op[0]
            : (!transform.any_op) -> !transform.any_op

        // 4. Find ReLU (User of Bias result)
        %relu = transform.get_consumers_of_result %bias[0]
            : (!transform.any_op) -> !transform.any_op

        // 5. Tile ReLU (Consumer) and fuse producers (Bias, Matmul, Fill)
        // L3
        %tiled_l3, %loops_l3:2 = transform.structured.fuse %relu { tile_sizes = [256, 256] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        
        // L2: Fuse to pull Matmul into L2 tiles
        %tiled_l2, %loops_l2:2 = transform.structured.fuse %tiled_l3 { tile_sizes = [128, 128] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // L1: Fuse to pull Matmul into L1 tiles
        %tiled_l1, %loops_l1:2 = transform.structured.fuse %tiled_l2 { tile_sizes = [32, 32] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
            
        // Reg: Fuse to pull Matmul into Reg tiles
        %tiled_reg, %loop_reg_i, %loop_reg_j = transform.structured.fuse %tiled_l1 { tile_sizes = [8, 8] }
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // K-Tiling: Match the fused matmul inside the INNERMOST register tile loop
        %fused_matmul = transform.structured.match ops{["linalg.matmul"]} in %loop_reg_j : (!transform.any_op) -> !transform.any_op
        %tiled_reduction, %loop_k = transform.structured.tile_using_for %fused_matmul tile_sizes [0, 0, 64]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    }
        
    transform.yield
  }
}

func.func @mlp_fusion(%A: tensor<128x128xf32>, %B1: tensor<128x128xf32>, %B2: tensor<128x128xf32>, %Bias1: tensor<128x128xf32>, %Bias2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  
  // Layer 1
  %empty1 = tensor.empty() : tensor<128x128xf32>
  %matmul1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%empty1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  
  %bias_res1 = tensor.empty() : tensor<128x128xf32>
  %add1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul1, %Bias1 : tensor<128x128xf32>, tensor<128x128xf32>) 
      outs(%bias_res1 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<128x128xf32>
  
  %relu_res1 = tensor.empty() : tensor<128x128xf32>
  %relu1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add1 : tensor<128x128xf32>) 
      outs(%relu_res1 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<128x128xf32>

  // Layer 2
  %empty2 = tensor.empty() : tensor<128x128xf32>
  %matmul2 = linalg.matmul ins(%relu1, %B2 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%empty2 : tensor<128x128xf32>) -> tensor<128x128xf32>
                     
  %bias_res2 = tensor.empty() : tensor<128x128xf32>
  %add2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul2, %Bias2 : tensor<128x128xf32>, tensor<128x128xf32>) 
      outs(%bias_res2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<128x128xf32>
  
  %relu_res2 = tensor.empty() : tensor<128x128xf32>
  %relu2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add2 : tensor<128x128xf32>) 
      outs(%relu_res2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<128x128xf32>
  
  return %relu2 : tensor<128x128xf32>
}
