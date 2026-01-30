module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // 1. Find all Matmuls
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    
    // 2. Iterate over each Matmul to fuse its chain
    transform.foreach %matmuls : !transform.any_op {
      ^bb0(%matmul: !transform.any_op):
        // Navigate to Bias (Add)
        %add = transform.get_consumers_of_result %matmul[0] : (!transform.any_op) -> (!transform.any_op)
        
        // Navigate to ReLU (Max)
        %relu = transform.get_consumers_of_result %add[0] : (!transform.any_op) -> (!transform.any_op)
        
        // Tile and Fuse the Terminal Consumer (ReLU)
        %tiled_l3, %loops_l3:2 = transform.structured.fuse %relu { tile_sizes = [256, 256] }
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
          
        // Note: For full multilevel, we would continue tiling %tiled_l3 here.
    }
    transform.yield
  }
}
func.func @two_layer_mlp(%input: tensor<512x512xf32>, %W1: tensor<512x512xf32>, %B1: tensor<1x512xf32>, %W2: tensor<512x512xf32>, %B2: tensor<1x512xf32>) -> tensor<512x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %init1 = tensor.empty() : tensor<512x512xf32>
  %init2 = tensor.empty() : tensor<512x512xf32>
  
  // Layer 1
  %matmul1 = linalg.matmul ins(%input, %W1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%init1 : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  %bias_res1 = tensor.empty() : tensor<512x512xf32>
  %add1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul1, %B1 : tensor<512x512xf32>, tensor<1x512xf32>) 
      outs(%bias_res1 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<512x512xf32>
  
  %relu_res1 = tensor.empty() : tensor<512x512xf32>
  %relu1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add1 : tensor<512x512xf32>) 
      outs(%relu_res1 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<512x512xf32>
  
  // Layer 2
  %matmul2 = linalg.matmul ins(%relu1, %W2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%init2 : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  %bias_res2 = tensor.empty() : tensor<512x512xf32>
  %add2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul2, %B2 : tensor<512x512xf32>, tensor<1x512xf32>) 
      outs(%bias_res2 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
  } -> tensor<512x512xf32>
  
  %relu_res2 = tensor.empty() : tensor<512x512xf32>
  %relu2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add2 : tensor<512x512xf32>) 
      outs(%relu_res2 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
  } -> tensor<512x512xf32>

  return %relu2 : tensor<512x512xf32>
}
