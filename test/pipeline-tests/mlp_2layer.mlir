// 2-Layer MLP Pipeline Test
//
// Architecture:
//   input [batch=32, in=256]
//   Layer 1: matmul [32x256] x [256x512] + bias [512] → ReLU → [32x512]
//   Layer 2: matmul [32x512] x [512x128] + bias [128]         → [32x128]
//
// Pipeline: nova-tile-and-distribute → canonicalize → cse → nova-gpu-pad-operands
//
// Expected behavior:
//   - Layer 2 matmul is the root (last compute op = computeOps.back())
//   - Layer 1 matmul + relu are fused as producers into the forall
//   - Bias adds fused as consumers
//   - Full-tile opt: batch=32 < 128 so no zeroing; N=512 > 128 so tiled
//   - Padding: batch 32→128, hidden 512→512 (already aligned), out 128→128

func.func @mlp_2layer(
    %input:  tensor<32x256xf32>,   // [batch, in_features]
    %W1:     tensor<256x512xf32>,  // Layer 1 weights [in, hidden]
    %b1:     tensor<32x512xf32>,   // Layer 1 bias    [batch, hidden]
    %W2:     tensor<512x128xf32>,  // Layer 2 weights [hidden, out]
    %b2:     tensor<32x128xf32>    // Layer 2 bias    [batch, out]
) -> tensor<32x128xf32> {

  %cst = arith.constant 0.0 : f32

  // --- Layer 1 ---
  %empty1 = tensor.empty() : tensor<32x512xf32>
  %zero1  = linalg.fill ins(%cst : f32) outs(%empty1 : tensor<32x512xf32>) -> tensor<32x512xf32>

  // Linear: input [32x256] x W1 [256x512] → [32x512]
  %linear1 = linalg.matmul
      ins(%input, %W1 : tensor<32x256xf32>, tensor<256x512xf32>)
      outs(%zero1 : tensor<32x512xf32>) -> tensor<32x512xf32>

  // Bias add: [32x512] + b1 [32x512]
  %bias1_out = tensor.empty() : tensor<32x512xf32>
  %biased1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%linear1, %b1 : tensor<32x512xf32>, tensor<32x512xf32>)
    outs(%bias1_out : tensor<32x512xf32>) {
  ^bb0(%x: f32, %b: f32, %o: f32):
    %add = arith.addf %x, %b : f32
    linalg.yield %add : f32
  } -> tensor<32x512xf32>

  // ReLU: max(0, x)
  %relu_out = tensor.empty() : tensor<32x512xf32>
  %relu1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%biased1 : tensor<32x512xf32>)
    outs(%relu_out : tensor<32x512xf32>) {
  ^bb0(%x: f32, %o: f32):
    %r = arith.maximumf %x, %cst : f32
    linalg.yield %r : f32
  } -> tensor<32x512xf32>

  // --- Layer 2 ---
  %empty2 = tensor.empty() : tensor<32x128xf32>
  %zero2  = linalg.fill ins(%cst : f32) outs(%empty2 : tensor<32x128xf32>) -> tensor<32x128xf32>

  // Linear: relu [32x512] x W2 [512x128] → [32x128]
  %linear2 = linalg.matmul
      ins(%relu1, %W2 : tensor<32x512xf32>, tensor<512x128xf32>)
      outs(%zero2 : tensor<32x128xf32>) -> tensor<32x128xf32>

  // Bias add: [32x128] + b2 [32x128]
  %out = tensor.empty() : tensor<32x128xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%linear2, %b2 : tensor<32x128xf32>, tensor<32x128xf32>)
    outs(%out : tensor<32x128xf32>) {
  ^bb0(%x: f32, %b: f32, %o: f32):
    %add = arith.addf %x, %b : f32
    linalg.yield %add : f32
  } -> tensor<32x128xf32>

  return %result : tensor<32x128xf32>
}
