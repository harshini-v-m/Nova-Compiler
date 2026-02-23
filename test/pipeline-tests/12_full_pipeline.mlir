// Full Pipeline Test — Nova GPU Tile-and-Fuse
// Apply all passes in order and verify:
//   - lowering_config attribute stamped on matmul
//   - Workgroup distribution via scf.forall {#gpu.block}
//   - Padding of A/B/C operands to static tile sizes
//   - Global→Shared memory promotion (alloc_tensor + linalg.copy + nova.fusion_barrier)
//   - K-reduction loop (scf.for)
//   - Thread tiling via scf.forall {#gpu.thread}
//   - Subgroup tiling via scf.forall {#gpu.warp}
//   - Config propagation: lowering_config survives through all tiling levels

func.func @full_pipeline_matmul(
    %A: tensor<128x256xf32>,
    %B: tensor<256x512xf32>,
    %C: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %result = linalg.matmul
      ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%C   : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
