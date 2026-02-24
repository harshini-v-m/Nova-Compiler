#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @matmul_bias_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x512xf32>) -> tensor<128x512xf32> {
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4) = (0) to (512) step (128) shared_outs(%arg5 = %arg3) -> (tensor<128x512xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[0, %arg4] [256, 128] [1, 1] : tensor<256x512xf32> to tensor<256x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[0, %arg4] [128, 128] [1, 1] : tensor<128x512xf32> to tensor<128x128xf32>
      %1 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x256xf32>
      %2 = linalg.copy ins(%arg0 : tensor<128x256xf32>) outs(%1 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %3 = nova.fusion_barrier %2 : tensor<128x256xf32>
      %4 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<256x128xf32>) outs(%4 : tensor<256x128xf32>) -> tensor<256x128xf32>
      %6 = nova.fusion_barrier %5 : tensor<256x128xf32>
      %7 = scf.for %arg6 = %c0 to %c256 step %c8 iter_args(%arg7 = %extracted_slice_0) -> (tensor<128x128xf32>) {
        %10 = scf.forall (%arg8, %arg9) = (0, 0) to (128, 128) step (4, 4) shared_outs(%arg10 = %arg7) -> (tensor<128x128xf32>) {
          %extracted_slice_4 = tensor.extract_slice %3[%arg8, %arg6] [4, 8] [1, 1] : tensor<128x256xf32> to tensor<4x8xf32>
          %11 = tensor.empty() : tensor<4x8xf32>
          %12 = linalg.copy ins(%extracted_slice_4 : tensor<4x8xf32>) outs(%11 : tensor<4x8xf32>) -> tensor<4x8xf32>
          %extracted_slice_5 = tensor.extract_slice %6[%arg6, %arg9] [8, 4] [1, 1] : tensor<256x128xf32> to tensor<8x4xf32>
          %13 = tensor.empty() : tensor<8x4xf32>
          %14 = linalg.copy ins(%extracted_slice_5 : tensor<8x4xf32>) outs(%13 : tensor<8x4xf32>) -> tensor<8x4xf32>
          %extracted_slice_6 = tensor.extract_slice %arg10[%arg8, %arg9] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %15 = linalg.matmul {lowering_config = {mma_kind = 0 : i32, promoted_operands = [0, 1], reduction = [0, 0, 8], subgroup = [16, 16, 0], thread = [4, 4, 0], workgroup = [128, 64, 0]}} ins(%12, %14 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%extracted_slice_6 : tensor<4x4xf32>) -> tensor<4x4xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %15 into %arg10[%arg8, %arg9] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<128x128xf32>
          }
        } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
        scf.yield %10 : tensor<128x128xf32>
      }
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4] [128] [1] : tensor<512xf32> to tensor<128xf32>
      %extracted_slice_2 = tensor.extract_slice %arg3[0, %arg4] [128, 128] [1, 1] : tensor<128x512xf32> to tensor<128x128xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %extracted_slice_1 : tensor<128x128xf32>, tensor<128xf32>) outs(%extracted_slice_2 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %10 = arith.addf %in, %in_4 : f32
        linalg.yield %10 : f32
      } -> tensor<128x128xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[0, %arg4] [128, 128] [1, 1] : tensor<128x512xf32> to tensor<128x128xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<128x128xf32>) outs(%extracted_slice_3 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.maximumf %in, %cst : f32
        linalg.yield %10 : f32
      } -> tensor<128x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg5[0, %arg4] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<128x512xf32>
      }
    } {mapping = [#gpu.block<x>]}
    return %0 : tensor<128x512xf32>
  }
}

