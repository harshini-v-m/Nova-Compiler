#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1x1024xf32>) -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = tensor.empty() : tensor<1024x1024xf32>
    %2 = tensor.empty() : tensor<1024x1024xf32>
    %3 = tensor.empty() : tensor<1024x1024xf32>
    %4 = scf.for %arg3 = %c0 to %c1024 step %c256 iter_args(%arg4 = %3) -> (tensor<1024x1024xf32>) {
      %5 = scf.for %arg5 = %c0 to %c1024 step %c256 iter_args(%arg6 = %arg4) -> (tensor<1024x1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [256, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<256x1024xf32>
        %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg5] [1024, 256] [1, 1] : tensor<1024x1024xf32> to tensor<1024x256xf32>
        %extracted_slice_2 = tensor.extract_slice %0[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %extracted_slice_3 = tensor.extract_slice %arg2[0, %arg5] [1, 256] [1, 1] : tensor<1x1024xf32> to tensor<1x256xf32>
        %extracted_slice_4 = tensor.extract_slice %1[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %extracted_slice_5 = tensor.extract_slice %2[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %extracted_slice_6 = tensor.extract_slice %arg6[%arg3, %arg5] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
        %6 = scf.for %arg7 = %c0 to %c256 step %c128 iter_args(%arg8 = %extracted_slice_6) -> (tensor<256x256xf32>) {
          %7 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %arg8) -> (tensor<256x256xf32>) {
            %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg7, 0] [128, 1024] [1, 1] : tensor<256x1024xf32> to tensor<128x1024xf32>
            %extracted_slice_8 = tensor.extract_slice %extracted_slice_1[0, %arg9] [1024, 128] [1, 1] : tensor<1024x256xf32> to tensor<1024x128xf32>
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_2[%arg7, %arg9] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
            %extracted_slice_10 = tensor.extract_slice %extracted_slice_3[0, %arg9] [1, 128] [1, 1] : tensor<1x256xf32> to tensor<1x128xf32>
            %extracted_slice_11 = tensor.extract_slice %extracted_slice_4[%arg7, %arg9] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
            %extracted_slice_12 = tensor.extract_slice %extracted_slice_5[%arg7, %arg9] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
            %extracted_slice_13 = tensor.extract_slice %arg10[%arg7, %arg9] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
            %8 = scf.for %arg11 = %c0 to %c128 step %c32 iter_args(%arg12 = %extracted_slice_13) -> (tensor<128x128xf32>) {
              %9 = scf.for %arg13 = %c0 to %c128 step %c32 iter_args(%arg14 = %arg12) -> (tensor<128x128xf32>) {
                %extracted_slice_15 = tensor.extract_slice %extracted_slice_7[%arg11, 0] [32, 1024] [1, 1] : tensor<128x1024xf32> to tensor<32x1024xf32>
                %extracted_slice_16 = tensor.extract_slice %extracted_slice_8[0, %arg13] [1024, 32] [1, 1] : tensor<1024x128xf32> to tensor<1024x32xf32>
                %extracted_slice_17 = tensor.extract_slice %extracted_slice_9[%arg11, %arg13] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
                %extracted_slice_18 = tensor.extract_slice %extracted_slice_10[0, %arg13] [1, 32] [1, 1] : tensor<1x128xf32> to tensor<1x32xf32>
                %extracted_slice_19 = tensor.extract_slice %extracted_slice_11[%arg11, %arg13] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
                %extracted_slice_20 = tensor.extract_slice %extracted_slice_12[%arg11, %arg13] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
                %extracted_slice_21 = tensor.extract_slice %arg14[%arg11, %arg13] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
                %10 = scf.for %arg15 = %c0 to %c32 step %c8 iter_args(%arg16 = %extracted_slice_21) -> (tensor<32x32xf32>) {
                  %11 = scf.for %arg17 = %c0 to %c32 step %c8 iter_args(%arg18 = %arg16) -> (tensor<32x32xf32>) {
                    %extracted_slice_23 = tensor.extract_slice %extracted_slice_15[%arg15, 0] [8, 1024] [1, 1] : tensor<32x1024xf32> to tensor<8x1024xf32>
                    %extracted_slice_24 = tensor.extract_slice %extracted_slice_16[0, %arg17] [1024, 8] [1, 1] : tensor<1024x32xf32> to tensor<1024x8xf32>
                    %extracted_slice_25 = tensor.extract_slice %extracted_slice_17[%arg15, %arg17] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                    %12 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice_25 : tensor<8x8xf32>) -> tensor<8x8xf32>
                    %13 = scf.for %arg19 = %c0 to %c1024 step %c64 iter_args(%arg20 = %12) -> (tensor<8x8xf32>) {
                      %extracted_slice_31 = tensor.extract_slice %extracted_slice_23[0, %arg19] [8, 64] [1, 1] : tensor<8x1024xf32> to tensor<8x64xf32>
                      %extracted_slice_32 = tensor.extract_slice %extracted_slice_24[%arg19, 0] [64, 8] [1, 1] : tensor<1024x8xf32> to tensor<64x8xf32>
                      %17 = linalg.matmul ins(%extracted_slice_31, %extracted_slice_32 : tensor<8x64xf32>, tensor<64x8xf32>) outs(%arg20 : tensor<8x8xf32>) -> tensor<8x8xf32>
                      scf.yield %17 : tensor<8x8xf32>
                    }
                    %extracted_slice_26 = tensor.extract_slice %extracted_slice_18[0, %arg17] [1, 8] [1, 1] : tensor<1x32xf32> to tensor<1x8xf32>
                    %extracted_slice_27 = tensor.extract_slice %extracted_slice_19[%arg15, %arg17] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_26 : tensor<1x8xf32>) outs(%extracted_slice_27 : tensor<8x8xf32>) {
                    ^bb0(%in: f32, %out: f32):
                      linalg.yield %in : f32
                    } -> tensor<8x8xf32>
                    %extracted_slice_28 = tensor.extract_slice %extracted_slice_20[%arg15, %arg17] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                    %15 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%13, %14 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%extracted_slice_28 : tensor<8x8xf32>) {
                    ^bb0(%in: f32, %in_31: f32, %out: f32):
                      %17 = arith.addf %in, %in_31 : f32
                      linalg.yield %17 : f32
                    } -> tensor<8x8xf32>
                    %extracted_slice_29 = tensor.extract_slice %arg18[%arg15, %arg17] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %cst : tensor<8x8xf32>, tensor<8x8xf32>) outs(%extracted_slice_29 : tensor<8x8xf32>) {
                    ^bb0(%in: f32, %in_31: f32, %out: f32):
                      %17 = arith.maximumf %in, %in_31 : f32
                      linalg.yield %17 : f32
                    } -> tensor<8x8xf32>
                    %inserted_slice_30 = tensor.insert_slice %16 into %arg18[%arg15, %arg17] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x32xf32>
                    scf.yield %inserted_slice_30 : tensor<32x32xf32>
                  }
                  scf.yield %11 : tensor<32x32xf32>
                }
                %inserted_slice_22 = tensor.insert_slice %10 into %arg14[%arg11, %arg13] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
                scf.yield %inserted_slice_22 : tensor<128x128xf32>
              }
              scf.yield %9 : tensor<128x128xf32>
            }
            %inserted_slice_14 = tensor.insert_slice %8 into %arg10[%arg7, %arg9] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x256xf32>
            scf.yield %inserted_slice_14 : tensor<256x256xf32>
          }
          scf.yield %7 : tensor<256x256xf32>
        }
        %inserted_slice = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<1024x1024xf32>
        scf.yield %inserted_slice : tensor<1024x1024xf32>
      }
      scf.yield %5 : tensor<1024x1024xf32>
    }
    return %4 : tensor<1024x1024xf32>
  }
}

