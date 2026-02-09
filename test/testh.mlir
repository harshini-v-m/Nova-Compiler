#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<1024x2048xf32>, %arg1: memref<2048x512xf32>, %arg2: memref<1x512xf32>) -> memref<1024x512xf32> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x512xf32>
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c8, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c64, %arg13 = %c1, %arg14 = %c1) {
      %1 = arith.muli %arg3, %c128 : index
      %2 = arith.muli %arg6, %c8 : index
      %c128_0 = arith.constant 128 : index
      %3 = arith.addi %1, %c128_0 : index
      %c16 = arith.constant 16 : index
      scf.for %arg15 = %1 to %3 step %c16 {
        %c8_1 = arith.constant 8 : index
        %4 = arith.addi %2, %c8_1 : index
        %c16_2 = arith.constant 16 : index
        scf.for %arg16 = %2 to %4 step %c16_2 {
          %5 = vector.transfer_read %arg2[%c0, %arg16], %0 {permutation_map = #map} : memref<1x512xf32>, vector<16x16xf32>
          %c0_3 = arith.constant 0 : index
          %c2048 = arith.constant 2048 : index
          %c8_4 = arith.constant 8 : index
          %6 = scf.for %arg17 = %c0_3 to %c2048 step %c8_4 iter_args(%arg18 = %5) -> (vector<16x16xf32>) {
            %8 = vector.transfer_read %arg0[%arg15, %arg17], %0 : memref<1024x2048xf32>, vector<16x16xf32>
            %9 = vector.transfer_read %arg1[%arg17, %arg16], %0 : memref<2048x512xf32>, vector<16x16xf32>
            %10 = vector.transpose %9, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
            %11 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>, tf32_enabled = true} %8, %10, %arg18 : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
            scf.yield %11 : vector<16x16xf32>
          }
          %7 = arith.maximumf %6, %cst : vector<16x16xf32>
          vector.transfer_write %7, %alloc[%arg15, %arg16] : vector<16x16xf32>, memref<1024x512xf32>
        }
      }
      gpu.terminator
    }
    return %alloc : memref<1024x512xf32>
  }
}