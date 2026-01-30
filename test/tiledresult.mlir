module {
  func.func @main(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32>
    scf.for %arg3 = %c0 to %c1024 step %c256 {
      scf.for %arg4 = %c0 to %c1024 step %c256 {
        %subview = memref.subview %arg0[%arg3, 0] [256, 1024] [1, 1] : memref<1024x1024xf32> to memref<256x1024xf32, strided<[1024, 1], offset: ?>>
        %subview_0 = memref.subview %arg1[0, %arg4] [1024, 256] [1, 1] : memref<1024x1024xf32> to memref<1024x256xf32, strided<[1024, 1], offset: ?>>
        %subview_1 = memref.subview %arg2[0, %arg4] [1, 256] [1, 1] : memref<1x1024xf32> to memref<1x256xf32, strided<[1024, 1], offset: ?>>
        %subview_2 = memref.subview %alloc[%arg3, %arg4] [256, 256] [1, 1] : memref<1024x1024xf32> to memref<256x256xf32, strided<[1024, 1], offset: ?>>
        scf.for %arg5 = %c0 to %c256 step %c128 {
          scf.for %arg6 = %c0 to %c256 step %c128 {
            %subview_3 = memref.subview %subview[%arg5, 0] [128, 1024] [1, 1] : memref<256x1024xf32, strided<[1024, 1], offset: ?>> to memref<128x1024xf32, strided<[1024, 1], offset: ?>>
            %subview_4 = memref.subview %subview_0[0, %arg6] [1024, 128] [1, 1] : memref<1024x256xf32, strided<[1024, 1], offset: ?>> to memref<1024x128xf32, strided<[1024, 1], offset: ?>>
            %subview_5 = memref.subview %subview_1[0, %arg6] [1, 128] [1, 1] : memref<1x256xf32, strided<[1024, 1], offset: ?>> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
            %subview_6 = memref.subview %subview_2[%arg5, %arg6] [128, 128] [1, 1] : memref<256x256xf32, strided<[1024, 1], offset: ?>> to memref<128x128xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg7 = %c0 to %c128 step %c32 {
              scf.for %arg8 = %c0 to %c128 step %c32 {
                %subview_7 = memref.subview %subview_3[%arg7, 0] [32, 1024] [1, 1] : memref<128x1024xf32, strided<[1024, 1], offset: ?>> to memref<32x1024xf32, strided<[1024, 1], offset: ?>>
                %subview_8 = memref.subview %subview_4[0, %arg8] [1024, 32] [1, 1] : memref<1024x128xf32, strided<[1024, 1], offset: ?>> to memref<1024x32xf32, strided<[1024, 1], offset: ?>>
                %subview_9 = memref.subview %subview_5[0, %arg8] [1, 32] [1, 1] : memref<1x128xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
                %subview_10 = memref.subview %subview_6[%arg7, %arg8] [32, 32] [1, 1] : memref<128x128xf32, strided<[1024, 1], offset: ?>> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
                scf.for %arg9 = %c0 to %c32 step %c8 {
                  scf.for %arg10 = %c0 to %c32 step %c8 {
                    %subview_11 = memref.subview %subview_7[%arg9, 0] [8, 1024] [1, 1] : memref<32x1024xf32, strided<[1024, 1], offset: ?>> to memref<8x1024xf32, strided<[1024, 1], offset: ?>>
                    %subview_12 = memref.subview %subview_8[0, %arg10] [1024, 8] [1, 1] : memref<1024x32xf32, strided<[1024, 1], offset: ?>> to memref<1024x8xf32, strided<[1024, 1], offset: ?>>
                    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
                    affine.for %arg11 = 0 to 8 {
                      affine.for %arg12 = 0 to 8 {
                        affine.store %cst, %alloc_13[%arg11, %arg12] : memref<8x8xf32>
                      }
                    }
                    scf.for %arg11 = %c0 to %c1024 step %c64 {
                      %subview_17 = memref.subview %subview_11[0, %arg11] [8, 64] [1, 1] : memref<8x1024xf32, strided<[1024, 1], offset: ?>> to memref<8x64xf32, strided<[1024, 1], offset: ?>>
                      %subview_18 = memref.subview %subview_12[%arg11, 0] [64, 8] [1, 1] : memref<1024x8xf32, strided<[1024, 1], offset: ?>> to memref<64x8xf32, strided<[1024, 1], offset: ?>>
                      affine.for %arg12 = 0 to 8 {
                        affine.for %arg13 = 0 to 8 {
                          affine.for %arg14 = 0 to 64 {
                            %0 = affine.load %subview_17[%arg12, %arg14] : memref<8x64xf32, strided<[1024, 1], offset: ?>>
                            %1 = affine.load %subview_18[%arg14, %arg13] : memref<64x8xf32, strided<[1024, 1], offset: ?>>
                            %2 = affine.load %alloc_13[%arg12, %arg13] : memref<8x8xf32>
                            %3 = arith.mulf %0, %1 : f32
                            %4 = arith.addf %2, %3 : f32
                            affine.store %4, %alloc_13[%arg12, %arg13] : memref<8x8xf32>
                          }
                        }
                      }
                    }
                    %subview_14 = memref.subview %subview_9[0, %arg10] [1, 8] [1, 1] : memref<1x32xf32, strided<[1024, 1], offset: ?>> to memref<1x8xf32, strided<[1024, 1], offset: ?>>
                    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
                    affine.for %arg11 = 0 to 8 {
                      affine.for %arg12 = 0 to 8 {
                        %0 = affine.load %alloc_13[%arg11, %arg12] : memref<8x8xf32>
                        %1 = affine.load %subview_14[0, %arg12] : memref<1x8xf32, strided<[1024, 1], offset: ?>>
                        %2 = arith.addf %0, %1 : f32
                        %3 = arith.maximumf %2, %cst : f32
                        affine.store %3, %alloc_15[%arg11, %arg12] : memref<8x8xf32>
                      }
                    }
                    %subview_16 = memref.subview %subview_10[%arg9, %arg10] [8, 8] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>> to memref<8x8xf32, strided<[1024, 1], offset: ?>>
                    memref.copy %alloc_15, %subview_16 : memref<8x8xf32> to memref<8x8xf32, strided<[1024, 1], offset: ?>>
                    memref.dealloc %alloc_13 : memref<8x8xf32>
                    memref.dealloc %alloc_15 : memref<8x8xf32>
                  }
                }
              }
            }
          }
        }
      }
    }
    return %alloc : memref<1024x1024xf32>
  }
}

