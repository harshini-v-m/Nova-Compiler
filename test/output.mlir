#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
module attributes {gpu.container_module} {
  llvm.func @cudaMemsetAsync(!llvm.ptr, i32, i64, !llvm.ptr) -> i32
  llvm.func @cudaMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
  llvm.func @cudaMallocAsync(!llvm.ptr, i64, !llvm.ptr) -> i32
  llvm.func @cudaMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
  llvm.func @cudaFreeAsync(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @main(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<1>, %arg10: !llvm.ptr<1>, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr<1>, %arg19: !llvm.ptr<1>, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: !llvm.ptr<1>, %arg26: !llvm.ptr<1>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: !llvm.ptr<1>, %arg35: !llvm.ptr<1>, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: !llvm.ptr<1>, %arg44: !llvm.ptr<1>, %arg45: i64, %arg46: i64, %arg47: i64, %arg48: i64, %arg49: i64, %arg50: !llvm.ptr<1>, %arg51: !llvm.ptr<1>, %arg52: i64, %arg53: i64, %arg54: i64, %arg55: !llvm.ptr<1>, %arg56: !llvm.ptr<1>, %arg57: i64, %arg58: i64, %arg59: i64, %arg60: i64, %arg61: i64, %arg62: !llvm.ptr<1>, %arg63: !llvm.ptr<1>, %arg64: i64, %arg65: i64, %arg66: i64, %arg67: !llvm.ptr<1>, %arg68: !llvm.ptr<1>, %arg69: i64, %arg70: i64, %arg71: i64, %arg72: !llvm.ptr<1>, %arg73: !llvm.ptr<1>, %arg74: i64, %arg75: i64, %arg76: i64, %arg77: i64, %arg78: i64, %arg79: !llvm.ptr<1>, %arg80: !llvm.ptr<1>, %arg81: i64, %arg82: i64, %arg83: i64, %arg84: !llvm.ptr<1>, %arg85: !llvm.ptr<1>, %arg86: i64, %arg87: i64, %arg88: i64, %arg89: i64, %arg90: i64, %arg91: !llvm.ptr<1>, %arg92: !llvm.ptr<1>, %arg93: i64, %arg94: i64, %arg95: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.mlir.undef : !llvm.array<2 x i64>
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.undef : !llvm.array<1 x i64>
    %4 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.undef : !llvm.array<3 x i64>
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(384 : i64) : i64
    %14 = llvm.mlir.constant(32 : i64) : i64
    %15 = llvm.mlir.constant(8 : i64) : i64
    %16 = llvm.mlir.constant(4 : i64) : i64
    %17 = llvm.mlir.constant(12288 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %20 = llvm.mlir.constant(8 : index) : i64
    %21 = llvm.mlir.constant(1024 : index) : i64
    %22 = llvm.mlir.constant(8192 : index) : i64
    %23 = llvm.mlir.constant(32 : index) : i64
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(8.192000e+03 : f32) : f32
    %26 = llvm.mlir.poison : f32
    %27 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %28 = llvm.mlir.constant(dense<2.000000e+00> : vector<8xf32>) : vector<8xf32>
    %29 = llvm.mlir.constant(4 : index) : i64
    %30 = llvm.mlir.constant(96 : index) : i64
    %31 = llvm.mlir.constant(128 : index) : i64
    %32 = llvm.mlir.constant(384 : index) : i64
    %33 = llvm.mlir.constant(48 : index) : i64
    %34 = llvm.mlir.constant(3 : index) : i64
    %35 = llvm.mlir.constant(16 : index) : i64
    %36 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    %37 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %arg62, %38[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %arg63, %39[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.insertvalue %arg64, %40[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.insertvalue %arg65, %41[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %arg66, %42[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = builtin.unrealized_conversion_cast %43 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> to memref<32xf32, 1>
    %45 = llvm.insertvalue %arg55, %37[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %arg56, %45[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %arg57, %46[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.insertvalue %arg58, %47[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.insertvalue %arg60, %48[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %arg59, %49[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %arg61, %50[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = builtin.unrealized_conversion_cast %51 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> to memref<384x32xf32, 1>
    %53 = llvm.insertvalue %arg50, %38[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %arg51, %53[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %arg52, %54[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %arg53, %55[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %arg54, %56[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = builtin.unrealized_conversion_cast %57 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> to memref<384xf32, 1>
    %59 = llvm.insertvalue %arg43, %37[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %arg44, %59[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %arg45, %60[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %arg46, %61[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %arg48, %62[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %arg47, %63[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %arg49, %64[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = builtin.unrealized_conversion_cast %65 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x384xf32, 1>
    %67 = llvm.insertvalue %arg34, %36[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %arg35, %67[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %arg36, %68[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.insertvalue %arg37, %69[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.insertvalue %arg40, %70[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = llvm.insertvalue %arg38, %71[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.insertvalue %arg41, %72[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.insertvalue %arg39, %73[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %75 = llvm.insertvalue %arg42, %74[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %76 = builtin.unrealized_conversion_cast %75 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x384xf32, 1>
    %77 = llvm.insertvalue %arg25, %36[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %78 = llvm.insertvalue %arg26, %77[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %79 = llvm.insertvalue %arg27, %78[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %80 = llvm.insertvalue %arg28, %79[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %81 = llvm.insertvalue %arg31, %80[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %82 = llvm.insertvalue %arg29, %81[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %83 = llvm.insertvalue %arg32, %82[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %84 = llvm.insertvalue %arg30, %83[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %85 = llvm.insertvalue %arg33, %84[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %86 = builtin.unrealized_conversion_cast %85 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x32xf32, 1>
    %87 = llvm.insertvalue %arg9, %36[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %88 = llvm.insertvalue %arg10, %87[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %89 = llvm.insertvalue %arg11, %88[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %90 = llvm.insertvalue %arg12, %89[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %91 = llvm.insertvalue %arg15, %90[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %92 = llvm.insertvalue %arg13, %91[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %93 = llvm.insertvalue %arg16, %92[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %94 = llvm.insertvalue %arg14, %93[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %95 = llvm.insertvalue %arg17, %94[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %96 = builtin.unrealized_conversion_cast %95 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x32xf32, 1>
    %97 = llvm.insertvalue %arg0, %36[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %98 = llvm.insertvalue %arg1, %97[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %99 = llvm.insertvalue %arg2, %98[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %100 = llvm.insertvalue %arg3, %99[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %101 = llvm.insertvalue %arg6, %100[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %102 = llvm.insertvalue %arg4, %101[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.insertvalue %arg7, %102[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %104 = llvm.insertvalue %arg5, %103[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %105 = llvm.insertvalue %arg8, %104[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %106 = builtin.unrealized_conversion_cast %105 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x32xf32, 1>
    %107 = builtin.unrealized_conversion_cast %24 : i64 to index
    %108 = llvm.mul %16, %15 : i64
    %109 = llvm.mul %108, %14 : i64
    %110 = llvm.mul %109, %13 : i64
    %111 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %112 = llvm.call @cudaMallocAsync(%111, %110, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %113 = llvm.load %111 : !llvm.ptr -> !llvm.ptr
    %114 = llvm.insertvalue %113, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %115 = llvm.insertvalue %113, %114[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %116 = llvm.insertvalue %9, %115[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %117 = llvm.insertvalue %15, %8[0] : !llvm.array<3 x i64> 
    %118 = llvm.insertvalue %14, %117[1] : !llvm.array<3 x i64> 
    %119 = llvm.insertvalue %13, %118[2] : !llvm.array<3 x i64> 
    %120 = llvm.insertvalue %119, %116[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %121 = llvm.insertvalue %7, %8[2] : !llvm.array<3 x i64> 
    %122 = llvm.mul %7, %13 : i64
    %123 = llvm.insertvalue %122, %121[1] : !llvm.array<3 x i64> 
    %124 = llvm.mul %122, %14 : i64
    %125 = llvm.insertvalue %124, %123[0] : !llvm.array<3 x i64> 
    %126 = llvm.insertvalue %125, %120[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %127 = builtin.unrealized_conversion_cast %126 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x384xf32>
    llvm.br ^bb1(%24 : i64)
  ^bb1(%128: i64):  // 2 preds: ^bb0, ^bb11
    %129 = llvm.icmp "slt" %128, %29 : i64
    llvm.cond_br %129, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%24 : i64)
  ^bb3(%130: i64):  // 2 preds: ^bb2, ^bb10
    %131 = llvm.icmp "slt" %130, %30 : i64
    llvm.cond_br %131, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%24 : i64)
  ^bb5(%132: i64):  // 2 preds: ^bb4, ^bb9
    %133 = builtin.unrealized_conversion_cast %132 : i64 to index
    %134 = llvm.icmp "slt" %132, %20 : i64
    llvm.cond_br %134, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%24 : i64)
  ^bb7(%135: i64):  // 2 preds: ^bb6, ^bb8
    %136 = llvm.icmp "slt" %135, %20 : i64
    llvm.cond_br %136, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %137 = llvm.mul %128, %20 overflow<nsw> : i64
    %138 = llvm.add %135, %137 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    %140 = llvm.mul %130, %29 overflow<nsw> : i64
    %141 = builtin.unrealized_conversion_cast %140 : i64 to index
    %142 = vector.transfer_read %58[%141], %26 : memref<384xf32, 1>, vector<8xf32>
    vector.transfer_write %142, %127[%133, %139, %141] : vector<8xf32>, memref<8x32x384xf32>
    %143 = llvm.add %135, %18 : i64
    llvm.br ^bb7(%143 : i64)
  ^bb9:  // pred: ^bb7
    %144 = llvm.add %132, %18 : i64
    llvm.br ^bb5(%144 : i64)
  ^bb10:  // pred: ^bb5
    %145 = llvm.add %130, %18 : i64
    llvm.br ^bb3(%145 : i64)
  ^bb11:  // pred: ^bb3
    %146 = llvm.add %128, %18 : i64
    llvm.br ^bb1(%146 : i64)
  ^bb12:  // pred: ^bb1
    llvm.br ^bb13(%24 : i64)
  ^bb13(%147: i64):  // 2 preds: ^bb12, ^bb26
    %148 = llvm.icmp "slt" %147, %29 : i64
    llvm.cond_br %148, ^bb14, ^bb27
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%24 : i64)
  ^bb15(%149: i64):  // 2 preds: ^bb14, ^bb25
    %150 = llvm.icmp "slt" %149, %30 : i64
    llvm.cond_br %150, ^bb16, ^bb26
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%24 : i64)
  ^bb17(%151: i64):  // 2 preds: ^bb16, ^bb24
    %152 = builtin.unrealized_conversion_cast %151 : i64 to index
    %153 = llvm.icmp "slt" %151, %20 : i64
    llvm.cond_br %153, ^bb18, ^bb25
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%24 : i64)
  ^bb19(%154: i64):  // 2 preds: ^bb18, ^bb23
    %155 = llvm.icmp "slt" %154, %20 : i64
    llvm.cond_br %155, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    %156 = llvm.mul %147, %20 overflow<nsw> : i64
    %157 = llvm.add %154, %156 : i64
    %158 = builtin.unrealized_conversion_cast %157 : i64 to index
    %159 = llvm.mul %149, %29 overflow<nsw> : i64
    %160 = builtin.unrealized_conversion_cast %159 : i64 to index
    %161 = vector.transfer_read %127[%152, %158, %160], %26 : memref<8x32x384xf32>, vector<8xf32>
    llvm.br ^bb21(%24, %161 : i64, vector<8xf32>)
  ^bb21(%162: i64, %163: vector<8xf32>):  // 2 preds: ^bb20, ^bb22
    %164 = builtin.unrealized_conversion_cast %162 : i64 to index
    %165 = llvm.icmp "slt" %162, %23 : i64
    llvm.cond_br %165, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %166 = llvm.mul %151, %21 : i64
    %167 = llvm.mul %157, %23 : i64
    %168 = llvm.add %166, %167 : i64
    %169 = llvm.add %168, %162 : i64
    %170 = llvm.getelementptr %arg1[%169] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %171 = llvm.alloca %12 x vector<1xf32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %172 = llvm.addrspacecast %170 : !llvm.ptr<1> to !llvm.ptr
    %173 = llvm.call @cudaMemcpy(%171, %172, %16, %6) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %174 = llvm.load %171 : !llvm.ptr -> vector<1xf32>
    %175 = vector.broadcast %174 : vector<1xf32> to vector<8xf32>
    %176 = vector.transfer_read %66[%164, %160], %26 : memref<32x384xf32, 1>, vector<8xf32>
    %177 = llvm.fmul %175, %176 : vector<8xf32>
    %178 = llvm.fadd %163, %177 : vector<8xf32>
    %179 = llvm.add %162, %18 : i64
    llvm.br ^bb21(%179, %178 : i64, vector<8xf32>)
  ^bb23:  // pred: ^bb21
    vector.transfer_write %163, %127[%152, %158, %160] : vector<8xf32>, memref<8x32x384xf32>
    %180 = llvm.add %154, %18 : i64
    llvm.br ^bb19(%180 : i64)
  ^bb24:  // pred: ^bb19
    %181 = llvm.add %151, %18 : i64
    llvm.br ^bb17(%181 : i64)
  ^bb25:  // pred: ^bb17
    %182 = llvm.add %149, %18 : i64
    llvm.br ^bb15(%182 : i64)
  ^bb26:  // pred: ^bb15
    %183 = llvm.add %147, %18 : i64
    llvm.br ^bb13(%183 : i64)
  ^bb27:  // pred: ^bb13
    %184 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %185 = llvm.call @cudaMallocAsync(%184, %110, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %186 = llvm.load %184 : !llvm.ptr -> !llvm.ptr
    %187 = llvm.insertvalue %186, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %188 = llvm.insertvalue %186, %187[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %189 = llvm.insertvalue %9, %188[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %190 = llvm.insertvalue %119, %189[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %191 = llvm.insertvalue %125, %190[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %192 = builtin.unrealized_conversion_cast %191 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x384xf32>
    llvm.br ^bb28(%24 : i64)
  ^bb28(%193: i64):  // 2 preds: ^bb27, ^bb38
    %194 = llvm.icmp "slt" %193, %29 : i64
    llvm.cond_br %194, ^bb29, ^bb39
  ^bb29:  // pred: ^bb28
    llvm.br ^bb30(%24 : i64)
  ^bb30(%195: i64):  // 2 preds: ^bb29, ^bb37
    %196 = llvm.icmp "slt" %195, %30 : i64
    llvm.cond_br %196, ^bb31, ^bb38
  ^bb31:  // pred: ^bb30
    llvm.br ^bb32(%24 : i64)
  ^bb32(%197: i64):  // 2 preds: ^bb31, ^bb36
    %198 = builtin.unrealized_conversion_cast %197 : i64 to index
    %199 = llvm.icmp "slt" %197, %20 : i64
    llvm.cond_br %199, ^bb33, ^bb37
  ^bb33:  // pred: ^bb32
    llvm.br ^bb34(%24 : i64)
  ^bb34(%200: i64):  // 2 preds: ^bb33, ^bb35
    %201 = llvm.icmp "slt" %200, %20 : i64
    llvm.cond_br %201, ^bb35, ^bb36
  ^bb35:  // pred: ^bb34
    %202 = llvm.mul %193, %20 overflow<nsw> : i64
    %203 = llvm.add %200, %202 : i64
    %204 = builtin.unrealized_conversion_cast %203 : i64 to index
    %205 = llvm.mul %195, %29 overflow<nsw> : i64
    %206 = builtin.unrealized_conversion_cast %205 : i64 to index
    %207 = vector.transfer_read %127[%198, %204, %206], %26 : memref<8x32x384xf32>, vector<8xf32>
    %208 = llvm.intr.maximum(%207, %27) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    vector.transfer_write %208, %192[%198, %204, %206] : vector<8xf32>, memref<8x32x384xf32>
    %209 = llvm.add %200, %18 : i64
    llvm.br ^bb34(%209 : i64)
  ^bb36:  // pred: ^bb34
    %210 = llvm.add %197, %18 : i64
    llvm.br ^bb32(%210 : i64)
  ^bb37:  // pred: ^bb32
    %211 = llvm.add %195, %18 : i64
    llvm.br ^bb30(%211 : i64)
  ^bb38:  // pred: ^bb30
    %212 = llvm.add %193, %18 : i64
    llvm.br ^bb28(%212 : i64)
  ^bb39:  // pred: ^bb28
    %213 = llvm.mul %109, %14 : i64
    %214 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %215 = llvm.call @cudaMallocAsync(%214, %213, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %216 = llvm.load %214 : !llvm.ptr -> !llvm.ptr
    %217 = llvm.insertvalue %216, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %218 = llvm.insertvalue %216, %217[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %219 = llvm.insertvalue %9, %218[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %220 = llvm.insertvalue %14, %118[2] : !llvm.array<3 x i64> 
    %221 = llvm.insertvalue %220, %219[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %222 = llvm.mul %7, %14 : i64
    %223 = llvm.insertvalue %222, %121[1] : !llvm.array<3 x i64> 
    %224 = llvm.mul %222, %14 : i64
    %225 = llvm.insertvalue %224, %223[0] : !llvm.array<3 x i64> 
    %226 = llvm.insertvalue %225, %221[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %227 = builtin.unrealized_conversion_cast %226 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x32xf32>
    llvm.br ^bb40(%24 : i64)
  ^bb40(%228: i64):  // 2 preds: ^bb39, ^bb50
    %229 = llvm.icmp "slt" %228, %29 : i64
    llvm.cond_br %229, ^bb41, ^bb51
  ^bb41:  // pred: ^bb40
    llvm.br ^bb42(%24 : i64)
  ^bb42(%230: i64):  // 2 preds: ^bb41, ^bb49
    %231 = llvm.icmp "slt" %230, %20 : i64
    llvm.cond_br %231, ^bb43, ^bb50
  ^bb43:  // pred: ^bb42
    llvm.br ^bb44(%24 : i64)
  ^bb44(%232: i64):  // 2 preds: ^bb43, ^bb48
    %233 = builtin.unrealized_conversion_cast %232 : i64 to index
    %234 = llvm.icmp "slt" %232, %20 : i64
    llvm.cond_br %234, ^bb45, ^bb49
  ^bb45:  // pred: ^bb44
    llvm.br ^bb46(%24 : i64)
  ^bb46(%235: i64):  // 2 preds: ^bb45, ^bb47
    %236 = llvm.icmp "slt" %235, %20 : i64
    llvm.cond_br %236, ^bb47, ^bb48
  ^bb47:  // pred: ^bb46
    %237 = llvm.mul %228, %20 overflow<nsw> : i64
    %238 = llvm.add %235, %237 : i64
    %239 = builtin.unrealized_conversion_cast %238 : i64 to index
    %240 = llvm.mul %230, %29 overflow<nsw> : i64
    %241 = builtin.unrealized_conversion_cast %240 : i64 to index
    %242 = vector.transfer_read %44[%241], %26 : memref<32xf32, 1>, vector<8xf32>
    vector.transfer_write %242, %227[%233, %239, %241] : vector<8xf32>, memref<8x32x32xf32>
    %243 = llvm.add %235, %18 : i64
    llvm.br ^bb46(%243 : i64)
  ^bb48:  // pred: ^bb46
    %244 = llvm.add %232, %18 : i64
    llvm.br ^bb44(%244 : i64)
  ^bb49:  // pred: ^bb44
    %245 = llvm.add %230, %18 : i64
    llvm.br ^bb42(%245 : i64)
  ^bb50:  // pred: ^bb42
    %246 = llvm.add %228, %18 : i64
    llvm.br ^bb40(%246 : i64)
  ^bb51:  // pred: ^bb40
    llvm.br ^bb52(%24 : i64)
  ^bb52(%247: i64):  // 2 preds: ^bb51, ^bb65
    %248 = llvm.icmp "slt" %247, %29 : i64
    llvm.cond_br %248, ^bb53, ^bb66
  ^bb53:  // pred: ^bb52
    llvm.br ^bb54(%24 : i64)
  ^bb54(%249: i64):  // 2 preds: ^bb53, ^bb64
    %250 = llvm.icmp "slt" %249, %20 : i64
    llvm.cond_br %250, ^bb55, ^bb65
  ^bb55:  // pred: ^bb54
    llvm.br ^bb56(%24 : i64)
  ^bb56(%251: i64):  // 2 preds: ^bb55, ^bb63
    %252 = builtin.unrealized_conversion_cast %251 : i64 to index
    %253 = llvm.icmp "slt" %251, %20 : i64
    llvm.cond_br %253, ^bb57, ^bb64
  ^bb57:  // pred: ^bb56
    llvm.br ^bb58(%24 : i64)
  ^bb58(%254: i64):  // 2 preds: ^bb57, ^bb62
    %255 = llvm.icmp "slt" %254, %20 : i64
    llvm.cond_br %255, ^bb59, ^bb63
  ^bb59:  // pred: ^bb58
    %256 = llvm.mul %247, %20 overflow<nsw> : i64
    %257 = llvm.add %254, %256 : i64
    %258 = builtin.unrealized_conversion_cast %257 : i64 to index
    %259 = llvm.mul %249, %29 overflow<nsw> : i64
    %260 = builtin.unrealized_conversion_cast %259 : i64 to index
    %261 = vector.transfer_read %227[%252, %258, %260], %26 : memref<8x32x32xf32>, vector<8xf32>
    llvm.br ^bb60(%24, %261 : i64, vector<8xf32>)
  ^bb60(%262: i64, %263: vector<8xf32>):  // 2 preds: ^bb59, ^bb61
    %264 = builtin.unrealized_conversion_cast %262 : i64 to index
    %265 = llvm.icmp "slt" %262, %32 : i64
    llvm.cond_br %265, ^bb61, ^bb62
  ^bb61:  // pred: ^bb60
    %266 = llvm.mul %251, %17 : i64
    %267 = llvm.mul %257, %32 : i64
    %268 = llvm.add %266, %267 : i64
    %269 = llvm.add %268, %262 : i64
    %270 = llvm.getelementptr %186[%269] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %271 = llvm.load %270 {alignment = 4 : i64} : !llvm.ptr -> vector<1xf32>
    %272 = vector.broadcast %271 : vector<1xf32> to vector<8xf32>
    %273 = vector.transfer_read %52[%264, %260], %26 : memref<384x32xf32, 1>, vector<8xf32>
    %274 = llvm.fmul %272, %273 : vector<8xf32>
    %275 = llvm.fadd %263, %274 : vector<8xf32>
    %276 = llvm.add %262, %18 : i64
    llvm.br ^bb60(%276, %275 : i64, vector<8xf32>)
  ^bb62:  // pred: ^bb60
    vector.transfer_write %263, %227[%252, %258, %260] : vector<8xf32>, memref<8x32x32xf32>
    %277 = llvm.add %254, %18 : i64
    llvm.br ^bb58(%277 : i64)
  ^bb63:  // pred: ^bb58
    %278 = llvm.add %251, %18 : i64
    llvm.br ^bb56(%278 : i64)
  ^bb64:  // pred: ^bb56
    %279 = llvm.add %249, %18 : i64
    llvm.br ^bb54(%279 : i64)
  ^bb65:  // pred: ^bb54
    %280 = llvm.add %247, %18 : i64
    llvm.br ^bb52(%280 : i64)
  ^bb66:  // pred: ^bb52
    %281 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %282 = llvm.call @cudaMallocAsync(%281, %213, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %283 = llvm.load %281 : !llvm.ptr -> !llvm.ptr
    %284 = llvm.insertvalue %283, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %285 = llvm.insertvalue %283, %284[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %286 = llvm.insertvalue %9, %285[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %287 = llvm.insertvalue %220, %286[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %288 = llvm.insertvalue %225, %287[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %289 = builtin.unrealized_conversion_cast %288 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x32xf32>
    llvm.br ^bb67(%24 : i64)
  ^bb67(%290: i64):  // 2 preds: ^bb66, ^bb77
    %291 = llvm.icmp "slt" %290, %29 : i64
    llvm.cond_br %291, ^bb68, ^bb78
  ^bb68:  // pred: ^bb67
    llvm.br ^bb69(%24 : i64)
  ^bb69(%292: i64):  // 2 preds: ^bb68, ^bb76
    %293 = llvm.icmp "slt" %292, %20 : i64
    llvm.cond_br %293, ^bb70, ^bb77
  ^bb70:  // pred: ^bb69
    llvm.br ^bb71(%24 : i64)
  ^bb71(%294: i64):  // 2 preds: ^bb70, ^bb75
    %295 = builtin.unrealized_conversion_cast %294 : i64 to index
    %296 = llvm.icmp "slt" %294, %20 : i64
    llvm.cond_br %296, ^bb72, ^bb76
  ^bb72:  // pred: ^bb71
    llvm.br ^bb73(%24 : i64)
  ^bb73(%297: i64):  // 2 preds: ^bb72, ^bb74
    %298 = llvm.icmp "slt" %297, %20 : i64
    llvm.cond_br %298, ^bb74, ^bb75
  ^bb74:  // pred: ^bb73
    %299 = llvm.mul %290, %20 overflow<nsw> : i64
    %300 = llvm.add %297, %299 : i64
    %301 = builtin.unrealized_conversion_cast %300 : i64 to index
    %302 = llvm.mul %292, %29 overflow<nsw> : i64
    %303 = builtin.unrealized_conversion_cast %302 : i64 to index
    %304 = vector.transfer_read %227[%295, %301, %303], %26 : memref<8x32x32xf32>, vector<8xf32>
    %305 = vector.transfer_read %96[%295, %301, %303], %26 : memref<8x32x32xf32, 1>, vector<8xf32>
    %306 = llvm.fsub %304, %305 : vector<8xf32>
    %307 = llvm.intr.pow(%306, %28) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    vector.transfer_write %307, %289[%295, %301, %303] : vector<8xf32>, memref<8x32x32xf32>
    %308 = llvm.add %297, %18 : i64
    llvm.br ^bb73(%308 : i64)
  ^bb75:  // pred: ^bb73
    %309 = llvm.add %294, %18 : i64
    llvm.br ^bb71(%309 : i64)
  ^bb76:  // pred: ^bb71
    %310 = llvm.add %292, %18 : i64
    llvm.br ^bb69(%310 : i64)
  ^bb77:  // pred: ^bb69
    %311 = llvm.add %290, %18 : i64
    llvm.br ^bb67(%311 : i64)
  ^bb78:  // pred: ^bb67
    %312 = llvm.mul %16, %7 : i64
    %313 = llvm.alloca %12 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %314 = llvm.call @cudaMallocAsync(%313, %312, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %315 = llvm.load %313 : !llvm.ptr -> !llvm.ptr<1>
    %316 = llvm.addrspacecast %315 : !llvm.ptr<1> to !llvm.ptr
    %317 = llvm.call @cudaMemsetAsync(%316, %5, %312, %11) : (!llvm.ptr, i32, i64, !llvm.ptr) -> i32
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%20, %18, %18) threads in (%21, %18, %18) : i64 args(%23 : i64, %283 : !llvm.ptr, %283 : !llvm.ptr, %9 : i64, %15 : i64, %14 : i64, %14 : i64, %224 : i64, %222 : i64, %7 : i64, %22 : i64, %19 : f32, %24 : i64, %315 : !llvm.ptr<1>, %315 : !llvm.ptr<1>, %9 : i64, %7 : i64, %7 : i64)
    gpu.launch_func  @main_kernel_0::@main_kernel_0 blocks in (%18, %18, %18) threads in (%18, %18, %18) : i64 args(%315 : !llvm.ptr<1>, %315 : !llvm.ptr<1>, %9 : i64, %7 : i64, %7 : i64, %24 : i64, %25 : f32)
    %318 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %319 = llvm.call @cudaMallocAsync(%318, %213, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %320 = llvm.load %318 : !llvm.ptr -> !llvm.ptr
    %321 = llvm.insertvalue %320, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %322 = llvm.insertvalue %320, %321[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %323 = llvm.insertvalue %9, %322[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %324 = llvm.insertvalue %220, %323[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %325 = llvm.insertvalue %225, %324[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %326 = builtin.unrealized_conversion_cast %325 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x32xf32>
    %327 = llvm.add %24, %24 overflow<nsw, nuw> : i64
    %328 = llvm.getelementptr inbounds|nuw %arg19[%327] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %329 = llvm.alloca %12 x f32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %330 = llvm.addrspacecast %328 : !llvm.ptr<1> to !llvm.ptr
    %331 = llvm.call @cudaMemcpy(%329, %330, %16, %6) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %332 = llvm.load %329 : !llvm.ptr -> f32
    %333 = vector.broadcast %332 : f32 to vector<8xf32>
    llvm.br ^bb79(%24 : i64)
  ^bb79(%334: i64):  // 2 preds: ^bb78, ^bb89
    %335 = llvm.icmp "slt" %334, %29 : i64
    llvm.cond_br %335, ^bb80, ^bb90
  ^bb80:  // pred: ^bb79
    llvm.br ^bb81(%24 : i64)
  ^bb81(%336: i64):  // 2 preds: ^bb80, ^bb88
    %337 = llvm.icmp "slt" %336, %20 : i64
    llvm.cond_br %337, ^bb82, ^bb89
  ^bb82:  // pred: ^bb81
    llvm.br ^bb83(%24 : i64)
  ^bb83(%338: i64):  // 2 preds: ^bb82, ^bb87
    %339 = builtin.unrealized_conversion_cast %338 : i64 to index
    %340 = llvm.icmp "slt" %338, %20 : i64
    llvm.cond_br %340, ^bb84, ^bb88
  ^bb84:  // pred: ^bb83
    llvm.br ^bb85(%24 : i64)
  ^bb85(%341: i64):  // 2 preds: ^bb84, ^bb86
    %342 = llvm.icmp "slt" %341, %20 : i64
    llvm.cond_br %342, ^bb86, ^bb87
  ^bb86:  // pred: ^bb85
    %343 = llvm.mul %334, %20 overflow<nsw> : i64
    %344 = llvm.add %341, %343 : i64
    %345 = builtin.unrealized_conversion_cast %344 : i64 to index
    %346 = llvm.mul %336, %29 overflow<nsw> : i64
    %347 = builtin.unrealized_conversion_cast %346 : i64 to index
    %348 = vector.transfer_read %227[%339, %345, %347], %26 : memref<8x32x32xf32>, vector<8xf32>
    %349 = vector.transfer_read %96[%339, %345, %347], %26 : memref<8x32x32xf32, 1>, vector<8xf32>
    %350 = vector.transfer_read %86[%339, %345, %347], %26 : memref<8x32x32xf32, 1>, vector<8xf32>
    %351 = llvm.fsub %348, %349 : vector<8xf32>
    %352 = llvm.fmul %351, %350 : vector<8xf32>
    %353 = llvm.fmul %352, %333 : vector<8xf32>
    vector.transfer_write %353, %326[%339, %345, %347] : vector<8xf32>, memref<8x32x32xf32>
    %354 = llvm.add %341, %18 : i64
    llvm.br ^bb85(%354 : i64)
  ^bb87:  // pred: ^bb85
    %355 = llvm.add %338, %18 : i64
    llvm.br ^bb83(%355 : i64)
  ^bb88:  // pred: ^bb83
    %356 = llvm.add %336, %18 : i64
    llvm.br ^bb81(%356 : i64)
  ^bb89:  // pred: ^bb81
    %357 = llvm.add %334, %18 : i64
    llvm.br ^bb79(%357 : i64)
  ^bb90:  // pred: ^bb79
    %358 = llvm.mul %16, %14 : i64
    %359 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %360 = llvm.call @cudaMallocAsync(%359, %358, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %361 = llvm.load %359 : !llvm.ptr -> !llvm.ptr
    %362 = llvm.insertvalue %361, %4[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %363 = llvm.insertvalue %361, %362[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %364 = llvm.insertvalue %9, %363[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %365 = llvm.insertvalue %14, %3[0] : !llvm.array<1 x i64> 
    %366 = llvm.insertvalue %365, %364[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %367 = llvm.insertvalue %7, %3[0] : !llvm.array<1 x i64> 
    %368 = llvm.insertvalue %367, %366[4] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %369 = builtin.unrealized_conversion_cast %368 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<32xf32>
    llvm.br ^bb91(%24 : i64)
  ^bb91(%370: i64):  // 2 preds: ^bb90, ^bb92
    %371 = llvm.icmp "slt" %370, %29 : i64
    llvm.cond_br %371, ^bb92, ^bb93
  ^bb92:  // pred: ^bb91
    %372 = llvm.mul %370, %20 overflow<nsw> : i64
    %373 = builtin.unrealized_conversion_cast %372 : i64 to index
    vector.transfer_write %27, %369[%373] : vector<8xf32>, memref<32xf32>
    %374 = llvm.add %370, %18 : i64
    llvm.br ^bb91(%374 : i64)
  ^bb93:  // pred: ^bb91
    llvm.br ^bb94(%24 : i64)
  ^bb94(%375: i64):  // 2 preds: ^bb93, ^bb101
    %376 = builtin.unrealized_conversion_cast %375 : i64 to index
    %377 = llvm.icmp "slt" %375, %23 : i64
    llvm.cond_br %377, ^bb95, ^bb102
  ^bb95:  // pred: ^bb94
    llvm.br ^bb96(%24 : i64)
  ^bb96(%378: i64):  // 2 preds: ^bb95, ^bb100
    %379 = builtin.unrealized_conversion_cast %378 : i64 to index
    %380 = llvm.icmp "slt" %378, %20 : i64
    llvm.cond_br %380, ^bb97, ^bb101
  ^bb97:  // pred: ^bb96
    %381 = llvm.getelementptr inbounds|nuw %361[%375] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %382 = llvm.load %381 : !llvm.ptr -> f32
    llvm.br ^bb98(%24, %27 : i64, vector<8xf32>)
  ^bb98(%383: i64, %384: vector<8xf32>):  // 2 preds: ^bb97, ^bb99
    %385 = llvm.icmp "slt" %383, %29 : i64
    llvm.cond_br %385, ^bb99, ^bb100
  ^bb99:  // pred: ^bb98
    %386 = llvm.mul %383, %20 overflow<nsw> : i64
    %387 = builtin.unrealized_conversion_cast %386 : i64 to index
    %388 = vector.transfer_read %326[%379, %387, %376], %26 {permutation_map = #map} : memref<8x32x32xf32>, vector<8xf32>
    %389 = llvm.fadd %388, %384 : vector<8xf32>
    %390 = llvm.add %383, %18 : i64
    llvm.br ^bb98(%390, %389 : i64, vector<8xf32>)
  ^bb100:  // pred: ^bb98
    %391 = "llvm.intr.vector.reduce.fadd"(%19, %384) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %392 = llvm.fadd %391, %382 : f32
    llvm.store %392, %381 : f32, !llvm.ptr
    %393 = llvm.add %378, %18 : i64
    llvm.br ^bb96(%393 : i64)
  ^bb101:  // pred: ^bb96
    %394 = llvm.add %375, %18 : i64
    llvm.br ^bb94(%394 : i64)
  ^bb102:  // pred: ^bb94
    %395 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %396 = llvm.call @cudaMallocAsync(%395, %110, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %397 = llvm.load %395 : !llvm.ptr -> !llvm.ptr
    %398 = llvm.insertvalue %397, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %399 = llvm.insertvalue %397, %398[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %400 = llvm.insertvalue %9, %399[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %401 = llvm.insertvalue %119, %400[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %402 = llvm.insertvalue %125, %401[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %403 = builtin.unrealized_conversion_cast %402 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x384xf32>
    llvm.br ^bb103(%24 : i64)
  ^bb103(%404: i64):  // 2 preds: ^bb102, ^bb113
    %405 = llvm.icmp "slt" %404, %29 : i64
    llvm.cond_br %405, ^bb104, ^bb114
  ^bb104:  // pred: ^bb103
    llvm.br ^bb105(%24 : i64)
  ^bb105(%406: i64):  // 2 preds: ^bb104, ^bb112
    %407 = llvm.icmp "slt" %406, %30 : i64
    llvm.cond_br %407, ^bb106, ^bb113
  ^bb106:  // pred: ^bb105
    llvm.br ^bb107(%24 : i64)
  ^bb107(%408: i64):  // 2 preds: ^bb106, ^bb111
    %409 = builtin.unrealized_conversion_cast %408 : i64 to index
    %410 = llvm.icmp "slt" %408, %20 : i64
    llvm.cond_br %410, ^bb108, ^bb112
  ^bb108:  // pred: ^bb107
    llvm.br ^bb109(%24 : i64)
  ^bb109(%411: i64):  // 2 preds: ^bb108, ^bb110
    %412 = llvm.icmp "slt" %411, %20 : i64
    llvm.cond_br %412, ^bb110, ^bb111
  ^bb110:  // pred: ^bb109
    %413 = llvm.mul %404, %20 overflow<nsw> : i64
    %414 = llvm.add %411, %413 : i64
    %415 = builtin.unrealized_conversion_cast %414 : i64 to index
    %416 = llvm.mul %406, %29 overflow<nsw> : i64
    %417 = builtin.unrealized_conversion_cast %416 : i64 to index
    vector.transfer_write %27, %403[%409, %415, %417] : vector<8xf32>, memref<8x32x384xf32>
    %418 = llvm.add %411, %18 : i64
    llvm.br ^bb109(%418 : i64)
  ^bb111:  // pred: ^bb109
    %419 = llvm.add %408, %18 : i64
    llvm.br ^bb107(%419 : i64)
  ^bb112:  // pred: ^bb107
    %420 = llvm.add %406, %18 : i64
    llvm.br ^bb105(%420 : i64)
  ^bb113:  // pred: ^bb105
    %421 = llvm.add %404, %18 : i64
    llvm.br ^bb103(%421 : i64)
  ^bb114:  // pred: ^bb103
    llvm.br ^bb115(%24 : i64)
  ^bb115(%422: i64):  // 2 preds: ^bb114, ^bb131
    %423 = llvm.icmp "slt" %422, %29 : i64
    llvm.cond_br %423, ^bb116, ^bb132
  ^bb116:  // pred: ^bb115
    llvm.br ^bb117(%24 : i64)
  ^bb117(%424: i64):  // 2 preds: ^bb116, ^bb130
    %425 = llvm.icmp "slt" %424, %30 : i64
    llvm.cond_br %425, ^bb118, ^bb131
  ^bb118:  // pred: ^bb117
    llvm.br ^bb119(%24 : i64)
  ^bb119(%426: i64):  // 2 preds: ^bb118, ^bb129
    %427 = builtin.unrealized_conversion_cast %426 : i64 to index
    %428 = llvm.icmp "slt" %426, %20 : i64
    llvm.cond_br %428, ^bb120, ^bb130
  ^bb120:  // pred: ^bb119
    llvm.br ^bb121(%24 : i64)
  ^bb121(%429: i64):  // 2 preds: ^bb120, ^bb128
    %430 = llvm.icmp "slt" %429, %20 : i64
    llvm.cond_br %430, ^bb122, ^bb129
  ^bb122:  // pred: ^bb121
    %431 = llvm.mul %422, %20 overflow<nsw> : i64
    %432 = llvm.add %429, %431 : i64
    %433 = builtin.unrealized_conversion_cast %432 : i64 to index
    llvm.br ^bb123(%24 : i64)
  ^bb123(%434: i64):  // 2 preds: ^bb122, ^bb127
    %435 = llvm.icmp "slt" %434, %29 : i64
    llvm.cond_br %435, ^bb124, ^bb128
  ^bb124:  // pred: ^bb123
    %436 = llvm.mul %424, %29 overflow<nsw> : i64
    %437 = llvm.add %434, %436 : i64
    %438 = builtin.unrealized_conversion_cast %437 : i64 to index
    %439 = llvm.mul %426, %17 overflow<nsw, nuw> : i64
    %440 = llvm.mul %432, %32 overflow<nsw, nuw> : i64
    %441 = llvm.add %439, %440 overflow<nsw, nuw> : i64
    %442 = llvm.add %441, %437 overflow<nsw, nuw> : i64
    %443 = llvm.getelementptr inbounds|nuw %397[%442] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %444 = llvm.load %443 : !llvm.ptr -> f32
    llvm.br ^bb125(%24, %27 : i64, vector<8xf32>)
  ^bb125(%445: i64, %446: vector<8xf32>):  // 2 preds: ^bb124, ^bb126
    %447 = llvm.icmp "slt" %445, %29 : i64
    llvm.cond_br %447, ^bb126, ^bb127
  ^bb126:  // pred: ^bb125
    %448 = llvm.mul %445, %20 overflow<nsw> : i64
    %449 = builtin.unrealized_conversion_cast %448 : i64 to index
    %450 = vector.transfer_read %326[%427, %433, %449], %26 : memref<8x32x32xf32>, vector<8xf32>
    %451 = vector.transfer_read %52[%438, %449], %26 : memref<384x32xf32, 1>, vector<8xf32>
    %452 = llvm.fmul %450, %451 : vector<8xf32>
    %453 = llvm.fadd %446, %452 : vector<8xf32>
    %454 = llvm.add %445, %18 : i64
    llvm.br ^bb125(%454, %453 : i64, vector<8xf32>)
  ^bb127:  // pred: ^bb125
    %455 = "llvm.intr.vector.reduce.fadd"(%19, %446) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %456 = llvm.fadd %455, %444 : f32
    llvm.store %456, %443 : f32, !llvm.ptr
    %457 = llvm.add %434, %18 : i64
    llvm.br ^bb123(%457 : i64)
  ^bb128:  // pred: ^bb123
    %458 = llvm.add %429, %18 : i64
    llvm.br ^bb121(%458 : i64)
  ^bb129:  // pred: ^bb121
    %459 = llvm.add %426, %18 : i64
    llvm.br ^bb119(%459 : i64)
  ^bb130:  // pred: ^bb119
    %460 = llvm.add %424, %18 : i64
    llvm.br ^bb117(%460 : i64)
  ^bb131:  // pred: ^bb117
    %461 = llvm.add %422, %18 : i64
    llvm.br ^bb115(%461 : i64)
  ^bb132:  // pred: ^bb115
    %462 = llvm.mul %108, %13 : i64
    %463 = llvm.mul %462, %14 : i64
    %464 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %465 = llvm.call @cudaMallocAsync(%464, %463, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %466 = llvm.load %464 : !llvm.ptr -> !llvm.ptr
    %467 = llvm.insertvalue %466, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %468 = llvm.insertvalue %466, %467[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %469 = llvm.insertvalue %9, %468[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %470 = llvm.insertvalue %13, %117[1] : !llvm.array<3 x i64> 
    %471 = llvm.insertvalue %14, %470[2] : !llvm.array<3 x i64> 
    %472 = llvm.insertvalue %471, %469[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %473 = llvm.mul %222, %13 : i64
    %474 = llvm.insertvalue %473, %223[0] : !llvm.array<3 x i64> 
    %475 = llvm.insertvalue %474, %472[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %476 = builtin.unrealized_conversion_cast %475 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x384x32xf32>
    llvm.br ^bb133(%24 : i64)
  ^bb133(%477: i64):  // 2 preds: ^bb132, ^bb143
    %478 = llvm.icmp "slt" %477, %33 : i64
    llvm.cond_br %478, ^bb134, ^bb144
  ^bb134:  // pred: ^bb133
    llvm.br ^bb135(%24 : i64)
  ^bb135(%479: i64):  // 2 preds: ^bb134, ^bb142
    %480 = llvm.icmp "slt" %479, %20 : i64
    llvm.cond_br %480, ^bb136, ^bb143
  ^bb136:  // pred: ^bb135
    llvm.br ^bb137(%24 : i64)
  ^bb137(%481: i64):  // 2 preds: ^bb136, ^bb141
    %482 = builtin.unrealized_conversion_cast %481 : i64 to index
    %483 = llvm.icmp "slt" %481, %20 : i64
    llvm.cond_br %483, ^bb138, ^bb142
  ^bb138:  // pred: ^bb137
    llvm.br ^bb139(%24 : i64)
  ^bb139(%484: i64):  // 2 preds: ^bb138, ^bb140
    %485 = llvm.icmp "slt" %484, %20 : i64
    llvm.cond_br %485, ^bb140, ^bb141
  ^bb140:  // pred: ^bb139
    %486 = llvm.mul %477, %20 overflow<nsw> : i64
    %487 = llvm.add %484, %486 : i64
    %488 = builtin.unrealized_conversion_cast %487 : i64 to index
    %489 = llvm.mul %479, %29 overflow<nsw> : i64
    %490 = builtin.unrealized_conversion_cast %489 : i64 to index
    vector.transfer_write %27, %476[%482, %488, %490] : vector<8xf32>, memref<8x384x32xf32>
    %491 = llvm.add %484, %18 : i64
    llvm.br ^bb139(%491 : i64)
  ^bb141:  // pred: ^bb139
    %492 = llvm.add %481, %18 : i64
    llvm.br ^bb137(%492 : i64)
  ^bb142:  // pred: ^bb137
    %493 = llvm.add %479, %18 : i64
    llvm.br ^bb135(%493 : i64)
  ^bb143:  // pred: ^bb135
    %494 = llvm.add %477, %18 : i64
    llvm.br ^bb133(%494 : i64)
  ^bb144:  // pred: ^bb133
    llvm.br ^bb145(%24 : i64)
  ^bb145(%495: i64):  // 2 preds: ^bb144, ^bb161
    %496 = llvm.icmp "slt" %495, %33 : i64
    llvm.cond_br %496, ^bb146, ^bb162
  ^bb146:  // pred: ^bb145
    llvm.br ^bb147(%24 : i64)
  ^bb147(%497: i64):  // 2 preds: ^bb146, ^bb160
    %498 = llvm.icmp "slt" %497, %20 : i64
    llvm.cond_br %498, ^bb148, ^bb161
  ^bb148:  // pred: ^bb147
    llvm.br ^bb149(%24 : i64)
  ^bb149(%499: i64):  // 2 preds: ^bb148, ^bb159
    %500 = builtin.unrealized_conversion_cast %499 : i64 to index
    %501 = llvm.icmp "slt" %499, %20 : i64
    llvm.cond_br %501, ^bb150, ^bb160
  ^bb150:  // pred: ^bb149
    llvm.br ^bb151(%24 : i64)
  ^bb151(%502: i64):  // 2 preds: ^bb150, ^bb158
    %503 = llvm.icmp "slt" %502, %20 : i64
    llvm.cond_br %503, ^bb152, ^bb159
  ^bb152:  // pred: ^bb151
    %504 = llvm.mul %495, %20 overflow<nsw> : i64
    %505 = llvm.add %502, %504 : i64
    %506 = builtin.unrealized_conversion_cast %505 : i64 to index
    llvm.br ^bb153(%24 : i64)
  ^bb153(%507: i64):  // 2 preds: ^bb152, ^bb157
    %508 = llvm.icmp "slt" %507, %29 : i64
    llvm.cond_br %508, ^bb154, ^bb158
  ^bb154:  // pred: ^bb153
    %509 = llvm.mul %497, %29 overflow<nsw> : i64
    %510 = llvm.add %507, %509 : i64
    %511 = builtin.unrealized_conversion_cast %510 : i64 to index
    %512 = llvm.mul %499, %17 overflow<nsw, nuw> : i64
    %513 = llvm.mul %505, %23 overflow<nsw, nuw> : i64
    %514 = llvm.add %512, %513 overflow<nsw, nuw> : i64
    %515 = llvm.add %514, %510 overflow<nsw, nuw> : i64
    %516 = llvm.getelementptr inbounds|nuw %466[%515] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %517 = llvm.load %516 : !llvm.ptr -> f32
    llvm.br ^bb155(%24, %27 : i64, vector<8xf32>)
  ^bb155(%518: i64, %519: vector<8xf32>):  // 2 preds: ^bb154, ^bb156
    %520 = llvm.icmp "slt" %518, %29 : i64
    llvm.cond_br %520, ^bb156, ^bb157
  ^bb156:  // pred: ^bb155
    %521 = llvm.mul %518, %20 overflow<nsw> : i64
    %522 = builtin.unrealized_conversion_cast %521 : i64 to index
    %523 = vector.transfer_read %192[%500, %522, %506], %26 {permutation_map = #map} : memref<8x32x384xf32>, vector<8xf32>
    %524 = vector.transfer_read %326[%500, %522, %511], %26 {permutation_map = #map} : memref<8x32x32xf32>, vector<8xf32>
    %525 = llvm.fmul %523, %524 : vector<8xf32>
    %526 = llvm.fadd %519, %525 : vector<8xf32>
    %527 = llvm.add %518, %18 : i64
    llvm.br ^bb155(%527, %526 : i64, vector<8xf32>)
  ^bb157:  // pred: ^bb155
    %528 = "llvm.intr.vector.reduce.fadd"(%19, %519) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %529 = llvm.fadd %528, %517 : f32
    llvm.store %529, %516 : f32, !llvm.ptr
    %530 = llvm.add %507, %18 : i64
    llvm.br ^bb153(%530 : i64)
  ^bb158:  // pred: ^bb153
    %531 = llvm.add %502, %18 : i64
    llvm.br ^bb151(%531 : i64)
  ^bb159:  // pred: ^bb151
    %532 = llvm.add %499, %18 : i64
    llvm.br ^bb149(%532 : i64)
  ^bb160:  // pred: ^bb149
    %533 = llvm.add %497, %18 : i64
    llvm.br ^bb147(%533 : i64)
  ^bb161:  // pred: ^bb147
    %534 = llvm.add %495, %18 : i64
    llvm.br ^bb145(%534 : i64)
  ^bb162:  // pred: ^bb145
    %535 = llvm.mul %16, %13 : i64
    %536 = llvm.mul %535, %14 : i64
    %537 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %538 = llvm.call @cudaMallocAsync(%537, %536, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %539 = llvm.load %537 : !llvm.ptr -> !llvm.ptr
    %540 = llvm.insertvalue %539, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %541 = llvm.insertvalue %539, %540[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %542 = llvm.insertvalue %9, %541[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %543 = llvm.insertvalue %13, %1[0] : !llvm.array<2 x i64> 
    %544 = llvm.insertvalue %14, %543[1] : !llvm.array<2 x i64> 
    %545 = llvm.insertvalue %544, %542[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %546 = llvm.insertvalue %7, %1[1] : !llvm.array<2 x i64> 
    %547 = llvm.insertvalue %222, %546[0] : !llvm.array<2 x i64> 
    %548 = llvm.insertvalue %547, %545[4] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %549 = builtin.unrealized_conversion_cast %548 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<384x32xf32>
    llvm.br ^bb163(%24 : i64)
  ^bb163(%550: i64):  // 2 preds: ^bb162, ^bb170
    %551 = llvm.icmp "slt" %550, %34 : i64
    llvm.cond_br %551, ^bb164, ^bb171
  ^bb164:  // pred: ^bb163
    llvm.br ^bb165(%24 : i64)
  ^bb165(%552: i64):  // 2 preds: ^bb164, ^bb169
    %553 = llvm.icmp "slt" %552, %29 : i64
    llvm.cond_br %553, ^bb166, ^bb170
  ^bb166:  // pred: ^bb165
    llvm.br ^bb167(%24 : i64)
  ^bb167(%554: i64):  // 2 preds: ^bb166, ^bb168
    %555 = llvm.icmp "slt" %554, %31 : i64
    llvm.cond_br %555, ^bb168, ^bb169
  ^bb168:  // pred: ^bb167
    %556 = llvm.mul %550, %31 overflow<nsw> : i64
    %557 = llvm.add %554, %556 : i64
    %558 = builtin.unrealized_conversion_cast %557 : i64 to index
    %559 = llvm.mul %552, %20 overflow<nsw> : i64
    %560 = builtin.unrealized_conversion_cast %559 : i64 to index
    vector.transfer_write %27, %549[%558, %560] : vector<8xf32>, memref<384x32xf32>
    %561 = llvm.add %554, %18 : i64
    llvm.br ^bb167(%561 : i64)
  ^bb169:  // pred: ^bb167
    %562 = llvm.add %552, %18 : i64
    llvm.br ^bb165(%562 : i64)
  ^bb170:  // pred: ^bb165
    %563 = llvm.add %550, %18 : i64
    llvm.br ^bb163(%563 : i64)
  ^bb171:  // pred: ^bb163
    llvm.br ^bb172(%24 : i64)
  ^bb172(%564: i64):  // 2 preds: ^bb171, ^bb182
    %565 = llvm.icmp "slt" %564, %34 : i64
    llvm.cond_br %565, ^bb173, ^bb183
  ^bb173:  // pred: ^bb172
    llvm.br ^bb174(%24 : i64)
  ^bb174(%566: i64):  // 2 preds: ^bb173, ^bb181
    %567 = llvm.icmp "slt" %566, %29 : i64
    llvm.cond_br %567, ^bb175, ^bb182
  ^bb175:  // pred: ^bb174
    llvm.br ^bb176(%24 : i64)
  ^bb176(%568: i64):  // 2 preds: ^bb175, ^bb180
    %569 = llvm.icmp "slt" %568, %31 : i64
    llvm.cond_br %569, ^bb177, ^bb181
  ^bb177:  // pred: ^bb176
    %570 = llvm.mul %564, %31 overflow<nsw> : i64
    %571 = llvm.add %568, %570 : i64
    %572 = builtin.unrealized_conversion_cast %571 : i64 to index
    llvm.br ^bb178(%24 : i64)
  ^bb178(%573: i64):  // 2 preds: ^bb177, ^bb179
    %574 = llvm.icmp "slt" %573, %20 : i64
    llvm.cond_br %574, ^bb179, ^bb180
  ^bb179:  // pred: ^bb178
    %575 = llvm.mul %566, %20 overflow<nsw> : i64
    %576 = llvm.add %573, %575 : i64
    %577 = builtin.unrealized_conversion_cast %576 : i64 to index
    %578 = llvm.mul %571, %23 overflow<nsw, nuw> : i64
    %579 = llvm.add %578, %576 overflow<nsw, nuw> : i64
    %580 = llvm.getelementptr inbounds|nuw %539[%579] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %581 = llvm.load %580 : !llvm.ptr -> f32
    %582 = vector.transfer_read %476[%107, %572, %577], %26 {in_bounds = [true], permutation_map = #map1} : memref<8x384x32xf32>, vector<8xf32>
    %583 = llvm.fadd %582, %27 : vector<8xf32>
    %584 = "llvm.intr.vector.reduce.fadd"(%19, %583) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %585 = llvm.fadd %584, %581 : f32
    llvm.store %585, %580 : f32, !llvm.ptr
    %586 = llvm.add %573, %18 : i64
    llvm.br ^bb178(%586 : i64)
  ^bb180:  // pred: ^bb178
    %587 = llvm.add %568, %18 : i64
    llvm.br ^bb176(%587 : i64)
  ^bb181:  // pred: ^bb176
    %588 = llvm.add %566, %18 : i64
    llvm.br ^bb174(%588 : i64)
  ^bb182:  // pred: ^bb174
    %589 = llvm.add %564, %18 : i64
    llvm.br ^bb172(%589 : i64)
  ^bb183:  // pred: ^bb172
    %590 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %591 = llvm.call @cudaMallocAsync(%590, %110, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %592 = llvm.load %590 : !llvm.ptr -> !llvm.ptr
    %593 = llvm.insertvalue %592, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %594 = llvm.insertvalue %592, %593[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %595 = llvm.insertvalue %9, %594[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %596 = llvm.insertvalue %119, %595[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %597 = llvm.insertvalue %125, %596[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %598 = builtin.unrealized_conversion_cast %597 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x384xf32>
    llvm.br ^bb184(%24 : i64)
  ^bb184(%599: i64):  // 2 preds: ^bb183, ^bb194
    %600 = llvm.icmp "slt" %599, %29 : i64
    llvm.cond_br %600, ^bb185, ^bb195
  ^bb185:  // pred: ^bb184
    llvm.br ^bb186(%24 : i64)
  ^bb186(%601: i64):  // 2 preds: ^bb185, ^bb193
    %602 = llvm.icmp "slt" %601, %30 : i64
    llvm.cond_br %602, ^bb187, ^bb194
  ^bb187:  // pred: ^bb186
    llvm.br ^bb188(%24 : i64)
  ^bb188(%603: i64):  // 2 preds: ^bb187, ^bb192
    %604 = builtin.unrealized_conversion_cast %603 : i64 to index
    %605 = llvm.icmp "slt" %603, %20 : i64
    llvm.cond_br %605, ^bb189, ^bb193
  ^bb189:  // pred: ^bb188
    llvm.br ^bb190(%24 : i64)
  ^bb190(%606: i64):  // 2 preds: ^bb189, ^bb191
    %607 = llvm.icmp "slt" %606, %20 : i64
    llvm.cond_br %607, ^bb191, ^bb192
  ^bb191:  // pred: ^bb190
    %608 = llvm.mul %599, %20 overflow<nsw> : i64
    %609 = llvm.add %606, %608 : i64
    %610 = builtin.unrealized_conversion_cast %609 : i64 to index
    %611 = llvm.mul %601, %29 overflow<nsw> : i64
    %612 = builtin.unrealized_conversion_cast %611 : i64 to index
    %613 = vector.transfer_read %403[%604, %610, %612], %26 : memref<8x32x384xf32>, vector<8xf32>
    %614 = vector.transfer_read %127[%604, %610, %612], %26 : memref<8x32x384xf32>, vector<8xf32>
    %615 = vector.transfer_read %76[%604, %610, %612], %26 : memref<8x32x384xf32, 1>, vector<8xf32>
    %616 = llvm.fcmp "ugt" %614, %615 : vector<8xf32>
    %617 = llvm.zext %616 : vector<8xi1> to vector<8xi32>
    %618 = llvm.sitofp %617 : vector<8xi32> to vector<8xf32>
    %619 = llvm.fmul %613, %618 : vector<8xf32>
    vector.transfer_write %619, %598[%604, %610, %612] : vector<8xf32>, memref<8x32x384xf32>
    %620 = llvm.add %606, %18 : i64
    llvm.br ^bb190(%620 : i64)
  ^bb192:  // pred: ^bb190
    %621 = llvm.add %603, %18 : i64
    llvm.br ^bb188(%621 : i64)
  ^bb193:  // pred: ^bb188
    %622 = llvm.add %601, %18 : i64
    llvm.br ^bb186(%622 : i64)
  ^bb194:  // pred: ^bb186
    %623 = llvm.add %599, %18 : i64
    llvm.br ^bb184(%623 : i64)
  ^bb195:  // pred: ^bb184
    %624 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %625 = llvm.call @cudaMallocAsync(%624, %535, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %626 = llvm.load %624 : !llvm.ptr -> !llvm.ptr
    %627 = llvm.insertvalue %626, %4[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %628 = llvm.insertvalue %626, %627[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %629 = llvm.insertvalue %9, %628[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %630 = llvm.insertvalue %13, %3[0] : !llvm.array<1 x i64> 
    %631 = llvm.insertvalue %630, %629[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %632 = llvm.insertvalue %367, %631[4] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %633 = builtin.unrealized_conversion_cast %632 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<384xf32>
    llvm.br ^bb196(%24 : i64)
  ^bb196(%634: i64):  // 2 preds: ^bb195, ^bb200
    %635 = llvm.icmp "slt" %634, %34 : i64
    llvm.cond_br %635, ^bb197, ^bb201
  ^bb197:  // pred: ^bb196
    llvm.br ^bb198(%24 : i64)
  ^bb198(%636: i64):  // 2 preds: ^bb197, ^bb199
    %637 = llvm.icmp "slt" %636, %35 : i64
    llvm.cond_br %637, ^bb199, ^bb200
  ^bb199:  // pred: ^bb198
    %638 = llvm.mul %636, %20 overflow<nsw> : i64
    %639 = llvm.mul %634, %31 overflow<nsw> : i64
    %640 = llvm.add %638, %639 : i64
    %641 = builtin.unrealized_conversion_cast %640 : i64 to index
    vector.transfer_write %27, %633[%641] : vector<8xf32>, memref<384xf32>
    %642 = llvm.add %636, %18 : i64
    llvm.br ^bb198(%642 : i64)
  ^bb200:  // pred: ^bb198
    %643 = llvm.add %634, %18 : i64
    llvm.br ^bb196(%643 : i64)
  ^bb201:  // pred: ^bb196
    llvm.br ^bb202(%24 : i64)
  ^bb202(%644: i64):  // 2 preds: ^bb201, ^bb212
    %645 = llvm.icmp "slt" %644, %34 : i64
    llvm.cond_br %645, ^bb203, ^bb213
  ^bb203:  // pred: ^bb202
    llvm.br ^bb204(%24 : i64)
  ^bb204(%646: i64):  // 2 preds: ^bb203, ^bb211
    %647 = llvm.icmp "slt" %646, %31 : i64
    llvm.cond_br %647, ^bb205, ^bb212
  ^bb205:  // pred: ^bb204
    %648 = llvm.mul %644, %31 overflow<nsw> : i64
    %649 = llvm.add %646, %648 : i64
    %650 = builtin.unrealized_conversion_cast %649 : i64 to index
    llvm.br ^bb206(%24 : i64)
  ^bb206(%651: i64):  // 2 preds: ^bb205, ^bb210
    %652 = builtin.unrealized_conversion_cast %651 : i64 to index
    %653 = llvm.icmp "slt" %651, %20 : i64
    llvm.cond_br %653, ^bb207, ^bb211
  ^bb207:  // pred: ^bb206
    %654 = llvm.getelementptr inbounds|nuw %626[%649] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %655 = llvm.load %654 : !llvm.ptr -> f32
    llvm.br ^bb208(%24, %27 : i64, vector<8xf32>)
  ^bb208(%656: i64, %657: vector<8xf32>):  // 2 preds: ^bb207, ^bb209
    %658 = llvm.icmp "slt" %656, %29 : i64
    llvm.cond_br %658, ^bb209, ^bb210
  ^bb209:  // pred: ^bb208
    %659 = llvm.mul %656, %20 overflow<nsw> : i64
    %660 = builtin.unrealized_conversion_cast %659 : i64 to index
    %661 = vector.transfer_read %598[%652, %660, %650], %26 {permutation_map = #map} : memref<8x32x384xf32>, vector<8xf32>
    %662 = llvm.fadd %661, %657 : vector<8xf32>
    %663 = llvm.add %656, %18 : i64
    llvm.br ^bb208(%663, %662 : i64, vector<8xf32>)
  ^bb210:  // pred: ^bb208
    %664 = "llvm.intr.vector.reduce.fadd"(%19, %657) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %665 = llvm.fadd %664, %655 : f32
    llvm.store %665, %654 : f32, !llvm.ptr
    %666 = llvm.add %651, %18 : i64
    llvm.br ^bb206(%666 : i64)
  ^bb211:  // pred: ^bb206
    %667 = llvm.add %646, %18 : i64
    llvm.br ^bb204(%667 : i64)
  ^bb212:  // pred: ^bb204
    %668 = llvm.add %644, %18 : i64
    llvm.br ^bb202(%668 : i64)
  ^bb213:  // pred: ^bb202
    %669 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %670 = llvm.call @cudaMallocAsync(%669, %110, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %671 = llvm.load %669 : !llvm.ptr -> !llvm.ptr
    %672 = llvm.insertvalue %671, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %673 = llvm.insertvalue %671, %672[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %674 = llvm.insertvalue %9, %673[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %675 = llvm.insertvalue %119, %674[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %676 = llvm.insertvalue %125, %675[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %677 = builtin.unrealized_conversion_cast %676 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<8x32x384xf32>
    llvm.br ^bb214(%24 : i64)
  ^bb214(%678: i64):  // 2 preds: ^bb213, ^bb224
    %679 = llvm.icmp "slt" %678, %29 : i64
    llvm.cond_br %679, ^bb215, ^bb225
  ^bb215:  // pred: ^bb214
    llvm.br ^bb216(%24 : i64)
  ^bb216(%680: i64):  // 2 preds: ^bb215, ^bb223
    %681 = llvm.icmp "slt" %680, %30 : i64
    llvm.cond_br %681, ^bb217, ^bb224
  ^bb217:  // pred: ^bb216
    llvm.br ^bb218(%24 : i64)
  ^bb218(%682: i64):  // 2 preds: ^bb217, ^bb222
    %683 = builtin.unrealized_conversion_cast %682 : i64 to index
    %684 = llvm.icmp "slt" %682, %20 : i64
    llvm.cond_br %684, ^bb219, ^bb223
  ^bb219:  // pred: ^bb218
    llvm.br ^bb220(%24 : i64)
  ^bb220(%685: i64):  // 2 preds: ^bb219, ^bb221
    %686 = llvm.icmp "slt" %685, %20 : i64
    llvm.cond_br %686, ^bb221, ^bb222
  ^bb221:  // pred: ^bb220
    %687 = llvm.mul %678, %20 overflow<nsw> : i64
    %688 = llvm.add %685, %687 : i64
    %689 = builtin.unrealized_conversion_cast %688 : i64 to index
    %690 = llvm.mul %680, %29 overflow<nsw> : i64
    %691 = builtin.unrealized_conversion_cast %690 : i64 to index
    vector.transfer_write %27, %677[%683, %689, %691] : vector<8xf32>, memref<8x32x384xf32>
    %692 = llvm.add %685, %18 : i64
    llvm.br ^bb220(%692 : i64)
  ^bb222:  // pred: ^bb220
    %693 = llvm.add %682, %18 : i64
    llvm.br ^bb218(%693 : i64)
  ^bb223:  // pred: ^bb218
    %694 = llvm.add %680, %18 : i64
    llvm.br ^bb216(%694 : i64)
  ^bb224:  // pred: ^bb216
    %695 = llvm.add %678, %18 : i64
    llvm.br ^bb214(%695 : i64)
  ^bb225:  // pred: ^bb214
    llvm.br ^bb226(%24 : i64)
  ^bb226(%696: i64):  // 2 preds: ^bb225, ^bb242
    %697 = llvm.icmp "slt" %696, %29 : i64
    llvm.cond_br %697, ^bb227, ^bb243
  ^bb227:  // pred: ^bb226
    llvm.br ^bb228(%24 : i64)
  ^bb228(%698: i64):  // 2 preds: ^bb227, ^bb241
    %699 = llvm.icmp "slt" %698, %30 : i64
    llvm.cond_br %699, ^bb229, ^bb242
  ^bb229:  // pred: ^bb228
    llvm.br ^bb230(%24 : i64)
  ^bb230(%700: i64):  // 2 preds: ^bb229, ^bb240
    %701 = builtin.unrealized_conversion_cast %700 : i64 to index
    %702 = llvm.icmp "slt" %700, %20 : i64
    llvm.cond_br %702, ^bb231, ^bb241
  ^bb231:  // pred: ^bb230
    llvm.br ^bb232(%24 : i64)
  ^bb232(%703: i64):  // 2 preds: ^bb231, ^bb239
    %704 = llvm.icmp "slt" %703, %20 : i64
    llvm.cond_br %704, ^bb233, ^bb240
  ^bb233:  // pred: ^bb232
    %705 = llvm.mul %696, %20 overflow<nsw> : i64
    %706 = llvm.add %703, %705 : i64
    %707 = builtin.unrealized_conversion_cast %706 : i64 to index
    llvm.br ^bb234(%24 : i64)
  ^bb234(%708: i64):  // 2 preds: ^bb233, ^bb238
    %709 = llvm.icmp "slt" %708, %29 : i64
    llvm.cond_br %709, ^bb235, ^bb239
  ^bb235:  // pred: ^bb234
    %710 = llvm.mul %698, %29 overflow<nsw> : i64
    %711 = llvm.add %708, %710 : i64
    %712 = builtin.unrealized_conversion_cast %711 : i64 to index
    %713 = llvm.mul %700, %17 overflow<nsw, nuw> : i64
    %714 = llvm.mul %706, %32 overflow<nsw, nuw> : i64
    %715 = llvm.add %713, %714 overflow<nsw, nuw> : i64
    %716 = llvm.add %715, %711 overflow<nsw, nuw> : i64
    %717 = llvm.getelementptr inbounds|nuw %671[%716] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %718 = llvm.load %717 : !llvm.ptr -> f32
    llvm.br ^bb236(%24, %27 : i64, vector<8xf32>)
  ^bb236(%719: i64, %720: vector<8xf32>):  // 2 preds: ^bb235, ^bb237
    %721 = llvm.icmp "slt" %719, %29 : i64
    llvm.cond_br %721, ^bb237, ^bb238
  ^bb237:  // pred: ^bb236
    %722 = llvm.mul %719, %20 overflow<nsw> : i64
    %723 = builtin.unrealized_conversion_cast %722 : i64 to index
    %724 = vector.transfer_read %106[%701, %723, %707], %26 {permutation_map = #map} : memref<8x32x32xf32, 1>, vector<8xf32>
    %725 = vector.transfer_read %598[%701, %723, %712], %26 {permutation_map = #map} : memref<8x32x384xf32>, vector<8xf32>
    %726 = llvm.fmul %724, %725 : vector<8xf32>
    %727 = llvm.fadd %720, %726 : vector<8xf32>
    %728 = llvm.add %719, %18 : i64
    llvm.br ^bb236(%728, %727 : i64, vector<8xf32>)
  ^bb238:  // pred: ^bb236
    %729 = "llvm.intr.vector.reduce.fadd"(%19, %720) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %730 = llvm.fadd %729, %718 : f32
    llvm.store %730, %717 : f32, !llvm.ptr
    %731 = llvm.add %708, %18 : i64
    llvm.br ^bb234(%731 : i64)
  ^bb239:  // pred: ^bb234
    %732 = llvm.add %703, %18 : i64
    llvm.br ^bb232(%732 : i64)
  ^bb240:  // pred: ^bb232
    %733 = llvm.add %700, %18 : i64
    llvm.br ^bb230(%733 : i64)
  ^bb241:  // pred: ^bb230
    %734 = llvm.add %698, %18 : i64
    llvm.br ^bb228(%734 : i64)
  ^bb242:  // pred: ^bb228
    %735 = llvm.add %696, %18 : i64
    llvm.br ^bb226(%735 : i64)
  ^bb243:  // pred: ^bb226
    %736 = llvm.mul %358, %13 : i64
    %737 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %738 = llvm.call @cudaMallocAsync(%737, %736, %11) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %739 = llvm.load %737 : !llvm.ptr -> !llvm.ptr
    %740 = llvm.insertvalue %739, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %741 = llvm.insertvalue %739, %740[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %742 = llvm.insertvalue %9, %741[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %743 = llvm.insertvalue %14, %1[0] : !llvm.array<2 x i64> 
    %744 = llvm.insertvalue %13, %743[1] : !llvm.array<2 x i64> 
    %745 = llvm.insertvalue %744, %742[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %746 = llvm.insertvalue %122, %546[0] : !llvm.array<2 x i64> 
    %747 = llvm.insertvalue %746, %745[4] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %748 = builtin.unrealized_conversion_cast %747 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x384xf32>
    llvm.br ^bb244(%24 : i64)
  ^bb244(%749: i64):  // 2 preds: ^bb243, ^bb248
    %750 = llvm.icmp "slt" %749, %33 : i64
    llvm.cond_br %750, ^bb245, ^bb249
  ^bb245:  // pred: ^bb244
    llvm.br ^bb246(%24 : i64)
  ^bb246(%751: i64):  // 2 preds: ^bb245, ^bb247
    %752 = builtin.unrealized_conversion_cast %751 : i64 to index
    %753 = llvm.icmp "slt" %751, %23 : i64
    llvm.cond_br %753, ^bb247, ^bb248
  ^bb247:  // pred: ^bb246
    %754 = llvm.mul %749, %20 overflow<nsw> : i64
    %755 = builtin.unrealized_conversion_cast %754 : i64 to index
    vector.transfer_write %27, %748[%752, %755] : vector<8xf32>, memref<32x384xf32>
    %756 = llvm.add %751, %18 : i64
    llvm.br ^bb246(%756 : i64)
  ^bb248:  // pred: ^bb246
    %757 = llvm.add %749, %18 : i64
    llvm.br ^bb244(%757 : i64)
  ^bb249:  // pred: ^bb244
    llvm.br ^bb250(%24 : i64)
  ^bb250(%758: i64):  // 2 preds: ^bb249, ^bb257
    %759 = llvm.icmp "slt" %758, %33 : i64
    llvm.cond_br %759, ^bb251, ^bb258
  ^bb251:  // pred: ^bb250
    llvm.br ^bb252(%24 : i64)
  ^bb252(%760: i64):  // 2 preds: ^bb251, ^bb256
    %761 = builtin.unrealized_conversion_cast %760 : i64 to index
    %762 = llvm.icmp "slt" %760, %23 : i64
    llvm.cond_br %762, ^bb253, ^bb257
  ^bb253:  // pred: ^bb252
    llvm.br ^bb254(%24 : i64)
  ^bb254(%763: i64):  // 2 preds: ^bb253, ^bb255
    %764 = llvm.icmp "slt" %763, %20 : i64
    llvm.cond_br %764, ^bb255, ^bb256
  ^bb255:  // pred: ^bb254
    %765 = llvm.mul %758, %20 overflow<nsw> : i64
    %766 = llvm.add %763, %765 : i64
    %767 = builtin.unrealized_conversion_cast %766 : i64 to index
    %768 = llvm.mul %760, %32 overflow<nsw, nuw> : i64
    %769 = llvm.add %768, %766 overflow<nsw, nuw> : i64
    %770 = llvm.getelementptr inbounds|nuw %739[%769] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %771 = llvm.load %770 : !llvm.ptr -> f32
    %772 = vector.transfer_read %677[%107, %761, %767], %26 {in_bounds = [true], permutation_map = #map1} : memref<8x32x384xf32>, vector<8xf32>
    %773 = llvm.fadd %772, %27 : vector<8xf32>
    %774 = "llvm.intr.vector.reduce.fadd"(%19, %773) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    %775 = llvm.fadd %774, %771 : f32
    llvm.store %775, %770 : f32, !llvm.ptr
    %776 = llvm.add %763, %18 : i64
    llvm.br ^bb254(%776 : i64)
  ^bb256:  // pred: ^bb254
    %777 = llvm.add %760, %18 : i64
    llvm.br ^bb252(%777 : i64)
  ^bb257:  // pred: ^bb252
    %778 = llvm.add %758, %18 : i64
    llvm.br ^bb250(%778 : i64)
  ^bb258:  // pred: ^bb250
    %779 = llvm.addrspacecast %arg68 : !llvm.ptr<1> to !llvm.ptr
    %780 = llvm.call @cudaMemcpyAsync(%779, %316, %312, %0, %11) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %781 = llvm.addrspacecast %arg73 : !llvm.ptr<1> to !llvm.ptr
    %782 = llvm.call @cudaMemcpyAsync(%781, %739, %736, %12, %11) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %783 = llvm.addrspacecast %arg80 : !llvm.ptr<1> to !llvm.ptr
    %784 = llvm.call @cudaMemcpyAsync(%783, %626, %535, %12, %11) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %785 = llvm.addrspacecast %arg85 : !llvm.ptr<1> to !llvm.ptr
    %786 = llvm.call @cudaMemcpyAsync(%785, %539, %536, %12, %11) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %787 = llvm.addrspacecast %arg92 : !llvm.ptr<1> to !llvm.ptr
    %788 = llvm.call @cudaMemcpyAsync(%787, %361, %358, %12, %11) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %789 = llvm.addrspacecast %arg1 : !llvm.ptr<1> to !llvm.ptr
    %790 = llvm.call @cudaFreeAsync(%789, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %791 = llvm.addrspacecast %arg10 : !llvm.ptr<1> to !llvm.ptr
    %792 = llvm.call @cudaFreeAsync(%791, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %793 = llvm.addrspacecast %arg19 : !llvm.ptr<1> to !llvm.ptr
    %794 = llvm.call @cudaFreeAsync(%793, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %795 = llvm.addrspacecast %arg26 : !llvm.ptr<1> to !llvm.ptr
    %796 = llvm.call @cudaFreeAsync(%795, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %797 = llvm.addrspacecast %arg35 : !llvm.ptr<1> to !llvm.ptr
    %798 = llvm.call @cudaFreeAsync(%797, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %799 = llvm.addrspacecast %arg44 : !llvm.ptr<1> to !llvm.ptr
    %800 = llvm.call @cudaFreeAsync(%799, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %801 = llvm.addrspacecast %arg51 : !llvm.ptr<1> to !llvm.ptr
    %802 = llvm.call @cudaFreeAsync(%801, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %803 = llvm.addrspacecast %arg56 : !llvm.ptr<1> to !llvm.ptr
    %804 = llvm.call @cudaFreeAsync(%803, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %805 = llvm.addrspacecast %arg63 : !llvm.ptr<1> to !llvm.ptr
    %806 = llvm.call @cudaFreeAsync(%805, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %807 = llvm.call @cudaFreeAsync(%113, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %808 = llvm.call @cudaFreeAsync(%186, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %809 = llvm.call @cudaFreeAsync(%216, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %810 = llvm.call @cudaFreeAsync(%283, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %811 = llvm.call @cudaFreeAsync(%316, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %812 = llvm.call @cudaFreeAsync(%320, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %813 = llvm.call @cudaFreeAsync(%361, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %814 = llvm.call @cudaFreeAsync(%397, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %815 = llvm.call @cudaFreeAsync(%466, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %816 = llvm.call @cudaFreeAsync(%539, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %817 = llvm.call @cudaFreeAsync(%592, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %818 = llvm.call @cudaFreeAsync(%626, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %819 = llvm.call @cudaFreeAsync(%671, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    %820 = llvm.call @cudaFreeAsync(%739, %11) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @_mlir_ciface_main_impl(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.extractvalue %20[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.extractvalue %20[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    %29 = llvm.extractvalue %28[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.extractvalue %28[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.extractvalue %28[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.extractvalue %28[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.extractvalue %28[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %34 = llvm.extractvalue %28[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.extractvalue %28[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %36 = llvm.extractvalue %28[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.extractvalue %28[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.load %arg4 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    %39 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.extractvalue %38[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.extractvalue %38[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.extractvalue %38[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.extractvalue %38[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.extractvalue %38[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.extractvalue %38[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = llvm.extractvalue %38[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> 
    %48 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.extractvalue %48[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.extractvalue %48[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.extractvalue %48[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %48[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.extractvalue %48[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.extractvalue %48[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.extractvalue %48[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.load %arg6 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %57 = llvm.extractvalue %56[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.extractvalue %56[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.extractvalue %56[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.extractvalue %56[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %61 = llvm.extractvalue %56[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.load %arg7 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.extractvalue %62[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %62[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.extractvalue %62[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.extractvalue %62[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.extractvalue %62[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.extractvalue %62[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.extractvalue %62[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.load %arg8 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %71 = llvm.extractvalue %70[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.extractvalue %70[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %73 = llvm.extractvalue %70[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.extractvalue %70[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.extractvalue %70[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.load %arg9 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.extractvalue %76[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.extractvalue %76[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.extractvalue %76[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %80 = llvm.extractvalue %76[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.extractvalue %76[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.load %arg10 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %83 = llvm.extractvalue %82[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.extractvalue %82[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.extractvalue %82[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.extractvalue %82[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.extractvalue %82[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.extractvalue %82[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.extractvalue %82[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.load %arg11 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %91 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %94 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.load %arg12 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %97 = llvm.extractvalue %96[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.extractvalue %96[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.extractvalue %96[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %96[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %96[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.extractvalue %96[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.extractvalue %96[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.load %arg13 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %105 = llvm.extractvalue %104[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.extractvalue %104[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.extractvalue %104[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.extractvalue %104[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %109 = llvm.extractvalue %104[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @main(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25, %26, %27, %29, %30, %31, %32, %33, %34, %35, %36, %37, %39, %40, %41, %42, %43, %44, %45, %46, %47, %49, %50, %51, %52, %53, %54, %55, %57, %58, %59, %60, %61, %63, %64, %65, %66, %67, %68, %69, %71, %72, %73, %74, %75, %77, %78, %79, %80, %81, %83, %84, %85, %86, %87, %88, %89, %91, %92, %93, %94, %95, %97, %98, %99, %100, %101, %102, %103, %105, %106, %107, %108, %109) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %1 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr %arg1[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %4 = llvm.load %3 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr %arg1[3] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %6 = llvm.load %5 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr %arg1[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr %arg1[5] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr %arg1[6] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %12 = llvm.load %11 : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr %arg1[7] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr %arg1[8] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %17 = llvm.getelementptr %arg1[9] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %18 = llvm.load %17 : !llvm.ptr -> !llvm.ptr
    %19 = llvm.getelementptr %arg1[10] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.ptr
    %21 = llvm.getelementptr %arg1[11] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %22 = llvm.load %21 : !llvm.ptr -> !llvm.ptr
    %23 = llvm.getelementptr %arg1[12] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %24 = llvm.load %23 : !llvm.ptr -> !llvm.ptr
    llvm.call @_mlir_ciface_main_impl(%arg0, %0, %2, %4, %6, %8, %10, %12, %14, %16, %18, %20, %22, %24) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 3 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main_kernel\0A// __wg_main_kernel_0 has been demoted\0A\0A.visible .entry main_kernel(\0A\09.param .u64 main_kernel_param_0,\0A\09.param .u64 .ptr .align 1 main_kernel_param_1,\0A\09.param .u64 .ptr .align 1 main_kernel_param_2,\0A\09.param .u64 main_kernel_param_3,\0A\09.param .u64 main_kernel_param_4,\0A\09.param .u64 main_kernel_param_5,\0A\09.param .u64 main_kernel_param_6,\0A\09.param .u64 main_kernel_param_7,\0A\09.param .u64 main_kernel_param_8,\0A\09.param .u64 main_kernel_param_9,\0A\09.param .u64 main_kernel_param_10,\0A\09.param .f32 main_kernel_param_11,\0A\09.param .u64 main_kernel_param_12,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_13,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_14,\0A\09.param .u64 main_kernel_param_15,\0A\09.param .u64 main_kernel_param_16,\0A\09.param .u64 main_kernel_param_17\0A)\0A.maxntid 1024, 1, 1\0A{\0A\09.reg .pred \09%p<20>;\0A\09.reg .b32 \09%r<94>;\0A\09.reg .b64 \09%rd<46>;\0A\09// demoted variable\0A\09.shared .align 4 .b8 __wg_main_kernel_0[128];\0A\09ld.param.b32 \09%r91, [main_kernel_param_11];\0A\09ld.param.b64 \09%rd20, [main_kernel_param_10];\0A\09ld.param.b64 \09%rd23, [main_kernel_param_2];\0A\09cvta.to.global.u64 \09%rd1, %rd23;\0A\09mov.u32 \09%r16, %tid.x;\0A\09cvt.u64.u32 \09%rd2, %r16;\0A\09mov.u32 \09%r17, %ctaid.x;\0A\09mov.u32 \09%r1, %ntid.x;\0A\09mad.lo.s32 \09%r19, %r17, %r1, %r16;\0A\09cvt.u64.u32 \09%rd43, %r19;\0A\09setp.le.s64 \09%p1, %rd20, %rd43;\0A\09@%p1 bra \09$L__BB0_9;\0A\09ld.param.b64 \09%rd19, [main_kernel_param_0];\0A\09mov.u32 \09%r18, %nctaid.x;\0A\09mul.lo.s32 \09%r20, %r1, %r18;\0A\09cvt.u64.u32 \09%rd4, %r20;\0A\09cvt.u32.u64 \09%r21, %rd19;\0A\09bra.uni \09$L__BB0_2;\0A$L__BB0_7:\0A\09div.u64 \09%rd45, %rd44, %rd19;\0A$L__BB0_8:\0A\09mul.lo.s64 \09%rd29, %rd45, %rd19;\0A\09sub.s64 \09%rd30, %rd44, %rd29;\0A\09shl.b64 \09%rd31, %rd45, 12;\0A\09add.s64 \09%rd32, %rd1, %rd31;\0A\09shl.b64 \09%rd33, %rd30, 7;\0A\09add.s64 \09%rd34, %rd32, %rd33;\0A\09shl.b64 \09%rd35, %rd12, 2;\0A\09add.s64 \09%rd36, %rd34, %rd35;\0A\09ld.global.b32 \09%r27, [%rd36];\0A\09add.rn.f32 \09%r91, %r91, %r27;\0A\09add.s64 \09%rd43, %rd43, %rd4;\0A\09setp.lt.s64 \09%p4, %rd43, %rd20;\0A\09@%p4 bra \09$L__BB0_2;\0A\09bra.uni \09$L__BB0_9;\0A$L__BB0_2:\0A\09or.b64 \09%rd24, %rd43, %rd19;\0A\09and.b64 \09%rd25, %rd24, -4294967296;\0A\09setp.ne.b64 \09%p2, %rd25, 0;\0A\09@%p2 bra \09$L__BB0_4;\0A\09bra.uni \09$L__BB0_3;\0A$L__BB0_4:\0A\09div.u64 \09%rd44, %rd43, %rd19;\0A\09bra.uni \09$L__BB0_5;\0A$L__BB0_3:\0A\09cvt.u32.u64 \09%r22, %rd43;\0A\09div.u32 \09%r23, %r22, %r21;\0A\09cvt.u64.u32 \09%rd44, %r23;\0A$L__BB0_5:\0A\09mul.lo.s64 \09%rd26, %rd44, %rd19;\0A\09sub.s64 \09%rd12, %rd43, %rd26;\0A\09or.b64 \09%rd27, %rd44, %rd19;\0A\09and.b64 \09%rd28, %rd27, -4294967296;\0A\09setp.ne.b64 \09%p3, %rd28, 0;\0A\09@%p3 bra \09$L__BB0_7;\0A\09cvt.u32.u64 \09%r25, %rd44;\0A\09div.u32 \09%r26, %r25, %r21;\0A\09cvt.u64.u32 \09%rd45, %r26;\0A\09bra.uni \09$L__BB0_8;\0A$L__BB0_9:\0A\09cvt.u32.u64 \09%r28, %rd2;\0A\09and.b32 \09%r5, %r28, 31;\0A\09and.b32 \09%r29, %r28, 992;\0A\09sub.s32 \09%r6, %r1, %r29;\0A\09setp.gt.s32 \09%p5, %r6, 31;\0A\09@%p5 bra \09$L__BB0_11;\0A\09sub.s32 \09%r39, 32, %r6;\0A\09mov.b32 \09%r40, -1;\0A\09shr.u32 \09%r41, %r40, %r39;\0A\09add.s32 \09%r42, %r6, -1;\0A\09shfl.sync.bfly.b32 \09%r43|%p6, %r91, 1, %r42, %r41;\0A\09add.rn.f32 \09%r44, %r91, %r43;\0A\09selp.f32 \09%r45, %r44, %r91, %p6;\0A\09shfl.sync.bfly.b32 \09%r46|%p7, %r45, 2, %r42, %r41;\0A\09add.rn.f32 \09%r47, %r46, %r45;\0A\09selp.f32 \09%r48, %r47, %r45, %p7;\0A\09shfl.sync.bfly.b32 \09%r49|%p8, %r48, 4, %r42, %r41;\0A\09add.rn.f32 \09%r50, %r49, %r48;\0A\09selp.f32 \09%r51, %r50, %r48, %p8;\0A\09shfl.sync.bfly.b32 \09%r52|%p9, %r51, 8, %r42, %r41;\0A\09add.rn.f32 \09%r53, %r52, %r51;\0A\09selp.f32 \09%r54, %r53, %r51, %p9;\0A\09shfl.sync.bfly.b32 \09%r55|%p10, %r54, 16, %r42, %r41;\0A\09add.rn.f32 \09%r56, %r55, %r54;\0A\09selp.f32 \09%r92, %r56, %r54, %p10;\0A\09bra.uni \09$L__BB0_12;\0A$L__BB0_11:\0A\09shfl.sync.bfly.b32 \09%r30, %r91, 1, 31, -1;\0A\09add.rn.f32 \09%r31, %r91, %r30;\0A\09shfl.sync.bfly.b32 \09%r32, %r31, 2, 31, -1;\0A\09add.rn.f32 \09%r33, %r31, %r32;\0A\09shfl.sync.bfly.b32 \09%r34, %r33, 4, 31, -1;\0A\09add.rn.f32 \09%r35, %r33, %r34;\0A\09shfl.sync.bfly.b32 \09%r36, %r35, 8, 31, -1;\0A\09add.rn.f32 \09%r37, %r35, %r36;\0A\09shfl.sync.bfly.b32 \09%r38, %r37, 16, 31, -1;\0A\09add.rn.f32 \09%r92, %r37, %r38;\0A$L__BB0_12:\0A\09setp.ne.b32 \09%p11, %r5, 0;\0A\09@%p11 bra \09$L__BB0_14;\0A\09shr.u32 \09%r58, %r28, 5;\0A\09mul.wide.u32 \09%rd37, %r58, 4;\0A\09mov.b64 \09%rd38, __wg_main_kernel_0;\0A\09add.s64 \09%rd17, %rd38, %rd37;\0A\09st.shared.b32 \09[%rd17], %r92;\0A$L__BB0_14:\0A\09ld.param.b64 \09%rd21, [main_kernel_param_12];\0A\09bar.sync \090;\0A\09add.s32 \09%r60, %r1, 31;\0A\09shr.u32 \09%r10, %r60, 5;\0A\09setp.ge.u32 \09%p12, %r28, %r10;\0A\09@%p12 bra \09$L__BB0_19;\0A\09shl.b64 \09%rd39, %rd2, 2;\0A\09mov.b64 \09%rd40, __wg_main_kernel_0;\0A\09add.s64 \09%rd41, %rd40, %rd39;\0A\09ld.shared.b32 \09%r11, [%rd41];\0A\09setp.gt.u32 \09%p13, %r1, 992;\0A\09@%p13 bra \09$L__BB0_17;\0A\09sub.s32 \09%r70, 32, %r10;\0A\09mov.b32 \09%r71, -1;\0A\09shr.u32 \09%r72, %r71, %r70;\0A\09add.s32 \09%r73, %r10, -1;\0A\09shfl.sync.bfly.b32 \09%r74|%p14, %r11, 1, %r73, %r72;\0A\09add.rn.f32 \09%r75, %r11, %r74;\0A\09selp.f32 \09%r76, %r75, %r11, %p14;\0A\09shfl.sync.bfly.b32 \09%r77|%p15, %r76, 2, %r73, %r72;\0A\09add.rn.f32 \09%r78, %r77, %r76;\0A\09selp.f32 \09%r79, %r78, %r76, %p15;\0A\09shfl.sync.bfly.b32 \09%r80|%p16, %r79, 4, %r73, %r72;\0A\09add.rn.f32 \09%r81, %r80, %r79;\0A\09selp.f32 \09%r82, %r81, %r79, %p16;\0A\09shfl.sync.bfly.b32 \09%r83|%p17, %r82, 8, %r73, %r72;\0A\09add.rn.f32 \09%r84, %r83, %r82;\0A\09selp.f32 \09%r85, %r84, %r82, %p17;\0A\09shfl.sync.bfly.b32 \09%r86|%p18, %r85, 16, %r73, %r72;\0A\09add.rn.f32 \09%r87, %r86, %r85;\0A\09selp.f32 \09%r93, %r87, %r85, %p18;\0A\09bra.uni \09$L__BB0_18;\0A$L__BB0_17:\0A\09shfl.sync.bfly.b32 \09%r61, %r11, 1, 31, -1;\0A\09add.rn.f32 \09%r62, %r11, %r61;\0A\09shfl.sync.bfly.b32 \09%r63, %r62, 2, 31, -1;\0A\09add.rn.f32 \09%r64, %r62, %r63;\0A\09shfl.sync.bfly.b32 \09%r65, %r64, 4, 31, -1;\0A\09add.rn.f32 \09%r66, %r64, %r65;\0A\09shfl.sync.bfly.b32 \09%r67, %r66, 8, 31, -1;\0A\09add.rn.f32 \09%r68, %r66, %r67;\0A\09shfl.sync.bfly.b32 \09%r69, %r68, 16, 31, -1;\0A\09add.rn.f32 \09%r93, %r68, %r69;\0A$L__BB0_18:\0A\09st.shared.b32 \09[__wg_main_kernel_0], %r93;\0A$L__BB0_19:\0A\09bar.sync \090;\0A\09setp.ne.b64 \09%p19, %rd21, %rd2;\0A\09@%p19 bra \09$L__BB0_21;\0A\09ld.param.b64 \09%rd22, [main_kernel_param_14];\0A\09shl.b64 \09%rd42, %rd21, 2;\0A\09add.s64 \09%rd18, %rd22, %rd42;\0A\09ld.shared.b32 \09%r88, [__wg_main_kernel_0];\0A\09atom.global.add.f32 \09%r89, [%rd18], %r88;\0A$L__BB0_21:\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main_kernel_0  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main_kernel_0\0A\0A.visible .entry main_kernel_0(\0A\09.param .u64 .ptr .global .align 1 main_kernel_0_param_0,\0A\09.param .u64 .ptr .global .align 1 main_kernel_0_param_1,\0A\09.param .u64 main_kernel_0_param_2,\0A\09.param .u64 main_kernel_0_param_3,\0A\09.param .u64 main_kernel_0_param_4,\0A\09.param .u64 main_kernel_0_param_5,\0A\09.param .f32 main_kernel_0_param_6\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<4>;\0A\09.reg .b64 \09%rd<5>;\0A\0A\09ld.param.b64 \09%rd1, [main_kernel_0_param_5];\0A\09shl.b64 \09%rd2, %rd1, 2;\0A\09ld.param.b64 \09%rd3, [main_kernel_0_param_1];\0A\09add.s64 \09%rd4, %rd3, %rd2;\0A\09ld.global.b32 \09%r1, [%rd4];\0A\09ld.param.b32 \09%r2, [main_kernel_0_param_6];\0A\09div.rn.f32 \09%r3, %r1, %r2;\0A\09st.global.b32 \09[%rd4], %r3;\0A\09ret;\0A\0A}\0A">]
}

