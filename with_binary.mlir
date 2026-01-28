module attributes {gpu.container_module} {
  llvm.func @cudaMalloc(!llvm.ptr, i64) -> i32
  llvm.func @cudaMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
  llvm.func @cudaFree(!llvm.ptr) -> i32
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1xf32_0(dense<1.000000e+00> : tensor<1xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x f32>
  llvm.mlir.global private constant @__constant_1xf32(dense<1.250000e-01> : tensor<1xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x f32>
  llvm.mlir.global private constant @__constant_8x1xf32(dense<1.000000e+00> : tensor<8x1xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<8 x array<1 x f32>>
  llvm.mlir.global private constant @__constant_8x16xf32(dense<1.000000e+00> : tensor<8x16xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<8 x array<16 x f32>>
  llvm.func @main1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(16 : i64) : i64
    %5 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.mlir.poison : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)>
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.addressof @__constant_1xf32_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @__constant_1xf32 : !llvm.ptr
    %10 = llvm.mlir.addressof @__constant_8x1xf32 : !llvm.ptr
    %11 = llvm.mlir.addressof @__constant_8x16xf32 : !llvm.ptr
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.mlir.constant(16 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(8 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %18 = llvm.mlir.constant(1.250000e-01 : f32) : f32
    %19 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.constant(4 : i64) : i64
    %21 = llvm.mlir.constant(8 : i64) : i64
    %22 = llvm.mul %20, %21 : i64
    %23 = llvm.mul %22, %4 : i64
    %24 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %25 = llvm.call @cudaMalloc(%24, %23) : (!llvm.ptr, i64) -> i32
    %26 = llvm.load %24 : !llvm.ptr -> !llvm.ptr<1>
    %27 = llvm.mul %1, %4 : i64
    %28 = llvm.addrspacecast %26 : !llvm.ptr<1> to !llvm.ptr
    %29 = llvm.call @cudaMemcpy(%28, %arg1, %23, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %30 = llvm.getelementptr %11[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<16 x f32>>
    %31 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %32 = llvm.call @cudaMalloc(%31, %23) : (!llvm.ptr, i64) -> i32
    %33 = llvm.load %31 : !llvm.ptr -> !llvm.ptr<1>
    %34 = llvm.addrspacecast %33 : !llvm.ptr<1> to !llvm.ptr
    %35 = llvm.call @cudaMemcpy(%34, %30, %23, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %36 = llvm.getelementptr %10[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<1 x f32>>
    %37 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %38 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %39 = llvm.getelementptr %12[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.add %40, %7 : i64
    %42 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.sub %7, %14 : i64
    %45 = llvm.add %43, %44 : i64
    %46 = llvm.urem %45, %7 : i64
    %47 = llvm.sub %45, %46 : i64
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr
    %49 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %50 = llvm.call @cudaMalloc(%49, %22) : (!llvm.ptr, i64) -> i32
    %51 = llvm.load %49 : !llvm.ptr -> !llvm.ptr<1>
    %52 = llvm.addrspacecast %51 : !llvm.ptr<1> to !llvm.ptr
    %53 = llvm.call @cudaMemcpy(%52, %48, %22, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    gpu.launch_func  @main1_kernel::@main1_kernel blocks in (%15, %14, %14) threads in (%14, %14, %14) : i64 args(%17 : f32, %51 : !llvm.ptr<1>, %51 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64)
    gpu.launch_func  @main1_kernel_0::@main1_kernel blocks in (%15, %14, %14) threads in (%14, %14, %14) : i64 args(%26 : !llvm.ptr<1>, %26 : !llvm.ptr<1>, %2 : i64, %21 : i64, %4 : i64, %27 : i64, %1 : i64, %51 : !llvm.ptr<1>, %51 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64, %16 : i64, %13 : i64, %14 : i64)
    %54 = llvm.call @cudaFree(%28) : (!llvm.ptr) -> i32
    %55 = llvm.call @cudaFree(%52) : (!llvm.ptr) -> i32
    %56 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %57 = llvm.ptrtoint %56 : !llvm.ptr to i64
    %58 = llvm.add %57, %7 : i64
    %59 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.add %60, %44 : i64
    %62 = llvm.urem %61, %7 : i64
    %63 = llvm.sub %61, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    llvm.store %17, %64 : f32, !llvm.ptr
    llvm.br ^bb1(%16 : i64)
  ^bb1(%65: i64):  // 2 preds: ^bb0, ^bb2
    %66 = llvm.icmp "slt" %65, %15 : i64
    llvm.cond_br %66, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %67 = llvm.getelementptr inbounds|nuw %48[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %68 = llvm.load %67 : !llvm.ptr -> f32
    %69 = llvm.load %64 : !llvm.ptr -> f32
    %70 = llvm.fadd %68, %69 : f32
    llvm.store %70, %64 : f32, !llvm.ptr
    %71 = llvm.add %65, %14 : i64
    llvm.br ^bb1(%71 : i64)
  ^bb3:  // pred: ^bb1
    %72 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.add %73, %44 : i64
    %75 = llvm.urem %74, %7 : i64
    %76 = llvm.sub %74, %75 : i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    %78 = llvm.insertvalue %72, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %79 = llvm.insertvalue %77, %78[1] : !llvm.struct<(ptr, ptr, i64)> 
    %80 = llvm.insertvalue %16, %79[2] : !llvm.struct<(ptr, ptr, i64)> 
    %81 = llvm.load %64 : !llvm.ptr -> f32
    %82 = llvm.fmul %81, %18 : f32
    llvm.store %82, %77 : f32, !llvm.ptr
    %83 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.add %84, %44 : i64
    %86 = llvm.urem %85, %7 : i64
    %87 = llvm.sub %85, %86 : i64
    %88 = llvm.inttoptr %87 : i64 to !llvm.ptr
    %89 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %90 = llvm.call @cudaMalloc(%89, %20) : (!llvm.ptr, i64) -> i32
    %91 = llvm.load %89 : !llvm.ptr -> !llvm.ptr<1>
    %92 = llvm.addrspacecast %91 : !llvm.ptr<1> to !llvm.ptr
    %93 = llvm.call @cudaMemcpy(%92, %88, %20, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %94 = llvm.load %38 : !llvm.ptr -> f32
    %95 = llvm.load %37 : !llvm.ptr -> f32
    %96 = llvm.fmul %94, %95 : f32
    llvm.store %96, %88 : f32, !llvm.ptr
    %97 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
    %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
    %99 = llvm.add %98, %44 : i64
    %100 = llvm.urem %99, %7 : i64
    %101 = llvm.sub %99, %100 : i64
    %102 = llvm.inttoptr %101 : i64 to !llvm.ptr
    %103 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %104 = llvm.call @cudaMalloc(%103, %22) : (!llvm.ptr, i64) -> i32
    %105 = llvm.load %103 : !llvm.ptr -> !llvm.ptr<1>
    %106 = llvm.addrspacecast %105 : !llvm.ptr<1> to !llvm.ptr
    %107 = llvm.call @cudaMemcpy(%106, %102, %22, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    gpu.launch_func  @main1_kernel_1::@main1_kernel blocks in (%15, %14, %14) threads in (%14, %14, %14) : i64 args(%91 : !llvm.ptr<1>, %91 : !llvm.ptr<1>, %2 : i64, %105 : !llvm.ptr<1>, %105 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64)
    %108 = llvm.call @cudaFree(%92) : (!llvm.ptr) -> i32
    %109 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
    %110 = llvm.ptrtoint %109 : !llvm.ptr to i64
    %111 = llvm.add %110, %44 : i64
    %112 = llvm.urem %111, %7 : i64
    %113 = llvm.sub %111, %112 : i64
    %114 = llvm.inttoptr %113 : i64 to !llvm.ptr
    %115 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %116 = llvm.call @cudaMalloc(%115, %22) : (!llvm.ptr, i64) -> i32
    %117 = llvm.load %115 : !llvm.ptr -> !llvm.ptr<1>
    %118 = llvm.addrspacecast %117 : !llvm.ptr<1> to !llvm.ptr
    %119 = llvm.call @cudaMemcpy(%118, %36, %22, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %120 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %121 = llvm.call @cudaMalloc(%120, %22) : (!llvm.ptr, i64) -> i32
    %122 = llvm.load %120 : !llvm.ptr -> !llvm.ptr<1>
    %123 = llvm.addrspacecast %122 : !llvm.ptr<1> to !llvm.ptr
    %124 = llvm.call @cudaMemcpy(%123, %114, %22, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    gpu.launch_func  @main1_kernel_2::@main1_kernel blocks in (%15, %14, %14) threads in (%14, %14, %14) : i64 args(%105 : !llvm.ptr<1>, %105 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64, %117 : !llvm.ptr<1>, %117 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64, %122 : !llvm.ptr<1>, %122 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64)
    %125 = llvm.call @cudaFree(%118) : (!llvm.ptr) -> i32
    %126 = llvm.call @cudaFree(%106) : (!llvm.ptr) -> i32
    %127 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %128 = llvm.call @cudaMalloc(%127, %23) : (!llvm.ptr, i64) -> i32
    %129 = llvm.load %127 : !llvm.ptr -> !llvm.ptr<1>
    gpu.launch_func  @main1_kernel_3::@main1_kernel blocks in (%15, %13, %14) threads in (%14, %14, %14) : i64 args(%122 : !llvm.ptr<1>, %122 : !llvm.ptr<1>, %2 : i64, %21 : i64, %1 : i64, %129 : !llvm.ptr<1>, %129 : !llvm.ptr<1>, %2 : i64, %21 : i64, %4 : i64, %27 : i64, %1 : i64)
    %130 = llvm.call @cudaFree(%123) : (!llvm.ptr) -> i32
    %131 = llvm.alloca %3 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %132 = llvm.call @cudaMalloc(%131, %23) : (!llvm.ptr, i64) -> i32
    %133 = llvm.load %131 : !llvm.ptr -> !llvm.ptr<1>
    gpu.launch_func  @main1_kernel_4::@main1_kernel blocks in (%15, %13, %14) threads in (%14, %14, %14) : i64 args(%129 : !llvm.ptr<1>, %129 : !llvm.ptr<1>, %2 : i64, %21 : i64, %4 : i64, %27 : i64, %1 : i64, %33 : !llvm.ptr<1>, %33 : !llvm.ptr<1>, %2 : i64, %21 : i64, %4 : i64, %27 : i64, %1 : i64, %133 : !llvm.ptr<1>, %133 : !llvm.ptr<1>, %2 : i64, %21 : i64, %4 : i64, %27 : i64, %1 : i64)
    %134 = llvm.call @cudaFree(%34) : (!llvm.ptr) -> i32
    llvm.call @free(%42) : (!llvm.ptr) -> ()
    llvm.call @free(%59) : (!llvm.ptr) -> ()
    llvm.call @free(%83) : (!llvm.ptr) -> ()
    llvm.call @free(%97) : (!llvm.ptr) -> ()
    llvm.call @free(%109) : (!llvm.ptr) -> ()
    %135 = llvm.addrspacecast %129 : !llvm.ptr<1> to !llvm.ptr
    %136 = llvm.call @cudaFree(%135) : (!llvm.ptr) -> i32
    %137 = llvm.getelementptr %12[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %138 = llvm.ptrtoint %137 : !llvm.ptr to i64
    %139 = llvm.call @malloc(%138) : (i64) -> !llvm.ptr
    %140 = llvm.insertvalue %139, %19[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.insertvalue %139, %140[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.insertvalue %16, %141[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.insertvalue %15, %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.insertvalue %13, %143[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %13, %144[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.insertvalue %14, %145[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.addrspacecast %133 : !llvm.ptr<1> to !llvm.ptr
    %148 = llvm.call @cudaMemcpy(%139, %147, %23, %0) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %149 = llvm.call @cudaFree(%147) : (!llvm.ptr) -> i32
    %150 = llvm.insertvalue %80, %6[0] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)> 
    %151 = llvm.insertvalue %146, %150[1] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)> 
    llvm.return %151 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)>
  }
  llvm.func @_mlir_ciface_main1(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.call @main1(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)>
    llvm.store %8, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)>, !llvm.ptr
    llvm.return
  }
  gpu.binary @main1_kernel  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 1 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main1_kernel\0A\0A.visible .entry main1_kernel(\0A\09.param .f32 main1_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_1,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_2,\0A\09.param .u64 main1_kernel_param_3,\0A\09.param .u64 main1_kernel_param_4,\0A\09.param .u64 main1_kernel_param_5\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<3>;\0A\09.reg .b64 \09%rd<4>;\0A\0A\09ld.param.b32 \09%r1, [main1_kernel_param_0];\0A\09mov.u32 \09%r2, %ctaid.x;\0A\09ld.param.b64 \09%rd1, [main1_kernel_param_2];\0A\09mul.wide.u32 \09%rd2, %r2, 4;\0A\09add.s64 \09%rd3, %rd1, %rd2;\0A\09st.global.b32 \09[%rd3], %r1;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main1_kernel_0  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 1 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main1_kernel\0A\0A.visible .entry main1_kernel(\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_1,\0A\09.param .u64 main1_kernel_param_2,\0A\09.param .u64 main1_kernel_param_3,\0A\09.param .u64 main1_kernel_param_4,\0A\09.param .u64 main1_kernel_param_5,\0A\09.param .u64 main1_kernel_param_6,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_7,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_8,\0A\09.param .u64 main1_kernel_param_9,\0A\09.param .u64 main1_kernel_param_10,\0A\09.param .u64 main1_kernel_param_11,\0A\09.param .u64 main1_kernel_param_12,\0A\09.param .u64 main1_kernel_param_13,\0A\09.param .u64 main1_kernel_param_14\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .pred \09%p<3>;\0A\09.reg .b32 \09%r<7>;\0A\09.reg .b64 \09%rd<19>;\0A\0A\09ld.param.b64 \09%rd11, [main1_kernel_param_13];\0A\09ld.param.b64 \09%rd18, [main1_kernel_param_12];\0A\09setp.ge.s64 \09%p1, %rd18, %rd11;\0A\09@%p1 bra \09$L__BB0_3;\0A\09ld.param.b64 \09%rd12, [main1_kernel_param_14];\0A\09ld.param.b64 \09%rd9, [main1_kernel_param_8];\0A\09ld.param.b64 \09%rd8, [main1_kernel_param_1];\0A\09mov.u32 \09%r4, %ctaid.x;\0A\09mul.wide.u32 \09%rd13, %r4, 64;\0A\09mul.wide.u32 \09%rd14, %r4, 4;\0A\09add.s64 \09%rd1, %rd9, %rd14;\0A\09ld.global.b32 \09%r6, [%rd1];\0A\09shl.b64 \09%rd15, %rd18, 2;\0A\09add.s64 \09%rd16, %rd13, %rd15;\0A\09add.s64 \09%rd17, %rd8, %rd16;\0A\09shl.b64 \09%rd3, %rd12, 2;\0A$L__BB0_2:\0A\09ld.global.b32 \09%r5, [%rd17];\0A\09add.rn.f32 \09%r6, %r5, %r6;\0A\09st.global.b32 \09[%rd1], %r6;\0A\09add.s64 \09%rd18, %rd18, %rd12;\0A\09add.s64 \09%rd17, %rd17, %rd3;\0A\09setp.lt.s64 \09%p2, %rd18, %rd11;\0A\09@%p2 bra \09$L__BB0_2;\0A$L__BB0_3:\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main1_kernel_1  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main1_kernel\0A\0A.visible .entry main1_kernel(\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_1,\0A\09.param .u64 main1_kernel_param_2,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_3,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_4,\0A\09.param .u64 main1_kernel_param_5,\0A\09.param .u64 main1_kernel_param_6,\0A\09.param .u64 main1_kernel_param_7\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<3>;\0A\09.reg .b64 \09%rd<5>;\0A\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09ld.param.b64 \09%rd1, [main1_kernel_param_1];\0A\09ld.global.b32 \09%r2, [%rd1];\0A\09mul.wide.u32 \09%rd2, %r1, 4;\0A\09ld.param.b64 \09%rd3, [main1_kernel_param_4];\0A\09add.s64 \09%rd4, %rd3, %rd2;\0A\09st.global.b32 \09[%rd4], %r2;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main1_kernel_2  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main1_kernel\0A\0A.visible .entry main1_kernel(\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_1,\0A\09.param .u64 main1_kernel_param_2,\0A\09.param .u64 main1_kernel_param_3,\0A\09.param .u64 main1_kernel_param_4,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_5,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_6,\0A\09.param .u64 main1_kernel_param_7,\0A\09.param .u64 main1_kernel_param_8,\0A\09.param .u64 main1_kernel_param_9,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_10,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_11,\0A\09.param .u64 main1_kernel_param_12,\0A\09.param .u64 main1_kernel_param_13,\0A\09.param .u64 main1_kernel_param_14\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<5>;\0A\09.reg .b64 \09%rd<8>;\0A\0A\09ld.param.b64 \09%rd1, [main1_kernel_param_1];\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09mul.wide.u32 \09%rd2, %r1, 4;\0A\09add.s64 \09%rd3, %rd1, %rd2;\0A\09ld.global.b32 \09%r2, [%rd3];\0A\09ld.param.b64 \09%rd4, [main1_kernel_param_6];\0A\09add.s64 \09%rd5, %rd4, %rd2;\0A\09ld.global.b32 \09%r3, [%rd5];\0A\09mul.rn.f32 \09%r4, %r2, %r3;\0A\09ld.param.b64 \09%rd6, [main1_kernel_param_11];\0A\09add.s64 \09%rd7, %rd6, %rd2;\0A\09st.global.b32 \09[%rd7], %r4;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main1_kernel_3  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main1_kernel\0A\0A.visible .entry main1_kernel(\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_1,\0A\09.param .u64 main1_kernel_param_2,\0A\09.param .u64 main1_kernel_param_3,\0A\09.param .u64 main1_kernel_param_4,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_5,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_6,\0A\09.param .u64 main1_kernel_param_7,\0A\09.param .u64 main1_kernel_param_8,\0A\09.param .u64 main1_kernel_param_9,\0A\09.param .u64 main1_kernel_param_10,\0A\09.param .u64 main1_kernel_param_11\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<4>;\0A\09.reg .b64 \09%rd<9>;\0A\0A\09ld.param.b64 \09%rd1, [main1_kernel_param_1];\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09mov.u32 \09%r2, %ctaid.y;\0A\09mul.wide.u32 \09%rd2, %r1, 4;\0A\09add.s64 \09%rd3, %rd1, %rd2;\0A\09ld.global.b32 \09%r3, [%rd3];\0A\09ld.param.b64 \09%rd4, [main1_kernel_param_6];\0A\09mul.wide.u32 \09%rd5, %r1, 64;\0A\09add.s64 \09%rd6, %rd4, %rd5;\0A\09mul.wide.u32 \09%rd7, %r2, 4;\0A\09add.s64 \09%rd8, %rd6, %rd7;\0A\09st.global.b32 \09[%rd8], %r3;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main1_kernel_4  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main1_kernel\0A\0A.visible .entry main1_kernel(\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_1,\0A\09.param .u64 main1_kernel_param_2,\0A\09.param .u64 main1_kernel_param_3,\0A\09.param .u64 main1_kernel_param_4,\0A\09.param .u64 main1_kernel_param_5,\0A\09.param .u64 main1_kernel_param_6,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_7,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_8,\0A\09.param .u64 main1_kernel_param_9,\0A\09.param .u64 main1_kernel_param_10,\0A\09.param .u64 main1_kernel_param_11,\0A\09.param .u64 main1_kernel_param_12,\0A\09.param .u64 main1_kernel_param_13,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_14,\0A\09.param .u64 .ptr .global .align 1 main1_kernel_param_15,\0A\09.param .u64 main1_kernel_param_16,\0A\09.param .u64 main1_kernel_param_17,\0A\09.param .u64 main1_kernel_param_18,\0A\09.param .u64 main1_kernel_param_19,\0A\09.param .u64 main1_kernel_param_20\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<8>;\0A\09.reg .b64 \09%rd<8>;\0A\0A\09ld.param.b64 \09%rd1, [main1_kernel_param_1];\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09mov.u32 \09%r2, %ctaid.y;\0A\09shl.b32 \09%r3, %r1, 4;\0A\09or.b32 \09%r4, %r3, %r2;\0A\09mul.wide.u32 \09%rd2, %r4, 4;\0A\09add.s64 \09%rd3, %rd1, %rd2;\0A\09ld.global.b32 \09%r5, [%rd3];\0A\09ld.param.b64 \09%rd4, [main1_kernel_param_8];\0A\09add.s64 \09%rd5, %rd4, %rd2;\0A\09ld.global.b32 \09%r6, [%rd5];\0A\09mul.rn.f32 \09%r7, %r5, %r6;\0A\09ld.param.b64 \09%rd6, [main1_kernel_param_15];\0A\09add.s64 \09%rd7, %rd6, %rd2;\0A\09st.global.b32 \09[%rd7], %r7;\0A\09ret;\0A\0A}\0A">]
}

