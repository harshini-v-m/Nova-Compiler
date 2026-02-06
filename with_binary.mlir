module attributes {gpu.container_module} {
  llvm.func @cudaMalloc(!llvm.ptr, i64) -> i32
  llvm.func @cudaMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
  llvm.func @cudaFree(!llvm.ptr) -> i32
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.array<2 x i64>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.mlir.constant(8 : i64) : i64
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(10 : i64) : i64
    %8 = llvm.mlir.constant(10 : index) : i64
    %9 = llvm.mlir.constant(8 : index) : i64
    %10 = llvm.mlir.constant(16 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(4 : i64) : i64
    %15 = llvm.mlir.constant(16 : i64) : i64
    %16 = llvm.mul %14, %15 : i64
    %17 = llvm.mul %16, %7 : i64
    %18 = llvm.alloca %6 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %19 = llvm.call @cudaMalloc(%18, %17) : (!llvm.ptr, i64) -> i32
    %20 = llvm.load %18 : !llvm.ptr -> !llvm.ptr<1>
    %21 = llvm.mul %4, %7 : i64
    %22 = llvm.addrspacecast %20 : !llvm.ptr<1> to !llvm.ptr
    %23 = llvm.call @cudaMemcpy(%22, %arg8, %17, %3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %24 = llvm.mul %14, %2 : i64
    %25 = llvm.mul %24, %15 : i64
    %26 = llvm.alloca %6 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %27 = llvm.call @cudaMalloc(%26, %25) : (!llvm.ptr, i64) -> i32
    %28 = llvm.load %26 : !llvm.ptr -> !llvm.ptr<1>
    %29 = llvm.mul %4, %15 : i64
    %30 = llvm.addrspacecast %28 : !llvm.ptr<1> to !llvm.ptr
    %31 = llvm.call @cudaMemcpy(%30, %arg1, %25, %3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %32 = llvm.mul %14, %4 : i64
    %33 = llvm.mul %32, %7 : i64
    %34 = llvm.alloca %6 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %35 = llvm.call @cudaMalloc(%34, %33) : (!llvm.ptr, i64) -> i32
    %36 = llvm.load %34 : !llvm.ptr -> !llvm.ptr<1>
    %37 = llvm.addrspacecast %36 : !llvm.ptr<1> to !llvm.ptr
    %38 = llvm.call @cudaMemcpy(%37, %arg15, %33, %3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %39 = llvm.mul %24, %7 : i64
    %40 = llvm.alloca %6 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %41 = llvm.call @cudaMalloc(%40, %39) : (!llvm.ptr, i64) -> i32
    %42 = llvm.load %40 : !llvm.ptr -> !llvm.ptr
    %43 = llvm.alloca %6 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %44 = llvm.call @cudaMalloc(%43, %39) : (!llvm.ptr, i64) -> i32
    %45 = llvm.load %43 : !llvm.ptr -> !llvm.ptr<1>
    %46 = llvm.addrspacecast %45 : !llvm.ptr<1> to !llvm.ptr
    %47 = llvm.call @cudaMemcpy(%46, %42, %39, %3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%9, %8, %11) threads in (%11, %11, %11) : i64 args(%13 : f32, %45 : !llvm.ptr<1>, %45 : !llvm.ptr<1>, %5 : i64, %2 : i64, %7 : i64, %21 : i64, %4 : i64)
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%9, %8, %11) threads in (%11, %11, %11) : i64 args(%28 : !llvm.ptr<1>, %28 : !llvm.ptr<1>, %5 : i64, %2 : i64, %15 : i64, %29 : i64, %4 : i64, %20 : !llvm.ptr<1>, %20 : !llvm.ptr<1>, %5 : i64, %15 : i64, %7 : i64, %21 : i64, %4 : i64, %45 : !llvm.ptr<1>, %45 : !llvm.ptr<1>, %5 : i64, %2 : i64, %7 : i64, %21 : i64, %4 : i64, %12 : i64, %10 : i64, %11 : i64)
    %48 = llvm.call @cudaFree(%22) : (!llvm.ptr) -> i32
    %49 = llvm.call @cudaFree(%30) : (!llvm.ptr) -> i32
    %50 = llvm.alloca %6 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %51 = llvm.call @cudaMalloc(%50, %39) : (!llvm.ptr, i64) -> i32
    %52 = llvm.load %50 : !llvm.ptr -> !llvm.ptr
    %53 = llvm.insertvalue %52, %1[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %52, %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %5, %54[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %2, %0[0] : !llvm.array<2 x i64> 
    %57 = llvm.insertvalue %7, %56[1] : !llvm.array<2 x i64> 
    %58 = llvm.insertvalue %57, %55[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %4, %0[1] : !llvm.array<2 x i64> 
    %60 = llvm.insertvalue %21, %59[0] : !llvm.array<2 x i64> 
    %61 = llvm.insertvalue %60, %58[4] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.alloca %6 x !llvm.ptr<1> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %63 = llvm.call @cudaMalloc(%62, %39) : (!llvm.ptr, i64) -> i32
    %64 = llvm.load %62 : !llvm.ptr -> !llvm.ptr<1>
    %65 = llvm.addrspacecast %64 : !llvm.ptr<1> to !llvm.ptr
    %66 = llvm.call @cudaMemcpy(%65, %52, %39, %3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%9, %8, %11) threads in (%11, %11, %11) : i64 args(%45 : !llvm.ptr<1>, %45 : !llvm.ptr<1>, %5 : i64, %2 : i64, %7 : i64, %21 : i64, %4 : i64, %36 : !llvm.ptr<1>, %36 : !llvm.ptr<1>, %12 : i64, %8 : i64, %11 : i64, %13 : f32, %64 : !llvm.ptr<1>, %64 : !llvm.ptr<1>, %5 : i64, %2 : i64, %7 : i64, %21 : i64, %4 : i64)
    %67 = llvm.call @cudaFree(%65) : (!llvm.ptr) -> i32
    %68 = llvm.call @cudaFree(%46) : (!llvm.ptr) -> i32
    %69 = llvm.call @cudaFree(%37) : (!llvm.ptr) -> i32
    %70 = llvm.call @cudaFree(%42) : (!llvm.ptr) -> i32
    llvm.return %61 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main_kernel\0A\0A.visible .entry main_kernel(\0A\09.param .f32 main_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_1,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_2,\0A\09.param .u64 main_kernel_param_3,\0A\09.param .u64 main_kernel_param_4,\0A\09.param .u64 main_kernel_param_5,\0A\09.param .u64 main_kernel_param_6,\0A\09.param .u64 main_kernel_param_7\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<5>;\0A\09.reg .b64 \09%rd<4>;\0A\0A\09ld.param.b32 \09%r1, [main_kernel_param_0];\0A\09mov.u32 \09%r2, %ctaid.x;\0A\09ld.param.b64 \09%rd1, [main_kernel_param_2];\0A\09mov.u32 \09%r3, %ctaid.y;\0A\09mad.lo.s32 \09%r4, %r2, 10, %r3;\0A\09mul.wide.u32 \09%rd2, %r4, 4;\0A\09add.s64 \09%rd3, %rd1, %rd2;\0A\09st.global.b32 \09[%rd3], %r1;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main_kernel_0  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 1 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main_kernel\0A\0A.visible .entry main_kernel(\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_1,\0A\09.param .u64 main_kernel_param_2,\0A\09.param .u64 main_kernel_param_3,\0A\09.param .u64 main_kernel_param_4,\0A\09.param .u64 main_kernel_param_5,\0A\09.param .u64 main_kernel_param_6,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_7,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_8,\0A\09.param .u64 main_kernel_param_9,\0A\09.param .u64 main_kernel_param_10,\0A\09.param .u64 main_kernel_param_11,\0A\09.param .u64 main_kernel_param_12,\0A\09.param .u64 main_kernel_param_13,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_14,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_15,\0A\09.param .u64 main_kernel_param_16,\0A\09.param .u64 main_kernel_param_17,\0A\09.param .u64 main_kernel_param_18,\0A\09.param .u64 main_kernel_param_19,\0A\09.param .u64 main_kernel_param_20,\0A\09.param .u64 main_kernel_param_21,\0A\09.param .u64 main_kernel_param_22,\0A\09.param .u64 main_kernel_param_23\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .pred \09%p<3>;\0A\09.reg .b32 \09%r<10>;\0A\09.reg .b64 \09%rd<29>;\0A\0A\09ld.param.b64 \09%rd17, [main_kernel_param_22];\0A\09ld.param.b64 \09%rd28, [main_kernel_param_21];\0A\09setp.ge.s64 \09%p1, %rd28, %rd17;\0A\09@%p1 bra \09$L__BB0_3;\0A\09ld.param.b64 \09%rd18, [main_kernel_param_23];\0A\09ld.param.b64 \09%rd15, [main_kernel_param_15];\0A\09ld.param.b64 \09%rd14, [main_kernel_param_8];\0A\09ld.param.b64 \09%rd13, [main_kernel_param_1];\0A\09mov.u32 \09%r4, %ctaid.y;\0A\09cvt.u64.u32 \09%rd1, %r4;\0A\09shl.b64 \09%rd19, %rd1, 2;\0A\09add.s64 \09%rd20, %rd15, %rd19;\0A\09mov.u32 \09%r5, %ctaid.x;\0A\09mul.wide.u32 \09%rd21, %r5, 64;\0A\09mul.wide.u32 \09%rd22, %r5, 40;\0A\09add.s64 \09%rd2, %rd20, %rd22;\0A\09ld.global.b32 \09%r9, [%rd2];\0A\09shl.b64 \09%rd23, %rd28, 2;\0A\09add.s64 \09%rd24, %rd21, %rd23;\0A\09add.s64 \09%rd27, %rd13, %rd24;\0A\09shl.b64 \09%rd4, %rd18, 2;\0A\09mad.lo.s64 \09%rd25, %rd28, 40, %rd19;\0A\09add.s64 \09%rd26, %rd14, %rd25;\0A\09mul.lo.s64 \09%rd6, %rd18, 40;\0A$L__BB0_2:\0A\09ld.global.b32 \09%r6, [%rd27];\0A\09ld.global.b32 \09%r7, [%rd26];\0A\09mul.rn.f32 \09%r8, %r6, %r7;\0A\09add.rn.f32 \09%r9, %r9, %r8;\0A\09st.global.b32 \09[%rd2], %r9;\0A\09add.s64 \09%rd28, %rd28, %rd18;\0A\09add.s64 \09%rd27, %rd27, %rd4;\0A\09add.s64 \09%rd26, %rd26, %rd6;\0A\09setp.lt.s64 \09%p2, %rd28, %rd17;\0A\09@%p2 bra \09$L__BB0_2;\0A$L__BB0_3:\0A\09ret;\0A\0A}\0A">]
  gpu.binary @main_kernel_1  [#gpu.object<#nvvm.target<chip = "sm_86">, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 2 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09main_kernel\0A\0A.visible .entry main_kernel(\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_1,\0A\09.param .u64 main_kernel_param_2,\0A\09.param .u64 main_kernel_param_3,\0A\09.param .u64 main_kernel_param_4,\0A\09.param .u64 main_kernel_param_5,\0A\09.param .u64 main_kernel_param_6,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_7,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_8,\0A\09.param .u64 main_kernel_param_9,\0A\09.param .u64 main_kernel_param_10,\0A\09.param .u64 main_kernel_param_11,\0A\09.param .f32 main_kernel_param_12,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_13,\0A\09.param .u64 .ptr .global .align 1 main_kernel_param_14,\0A\09.param .u64 main_kernel_param_15,\0A\09.param .u64 main_kernel_param_16,\0A\09.param .u64 main_kernel_param_17,\0A\09.param .u64 main_kernel_param_18,\0A\09.param .u64 main_kernel_param_19\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<9>;\0A\09.reg .b64 \09%rd<9>;\0A\0A\09ld.param.b64 \09%rd1, [main_kernel_param_1];\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09mov.u32 \09%r2, %ctaid.y;\0A\09mad.lo.s32 \09%r3, %r1, 10, %r2;\0A\09mul.wide.u32 \09%rd2, %r3, 4;\0A\09add.s64 \09%rd3, %rd1, %rd2;\0A\09ld.global.b32 \09%r4, [%rd3];\0A\09ld.param.b64 \09%rd4, [main_kernel_param_8];\0A\09mul.wide.u32 \09%rd5, %r2, 4;\0A\09add.s64 \09%rd6, %rd4, %rd5;\0A\09ld.global.b32 \09%r5, [%rd6];\0A\09add.rn.f32 \09%r6, %r4, %r5;\0A\09ld.param.b32 \09%r7, [main_kernel_param_12];\0A\09max.NaN.f32 \09%r8, %r6, %r7;\0A\09ld.param.b64 \09%rd7, [main_kernel_param_14];\0A\09add.s64 \09%rd8, %rd7, %rd2;\0A\09st.global.b32 \09[%rd8], %r8;\0A\09ret;\0A\0A}\0A">]
}

