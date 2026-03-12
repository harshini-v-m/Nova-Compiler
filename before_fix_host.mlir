[GpuRuntimeLowering] Done.
// -----// IR Dump Before mlir::nova::FixHostGpuMemoryPass (fix-host-gpu-memory) //----- //
module attributes {gpu.container_module} {
  llvm.func @cudaMemsetAsync(!llvm.ptr, i32, i64, !llvm.ptr) -> i32
  llvm.func @cudaMallocAsync(!llvm.ptr, i64, !llvm.ptr) -> i32
  llvm.func @cudaFreeAsync(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @cudaMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
  llvm.mlir.global private constant @__constant_4xi64(dense<[0, 10, 20, 30]> : tensor<4xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x i64>
  llvm.mlir.global private constant @__constant_1xindex(dense<40> : tensor<1xindex>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x i64>
  llvm.mlir.global private constant @__constant_2xindex(dense<[4, 10]> : tensor<2xindex>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x i64>
  llvm.func @test_sce(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(10 : index) : i64
    %4 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(4 : index) : i64
    %8 = llvm.mlir.constant(256 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(4 : i64) : i64
    %11 = llvm.mlir.constant(10 : i64) : i64
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.mlir.undef : !llvm.array<1 x i64>
    %16 = llvm.mlir.constant(3 : i32) : i32
    %17 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.mlir.constant(2 : i32) : i32
    %19 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %arg7, %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %arg8, %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %arg9, %22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %arg10, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %arg11, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg0, %19[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg1, %26[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %arg3, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %arg5, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %arg4, %30[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %arg6, %31[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.alloca %9 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %32, %33 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %34 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    llvm.call @mgpuMemHostRegisterMemRef(%1, %33, %35) : (i64, !llvm.ptr, i64) -> ()
    %36 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %37 = llvm.mul %10, %10 : i64
    %38 = llvm.mul %37, %11 : i64
    %39 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %40 = llvm.call @cudaMallocAsync(%39, %38, %36) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %41 = llvm.load %39 : !llvm.ptr -> !llvm.ptr
    %42 = llvm.call @cudaMemcpyAsync(%41, %arg1, %38, %12, %36) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %43 = llvm.alloca %9 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %25, %43 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @mgpuStreamSynchronize(%36) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%36) : (!llvm.ptr) -> ()
    %44 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    llvm.call @mgpuMemHostRegisterMemRef(%9, %43, %45) : (i64, !llvm.ptr, i64) -> ()
    %46 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %47 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %48 = llvm.call @cudaMallocAsync(%47, %37, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %49 = llvm.load %47 : !llvm.ptr -> !llvm.ptr
    %50 = llvm.call @cudaMemcpyAsync(%49, %arg8, %37, %12, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %51 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %52 = llvm.call @cudaMallocAsync(%51, %37, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %53 = llvm.load %51 : !llvm.ptr -> !llvm.ptr
    %54 = llvm.call @cudaMemcpyAsync(%53, %arg8, %37, %12, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %55 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %56 = llvm.call @cudaMallocAsync(%55, %38, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %57 = llvm.load %55 : !llvm.ptr -> !llvm.ptr
    %58 = llvm.call @cudaMemcpyAsync(%57, %arg1, %38, %12, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %59 = llvm.mul %10, %13 : i64
    %60 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %61 = llvm.call @cudaMallocAsync(%60, %59, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %62 = llvm.load %60 : !llvm.ptr -> !llvm.ptr
    %63 = llvm.addrspacecast %62 : !llvm.ptr to !llvm.ptr<1>
    %64 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %65 = llvm.call @cudaMallocAsync(%64, %38, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %66 = llvm.load %64 : !llvm.ptr -> !llvm.ptr
    %67 = llvm.mul %13, %11 : i64
    %68 = llvm.call @cudaMemcpyAsync(%66, %57, %38, %16, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %69 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %70 = llvm.call @cudaMallocAsync(%69, %37, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %71 = llvm.load %69 : !llvm.ptr -> !llvm.ptr
    %72 = llvm.call @cudaMemcpyAsync(%71, %53, %37, %16, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    gpu.launch_func <%46 : !llvm.ptr> @test_sce_kernel::@test_sce_kernel blocks in (%9, %9, %9) threads in (%8, %9, %9) : i64 args(%71 : !llvm.ptr, %71 : !llvm.ptr, %14 : i64, %10 : i64, %13 : i64, %66 : !llvm.ptr, %66 : !llvm.ptr, %14 : i64, %10 : i64, %11 : i64, %67 : i64, %13 : i64, %3 : i64, %4 : f32, %5 : f32, %6 : i64, %7 : i64, %9 : i64, %2 : f32, %63 : !llvm.ptr<1>, %63 : !llvm.ptr<1>, %14 : i64, %13 : i64, %13 : i64)
    %73 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %74 = llvm.call @cudaMallocAsync(%73, %59, %46) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %75 = llvm.load %73 : !llvm.ptr -> !llvm.ptr
    %76 = llvm.insertvalue %75, %17[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.insertvalue %75, %76[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.insertvalue %14, %77[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.insertvalue %13, %15[0] : !llvm.array<1 x i64> 
    %80 = llvm.insertvalue %79, %78[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.insertvalue %79, %80[4] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.call @cudaMemcpyAsync(%75, %62, %59, %16, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %83 = llvm.call @cudaFreeAsync(%53, %46) : (!llvm.ptr, !llvm.ptr) -> i32
    %84 = llvm.call @cudaFreeAsync(%57, %46) : (!llvm.ptr, !llvm.ptr) -> i32
    %85 = llvm.call @cudaFreeAsync(%62, %46) : (!llvm.ptr, !llvm.ptr) -> i32
    %86 = llvm.call @cudaFreeAsync(%66, %46) : (!llvm.ptr, !llvm.ptr) -> i32
    %87 = llvm.call @cudaFreeAsync(%71, %46) : (!llvm.ptr, !llvm.ptr) -> i32
    %88 = llvm.call @cudaMemcpyAsync(%arg1, %41, %38, %18, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %89 = llvm.call @cudaMemcpyAsync(%arg8, %49, %37, %18, %46) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    llvm.call @mgpuStreamSynchronize(%46) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%46) : (!llvm.ptr) -> ()
    llvm.return %81 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  gpu.binary @test_sce_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 4 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_kernel\0A// __wg_test_sce_kernel_0 has been demoted\0A// __wg_test_sce_kernel_1 has been demoted\0A// __wg_test_sce_kernel_2 has been demoted\0A\0A.visible .entry test_sce_kernel(\0A\09.param .u64 .ptr .align 1 test_sce_kernel_param_0,\0A\09.param .u64 .ptr .align 1 test_sce_kernel_param_1,\0A\09.param .u64 test_sce_kernel_param_2,\0A\09.param .u64 test_sce_kernel_param_3,\0A\09.param .u64 test_sce_kernel_param_4,\0A\09.param .u64 .ptr .align 1 test_sce_kernel_param_5,\0A\09.param .u64 .ptr .align 1 test_sce_kernel_param_6,\0A\09.param .u64 test_sce_kernel_param_7,\0A\09.param .u64 test_sce_kernel_param_8,\0A\09.param .u64 test_sce_kernel_param_9,\0A\09.param .u64 test_sce_kernel_param_10,\0A\09.param .u64 test_sce_kernel_param_11,\0A\09.param .u64 test_sce_kernel_param_12,\0A\09.param .f32 test_sce_kernel_param_13,\0A\09.param .f32 test_sce_kernel_param_14,\0A\09.param .u64 test_sce_kernel_param_15,\0A\09.param .u64 test_sce_kernel_param_16,\0A\09.param .u64 test_sce_kernel_param_17,\0A\09.param .f32 test_sce_kernel_param_18,\0A\09.param .u64 .ptr .global .align 1 test_sce_kernel_param_19,\0A\09.param .u64 .ptr .global .align 1 test_sce_kernel_param_20,\0A\09.param .u64 test_sce_kernel_param_21,\0A\09.param .u64 test_sce_kernel_param_22,\0A\09.param .u64 test_sce_kernel_param_23\0A)\0A.maxntid 256, 1, 1\0A{\0A\09.reg .pred \09%p<51>;\0A\09.reg .b32 \09%r<224>;\0A\09.reg .b64 \09%rd<54>;\0A\09// demoted variable\0A\09.shared .align 4 .b8 __wg_test_sce_kernel_0[128];\0A\09// demoted variable\0A\09.shared .align 4 .b8 __wg_test_sce_kernel_1[128];\0A\09// demoted variable\0A\09.shared .align 4 .b8 __wg_test_sce_kernel_2[128];\0A\09ld.param.b64 \09%rd33, [test_sce_kernel_param_16];\0A\09ld.param.b64 \09%rd32, [test_sce_kernel_param_15];\0A\09ld.param.b32 \09%r40, [test_sce_kernel_param_14];\0A\09ld.param.b64 \09%rd36, [test_sce_kernel_param_1];\0A\09cvta.to.global.u64 \09%rd1, %rd36;\0A\09ld.param.b64 \09%rd37, [test_sce_kernel_param_6];\0A\09cvta.to.global.u64 \09%rd2, %rd37;\0A\09mov.u32 \09%r42, %tid.x;\0A\09cvt.u64.u32 \09%rd3, %r42;\0A\09setp.ge.s64 \09%p1, %rd32, %rd33;\0A\09mov.b32 \09%r223, %r40;\0A\09@%p1 bra \09$L__BB0_33;\0A\09ld.param.b64 \09%rd34, [test_sce_kernel_param_17];\0A\09ld.param.b32 \09%r39, [test_sce_kernel_param_13];\0A\09ld.param.b64 \09%rd31, [test_sce_kernel_param_12];\0A\09mov.u32 \09%r43, %ntid.x;\0A\09cvt.u64.u32 \09%rd4, %r43;\0A\09cvt.u32.u64 \09%r44, %rd3;\0A\09cvt.u32.u64 \09%r45, %rd4;\0A\09and.b32 \09%r1, %r44, 31;\0A\09and.b32 \09%r46, %r44, 224;\0A\09sub.s32 \09%r2, %r45, %r46;\0A\09sub.s32 \09%r47, 32, %r2;\0A\09mov.b32 \09%r48, -1;\0A\09shr.u32 \09%r3, %r48, %r47;\0A\09add.s32 \09%r4, %r2, -1;\0A\09shr.u32 \09%r49, %r44, 5;\0A\09mul.wide.u32 \09%rd38, %r49, 4;\0A\09mov.b64 \09%rd39, __wg_test_sce_kernel_0;\0A\09add.s64 \09%rd5, %rd39, %rd38;\0A\09add.s32 \09%r50, %r45, 31;\0A\09shr.u32 \09%r5, %r50, 5;\0A\09shl.b64 \09%rd40, %rd3, 2;\0A\09add.s64 \09%rd6, %rd39, %rd40;\0A\09sub.s32 \09%r51, 32, %r5;\0A\09shr.u32 \09%r6, %r48, %r51;\0A\09add.s32 \09%r7, %r5, -1;\0A\09mov.b64 \09%rd41, __wg_test_sce_kernel_1;\0A\09add.s64 \09%rd7, %rd41, %rd38;\0A\09add.s64 \09%rd8, %rd41, %rd40;\0A\09mov.b64 \09%rd42, __wg_test_sce_kernel_2;\0A\09add.s64 \09%rd9, %rd42, %rd38;\0A\09add.s64 \09%rd10, %rd42, %rd40;\0A\09mad.lo.s64 \09%rd43, %rd32, 40, %rd40;\0A\09add.s64 \09%rd47, %rd2, %rd43;\0A\09mul.lo.s64 \09%rd12, %rd34, 40;\0A\09shl.b64 \09%rd13, %rd4, 2;\0A\09setp.le.s64 \09%p2, %rd31, %rd3;\0A\09setp.gt.s32 \09%p4, %r2, 31;\0A\09setp.ne.b32 \09%p10, %r1, 0;\0A\09mov.b32 \09%r223, %r40;\0A\09mov.b64 \09%rd48, %rd32;\0A\09bra.uni \09$L__BB0_2;\0A$L__BB0_31:\0A\09ld.shared.b32 \09%r170, [%rd10];\0A\09shfl.sync.bfly.b32 \09%r171|%p41, %r170, 1, %r7, %r6;\0A\09add.rn.f32 \09%r172, %r170, %r171;\0A\09selp.f32 \09%r173, %r172, %r170, %p41;\0A\09shfl.sync.bfly.b32 \09%r174|%p42, %r173, 2, %r7, %r6;\0A\09add.rn.f32 \09%r175, %r174, %r173;\0A\09selp.f32 \09%r176, %r175, %r173, %p42;\0A\09shfl.sync.bfly.b32 \09%r177|%p43, %r176, 4, %r7, %r6;\0A\09add.rn.f32 \09%r178, %r177, %r176;\0A\09selp.f32 \09%r179, %r178, %r176, %p43;\0A\09shfl.sync.bfly.b32 \09%r180|%p44, %r179, 8, %r7, %r6;\0A\09add.rn.f32 \09%r181, %r180, %r179;\0A\09selp.f32 \09%r182, %r181, %r179, %p44;\0A\09shfl.sync.bfly.b32 \09%r183|%p45, %r182, 16, %r7, %r6;\0A\09add.rn.f32 \09%r184, %r183, %r182;\0A\09selp.f32 \09%r222, %r184, %r182, %p45;\0A\09st.shared.b32 \09[__wg_test_sce_kernel_2], %r222;\0A$L__BB0_32:\0A\09setp.lt.f32 \09%p46, %r220, 0f00800000;\0A\09mul.rn.f32 \09%r185, %r220, 0f4B000000;\0A\09selp.f32 \09%r186, %r185, %r220, %p46;\0A\09selp.f32 \09%r187, 0fC1B80000, 0f00000000, %p46;\0A\09add.s32 \09%r188, %r186, -1059760811;\0A\09and.b32 \09%r189, %r188, -8388608;\0A\09sub.s32 \09%r190, %r186, %r189;\0A\09cvt.rn.f32.s32 \09%r191, %r189;\0A\09fma.rn.f32 \09%r192, %r191, 0f34000000, %r187;\0A\09add.rn.f32 \09%r193, %r190, 0fBF800000;\0A\09fma.rn.f32 \09%r194, %r193, 0fBE055027, 0f3E1039F6;\0A\09fma.rn.f32 \09%r195, %r194, %r193, 0fBDF8CDCC;\0A\09fma.rn.f32 \09%r196, %r195, %r193, 0f3E0F2955;\0A\09fma.rn.f32 \09%r197, %r196, %r193, 0fBE2AD8B9;\0A\09fma.rn.f32 \09%r198, %r197, %r193, 0f3E4CED0B;\0A\09fma.rn.f32 \09%r199, %r198, %r193, 0fBE7FFF22;\0A\09fma.rn.f32 \09%r200, %r199, %r193, 0f3EAAAA78;\0A\09fma.rn.f32 \09%r201, %r200, %r193, 0fBF000000;\0A\09mul.rn.f32 \09%r202, %r193, %r201;\0A\09fma.rn.f32 \09%r203, %r202, %r193, %r193;\0A\09fma.rn.f32 \09%r204, %r192, 0f3F317218, %r203;\0A\09setp.gt.u32 \09%p47, %r186, 2139095039;\0A\09fma.rn.f32 \09%r205, %r186, 0f7F800000, 0f7F800000;\0A\09selp.f32 \09%r206, %r205, %r204, %p47;\0A\09setp.eq.f32 \09%p48, %r186, 0f00000000;\0A\09selp.f32 \09%r207, 0fFF800000, %r206, %p48;\0A\09sub.rn.f32 \09%r208, %r222, %r214;\0A\09sub.rn.f32 \09%r209, %r207, %r208;\0A\09add.rn.f32 \09%r223, %r223, %r209;\0A\09add.s64 \09%rd48, %rd48, %rd34;\0A\09add.s64 \09%rd47, %rd47, %rd12;\0A\09setp.lt.s64 \09%p49, %rd48, %rd33;\0A\09@%p49 bra \09$L__BB0_2;\0A\09bra.uni \09$L__BB0_33;\0A$L__BB0_2:\0A\09mov.b32 \09%r212, %r39;\0A\09@%p2 bra \09$L__BB0_5;\0A\09mov.b64 \09%rd49, %rd47;\0A\09mov.b32 \09%r212, %r39;\0A\09mov.b64 \09%rd50, %rd3;\0A$L__BB0_4:\0A\09ld.global.b32 \09%r52, [%rd49];\0A\09max.NaN.f32 \09%r212, %r212, %r52;\0A\09add.s64 \09%rd50, %rd50, %rd4;\0A\09add.s64 \09%rd49, %rd49, %rd13;\0A\09setp.lt.s64 \09%p3, %rd50, %rd31;\0A\09@%p3 bra \09$L__BB0_4;\0A$L__BB0_5:\0A\09shl.b64 \09%rd44, %rd48, 2;\0A\09add.s64 \09%rd45, %rd1, %rd44;\0A\09ld.global.s32 \09%rd16, [%rd45];\0A\09@%p4 bra \09$L__BB0_7;\0A\09shfl.sync.bfly.b32 \09%r62|%p5, %r212, 1, %r4, %r3;\0A\09max.NaN.f32 \09%r63, %r212, %r62;\0A\09selp.f32 \09%r64, %r63, %r212, %p5;\0A\09shfl.sync.bfly.b32 \09%r65|%p6, %r64, 2, %r4, %r3;\0A\09max.NaN.f32 \09%r66, %r64, %r65;\0A\09selp.f32 \09%r67, %r66, %r64, %p6;\0A\09shfl.sync.bfly.b32 \09%r68|%p7, %r67, 4, %r4, %r3;\0A\09max.NaN.f32 \09%r69, %r67, %r68;\0A\09selp.f32 \09%r70, %r69, %r67, %p7;\0A\09shfl.sync.bfly.b32 \09%r71|%p8, %r70, 8, %r4, %r3;\0A\09max.NaN.f32 \09%r72, %r70, %r71;\0A\09selp.f32 \09%r73, %r72, %r70, %p8;\0A\09shfl.sync.bfly.b32 \09%r74|%p9, %r73, 16, %r4, %r3;\0A\09max.NaN.f32 \09%r75, %r73, %r74;\0A\09selp.f32 \09%r213, %r75, %r73, %p9;\0A\09bra.uni \09$L__BB0_8;\0A$L__BB0_7:\0A\09shfl.sync.bfly.b32 \09%r53, %r212, 1, 31, -1;\0A\09max.NaN.f32 \09%r54, %r212, %r53;\0A\09shfl.sync.bfly.b32 \09%r55, %r54, 2, 31, -1;\0A\09max.NaN.f32 \09%r56, %r54, %r55;\0A\09shfl.sync.bfly.b32 \09%r57, %r56, 4, 31, -1;\0A\09max.NaN.f32 \09%r58, %r56, %r57;\0A\09shfl.sync.bfly.b32 \09%r59, %r58, 8, 31, -1;\0A\09max.NaN.f32 \09%r60, %r58, %r59;\0A\09shfl.sync.bfly.b32 \09%r61, %r60, 16, 31, -1;\0A\09max.NaN.f32 \09%r213, %r60, %r61;\0A$L__BB0_8:\0A\09@%p10 bra \09$L__BB0_10;\0A\09st.shared.b32 \09[%rd5], %r213;\0A$L__BB0_10:\0A\09setp.lt.u32 \09%p11, %r44, %r5;\0A\09@%p11 bra \09$L__BB0_12;\0A\09bra.uni \09$L__BB0_11;\0A$L__BB0_12:\0A\09ld.shared.b32 \09%r77, [%rd6];\0A\09shfl.sync.bfly.b32 \09%r78|%p12, %r77, 1, %r7, %r6;\0A\09max.NaN.f32 \09%r79, %r77, %r78;\0A\09selp.f32 \09%r80, %r79, %r77, %p12;\0A\09shfl.sync.bfly.b32 \09%r81|%p13, %r80, 2, %r7, %r6;\0A\09max.NaN.f32 \09%r82, %r80, %r81;\0A\09selp.f32 \09%r83, %r82, %r80, %p13;\0A\09shfl.sync.bfly.b32 \09%r84|%p14, %r83, 4, %r7, %r6;\0A\09max.NaN.f32 \09%r85, %r83, %r84;\0A\09selp.f32 \09%r86, %r85, %r83, %p14;\0A\09shfl.sync.bfly.b32 \09%r87|%p15, %r86, 8, %r7, %r6;\0A\09max.NaN.f32 \09%r88, %r86, %r87;\0A\09selp.f32 \09%r89, %r88, %r86, %p15;\0A\09shfl.sync.bfly.b32 \09%r90|%p16, %r89, 16, %r7, %r6;\0A\09max.NaN.f32 \09%r91, %r89, %r90;\0A\09selp.f32 \09%r214, %r91, %r89, %p16;\0A\09st.shared.b32 \09[__wg_test_sce_kernel_0], %r214;\0A\09bra.uni \09$L__BB0_13;\0A$L__BB0_11:\0A\09ld.shared.b32 \09%r214, [__wg_test_sce_kernel_0];\0A$L__BB0_13:\0A\09mov.b32 \09%r217, %r40;\0A\09mov.b32 \09%r218, %r40;\0A\09@%p2 bra \09$L__BB0_16;\0A\09sub.s64 \09%rd51, %rd3, %rd16;\0A\09mov.b64 \09%rd52, %rd47;\0A\09mov.b32 \09%r218, %r40;\0A\09mov.b32 \09%r217, %r40;\0A\09mov.b64 \09%rd53, %rd3;\0A$L__BB0_15:\0A\09ld.global.b32 \09%r92, [%rd52];\0A\09sub.rn.f32 \09%r93, %r92, %r214;\0A\09fma.rn.f32 \09%r94, %r93, 0f3BBB989D, 0f3F000000;\0A\09cvt.sat.f32.f32 \09%r95, %r94;\0A\09mov.b32 \09%r96, 0f4B400001;\0A\09mov.b32 \09%r97, 0f437C0000;\0A\09fma.rm.f32 \09%r98, %r95, %r97, %r96;\0A\09add.rn.f32 \09%r99, %r98, 0fCB40007F;\0A\09neg.f32 \09%r100, %r99;\0A\09fma.rn.f32 \09%r101, %r93, 0f3FB8AA3B, %r100;\0A\09fma.rn.f32 \09%r102, %r93, 0f32A57060, %r101;\0A\09shl.b32 \09%r103, %r98, 23;\0A\09ex2.approx.ftz.f32 \09%r104, %r102;\0A\09mul.rn.f32 \09%r105, %r104, %r103;\0A\09add.rn.f32 \09%r217, %r217, %r105;\0A\09setp.eq.b64 \09%p18, %rd51, 0;\0A\09selp.f32 \09%r106, %r92, %r40, %p18;\0A\09add.rn.f32 \09%r218, %r218, %r106;\0A\09add.s64 \09%rd53, %rd53, %rd4;\0A\09add.s64 \09%rd52, %rd52, %rd13;\0A\09add.s64 \09%rd51, %rd51, %rd4;\0A\09setp.lt.s64 \09%p19, %rd53, %rd31;\0A\09@%p19 bra \09$L__BB0_15;\0A$L__BB0_16:\0A\09@%p4 bra \09$L__BB0_18;\0A\09shfl.sync.bfly.b32 \09%r116|%p21, %r217, 1, %r4, %r3;\0A\09add.rn.f32 \09%r117, %r217, %r116;\0A\09selp.f32 \09%r118, %r117, %r217, %p21;\0A\09shfl.sync.bfly.b32 \09%r119|%p22, %r118, 2, %r4, %r3;\0A\09add.rn.f32 \09%r120, %r119, %r118;\0A\09selp.f32 \09%r121, %r120, %r118, %p22;\0A\09shfl.sync.bfly.b32 \09%r122|%p23, %r121, 4, %r4, %r3;\0A\09add.rn.f32 \09%r123, %r122, %r121;\0A\09selp.f32 \09%r124, %r123, %r121, %p23;\0A\09shfl.sync.bfly.b32 \09%r125|%p24, %r124, 8, %r4, %r3;\0A\09add.rn.f32 \09%r126, %r125, %r124;\0A\09selp.f32 \09%r127, %r126, %r124, %p24;\0A\09shfl.sync.bfly.b32 \09%r128|%p25, %r127, 16, %r4, %r3;\0A\09add.rn.f32 \09%r129, %r128, %r127;\0A\09selp.f32 \09%r219, %r129, %r127, %p25;\0A\09bra.uni \09$L__BB0_19;\0A$L__BB0_18:\0A\09shfl.sync.bfly.b32 \09%r107, %r217, 1, 31, -1;\0A\09add.rn.f32 \09%r108, %r217, %r107;\0A\09shfl.sync.bfly.b32 \09%r109, %r108, 2, 31, -1;\0A\09add.rn.f32 \09%r110, %r108, %r109;\0A\09shfl.sync.bfly.b32 \09%r111, %r110, 4, 31, -1;\0A\09add.rn.f32 \09%r112, %r110, %r111;\0A\09shfl.sync.bfly.b32 \09%r113, %r112, 8, 31, -1;\0A\09add.rn.f32 \09%r114, %r112, %r113;\0A\09shfl.sync.bfly.b32 \09%r115, %r114, 16, 31, -1;\0A\09add.rn.f32 \09%r219, %r114, %r115;\0A$L__BB0_19:\0A\09@%p10 bra \09$L__BB0_21;\0A\09st.shared.b32 \09[%rd7], %r219;\0A$L__BB0_21:\0A\09@%p11 bra \09$L__BB0_23;\0A\09bra.uni \09$L__BB0_22;\0A$L__BB0_23:\0A\09ld.shared.b32 \09%r131, [%rd8];\0A\09shfl.sync.bfly.b32 \09%r132|%p28, %r131, 1, %r7, %r6;\0A\09add.rn.f32 \09%r133, %r131, %r132;\0A\09selp.f32 \09%r134, %r133, %r131, %p28;\0A\09shfl.sync.bfly.b32 \09%r135|%p29, %r134, 2, %r7, %r6;\0A\09add.rn.f32 \09%r136, %r135, %r134;\0A\09selp.f32 \09%r137, %r136, %r134, %p29;\0A\09shfl.sync.bfly.b32 \09%r138|%p30, %r137, 4, %r7, %r6;\0A\09add.rn.f32 \09%r139, %r138, %r137;\0A\09selp.f32 \09%r140, %r139, %r137, %p30;\0A\09shfl.sync.bfly.b32 \09%r141|%p31, %r140, 8, %r7, %r6;\0A\09add.rn.f32 \09%r142, %r141, %r140;\0A\09selp.f32 \09%r143, %r142, %r140, %p31;\0A\09shfl.sync.bfly.b32 \09%r144|%p32, %r143, 16, %r7, %r6;\0A\09add.rn.f32 \09%r145, %r144, %r143;\0A\09selp.f32 \09%r220, %r145, %r143, %p32;\0A\09st.shared.b32 \09[__wg_test_sce_kernel_1], %r220;\0A\09bra.uni \09$L__BB0_24;\0A$L__BB0_22:\0A\09ld.shared.b32 \09%r220, [__wg_test_sce_kernel_1];\0A$L__BB0_24:\0A\09@%p4 bra \09$L__BB0_26;\0A\09shfl.sync.bfly.b32 \09%r155|%p34, %r218, 1, %r4, %r3;\0A\09add.rn.f32 \09%r156, %r218, %r155;\0A\09selp.f32 \09%r157, %r156, %r218, %p34;\0A\09shfl.sync.bfly.b32 \09%r158|%p35, %r157, 2, %r4, %r3;\0A\09add.rn.f32 \09%r159, %r158, %r157;\0A\09selp.f32 \09%r160, %r159, %r157, %p35;\0A\09shfl.sync.bfly.b32 \09%r161|%p36, %r160, 4, %r4, %r3;\0A\09add.rn.f32 \09%r162, %r161, %r160;\0A\09selp.f32 \09%r163, %r162, %r160, %p36;\0A\09shfl.sync.bfly.b32 \09%r164|%p37, %r163, 8, %r4, %r3;\0A\09add.rn.f32 \09%r165, %r164, %r163;\0A\09selp.f32 \09%r166, %r165, %r163, %p37;\0A\09shfl.sync.bfly.b32 \09%r167|%p38, %r166, 16, %r4, %r3;\0A\09add.rn.f32 \09%r168, %r167, %r166;\0A\09selp.f32 \09%r221, %r168, %r166, %p38;\0A\09bra.uni \09$L__BB0_27;\0A$L__BB0_26:\0A\09shfl.sync.bfly.b32 \09%r146, %r218, 1, 31, -1;\0A\09add.rn.f32 \09%r147, %r218, %r146;\0A\09shfl.sync.bfly.b32 \09%r148, %r147, 2, 31, -1;\0A\09add.rn.f32 \09%r149, %r147, %r148;\0A\09shfl.sync.bfly.b32 \09%r150, %r149, 4, 31, -1;\0A\09add.rn.f32 \09%r151, %r149, %r150;\0A\09shfl.sync.bfly.b32 \09%r152, %r151, 8, 31, -1;\0A\09add.rn.f32 \09%r153, %r151, %r152;\0A\09shfl.sync.bfly.b32 \09%r154, %r153, 16, 31, -1;\0A\09add.rn.f32 \09%r221, %r153, %r154;\0A$L__BB0_27:\0A\09@%p10 bra \09$L__BB0_29;\0A\09st.shared.b32 \09[%rd9], %r221;\0A$L__BB0_29:\0A\09@%p11 bra \09$L__BB0_31;\0A\09ld.shared.b32 \09%r222, [__wg_test_sce_kernel_2];\0A\09bra.uni \09$L__BB0_32;\0A$L__BB0_33:\0A\09setp.ne.b64 \09%p50, %rd32, %rd3;\0A\09@%p50 bra \09$L__BB0_35;\0A\09ld.param.b64 \09%rd35, [test_sce_kernel_param_20];\0A\09ld.param.b32 \09%r41, [test_sce_kernel_param_18];\0A\09shl.b64 \09%rd46, %rd32, 2;\0A\09add.s64 \09%rd30, %rd35, %rd46;\0A\09div.rn.f32 \09%r38, %r223, %r41;\0A\09st.global.b32 \09[%rd30], %r38;\0A$L__BB0_35:\0A\09ret;\0A\0A}\0A">]
  llvm.func @test_sce_backward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.addressof @__constant_4xi64 : !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(3 : index) : i64
    %4 = llvm.mlir.constant(-4 : index) : i64
    %5 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(4 : index) : i64
    %9 = llvm.mlir.constant(32 : index) : i64
    %10 = llvm.mlir.constant(10 : index) : i64
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %13 = llvm.mlir.constant(256 : index) : i64
    %14 = llvm.mlir.constant(40 : index) : i64
    %15 = llvm.mlir.constant(2.500000e-01 : f32) : f32
    %16 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %17 = llvm.mlir.constant(8 : i64) : i64
    %18 = llvm.mlir.constant(4 : i64) : i64
    %19 = llvm.mlir.constant(1 : i32) : i32
    %20 = llvm.mlir.constant(0 : i64) : i64
    %21 = llvm.mlir.constant(1 : i64) : i64
    %22 = llvm.mlir.constant(10 : i64) : i64
    %23 = llvm.mlir.undef : !llvm.array<2 x i64>
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %25 = llvm.mlir.undef : !llvm.array<3 x i64>
    %26 = llvm.mlir.constant(40 : i64) : i64
    %27 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.mlir.constant(2 : i32) : i32
    %29 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %arg7, %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %arg8, %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %arg9, %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %arg10, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg11, %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %arg0, %29[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %arg1, %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %arg2, %37[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %arg3, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %arg5, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %arg4, %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %arg6, %41[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %44 = llvm.mul %17, %18 : i64
    %45 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %46 = llvm.call @cudaMallocAsync(%45, %44, %43) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %47 = llvm.load %45 : !llvm.ptr -> !llvm.ptr
    %48 = llvm.addrspacecast %47 : !llvm.ptr to !llvm.ptr<1>
    %49 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
    %50 = llvm.call @cudaMemcpyAsync(%47, %49, %44, %19, %43) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %51 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %42, %51 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @mgpuStreamSynchronize(%43) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%43) : (!llvm.ptr) -> ()
    %52 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    llvm.call @mgpuMemHostRegisterMemRef(%0, %51, %53) : (i64, !llvm.ptr, i64) -> ()
    %54 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %55 = llvm.mul %18, %18 : i64
    %56 = llvm.mul %55, %22 : i64
    %57 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %58 = llvm.call @cudaMallocAsync(%57, %56, %54) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %59 = llvm.load %57 : !llvm.ptr -> !llvm.ptr
    %60 = llvm.call @cudaMemcpyAsync(%59, %arg1, %56, %19, %54) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %61 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %35, %61 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @mgpuStreamSynchronize(%54) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%54) : (!llvm.ptr) -> ()
    %62 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    llvm.call @mgpuMemHostRegisterMemRef(%7, %61, %63) : (i64, !llvm.ptr, i64) -> ()
    %64 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %65 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %66 = llvm.call @cudaMallocAsync(%65, %55, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %67 = llvm.load %65 : !llvm.ptr -> !llvm.ptr
    %68 = llvm.call @cudaMemcpyAsync(%67, %arg8, %55, %19, %64) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %69 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %70 = llvm.call @cudaMallocAsync(%69, %55, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %71 = llvm.load %69 : !llvm.ptr -> !llvm.ptr
    %72 = llvm.addrspacecast %71 : !llvm.ptr to !llvm.ptr<1>
    %73 = llvm.call @cudaMemcpyAsync(%71, %arg8, %55, %19, %64) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %74 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %75 = llvm.call @cudaMallocAsync(%74, %56, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %76 = llvm.load %74 : !llvm.ptr -> !llvm.ptr
    %77 = llvm.addrspacecast %76 : !llvm.ptr to !llvm.ptr<1>
    %78 = llvm.mul %21, %22 : i64
    %79 = llvm.call @cudaMemcpyAsync(%76, %arg1, %56, %19, %64) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %80 = llvm.mul %55, %21 : i64
    %81 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %82 = llvm.call @cudaMallocAsync(%81, %80, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %83 = llvm.load %81 : !llvm.ptr -> !llvm.ptr
    %84 = llvm.addrspacecast %83 : !llvm.ptr to !llvm.ptr<1>
    %85 = llvm.mul %21, %21 : i64
    %86 = llvm.bitcast %6 : f32 to i32
    %87 = llvm.call @cudaMemsetAsync(%83, %86, %80, %64) : (!llvm.ptr, i32, i64, !llvm.ptr) -> i32
    gpu.launch_func <%64 : !llvm.ptr> @test_sce_backward_kernel::@test_sce_backward_kernel blocks in (%8, %7, %7) threads in (%9, %7, %7) : i64 args(%77 : !llvm.ptr<1>, %77 : !llvm.ptr<1>, %20 : i64, %18 : i64, %22 : i64, %78 : i64, %21 : i64, %10 : i64, %6 : f32, %11 : i64, %84 : !llvm.ptr<1>, %84 : !llvm.ptr<1>, %20 : i64, %18 : i64, %21 : i64, %85 : i64, %21 : i64)
    %88 = llvm.mul %80, %22 : i64
    %89 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %90 = llvm.call @cudaMallocAsync(%89, %88, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %91 = llvm.load %89 : !llvm.ptr -> !llvm.ptr
    %92 = llvm.mul %78, %21 : i64
    gpu.launch_func <%64 : !llvm.ptr> @test_sce_backward_kernel_0::@test_sce_backward_kernel_0 blocks in (%8, %10, %7) threads in (%7, %7, %7) : i64 args(%77 : !llvm.ptr<1>, %77 : !llvm.ptr<1>, %20 : i64, %18 : i64, %22 : i64, %78 : i64, %21 : i64, %84 : !llvm.ptr<1>, %84 : !llvm.ptr<1>, %20 : i64, %18 : i64, %21 : i64, %85 : i64, %21 : i64, %11 : i64, %91 : !llvm.ptr, %91 : !llvm.ptr, %20 : i64, %18 : i64, %21 : i64, %22 : i64, %92 : i64, %78 : i64, %21 : i64)
    %93 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %94 = llvm.call @cudaMallocAsync(%93, %80, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %95 = llvm.load %93 : !llvm.ptr -> !llvm.ptr
    %96 = llvm.addrspacecast %95 : !llvm.ptr to !llvm.ptr<1>
    %97 = llvm.bitcast %12 : f32 to i32
    %98 = llvm.call @cudaMemsetAsync(%95, %97, %80, %64) : (!llvm.ptr, i32, i64, !llvm.ptr) -> i32
    gpu.launch_func <%64 : !llvm.ptr> @test_sce_backward_kernel_1::@test_sce_backward_kernel_1 blocks in (%8, %7, %7) threads in (%9, %7, %7) : i64 args(%91 : !llvm.ptr, %91 : !llvm.ptr, %20 : i64, %18 : i64, %21 : i64, %22 : i64, %92 : i64, %78 : i64, %21 : i64, %11 : i64, %10 : i64, %12 : f32, %96 : !llvm.ptr<1>, %96 : !llvm.ptr<1>, %20 : i64, %18 : i64, %21 : i64, %85 : i64, %21 : i64)
    %99 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %100 = llvm.call @cudaMallocAsync(%99, %88, %64) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %101 = llvm.load %99 : !llvm.ptr -> !llvm.ptr
    %102 = llvm.insertvalue %101, %24[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.insertvalue %101, %102[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %104 = llvm.insertvalue %20, %103[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %105 = llvm.insertvalue %18, %25[0] : !llvm.array<3 x i64> 
    %106 = llvm.insertvalue %21, %105[1] : !llvm.array<3 x i64> 
    %107 = llvm.insertvalue %22, %106[2] : !llvm.array<3 x i64> 
    %108 = llvm.insertvalue %107, %104[3] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %109 = llvm.insertvalue %21, %25[2] : !llvm.array<3 x i64> 
    %110 = llvm.insertvalue %78, %109[1] : !llvm.array<3 x i64> 
    %111 = llvm.insertvalue %92, %110[0] : !llvm.array<3 x i64> 
    %112 = llvm.insertvalue %111, %108[4] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    gpu.launch_func <%64 : !llvm.ptr> @test_sce_backward_kernel_2::@test_sce_backward_kernel_2 blocks in (%8, %10, %7) threads in (%7, %7, %7) : i64 args(%91 : !llvm.ptr, %91 : !llvm.ptr, %20 : i64, %18 : i64, %21 : i64, %22 : i64, %92 : i64, %78 : i64, %21 : i64, %11 : i64, %96 : !llvm.ptr<1>, %96 : !llvm.ptr<1>, %20 : i64, %18 : i64, %21 : i64, %85 : i64, %21 : i64, %5 : f32, %101 : !llvm.ptr, %101 : !llvm.ptr, %20 : i64, %18 : i64, %21 : i64, %22 : i64, %92 : i64, %78 : i64, %21 : i64)
    %113 = llvm.extractvalue %112[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %114 = llvm.extractvalue %112[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %115 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %116 = llvm.insertvalue %113, %115[0] : !llvm.struct<(ptr, ptr, i64)> 
    %117 = llvm.insertvalue %114, %116[1] : !llvm.struct<(ptr, ptr, i64)> 
    %118 = llvm.mlir.constant(0 : index) : i64
    %119 = llvm.insertvalue %118, %117[2] : !llvm.struct<(ptr, ptr, i64)> 
    %120 = llvm.extractvalue %112[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %121 = llvm.extractvalue %112[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %122 = llvm.extractvalue %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %123 = llvm.extractvalue %112[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %124 = llvm.extractvalue %112[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %125 = llvm.extractvalue %112[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %126 = llvm.extractvalue %112[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %127 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.extractvalue %119[0] : !llvm.struct<(ptr, ptr, i64)> 
    %129 = llvm.extractvalue %119[1] : !llvm.struct<(ptr, ptr, i64)> 
    %130 = llvm.insertvalue %128, %127[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %129, %130[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(0 : index) : i64
    %133 = llvm.insertvalue %132, %131[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.constant(4 : index) : i64
    %135 = llvm.insertvalue %134, %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(10 : index) : i64
    %137 = llvm.insertvalue %136, %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mlir.constant(10 : index) : i64
    %139 = llvm.insertvalue %138, %137[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mlir.constant(1 : index) : i64
    %141 = llvm.insertvalue %140, %139[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = builtin.unrealized_conversion_cast %141 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x10xf32>
    %143 = builtin.unrealized_conversion_cast %142 : memref<4x10xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @mgpuStreamSynchronize(%64) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%64) : (!llvm.ptr) -> ()
    %144 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %146 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %147 = llvm.call @cudaMallocAsync(%146, %44, %145) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %148 = llvm.load %146 : !llvm.ptr -> !llvm.ptr
    gpu.launch_func <%145 : !llvm.ptr> @test_sce_backward_kernel_3::@test_sce_backward_kernel_3 blocks in (%8, %7, %7) threads in (%7, %7, %7) : i64 args(%72 : !llvm.ptr<1>, %72 : !llvm.ptr<1>, %20 : i64, %18 : i64, %21 : i64, %48 : !llvm.ptr<1>, %48 : !llvm.ptr<1>, %20 : i64, %18 : i64, %21 : i64, %148 : !llvm.ptr, %148 : !llvm.ptr, %20 : i64, %18 : i64, %21 : i64)
    %149 = llvm.mul %18, %26 : i64
    %150 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %151 = llvm.call @cudaMallocAsync(%150, %149, %145) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %152 = llvm.load %150 : !llvm.ptr -> !llvm.ptr
    %153 = llvm.addrspacecast %152 : !llvm.ptr to !llvm.ptr<1>
    %154 = llvm.call @cudaMemcpyAsync(%152, %144, %149, %19, %145) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    gpu.launch_func <%145 : !llvm.ptr> @test_sce_backward_kernel_4::@test_sce_backward_kernel_4 blocks in (%7, %7, %7) threads in (%13, %7, %7) : i64 args(%8 : i64, %148 : !llvm.ptr, %148 : !llvm.ptr, %20 : i64, %18 : i64, %21 : i64, %11 : i64, %14 : i64, %16 : f32, %153 : !llvm.ptr<1>, %153 : !llvm.ptr<1>, %20 : i64, %26 : i64, %21 : i64)
    llvm.call @mgpuStreamSynchronize(%145) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%145) : (!llvm.ptr) -> ()
    %155 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %156 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %157 = llvm.call @cudaMallocAsync(%156, %56, %155) : (!llvm.ptr, i64, !llvm.ptr) -> i32
    %158 = llvm.load %156 : !llvm.ptr -> !llvm.ptr
    %159 = llvm.insertvalue %158, %27[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.insertvalue %158, %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %20, %160[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %18, %23[0] : !llvm.array<2 x i64> 
    %163 = llvm.insertvalue %22, %162[1] : !llvm.array<2 x i64> 
    %164 = llvm.insertvalue %163, %161[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %21, %23[1] : !llvm.array<2 x i64> 
    %166 = llvm.insertvalue %78, %165[0] : !llvm.array<2 x i64> 
    %167 = llvm.insertvalue %166, %164[4] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    gpu.launch_func <%155 : !llvm.ptr> @test_sce_backward_kernel_5::@test_sce_backward_kernel_5 blocks in (%8, %3, %7) threads in (%8, %7, %7) : i64 args(%4 : i64, %10 : i64, %8 : i64, %153 : !llvm.ptr<1>, %153 : !llvm.ptr<1>, %11 : i64, %8 : i64, %10 : i64, %10 : i64, %7 : i64, %15 : f32, %158 : !llvm.ptr, %158 : !llvm.ptr, %20 : i64, %18 : i64, %22 : i64, %78 : i64, %21 : i64)
    %168 = llvm.call @cudaFreeAsync(%71, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %169 = llvm.call @cudaFreeAsync(%76, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %170 = llvm.call @cudaFreeAsync(%83, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %171 = llvm.call @cudaFreeAsync(%91, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %172 = llvm.call @cudaFreeAsync(%95, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %173 = llvm.call @cudaFreeAsync(%101, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %174 = llvm.call @cudaFreeAsync(%148, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %175 = llvm.call @cudaFreeAsync(%152, %155) : (!llvm.ptr, !llvm.ptr) -> i32
    %176 = llvm.call @cudaMemcpyAsync(%arg1, %59, %56, %28, %155) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    %177 = llvm.call @cudaMemcpyAsync(%arg8, %67, %55, %28, %155) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
    llvm.call @mgpuStreamSynchronize(%155) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%155) : (!llvm.ptr) -> ()
    llvm.return %167 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  gpu.binary @test_sce_backward_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 1 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel\0A// __wg_test_sce_backward_kernel_0_$_0 has been demoted\0A\0A.visible .entry test_sce_backward_kernel(\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_param_0,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_param_1,\0A\09.param .u64 test_sce_backward_kernel_param_2,\0A\09.param .u64 test_sce_backward_kernel_param_3,\0A\09.param .u64 test_sce_backward_kernel_param_4,\0A\09.param .u64 test_sce_backward_kernel_param_5,\0A\09.param .u64 test_sce_backward_kernel_param_6,\0A\09.param .u64 test_sce_backward_kernel_param_7,\0A\09.param .f32 test_sce_backward_kernel_param_8,\0A\09.param .u64 test_sce_backward_kernel_param_9,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_param_10,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_param_11,\0A\09.param .u64 test_sce_backward_kernel_param_12,\0A\09.param .u64 test_sce_backward_kernel_param_13,\0A\09.param .u64 test_sce_backward_kernel_param_14,\0A\09.param .u64 test_sce_backward_kernel_param_15,\0A\09.param .u64 test_sce_backward_kernel_param_16\0A)\0A.maxntid 32, 1, 1\0A{\0A\09.reg .pred \09%p<17>;\0A\09.reg .b32 \09%r<66>;\0A\09.reg .b64 \09%rd<22>;\0A\09// demoted variable\0A\09.shared .align 4 .f32 __wg_test_sce_backward_kernel_0_$_0;\0A\09ld.param.b32 \09%r63, [test_sce_backward_kernel_param_8];\0A\09ld.param.b64 \09%rd12, [test_sce_backward_kernel_param_7];\0A\09mov.u32 \09%r12, %tid.x;\0A\09cvt.u64.u32 \09%rd1, %r12;\0A\09mov.u32 \09%r13, %ctaid.x;\0A\09cvt.u64.u32 \09%rd2, %r13;\0A\09mov.u32 \09%r14, %ntid.x;\0A\09cvt.u64.u32 \09%rd3, %r14;\0A\09setp.le.s64 \09%p1, %rd12, %rd1;\0A\09@%p1 bra \09$L__BB0_3;\0A\09ld.param.b64 \09%rd11, [test_sce_backward_kernel_param_1];\0A\09shl.b64 \09%rd15, %rd1, 2;\0A\09mad.lo.s64 \09%rd16, %rd2, 40, %rd15;\0A\09add.s64 \09%rd20, %rd11, %rd16;\0A\09shl.b64 \09%rd5, %rd3, 2;\0A\09mov.b64 \09%rd21, %rd1;\0A$L__BB0_2:\0A\09ld.global.b32 \09%r15, [%rd20];\0A\09max.NaN.f32 \09%r63, %r63, %r15;\0A\09add.s64 \09%rd21, %rd21, %rd3;\0A\09add.s64 \09%rd20, %rd20, %rd5;\0A\09setp.lt.s64 \09%p2, %rd21, %rd12;\0A\09@%p2 bra \09$L__BB0_2;\0A$L__BB0_3:\0A\09cvt.u32.u64 \09%r16, %rd3;\0A\09setp.gt.u32 \09%p3, %r16, 31;\0A\09@%p3 bra \09$L__BB0_5;\0A\09sub.s32 \09%r27, 32, %r16;\0A\09mov.b32 \09%r28, -1;\0A\09shr.u32 \09%r29, %r28, %r27;\0A\09add.s32 \09%r30, %r16, -1;\0A\09shfl.sync.bfly.b32 \09%r31|%p4, %r63, 1, %r30, %r29;\0A\09max.NaN.f32 \09%r32, %r63, %r31;\0A\09selp.f32 \09%r33, %r32, %r63, %p4;\0A\09shfl.sync.bfly.b32 \09%r34|%p5, %r33, 2, %r30, %r29;\0A\09max.NaN.f32 \09%r35, %r33, %r34;\0A\09selp.f32 \09%r36, %r35, %r33, %p5;\0A\09shfl.sync.bfly.b32 \09%r37|%p6, %r36, 4, %r30, %r29;\0A\09max.NaN.f32 \09%r38, %r36, %r37;\0A\09selp.f32 \09%r39, %r38, %r36, %p6;\0A\09shfl.sync.bfly.b32 \09%r40|%p7, %r39, 8, %r30, %r29;\0A\09max.NaN.f32 \09%r41, %r39, %r40;\0A\09selp.f32 \09%r42, %r41, %r39, %p7;\0A\09shfl.sync.bfly.b32 \09%r43|%p8, %r42, 16, %r30, %r29;\0A\09max.NaN.f32 \09%r44, %r42, %r43;\0A\09selp.f32 \09%r64, %r44, %r42, %p8;\0A\09bra.uni \09$L__BB0_6;\0A$L__BB0_5:\0A\09shfl.sync.bfly.b32 \09%r17, %r63, 1, 31, -1;\0A\09max.NaN.f32 \09%r18, %r63, %r17;\0A\09shfl.sync.bfly.b32 \09%r19, %r18, 2, 31, -1;\0A\09max.NaN.f32 \09%r20, %r18, %r19;\0A\09shfl.sync.bfly.b32 \09%r21, %r20, 4, 31, -1;\0A\09max.NaN.f32 \09%r22, %r20, %r21;\0A\09shfl.sync.bfly.b32 \09%r23, %r22, 8, 31, -1;\0A\09max.NaN.f32 \09%r24, %r22, %r23;\0A\09shfl.sync.bfly.b32 \09%r25, %r24, 16, 31, -1;\0A\09max.NaN.f32 \09%r64, %r24, %r25;\0A$L__BB0_6:\0A\09ld.param.b64 \09%rd13, [test_sce_backward_kernel_param_9];\0A\09cvt.u32.u64 \09%r45, %rd1;\0A\09setp.ne.b32 \09%p9, %r45, 0;\0A\09@%p9 bra \09$L__BB0_8;\0A\09shfl.sync.bfly.b32 \09%r46|%p10, %r64, 1, 0, 1;\0A\09max.NaN.f32 \09%r47, %r64, %r46;\0A\09selp.f32 \09%r48, %r47, %r64, %p10;\0A\09shfl.sync.bfly.b32 \09%r49|%p11, %r48, 2, 0, 1;\0A\09max.NaN.f32 \09%r50, %r48, %r49;\0A\09selp.f32 \09%r51, %r50, %r48, %p11;\0A\09shfl.sync.bfly.b32 \09%r52|%p12, %r51, 4, 0, 1;\0A\09max.NaN.f32 \09%r53, %r51, %r52;\0A\09selp.f32 \09%r54, %r53, %r51, %p12;\0A\09shfl.sync.bfly.b32 \09%r55|%p13, %r54, 8, 0, 1;\0A\09max.NaN.f32 \09%r56, %r54, %r55;\0A\09selp.f32 \09%r57, %r56, %r54, %p13;\0A\09shfl.sync.bfly.b32 \09%r58|%p14, %r57, 16, 0, 1;\0A\09max.NaN.f32 \09%r59, %r57, %r58;\0A\09selp.f32 \09%r60, %r59, %r57, %p14;\0A\09st.shared.b32 \09[__wg_test_sce_backward_kernel_0_$_0], %r60;\0A$L__BB0_8:\0A\09setp.ne.b64 \09%p15, %rd13, %rd1;\0A\09@%p15 bra \09$L__BB0_11;\0A\09ld.param.b64 \09%rd14, [test_sce_backward_kernel_param_11];\0A\09shl.b64 \09%rd17, %rd13, 2;\0A\09add.s64 \09%rd18, %rd14, %rd17;\0A\09shl.b64 \09%rd19, %rd2, 2;\0A\09add.s64 \09%rd10, %rd18, %rd19;\0A\09ld.shared.b32 \09%r7, [__wg_test_sce_backward_kernel_0_$_0];\0A\09ld.global.b32 \09%r65, [%rd10];\0A$L__BB0_10:\0A\09max.NaN.f32 \09%r61, %r65, %r7;\0A\09atom.acq_rel.global.cas.b32 \09%r10, [%rd10], %r65, %r61;\0A\09setp.ne.b32 \09%p16, %r10, %r65;\0A\09mov.b32 \09%r65, %r10;\0A\09@%p16 bra \09$L__BB0_10;\0A$L__BB0_11:\0A\09ret;\0A\0A}\0A">]
  gpu.binary @test_sce_backward_kernel_0  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel_0\0A\0A.visible .entry test_sce_backward_kernel_0(\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_0_param_0,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_0_param_1,\0A\09.param .u64 test_sce_backward_kernel_0_param_2,\0A\09.param .u64 test_sce_backward_kernel_0_param_3,\0A\09.param .u64 test_sce_backward_kernel_0_param_4,\0A\09.param .u64 test_sce_backward_kernel_0_param_5,\0A\09.param .u64 test_sce_backward_kernel_0_param_6,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_0_param_7,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_0_param_8,\0A\09.param .u64 test_sce_backward_kernel_0_param_9,\0A\09.param .u64 test_sce_backward_kernel_0_param_10,\0A\09.param .u64 test_sce_backward_kernel_0_param_11,\0A\09.param .u64 test_sce_backward_kernel_0_param_12,\0A\09.param .u64 test_sce_backward_kernel_0_param_13,\0A\09.param .u64 test_sce_backward_kernel_0_param_14,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_0_param_15,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_0_param_16,\0A\09.param .u64 test_sce_backward_kernel_0_param_17,\0A\09.param .u64 test_sce_backward_kernel_0_param_18,\0A\09.param .u64 test_sce_backward_kernel_0_param_19,\0A\09.param .u64 test_sce_backward_kernel_0_param_20,\0A\09.param .u64 test_sce_backward_kernel_0_param_21,\0A\09.param .u64 test_sce_backward_kernel_0_param_22,\0A\09.param .u64 test_sce_backward_kernel_0_param_23\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<18>;\0A\09.reg .b64 \09%rd<18>;\0A\0A\09ld.param.b64 \09%rd1, [test_sce_backward_kernel_0_param_16];\0A\09cvta.to.global.u64 \09%rd2, %rd1;\0A\09ld.param.b64 \09%rd3, [test_sce_backward_kernel_0_param_1];\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09cvt.u64.u32 \09%rd4, %r1;\0A\09mov.u32 \09%r2, %ctaid.y;\0A\09mul.wide.u32 \09%rd5, %r1, 40;\0A\09add.s64 \09%rd6, %rd3, %rd5;\0A\09mul.wide.u32 \09%rd7, %r2, 4;\0A\09add.s64 \09%rd8, %rd6, %rd7;\0A\09ld.param.b64 \09%rd9, [test_sce_backward_kernel_0_param_8];\0A\09ld.global.b32 \09%r3, [%rd8];\0A\09ld.param.b64 \09%rd10, [test_sce_backward_kernel_0_param_14];\0A\09shl.b64 \09%rd11, %rd10, 2;\0A\09add.s64 \09%rd12, %rd9, %rd11;\0A\09mul.wide.u32 \09%rd13, %r1, 4;\0A\09add.s64 \09%rd14, %rd12, %rd13;\0A\09ld.global.b32 \09%r4, [%rd14];\0A\09sub.rn.f32 \09%r5, %r3, %r4;\0A\09fma.rn.f32 \09%r6, %r5, 0f3BBB989D, 0f3F000000;\0A\09cvt.sat.f32.f32 \09%r7, %r6;\0A\09mov.b32 \09%r8, 0f4B400001;\0A\09mov.b32 \09%r9, 0f437C0000;\0A\09fma.rm.f32 \09%r10, %r7, %r9, %r8;\0A\09add.rn.f32 \09%r11, %r10, 0fCB40007F;\0A\09neg.f32 \09%r12, %r11;\0A\09fma.rn.f32 \09%r13, %r5, 0f3FB8AA3B, %r12;\0A\09fma.rn.f32 \09%r14, %r5, 0f32A57060, %r13;\0A\09shl.b32 \09%r15, %r10, 23;\0A\09ex2.approx.ftz.f32 \09%r16, %r14;\0A\09mul.rn.f32 \09%r17, %r16, %r15;\0A\09add.s64 \09%rd15, %rd10, %rd4;\0A\09mad.lo.s64 \09%rd16, %rd15, 40, %rd2;\0A\09add.s64 \09%rd17, %rd16, %rd7;\0A\09st.global.b32 \09[%rd17], %r17;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @test_sce_backward_kernel_1  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 1 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel_1\0A// __wg_test_sce_backward_kernel_1_0_$_0 has been demoted\0A\0A.visible .entry test_sce_backward_kernel_1(\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_1_param_0,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_1_param_1,\0A\09.param .u64 test_sce_backward_kernel_1_param_2,\0A\09.param .u64 test_sce_backward_kernel_1_param_3,\0A\09.param .u64 test_sce_backward_kernel_1_param_4,\0A\09.param .u64 test_sce_backward_kernel_1_param_5,\0A\09.param .u64 test_sce_backward_kernel_1_param_6,\0A\09.param .u64 test_sce_backward_kernel_1_param_7,\0A\09.param .u64 test_sce_backward_kernel_1_param_8,\0A\09.param .u64 test_sce_backward_kernel_1_param_9,\0A\09.param .u64 test_sce_backward_kernel_1_param_10,\0A\09.param .f32 test_sce_backward_kernel_1_param_11,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_1_param_12,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_1_param_13,\0A\09.param .u64 test_sce_backward_kernel_1_param_14,\0A\09.param .u64 test_sce_backward_kernel_1_param_15,\0A\09.param .u64 test_sce_backward_kernel_1_param_16,\0A\09.param .u64 test_sce_backward_kernel_1_param_17,\0A\09.param .u64 test_sce_backward_kernel_1_param_18\0A)\0A.maxntid 32, 1, 1\0A{\0A\09.reg .pred \09%p<16>;\0A\09.reg .b32 \09%r<62>;\0A\09.reg .b64 \09%rd<24>;\0A\09// demoted variable\0A\09.shared .align 4 .f32 __wg_test_sce_backward_kernel_1_0_$_0;\0A\09ld.param.b32 \09%r60, [test_sce_backward_kernel_1_param_11];\0A\09ld.param.b64 \09%rd13, [test_sce_backward_kernel_1_param_10];\0A\09ld.param.b64 \09%rd12, [test_sce_backward_kernel_1_param_9];\0A\09ld.param.b64 \09%rd15, [test_sce_backward_kernel_1_param_1];\0A\09cvta.to.global.u64 \09%rd1, %rd15;\0A\09mov.u32 \09%r8, %tid.x;\0A\09cvt.u64.u32 \09%rd2, %r8;\0A\09mov.u32 \09%r9, %ctaid.x;\0A\09cvt.u64.u32 \09%rd3, %r9;\0A\09mov.u32 \09%r10, %ntid.x;\0A\09cvt.u64.u32 \09%rd4, %r10;\0A\09setp.le.s64 \09%p1, %rd13, %rd2;\0A\09@%p1 bra \09$L__BB0_3;\0A\09add.s64 \09%rd16, %rd12, %rd3;\0A\09shl.b64 \09%rd17, %rd2, 2;\0A\09mad.lo.s64 \09%rd18, %rd16, 40, %rd17;\0A\09add.s64 \09%rd22, %rd1, %rd18;\0A\09shl.b64 \09%rd6, %rd4, 2;\0A\09mov.b64 \09%rd23, %rd2;\0A$L__BB0_2:\0A\09ld.global.b32 \09%r11, [%rd22];\0A\09add.rn.f32 \09%r60, %r60, %r11;\0A\09add.s64 \09%rd23, %rd23, %rd4;\0A\09add.s64 \09%rd22, %rd22, %rd6;\0A\09setp.lt.s64 \09%p2, %rd23, %rd13;\0A\09@%p2 bra \09$L__BB0_2;\0A$L__BB0_3:\0A\09cvt.u32.u64 \09%r12, %rd4;\0A\09setp.gt.u32 \09%p3, %r12, 31;\0A\09@%p3 bra \09$L__BB0_5;\0A\09sub.s32 \09%r23, 32, %r12;\0A\09mov.b32 \09%r24, -1;\0A\09shr.u32 \09%r25, %r24, %r23;\0A\09add.s32 \09%r26, %r12, -1;\0A\09shfl.sync.bfly.b32 \09%r27|%p4, %r60, 1, %r26, %r25;\0A\09add.rn.f32 \09%r28, %r60, %r27;\0A\09selp.f32 \09%r29, %r28, %r60, %p4;\0A\09shfl.sync.bfly.b32 \09%r30|%p5, %r29, 2, %r26, %r25;\0A\09add.rn.f32 \09%r31, %r30, %r29;\0A\09selp.f32 \09%r32, %r31, %r29, %p5;\0A\09shfl.sync.bfly.b32 \09%r33|%p6, %r32, 4, %r26, %r25;\0A\09add.rn.f32 \09%r34, %r33, %r32;\0A\09selp.f32 \09%r35, %r34, %r32, %p6;\0A\09shfl.sync.bfly.b32 \09%r36|%p7, %r35, 8, %r26, %r25;\0A\09add.rn.f32 \09%r37, %r36, %r35;\0A\09selp.f32 \09%r38, %r37, %r35, %p7;\0A\09shfl.sync.bfly.b32 \09%r39|%p8, %r38, 16, %r26, %r25;\0A\09add.rn.f32 \09%r40, %r39, %r38;\0A\09selp.f32 \09%r61, %r40, %r38, %p8;\0A\09bra.uni \09$L__BB0_6;\0A$L__BB0_5:\0A\09shfl.sync.bfly.b32 \09%r13, %r60, 1, 31, -1;\0A\09add.rn.f32 \09%r14, %r60, %r13;\0A\09shfl.sync.bfly.b32 \09%r15, %r14, 2, 31, -1;\0A\09add.rn.f32 \09%r16, %r14, %r15;\0A\09shfl.sync.bfly.b32 \09%r17, %r16, 4, 31, -1;\0A\09add.rn.f32 \09%r18, %r16, %r17;\0A\09shfl.sync.bfly.b32 \09%r19, %r18, 8, 31, -1;\0A\09add.rn.f32 \09%r20, %r18, %r19;\0A\09shfl.sync.bfly.b32 \09%r21, %r20, 16, 31, -1;\0A\09add.rn.f32 \09%r61, %r20, %r21;\0A$L__BB0_6:\0A\09cvt.u32.u64 \09%r41, %rd2;\0A\09setp.ne.b32 \09%p9, %r41, 0;\0A\09@%p9 bra \09$L__BB0_8;\0A\09shfl.sync.bfly.b32 \09%r42|%p10, %r61, 1, 0, 1;\0A\09add.rn.f32 \09%r43, %r61, %r42;\0A\09selp.f32 \09%r44, %r43, %r61, %p10;\0A\09shfl.sync.bfly.b32 \09%r45|%p11, %r44, 2, 0, 1;\0A\09add.rn.f32 \09%r46, %r45, %r44;\0A\09selp.f32 \09%r47, %r46, %r44, %p11;\0A\09shfl.sync.bfly.b32 \09%r48|%p12, %r47, 4, 0, 1;\0A\09add.rn.f32 \09%r49, %r48, %r47;\0A\09selp.f32 \09%r50, %r49, %r47, %p12;\0A\09shfl.sync.bfly.b32 \09%r51|%p13, %r50, 8, 0, 1;\0A\09add.rn.f32 \09%r52, %r51, %r50;\0A\09selp.f32 \09%r53, %r52, %r50, %p13;\0A\09shfl.sync.bfly.b32 \09%r54|%p14, %r53, 16, 0, 1;\0A\09add.rn.f32 \09%r55, %r54, %r53;\0A\09selp.f32 \09%r56, %r55, %r53, %p14;\0A\09st.shared.b32 \09[__wg_test_sce_backward_kernel_1_0_$_0], %r56;\0A$L__BB0_8:\0A\09setp.ne.b64 \09%p15, %rd12, %rd2;\0A\09@%p15 bra \09$L__BB0_10;\0A\09ld.param.b64 \09%rd14, [test_sce_backward_kernel_1_param_13];\0A\09shl.b64 \09%rd19, %rd12, 2;\0A\09add.s64 \09%rd20, %rd14, %rd19;\0A\09shl.b64 \09%rd21, %rd3, 2;\0A\09add.s64 \09%rd11, %rd20, %rd21;\0A\09ld.shared.b32 \09%r57, [__wg_test_sce_backward_kernel_1_0_$_0];\0A\09atom.global.add.f32 \09%r58, [%rd11], %r57;\0A$L__BB0_10:\0A\09ret;\0A\0A}\0A">]
  gpu.binary @test_sce_backward_kernel_2  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel_2\0A\0A.visible .entry test_sce_backward_kernel_2(\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_2_param_0,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_2_param_1,\0A\09.param .u64 test_sce_backward_kernel_2_param_2,\0A\09.param .u64 test_sce_backward_kernel_2_param_3,\0A\09.param .u64 test_sce_backward_kernel_2_param_4,\0A\09.param .u64 test_sce_backward_kernel_2_param_5,\0A\09.param .u64 test_sce_backward_kernel_2_param_6,\0A\09.param .u64 test_sce_backward_kernel_2_param_7,\0A\09.param .u64 test_sce_backward_kernel_2_param_8,\0A\09.param .u64 test_sce_backward_kernel_2_param_9,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_2_param_10,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_2_param_11,\0A\09.param .u64 test_sce_backward_kernel_2_param_12,\0A\09.param .u64 test_sce_backward_kernel_2_param_13,\0A\09.param .u64 test_sce_backward_kernel_2_param_14,\0A\09.param .u64 test_sce_backward_kernel_2_param_15,\0A\09.param .u64 test_sce_backward_kernel_2_param_16,\0A\09.param .f32 test_sce_backward_kernel_2_param_17,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_2_param_18,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_2_param_19,\0A\09.param .u64 test_sce_backward_kernel_2_param_20,\0A\09.param .u64 test_sce_backward_kernel_2_param_21,\0A\09.param .u64 test_sce_backward_kernel_2_param_22,\0A\09.param .u64 test_sce_backward_kernel_2_param_23,\0A\09.param .u64 test_sce_backward_kernel_2_param_24,\0A\09.param .u64 test_sce_backward_kernel_2_param_25,\0A\09.param .u64 test_sce_backward_kernel_2_param_26\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<8>;\0A\09.reg .b64 \09%rd<18>;\0A\0A\09ld.param.b64 \09%rd1, [test_sce_backward_kernel_2_param_1];\0A\09cvta.to.global.u64 \09%rd2, %rd1;\0A\09ld.param.b64 \09%rd3, [test_sce_backward_kernel_2_param_19];\0A\09cvta.to.global.u64 \09%rd4, %rd3;\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09cvt.u64.u32 \09%rd5, %r1;\0A\09mov.u32 \09%r2, %ctaid.y;\0A\09cvt.u64.u32 \09%rd6, %r2;\0A\09ld.param.b64 \09%rd7, [test_sce_backward_kernel_2_param_9];\0A\09add.s64 \09%rd8, %rd7, %rd5;\0A\09mad.lo.s64 \09%rd9, %rd8, 10, %rd6;\0A\09shl.b64 \09%rd10, %rd9, 2;\0A\09add.s64 \09%rd11, %rd2, %rd10;\0A\09ld.global.b32 \09%r3, [%rd11];\0A\09ld.param.b64 \09%rd12, [test_sce_backward_kernel_2_param_11];\0A\09shl.b64 \09%rd13, %rd7, 2;\0A\09add.s64 \09%rd14, %rd12, %rd13;\0A\09mul.wide.u32 \09%rd15, %r1, 4;\0A\09add.s64 \09%rd16, %rd14, %rd15;\0A\09ld.global.b32 \09%r4, [%rd16];\0A\09ld.param.b32 \09%r5, [test_sce_backward_kernel_2_param_17];\0A\09div.rn.f32 \09%r6, %r5, %r4;\0A\09mul.rn.f32 \09%r7, %r3, %r6;\0A\09add.s64 \09%rd17, %rd4, %rd10;\0A\09st.global.b32 \09[%rd17], %r7;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @test_sce_backward_kernel_3  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel_3\0A\0A.visible .entry test_sce_backward_kernel_3(\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_3_param_0,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_3_param_1,\0A\09.param .u64 test_sce_backward_kernel_3_param_2,\0A\09.param .u64 test_sce_backward_kernel_3_param_3,\0A\09.param .u64 test_sce_backward_kernel_3_param_4,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_3_param_5,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_3_param_6,\0A\09.param .u64 test_sce_backward_kernel_3_param_7,\0A\09.param .u64 test_sce_backward_kernel_3_param_8,\0A\09.param .u64 test_sce_backward_kernel_3_param_9,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_3_param_10,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_3_param_11,\0A\09.param .u64 test_sce_backward_kernel_3_param_12,\0A\09.param .u64 test_sce_backward_kernel_3_param_13,\0A\09.param .u64 test_sce_backward_kernel_3_param_14\0A)\0A.maxntid 1, 1, 1\0A{\0A\09.reg .b32 \09%r<2>;\0A\09.reg .b64 \09%rd<17>;\0A\0A\09ld.param.b64 \09%rd1, [test_sce_backward_kernel_3_param_11];\0A\09cvta.to.global.u64 \09%rd2, %rd1;\0A\09ld.param.b64 \09%rd3, [test_sce_backward_kernel_3_param_1];\0A\09mov.u32 \09%r1, %ctaid.x;\0A\09mul.wide.u32 \09%rd4, %r1, 4;\0A\09add.s64 \09%rd5, %rd3, %rd4;\0A\09ld.global.s32 \09%rd6, [%rd5];\0A\09mul.wide.u32 \09%rd7, %r1, 8;\0A\09ld.param.b64 \09%rd8, [test_sce_backward_kernel_3_param_6];\0A\09add.s64 \09%rd9, %rd8, %rd7;\0A\09ld.global.b32 \09%rd10, [%rd9];\0A\09ld.global.b32 \09%rd11, [%rd9+4];\0A\09shl.b64 \09%rd12, %rd11, 32;\0A\09or.b64 \09%rd13, %rd12, %rd10;\0A\09add.s64 \09%rd14, %rd13, %rd6;\0A\09add.s64 \09%rd15, %rd2, %rd7;\0A\09st.global.b32 \09[%rd15], %rd14;\0A\09shr.u64 \09%rd16, %rd14, 32;\0A\09st.global.b32 \09[%rd15+4], %rd16;\0A\09ret;\0A\0A}\0A">]
  gpu.binary @test_sce_backward_kernel_4  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel_4\0A\0A.visible .entry test_sce_backward_kernel_4(\0A\09.param .u64 test_sce_backward_kernel_4_param_0,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_4_param_1,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_4_param_2,\0A\09.param .u64 test_sce_backward_kernel_4_param_3,\0A\09.param .u64 test_sce_backward_kernel_4_param_4,\0A\09.param .u64 test_sce_backward_kernel_4_param_5,\0A\09.param .u64 test_sce_backward_kernel_4_param_6,\0A\09.param .u64 test_sce_backward_kernel_4_param_7,\0A\09.param .f32 test_sce_backward_kernel_4_param_8,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_4_param_9,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_4_param_10,\0A\09.param .u64 test_sce_backward_kernel_4_param_11,\0A\09.param .u64 test_sce_backward_kernel_4_param_12,\0A\09.param .u64 test_sce_backward_kernel_4_param_13\0A)\0A.maxntid 256, 1, 1\0A{\0A\09.reg .pred \09%p<5>;\0A\09.reg .b32 \09%r<4>;\0A\09.reg .b64 \09%rd<16>;\0A\0A\09ld.param.b64 \09%rd7, [test_sce_backward_kernel_4_param_0];\0A\09ld.param.b64 \09%rd8, [test_sce_backward_kernel_4_param_2];\0A\09cvta.to.global.u64 \09%rd1, %rd8;\0A\09mov.u32 \09%r2, %tid.x;\0A\09cvt.u64.u32 \09%rd2, %r2;\0A\09setp.le.u64 \09%p1, %rd7, %rd2;\0A\09@%p1 bra \09$L__BB0_3;\0A\09ld.param.b64 \09%rd5, [test_sce_backward_kernel_4_param_7];\0A\09ld.param.b64 \09%rd4, [test_sce_backward_kernel_4_param_6];\0A\09shl.b64 \09%rd9, %rd2, 3;\0A\09add.s64 \09%rd10, %rd1, %rd9;\0A\09ld.global.b32 \09%rd11, [%rd10];\0A\09ld.global.b32 \09%rd12, [%rd10+4];\0A\09shl.b64 \09%rd13, %rd12, 32;\0A\09or.b64 \09%rd14, %rd13, %rd11;\0A\09setp.lt.s64 \09%p2, %rd14, %rd4;\0A\09setp.ge.s64 \09%p3, %rd14, %rd5;\0A\09or.pred \09%p4, %p2, %p3;\0A\09@%p4 bra \09$L__BB0_3;\0A\09ld.param.b64 \09%rd6, [test_sce_backward_kernel_4_param_10];\0A\09ld.param.b32 \09%r1, [test_sce_backward_kernel_4_param_8];\0A\09shl.b64 \09%rd15, %rd14, 2;\0A\09add.s64 \09%rd3, %rd6, %rd15;\0A\09atom.global.add.f32 \09%r3, [%rd3], %r1;\0A$L__BB0_3:\0A\09ret;\0A\0A}\0A">]
  gpu.binary @test_sce_backward_kernel_5  [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", flags = {fast, ftz}>, properties = {LLVMIRToISATimeInMs = 0 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 7.1\0A.target sm_86\0A.address_size 64\0A\0A\09// .globl\09test_sce_backward_kernel_5\0A\0A.visible .entry test_sce_backward_kernel_5(\0A\09.param .u64 test_sce_backward_kernel_5_param_0,\0A\09.param .u64 test_sce_backward_kernel_5_param_1,\0A\09.param .u64 test_sce_backward_kernel_5_param_2,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_5_param_3,\0A\09.param .u64 .ptr .global .align 1 test_sce_backward_kernel_5_param_4,\0A\09.param .u64 test_sce_backward_kernel_5_param_5,\0A\09.param .u64 test_sce_backward_kernel_5_param_6,\0A\09.param .u64 test_sce_backward_kernel_5_param_7,\0A\09.param .u64 test_sce_backward_kernel_5_param_8,\0A\09.param .u64 test_sce_backward_kernel_5_param_9,\0A\09.param .f32 test_sce_backward_kernel_5_param_10,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_5_param_11,\0A\09.param .u64 .ptr .align 1 test_sce_backward_kernel_5_param_12,\0A\09.param .u64 test_sce_backward_kernel_5_param_13,\0A\09.param .u64 test_sce_backward_kernel_5_param_14,\0A\09.param .u64 test_sce_backward_kernel_5_param_15,\0A\09.param .u64 test_sce_backward_kernel_5_param_16,\0A\09.param .u64 test_sce_backward_kernel_5_param_17\0A)\0A.maxntid 4, 1, 1\0A{\0A\09.reg .pred \09%p<2>;\0A\09.reg .b32 \09%r<8>;\0A\09.reg .b64 \09%rd<16>;\0A\0A\09ld.param.b64 \09%rd3, [test_sce_backward_kernel_5_param_2];\0A\09ld.param.b64 \09%rd5, [test_sce_backward_kernel_5_param_0];\0A\09ld.param.b64 \09%rd6, [test_sce_backward_kernel_5_param_12];\0A\09cvta.to.global.u64 \09%rd1, %rd6;\0A\09ld.param.b64 \09%rd7, [test_sce_backward_kernel_5_param_1];\0A\09mov.u32 \09%r3, %ctaid.y;\0A\09cvt.u64.u32 \09%rd2, %r3;\0A\09mov.u32 \09%r1, %tid.x;\0A\09cvt.u64.u32 \09%rd8, %r1;\0A\09mad.lo.s64 \09%rd9, %rd5, %rd2, %rd7;\0A\09min.s64 \09%rd10, %rd9, %rd3;\0A\09setp.le.s64 \09%p1, %rd10, %rd8;\0A\09@%p1 bra \09$L__BB0_2;\0A\09ld.param.b32 \09%r2, [test_sce_backward_kernel_5_param_10];\0A\09ld.param.b64 \09%rd4, [test_sce_backward_kernel_5_param_4];\0A\09mov.u32 \09%r4, %ctaid.x;\0A\09mad.lo.s32 \09%r5, %r4, 10, %r1;\0A\09cvt.u64.u32 \09%rd11, %r5;\0A\09mad.lo.s64 \09%rd12, %rd3, %rd2, %rd11;\0A\09shl.b64 \09%rd13, %rd12, 2;\0A\09add.s64 \09%rd14, %rd4, %rd13;\0A\09ld.global.b32 \09%r6, [%rd14];\0A\09mul.rn.f32 \09%r7, %r2, %r6;\0A\09add.s64 \09%rd15, %rd1, %rd13;\0A\09st.global.b32 \09[%rd15], %r7;\0A$L__BB0_2:\0A\09ret;\0A\0A}\0A">]
  llvm.func @mgpuMemHostRegisterMemRef(i64, !llvm.ptr, i64)
  llvm.func @mgpuStreamCreate() -> !llvm.ptr
  llvm.func @mgpuStreamSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamDestroy(!llvm.ptr)
}


[isGpuAllocPtr] Checking value: %149 = llvm.alloca %20 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %143 = llvm.alloca %20 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %139 = llvm.alloca %20 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %99 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %93 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %89 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %81 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %74 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %69 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %65 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %61 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %57 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %51 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %45 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %73 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %69 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %64 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %60 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %55 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %51 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %47 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %43 = llvm.alloca %9 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %39 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %33 = llvm.alloca %9 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %111 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %105 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %101 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %97 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %91 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %87 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %79 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %72 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %67 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %63 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %59 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %55 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %49 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %43 = llvm.alloca %19 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %73 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %69 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %64 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %60 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %55 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %51 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %47 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %43 = llvm.alloca %9 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %39 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
[isGpuAllocPtr] Checking value: %33 = llvm.alloca %9 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
[isGpuAllocPtr] Defining op: llvm.alloca
[isGpuAllocPtr] Return false.
