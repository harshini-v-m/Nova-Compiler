module attributes {gpu.container_module} {
  func.func @test_layernorm(%arg0: memref<2x3xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>) -> memref<2x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %cst_0 = arith.constant 3.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cast = memref.cast %arg0 : memref<2x3xf32> to memref<*xf32>
    gpu.host_register %cast : memref<*xf32>
    %memref = gpu.alloc  () : memref<2x3xf32, 1>
    gpu.memcpy  %memref, %arg0 : memref<2x3xf32, 1>, memref<2x3xf32>
    %cast_2 = memref.cast %arg1 : memref<3xf32> to memref<*xf32>
    gpu.host_register %cast_2 : memref<*xf32>
    %memref_3 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memcpy  %memref_3, %arg1 : memref<3xf32, 1>, memref<3xf32>
    %cast_4 = memref.cast %arg2 : memref<3xf32> to memref<*xf32>
    gpu.host_register %cast_4 : memref<*xf32>
    %memref_5 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memcpy  %memref_5, %arg2 : memref<3xf32, 1>, memref<3xf32>
    %memref_6 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memcpy  %memref_6, %arg2 : memref<3xf32, 1>, memref<3xf32>
    %memref_7 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memcpy  %memref_7, %arg1 : memref<3xf32, 1>, memref<3xf32>
    %memref_8 = gpu.alloc  () : memref<2x3xf32, 1>
    gpu.memcpy  %memref_8, %arg0 : memref<2x3xf32, 1>, memref<2x3xf32>
    %memref_9 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memset  %memref_9, %cst : memref<2x1xf32, 1>, f32
    gpu.launch_func  @test_layernorm_kernel::@test_layernorm_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_8 : memref<2x3xf32, 1>, %c3 : index, %cst : f32, %cst_0 : f32, %c0 : index, %memref_9 : memref<2x1xf32, 1>)
    %memref_10 = gpu.alloc  () : memref<2x3xf32>
    gpu.launch_func  @test_layernorm_kernel_0::@test_layernorm_kernel_0 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_8 : memref<2x3xf32, 1>, %memref_10 : memref<2x3xf32>)
    %memref_11 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memset  %memref_11, %cst : memref<2x1xf32, 1>, f32
    gpu.launch_func  @test_layernorm_kernel_1::@test_layernorm_kernel_1 blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_10 : memref<2x3xf32>, %c3 : index, %cst : f32, %cst_0 : f32, %c0 : index, %memref_11 : memref<2x1xf32, 1>)
    gpu.dealloc  %memref_10 : memref<2x3xf32>
    %memref_12 = gpu.alloc  () : memref<2x1x3xf32>
    gpu.launch_func  @test_layernorm_kernel_2::@test_layernorm_kernel_2 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_8 : memref<2x3xf32, 1>, %memref_9 : memref<2x1xf32, 1>, %c0 : index, %memref_11 : memref<2x1xf32, 1>, %memref_7 : memref<3xf32, 1>, %memref_6 : memref<3xf32, 1>, %cst_1 : f32, %memref_12 : memref<2x1x3xf32>)
    gpu.dealloc  %memref_6 : memref<3xf32, 1>
    gpu.dealloc  %memref_7 : memref<3xf32, 1>
    gpu.dealloc  %memref_8 : memref<2x3xf32, 1>
    gpu.dealloc  %memref_9 : memref<2x1xf32, 1>
    gpu.dealloc  %memref_11 : memref<2x1xf32, 1>
    %collapse_shape = memref.collapse_shape %memref_12 [[0, 1], [2]] : memref<2x1x3xf32> into memref<2x3xf32>
    gpu.memcpy  %arg0, %memref : memref<2x3xf32>, memref<2x3xf32, 1>
    gpu.memcpy  %arg1, %memref_3 : memref<3xf32>, memref<3xf32, 1>
    gpu.memcpy  %arg2, %memref_5 : memref<3xf32>, memref<3xf32, 1>
    return %collapse_shape : memref<2x3xf32>
  }
  gpu.module @test_layernorm_kernel {
    gpu.func @test_layernorm_kernel(%arg0: memref<2x3xf32, 1>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: index, %arg5: memref<2x1xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg6 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg7 = %arg2) -> (f32) {
        %4 = memref.load %arg0[%block_id_x_1, %arg6] : memref<2x3xf32, 1>
        %5 = arith.addf %arg7, %4 : f32
        scf.yield %5 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.divf %1, %arg3 : f32
      %3 = arith.cmpi eq, %thread_id_x_0, %arg4 : index
      scf.if %3 {
        %4 = memref.atomic_rmw addf %2, %arg5[%block_id_x_1, %arg4] : (f32, memref<2x1xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_kernel_0 {
    gpu.func @test_layernorm_kernel_0(%arg0: memref<2x3xf32, 1>, %arg1: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32, 1>
      %1 = arith.mulf %0, %0 : f32
      memref.store %1, %arg1[%block_id_x, %block_id_y] : memref<2x3xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_kernel_1 {
    gpu.func @test_layernorm_kernel_1(%arg0: memref<2x3xf32>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: index, %arg5: memref<2x1xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg6 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg7 = %arg2) -> (f32) {
        %4 = memref.load %arg0[%block_id_x_1, %arg6] : memref<2x3xf32>
        %5 = arith.addf %arg7, %4 : f32
        scf.yield %5 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.divf %1, %arg3 : f32
      %3 = arith.cmpi eq, %thread_id_x_0, %arg4 : index
      scf.if %3 {
        %4 = memref.atomic_rmw addf %2, %arg5[%block_id_x_1, %arg4] : (f32, memref<2x1xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_kernel_2 {
    gpu.func @test_layernorm_kernel_2(%arg0: memref<2x3xf32, 1>, %arg1: memref<2x1xf32, 1>, %arg2: index, %arg3: memref<2x1xf32, 1>, %arg4: memref<3xf32, 1>, %arg5: memref<3xf32, 1>, %arg6: f32, %arg7: memref<2x1x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32, 1>
      %1 = memref.load %arg1[%block_id_x, %arg2] : memref<2x1xf32, 1>
      %2 = memref.load %arg3[%block_id_x, %arg2] : memref<2x1xf32, 1>
      %3 = memref.load %arg4[%block_id_y] : memref<3xf32, 1>
      %4 = memref.load %arg5[%block_id_y] : memref<3xf32, 1>
      %5 = arith.mulf %1, %1 : f32
      %6 = arith.subf %2, %5 : f32
      %7 = arith.addf %6, %arg6 : f32
      %8 = math.rsqrt %7 : f32
      %9 = arith.subf %0, %1 : f32
      %10 = arith.mulf %9, %8 : f32
      %11 = arith.mulf %10, %3 : f32
      %12 = arith.addf %11, %4 : f32
      memref.store %12, %arg7[%block_id_x, %arg2, %block_id_y] : memref<2x1x3xf32>
      gpu.return
    }
  }
  func.func @test_layernorm_backward(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>, %arg2: memref<2x1xf32>, %arg3: memref<2x1xf32>, %arg4: memref<3xf32>) -> (memref<2x3xf32>, memref<3xf32>, memref<3xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %cst_0 = arith.constant 3.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cast = memref.cast %arg0 : memref<2x3xf32> to memref<*xf32>
    gpu.host_register %cast : memref<*xf32>
    %memref = gpu.alloc  () : memref<2x3xf32, 1>
    gpu.memcpy  %memref, %arg0 : memref<2x3xf32, 1>, memref<2x3xf32>
    %cast_2 = memref.cast %arg1 : memref<2x3xf32> to memref<*xf32>
    gpu.host_register %cast_2 : memref<*xf32>
    %memref_3 = gpu.alloc  () : memref<2x3xf32, 1>
    gpu.memcpy  %memref_3, %arg1 : memref<2x3xf32, 1>, memref<2x3xf32>
    %cast_4 = memref.cast %arg2 : memref<2x1xf32> to memref<*xf32>
    gpu.host_register %cast_4 : memref<*xf32>
    %memref_5 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memcpy  %memref_5, %arg2 : memref<2x1xf32, 1>, memref<2x1xf32>
    %cast_6 = memref.cast %arg3 : memref<2x1xf32> to memref<*xf32>
    gpu.host_register %cast_6 : memref<*xf32>
    %memref_7 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memcpy  %memref_7, %arg3 : memref<2x1xf32, 1>, memref<2x1xf32>
    %cast_8 = memref.cast %arg4 : memref<3xf32> to memref<*xf32>
    gpu.host_register %cast_8 : memref<*xf32>
    %memref_9 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memcpy  %memref_9, %arg4 : memref<3xf32, 1>, memref<3xf32>
    %memref_10 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memcpy  %memref_10, %arg4 : memref<3xf32, 1>, memref<3xf32>
    %memref_11 = gpu.alloc  () : memref<2x3xf32, 1>
    gpu.memcpy  %memref_11, %arg1 : memref<2x3xf32, 1>, memref<2x3xf32>
    %memref_12 = gpu.alloc  () : memref<2x3xf32, 1>
    gpu.memcpy  %memref_12, %arg0 : memref<2x3xf32, 1>, memref<2x3xf32>
    %memref_13 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memset  %memref_13, %cst : memref<2x1xf32, 1>, f32
    gpu.launch_func  @test_layernorm_backward_kernel::@test_layernorm_backward_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_11 : memref<2x3xf32, 1>, %c3 : index, %cst : f32, %cst_0 : f32, %c0 : index, %memref_13 : memref<2x1xf32, 1>)
    %memref_14 = gpu.alloc  () : memref<2x3xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_3::@test_layernorm_backward_kernel_3 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_11 : memref<2x3xf32, 1>, %memref_14 : memref<2x3xf32>)
    %memref_15 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memset  %memref_15, %cst : memref<2x1xf32, 1>, f32
    gpu.launch_func  @test_layernorm_backward_kernel_4::@test_layernorm_backward_kernel_4 blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_14 : memref<2x3xf32>, %c3 : index, %cst : f32, %cst_0 : f32, %c0 : index, %memref_15 : memref<2x1xf32, 1>)
    gpu.dealloc  %memref_14 : memref<2x3xf32>
    %memref_16 = gpu.alloc  () : memref<2x1xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_5::@test_layernorm_backward_kernel_5 blocks in (%c2, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref_15 : memref<2x1xf32, 1>, %c0 : index, %memref_13 : memref<2x1xf32, 1>, %cst_1 : f32, %memref_16 : memref<2x1xf32>)
    gpu.dealloc  %memref_15 : memref<2x1xf32, 1>
    %memref_17 = gpu.alloc  () : memref<2x1x3xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_6::@test_layernorm_backward_kernel_6 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_11 : memref<2x3xf32, 1>, %memref_13 : memref<2x1xf32, 1>, %c0 : index, %memref_16 : memref<2x1xf32>, %memref_17 : memref<2x1x3xf32>)
    gpu.dealloc  %memref_11 : memref<2x3xf32, 1>
    gpu.dealloc  %memref_13 : memref<2x1xf32, 1>
    %memref_18 = gpu.alloc  () : memref<2x3xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_7::@test_layernorm_backward_kernel_7 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_12 : memref<2x3xf32, 1>, %memref_17 : memref<2x1x3xf32>, %c0 : index, %memref_18 : memref<2x3xf32>)
    %memref_19 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memset  %memref_19, %cst : memref<3xf32, 1>, f32
    gpu.launch_func  @test_layernorm_backward_kernel_8::@test_layernorm_backward_kernel_8 blocks in (%c3, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_18 : memref<2x3xf32>, %c2 : index, %cst : f32, %c0 : index, %memref_19 : memref<3xf32, 1>)
    gpu.dealloc  %memref_18 : memref<2x3xf32>
    %memref_20 = gpu.alloc  () : memref<3xf32, 1>
    gpu.memset  %memref_20, %cst : memref<3xf32, 1>, f32
    gpu.launch_func  @test_layernorm_backward_kernel_9::@test_layernorm_backward_kernel_9 blocks in (%c3, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_12 : memref<2x3xf32, 1>, %c2 : index, %cst : f32, %c0 : index, %memref_20 : memref<3xf32, 1>)
    %memref_21 = gpu.alloc  () : memref<2x3xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_10::@test_layernorm_backward_kernel_10 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_12 : memref<2x3xf32, 1>, %memref_10 : memref<3xf32, 1>, %memref_21 : memref<2x3xf32>)
    gpu.dealloc  %memref_10 : memref<3xf32, 1>
    gpu.dealloc  %memref_12 : memref<2x3xf32, 1>
    %memref_22 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memset  %memref_22, %cst : memref<2x1xf32, 1>, f32
    gpu.launch_func  @test_layernorm_backward_kernel_11::@test_layernorm_backward_kernel_11 blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_21 : memref<2x3xf32>, %c3 : index, %cst : f32, %cst_0 : f32, %c0 : index, %memref_22 : memref<2x1xf32, 1>)
    %memref_23 = gpu.alloc  () : memref<2x3xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_12::@test_layernorm_backward_kernel_12 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_21 : memref<2x3xf32>, %memref_17 : memref<2x1x3xf32>, %c0 : index, %memref_23 : memref<2x3xf32>)
    %memref_24 = gpu.alloc  () : memref<2x1xf32, 1>
    gpu.memset  %memref_24, %cst : memref<2x1xf32, 1>, f32
    gpu.launch_func  @test_layernorm_backward_kernel_13::@test_layernorm_backward_kernel_13 blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref_23 : memref<2x3xf32>, %c3 : index, %cst : f32, %cst_0 : f32, %c0 : index, %memref_24 : memref<2x1xf32, 1>)
    gpu.dealloc  %memref_23 : memref<2x3xf32>
    %memref_25 = gpu.alloc  () : memref<2x1x3xf32>
    gpu.launch_func  @test_layernorm_backward_kernel_14::@test_layernorm_backward_kernel_14 blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%memref_21 : memref<2x3xf32>, %memref_22 : memref<2x1xf32, 1>, %c0 : index, %memref_17 : memref<2x1x3xf32>, %memref_24 : memref<2x1xf32, 1>, %memref_16 : memref<2x1xf32>, %memref_25 : memref<2x1x3xf32>)
    gpu.dealloc  %memref_16 : memref<2x1xf32>
    gpu.dealloc  %memref_17 : memref<2x1x3xf32>
    gpu.dealloc  %memref_21 : memref<2x3xf32>
    gpu.dealloc  %memref_22 : memref<2x1xf32, 1>
    gpu.dealloc  %memref_24 : memref<2x1xf32, 1>
    %collapse_shape = memref.collapse_shape %memref_25 [[0, 1], [2]] : memref<2x1x3xf32> into memref<2x3xf32>
    %memref_26 = gpu.alloc  () : memref<3xf32>
    gpu.memcpy  %memref_26, %memref_19 : memref<3xf32>, memref<3xf32, 1>
    gpu.dealloc  %memref_19 : memref<3xf32, 1>
    %memref_27 = gpu.alloc  () : memref<3xf32>
    gpu.memcpy  %memref_27, %memref_20 : memref<3xf32>, memref<3xf32, 1>
    gpu.dealloc  %memref_20 : memref<3xf32, 1>
    gpu.memcpy  %arg0, %memref : memref<2x3xf32>, memref<2x3xf32, 1>
    gpu.memcpy  %arg1, %memref_3 : memref<2x3xf32>, memref<2x3xf32, 1>
    gpu.memcpy  %arg2, %memref_5 : memref<2x1xf32>, memref<2x1xf32, 1>
    gpu.memcpy  %arg3, %memref_7 : memref<2x1xf32>, memref<2x1xf32, 1>
    gpu.memcpy  %arg4, %memref_9 : memref<3xf32>, memref<3xf32, 1>
    return %collapse_shape, %memref_26, %memref_27 : memref<2x3xf32>, memref<3xf32>, memref<3xf32>
  }
  gpu.module @test_layernorm_backward_kernel {
    gpu.func @test_layernorm_backward_kernel(%arg0: memref<2x3xf32, 1>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: index, %arg5: memref<2x1xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg6 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg7 = %arg2) -> (f32) {
        %4 = memref.load %arg0[%block_id_x_1, %arg6] : memref<2x3xf32, 1>
        %5 = arith.addf %arg7, %4 : f32
        scf.yield %5 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.divf %1, %arg3 : f32
      %3 = arith.cmpi eq, %thread_id_x_0, %arg4 : index
      scf.if %3 {
        %4 = memref.atomic_rmw addf %2, %arg5[%block_id_x_1, %arg4] : (f32, memref<2x1xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_3 {
    gpu.func @test_layernorm_backward_kernel_3(%arg0: memref<2x3xf32, 1>, %arg1: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32, 1>
      %1 = arith.mulf %0, %0 : f32
      memref.store %1, %arg1[%block_id_x, %block_id_y] : memref<2x3xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_4 {
    gpu.func @test_layernorm_backward_kernel_4(%arg0: memref<2x3xf32>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: index, %arg5: memref<2x1xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg6 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg7 = %arg2) -> (f32) {
        %4 = memref.load %arg0[%block_id_x_1, %arg6] : memref<2x3xf32>
        %5 = arith.addf %arg7, %4 : f32
        scf.yield %5 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.divf %1, %arg3 : f32
      %3 = arith.cmpi eq, %thread_id_x_0, %arg4 : index
      scf.if %3 {
        %4 = memref.atomic_rmw addf %2, %arg5[%block_id_x_1, %arg4] : (f32, memref<2x1xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_5 {
    gpu.func @test_layernorm_backward_kernel_5(%arg0: memref<2x1xf32, 1>, %arg1: index, %arg2: memref<2x1xf32, 1>, %arg3: f32, %arg4: memref<2x1xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %arg1] : memref<2x1xf32, 1>
      %1 = memref.load %arg2[%block_id_x, %arg1] : memref<2x1xf32, 1>
      %2 = arith.mulf %1, %1 : f32
      %3 = arith.subf %0, %2 : f32
      %4 = arith.addf %3, %arg3 : f32
      %5 = math.rsqrt %4 : f32
      memref.store %5, %arg4[%block_id_x, %arg1] : memref<2x1xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_6 {
    gpu.func @test_layernorm_backward_kernel_6(%arg0: memref<2x3xf32, 1>, %arg1: memref<2x1xf32, 1>, %arg2: index, %arg3: memref<2x1xf32>, %arg4: memref<2x1x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32, 1>
      %1 = memref.load %arg1[%block_id_x, %arg2] : memref<2x1xf32, 1>
      %2 = memref.load %arg3[%block_id_x, %arg2] : memref<2x1xf32>
      %3 = arith.subf %0, %1 : f32
      %4 = arith.mulf %3, %2 : f32
      memref.store %4, %arg4[%block_id_x, %arg2, %block_id_y] : memref<2x1x3xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_7 {
    gpu.func @test_layernorm_backward_kernel_7(%arg0: memref<2x3xf32, 1>, %arg1: memref<2x1x3xf32>, %arg2: index, %arg3: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32, 1>
      %1 = memref.load %arg1[%block_id_x, %arg2, %block_id_y] : memref<2x1x3xf32>
      %2 = arith.mulf %0, %1 : f32
      memref.store %2, %arg3[%block_id_x, %block_id_y] : memref<2x3xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_8 {
    gpu.func @test_layernorm_backward_kernel_8(%arg0: memref<2x3xf32>, %arg1: index, %arg2: f32, %arg3: index, %arg4: memref<3xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 3, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg5 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg6 = %arg2) -> (f32) {
        %3 = memref.load %arg0[%arg5, %block_id_x_1] : memref<2x3xf32>
        %4 = arith.addf %arg6, %3 : f32
        scf.yield %4 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.cmpi eq, %thread_id_x_0, %arg3 : index
      scf.if %2 {
        %3 = memref.atomic_rmw addf %1, %arg4[%block_id_x_1] : (f32, memref<3xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_9 {
    gpu.func @test_layernorm_backward_kernel_9(%arg0: memref<2x3xf32, 1>, %arg1: index, %arg2: f32, %arg3: index, %arg4: memref<3xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 3, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg5 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg6 = %arg2) -> (f32) {
        %3 = memref.load %arg0[%arg5, %block_id_x_1] : memref<2x3xf32, 1>
        %4 = arith.addf %arg6, %3 : f32
        scf.yield %4 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.cmpi eq, %thread_id_x_0, %arg3 : index
      scf.if %2 {
        %3 = memref.atomic_rmw addf %1, %arg4[%block_id_x_1] : (f32, memref<3xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_10 {
    gpu.func @test_layernorm_backward_kernel_10(%arg0: memref<2x3xf32, 1>, %arg1: memref<3xf32, 1>, %arg2: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32, 1>
      %1 = memref.load %arg1[%block_id_y] : memref<3xf32, 1>
      %2 = arith.mulf %0, %1 : f32
      memref.store %2, %arg2[%block_id_x, %block_id_y] : memref<2x3xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_11 {
    gpu.func @test_layernorm_backward_kernel_11(%arg0: memref<2x3xf32>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: index, %arg5: memref<2x1xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg6 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg7 = %arg2) -> (f32) {
        %4 = memref.load %arg0[%block_id_x_1, %arg6] : memref<2x3xf32>
        %5 = arith.addf %arg7, %4 : f32
        scf.yield %5 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.divf %1, %arg3 : f32
      %3 = arith.cmpi eq, %thread_id_x_0, %arg4 : index
      scf.if %3 {
        %4 = memref.atomic_rmw addf %2, %arg5[%block_id_x_1, %arg4] : (f32, memref<2x1xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_12 {
    gpu.func @test_layernorm_backward_kernel_12(%arg0: memref<2x3xf32>, %arg1: memref<2x1x3xf32>, %arg2: index, %arg3: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32>
      %1 = memref.load %arg1[%block_id_x, %arg2, %block_id_y] : memref<2x1x3xf32>
      %2 = arith.mulf %0, %1 : f32
      memref.store %2, %arg3[%block_id_x, %block_id_y] : memref<2x3xf32>
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_13 {
    gpu.func @test_layernorm_backward_kernel_13(%arg0: memref<2x3xf32>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: index, %arg5: memref<2x1xf32, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %block_id_x_1 = gpu.block_id  x
      %block_dim_x_2 = gpu.block_dim  x
      %0 = scf.for %arg6 = %thread_id_x_0 to %arg1 step %block_dim_x_2 iter_args(%arg7 = %arg2) -> (f32) {
        %4 = memref.load %arg0[%block_id_x_1, %arg6] : memref<2x3xf32>
        %5 = arith.addf %arg7, %4 : f32
        scf.yield %5 : f32
      }
      %1 = gpu.all_reduce  add %0 uniform {
      } : (f32) -> f32
      %2 = arith.divf %1, %arg3 : f32
      %3 = arith.cmpi eq, %thread_id_x_0, %arg4 : index
      scf.if %3 {
        %4 = memref.atomic_rmw addf %2, %arg5[%block_id_x_1, %arg4] : (f32, memref<2x1xf32, 1>) -> f32
      }
      gpu.return
    }
  }
  gpu.module @test_layernorm_backward_kernel_14 {
    gpu.func @test_layernorm_backward_kernel_14(%arg0: memref<2x3xf32>, %arg1: memref<2x1xf32, 1>, %arg2: index, %arg3: memref<2x1x3xf32>, %arg4: memref<2x1xf32, 1>, %arg5: memref<2x1xf32>, %arg6: memref<2x1x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<2x3xf32>
      %1 = memref.load %arg1[%block_id_x, %arg2] : memref<2x1xf32, 1>
      %2 = memref.load %arg3[%block_id_x, %arg2, %block_id_y] : memref<2x1x3xf32>
      %3 = memref.load %arg4[%block_id_x, %arg2] : memref<2x1xf32, 1>
      %4 = memref.load %arg5[%block_id_x, %arg2] : memref<2x1xf32>
      %5 = arith.mulf %2, %3 : f32
      %6 = arith.subf %0, %1 : f32
      %7 = arith.subf %6, %5 : f32
      %8 = arith.mulf %7, %4 : f32
      memref.store %8, %arg6[%block_id_x, %arg2, %block_id_y] : memref<2x1x3xf32>
      gpu.return
    }
  }
}

