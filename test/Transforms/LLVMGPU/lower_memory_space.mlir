// RUN: nova-opt %s -nova-gpu-lower-memory-space | FileCheck %s

// CHECK-LABEL: func @test_lower_memory_space
func.func @test_lower_memory_space(%arg0: memref<16xf32, #gpu.address_space<workgroup>>, 
                                   %arg1: memref<16xf32, #gpu.address_space<global>>, 
                                   %arg2: memref<16xf32, #gpu.address_space<private>>) {
  // CHECK: %{{.*}} = memref.alloc() : memref<16xf32, 3>
  %0 = memref.alloc() : memref<16xf32, #gpu.address_space<workgroup>>
  // CHECK: %{{.*}} = memref.alloc() : memref<16xf32, 1>
  %1 = memref.alloc() : memref<16xf32, #gpu.address_space<global>>
  // CHECK: %{{.*}} = memref.alloc() : memref<16xf32, 0>
  %2 = memref.alloc() : memref<16xf32, #gpu.address_space<private>>
  return
}
