// RUN: nova-opt %s -nova-gpu-infer-memory-space | FileCheck %s

func.func @reproduce_incorrect_workgroup_inference(%arg0: tensor<16x8xf32>, %arg1: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // This allocation is in the func.func parent (host scope).
  // It should NOT be assigned #gpu.address_space<workgroup>.
  // CHECK: bufferization.alloc_tensor() {memory_space = #gpu.address_space<private>}
  %0 = bufferization.alloc_tensor() : tensor<16x8xf32>
  
  %1 = scf.forall (%arg3, %arg4) in (1, 2) shared_outs(%arg5 = %arg1) -> (tensor<16x8xf32>) {
    // This nested forall is thread-mapped and inside a block-mapped forall.
    // It is the ONLY user of %0.
    %2 = scf.forall (%arg6, %arg7) in (16, 8) shared_outs(%arg8 = %0) -> (tensor<16x8xf32>) {
      %extracted = tensor.extract_slice %arg0[%arg6, %arg7] [1, 1] [1, 1] : tensor<16x8xf32> to tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %extracted into %arg8[%arg6, %arg7] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<16x8xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
    
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg5[0, 0] [16, 8] [1, 1] : tensor<16x8xf32> into tensor<16x8xf32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  
  return %1 : tensor<16x8xf32>
}
