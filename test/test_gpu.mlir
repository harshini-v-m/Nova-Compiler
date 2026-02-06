// RUN: nova-opt %s --nova-gpu-pipeline | FileCheck %s

module {
  func.func @main(
    %arg0: tensor<1024x1024xf32>, 
    %arg1: tensor<1024x1024xf32>,
    %bias: tensor<1024x1024xf32>
  ) -> (tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
    
    %0 = nova.matmul %arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
    %1 = nova.add %0, %bias : tensor<1024x1024xf32>, tensor<1024x1024xf32>
    %2 = nova.relu %1 : tensor<1024x1024xf32>
    return %2 : tensor<1024x1024xf32>
  }
}
