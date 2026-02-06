// Test file for binary operations: arithmetic and boolean
// This file contains 2 functions demonstrating binary operations

module {
  // Function 1: Binary Arithmetic Operations
  func.func @binary_arithmetic(
    %arg0: tensor<4x8xf32>,
    %arg1: tensor<4x8xf32>
  ) -> tensor<4x8xf32> {
    
    %add = nova.add %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %sub = nova.sub %add, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %mul = nova.mul %sub, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %div = nova.div %mul, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %mod = nova.mod %div, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %pow = nova.pow %mod, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %max = nova.max %pow, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    %min = nova.min %max, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
    
    return %add : tensor<4x8xf32>
  }

  // Function 2: Binary Boolean Operations
  func.func @binary_boolean(
    %arg0: tensor<4x8xf32>,
    %arg1: tensor<4x8xf32>
  ) -> tensor<4x8xi1> {
    
    %and = "nova.and"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
    %or  = "nova.or"(%and, %arg1) : (tensor<4x8xi1>, tensor<4x8xf32>) -> tensor<4x8xi1>
    %xor = "nova.xor"(%or,  %arg1) : (tensor<4x8xi1>, tensor<4x8xf32>) -> tensor<4x8xi1>
    
    return %xor : tensor<4x8xi1>
  }
}