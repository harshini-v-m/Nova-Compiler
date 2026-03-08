// Test file for unary operations: arithmetic, exponent, and trigonometry
// This file contains 3 functions demonstrating unary operations

module {
  // Function 1: Unary Arithmetic Operations


  // Function 2: Unary Exponent/Logarithm Operations
  func.func @unary_exponent(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4x8xf32> {
    
    %exp   = nova.exp   %arg0 : tensor<4x8xf32>
    %exp2  = nova.exp2  %exp : tensor<4x8xf32>
    %log   = nova.log   %exp2 : tensor<4x8xf32>
    %log2  = nova.log2  %log : tensor<4x8xf32>
    %log10 = nova.log10 %log2 : tensor<4x8xf32>
    
    return %log10 : tensor<4x8xf32>
  }

  // Function 3: Unary Trigonometry Operations
 
}