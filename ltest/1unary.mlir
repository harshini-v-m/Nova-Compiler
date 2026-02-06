// Test file for unary operations: arithmetic, exponent, and trigonometry
// This file contains 3 functions demonstrating unary operations

module {
  // Function 1: Unary Arithmetic Operations
  func.func @unary_arithmetic(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4x8xi1> {
    
    %square = nova.square %arg0 : tensor<4x8xf32>
    %sqrt   = nova.sqrt   %square : tensor<4x8xf32>
    %neg    = nova.neg    %sqrt : tensor<4x8xf32>
    %abs    = nova.abs    %neg : tensor<4x8xf32>
    %sign   = nova.sign   %abs : tensor<4x8xf32>
    %recip  = nova.reciprocal %sign : tensor<4x8xf32>
    %not    = "nova.not"(%recip) : (tensor<4x8xf32>) -> tensor<4x8xi1>
    
    return %not : tensor<4x8xi1>
  }

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
  func.func @unary_trigonometry(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4x8xf32> {
    
    // Basic trigonometric functions
    %sin  = nova.sin  %arg0 : tensor<4x8xf32>
    %cos  = nova.cos  %sin : tensor<4x8xf32>
    %tan  = nova.tan  %cos : tensor<4x8xf32>
    
    // Inverse trigonometric functions
    %asin = nova.asin %tan : tensor<4x8xf32>
    %acos = nova.acos %asin : tensor<4x8xf32>
    %atan = nova.atan %acos : tensor<4x8xf32>
    
    // Hyperbolic functions
    %sinh = nova.sinh %atan : tensor<4x8xf32>
    %cosh = nova.cosh %sinh : tensor<4x8xf32>
    %tanh = nova.tanh %cosh : tensor<4x8xf32>
    
    // Inverse hyperbolic functions
    %asinh = nova.asinh %tanh : tensor<4x8xf32>
    %acosh = nova.acosh %asinh : tensor<4x8xf32>
    %atanh = nova.atanh %acosh : tensor<4x8xf32>
    
    return %atanh : tensor<4x8xf32>
  }
}