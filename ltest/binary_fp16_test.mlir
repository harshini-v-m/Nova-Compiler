// Comprehensive test file for all binary operations with fp16 data type
// This file contains 3 functions testing all 11 binary operations in Nova

module {
  // Function 1: Arithmetic Binary Operations with fp16
  func.func @binary_arithmetic_fp16(
    %arg0: tensor<4x8xf16>,
    %arg1: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: add, sub, mul, div, mod, pow
    %add = nova.add %arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %sub = nova.sub %add, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %mul = nova.mul %sub, %arg0 : tensor<4x8xf16>, tensor<4x8xf16>
    %div = nova.div %mul, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %mod = nova.mod %div, %arg0 : tensor<4x8xf16>, tensor<4x8xf16>
    %pow = nova.pow %mod, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>  
    
    return %pow : tensor<4x8xf16>
  }

  // Function 2: Comparison Binary Operations with fp16
  func.func @binary_comparison_fp16(
    %arg0: tensor<4x8xf16>,
    %arg1: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: max, min
    %max = nova.max %arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %min = nova.min %max, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    
    return %min : tensor<4x8xf16>
  }

  // Function 3: Comprehensive chained test with fp16
  func.func @binary_comprehensive_fp16(
    %arg0: tensor<4x8xf16>,
    %arg1: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Chain multiple operations to test type propagation
    %0 = nova.add %arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %1 = nova.mul %0, %arg0 : tensor<4x8xf16>, tensor<4x8xf16>
    %2 = nova.max %1, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %3 = nova.sub %2, %arg0 : tensor<4x8xf16>, tensor<4x8xf16>
    %4 = nova.div %3, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>
    %5 = nova.min %4, %arg0 : tensor<4x8xf16>, tensor<4x8xf16>
    
    return %5 : tensor<4x8xf16>
  }

  // Function 4: Test with broadcasting (scalar-like tensor)
  func.func @binary_broadcast_fp16(
    %arg0: tensor<4x8xf16>,
    %arg1: tensor<1xf16>
  ) -> tensor<4x8xf16> {
    
    // Test broadcasting behavior with fp16
    %add = nova.add %arg0, %arg1 : tensor<4x8xf16>, tensor<1xf16>
    %mul = nova.mul %add, %arg1 : tensor<4x8xf16>, tensor<1xf16>
    
    return %mul : tensor<4x8xf16>
  }
}
// Comprehensive test file for all unary operations with fp16 data type
// This file contains 5 functions testing all 30+ unary operations in Nova

module {
  // Function 1: Unary Arithmetic Operations with fp16
  func.func @unary_arithmetic_fp16(
    %arg0: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: abs, square, sqrt, rsqrt, sign, neg, reciprocal
    %abs    = nova.abs    %arg0 : tensor<4x8xf16>
    %square = nova.square %abs : tensor<4x8xf16>
    %sqrt   = nova.sqrt   %square : tensor<4x8xf16>
    %rsqrt  = nova.rsqrt  %sqrt : tensor<4x8xf16>
    %sign   = nova.sign   %rsqrt : tensor<4x8xf16>
    %neg    = nova.neg    %sign : tensor<4x8xf16>
    %recip  = nova.reciprocal %neg : tensor<4x8xf16>
    
    return %recip : tensor<4x8xf16>
  }

  // Function 2: Unary Exponent/Logarithm Operations with fp16
  func.func @unary_exponent_fp16(
    %arg0: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: exp, exp2, log, log2, log10
    %exp   = nova.exp   %arg0 : tensor<4x8xf16>
    %exp2  = nova.exp2  %exp : tensor<4x8xf16>
    %log   = nova.log   %exp2 : tensor<4x8xf16>
    %log2  = nova.log2  %log : tensor<4x8xf16>
    %log10 = nova.log10 %log2 : tensor<4x8xf16>
    
    return %log10 : tensor<4x8xf16>
  }

  // Function 3: Unary Trigonometry Operations with fp16
  func.func @unary_trig_fp16(
    %arg0: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: sin, cos, tan, asin, acos, atan
    %sin  = nova.sin  %arg0 : tensor<4x8xf16>
    %cos  = nova.cos  %sin : tensor<4x8xf16>
    %tan  = nova.tan  %cos : tensor<4x8xf16>
    %asin = nova.asin %tan : tensor<4x8xf16>
    %acos = nova.acos %asin : tensor<4x8xf16>
    %atan = nova.atan %acos : tensor<4x8xf16>
    
    return %atan : tensor<4x8xf16>
  }

  // Function 4: Unary Hyperbolic Operations with fp16
  func.func @unary_hyperbolic_fp16(
    %arg0: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: sinh, cosh, tanh, asinh, acosh, atanh
    %sinh  = nova.sinh  %arg0 : tensor<4x8xf16>
    %cosh  = nova.cosh  %sinh : tensor<4x8xf16>
    %tanh  = nova.tanh  %cosh : tensor<4x8xf16>
    %asinh = nova.asinh %tanh : tensor<4x8xf16>
    %acosh = nova.acosh %asinh : tensor<4x8xf16>
    %atanh = nova.atanh %acosh : tensor<4x8xf16>
    
    return %atanh : tensor<4x8xf16>
  }

  // Function 5: Special Functions with fp16
  func.func @unary_special_fp16(
    %arg0: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Test: relu (special activation function)
    %relu = nova.relu %arg0 : tensor<4x8xf16>
    
    return %relu : tensor<4x8xf16>
  }

  // Function 6: Comprehensive chained test with fp16
  func.func @unary_comprehensive_fp16(
    %arg0: tensor<4x8xf16>
  ) -> tensor<4x8xf16> {
    
    // Chain multiple operations to test type propagation
    %0 = nova.abs %arg0 : tensor<4x8xf16>
    %1 = nova.sqrt %0 : tensor<4x8xf16>
    %2 = nova.exp %1 : tensor<4x8xf16>
    %3 = nova.log %2 : tensor<4x8xf16>
    %4 = nova.sin %3 : tensor<4x8xf16>
    %5 = nova.tanh %4 : tensor<4x8xf16>
    %6 = nova.relu %5 : tensor<4x8xf16>
    
    return %6 : tensor<4x8xf16>
  }
}
