// Test file for activation functions
// This file demonstrates all activation operations in nova dialect

module {
  func.func @activation_functions(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4x8xf32> {
    
    %relu = nova.relu %arg0 : tensor<4x8xf32>
    %sigmoid = nova.sigmoid %relu : tensor<4x8xf32>
    %softmax = nova.softmax %sigmoid : tensor<4x8xf32>
    %gelu = nova.gelu %softmax : tensor<4x8xf32>
    
    return %gelu : tensor<4x8xf32>
  }
}