module {
  func.func @activation_functions(
    %arg0: tensor<4x8xf32>
  ) -> tensor<4x8xf32> {
    
    %softmax = nova.softmax %arg0 : tensor<4x8xf32>

    return %softmax : tensor<4x8xf32>
  }
}