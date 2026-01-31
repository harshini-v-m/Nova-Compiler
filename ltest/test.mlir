module {
  // 2D Matmul
  func.func @matmul_2d(%arg0: tensor<10x8xf32,  #nova.device<"1">>) -> tensor<f32, #nova.device<"1">> attributes {llvm.emit_c_interface} {
    %0 = nova.reduce<sum> %arg0 dimension = [0,1]
      : tensor<10x8xf32, #nova.device<"1">>
        return %0 : tensor<f32, #nova.device<"1">>
  }
}
