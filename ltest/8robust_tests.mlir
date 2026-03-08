module {

  // 2. GPU-only test: Add and Matmul
  func.func @test_gpu_only(%arg0: tensor<16x16xf32, #nova.device<"1">>, %arg1: tensor<16x16xf32, #nova.device<"1">>) -> tensor<16x16xf32> attributes { llvm.emit_c_interface } {
    %sum = nova.add %arg0, %arg1 : tensor<16x16xf32,#nova.device<"1">>, tensor<16x16xf32,#nova.device<"1">>
    %prod = nova.matmul %sum, %arg0 : tensor<16x16xf32>, tensor<16x16xf32, #nova.device<"1">>
    return %prod : tensor<16x16xf32>
  }

}
