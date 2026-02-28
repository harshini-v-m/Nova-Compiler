module {
  // 1. CPU-only test: Transpose
  func.func @test_cpu_only(%arg0: tensor<8x8xf32, #nova.device<"0">>) -> tensor<8x8xf32> attributes { llvm.emit_c_interface } {
    %res = nova.transpose %arg0 : tensor<8x8xf32, #nova.device<"0">>
    return %res : tensor<8x8xf32>
  }

  // 2. GPU-only test: Add and Matmul
  func.func @test_gpu_only(%arg0: tensor<16x16xf32, #nova.device<"1">>, %arg1: tensor<16x16xf32, #nova.device<"1">>) -> tensor<16x16xf32> attributes { llvm.emit_c_interface } {
    %sum = nova.add %arg0, %arg1 : tensor<16x16xf32,#nova.device<"1">>, tensor<16x16xf32,#nova.device<"1">>
    %prod = nova.matmul %sum, %arg0 : tensor<16x16xf32>, tensor<16x16xf32, #nova.device<"1">>
    return %prod : tensor<16x16xf32>
  }

  // 3. Mixed H2D: CPU Constant -> H2D -> GPU Add
  func.func @test_mixed_h2d(%arg0: tensor<32x32xf32, #nova.device<"0">>) -> tensor<32x32xf32> attributes { llvm.emit_c_interface } {
    %gpu_in = nova.to_device %arg0 : tensor<32x32xf32, #nova.device<"0">> -> tensor<32x32xf32, #nova.device<"1">>
    %res = nova.add %gpu_in, %gpu_in : tensor<32x32xf32, #nova.device<"1">>, tensor<32x32xf32, #nova.device<"1">>
    return %res : tensor<32x32xf32>
  }

  // 4. Mixed D2H: GPU Add -> D2H -> CPU Transpose
  func.func @test_mixed_d2h(%arg0: tensor<32x32xf32, #nova.device<"1">>) -> tensor<32x32xf32> attributes { llvm.emit_c_interface } {
    %gpu_sum = nova.add %arg0, %arg0 : tensor<32x32xf32, #nova.device<"1">>, tensor<32x32xf32, #nova.device<"1">>
    %cpu_in = nova.to_device %gpu_sum : tensor<32x32xf32> -> tensor<32x32xf32, #nova.device<"0">>
    %res = nova.transpose %cpu_in : tensor<32x32xf32, #nova.device<"0">>
    return %res : tensor<32x32xf32>
  }

  // 5. Stress Chain: Add -> Matmul -> Gelu -> Softmax on GPU
  func.func @test_stress_chain(%arg0: tensor<64x64xf32, #nova.device<"1">>) -> tensor<64x64xf32> attributes { llvm.emit_c_interface } {
    %sum = nova.add %arg0, %arg0 : tensor<64x64xf32, #nova.device<"1">>, tensor<64x64xf32, #nova.device<"1">>
    %prod = nova.matmul %sum, %arg0 : tensor<64x64xf32>, tensor<64x64xf32, #nova.device<"1">>
    %gelu = nova.gelu %prod : tensor<64x64xf32>
    %soft = nova.softmax %gelu dimension = 1 : tensor<64x64xf32>
    return %soft : tensor<64x64xf32>
  }
}
