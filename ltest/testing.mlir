 module {
  func.func @test_main(%arg0: tensor<8x1x4xf32,#nova.device<"1">>, %arg1: tensor<1x4xf32>) -> tensor<1xf32> attributes {llvm.emit_c_interface} {
    %0 = nova.sce %arg0, %arg1 : tensor<8x1x4xf32,#nova.device<"1">>, tensor<1x4xf32>
   return %0 : tensor<1xf32>
  }
 }