 module {
  func.func @test_main(%arg0: tensor<8x8xf32,#nova.device<"1">>, %arg1: tensor<8x8xf32,#nova.device<"1">>,%target: tensor<8xi32,#nova.device<"1">>) -> tensor<1xf32> attributes {llvm.emit_c_interface} {
    %0=nova.matmul %arg0 ,%arg1 : tensor<8x8xf32,#nova.device<"1">>, tensor<8x8xf32,#nova.device<"1">>
    %1=nova.add %0 ,%arg1 : tensor<8x8xf32>, tensor<8x8xf32,#nova.device<"1">>
    %2 = nova.sce %1 ,%target : tensor<8x8xf32>, tensor<8xi32,#nova.device<"1">>
   return %2 : tensor<1xf32>
  }
 }