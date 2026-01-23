module {
  func.func @main(%arg0: tensor<8x16xf32>,
  %arg1: tensor<1x16xf32>) -> 
  tensor<8x16xf32> {
    %1 = nova.div %arg0 ,%arg1 : tensor<8x16xf32>,tensor<1x16xf32>
    return %1 : tensor<8x16xf32>
  }
}