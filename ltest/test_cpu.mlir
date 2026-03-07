// Mean reduction over all dimensions (COMMENTED OUT DUE TO LEGALIZATION ISSUE)
   func.func @reduce_mean_all(
     %arg0: tensor<4x8xf32>
   ) -> tensor<1xf32> {
     %mean_all = nova.reduce<mean> %arg0 dimension = [0, 1]
       : tensor<4x8xf32>
     return %mean_all : tensor<1xf32>
}