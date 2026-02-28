module {
  func.func @main(%arg0: memref<8x32x32xf32, 1>, %arg1: memref<8x32xi32, 1>, %arg2: memref<1x1xf32, 1>, %arg3: memref<8x32x384xf32, 1>, %arg4: memref<32x384xf32, 1>, %arg5: memref<384xf32, 1>, %arg6: memref<384x32xf32, 1>, %arg7: memref<32xf32, 1>, %arg8: memref<1xf32, 1>, %arg9: memref<32x384xf32, 1>, %arg10: memref<384xf32, 1>, %arg11: memref<384x32xf32, 1>, %arg12: memref<32xf32, 1>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8x32x32xf32, 1> to tensor<8x32x32xf32, #nova.device<"1">>
    %1 = bufferization.to_tensor %arg1 restrict : memref<8x32xi32, 1> to tensor<8x32xi32, #nova.device<"1">>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1x1xf32, 1> to tensor<1x1xf32, #nova.device<"1">>
    %3 = bufferization.to_tensor %arg3 restrict : memref<8x32x384xf32, 1> to tensor<8x32x384xf32, #nova.device<"1">>
    %4 = bufferization.to_tensor %arg4 restrict : memref<32x384xf32, 1> to tensor<32x384xf32, #nova.device<"1">>
    %5 = bufferization.to_tensor %arg5 restrict : memref<384xf32, 1> to tensor<384xf32, #nova.device<"1">>
    %6 = bufferization.to_tensor %arg6 restrict : memref<384x32xf32, 1> to tensor<384x32xf32, #nova.device<"1">>
    %7 = bufferization.to_tensor %arg7 restrict : memref<32xf32, 1> to tensor<32xf32, #nova.device<"1">>
    %8 = nova.matmul %0, %4 : tensor<8x32x32xf32, #nova.device<"1">>, tensor<32x384xf32, #nova.device<"1">>
    %9 = nova.add %8, %5 : tensor<8x32x384xf32>, tensor<384xf32, #nova.device<"1">>
    %10 = nova.relu %9 : tensor<8x32x384xf32>
    %11 = nova.matmul %10, %6 : tensor<8x32x384xf32>, tensor<384x32xf32, #nova.device<"1">>
    %12 = nova.add %11, %7 : tensor<8x32x32xf32>, tensor<32xf32, #nova.device<"1">>
    %13 = nova.cast %1 : tensor<8x32xi32, #nova.device<"1">> -> tensor<8x32xi32>
    %14 = nova.sce %12, %13 : tensor<8x32x32xf32>, tensor<8x32xi32>
    %15 = nova.softmax %12 dimension = 2 : tensor<8x32x32xf32>
    %16 = nova.reshape %15 : tensor<8x32x32xf32> -> tensor<8192xf32>
    %17 = nova.constant {value = dense<"0x0000000020000000400000006000000080000000A0000000C0000000E00000000001000020010000400100006001000080010000A0010000C0010000E00100000002000020020000400200006002000080020000A0020000C0020000E00200000003000020030000400300006003000080030000A0030000C0030000E00300000004000020040000400400006004000080040000A0040000C0040000E00400000005000020050000400500006005000080050000A0050000C0050000E00500000006000020060000400600006006000080060000A0060000C0060000E00600000007000020070000400700006007000080070000A0070000C0070000E00700000008000020080000400800006008000080080000A0080000C0080000E00800000009000020090000400900006009000080090000A0090000C0090000E0090000000A0000200A0000400A0000600A0000800A0000A00A0000C00A0000E00A0000000B0000200B0000400B0000600B0000800B0000A00B0000C00B0000E00B0000000C0000200C0000400C0000600C0000800C0000A00C0000C00C0000E00C0000000D0000200D0000400D0000600D0000800D0000A00D0000C00D0000E00D0000000E0000200E0000400E0000600E0000800E0000A00E0000C00E0000E00E0000000F0000200F0000400F0000600F0000800F0000A00F0000C00F0000E00F00000010000020100000401000006010000080100000A0100000C0100000E01000000011000020110000401100006011000080110000A0110000C0110000E01100000012000020120000401200006012000080120000A0120000C0120000E01200000013000020130000401300006013000080130000A0130000C0130000E01300000014000020140000401400006014000080140000A0140000C0140000E01400000015000020150000401500006015000080150000A0150000C0150000E01500000016000020160000401600006016000080160000A0160000C0160000E01600000017000020170000401700006017000080170000A0170000C0170000E01700000018000020180000401800006018000080180000A0180000C0180000E01800000019000020190000401900006019000080190000A0190000C0190000E0190000001A0000201A0000401A0000601A0000801A0000A01A0000C01A0000E01A0000001B0000201B0000401B0000601B0000801B0000A01B0000C01B0000E01B0000001C0000201C0000401C0000601C0000801C0000A01C0000C01C0000E01C0000001D0000201D0000401D0000601D0000801D0000A01D0000C01D0000E01D0000001E0000201E0000401E0000601E0000801E0000A01E0000C01E0000E01E0000001F0000201F0000401F0000601F0000801F0000A01F0000C01F0000E01F0000"> : tensor<256xi32>} : tensor<256xi32>
    %18 = nova.reshape %1 : tensor<8x32xi32, #nova.device<"1">> -> tensor<256xi32>
    %19 = nova.add %18, %17 : tensor<256xi32>, tensor<256xi32>
    %20 = nova.constant {value = dense<-1.000000e+00> : tensor<256xf32>} : tensor<256xf32>
    %21 = nova.scatter_add %16, %19, %20 : tensor<8192xf32>, tensor<256xi32>, tensor<256xf32> -> tensor<8192xf32>
    %22 = nova.reshape %21 : tensor<8192xf32> -> tensor<8x32x32xf32>
    %23 = nova.constant {value = dense<3.906250e-03> : tensor<1xf32>} : tensor<1xf32>
    %24 = nova.mul %2, %23 : tensor<1x1xf32, #nova.device<"1">>, tensor<1xf32>
    %25 = nova.mul %22, %24 : tensor<8x32x32xf32>, tensor<1x1xf32>
    %26 = nova.reduce<sum> %25 dimension = [0, 1] : tensor<8x32x32xf32>
    %27 = nova.transpose %6 axes1 = 1 axes2 = 0 : tensor<384x32xf32, #nova.device<"1">>
    %28 = nova.matmul %25, %27 : tensor<8x32x32xf32>, tensor<32x384xf32>
    %29 = nova.transpose %10 axes1 = 2 axes2 = 1 : tensor<8x32x384xf32>
    %30 = nova.matmul %29, %25 : tensor<8x384x32xf32>, tensor<8x32x32xf32>
    %31 = nova.reduce<sum> %30 dimension = [0] : tensor<8x384x32xf32>
    %32 = nova.compare<gt> %9, %3 : tensor<8x32x384xf32>, tensor<8x32x384xf32, #nova.device<"1">>
    %33 = nova.cast %32 : tensor<8x32x384xi1> -> tensor<8x32x384xf32>
    %34 = nova.mul %28, %33 : tensor<8x32x384xf32>, tensor<8x32x384xf32>
    %35 = nova.reduce<sum> %34 dimension = [0, 1] : tensor<8x32x384xf32>
    %36 = nova.transpose %0 axes1 = 2 axes2 = 1 : tensor<8x32x32xf32, #nova.device<"1">>
    %37 = nova.matmul %36, %34 : tensor<8x32x32xf32>, tensor<8x32x384xf32>
    %38 = nova.reduce<sum> %37 dimension = [0] : tensor<8x32x384xf32>
    bufferization.materialize_in_destination %14 in writable %arg8 : (tensor<1xf32>, memref<1xf32, 1>) -> ()
    bufferization.materialize_in_destination %38 in writable %arg9 : (tensor<32x384xf32>, memref<32x384xf32, 1>) -> ()
    bufferization.materialize_in_destination %35 in writable %arg10 : (tensor<384xf32>, memref<384xf32, 1>) -> ()
    bufferization.materialize_in_destination %31 in writable %arg11 : (tensor<384x32xf32>, memref<384x32xf32, 1>) -> ()
    bufferization.materialize_in_destination %26 in writable %arg12 : (tensor<32xf32>, memref<32xf32, 1>) -> ()
    return
  }
}