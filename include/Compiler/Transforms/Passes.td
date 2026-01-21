#ifndef NOVA_TRANSFORMS_PASSES
#define NOVA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"




def FuseMatmulBias : Pass<"fuse-matmul-bias", "func::FuncOp"> {
    let summary = "Fuse matmul + bias add operations";
    let description = [{
        Detects the pattern of matmul followed by elementwise add (bias)
        and fuses them into a single operation to eliminate intermediate
        memory traffic. This is a common pattern in neural networks.
    }];
    let dependentDialects = [
        "mlir::linalg::LinalgDialect",
        "mlir::arith::ArithDialect"
    ];
    let constructor = "::mlir::nova::createFuseMatmulBiasPass()";
}

def ConvertMemRefToGpu : Pass<"convert-memref-to-gpu", "ModuleOp"> {
    let summary = "Convert memref.alloc/dealloc with memory space 1 to gpu.alloc/dealloc";
    let description = [{
        Converts memref.alloc and memref.dealloc operations that have memory space 1
        to their corresponding gpu.alloc and gpu.dealloc operations.
    }];
    let dependentDialects = [
        "mlir::memref::MemRefDialect",
        "mlir::gpu::GPUDialect"
    ];
    let constructor = "::mlir::nova::createConvertMemRefToGpuPass()";
}

#endif // NOVA_TRANSFORMS_PASSES