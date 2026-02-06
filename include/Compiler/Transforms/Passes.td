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

def AffineScalarizeAccumulator : Pass<"affine-scalarize-accumulator", "func::FuncOp"> {
    let summary = "Convert buffer-based loop accumulation to register-based using iter_args";
    let description = [{
        Detects loop-carried accumulation through buffers and transforms them
        to use affine.for iter_args for register-based accumulation.
    }];
    let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::memref::MemRefDialect"
    ];
    let constructor = "::mlir::nova::createAffineScalarizeAccumulatorPass()";
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

def RemDevAttr : Pass<"rem-dev-attr", "func::FuncOp"> {
    let summary = "Remove device attribute from function arguments and add explicit to_device copies";
    let description = [{
        Normalizes function arguments by removing #nova.device attributes.
        If an argument has no attribute, it is considered a host tensor and 
        a nova.to_device operation is inserted at the start of the function.
    }];
    let dependentDialects = [
        "mlir::nova::NovaDialect"
    ];
    let constructor = "::mlir::nova::createRemDevAttrPass()";
}

#endif // NOVA_TRANSFORMS_PASSES