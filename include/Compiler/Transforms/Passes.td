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

def SCFScalarizeAccumulator : Pass<"scf-scalarize-accumulator", "func::FuncOp"> {
    let summary = "Convert buffer-based loop accumulation to register-based using iter_args";
    let description = [{
        Detects loop-carried accumulation through buffers and transforms them
        to use scf.for iter_args for register-based accumulation.
    }];
    let dependentDialects = [
        "mlir::scf::SCFDialect",
        "mlir::memref::MemRefDialect"
    ];
    let constructor = "::mlir::nova::createSCFScalarizeAccumulatorPass()";
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

def VectorOpt : Pass<"vector-opt", "func::FuncOp"> {
    let summary = "Custom vector optimization pass";
    let description = [{
        Optimizes vector operations, including fusing mul/add to contract
        and optionally unrolling loops.
    }];
    let dependentDialects = [
        "mlir::vector::VectorDialect",
        "mlir::affine::AffineDialect",
        "mlir::scf::SCFDialect",
        "mlir::arith::ArithDialect"
    ];
    let options = [
        Option<"unrollFactor", "unroll-factor", "int", /*default=*/"1",
               "Unrolling factor (1 to disable)">
    ];
    let constructor = "::mlir::nova::createVectorOptPass()";
}

def RenameGpuKernels : Pass<"rename-gpu-kernels", "ModuleOp"> {
    let summary = "Rename GPU kernels to match their module name";
    let description = [{
        Renames GPU kernels to match their containing module name.
        This ensures unique symbols in the generated binary output, which is useful for profiling.
    }];
    let dependentDialects = [
        "mlir::gpu::GPUDialect"
    ];
    let constructor = "::mlir::nova::createRenameGpuKernelsPass()";
}


def GpuRuntimeLowering : Pass<"gpu-runtime-lowering", "ModuleOp"> {
    let summary = "Lower abstract GPU memory operations to CUDA runtime calls";
    let description = [{
        Lower abstract GPU memory operations (gpu.alloc, gpu.memcpy, etc.) 
        to CUDA runtime calls (cudaMalloc, cudaMemcpy, etc.) or custom runtime wrappers.
    }];
    let dependentDialects = [
        "mlir::gpu::GPUDialect",
        "mlir::LLVM::LLVMDialect"
    ];
    let constructor = "::mlir::nova::createGpuRuntimeLoweringPass()";
}

def FixHostGpuMemory : Pass<"fix-host-gpu-memory", "ModuleOp"> {
    let summary = "Patch host-side access to GPU pointers via cudaMemcpy";
    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
    ];
    let constructor = "::mlir::nova::createFixHostGpuMemoryPass()";
}

#endif // NOVA_TRANSFORMS_PASSES
