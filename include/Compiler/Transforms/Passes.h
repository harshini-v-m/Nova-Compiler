#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "Compiler/Transforms/FuseMatmulBias.h"
#include "Compiler/Transforms/FixGpuLaunch.h"
#include "Compiler/Transforms/GenerateDynamicWrapper.h"
#include "Compiler/Translation/NovaReduceToGpu/NovaReduceToGpu.h"

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createConvertMemRefToGpuPass();

#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Passes.h.inc"

}  // namespace nova
}  // namespace mlir

#endif