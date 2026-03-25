#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "Compiler/Transforms/FuseMatmulBias.h"
#include "Compiler/Transforms/NovaFusionPass.h"
#include "Compiler/Transforms/FixGpuLaunch.h"
#include "Compiler/Transforms/GenerateDynamicWrapper.h"
#include "Compiler/Translation/NovaToGpu/NovaToGpu.h"

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createConvertMemRefToGpuPass();
std::unique_ptr<Pass> createAffineScalarizeAccumulatorPass();
std::unique_ptr<Pass> createRemDevAttrPass();
std::unique_ptr<Pass> createVectorOptPass();
std::unique_ptr<Pass> createRenameGpuKernelsPass();

std::unique_ptr<Pass> createNovaFuseReductionIntoProducerPass();
void registerNovaFuseReductionIntoProducerPass();

std::unique_ptr<Pass> createNovaElementwiseOpFusionPass();
void registerNovaElementwiseOpFusionPass();

#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Passes.h.inc"

}  // namespace nova
}  // namespace mlir

#endif