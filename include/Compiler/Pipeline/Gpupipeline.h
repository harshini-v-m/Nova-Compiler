#ifndef COMPILER_GPU_PIPELINE_PIPELINE_H
#define COMPILER_GPU_PIPELINE_PIPELINE_H

namespace mlir {
namespace nova {

// Register all Nova pipelines
void registerNovaGPUPipelines();
void createNovaGPUPipelines(OpPassManager &pm);
} // namespace nova
} // namespace mlir


#endif 