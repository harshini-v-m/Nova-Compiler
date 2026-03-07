#include "Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"

using namespace mlir;

namespace mlir
{
  namespace nova
  {

    // Helper: Check if an operation is directly/indirectly inside a gpu.launch
    static bool isInsideGpuLaunch(Operation *op)
    {
      Operation *parent = op->getParentOp();
      while (parent)
      {
        if (isa<gpu::LaunchOp>(parent))
          return true;
        parent = parent->getParentOp();
      }
      return false;
    }

    // Helper: Check if a loop already has GPU mapping attributes
    static bool hasGpuMapping(Operation *op)
    {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op))
      {
        return forallOp.getMappingAttr() != nullptr;
      }
      return false;
    }

    // Helper: Add GPU block-level mapping to a top-level scf.forall or scf.parallel
    // Maps innermost loop -> DimX, next -> DimY, next -> DimZ
    static void addGpuBlockMapping(Operation *op, MLIRContext *ctx)
    {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op))
      {
        // For scf.parallel, we just add the mapping attribute directly
        // The shape is the number of loops
        int64_t numLoops = parallelOp.getNumLoops();

        SmallVector<Attribute, 3> mapping;
        int dimIdx = 0;
        for (int64_t i = numLoops - 1; i >= 0 && dimIdx < 3; --i, ++dimIdx)
        {
          gpu::MappingId mappingId;
          switch (dimIdx)
          {
          case 0:
            mappingId = gpu::MappingId::DimX;
            break;
          case 1:
            mappingId = gpu::MappingId::DimY;
            break;
          case 2:
            mappingId = gpu::MappingId::DimZ;
            break;
          default:
            return;
          }
          mapping.push_back(gpu::GPUBlockMappingAttr::get(ctx, mappingId));
        }

        // Reverse to match loop order (outermost first)
        SmallVector<Attribute, 3> reversedMapping(llvm::reverse(mapping));

        // Set the mapping attribute on the parallel loop
        // This requires conversion to scf.forall, which the next pass will do
        // For now, we'll just add the attribute
        ArrayAttr mappingAttr = ArrayAttr::get(ctx, reversedMapping);
        parallelOp->setAttr("mapping", mappingAttr);
      }
    }

    struct NovaGpuMapParallelLoopPass
        : public PassWrapper<NovaGpuMapParallelLoopPass, OperationPass<func::FuncOp>>
    {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaGpuMapParallelLoopPass)

      void runOnOperation() override
      {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();

        // Step 1: Manually add GPU block mapping to top-level scf.parallel/scf.forall
        // This explicitly mirrors the logic of createGpuMapParallelLoopsPass
        // but with full control to skip nested loops and avoid IR corruption
        func.walk([&](Operation *op) {
          // Only process scf.parallel and unmapped scf.forall operations
          bool isParallel = isa<scf::ParallelOp>(op);
          bool isForall = isa<scf::ForallOp>(op) && !hasGpuMapping(op);

          if (!isParallel && !isForall)
            return;

          // Skip if already inside a gpu.launch (nested loop - illegal to convert)
          if (isInsideGpuLaunch(op))
            return;

          // Add GPU block-level mapping (DimX, DimY, DimZ)
          addGpuBlockMapping(op, ctx);
        });

        // Step 2: Convert mapped scf.parallel/scf.forall to gpu.launch
        // This pass only converts operations that have the GPU mapping attribute
        PassManager pm(func->getContext(), func.getOperationName());
        pm.addPass(createConvertParallelLoopToGpuPass());
        if (failed(pm.run(func)))
          return signalPassFailure();
      }

      StringRef getArgument() const final { return "nova-gpu-map-parallel-wrapper"; }
      StringRef getDescription() const final
      {
        return "Adds GPU block-level mapping to top-level scf.parallel/scf.forall "
               "loops, then converts them to gpu.launch blocks. Skips loops nested "
               "inside existing gpu.launch blocks.";
      }
    };

    std::unique_ptr<Pass> createNovaGpuMapParallelLoopPass()
    {
      return std::make_unique<NovaGpuMapParallelLoopPass>();
    }

    void registerNovaGpuMapParallelLoopPass()
    {
      PassRegistration<NovaGpuMapParallelLoopPass>();
    }

  } // namespace nova
} // namespace mlir