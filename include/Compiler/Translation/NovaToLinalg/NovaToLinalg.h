#ifndef MLIR_CONVERSION_NOVATOLINALG_H
#define MLIR_CONVERSION_NOVATOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir{
    class Pass;
    class RewritePatternSet;
    class TypeConverter;
    namespace nova{
        std::unique_ptr<Pass> createNovaToLinalgLoweringPass();
        void regsiterNovaToLinalgLoweringTemplatePass();
        void populateNovaToLinalgPatterns(RewritePatternSet &patterns);
        void populateNovaToLinalgPatternsTemplate(RewritePatternSet &patterns);

    }
}
#endif
/*
NovaMatmulOpLoweringgeneric, 
NovaBroadcastInDimOpLowering,
NovaTransposeOpLowering, 
NovaToDeviceOpLowering,
NovaScatterAddOpLowering,
NovaGatherOpLowering,
NovaDivOpLowering,
NovaMulOpLowering
nova::ModOp,
nova::Exp2Op,
nova::Log2Op,
nova::Log10Op
nova::TanOp,
nova::AsinOp,
nova::AcosOp,
nova::AtanOp,
nova::SinhOp,
nova::CoshOp
nova::AsinhOp,
nova::AcoshOp,
nova::AtanhOp,
nova::CompareOp,
nova::SignOp,
ArgMinConverter, 
ArgMaxConverter, 
ReduceOpConverter, 
AdamOpConverter
*/