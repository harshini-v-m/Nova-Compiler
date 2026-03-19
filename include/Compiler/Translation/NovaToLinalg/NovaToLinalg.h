#ifndef MLIR_CONVERSION_NOVATOLINALG_H
#define MLIR_CONVERSION_NOVATOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;
namespace nova {

// From TNovaToLinalg.cpp — elementwise Nova ops → linalg.generic
std::unique_ptr<Pass> createNovaElementwiseToLinalgPass();
void registerNovaElementwiseToLinalgPass();
void populateNovaToLinalgPatternsTemplate(RewritePatternSet &patterns);

// From NovaToLinalg.cpp — structural Nova ops (matmul, gather, etc.) → linalg
std::unique_ptr<Pass> createNovaToLinalgPass();
void registerNovaToLinalgPass();
void populateNovaToLinalgPatterns(RewritePatternSet &patterns);

// From NovaLinalgFusion.cpp — Horizontal fusion of linalg.generic ops
std::unique_ptr<Pass> createNovaLinalgHorizontalFusionPass();
void registerNovaLinalgHorizontalFusionPass();

} // namespace nova
} // namespace mlir

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