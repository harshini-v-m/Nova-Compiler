//===- NovaGPUDistributionPatterns.cpp - Basic GPU distribution patterns --===//
//
// Port of IREE's GPUDistributionPatterns.cpp.
// (iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp)
//
// Patterns:
//   DistributeConstants          — arith.constant splat vector → per-thread splat
//   DistributePoison             — ub.poison vector → per-thread poison
//   DistributeElementwise        — any elementwise op → per-thread op (same opcode)
//   DistributeScfFor             — scf.for iter_args distribution
//   DistributeTrivialLayoutConversions — to_layout(x)->y when same layout → erase
//   DistributeGather             — vector.gather
//   DistributeTrivialExtract     — 0-rank vector.extract → scalar extract
//
// Key differences from IREE:
//   - nova::vec_ext namespace instead of IREE::VectorExt
//   - DistributeTrivialLayoutConversions matches nova_vector_ext.to_layout
//   - DistributePoison: ub.poison is upstream MLIR, no change needed
//
//===----------------------------------------------------------------------===//

#include "NovaGPUVectorDistribution.h"

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::nova {

using vec_ext::VectorLayoutInterface;
using VectorValue = TypedValue<VectorType>;

namespace {

//===----------------------------------------------------------------------===//
// DistributeConstants
// arith.constant with splat vector value → per-thread splat of the same scalar.
//===----------------------------------------------------------------------===//
struct DistributeConstants final
    : OpDistributionPattern<arith::ConstantOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto constant = dyn_cast<VectorValue>(constantOp.getResult());
    if (!constant)
      return failure();

    // Only splat constants can be trivially distributed.
    auto attr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();

    VectorLayoutInterface layout = signature[constant];
    Type elemTy = constant.getType().getElementType();
    VectorType distType =
        VectorType::get(layout.getDistributedShape(), elemTy);
    auto distOp = arith::ConstantOp::create(
        rewriter, constantOp.getLoc(), distType,
        SplatElementsAttr::get(distType, attr.getSplatValue<Attribute>()));
    replaceOpWithDistributedValues(rewriter, constantOp, distOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributePoison
// ub.poison vector → per-thread poison of the distributed shape.
//===----------------------------------------------------------------------===//
struct DistributePoison final : OpDistributionPattern<ub::PoisonOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(ub::PoisonOp poisonOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto poisonVal = dyn_cast<VectorValue>(poisonOp.getResult());
    if (!poisonVal)
      return failure();

    VectorLayoutInterface layout = signature[poisonVal];
    VectorType distType = VectorType::get(layout.getDistributedShape(),
                                          poisonVal.getType().getElementType());
    auto distOp = ub::PoisonOp::create(rewriter, poisonOp.getLoc(), distType);
    replaceOpWithDistributedValues(rewriter, poisonOp, distOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeElementwise
// Any op with elementwise-mappable traits: distribute each operand/result
// independently (all have the same layout), then clone the op.
//===----------------------------------------------------------------------===//
struct DistributeElementwise final
    : OpTraitDistributionPattern<OpTrait::Elementwise> {
  using OpTraitDistributionPattern::OpTraitDistributionPattern;

  LogicalResult matchAndRewrite(Operation *op,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return failure();

    // Distribute each vector operand.
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (auto vecOperand = dyn_cast<VectorValue>(operand))
        operand = getDistributed(rewriter, vecOperand, signature[vecOperand]);
      operands.push_back(operand);
    }

    // Compute distributed result types.
    SmallVector<Type> resultTypes;
    for (Value result : op->getResults()) {
      Type ty = result.getType();
      if (auto vecResult = dyn_cast<VectorValue>(result)) {
        VectorLayoutInterface layout = signature[vecResult];
        ty = VectorType::get(layout.getDistributedShape(),
                             vecResult.getType().getElementType());
      }
      resultTypes.push_back(ty);
    }

    Operation *distOp = mlir::clone(rewriter, op, resultTypes, operands);
    replaceOpWithDistributedValues(rewriter, op, distOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeScfFor
// scf.for with vector iter_args: create a new loop over distributed vectors,
// wrap distributed bbArgs in to_simd so the body can be moved unchanged.
//===----------------------------------------------------------------------===//
struct DistributeScfFor final : OpDistributionPattern<scf::ForOp> {
  using OpDistributionPattern::OpDistributionPattern;

  // Wrap each distributed bbArg in to_simd so the block body sees SIMD types.
  SmallVector<Value>
  getBbArgsReplacements(RewriterBase &rewriter,
                        Block::BlockArgListType bbArgs,
                        ValueRange oldInits) const {
    SmallVector<Value> replacements;
    for (auto [bbArg, oldInit] : llvm::zip_equal(bbArgs, oldInits)) {
      Value val = bbArg;
      if (auto oldVecInit = dyn_cast<VectorValue>(oldInit))
        val = vec_ext::ToSIMDOp::create(rewriter, oldVecInit.getLoc(),
                                        oldVecInit.getType(), val);
      replacements.push_back(val);
    }
    return replacements;
  }

  // Distribute the scf.yield terminator of the new loop.
  LogicalResult distributeYield(PatternRewriter &rewriter,
                                scf::ForOp newForOp) const {
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    auto sig = getOpSignature(yieldOp);
    if (!sig)
      return failure();

    SmallVector<Value> operands;
    for (Value operand : yieldOp->getOperands()) {
      if (auto vecOperand = dyn_cast<VectorValue>(operand))
        operand = getDistributed(rewriter, vecOperand, (*sig)[vecOperand]);
      operands.push_back(operand);
    }
    auto distYield =
        scf::YieldOp::create(rewriter, yieldOp.getLoc(), operands);
    rewriter.replaceOp(yieldOp, distYield);
    return success();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    if (forOp.getInitArgs().empty())
      return failure();

    Block *oldBody = forOp.getBody();

    // Distribute init_args.
    SmallVector<Value> newInitArgs;
    for (Value initArg : forOp.getInitArgs()) {
      if (auto vecInit = dyn_cast<VectorValue>(initArg))
        initArg = getDistributed(rewriter, vecInit, signature[vecInit]);
      newInitArgs.push_back(initArg);
    }

    auto newForOp = scf::ForOp::create(rewriter, forOp.getLoc(),
                                       forOp.getLowerBound(),
                                       forOp.getUpperBound(),
                                       forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());

    // Wrap new iter_args in to_simd so the block body sees SIMD types.
    rewriter.setInsertionPointToStart(newForOp.getBody());
    SmallVector<Value> iterArgs = getBbArgsReplacements(
        rewriter, newForOp.getRegionIterArgs(), forOp.getInitArgs());
    iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

    rewriter.mergeBlocks(oldBody, newForOp.getBody(), iterArgs);

    if (failed(distributeYield(rewriter, newForOp)))
      return failure();

    replaceOpWithDistributedValues(rewriter, forOp, newForOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeTrivialLayoutConversions
// nova_vector_ext.to_layout(x) → erase when input and output have same layout.
//===----------------------------------------------------------------------===//
struct DistributeTrivialLayoutConversions final
    : OpDistributionPattern<vec_ext::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vec_ext::ToLayoutOp toLayoutOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto input  = cast<VectorValue>(toLayoutOp.getInput());
    auto output = cast<VectorValue>(toLayoutOp.getOutput());
    VectorLayoutInterface currentLayout = signature[input];
    VectorLayoutInterface targetLayout  = signature[output];

    if (!currentLayout)
      return rewriter.notifyMatchFailure(toLayoutOp, "no layout on input");
    if (!targetLayout)
      return rewriter.notifyMatchFailure(toLayoutOp, "no layout on output");
    if (currentLayout != targetLayout)
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "non-trivial layout conversion");

    rewriter.replaceOp(toLayoutOp, toLayoutOp.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeGather
// vector.gather: distribute indices, mask, passthru, and result independently.
//===----------------------------------------------------------------------===//
struct DistributeGather final : OpDistributionPattern<vector::GatherOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::GatherOp gatherOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto result   = gatherOp.getResult();
    auto indexVec = gatherOp.getIndexVec();   // the vector<idx> operand
    auto mask     = gatherOp.getMask();
    auto passThru = gatherOp.getPassThru();

    VectorLayoutInterface resultLayout   = signature[result];
    VectorLayoutInterface indicesLayout  = signature[indexVec];
    VectorLayoutInterface maskLayout     = signature[mask];
    VectorLayoutInterface passThruLayout = signature[passThru];

    if (!resultLayout || !indicesLayout || !maskLayout || !passThruLayout)
      return failure();

    VectorType distType =
        VectorType::get(resultLayout.getDistributedShape(),
                        result.getType().getElementType());
    // getIndices() returns the static-index operands (unchanged).
    auto distGather = vector::GatherOp::create(
        rewriter, gatherOp.getLoc(), distType, gatherOp.getBase(),
        gatherOp.getIndices(),
        getDistributed(rewriter, indexVec, indicesLayout),
        getDistributed(rewriter, mask, maskLayout),
        getDistributed(rewriter, passThru, passThruLayout));
    replaceOpWithDistributedValues(rewriter, gatherOp, distGather.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DistributeTrivialExtract
// 0-rank vector.extract: extract from the distributed source.
//===----------------------------------------------------------------------===//
struct DistributeTrivialExtract final
    : OpDistributionPattern<vector::ExtractOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    if (extractOp.getSourceVectorType().getRank() != 0)
      return rewriter.notifyMatchFailure(extractOp,
                                         "only 0-rank extraction supported");

    auto source = extractOp.getVector();
    VectorLayoutInterface sourceLayout = signature[source];
    Value distExtract = vector::ExtractOp::create(
        rewriter, extractOp.getLoc(),
        getDistributed(rewriter, source, sourceLayout), ArrayRef<int64_t>{});
    replaceOpWithDistributedValues(rewriter, extractOp, distExtract);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void populateNovaGPUDistributionPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributeConstants, DistributePoison, DistributeScfFor,
               DistributeTrivialExtract>(patterns.getContext());
  patterns.add<DistributeElementwise>(patterns.getContext());
  patterns.add<DistributeTrivialLayoutConversions>(patterns.getContext());
  patterns.add<DistributeGather>(patterns.getContext());
}

} // namespace mlir::nova
