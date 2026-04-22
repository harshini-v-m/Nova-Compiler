//===- NovaGPUVectorDistribution.h - Vector distribution engine -----------===//
//
// Infrastructure for distributing SIMD vector ops to per-thread SIMT slices.
//
// Port of IREE's GPUVectorDistribution.h
// (iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h)
//
// Usage:
//   1. ConfigureTensorLayouts placed nova_vector_ext.to_layout anchors.
//   2. Call distributeVectorOps(root, patterns, options).
//      - Runs propagateVectorLayoutInfo (NovaVectorLayoutAnalysis)
//      - Sets DistributionSignature on each op
//      - Applies distribution patterns via worklist
//      - Canonicalizes to_simd/to_simt pairs
//      - Verifies all conversion ops were eliminated
//
// Key types:
//   DistributionSignature — maps each vector Value of an op to its layout
//   DistributionPattern   — base class for per-op distribution patterns
//   VectorLayoutOptions   — abstract config (provides default layout + patterns)
//   distributeVectorOps   — main entry point
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_GPU_VECTOR_DISTRIBUTION_H
#define NOVA_TRANSFORMS_LLVMGPU_GPU_VECTOR_DISTRIBUTION_H

#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "Compiler/Transforms/LLVMGPU/NovaVectorLayoutAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::nova {

using vec_ext::VectorLayoutInterface;
using VectorValue = TypedValue<VectorType>;

/// Maps each vector-typed Value of an operation to its VectorLayoutInterface.
/// Non-vector Values are absent from the map.
using DistributionSignature =
    DenseMap<VectorValue, VectorLayoutInterface>;

//===----------------------------------------------------------------------===//
// DistributionPattern — base class for all distribution rewrite patterns
//===----------------------------------------------------------------------===//

struct DistributionPattern : RewritePattern {
  using RewritePattern::RewritePattern;

  /// Get the distributed (SIMT-sliced) form of a SIMD vector value.
  /// If `value` is the result of a to_simd op, returns its source directly.
  /// Otherwise, inserts a to_simt op producing `layout.getDistributedShape()`.
  VectorValue getDistributed(RewriterBase &rewriter, VectorValue value,
                             VectorLayoutInterface layout) const;

  /// Wrap each vector result in `values` with a to_simd op converting it back
  /// to the undistributed SIMD type.  Non-vector values are passed through.
  SmallVector<Value> getOpDistributedReplacements(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange values) const;

  /// Replace `op` with `values`, wrapping each vector result in to_simd.
  void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                      ValueRange values) const;

  /// Return the distribution signature for `op`, or nullopt if not set.
  std::optional<DistributionSignature> getOpSignature(Operation *op) const;

  /// Set a new signature on `op` and mark it for redistribution.
  /// `inputLayouts` / `outputLayouts` list layouts for the vector-typed
  /// operands/results of `op` in order.
  void setSignatureForRedistribution(
      RewriterBase &rewriter, Operation *op,
      ArrayRef<VectorLayoutInterface> inputLayouts,
      ArrayRef<VectorLayoutInterface> outputLayouts) const;

  /// Inline a vector::MaskOp body and distribute the results.
  /// Called by masked operation patterns after distributing the masked op.
  LogicalResult replaceParentMask(PatternRewriter &rewriter,
                                  vector::MaskOp maskOp) const;
};

//===----------------------------------------------------------------------===//
// OpDistributionPattern<SourceOp>
// Template base for patterns matching a single concrete op type.
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct OpDistributionPattern : DistributionPattern {
  OpDistributionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : DistributionPattern(SourceOp::getOperationName(), benefit, context) {}

  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        DistributionSignature &signature,
                                        PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto sig = getOpSignature(op);
    if (!sig)
      return failure();
    return matchAndRewrite(cast<SourceOp>(op), *sig, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// OpTraitDistributionPattern<Trait>
// Template base for patterns matching all ops with a given trait.
//===----------------------------------------------------------------------===//

template <template <typename> class TraitType>
struct OpTraitDistributionPattern : DistributionPattern {
  OpTraitDistributionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : DistributionPattern(Pattern::MatchTraitOpTypeTag(),
                            TypeID::get<TraitType>(), benefit, context) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        DistributionSignature &signature,
                                        PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto sig = getOpSignature(op);
    if (!sig)
      return failure();
    return matchAndRewrite(op, *sig, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// VectorLayoutOptions — abstract configuration for distributeVectorOps
//===----------------------------------------------------------------------===//

class VectorLayoutOptions {
public:
  explicit VectorLayoutOptions(Operation *root, bool fullConversion = true)
      : root(root), fullConversion(fullConversion) {
    assert(root && "root operation must be non-null");
  }
  virtual ~VectorLayoutOptions() = default;

  bool verifyConversion() const { return fullConversion; }

  /// Return a default layout for `type` when the analysis couldn't infer one.
  /// Return a null VectorLayoutInterface to indicate "no default" (op skipped).
  virtual VectorLayoutInterface
  getDefaultLayout(VectorType type) const = 0;

protected:
  Operation *root;
  bool fullConversion;
};

//===----------------------------------------------------------------------===//
// distributeVectorOps — main entry point
//===----------------------------------------------------------------------===//

/// Distribute vector ops in the IR rooted at `root`:
///   1. propagateVectorLayoutInfo — seed from to_layout anchors + fixpoint
///   2. setOpSignature on each op with fully-resolved layouts
///   3. applyVectorDistribution — worklist pattern application
///   4. Canonicalize to_simd/to_simt pairs
///   5. Remove signature attributes
///   6. Verify no stray to_simd/to_simt remain (if options.verifyConversion())
LogicalResult distributeVectorOps(Operation *root,
                                  RewritePatternSet &distributionPatterns,
                                  VectorLayoutOptions &options);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_GPU_VECTOR_DISTRIBUTION_H
