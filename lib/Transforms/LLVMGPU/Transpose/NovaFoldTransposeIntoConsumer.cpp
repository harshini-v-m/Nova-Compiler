//===- NovaFoldTransposeIntoConsumer.cpp -----------------------------------===//
//
// Two complementary patterns that eliminate transpose ops adjacent to matmuls
// by composing the permutation into the contraction's indexing maps.
//
// ------------------------------------------------------------------------------
// Pattern 1 — transpose producer(s) → matmul consumer (FoldTransposeIntoMatmul)
// ------------------------------------------------------------------------------
// Handles LHS-only, RHS-only, or both-transposed in a single application.
//
// Example — both transposed:
// Before:
//   %TA = linalg.generic {in:(d0,d1)->(d0,d1), out:(d0,d1)->(d1,d0)}
//             ins(%A : tensor<KxMxf32>) outs(%empty_A : tensor<MxKxf32>)
//   %TB = linalg.generic {in:(d0,d1)->(d0,d1), out:(d0,d1)->(d1,d0)}
//             ins(%B : tensor<NxKxf32>) outs(%empty_B : tensor<KxNxf32>)
//   %C  = linalg.matmul ins(%TA, %TB : tensor<MxKxf32>, tensor<KxNxf32>)
//                       outs(%init : tensor<MxNxf32>)
//
// After:
//   %C = linalg.generic {
//            indexing_maps = [(d0,d1,d2)->(d2,d0),   // A: transposed LHS
//                             (d0,d1,d2)->(d1,d2),   // B: transposed RHS
//                             (d0,d1,d2)->(d0,d1)],  // C: unchanged
//            iterator_types = [parallel, parallel, reduction]
//        } ins(%A, %B) outs(%init) { matmul body }
//
// Both intermediate transpose buffers are eliminated. PromoteMatmulOperands
// then generates direct coalesced global→shared copies from the originals.
//
// ---------------------------------------------------------------------------------
// Pattern 2 — matmul producer → transpose consumer (FoldMatmulIntoTransposeOutput)
// ---------------------------------------------------------------------------------
// Before:
//   %C = linalg.matmul ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
//                      outs(%init_C : tensor<MxNxf32>)
//   %T = linalg.generic {in:(d0,d1)->(d0,d1), out:(d0,d1)->(d1,d0)}
//            ins(%C) outs(%init_T : tensor<NxMxf32>)
//
// After:
//   %T = linalg.generic {
//            indexing_maps = [(d0,d1,d2)->(d0,d2),   // A: unchanged
//                             (d0,d1,d2)->(d2,d1),   // B: unchanged
//                             (d0,d1,d2)->(d1,d0)],  // output: transposed
//            iterator_types = [parallel, parallel, reduction]
//        } ins(%A, %B) outs(%init_T) { matmul body }
//
// The matmul writes its result directly into the transposed layout — no
// separate data-movement op for the transpose exists at all. %init_C and the
// transpose generic become dead and are removed by the greedy driver's DCE.
//
// Implementation notes
// --------------------
// Both patterns produce a single linalg.GenericOp in ONE replaceOp call.
// generalizeNamedOp is intentionally NOT used (would call replaceOp internally,
// causing a double-replacement segfault in the greedy driver).
//
// The dead ops (transpose producer / matmul producer) are NOT erased manually;
// the greedy driver's DCE handles them automatically.
//
// Mirrors IREE's FuseTransposeWithLinalgOpConsumer and
// FuseTransposeWithLinalgOpProducer patterns in
// iree/compiler/src/iree/compiler/Codegen/Common/PropagateLinalgTranspose.cpp,
// positioned after all FuseMatmulBias calls and before SelectLoweringStrategy
// so the config sees the correct contraction op with (possibly transposed) maps.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-fold-transpose-into-consumer"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::nova {
namespace {

// isTransposeGeneric
// Returns true when `op` is a pure copy-transpose:
//   - exactly 1 DPS input, 1 DPS output
//   - all iterator types parallel
//   - output indexing map is a bijective permutation
//   - body is: linalg.yield %blockArg0   (identity pass-through)
static bool isTransposeGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return false;

  for (auto iter : op.getIteratorTypesArray())
    if (iter != utils::IteratorType::parallel)
      return false;

  // Output map must be a full permutation so its inverse is well-defined.
  AffineMap outMap = op.getIndexingMapsArray().back();
  if (!outMap.isPermutation())
    return false;

  // Body must yield the first (input) block argument unchanged.
  Block &body = op.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  return yield.getNumOperands() == 1 &&
         yield.getOperand(0) == body.getArgument(0);
}

// buildMatmulBody
// Constructs a multiply-accumulate body suitable for any linalg.matmul
// element type.  Block args are (lhs, rhs, acc) — same order as matmul.
static void buildMatmulBody(OpBuilder &b, Location loc, ValueRange args) {
  Value lhs = args[0], rhs = args[1], acc = args[2];
  Value mul, result;
  if (mlir::isa<FloatType>(lhs.getType())) {
    mul    = b.create<arith::MulFOp>(loc, lhs, rhs);
    result = b.create<arith::AddFOp>(loc, mul, acc);
  } else {
    mul    = b.create<arith::MulIOp>(loc, lhs, rhs);
    result = b.create<arith::AddIOp>(loc, mul, acc);
  }
  b.create<linalg::YieldOp>(loc, result);
}

// isMatmulContraction
// Returns true for linalg.MatmulOp and for linalg.GenericOp that has the
// standard 3-loop [parallel, parallel, reduction] structure and a
// multiply-accumulate body (mulf+addf or muli+addi).  Accepts generics
// produced by a previous FoldTransposeIntoMatmul application.
static bool isMatmulContraction(linalg::LinalgOp op) {
  if (isa<linalg::MatmulOp>(op))
    return true;
  auto generic = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!generic) return false;
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1)
    return false;
  auto iters = generic.getIteratorTypesArray();
  if (iters.size() != 3 ||
      iters[0] != utils::IteratorType::parallel ||
      iters[1] != utils::IteratorType::parallel ||
      iters[2] != utils::IteratorType::reduction)
    return false;
  // Body must be exactly two ops: a multiply and an accumulating add.
  Block &body = generic.getRegion().front();
  SmallVector<Operation *> ops;
  for (Operation &innerOp : body)
    if (!isa<linalg::YieldOp>(innerOp))
      ops.push_back(&innerOp);
  if (ops.size() != 2) return false;
  Operation *mul = ops[0], *add = ops[1];
  if (!isa<arith::MulFOp>(mul) && !isa<arith::MulIOp>(mul)) return false;
  if (!isa<arith::AddFOp>(add) && !isa<arith::AddIOp>(add)) return false;
  Value mulResult = mul->getResult(0);
  for (Value operand : add->getOperands())
    if (operand == mulResult) return true;
  return false;
}

// FoldTransposeIntoMatmul
// Matches linalg.matmul whose LHS and/or RHS is produced by a transpose
// generic with a single use and folds all such transpose permutations into
// the matmul's indexing maps in one shot.
//
// Both operands are scanned before emitting any replacement, so a matmul
// with both LHS and RHS transposed is handled in a single pattern application
// rather than requiring two greedy-driver iterations.
//
// Map composition (per operand):
//   transposeInMap  : iter_space → original tensor coords
//   transposeOutMap : iter_space → transposed output coords  (permutation P)
//   consumerMap     : matmul iter_space → transposed output coords
//   new map         : transposeInMap ∘ inv(transposeOutMap) ∘ consumerMap
struct FoldTransposeIntoMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // Seed new maps and inputs from the matmul; update in-place for each
    // foldable operand so both LHS and RHS can be folded simultaneously.
    SmallVector<AffineMap> newMaps = matmulOp.getIndexingMapsArray();
    SmallVector<Value> newInputs;
    for (unsigned i = 0; i < matmulOp.getNumDpsInputs(); ++i)
      newInputs.push_back(matmulOp.getDpsInputOperand(i)->get());

    bool anyFolded = false;
    for (int operandIdx = 0; operandIdx < 2; ++operandIdx) {
      Value input = matmulOp.getDpsInputOperand(operandIdx)->get();
      auto transposeOp = input.getDefiningOp<linalg::GenericOp>();
      if (!transposeOp || !isTransposeGeneric(transposeOp))
        continue;
      // Only fold when this matmul is the transpose's sole consumer.
      if (!transposeOp->hasOneUse())
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "[nova-fold-transpose] folding operand " << operandIdx
                 << " of: " << matmulOp << "\n");

      // Compose the transpose permutation into the operand's existing map.
      SmallVector<AffineMap> transpMaps = transposeOp.getIndexingMapsArray();
      AffineMap transpInMap  = transpMaps[0];
      AffineMap transpOutMap = transpMaps[1];
      AffineMap resolve =
          transpInMap.compose(mlir::inversePermutation(transpOutMap));
      newMaps[operandIdx] = resolve.compose(newMaps[operandIdx]);

      // Bypass the transpose: use its source tensor directly.
      newInputs[operandIdx] = transposeOp.getDpsInputOperand(0)->get();
      anyFolded = true;
    }

    if (!anyFolded)
      return failure();

    SmallVector<Value> outputs;
    for (unsigned i = 0; i < matmulOp.getNumDpsInits(); ++i)
      outputs.push_back(matmulOp.getDpsInitOperand(i)->get());

    // Create the replacement linalg.generic with all composed maps and
    // a type-aware matmul body.  This is a SINGLE root replacement —
    // no intermediate ops, no double-replacement.
    auto newGen = rewriter.create<linalg::GenericOp>(
        matmulOp.getLoc(), matmulOp.getResultTypes(),
        newInputs, outputs, newMaps,
        matmulOp.getIteratorTypesArray(),
        buildMatmulBody);

    LLVM_DEBUG(llvm::dbgs() << "[nova-fold-transpose] result: "
                             << newGen << "\n");

    // Replace the root op.  Any folded transpose producers are now dead and
    // will be removed by the greedy driver's DCE — do NOT erase manually.
    rewriter.replaceOp(matmulOp, newGen.getResults());
    return success();
  }
};

// FoldMatmulIntoTransposeOutput
// Matches a linalg.generic transpose that consumes the sole result of a
// matmul-like op and folds the permutation into the producer's output map so
// the matmul writes its result directly in the transposed layout.
//
// Map composition (see file header for the before/after picture):
//   producerOutMap  : matmul iter_space → C coords  (e.g. (d0,d1,d2)->(d0,d1))
//   transposeInMap  : transpose iter_space → C coords  (often identity)
//   transposeOutMap : transpose iter_space → T coords  (permutation P)
//
//   c_to_t   = transposeOutMap ∘ inv(transposeInMap)   // C-index → T-index
//   newOutMap = c_to_t ∘ producerOutMap                // matmul-iter → T-index
//
// Both transposeInMap and transposeOutMap must be permutations so that
// inversePermutation is well-defined.
struct FoldMatmulIntoTransposeOutput
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Only proceed when this generic is a pure transpose.
    if (!isTransposeGeneric(transposeOp))
      return failure();

    // The transpose's sole input must be the result of a matmul-like op.
    Value transposeInput = transposeOp.getDpsInputOperand(0)->get();
    auto producer = transposeInput.getDefiningOp<linalg::LinalgOp>();
    if (!producer || !isMatmulContraction(producer))
      return failure();

    // Only fold when the matmul result is used exclusively by this transpose.
    if (!producer->hasOneUse())
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-fold-transpose] folding output transpose of: "
               << *producer << "\n");

    // Map composition
    SmallVector<AffineMap> transposeMaps = transposeOp.getIndexingMapsArray();
    AffineMap transposeInMap  = transposeMaps[0]; // transpose-iters → C coords
    AffineMap transposeOutMap = transposeMaps[1]; // transpose-iters → T coords

    // transposeInMap must also be a permutation for its inverse to exist.
    if (!transposeInMap.isPermutation())
      return failure();

    SmallVector<AffineMap> producerMaps = producer.getIndexingMapsArray();
    AffineMap producerOutMap = producerMaps.back(); // matmul-iters → C coords

    // c_to_t: C-index → T-index.
    AffineMap c_to_t   = transposeOutMap.compose(inversePermutation(transposeInMap));
    // newOutMap: matmul-iters → T-index.
    AffineMap newOutMap = c_to_t.compose(producerOutMap);

    SmallVector<AffineMap> newMaps(producerMaps);
    newMaps.back() = newOutMap;

    // Build replacement operands
    SmallVector<Value> newInputs;
    for (unsigned i = 0; i < producer.getNumDpsInputs(); ++i)
      newInputs.push_back(producer.getDpsInputOperand(i)->get());

    // The new output tensor is the transpose's init — already typed tensor<NxM>.
    SmallVector<Value> newOutputs;
    for (unsigned i = 0; i < transposeOp.getNumDpsInits(); ++i)
      newOutputs.push_back(transposeOp.getDpsInitOperand(i)->get());

    // Create the folded contraction
    // Result type is tensor<NxM> (transpose's result type), not tensor<MxN>.
    auto newGen = rewriter.create<linalg::GenericOp>(
        producer->getLoc(),
        transposeOp.getResultTypes(),
        newInputs, newOutputs, newMaps,
        producer.getIteratorTypesArray(),
        buildMatmulBody);

    LLVM_DEBUG(llvm::dbgs()
               << "[nova-fold-transpose] result: " << newGen << "\n");

    // Replace the transpose op.  The matmul producer is now dead and will
    // be removed by the greedy driver's DCE — do NOT erase it manually.
    rewriter.replaceOp(transposeOp, newGen.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaFoldTransposeIntoConsumerPass
    : public PassWrapper<NovaFoldTransposeIntoConsumerPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaFoldTransposeIntoConsumerPass)

  NovaFoldTransposeIntoConsumerPass() = default;
  NovaFoldTransposeIntoConsumerPass(
      const NovaFoldTransposeIntoConsumerPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<FoldTransposeIntoMatmul>(funcOp.getContext());
    patterns.add<FoldMatmulIntoTransposeOutput>(funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const override {
    return "nova-fold-transpose-into-consumer";
  }
  StringRef getDescription() const override {
    return "Fold linalg.generic transpose ops adjacent to linalg.matmul: "
           "(1) transpose-producer into matmul-consumer via LHS/RHS map "
           "composition; (2) matmul-producer into transpose-consumer by "
           "writing the matmul result directly in the transposed layout.";
  }
};

} // namespace

//===----------------------------------------------------------------------===//  
// Public API  
//===----------------------------------------------------------------------===//  

std::unique_ptr<Pass> createNovaFoldTransposeIntoConsumerPass() {
  return std::make_unique<NovaFoldTransposeIntoConsumerPass>();
}

void registerNovaFoldTransposeIntoConsumerPass() {
  PassRegistration<NovaFoldTransposeIntoConsumerPass>();
}

} // namespace mlir::nova
