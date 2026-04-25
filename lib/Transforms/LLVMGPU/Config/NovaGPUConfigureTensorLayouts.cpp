//===- NovaGPUConfigureTensorLayouts.cpp ----------------------------------===//
//
// Inserts nova_vector_ext.to_layout ops around linalg operands/results to
// anchor VectorDistribute. Reads mma_kind and tile sizes from the
// lowering_config DictionaryAttr set by NovaKernelConfig.
//
// Three categories of ops are handled:
//   1. MMA contraction (mma_kind != 0, isaContractionOpInterface):
//        setContractionAnchor — wraps each operand and the result in a
//        to_layout op carrying a NestedLayoutAttr built from PTX ISA data.
//        Also stamps nova.gpu.mma on the linalg op and on the result to_layout
//        so VectorDistribute can identify and lower it to nvgpu.mma.sync.
//   2. Derived-thread copy/fill (isDerivedThreadConfig):
//        setDerivedThreadConfigLayout — computes thread tiles from the total
//        thread count and annotates the result with a to_layout op.
//   3. Generic elementwise / fill (everything else with a lowering_config):
//        setGPULoweringConfigLayout — distributes subgroup × thread, wraps
//        all operands and results.
//
// Pipeline position:
//   AFTER  ApplyPaddingLevel  (static operand shapes required)
//   BEFORE NovaGPUGenericVectorization
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "Compiler/Transforms/LLVMGPU/NovaGPULoweringConfigUtils.h"
#include "Compiler/Transforms/LLVMGPU/NovaVectorOpUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nova-gpu-configure-tensor-layouts"

using namespace mlir;
using namespace mlir::nova;
using namespace mlir::nova::vec_ext;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Divide `bounds[i]` by `tile[i]` (treat 0 in tile as "don't divide").
/// Updates bounds in-place and returns the actual divisors used.
static SmallVector<int64_t> divideBounds(SmallVector<int64_t> &bounds,
                                         ArrayRef<int64_t> tile) {
  assert(bounds.size() >= tile.size());
  SmallVector<int64_t> divisors(bounds.size(), 1);
  for (size_t i = 0; i < tile.size(); ++i)
    if (tile[i] != 0)
      divisors[i] = tile[i];
  for (size_t i = 0; i < bounds.size(); ++i)
    bounds[i] = llvm::divideCeil(bounds[i], divisors[i]);
  return divisors;
}

/// Compute row-major strides from a counts vector.
/// Dims with count == 1 get stride 0.
static SmallVector<int64_t> stridesFromBasis(ArrayRef<int64_t> basis) {
  SmallVector<int64_t> strides(basis.size(), 0);
  int64_t stride = 1;
  for (int i = (int)basis.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= basis[i];
  }
  return strides;
}

//===----------------------------------------------------------------------===//
// setContractionAnchor — MMA contraction ops
//===----------------------------------------------------------------------===//

static LogicalResult setContractionAnchor(linalg::LinalgOp contract,
                                          DictionaryAttr config,
                                          RewriterBase &rewriter) {
  int32_t mmaKind = getMmaKindRaw(config);
  if (mmaKind == 0)
    return contract->emitError("setContractionAnchor: no mma_kind in config");

  NovaMNKShape mnkShape = getNovaMMKShape(mmaKind);
  if (mnkShape.m == 0)
    return contract->emitError("setContractionAnchor: unknown mma_kind");

  SmallVector<AffineMap> indexingMaps = contract.getIndexingMapsArray();
  auto maybeOpInfo = VectorContractOpInfo::inferFromIndexingMaps(indexingMaps);
  if (failed(maybeOpInfo))
    return contract->emitError(
        "setContractionAnchor: cannot infer contraction dims");

  VectorContractOpInfo opInfo = *maybeOpInfo;
  int64_t innerMDim = (int64_t)opInfo.getMDims().back();
  int64_t innerNDim = (int64_t)opInfo.getNDims().back();
  int64_t innerKDim = (int64_t)opInfo.getKDims().back();
  int64_t rank = (int64_t)indexingMaps[0].getNumDims();

  // Iteration-space bounds (static after padding).
  SmallVector<int64_t> bounds = contract.getStaticLoopRanges();
  if (ShapedType::isDynamicShape(bounds))
    return contract->emitError(
        "setContractionAnchor: dynamic iteration space");

  // numSubgroups comes from wg_subgroup (non-reduction dims only).
  // The contraction runs inside a warp forall of shape wg_subgroup, so each
  // warp tile is bounds[i] / wg_subgroup[i]. DistributeTransferWrite uses
  // numSubgroups to assign disjoint accumulator regions to each warp; if it
  // is all-ones the pass incorrectly predicates all writes on warp_id == 0.
  SmallVector<int64_t> wgSubgroup = getLoweringConfigTileSizes(config, kWgSubgroupKey);
  wgSubgroup.resize(rank, 1);

  SmallVector<int64_t> numSubgroups(rank, 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (wgSubgroup[i] > 1 && bounds[i] > 0)
      numSubgroups[i] = wgSubgroup[i];
  }

  // Divide bounds by subgroup count so batchCounts reflects the per-warp tile.
  for (int64_t i = 0; i < rank; ++i)
    if (numSubgroups[i] > 1)
      bounds[i] = llvm::divideCeil(bounds[i], numSubgroups[i]);

  // Row-major subgroup strides: dims with numSubgroups==1 get stride 0.
  SmallVector<int64_t> subgroupStrides = stridesFromBasis(numSubgroups);
  for (int64_t i = 0; i < rank; ++i)
    if (numSubgroups[i] <= 1)
      subgroupStrides[i] = 0;

  // batchCounts = ceil(bounds[i] / mmaShape[i]): how many MMA tiles each
  // warp executes per dimension.
  SmallVector<int64_t> batchCounts(rank, 1);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t mmaSize = 1;
    if (i == innerMDim) mmaSize = mnkShape.m;
    else if (i == innerNDim) mmaSize = mnkShape.n;
    else if (i == innerKDim) mmaSize = mnkShape.k;
    batchCounts[i] = (mmaSize > 0) ? llvm::divideCeil(bounds[i], mmaSize) : 1;
  }

  MLIRContext *ctx = rewriter.getContext();

  // Build a NestedLayoutAttr for each operand and project it through the
  // operand's indexing map.
  //
  // Operand dim mapping:
  //   0 = LHS: outerDim=M, innerDim=K
  //   1 = RHS: outerDim=K, innerDim=N
  //   2 = ACC: outerDim=M, innerDim=N
  struct DimPair { int64_t outer; int64_t inner; };
  DimPair operandDims[3] = {
    {innerMDim, innerKDim},
    {innerKDim, innerNDim},
    {innerMDim, innerNDim},
  };

  auto buildLayout = [&](int idx) -> VectorLayoutInterface {
    NovaMMASingleSubgroupLayout sgl = getNovaSubgroupLayout(mmaKind, idx);
    if (sgl.empty())
      return {};

    int64_t outerDim = operandDims[idx].outer;
    int64_t innerDim = operandDims[idx].inner;

    SmallVector<int64_t> outerCounts(rank, 1);
    SmallVector<int64_t> threadCounts(rank, 1);
    SmallVector<int64_t> threadStrides(rank, 0);
    SmallVector<int64_t> elemCounts(rank, 1);

    outerCounts[outerDim]   = sgl.outer[0];
    outerCounts[innerDim]   = sgl.outer[1];
    threadCounts[outerDim]  = sgl.thread[0];
    threadCounts[innerDim]  = sgl.thread[1];
    threadStrides[outerDim] = sgl.tstrides[0];
    threadStrides[innerDim] = sgl.tstrides[1];
    elemCounts[outerDim]    = sgl.element[0];
    elemCounts[innerDim]    = sgl.element[1];

    auto fullLayout = NestedLayoutAttr::get(ctx, numSubgroups, batchCounts,
                                            outerCounts, threadCounts,
                                            elemCounts, subgroupStrides,
                                            threadStrides);
    return fullLayout.apply(indexingMaps[idx]);
  };

  VectorLayoutInterface lhsLayout = buildLayout(0);
  VectorLayoutInterface rhsLayout = buildLayout(1);
  VectorLayoutInterface accLayout = buildLayout(2);

  if (!lhsLayout || !rhsLayout || !accLayout)
    return contract->emitError(
        "setContractionAnchor: failed to build layout for one or more operands");

  // Which operands are promoted to shared memory.
  SmallVector<bool> promoted(contract->getNumOperands(), false);
  if (auto promOpt = getPromotedOperandList(config))
    for (int64_t pidx : *promOpt)
      if (pidx < (int64_t)promoted.size())
        promoted[pidx] = true;

  Location loc = contract.getLoc();
  rewriter.setInsertionPoint(contract);

  auto mkToLayout = [&](Value v, VectorLayoutInterface layout,
                        bool sharedMem) -> ToLayoutOp {
    Attribute noIntrinsic;
    return ToLayoutOp::create(rewriter, loc, v, layout, noIntrinsic, sharedMem);
  };

  Value lhsIn = mkToLayout(contract->getOperand(0), lhsLayout, promoted[0]);
  Value rhsIn = mkToLayout(contract->getOperand(1), rhsLayout, promoted[1]);
  // C accumulator is always register-resident — never in shared memory,
  // regardless of what promoted_operands says. shared_memory_conversion on C
  // causes DistributeTransferRead/Write to round-trip through global every
  // K-step instead of keeping the accumulator in registers.
  Value accIn = mkToLayout(contract->getOperand(2), accLayout, /*sharedMem=*/false);

  contract->setOperand(0, lhsIn);
  contract->setOperand(1, rhsIn);
  contract->setOperand(2, accIn);

  // Mark the op with the MMA intrinsic so VectorDistribute can identify it.
  rewriter.modifyOpInPlace(contract, [&]() {
    contract->setAttr("nova.gpu.mma", rewriter.getI32IntegerAttr(mmaKind));
  });

  // Wrap the result in a to_layout carrying the mma_kind — the linalg op is
  // destroyed by GenericVectorization but this anchor survives as the
  // vector-level distribution point.
  rewriter.setInsertionPointAfter(contract);
  Attribute mmaKindAttr = rewriter.getI32IntegerAttr(mmaKind);
  auto resultLayout = ToLayoutOp::create(rewriter, loc,
                                          contract->getResult(0),
                                          accLayout, mmaKindAttr);
  rewriter.replaceAllUsesExcept(contract->getResult(0), resultLayout.getResult(),
                                resultLayout);

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-configure-tensor-layouts] contraction: "
                 << contract->getName() << "\n";
  });
  return success();
}

//===----------------------------------------------------------------------===//
// setDerivedThreadConfigLayout — copy/fill ops with derived_thread=true
//===----------------------------------------------------------------------===//

static LogicalResult setDerivedThreadConfigLayout(linalg::LinalgOp linalgOp,
                                                   DictionaryAttr config,
                                                   int64_t totalThreads,
                                                   RewriterBase &rewriter) {
  int64_t opRank = linalgOp.getNumLoops();
  SmallVector<int64_t> elementTile =
      getLoweringConfigTileSizes(config, kThreadKey);
  elementTile.resize(opRank, 1);

  // Override the innermost dim with a 128-bit vectorization width when the
  // resulting tile evenly distributes across threads. This produces
  // vector<1x4xf32> / vector<1x8xf16> loads instead of scalar loads.
  if (opRank > 0) {
    Type elemTy;
    for (Value operand : linalgOp->getOperands())
      if (auto shaped = dyn_cast<ShapedType>(operand.getType())) {
        elemTy = shaped.getElementType();
        break;
      }
    if (elemTy && elemTy.isIntOrFloat()) {
      unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
      int64_t vecElems = (bitWidth > 0) ? (128 / (int64_t)bitWidth) : 1;
      SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
      int64_t innermostRange = loopRanges.back();
      if (vecElems > 1 && !ShapedType::isDynamic(innermostRange) &&
          innermostRange % vecElems == 0) {
        int64_t remaining = innermostRange / vecElems;
        bool ok = (remaining % totalThreads == 0) ||
                  (totalThreads % remaining == 0);
        if (ok)
          elementTile.back() = vecElems;
      }
    }
  }

  SmallVector<int64_t> opShape = linalgOp.getStaticLoopRanges();

  for (auto [size, element] : llvm::zip(opShape, elementTile)) {
    if (ShapedType::isDynamic(size))
      return success(); // tail tile is dynamic — skip
    if (element == 0 || size % element != 0)
      return linalgOp->emitError(
          "derived thread: element tile doesn't divide shape");
    size /= element;
  }

  SmallVector<int64_t> threadTile(opRank, 1);
  SmallVector<int64_t> threadStrides(opRank, 0);
  int64_t residualThreads = totalThreads;
  int64_t currStride = 1;

  // Distribute threads innermost-first.
  for (int i = opRank - 1; i >= 0 && residualThreads > 1; --i) {
    int64_t size = opShape[i];
    int64_t threadBlock;
    if (residualThreads % size == 0)
      threadBlock = size;
    else if (size % residualThreads == 0)
      threadBlock = residualThreads;
    else
      return linalgOp->emitError(
          "derived thread: cannot distribute threads evenly");

    threadTile[i] = threadBlock;
    threadStrides[i] = currStride;
    opShape[i] /= threadBlock;
    currStride *= threadBlock;
    residualThreads /= threadBlock;
  }

  MLIRContext *ctx = rewriter.getContext();
  SmallVector<int64_t> subgroupTile(opRank, 1);
  SmallVector<int64_t> subgroupStrides(opRank, 0);
  SmallVector<int64_t> outerTile(opRank, 1);
  // opShape now holds the remaining batch counts.
  auto layout = NestedLayoutAttr::get(ctx, subgroupTile, opShape, outerTile,
                                      threadTile, elementTile, subgroupStrides,
                                      threadStrides);

  Location loc = linalgOp.getLoc();
  rewriter.setInsertionPointAfter(linalgOp);
  for (OpResult result : linalgOp->getResults()) {
    VectorLayoutInterface resultLayout =
        layout.apply(linalgOp.getIndexingMapMatchingResult(result));
    Attribute noIntrinsic;
    auto toLayout =
        ToLayoutOp::create(rewriter, loc, result, resultLayout, noIntrinsic);
    rewriter.replaceAllUsesExcept(result, toLayout.getResult(), toLayout);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-configure-tensor-layouts] derived-thread: "
                 << linalgOp->getName() << "\n";
  });
  return success();
}

//===----------------------------------------------------------------------===//
// setGPULoweringConfigLayout — generic elementwise/fill ops
//===----------------------------------------------------------------------===//

static LogicalResult setGPULoweringConfigLayout(linalg::LinalgOp linalgOp,
                                                 DictionaryAttr config,
                                                 RewriterBase &rewriter) {
  int64_t rank = linalgOp.getNumLoops();
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();

  SmallVector<int64_t> subgroupTile =
      getLoweringConfigTileSizes(config, kSubgroupKey);
  SmallVector<int64_t> threadTile =
      getLoweringConfigTileSizes(config, kThreadKey);
  subgroupTile.resize(rank, 1);
  threadTile.resize(rank, 1);

  SmallVector<int64_t> numSubgroups = divideBounds(bounds, subgroupTile);
  SmallVector<int64_t> subgroupStrides = stridesFromBasis(numSubgroups);
  for (int64_t i = 0; i < rank; ++i)
    if (numSubgroups[i] == 1)
      subgroupStrides[i] = 0;

  SmallVector<int64_t> numThreads = divideBounds(bounds, threadTile);
  SmallVector<int64_t> threadStrides = stridesFromBasis(numThreads);
  for (int64_t i = 0; i < rank; ++i)
    if (numThreads[i] == 1)
      threadStrides[i] = 0;

  // Remaining bounds are the batch counts.
  SmallVector<int64_t> batchCounts = bounds;
  SmallVector<int64_t> outerTile(rank, 1);

  MLIRContext *ctx = rewriter.getContext();
  auto layout = NestedLayoutAttr::get(ctx, numSubgroups, batchCounts, outerTile,
                                      numThreads, threadTile, subgroupStrides,
                                      threadStrides);

  SmallVector<bool> promoted(linalgOp->getNumOperands(), false);
  if (auto promOpt = getPromotedOperandList(config))
    for (int64_t pidx : *promOpt)
      if (pidx < (int64_t)promoted.size())
        promoted[pidx] = true;

  Location loc = linalgOp.getLoc();
  Attribute noIntrinsic;

  rewriter.setInsertionPoint(linalgOp);
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    VectorLayoutInterface opLayout =
        layout.apply(linalgOp.getMatchingIndexingMap(&operand));
    auto toLayout = ToLayoutOp::create(rewriter, loc, operand.get(), opLayout,
                                       noIntrinsic,
                                       promoted[operand.getOperandNumber()]);
    operand.set(toLayout.getResult());
  }

  rewriter.setInsertionPointAfter(linalgOp);
  for (OpResult result : linalgOp->getResults()) {
    VectorLayoutInterface resultLayout =
        layout.apply(linalgOp.getIndexingMapMatchingResult(result));
    auto toLayout =
        ToLayoutOp::create(rewriter, loc, result, resultLayout, noIntrinsic);
    rewriter.replaceAllUsesExcept(result, toLayout.getResult(), toLayout);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[nova-configure-tensor-layouts] generic: "
                 << linalgOp->getName() << "\n";
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NovaGPUConfigureTensorLayoutsPass
    : public PassWrapper<NovaGPUConfigureTensorLayoutsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NovaGPUConfigureTensorLayoutsPass)

  StringRef getArgument() const override {
    return "nova-gpu-configure-tensor-layouts";
  }
  StringRef getDescription() const override {
    return "Annotate linalg ops with nova_vector_ext.to_layout ops carrying "
           "NestedLayoutAttr for VectorDistribute";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NovaVectorExtDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<linalg::LinalgOp> candidates;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (getLoweringConfig(op))
        candidates.push_back(op);
    });

    for (linalg::LinalgOp candidate : candidates) {
      DictionaryAttr config = getLoweringConfig(candidate);
      if (!config)
        continue;

      int32_t mmaKind = getMmaKindRaw(config);
      LogicalResult result = failure();

      if (mmaKind != 0 && linalg::isaContractionOpInterface(candidate)) {
        result = setContractionAnchor(candidate, config, rewriter);
      } else if (mmaKind == 0 && linalg::isaContractionOpInterface(candidate)) {
        // SIMT contraction: distributed by thread tiling; no layout anchor.
        result = success();
      } else if (isDerivedThreadConfig(config)) {
        // If the copy is already inside a thread-mapped forall, the forall
        // itself distributes one thread per slice — adding a to_layout anchor
        // would double-distribute and produce a degenerate zero-stride layout.
        bool insideThreadForall = false;
        if (auto parentForall =
                candidate->getParentOfType<scf::ForallOp>()) {
          if (auto mapping = parentForall.getMappingAttr()) {
            for (Attribute attr : mapping) {
              if (isa<gpu::GPUThreadMappingAttr>(attr)) {
                insideThreadForall = true;
                break;
              }
            }
          }
        }
        if (insideThreadForall) {
          result = success();
        } else {
          int64_t totalThreads = getTargetThreadCount(config);
          if (totalThreads == 0)
            totalThreads = 256;
          result = setDerivedThreadConfigLayout(candidate, config,
                                                totalThreads, rewriter);
        }
      } else {
        result = setGPULoweringConfigLayout(candidate, config, rewriter);
      }

      if (failed(result)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nova-configure-tensor-layouts] failed on: "
                   << candidate->getName() << "\n");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

namespace mlir::nova {

std::unique_ptr<Pass> createNovaGPUConfigureTensorLayoutsPass() {
  return std::make_unique<NovaGPUConfigureTensorLayoutsPass>();
}

void registerNovaGPUConfigureTensorLayoutsPass() {
  PassRegistration<NovaGPUConfigureTensorLayoutsPass>();
}

} // namespace mlir::nova
