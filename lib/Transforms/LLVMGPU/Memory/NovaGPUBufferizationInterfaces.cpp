//===- NovaGPUBufferizationInterfaces.cpp - Bufferization for Nova GPU ops ===//
//
// Registers BufferizableOpInterface external model for nova::ValueBarrierOp.
//
// nova::ValueBarrierOp (tensor-semantic) bufferizes as:
//   1. Emit gpu::BarrierOp at the current position — the read-sync fence.
//   2. Replace each result with an in-place memref alias of the corresponding
//      input (no copy — BufferRelation::Equivalent).
//
// This mirrors IREE's ValueBarrierOpBufferizationInterface in
// Codegen/Dialect/GPU/Transforms/BufferizationInterfaces.cpp.
//
// Write barriers (formerly vector-semantic nova.value_barrier) no longer exist
// at bufferization time: NovaGPUVectorAllocPass now emits gpu::BarrierOp
// directly at block start, exactly like IREE's GPUVectorAllocPass.
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir::nova {

namespace {

/// BufferizableOpInterface external model for nova::ValueBarrierOp.
///
/// The op is value-semantic (Pure) pre-bufferization. At bufferization time
/// the tensor inputs become memrefs and the op emits a gpu::BarrierOp to
/// synchronize threads before any consumer reads the result memrefs.
struct ValueBarrierOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          ValueBarrierOpBufferizationInterface, nova::ValueBarrierOp> {

  // The barrier aliases its inputs in-place — it never reads or writes to
  // generate a distinct buffer; it only enforces ordering.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  // result[i] is an in-place alias of input[i] — equivalent relation.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(opOperand.getOperandNumber()),
             BufferRelation::Equivalent}};
  }

  // The buffer type of result[i] is the same as the buffer type of input[i].
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto barrierOp = cast<nova::ValueBarrierOp>(op);
    if (!barrierOp.hasTensorSemantics())
      return failure();
    int idx = cast<OpResult>(value).getResultNumber();
    auto bufType = bufferization::getBufferType(barrierOp.getInputs()[idx],
                                               options, state, invocationStack);
    if (failed(bufType))
      return failure();
    return cast<BaseMemRefType>(*bufType);
  }

  /// Emit gpu::BarrierOp then replace the op with in-place memref aliases.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto barrierOp = cast<nova::ValueBarrierOp>(op);

    // Only tensor-semantic value_barriers reach bufferization.
    // Vector-semantic ones were replaced by gpu::BarrierOp in
    // NovaGPUVectorAllocPass and no longer exist at this point.
    if (!barrierOp.hasTensorSemantics())
      return failure();

    // Get memref buffers for each input tensor (in-place, no copy).
    SmallVector<Value> buffers;
    buffers.reserve(barrierOp.getNumOperands());
    for (Value input : barrierOp.getInputs()) {
      FailureOr<Value> buf = getBuffer(rewriter, input, options, state);
      if (failed(buf))
        return failure();
      buffers.push_back(*buf);
    }

    // Emit the read-sync barrier at the precise SSA location.
    // Synchronizes workgroup (shared) memory: ensures all threads have
    // completed their writes before any thread reads the result memrefs.
    auto readBarrier = gpu::BarrierOp::create(rewriter, barrierOp.getLoc());
    readBarrier->setAttr(
        "address_spaces",
        rewriter.getArrayAttr({gpu::AddressSpaceAttr::get(
            rewriter.getContext(), gpu::AddressSpace::Workgroup)}));

    // Replace all value_barrier results with their memref aliases (in-place).
    replaceOpWithBufferizedValues(rewriter, op, buffers);
    return success();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// ToLayoutOp — in-place alias, result type == input type (AllTypesMatch).
//
// tensor-semantic to_layout is a Pure passthrough: it carries layout metadata
// for VectorDistribute but does not copy data. Declaring result ≡ input lets
// OneShotBufferize prove that parallel_insert_slice destinations sharing the
// same shared_outs slot are in-place, eliminating the tensor.empty staging
// copies that would otherwise appear after linalg.copy / linalg.generic in
// ConfigureTensorLayouts-annotated forall bodies.
// ---------------------------------------------------------------------------
namespace {
struct ToLayoutOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          ToLayoutOpBufferizationInterface, nova::vec_ext::ToLayoutOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  // Output aliases the input in-place — no copy ever needed.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  // Buffer type of the result is identical to the buffer type of the input.
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto toLayout = cast<nova::vec_ext::ToLayoutOp>(op);
    if (!toLayout.hasTensorSemantics())
      return failure();
    auto bufType = bufferization::getBufferType(toLayout.getInput(), options,
                                               state, invocationStack);
    if (failed(bufType))
      return failure();
    return cast<BaseMemRefType>(*bufType);
  }

  // Replace the op with a no-op: output is the same buffer as input.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto toLayout = cast<nova::vec_ext::ToLayoutOp>(op);
    if (!toLayout.hasTensorSemantics())
      return failure();
    FailureOr<Value> buf =
        getBuffer(rewriter, toLayout.getInput(), options, state);
    if (failed(buf))
      return failure();
    replaceOpWithBufferizedValues(rewriter, op, *buf);
    return success();
  }
};
} // namespace

void registerNovaValueBarrierBufferizationInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, nova::NovaDialect *) {
    nova::ValueBarrierOp::attachInterface<
        ValueBarrierOpBufferizationInterface>(*ctx);
  });
  registry.addExtension(
      +[](MLIRContext *ctx, nova::vec_ext::NovaVectorExtDialect *) {
        nova::vec_ext::ToLayoutOp::attachInterface<
            ToLayoutOpBufferizationInterface>(*ctx);
      });
}

} // namespace mlir::nova
