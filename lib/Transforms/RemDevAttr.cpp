#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nova;

namespace mlir {
namespace nova {

class RemDevAttrPass
    : public PassWrapper<RemDevAttrPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemDevAttrPass)

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    Operation *fOp = func;
    MLIRContext *ctx = fOp->getContext();
    OpBuilder builder(ctx);

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](Type type) -> std::optional<Type> {
      if (auto rtt = llvm::dyn_cast<RankedTensorType>(type)) {
        Attribute encoding = rtt.getEncoding();
        if (encoding && llvm::isa<mlir::nova::NovaDeviceAttr>(encoding)) {
          return RankedTensorType::get(rtt.getShape(), rtt.getElementType());
        }
      }
      return std::nullopt;
    });

    // Phase 1: Explicit Copies (Insertion)
    // Insertion points: to_tensor and BlockArguments
    fOp->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "bufferization.to_tensor") {
        Value memref = op->getOperand(0);
        auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());
        if (memrefType) {
          Attribute space = memrefType.getMemorySpace();
          bool isHost = !space;
          if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(space)) {
            if (intAttr.getInt() == 0)
              isHost = true;
          }
          if (isHost) {
            builder.setInsertionPointAfter(op);
            auto tensorType =
                llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
            if (tensorType) {
              auto plainType = RankedTensorType::get(
                  tensorType.getShape(), tensorType.getElementType());
#if MLIR_VERSION_HAS_PROPERTIES
              // If your MLIR version has properties, you might need to handle
              // them here
#endif
              auto toDev = builder.create<mlir::nova::ToDeviceOp>(
                  op->getLoc(), plainType, op->getResult(0));
              op->getResult(0).replaceAllUsesExcept(toDev.getResult(), toDev);
            }
          }
        }
      }
    });

    if (!fOp->getRegions().empty() && !fOp->getRegion(0).empty()) {
      for (auto arg : fOp->getRegion(0).getArguments()) {
        auto type = llvm::dyn_cast<RankedTensorType>(arg.getType());
        if (type && !type.getEncoding()) {
          builder.setInsertionPointToStart(&fOp->getRegion(0).front());
          auto resultType =
              RankedTensorType::get(type.getShape(), type.getElementType());
          auto toDev = builder.create<mlir::nova::ToDeviceOp>(
              fOp->getLoc(), resultType, (Value)arg);
          Value res = toDev.getResult();
          arg.replaceAllUsesExcept(res, (Operation *)toDev);
        }
      }
    }

    // Phase 2: Update all Result Types and Block Arguments
    fOp->walk([&](Operation *op) {
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        op->getResult(i).setType(replacer.replace(op->getResult(i).getType()));
      }
      for (auto &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto arg : block.getArguments()) {
            arg.setType(replacer.replace(arg.getType()));
          }
        }
      }
    });

    // Phase 3: Update all Attributes and potentially other elements
    fOp->walk([&](Operation *op) {
      DictionaryAttr oldAttrs = op->getAttrDictionary();
      DictionaryAttr newAttrs =
          llvm::dyn_cast_or_null<DictionaryAttr>(replacer.replace(oldAttrs));
      if (newAttrs && newAttrs != oldAttrs) {
        op->setAttrs(newAttrs);
      }
    });

    // Phase 4: Function Signature
    auto typeAttr = fOp->getAttrOfType<TypeAttr>("function_type");
    if (typeAttr) {
      auto oldFuncType =
          llvm::dyn_cast<mlir::FunctionType>(typeAttr.getValue());
      if (oldFuncType) {
        SmallVector<Type> newArgTypes;
        for (auto type : oldFuncType.getInputs())
          newArgTypes.push_back(replacer.replace(type));
        SmallVector<Type> newResultTypes;
        for (auto type : oldFuncType.getResults())
          newResultTypes.push_back(replacer.replace(type));
        auto newFuncType =
            mlir::FunctionType::get(ctx, newArgTypes, newResultTypes);
        fOp->setAttr("function_type", TypeAttr::get(newFuncType));
      }
    }
  }

  StringRef getArgument() const final { return "rem-dev-attr"; }
  StringRef getDescription() const final {
    return "Remove device attribute from function arguments and add explicit "
           "to_device copies";
  }
};

std::unique_ptr<Pass> createRemDevAttrPass() {
  return std::make_unique<RemDevAttrPass>();
}

} // namespace nova
} // namespace mlir
