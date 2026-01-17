//===- GenerateDynamicWrapper.cpp - Dynamic args wrapper pass -----------===//
//
// Nova Compiler - Transforms
// 
// This pass generates a wrapper function that takes a pointer array,
// enabling unlimited function arguments without libffi.
//
// Must run AFTER convert-func-to-llvm when we have LLVM::LLVMFuncOp.
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/GenerateDynamicWrapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {

struct GenerateDynamicWrapperPass 
    : public PassWrapper<GenerateDynamicWrapperPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateDynamicWrapperPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Find the _mlir_ciface_main function (the C interface wrapper)
    LLVM::LLVMFuncOp cifaceFunc = nullptr;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func.getName() == "_mlir_ciface_main") {
        cifaceFunc = func;
      }
    });
    
    if (!cifaceFunc) {
      // No C interface function found, nothing to do
      return;
    }
    
    // Get the number of arguments
    size_t numArgs = cifaceFunc.getNumArguments();
    
    // If already has 2 args (result, args_array), we're done
    if (numArgs <= 2) {
      return;
    }
    
    OpBuilder builder(module.getContext());
    auto loc = cifaceFunc.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(module.getContext());
    auto i64Type = builder.getI64Type();
    auto voidType = LLVM::LLVMVoidType::get(module.getContext());
    
    // Step 1: Rename existing ciface to _mlir_ciface_main_impl
    cifaceFunc.setName("_mlir_ciface_main_impl");
    
    // Step 2: Create new wrapper: _mlir_ciface_main(result_ptr, args_array)
    auto wrapperFuncType = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    
    builder.setInsertionPointAfter(cifaceFunc);
    auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(loc, "_mlir_ciface_main", wrapperFuncType);
    
    // Step 3: Create wrapper body
    Block *entryBlock = wrapperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);
    
    Value resultPtr = entryBlock->getArgument(0);
    Value argsArray = entryBlock->getArgument(1);
    
    // Load each argument from the array and call the impl function
    SmallVector<Value> implArgs;
    implArgs.push_back(resultPtr); // First arg is always result struct pointer
    
    // Load remaining args from the array (indices 0..numArgs-2)
    for (size_t i = 1; i < numArgs; ++i) {
      Value idx = builder.create<LLVM::ConstantOp>(loc, i64Type, 
                                                   builder.getI64IntegerAttr(i - 1));
      Value elemPtr = builder.create<LLVM::GEPOp>(loc, ptrType, ptrType,
                                                  argsArray, ValueRange{idx});
      Value descPtr = builder.create<LLVM::LoadOp>(loc, ptrType, elemPtr);
      implArgs.push_back(descPtr);
    }
    
    // Call the implementation function
    builder.create<LLVM::CallOp>(loc, cifaceFunc, implArgs);
    
    // Return void
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
  
  StringRef getArgument() const final { return "generate-dynamic-wrapper"; }
  StringRef getDescription() const final { 
    return "Generate wrapper for dynamic argument dispatch"; 
  }
};

} // namespace

std::unique_ptr<Pass> mlir::nova::createGenerateDynamicWrapperPass() {
  return std::make_unique<GenerateDynamicWrapperPass>();
}
