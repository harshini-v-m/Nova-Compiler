#include "Compiler/API/NovaCompilerAPI.h"
#include "Compiler/Pipeline/Pipeline.h"
#include "Compiler/Pipeline/Gpupipeline.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InitLLVM.h"

#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::nova;

static llvm::cl::opt<std::string> inputFile(
    llvm::cl::Positional, llvm::cl::desc("<input .mlir file>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> outputFile(
    "o", llvm::cl::desc("Output file"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> device(
    "device", llvm::cl::desc("Target device (cpu or gpu)"),
    llvm::cl::value_desc("device"), llvm::cl::init("cpu"));

static llvm::cl::opt<bool> outputLLVMIR(
    "emit-llvm", llvm::cl::desc("Output LLVM IR instead of MLIR"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verbose(
    "verbose", llvm::cl::desc("Enable verbose output"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Nova API Test Tool\n");

  // Create Nova API instance
  NovaCompilerAPI compiler;

  // Set up compiler options
  CompilerOptions options;
  options.device = device.getValue();
  options.runFullPipeline = true;
  options.verbose = verbose.getValue();

  std::error_code ec;
  llvm::raw_fd_ostream outputStream(outputFile.getValue(), ec);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return 1;
  }

  if (outputLLVMIR.getValue()) {
    // Compile to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = compiler.compileFile(inputFile.getValue(), llvmContext, options);
    
    if (!llvmModule) {
      llvm::errs() << "Compilation failed\n";
      return 1;
    }

    // Print LLVM IR
    llvmModule->print(outputStream, nullptr);
  } else {
    // Compile and output MLIR after lowering
    // We need to parse, run pipeline, and print MLIR
    auto fileOrErr = llvm::MemoryBuffer::getFile(inputFile.getValue());
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Failed to open file: " << ec.message() << "\n";
      return 1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    MLIRContext context;
    DialectRegistry registry;
    NovaCompilerAPI::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Failed to parse MLIR file\n";
      return 1;
    }

    if (failed(verify(*module))) {
      llvm::errs() << "Module verification failed\n";
      return 1;
    }

    // Run the pipeline
    PassManager pm(&context);
    if (options.device == "gpu") {
      createNovaGPUPipelines(pm);
    } else {
      createNovaPipelines(pm);
    }

    if (failed(pm.run(*module))) {
      llvm::errs() << "Pipeline execution failed\n";
      return 1;
    }

    // Print the lowered MLIR
    module->print(outputStream);
  }

  return 0;
}
