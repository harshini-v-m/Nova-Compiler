# Running Tests with Nova API

This directory contains a script `run_with_api.sh` that allows you to run test files through the Nova API instead of using `nova-opt` directly.

## Usage

```bash
./run_with_api.sh <test_file.mlir> [options]
```

## Options

- `--device <cpu|gpu>` - Target device (default: `cpu`)
- `--emit-llvm` - Output LLVM IR instead of MLIR
- `--output <file>` or `-o <file>` - Output file (default: stdout)
- `--verbose` - Enable verbose output
- `--help` or `-h` - Show help message

## Examples

### Basic usage (output MLIR to stdout)
```bash
./run_with_api.sh 1unary.mlir
```

### Output to file
```bash
./run_with_api.sh 1unary.mlir --output output.mlir
```

### GPU target with LLVM IR output
```bash
./run_with_api.sh test_gpu.mlir --device gpu --emit-llvm --output output.ll
```

### CPU target with verbose output
```bash
./run_with_api.sh 2binary.mlir --device cpu --verbose --output lowered.mlir
```

## Building

Before using the script, you need to build the `nova-api-test` tool:

```bash
cd /path/to/mlir-compiler
mkdir -p build && cd build
cmake ..
make nova-api-test
```

The script will automatically find the binary in `build/tools/nova-api-test/nova-api-test`.

## Differences from nova-opt

- Uses the Nova API programmatically instead of command-line tool
- Provides the same lowering pipeline functionality
- Can output both MLIR (after lowering) and LLVM IR
- More flexible for integration into other tools
