#!/bin/bash

# Configuration
MLIR_OPT="./build/tools/nova-opt/nova-opt"
MLIR_TRANSLATE="/home/blu-bridge023/Desktop/llvm-project/build/bin/mlir-translate"
LLC="/home/blu-bridge023/Desktop/llvm-project/build/bin/llc"
LLVM_LIB="/home/blu-bridge023/Desktop/llvm-project/build/lib"

INPUT_MLIR="ltest/matmul_correctness.mlir"
INTERMEDIATE_MLIR="ltest/intermediate.mlir"
BINARY_MLIR="ltest/with_binary.mlir"
OUTPUT_LL="ltest/output.ll"
OUTPUT_OBJ="ltest/output.o"
DRIVER_CPP="ltest/matmul_driver.cpp"
DRIVER_OBJ="ltest/driver.o"
EXECUTABLE="ltest/matmul_test"

echo "=== Nova Matmul Correctness Test ==="

# Step 1: Lowering
echo "1. Lowering to Linalg and GPU dialect..."
$MLIR_OPT $INPUT_MLIR --nova-gpu-optimized-pipeline \
  --convert-bufferization-to-memref \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o $INTERMEDIATE_MLIR
if [ $? -ne 0 ]; then echo "Lowering failed"; exit 1; fi

# Step 2: GPU Serialization
echo "2. Serializing GPU module..."
$MLIR_OPT $INTERMEDIATE_MLIR --gpu-module-to-binary="format=isa" -o $BINARY_MLIR
if [ $? -ne 0 ]; then echo "GPU serialization failed"; exit 1; fi

# Step 3: Translate to LLVM IR
echo "3. Translating to LLVM IR..."
$MLIR_TRANSLATE --mlir-to-llvmir $BINARY_MLIR -o $OUTPUT_LL
if [ $? -ne 0 ]; then echo "Translation failed"; exit 1; fi

# Step 4: Compile to Object file
echo "4. Compiling LLVM IR to object..."
$LLC -filetype=obj -relocation-model=pic $OUTPUT_LL -o $OUTPUT_OBJ
if [ $? -ne 0 ]; then echo "LLC failed"; exit 1; fi

# Step 5: Compile Driver
echo "5. Compiling C++ driver..."
clang++ -c $DRIVER_CPP -o $DRIVER_OBJ -I/usr/local/cuda/include
if [ $? -ne 0 ]; then echo "Driver compilation failed"; exit 1; fi

# Step 6: Link
echo "6. Linking..."
clang++ $DRIVER_OBJ $OUTPUT_OBJ -o $EXECUTABLE \
  -L/usr/local/cuda/lib64 -lcudart -ldl -lm -lmlir_cuda_runtime \
  -L$LLVM_LIB
if [ $? -ne 0 ]; then echo "Linking failed"; exit 1; fi

# Step 7: Run
echo "7. Running verification..."
LD_LIBRARY_PATH=$LLVM_LIB:$LD_LIBRARY_PATH ./$EXECUTABLE
