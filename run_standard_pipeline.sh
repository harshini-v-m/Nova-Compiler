
# Complete MLIR to LLVM IR Pipeline for GPU Code
# This script converts MLIR with Linalg operations to LLVM IR with GPU kernel support

# Adjusted paths for this environment
# USE CUSTOM NOVA-OPT for the first step
MLIR_OPT="./build/tools/nova-opt/nova-opt"
MLIR_TRANSLATE="../mlir/llvm-project/build/bin/mlir-translate"
LLC="../mlir/llvm-project/build/bin/llc"
INPUT_FILE="test/test.mlir"

echo "=== MLIR to LLVM IR Conversion Pipeline (using Nova Compiler) ==="
echo ""

# Full pipeline command
echo "Step 1: Running optimization and lowering passes..."
# Using --nova-gpu-pipeline directly!
$MLIR_OPT $INPUT_FILE \
  --nova-gpu-pipeline \
  -o intermediate.mlir

if [ $? -eq 0 ]; then
    echo "✓ Lowering passes completed successfully (Intermediate: intermediate.mlir)"
    echo ""
    echo "Step 2: Extracting GPU module..."
    
    # Extract the GPU module separately
    $MLIR_OPT intermediate.mlir \
      --gpu-module-to-binary="format=isa" \
      -o with_binary.mlir 
    
    if [ $? -eq 0 ]; then
        echo "✓ GPU module serialized"
        echo ""
        echo "Step 3: Translating to LLVM IR..."
        
        # Translate to LLVM IR
        $MLIR_TRANSLATE --mlir-to-llvmir with_binary.mlir -o output.ll
        
        if [ $? -eq 0 ]; then
            echo "✓ Translation to LLVM IR successful"
            echo "Output saved to output.ll"
            echo ""
            
            echo "Step 4: Compiling and executing..."
            
            # Compile LLVM IR to object file
            $LLC -filetype=obj -relocation-model=pic output.ll -o output.o
            
            # Compile driver
            clang++ -c driver.cpp -o driver.o
            
            # Link and produce executable (renamed to a.out as requested to remove 'pipelinecheck' name)
            clang++ driver.o output.o -o a.out \
              -L/usr/local/cuda/lib64 -lcudart -ldl -lm -lmlir_cuda_runtime \
              -L/home/blu-bridge021/Desktop/mlir/llvm-project/build/lib
            
            if [ $? -eq 0 ]; then
                echo "✓ Build successful! Running executable..."
                echo ""
                # Execute with Rank 3 (8x128x768)
                LD_LIBRARY_PATH=/home/blu-bridge021/Desktop/mlir/llvm-project/build/lib:$LD_LIBRARY_PATH ./a.out
            else
                echo "✗ Linking failed"
                exit 1
            fi
        else
            echo "✗ Translation to LLVM IR failed"
            exit 1
        fi
    else
        echo "✗ GPU module serialization failed"
        exit 1
    fi
else
    echo "✗ Lowering passes failed"
    exit 1
fi
