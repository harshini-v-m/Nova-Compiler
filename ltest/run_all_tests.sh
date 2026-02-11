#!/bin/bash

# Define the path to the nova-opt binary
# Assuming this script is run from the ltest directory or the project root
if [ -f "build/tools/nova-opt/nova-opt" ]; then
    NOVA_OPT="build/tools/nova-opt/nova-opt"
    TEST_DIR="ltest"
elif [ -f "../build/tools/nova-opt/nova-opt" ]; then
    NOVA_OPT="../build/tools/nova-opt/nova-opt"
    TEST_DIR="."
else
    echo "Error: Could not find nova-opt binary."
    echo "Please run this script from the project root or the ltest directory."
    exit 1
fi

echo "Running tests using $NOVA_OPT in $TEST_DIR..."

# Iterate over all .mlir files in the test directory
for file in "$TEST_DIR"/*.mlir; do
    # Check if file exists (in case of no matches)
    if [ ! -e "$file" ]; then
        echo "No .mlir files found in $TEST_DIR"
        exit 1
    fi

    echo "Testing $file..."
    
    # Run the pipeline command, discarding output unless there is an error
    "$NOVA_OPT" --nova-gpu-pipeline "$file" > /dev/null 2>&1
    
    # Check return code
    if [ $? -ne 0 ]; then
        echo "Error: Failed to lower $file"
        echo "Stopping..."
        exit 1
    fi
done

echo "Successfully lowered all files."
