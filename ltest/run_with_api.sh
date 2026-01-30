#!/bin/bash

# Script to run test files through Nova API instead of nova-opt
# Usage: ./run_with_api.sh <test_file.mlir> [options]
# Options:
#   --device <cpu|gpu>    Target device (default: cpu)
#   --emit-llvm           Output LLVM IR instead of MLIR
#   --output <file>       Output file (default: stdout)
#   --verbose             Enable verbose output

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
DEVICE="cpu"
EMIT_LLVM=false
OUTPUT_FILE="-"
VERBOSE=false
INPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --emit-llvm)
      EMIT_LLVM=true
      shift
      ;;
    --output|-o)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 <test_file.mlir> [options]"
      echo ""
      echo "Options:"
      echo "  --device <cpu|gpu>    Target device (default: cpu)"
      echo "  --emit-llvm           Output LLVM IR instead of MLIR"
      echo "  --output <file>       Output file (default: stdout)"
      echo "  --verbose             Enable verbose output"
      echo "  --help, -h            Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 1unary.mlir"
      echo "  $0 1unary.mlir --device gpu --output output.mlir"
      echo "  $0 1unary.mlir --emit-llvm --output output.ll"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
    *)
      if [ -z "$INPUT_FILE" ]; then
        INPUT_FILE="$1"
      else
        echo "Error: Multiple input files specified"
        exit 1
      fi
      shift
      ;;
  esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
  echo "Error: No input file specified"
  echo "Use --help for usage information"
  exit 1
fi

# Resolve input file path (relative to script directory or absolute)
if [[ "$INPUT_FILE" == /* ]]; then
  # Absolute path
  INPUT_PATH="$INPUT_FILE"
else
  # Relative path - try script directory first, then current directory
  if [ -f "$SCRIPT_DIR/$INPUT_FILE" ]; then
    INPUT_PATH="$SCRIPT_DIR/$INPUT_FILE"
  elif [ -f "$INPUT_FILE" ]; then
    INPUT_PATH="$INPUT_FILE"
  else
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
  fi
fi

# Find nova-api-test binary
NOVA_API_TEST=""
if [ -f "$PROJECT_ROOT/build/tools/nova-api-test/nova-api-test" ]; then
  NOVA_API_TEST="$PROJECT_ROOT/build/tools/nova-api-test/nova-api-test"
elif [ -f "./build/tools/nova-api-test/nova-api-test" ]; then
  NOVA_API_TEST="./build/tools/nova-api-test/nova-api-test"
elif command -v nova-api-test &> /dev/null; then
  NOVA_API_TEST="nova-api-test"
else
  echo "Error: nova-api-test binary not found"
  echo "Please build the project first:"
  echo "  cd $PROJECT_ROOT"
  echo "  mkdir -p build && cd build"
  echo "  cmake .."
  echo "  make nova-api-test"
  exit 1
fi

# Build command
CMD="$NOVA_API_TEST $INPUT_PATH --device $DEVICE"

if [ "$EMIT_LLVM" = true ]; then
  CMD="$CMD --emit-llvm"
fi

if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi

if [ "$OUTPUT_FILE" != "-" ]; then
  CMD="$CMD -o $OUTPUT_FILE"
fi

# Execute command
echo "=== Running Nova API Test ==="
echo "Input:  $INPUT_PATH"
echo "Device: $DEVICE"
echo "Output: $([ "$OUTPUT_FILE" = "-" ] && echo "stdout" || echo "$OUTPUT_FILE")"
echo "Format: $([ "$EMIT_LLVM" = true ] && echo "LLVM IR" || echo "MLIR")"
echo ""
echo "Command: $CMD"
echo ""

$CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✓ Compilation successful"
  if [ "$OUTPUT_FILE" != "-" ]; then
    echo "Output saved to: $OUTPUT_FILE"
  fi
else
  echo ""
  echo "✗ Compilation failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
