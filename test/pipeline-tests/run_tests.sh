#!/usr/bin/env bash
# run_tests.sh — Run all Nova GPU pipeline tests.
#
# Usage:
#   ./run_tests.sh                  # auto-detect nova-opt from build/
#   NOVA_OPT=/path/to/nova-opt ./run_tests.sh
#
# Exit code: 0 = all passed, non-zero = one or more failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Locate nova-opt ---
if [[ -z "${NOVA_OPT:-}" ]]; then
  NOVA_OPT="$REPO_ROOT/build/tools/nova-opt/nova-opt"
fi

if [[ ! -x "$NOVA_OPT" ]]; then
  echo "ERROR: nova-opt not found at: $NOVA_OPT"
  echo "  Build with: cd $REPO_ROOT/build && make -j8 nova-opt"
  echo "  Or set: NOVA_OPT=/path/to/nova-opt"
  exit 1
fi

echo "Using nova-opt: $NOVA_OPT"
echo ""

# --- Test runner ---
PASS=0
FAIL=0
ERRORS=()

run_test() {
  local name="$1"
  local file="$2"
  local passes="$3"
  local checks="$4"   # grep pattern(s) to verify in output (comma-separated)
  local absent="$5"   # grep pattern(s) that must NOT appear (comma-separated)

  printf "  %-50s " "$name"

  # Run the pass pipeline
  local output
  if ! output=$("$NOVA_OPT" "$file" $passes 2>&1); then
    echo "FAIL (nova-opt error)"
    ERRORS+=("$name: nova-opt returned non-zero")
    ((FAIL++)) || true
    return
  fi

  # Check required patterns
  if [[ -n "$checks" ]]; then
    IFS=',' read -ra patterns <<< "$checks"
    for pat in "${patterns[@]}"; do
      pat="${pat#"${pat%%[![:space:]]*}"}"  # trim leading space
      if ! echo "$output" | grep -q "$pat"; then
        echo "FAIL (missing: '$pat')"
        ERRORS+=("$name: expected pattern not found: '$pat'")
        ((FAIL++)) || true
        return
      fi
    done
  fi

  # Check absent patterns
  if [[ -n "$absent" ]]; then
    IFS=',' read -ra patterns <<< "$absent"
    for pat in "${patterns[@]}"; do
      pat="${pat#"${pat%%[![:space:]]*}"}"
      if echo "$output" | grep -q "$pat"; then
        echo "FAIL (unexpected: '$pat')"
        ERRORS+=("$name: unexpected pattern found: '$pat'")
        ((FAIL++)) || true
        return
      fi
    done
  fi

  echo "PASS"
  ((PASS++)) || true
}

PIPELINE="-nova-tile-and-distribute -nova-gpu-pad-operands"
TILE_ONLY="-nova-tile-and-distribute"

echo "========================================"
echo " Nova GPU Pipeline Tests"
echo "========================================"
echo ""

echo "--- Tiling + Padding (combined pipeline) ---"

run_test \
  "01: matmul non-aligned (100x300 x 300x200)" \
  "$SCRIPT_DIR/01_matmul_non_aligned.mlir" \
  "$PIPELINE" \
  "to (100, 200) step (128, 128),tensor<128x300xf32>,tensor<300x128xf32>,tensor<128x128xf32>,linalg.matmul" \
  ""

run_test \
  "02: matmul aligned (256x128 x 128x512)" \
  "$SCRIPT_DIR/02_matmul_aligned.mlir" \
  "$PIPELINE" \
  "to (256, 512) step (128, 128),linalg.matmul" \
  "tensor.pad"

run_test \
  "03: batch_matmul (4x64x256 x 4x256x64)" \
  "$SCRIPT_DIR/03_batch_matmul.mlir" \
  "$PIPELINE" \
  "to (4, 64, 64) step (1, 128, 128),tensor<1x128x256xf32>,tensor<1x256x128xf32>,linalg.batch_matmul,gpu.block<z>" \
  ""

echo ""
echo "--- Consumer Fusion ---"

run_test \
  "04: matmul + bias (consumer fused)" \
  "$SCRIPT_DIR/04_matmul_bias_fusion.mlir" \
  "$PIPELINE" \
  "linalg.matmul,linalg.generic,scf.forall.in_parallel,gpu.block<y>" \
  ""

run_test \
  "05: matmul + bias + relu (two consumers fused)" \
  "$SCRIPT_DIR/05_matmul_bias_relu.mlir" \
  "$PIPELINE" \
  "linalg.matmul,addf,maximumf,scf.forall.in_parallel" \
  ""

echo ""
echo "--- Full-Tile Optimization ---"

run_test \
  "06: matmul exact tile (128x128) — 1D forall" \
  "$SCRIPT_DIR/06_matmul_exact_tile.mlir" \
  "$PIPELINE" \
  "to (128) step (128),linalg.matmul,gpu.block<x>" \
  "tensor.pad"

echo ""
echo "--- K-Tiling, Promotion, Thread, and Subgroup Tiling ---"

run_test \
  "07: k-dimension tiling reduction step 32" \
  "$SCRIPT_DIR/07_k_tiling_reduction.mlir" \
  "-nova-tile-and-distribute -canonicalize -cse -nova-gpu-pad-operands -canonicalize -cse -nova-gpu-promote-matmul-operands -canonicalize -cse -nova-gpu-apply-tiling-level-reduction -canonicalize -cse" \
  "scf.for.*step %c32,linalg.matmul" \
  ""

run_test \
  "08: gpu promote matmul operands" \
  "$SCRIPT_DIR/08_promotion.mlir" \
  "-nova-tile-and-distribute -canonicalize -cse -nova-gpu-pad-operands -canonicalize -cse -nova-gpu-promote-matmul-operands -canonicalize -cse" \
  "bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>},linalg.copy,nova.fusion_barrier,tensor.empty,linalg.copy,linalg.matmul" \
  ""

run_test \
  "09: gpu apply tiling level thread step 4" \
  "$SCRIPT_DIR/09_thread_tiling.mlir" \
  "-nova-tile-and-distribute -canonicalize -cse -nova-gpu-pad-operands -canonicalize -cse -nova-gpu-promote-matmul-operands -canonicalize -cse -nova-gpu-apply-tiling-level-reduction -canonicalize -cse -nova-gpu-apply-tiling-level-thread -canonicalize -cse" \
  "scf.for.*step %c32,scf.for.*step %c4,scf.for.*step %c4,linalg.matmul" \
  ""

run_test \
  "10: gpu apply tiling level subgroup step 16" \
  "$SCRIPT_DIR/10_subgroup_tiling.mlir" \
  "-nova-tile-and-distribute -canonicalize -cse -nova-gpu-pad-operands -canonicalize -cse -nova-gpu-promote-matmul-operands -canonicalize -cse -nova-gpu-apply-tiling-level-reduction -canonicalize -cse -nova-gpu-apply-tiling-level-subgroup -canonicalize -cse" \
  "scf.for.*step %c32,scf.for.*step %c16,scf.for.*step %c16,linalg.matmul" \
  ""

echo ""
echo "========================================"
if [[ $FAIL -eq 0 ]]; then
  echo " RESULT: All $PASS tests PASSED ✓"
else
  echo " RESULT: $PASS passed, $FAIL FAILED"
  echo ""
  echo "Failures:"
  for err in "${ERRORS[@]}"; do
    echo "  ✗ $err"
  done
  exit 1
fi
echo "========================================"
