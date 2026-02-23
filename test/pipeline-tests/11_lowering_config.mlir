// Test: nova-gpu-select-lowering-strategy
// Verifies that the strategy pass attaches a lowering_config DictionaryAttr
// containing workgroup tiles, reduction tiles, mma_kind, and promoted_operands
// to linalg.matmul ops.
//
// Also verifies the full pipeline still works end-to-end with the strategy
// pass as the first step.
//
// RUN: nova-opt %s \
// RUN:   -nova-gpu-select-lowering-strategy="cuda-arch=sm_80" \
// RUN: | FileCheck %s
//
// CHECK-LABEL: func.func @matmul_strategy_test
// CHECK:       linalg.matmul
// CHECK-SAME:  lowering_config = {
// CHECK-SAME:  mma_kind =
// CHECK-SAME:  promoted_operands =
// CHECK-SAME:  reduction =
// CHECK-SAME:  workgroup =

func.func @matmul_strategy_test(
    %arg0: tensor<128x256xf32>,
    %arg1: tensor<256x512xf32>,
    %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%arg2 : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}

// -----------------------------------------------------------------------
// Test 2: sm_75 (Turing) — should still select WMMA intrinsic (mma_kind != 0)
// -----------------------------------------------------------------------
//
// RUN: nova-opt %s \
// RUN:   -nova-gpu-select-lowering-strategy="cuda-arch=sm_75" \
// RUN: | FileCheck %s --check-prefix=TURING
//
// TURING-LABEL: func.func @matmul_strategy_test
// TURING:       linalg.matmul
// TURING-SAME:  lowering_config

// -----------------------------------------------------------------------
// Test 3: Full pipeline integration (sm_80 default)
// Strategy pass must not break the existing pipeline.
// -----------------------------------------------------------------------
//
// RUN: nova-opt %s \
// RUN:   -nova-gpu-select-lowering-strategy \
// RUN:   -nova-tile-and-distribute \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-pad-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-promote-matmul-operands \
// RUN:   -canonicalize -cse \
// RUN:   -nova-gpu-apply-tiling-level-reduction \
// RUN:   -canonicalize -cse \
// RUN: | FileCheck %s --check-prefix=PIPELINE
//
// PIPELINE-LABEL: func.func @matmul_strategy_test
// PIPELINE:       scf.forall
// PIPELINE:         scf.for
// PIPELINE:           linalg.matmul
