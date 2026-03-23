# NovaGPUApplyTilingLevelReductionPass — Detailed IR Analysis Report

**Pass:** `mlir::nova::NovaGPUApplyTilingLevelReductionPass` (`nova-gpu-apply-tiling-level-reduction`)
**Source:** `lib/Transforms/LLVMGPU/NovaGPUApplyTilingLevelReduction.cpp`
**IR dump location:** `debug2.log` lines 42230–44729
**Pipeline position:** Step 4 (after PromoteMatmulOperands, before Thread tiling)
**Date:** 2026-03-21

---

## 1. Pass Overview

The reduction tiling pass tiles the K (reduction) dimension for all linalg ops that have non-zero `reduction` tile sizes in their `lowering_config`. It creates `scf.for` loops (sequential iteration) for the K dimension, with producer fusion pulling copy ops and other producers into the loop body.

### Key behavior:
- **Contraction ops** (batch_matmul, matmul): K-tiled with kStep=32 → `scf.for 0 to K step 32`
- **Reduction generics** (LN mean/var, softmax): K-tiled with kStep=4 → `scf.for 0 to D step 4`
- **d(gamma)/d(beta) reductions**: Nested `scf.for` loops over batch and sequence dims (kStep=4 each)
- **Promoted `linalg.copy` ops**: Fused as producers into the K-loop → copies shrink from `[wgM × K]` to `[wgM × kStep]`
- **Elementwise ops** (no reduction dim): NOT tiled, pass through unchanged

### Fusion control:
- Producers NOT in the payload set → fused into the K-loop
- `tensor.pad` ops → NOT fused (handled by ExtractSliceOfPadTensorSwap cleanup)
- Destination operands → NOT fused at reduction level

---

## 2. Summary Statistics

| Metric | Count |
|--------|-------|
| Total `scf.for` loops created | **~55** (includes nested loops) |
| K-loops for contraction ops (kStep=32) | **21** (18 batch_matmul + 3 matmul) |
| K-loops for reduction generics (kStep=4) | **~34** |
| Contraction ops inside scf.for | **21/21** (100%) |
| Contraction ops NOT K-tiled | **0** |
| `linalg.copy` ops fused into K-loops | **42/42** (100%) |
| `linalg.copy` ops surviving outside K-loops | **0** |
| Foralls with NO scf.for (elementwise only) | **~15** |

---

## 3. K-Tiling Shapes Summary

### Before K-tiling (from PromoteMatmulOperands):
```
copy A: [1 × 32 × 384]     (full K=384)
copy B: [1 × 384 × 128]    (full K=384)
batch_matmul: [1×32×384] × [1×384×128] → [1×32×128]
```

### After K-tiling (inside scf.for loop):
```
copy A: [1 × 32 × 32]      (kStep=32, shrunk 12×)
copy B: [1 × 32 × 128]     (kStep=32, shrunk 12×)
batch_matmul: [1×32×32] × [1×32×128] → [1×32×128]
```

### Shared memory per K-step (after thread tiling + bufferize):
```
A tile: 32 × 32 × 4 bytes  =  4 KB
B tile: 32 × 128 × 4 bytes = 16 KB
Total per K-step:             20 KB  (well within 48 KB limit)
```

---

## 4. Complete Forall Inventory with K-Tiling Analysis

### Forward Pass — Embedding + LN0

| # | SSA | Bounds | scf.for? | K-loop bounds | kStep | Ops inside loop | Copy fusion |
|---|-----|--------|----------|---------------|-------|-----------------|-------------|
| 1 | %28 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | embedding(1×256×4) + sum_reduction | N/A |
| 2 | %29 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | embedding(1×256×4) + var_reduction | N/A |

### Forward Pass — Transformer Block 0

| # | SSA | Bounds | scf.for? | K-loop bounds | kStep | Ops inside loop | Copy fusion |
|---|-----|--------|----------|---------------|-------|-----------------|-------------|
| 3 | %32:4 | (0,0,0)→(8,1024,1536) step(1,32,128) | YES | 0→384 step 32 | 32 | embed(1×32×32) + LN_norm(1×32×32) + **copy_A**(1×32×32) + W_bcast(1×32×128) + **copy_B**(1×32×128) + **batch_matmul** | **BOTH FUSED** |
| 4 | %34 | (0,0,0)→(8,1024,384) step(1,32,128) | YES | 0→1536 step 32 | 32 | extract_slice(1×32×32) + **copy_A**(1×32×32) + W_bcast(1×32×128) + **copy_B**(1×32×128) + **batch_matmul** | **BOTH FUSED** |

**Forall #3 detail (FFN-up with mega-fusion):**
```
OUTSIDE scf.for:
  %embed_full = linalg.generic (embedding, 1×32×384)  ← full size for LN
  %ln_full    = linalg.generic (LN normalize, 1×32×384) ← full size
  %fill       = linalg.fill (zero-init accumulator, 1×32×128)

INSIDE scf.for (0 to 384 step 32):
  %embed_tile = linalg.generic (embedding, 1×32×32)    ← K-sliced, FUSED
  %ln_tile    = linalg.generic (LN normalize, 1×32×32)  ← K-sliced, FUSED
  %copy_a     = linalg.copy (%ln_tile → tensor.empty)   ← 1×32×32, FUSED
  %w_tile     = linalg.generic (W broadcast, 1×32×128)  ← K-sliced, FUSED
  %copy_b     = linalg.copy (%w_tile → tensor.empty)    ← 1×32×128, FUSED
  %acc        = linalg.batch_matmul(%copy_a, %copy_b, %iter_arg)

OUTSIDE scf.for (post-matmul epilogues):
  %bias = linalg.generic (bias add, 1×32×128)
  %gelu = linalg.generic (GELU, 1×32×128)
```

### Forward Pass — Transformer Block 1

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 5 | %35 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A (LN mean) |
| 6 | %36 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A (LN var) |
| 7 | %37:3 | (0,0,0)→(8,1024,1536) step(1,32,128) | YES | 0→384 step 32 | 32 | **BOTH FUSED** |
| 8 | %38 | (0,0,0)→(8,1024,384) step(1,32,128) | YES | 0→1536 step 32 | 32 | **BOTH FUSED** |

### Forward Pass — Transformer Block 2

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 9 | %39 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A |
| 10 | %40 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A |
| 11 | %41:3 | (0,0,0)→(8,1024,1536) step(1,32,128) | YES | 0→384 step 32 | 32 | **BOTH FUSED** |
| 12 | %42 | (0,0,0)→(8,1024,384) step(1,32,128) | YES | 0→1536 step 32 | 32 | **BOTH FUSED** |

### Forward Pass — Final LN + Output Projection

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 13 | %43 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A |
| 14 | %44 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A |
| 15 | %45 | in(1) | NO | — | — | Single-block LN normalize (elementwise) |
| 16 | %47 | (0,0)→(8192,50304) step(32,128) | YES | 0→384 step 32 | 32 | **BOTH FUSED** (2D matmul: 32×32, 32×128) |

### Forward Pass — Softmax + Loss

| # | SSA | Bounds | scf.for? | K-loop | kStep | Notes |
|---|-----|--------|----------|--------|-------|-------|
| 17 | %48 | (0,0)→(8,1024) step(1,256) | YES | 0→50304 step 4 | 4 | max reduction |
| 18 | %49 | (0,0)→(8,1024) step(1,256) | YES | 0→50304 step 4 | 4 | sum-exp reduction |
| 19 | %51 | in(8,1024,8) | NO | — | — | Elementwise CE loss |
| 20 | — | in(64) | YES | partial sum loop | — | Scalar accumulation |
| 21 | — | in(1) | YES | 0→64 step 1 | 1 | Final loss reduce |
| 22 | %54 | (0,0)→(8,1024) step(1,256) | YES | 0→50304 step 4 | 4 | Backward exp+sum |
| 23 | %55 | (0,0,0)→(8,1024,50304) step(1,1,128) | NO | — | — | Elementwise normalize |

### Backward Pass — Index/Scatter + Output Projection

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 24 | %57 | in(8) | NO | — | — | Label cast |
| 25 | %59 | (0)→(8192) step(1024) | NO | — | — | Index arith |
| 26 | nested | in(8){in(1024)} | NO | — | — | Atomic scatter |
| 27 | %62 | (0,0,0)→(8,1024,50304) step(1,1,128) | NO | — | — | Grad scale |
| 28 | %65 | (0,0)→(8192,384) step(32,128) | YES | 0→50304 step 32 | 32 | **BOTH FUSED** (dX matmul) |
| 29 | %67 | (0,0)→(384,50304) step(32,128) | YES | 0→8192 step 32 | 32 | **BOTH FUSED** (dW matmul) |

### Backward Pass — Final LN

| # | SSA | Bounds | scf.for? | K-loop | kStep | Notes |
|---|-----|--------|----------|--------|-------|-------|
| 30 | %68 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | LN bwd sum#1 |
| 31 | %69 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | LN bwd sum#2 |
| 32 | %71 | (0)→(384) step(128) | YES | nested 0→8 step 4, 0→1024 step 4 | 4 | d(gamma) |
| 33 | %72 | (0)→(384) step(128) | YES | nested 0→8 step 4, 0→1024 step 4 | 4 | d(beta) |

### Backward Pass — FFN Block 3

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 34 | %75 | in(1) | NO | — | — | Transpose (single-block) |
| 35 | %76:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | YES | 0→384 step 32 | 32 | **BOTH FUSED** (dX_up) |
| 36 | %77 | (0,0,0)→(8,1536,384) step(1,32,128) | YES | 0→1024 step 32 | 32 | **BOTH FUSED** (dW_down) |
| 37 | %79 | (0)→(384) step(128) | YES | nested 0→8 step 4, 0→1024 step 4 | 4 | d(bias_down) |
| 38 | %81 | in(1) | NO | — | — | Transpose (single-block) |
| 39 | %82:2 | (0,0,0)→(8,1024,384) step(1,32,128) | YES | 0→1536 step 32 | 32 | **BOTH FUSED** (dX_down) |
| 40 | %83 | (0,0,0)→(8,384,1536) step(1,32,128) | YES | 0→1024 step 32 | 32 | **BOTH FUSED** (dW_up) |
| 41 | %85 | (0)→(1536) step(512) | YES | nested 0→8 step 4, 0→1024 step 4 | 4 | d(bias_up) |

### Backward Pass — LN Block 3 + FFN Block 2

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 42 | %86 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A (LN bwd) |
| 43 | %87 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A (LN bwd) |
| 44 | %88 | in(1) | NO | — | — | Transpose |
| 45 | %89:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | YES | 0→384 step 32 | 32 | **BOTH FUSED** |
| 46 | %90 | (0,0,0)→(8,1536,384) step(1,32,128) | YES | 0→1024 step 32 | 32 | **BOTH FUSED** |
| 47 | %91 | (0)→(384) step(128) | YES | nested | 4 | d(bias_down) |
| 48 | %92 | in(1) | NO | — | — | Transpose |
| 49 | %93:2 | (0,0,0)→(8,1024,384) step(1,32,128) | YES | 0→1536 step 32 | 32 | **BOTH FUSED** |
| 50 | %94 | (0,0,0)→(8,384,1536) step(1,32,128) | YES | 0→1024 step 32 | 32 | **BOTH FUSED** |
| 51 | %95 | (0)→(1536) step(512) | YES | nested | 4 | d(bias_up) |

### Backward Pass — LN Block 2 + FFN Block 1

| # | SSA | Bounds | scf.for? | K-loop | kStep | Copy fusion |
|---|-----|--------|----------|--------|-------|-------------|
| 52 | %96 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A |
| 53 | %97 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | N/A |
| 54 | %98 | in(1) | NO | — | — | Transpose |
| 55 | %99:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | YES | 0→384 step 32 | 32 | **BOTH FUSED** |
| 56 | %100 | (0,0,0)→(8,1536,384) step(1,32,128) | YES | 0→1024 step 32 | 32 | **BOTH FUSED** |
| 57 | %101 | (0)→(384) step(128) | YES | nested | 4 | d(bias_down) |
| 58 | %102 | in(1) | NO | — | — | Transpose |
| 59 | %103:2 | (0,0,0)→(8,1024,384) step(1,32,128) | YES | 0→1536 step 32 | 32 | **BOTH FUSED** |
| 60 | %104 | (0,0,0)→(8,384,1536) step(1,32,128) | YES | 0→1024 step 32 | 32 | **BOTH FUSED** |
| 61 | %105 | (0)→(1536) step(512) | YES | nested | 4 | d(bias_up) |

### Backward Pass — LN Block 0 + Embedding Gradients

| # | SSA | Bounds | scf.for? | K-loop | kStep | Notes |
|---|-----|--------|----------|--------|-------|-------|
| 62 | %106 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | LN bwd |
| 63 | %107 | (0,0)→(8,1024) step(1,256) | YES | 0→384 step 4 | 4 | LN bwd |
| 64 | %109:3 | (0,0)→(1024,384) step(1,128) | YES | 0→8 step 4 | 4 | Pos embed grad |
| 65–67 | nested/scatter | various | NO | — | — | Atomic scatter |

### Optimizer — Weight/Bias Updates

| # | SSA | Bounds | scf.for? | K-loop | kStep | Notes |
|---|-----|--------|----------|--------|-------|-------|
| 68 | %116 | (0,0)→(50304,384) step(1,128) | NO | — | — | Elementwise add |
| 69 | %118 | (0,0)→(1024,384) step(1,128) | NO | — | — | Elementwise add |
| 70 | %120 | (0)→(384) step(128) | YES | nested 0→8 step 4, 0→1024 step 4 | 4 | Fused d(gamma) + update |
| 71 | %122 | (0)→(384) step(128) | YES | nested | 4 | d(beta) + update |
| 72 | %124 | (0,0)→(384,1536) step(1,256) | YES | 0→8 step 4 | 4 | Weight grad reduce |
| 73 | %126 | (0)→(1536) step(512) | NO | — | — | Bias update |
| 74 | %128 | (0,0)→(1536,384) step(1,128) | YES | 0→8 step 4 | 4 | Weight grad reduce |
| 75 | %130 | (0)→(384) step(128) | NO | — | — | Bias update |
| 76–87 | various | various | mixed | — | — | Block 1 & 2 updates (same pattern) |
| 88 | %160 | in(1) | NO | — | — | Single-block W_vocab update |

---

## 5. K-Tiling Iteration Counts

| Matmul Type | K | kStep | Iterations | Ops per Iteration |
|-------------|---|-------|------------|-------------------|
| FFN-up (fwd/bwd dX_up) | 384 | 32 | 12 | 2 copies + 2 producers + matmul |
| FFN-down (fwd/bwd dX_down) | 1536 | 32 | 48 | 2 copies + 1 producer + matmul |
| dW (bwd weight grad) | 1024 | 32 | 32 | transpose + 2 copies + matmul |
| Output projection | 384 | 32 | 12 | 2 copies + matmul |
| dX (output proj bwd) | 50304 | 32 | 1572 | 2 copies + transpose + matmul |
| dW (output proj bwd) | 8192 | 32 | 256 | transpose + 2 copies + matmul |
| LN mean/var | 384 | 4 | 96 | producer + reduction |
| Softmax max/sum | 50304 | 4 | 12576 | reduction |
| d(gamma)/d(beta) | 8×1024 | 4×4 | 2×256 | reduction (nested) |

---

## 6. Producer Fusion into K-Loop

### What gets fused INSIDE the scf.for:

| Producer Type | Fused? | Shape (inside loop) | Notes |
|---------------|--------|-------------------|-------|
| `linalg.copy` (promoted A) | YES | 1×32×32 or 32×32 | Always fused — this is the key result |
| `linalg.copy` (promoted B) | YES | 1×32×128 or 32×128 | Always fused |
| `linalg.generic` (W broadcast) | YES | 1×32×128 | Sliced along K, fused as producer |
| `linalg.generic` (LN normalize) | YES | 1×32×32 | K-sliced, fused (recomputed per tile) |
| `linalg.generic` (GELU backward) | YES | 1×32×32 | K-sliced, fused |
| `linalg.generic` (embedding) | YES | 1×32×32 | K-sliced, fused |
| `linalg.transpose` | YES | 1×32×32 | Fused inside dW K-loops |
| `tensor.extract_slice` | YES | Various | Sliced along K for operand reads |

### What stays OUTSIDE the scf.for:

| Op | Shape | Reason |
|----|-------|--------|
| `linalg.fill` | 1×32×128 | Accumulator init — must be before loop |
| `linalg.generic` (bias add) | 1×32×128 | Post-matmul epilogue — after loop |
| `linalg.generic` (GELU) | 1×32×128 | Post-matmul epilogue — after loop |
| `linalg.generic` (residual) | 1×32×128 | Post-matmul epilogue — after loop |
| Full-size producers | 1×32×384 | LN/embed computed once outside, K-sliced inside |

---

## 7. Issues and Observations

### 7.1 K-Tiling: 100% Success Rate

Every contraction op (21/21) and every reduction op with non-zero reduction tiles was successfully K-tiled. All 42 promoted `linalg.copy` ops are fused inside K-loops with properly shrunk shapes.

### 7.2 Producer Duplication Pattern

In foralls like %32 (FFN-up mega-fusion), producers like embedding lookup and LN normalize appear TWICE:
- **Outside scf.for**: Full `[1×32×384]` computation (needed for epilogues or downstream foralls)
- **Inside scf.for**: K-sliced `[1×32×32]` computation (fused as producer for copy+matmul chain)

This is correct and expected — the K-sliced version inside the loop avoids reloading the full tensor each iteration.

### 7.3 Single-Block Foralls Still Present

7 single-block foralls remain (1 LN normalize, 6 transposes) — unchanged from previous passes. These will be fixed by the `isContractionPrologue` fix applied earlier.

### 7.4 Nested Reduction Loops for d(gamma)/d(beta)

The d(gamma) and d(beta) reductions have TWO reduction dimensions (batch=8, seq=1024), producing nested `scf.for` loops:
- Outer: `0 to 8 step 4` (2 iterations over batch)
- Inner: `0 to 1024 step 4` (256 iterations over sequence)
- Total: 512 iterations with tiles of `[4×4×128]`

### 7.5 Large K Iteration Count for dX Backward

The dX backward matmul (8192×50304 × 50304^T) has K=50304 with kStep=32, producing **1572 K-iterations**. Each iteration loads 32×32 + 32×128 = 5120 elements from global memory. This is the most memory-bandwidth-intensive kernel.

---

## 8. Summary

The reduction tiling pass successfully:
- Created **~55 `scf.for` loops** across all foralls with reduction dims
- K-tiled all **21 contraction ops** with kStep=32
- **Fused all 42 `linalg.copy` ops** into K-loops, shrinking shapes from `[wgM × K]` to `[wgM × 32]`
- Fused additional producers (broadcasts, LN normalize, GELU backward, transposes) into K-loops
- Maintained correct accumulator patterns (fill before loop, epilogues after loop)

The next pass (Thread tiling, Step 5) will create `scf.forall` with thread mapping for the promoted copies (using their thread-only lowering_config) and for the contraction ops themselves.
