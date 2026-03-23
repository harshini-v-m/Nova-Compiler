# NovaTileAndDistributePass — Detailed IR Analysis Report

**Pass:** `mlir::nova::NovaTileAndDistributePass` (`nova-tile-and-distribute`)
**Source:** `lib/Transforms/LLVMGPU/NovaGPUTileDispatchUsingForall.cpp`
**IR dump location:** `debug2.log` lines 23509–25487
**Date:** 2026-03-21

---

## 1. Pass Overview

The TileAndDistribute pass converts a flat sequence of linalg ops into workgroup-tiled `scf.forall` loops with `#gpu.block` mapping. It operates in 4 phases:

```
Phase 1a:  Tile contraction ops (matmul, batch_matmul) first.
           Consumer fusion (bias, GELU, residual) is immediate if safe,
           deferred if a sibling contraction sits between root and consumer.
Phase 1a-2: Deferred consumer fusion for contraction foralls.
Phase 1b:  Tile remaining ops with lowering_config (reductions, elementwise) in reverse order.
           Consumer fusion enabled since all contractions are now in foralls.
Phase 2:   Wrap any remaining un-distributed compute ops in single-block foralls.
```

### Fusion control rules:
- **Do NOT fuse** pad ops, contraction ops (stay as independent roots), or ops with reduction iterators
- **DO fuse** all-parallel producers (weight broadcasts, LN apply, bias add, GELU, residual connections)
- Consumer fusion also pulls in the new consumers' producers transitively

---

## 2. Summary Statistics

| Metric | Count |
|--------|-------|
| Total `scf.forall` blocks | 93 (including 4 nested scatter patterns) |
| Top-level foralls | ~82 unique workgroup dispatches |
| Foralls with fused ops (>1 compute op) | ~26 |
| Single-op foralls | ~56 |
| Single-block foralls (in(1)) | 8 |
| Nested foralls (scatter patterns) | 4 |

---

## 3. Complete Forall Inventory

### Forward Pass — Embedding + LN0

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 1 | %28:2 | (0,0)→(8,1024) step(1,256) | 8×4=**32** | embedding_lookup + fill + **LN0_mean_reduction** (root) | LN0 mean with embedding fused as producer |
| 2 | %29:2 | (0,0)→(8,1024) step(1,256) | 8×4=**32** | embedding_lookup (recomputed) + **LN0_var_reduction** (root) | LN0 variance with embedding recomputed |

**Note:** Embedding lookup is fused as producer into both LN reductions. The embedding is recomputed in forall #2 since it was already consumed by #1.

### Forward Pass — Transformer Block 0

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 3 | %32:5 | (0,0,0)→(8,1024,1536) step(1,32,128) | 8×32×12=**3072** | embedding + LN_normalize + W_up_broadcast + fill + **batch_matmul_up** + bias_add + GELU | **MEGA-FUSION**: 7 ops fused into FFN-up matmul |
| 4 | %34:2 | (0,0,0)→(8,1024,384) step(1,32,128) | 8×32×3=**768** | GELU_out_slice + W_down_broadcast + fill + **batch_matmul_down** + residual_bias_add | FFN-down matmul + residual epilogue |

**5 outputs from %32**: batch_matmul result, embedding (for reuse), LN apply (for reuse), bias_add result, GELU result
**2 outputs from %34**: batch_matmul+residual result, input pass-through

### Forward Pass — Transformer Block 1

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 5 | %35 | (0,0)→(8,1024) step(1,256) | 8×4=**32** | fill + **LN1_mean** | Standalone reduction |
| 6 | %36 | (0,0)→(8,1024) step(1,256) | 8×4=**32** | fill + **LN1_var** | Standalone reduction |
| 7 | %37:4 | (0,0,0)→(8,1024,1536) step(1,32,128) | 8×32×12=**3072** | LN_normalize + W_up_broadcast + fill + **batch_matmul_up** + bias_add + GELU | MEGA-FUSION (6 ops) |
| 8 | %38:2 | (0,0,0)→(8,1024,384) step(1,32,128) | 8×32×3=**768** | W_down_broadcast + fill + **batch_matmul_down** + residual_bias_add | Fused matmul + residual |

### Forward Pass — Transformer Block 2

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 9 | %39 | (0,0)→(8,1024) step(1,256) | **32** | **LN2_mean** | Standalone |
| 10 | %40 | (0,0)→(8,1024) step(1,256) | **32** | **LN2_var** | Standalone |
| 11 | %41:4 | (0,0,0)→(8,1024,1536) step(1,32,128) | **3072** | MEGA-FUSION: LN + FFN-up + bias + GELU | Same pattern as block 0/1 |
| 12 | %42:2 | (0,0,0)→(8,1024,384) step(1,32,128) | **768** | FFN-down + residual | Same pattern |

### Forward Pass — Final LN + Output Projection

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 13 | %43 | (0,0)→(8,1024) step(1,256) | **32** | **Final_LN_mean** | Standalone |
| 14 | %44 | (0,0)→(8,1024) step(1,256) | **32** | **Final_LN_var** | Standalone |
| 15 | %46 | in(1) | **1** | LN_normalize (8×1024×384 elementwise) | **BUG: single-block forall** |
| 16 | %48 | (0,0)→(8192,50304) step(32,128) | 256×393=**100,608** | fill + **linalg.matmul** | Output projection (forward) |

### Forward Pass — Softmax + Cross-Entropy Loss

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 17 | %49 | (0,0)→(8,1024) step(1,256) | **32** | fill + **max_reduction** | Softmax: max(logits) |
| 18 | %50 | (0,0)→(8,1024) step(1,256) | **32** | fill + **exp_sum_reduction** | Softmax: sum(exp(x-max)) |
| 19 | %52 | in(8,1024,8) | **65,536** | **CE_loss_generic** (with tensor.extract) | SCE loss per-position |
| 20 | - | in(64) | **64** | scf.for loop (partial reduce) | Loss: partial sum |
| 21 | - | in(1) | **1** | final reduce + divide | Loss: mean |

### Backward Pass — Softmax Grad + SCE Backward

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 22 | %55:2 | (0,0)→(8,1024) step(1,256) | **32** | **Fused**: exp_generic (producer) + exp_sum_reduction (root) | Softmax backward reduction |
| 23 | %56 | (0,0,0)→(8,1024,50304) step(1,1,128) | 8×1024×393=**3,175,424** | **Fused**: exp + divide (normalize) | Softmax normalize backward |

### Backward Pass — Index/Scatter Ops

| # | SSA | Bounds | Grid | Semantic |
|---|-----|--------|------|----------|
| 24 | %58 | in(8) | **8** | Label cast i16→i64 |
| 25 | %60 | (0)→(8192) step(1024) | **8** | Index offset arithmetic |
| 26 | nested | in(8){in(1024)} | **8+1024** | Atomic scatter (one_hot subtract) |
| 27 | %63 | (0,0,0)→(8,1024,50304) step(1,1,128) | **3,175,424** | Grad scale multiply |

### Backward Pass — Output Projection

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 28 | %66 | (0,0)→(8192,384) step(32,128) | 256×3=**768** | transpose_fused + fill + **matmul** (dX) | dX = dY × W_vocab^T |
| 29 | %68 | (0,0)→(384,50304) step(32,128) | 12×393=**4,716** | transpose_fused + fill + **matmul** (dW) | dW_vocab = X^T × dY |

### Backward Pass — Final LN Backward

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 30 | %69:3 | (0,0)→(8,1024) step(1,256) | **32** | **Fused 3 ops**: LN_bwd_generic (2 outputs) + reduction#1 | LN bwd: sum(grad*(x-mean)) |
| 31 | %70 | (0,0)→(8,1024) step(1,256) | **32** | LN_bwd_generic (recomputed) + reduction#2 | LN bwd: sum(grad) |
| 32 | %72 | (0)→(384) step(128) | **3** | d(gamma) reduction | Weight grad |
| 33 | %73 | (0)→(384) step(128) | **3** | d(beta) reduction | Bias grad |

### Backward Pass — FFN Block 3 (pattern repeats for blocks 2, 1)

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 34 | %77 | in(1) | **1** | W_up transpose/reshape | **BUG: single-block** |
| 35 | %78:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | **3072** | **Fused**: LN_bwd_normalize + W_up_broadcast + fill + batch_matmul_up | Backward FFN dX_up |
| 36 | %79 | (0,0,0)→(8,1536,384) step(1,32,128) | 8×48×3=**1152** | fill + **batch_matmul** (dW_down) | Weight grad |
| 37 | %81 | (0)→(384) step(128) | **3** | d(bias_down) reduction | Bias grad |
| 38 | %84 | in(1) | **1** | W_down transpose/reshape | **BUG: single-block** |
| 39 | %85:2 | (0,0,0)→(8,1024,384) step(1,32,128) | **768** | **Fused**: GELU_bwd + W_down_broadcast + fill + batch_matmul_down | Backward FFN dX_down |
| 40 | %86 | (0,0,0)→(8,384,1536) step(1,32,128) | 8×12×12=**1152** | fill + **batch_matmul** (dW_up) | Weight grad |
| 41 | %88 | (0)→(1536) step(512) | **3** | d(bias_up) reduction | Bias grad |

### Backward Pass — LN Block 3

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 42 | %89:3 | (0,0)→(8,1024) step(1,256) | **32** | LN_bwd_generic + reduction | sum(grad*(x-mean)) |
| 43 | %90 | (0,0)→(8,1024) step(1,256) | **32** | LN_bwd_generic (recompute) + reduction | sum(grad) |

### Backward Pass — FFN Block 2 (same structure as Block 3)

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 44 | %92 | in(1) | **1** | W_up transpose | single-block |
| 45 | %93:2 | →(8,1024,1536) step(1,32,128) | **3072** | LN_bwd + W_up_bcast + matmul_up | Fused backward FFN |
| 46 | %94 | →(8,1536,384) step(1,32,128) | **1152** | matmul (dW_down) | Weight grad |
| 47 | %95 | →(384) step(128) | **3** | d(bias_down) | |
| 48 | %97 | in(1) | **1** | W_down transpose | single-block |
| 49 | %98:2 | →(8,1024,384) step(1,32,128) | **768** | GELU_bwd + matmul_down | Fused backward FFN |
| 50 | %99 | →(8,384,1536) step(1,32,128) | **1152** | matmul (dW_up) | Weight grad |
| 51 | %100 | →(1536) step(512) | **3** | d(bias_up) | |

### Backward Pass — LN Block 2

| # | SSA | Bounds | Grid | Semantic |
|---|-----|--------|------|----------|
| 52 | %101:3 | →(8,1024) step(1,256) | **32** | LN bwd reduction #1 |
| 53 | %102 | →(8,1024) step(1,256) | **32** | LN bwd reduction #2 |

### Backward Pass — FFN Block 1 (same structure)

| # | SSA | Bounds | Grid | Semantic |
|---|-----|--------|------|----------|
| 54 | %104 | in(1) | **1** | W_up transpose |
| 55 | %105:2 | →(8,1024,1536) step(1,32,128) | **3072** | LN_bwd + matmul_up |
| 56 | %106 | →(8,1536,384) step(1,32,128) | **1152** | matmul (dW_down) |
| 57 | %107 | →(384) step(128) | **3** | d(bias_down) |
| 58 | %109 | in(1) | **1** | W_down transpose |
| 59 | %110:2 | →(8,1024,384) step(1,32,128) | **768** | GELU_bwd + matmul_down |
| 60 | %111 | →(8,384,1536) step(1,32,128) | **1152** | matmul (dW_up) |
| 61 | %112 | →(1536) step(512) | **3** | d(bias_up) |

### Backward Pass — LN Block 0 (Embedding Layer)

| # | SSA | Bounds | Grid | Semantic |
|---|-----|--------|------|----------|
| 62 | %113:3 | →(8,1024) step(1,256) | **32** | LN bwd reduction #1 |
| 63 | %114 | →(8,1024) step(1,256) | **32** | LN bwd reduction #2 |

### Backward Pass — Embedding Gradient

| # | SSA | Bounds | Grid | Semantic |
|---|-----|--------|------|----------|
| 64 | %116:3 | (0,0)→(1024,384) step(1,128) | 1024×3=**3072** | Position embedding grad reduction + LN bwd dx fused |
| 65 | nested | in(1024){in(384)} | 1024+384 | Atomic scatter: pos embed grad |
| 66 | %120 | in(1) | **1** | Init zero tensor (50304×384) |
| 67 | nested | in(8192){in(384)} | 8192+384 | Atomic scatter: token embed grad |

### Optimizer — Weight/Bias Updates

| # | SSA | Bounds | Grid | Fused Ops | Semantic |
|---|-----|--------|------|-----------|----------|
| 68 | %124 | (0,0)→(50304,384) step(1,128) | 50304×3=**150,912** | **Fused**: old_weight read + grad_add | Token embedding update |
| 69 | %126 | (0,0)→(1024,384) step(1,128) | 1024×3=**3072** | **Fused**: old_weight + grad_add | Position embedding update |
| 70 | %128:2 | (0)→(384) step(128) | **3** | **Fused**: gamma + beta update (LN0) | 2-output forall |
| 71 | %130:2 | (0)→(384) step(128) | **3** | d(gamma) + d(beta) accumulate (LN0) | |
| 72 | %132:2 | (0,0)→(384,1536) step(1,256) | 384×6=**2304** | **Fused**: W_up + W_up_grad_add | Block 0 FFN-up weight update |
| 73 | %134 | (0)→(1536) step(512) | **3** | b_up update | Block 0 FFN-up bias update |
| 74 | %136:2 | (0,0)→(1536,384) step(1,128) | 1536×3=**4608** | W_down + grad_add | Block 0 FFN-down weight update |
| 75 | %138 | (0)→(384) step(128) | **3** | b_down update | |
| 76-81 | %140-%154 | (varies) | (varies) | Same pattern | Block 1 updates |
| 82-87 | %156-%166 | (varies) | (varies) | Same pattern | Block 2 updates |
| 88 | %169 | in(1) | **1** | W_vocab grad zero tensor | Placeholder for scatter |

---

## 4. Fusion Analysis

### Successful Fusions

**Forward FFN mega-fusions** (foralls #3, #7, #11):
```
LN_normalize (producer) → W_up_broadcast (producer) → fill (producer) →
batch_matmul (ROOT) → bias_add (epilogue) → GELU (epilogue)
```
- 6-7 ops fused into a single 3072-block forall
- LN normalize is all-parallel → fusable as producer
- bias_add and GELU are all-parallel → fusable as epilogues
- Excellent: no extra kernel launches for the entire LN→matmul→GELU chain

**Forward FFN-down fusions** (foralls #4, #8, #12):
```
W_down_broadcast (producer) → fill (producer) → batch_matmul (ROOT) → residual+bias_add (epilogue)
```
- 4-5 ops fused

**Backward FFN fusions** (foralls #35, #39, #45, #49, #55, #59):
```
LN_bwd_normalize / GELU_bwd (producer) → W_broadcast (producer) → batch_matmul (ROOT)
```
- 3-4 ops fused per backward matmul

**Softmax backward fusion** (#22):
```
exp_generic (producer) → exp_sum_reduction (ROOT)
```

**Optimizer weight update fusions** (foralls #68-#87):
```
old_weight_read + new_grad (producers) → weight_add (ROOT)
```

### NOT Fused (Separate Foralls)

1. **LN mean and variance** are always in separate foralls — cannot share the input read because each is an independent reduction root

2. **LN backward dual-output generic** is recomputed between foralls #30/#31 — the 2-output elementwise (dy*gamma, dy*gamma*xhat) is computed once for each downstream reduction

3. **d(gamma) and d(beta)** are separate foralls (#32, #33) — two independent reductions over the same input

---

## 5. Issues Found

### 5.1 CRITICAL: Final LN Normalize on Single Workgroup (#15)

```
%46 = scf.forall (%arg51) in (1) shared_outs(%arg52 = %45) -> (tensor<8x1024x384xf32>)
```

The final LayerNorm normalize (8×1024×384 = 3.1M elements) runs on **1 workgroup (1 thread block)**. This is a full elementwise operation that should be distributed across thousands of blocks.

**Root cause:** The LN normalize generic (#38 in strategy report, `%67`) has **no lowering_config** — it's classified as a contraction prologue (feeds into the output projection matmul). But the output projection matmul is `linalg.matmul` operating on **reshaped** 8192×384 (collapsed from 8×1024×384), not directly on the 3D tensor. So the fusion fails — the LN normalize doesn't get pulled into the matmul's forall because the shapes don't match after the reshape.

The pass falls through to Phase 2: "wrap remaining un-distributed compute ops in single-block forall."

**Impact:** The final LN normalize before the output projection is serialized to 1 thread block. For 3.1M elements at 256 threads, this takes ~12K cycles vs ~12 cycles if distributed across 3072 blocks.

### 5.2 Weight Transpose on Single Workgroup (6 instances)

```
%77 = scf.forall (%arg51) in (1) shared_outs(...) -> (tensor<1x384x1536xf32>)
%84 = scf.forall (%arg51) in (1) shared_outs(...) -> (tensor<1x1536x384xf32>)
%92, %97, %104, %109 — same pattern
```

Each backward FFN block has 2 weight transpose/reshape ops running on 1 workgroup. While these are small (384×1536 = 590K elements), running on 1 block is still 2300x slower than optimal.

**Root cause:** Same as above — these generics have no lowering_config (classified as contraction prologues) but don't get fused into the matmul forall due to shape mismatches.

### 5.3 Embedding Lookup Recomputation

The embedding lookup (token + position gather) is fused as a producer into:
- LN0 mean reduction (forall #1)
- LN0 variance reduction (forall #2)
- FFN-up matmul (forall #3)

It's **recomputed** in each forall rather than computed once and shared. For the matmul forall (#3), the embedding is computed for each [1,32,128] tile, which is fine since it's a simple gather. But it's a pattern to watch for more expensive producers.

### 5.4 LN Backward Dual-Output Recomputation

Each LN backward has a 2-output generic that computes both `dy*gamma` and `dy*gamma*(x-mean)/std`. This generic is computed twice:
- Once fused into reduction of output #1 (forall #30)
- Once fused into reduction of output #2 (forall #31)

This doubles the compute for LN backward. IREE solves this with multi-result tiling, but Nova's current tiling infrastructure processes one reduction root at a time.

### 5.5 Very Large Grid for Softmax/Grad Ops

```
%56 = scf.forall → (8,1024,50304) step(1,1,128)  — Grid: 3,175,424 blocks
%63 = scf.forall → (8,1024,50304) step(1,1,128)  — Grid: 3,175,424 blocks
```

These softmax elementwise ops launch 3.1M thread blocks with only 64 threads each. The per-block work is tiny (128 elements × 1 row). Could benefit from larger tiles to reduce launch overhead.

---

## 6. Grid Dimension Summary

| Category | Grid Size | Tile | Blocks | Threads/Block* |
|----------|-----------|------|--------|----------------|
| LN reductions | [1,256] | 8×4 | 32 | 256 |
| FFN-up matmul (fused) | [1,32,128] | 8×32×12 | 3,072 | 256 |
| FFN-down matmul (fused) | [1,32,128] | 8×32×3 | 768 | 256 |
| LM head matmul | [32,128] | 256×393 | 100,608 | 256 |
| LM head dX matmul | [32,128] | 256×3 | 768 | 256 |
| LM head dW matmul | [32,128] | 12×393 | 4,716 | 256 |
| Backward dW batch_matmul | [1,32,128] | varies | 1,152 | 256 |
| Softmax elementwise | [1,1,128] | 8×1024×393 | 3,175,424 | 64 |
| CE loss | [1,1,1] | 8×1024×8 | 65,536 | varies |
| d(gamma)/d(beta) | [128] | 3 | 3 | 64 |
| d(bias) 1536 | [512] | 3 | 3 | 256 |
| Weight updates | varies | varies | varies | 64-256 |
| **Single-block BUGs** | [full] | 1 | **1** | 256 |

*Thread count determined by subsequent thread tiling pass, shown here for reference.

---

## 7. Forall Output Counts

The pass creates multi-result foralls when fusion yields values consumed outside:

| Outputs | Count | Typical Use |
|---------|-------|-------------|
| 1 | ~56 | Standalone reductions, matmul dW, elementwise |
| 2 | ~16 | Matmul + residual, weight+bias update pairs |
| 3 | 5 | LN backward (grad + 2 intermediates) |
| 4 | 3 | FFN-up fused (matmul + LN + bias + GELU results) |
| 5 | 1 | Block 0 FFN-up mega-fusion (includes embedding pass-through) |

---

## 8. Source Code Key Points

**File:** `NovaGPUTileDispatchUsingForall.cpp` (725 lines)

### Phase 1a Contraction Safety Check (lines 452-488)
The pass checks if consumer fusion is safe for each contraction by walking forward in the block: if an untiled sibling contraction appears before any consumer, consumer fusion is deferred to Phase 1a-2. This prevents MLIR's `tileAndFuseConsumerOfSlices` from accidentally dragging sibling contractions into the forall.

### Fusion Control Function (lines 333-360)
```cpp
// Don't fuse: pad ops, contraction ops, ops with reduction iterators
// Do fuse: all-parallel producers (broadcasts, LN apply, bias, GELU)
```

### Full-Tile Optimization (lines 189-222)
When `staticLoopSize == tileSize`, the tile dimension is zeroed out to avoid single-trip forall loops. At least one non-zero tile is kept. Exception: ops with reduction iterators always stay in foralls.

### Phase 2 Single-Block Wrapping (lines 613-676)
Remaining un-distributed compute ops get individually wrapped in `scf.forall(0 to 1)` with GPU block mapping. This is the source of the single-block performance bugs.
