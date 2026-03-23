# NovaGPUPadOperandsPass — Detailed IR Analysis Report

**Pass:** `mlir::nova::NovaGPUPadOperandsPass` (`nova-gpu-pad-operands`)
**Source:** `lib/Transforms/LLVMGPU/NovaGPUPadOperands.cpp`
**IR dump location:** `debug2.log` lines 30376–32307
**Pipeline position:** Step 2 (after TileAndDistribute + canonicalize + CSE, before PromoteMatmulOperands)
**Date:** 2026-03-21

---

## 1. Pass Overview

The PadOperands pass reads `padding` values from each linalg op's `lowering_config` and pads operands to static multiples of those tile sizes using `linalg::rewriteAsPaddedOp()`. Zero-padding is used (safe for matmul/reduction). A cleanup pattern (`FoldFillIntoPad`) folds `fill+pad` sequences to avoid materializing large padded tensors.

### Pipeline context:
```
Step 0: SelectLoweringStrategy  → stamps lowering_config (with padding list)
Step 1: TileAndDistribute       → tiles into scf.forall, fuses producers/consumers
Step 2: PadOperands             → pads operands to tile-aligned sizes  ← THIS PASS
Step 3: PromoteMatmulOperands   → inserts shared memory copies
Step 4: K-tiling (reduction)    → tiles along K dimension
```

---

## 2. Key Finding: Zero tensor.pad Ops Inserted

**There are ZERO `tensor.pad` operations in the output IR.** All operand dimensions are already naturally aligned with the padding tile sizes, so `linalg::rewriteAsPaddedOp()` found nothing to pad.

### Why all dimensions are aligned:

| Dimension | Size | Pad-to | Divisible? |
|-----------|------|--------|------------|
| Batch | 8 | 1 | 8/1 = 8 |
| M (matmul) | 32 (tiled) | 32 | 32/32 = 1 |
| N (matmul) | 128 (tiled) | 128 | 128/128 = 1 |
| K = 384 | 384 | 32 | 384/32 = 12 |
| K = 1536 | 1536 | 32 | 1536/32 = 48 |
| K = 1024 | 1024 | 32 | 1024/32 = 32 |
| K = 8192 | 8192 | 32 | 8192/32 = 256 |
| K = 50304 | 50304 | 32 | 50304/32 = 1572 |
| Rows (LN) | 256 (tiled) | 256 | 256/256 = 1 |
| Cols (LN) | 384 | 4 | 384/4 = 96 |
| Vocab | 50304 | 128 | 50304/128 = 393 |

The model dimensions (384, 1536, 1024, 50304) are all multiples of the tile sizes (32, 128, 4). This is typical for transformer models with power-of-2-friendly hidden dimensions.

---

## 3. Complete Forall Inventory with Padding Analysis

### Forward Pass — Embedding + LN0

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 1 | %28 | (0,0)→(8,1024) step(1,256) | embedding_lookup(`[1,1,128]`) + fill + LN0_mean(`[1,256,4]`) | Aligned | No pad needed |
| 2 | %29 | (0,0)→(8,1024) step(1,256) | embedding_lookup(`[1,1,128]`) + fill + LN0_var(`[1,256,4]`) | Aligned | No pad needed |

### Forward Pass — Transformer Block 0

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 3 | %32:4 | (0,0,0)→(8,1024,1536) step(1,32,128) | embedding(config) + LN_normalize(no config) + W_broadcast(no config) + fill + **batch_matmul**(`[1,32,128,32]`, promo=[0,1]) + bias(no config) + GELU(no config) | K=384, 384%32=0 | No pad needed |
| 4 | %34 | (0,0,0)→(8,1024,384) step(1,32,128) | W_broadcast(no config) + fill + **batch_matmul**(`[1,32,128,32]`, promo=[0,1]) + residual(no config) | K=1536, 1536%32=0 | No pad needed |

### Forward Pass — Transformer Block 1

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 5 | %35 | (0,0)→(8,1024) step(1,256) | fill + LN1_mean(`[1,256,4]`) | 384%4=0 | No pad needed |
| 6 | %36 | (0,0)→(8,1024) step(1,256) | fill + LN1_var(`[1,256,4]`) | 384%4=0 | No pad needed |
| 7 | %37:3 | (0,0,0)→(8,1024,1536) step(1,32,128) | LN_norm(no config) + W_bcast(no config) + fill + **batch_matmul**(`[1,32,128,32]`) + bias(no config) + GELU(no config) | Aligned | No pad needed |
| 8 | %38 | (0,0,0)→(8,1024,384) step(1,32,128) | W_bcast(no config) + fill + **batch_matmul**(`[1,32,128,32]`) + residual(no config) | Aligned | No pad needed |

### Forward Pass — Transformer Block 2

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 9 | %39 | (0,0)→(8,1024) step(1,256) | fill + LN2_mean(`[1,256,4]`) | Aligned | No pad needed |
| 10 | %40 | (0,0)→(8,1024) step(1,256) | fill + LN2_var(`[1,256,4]`) | Aligned | No pad needed |
| 11 | %41:3 | (0,0,0)→(8,1024,1536) step(1,32,128) | LN_norm + W_bcast + fill + **batch_matmul**(`[1,32,128,32]`) + bias + GELU | Aligned | No pad needed |
| 12 | %42 | (0,0,0)→(8,1024,384) step(1,32,128) | W_bcast + fill + **batch_matmul**(`[1,32,128,32]`) + residual | Aligned | No pad needed |

### Forward Pass — Final LN + Output Projection

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 13 | %43 | (0,0)→(8,1024) step(1,256) | fill + Final_LN_mean(`[1,256,4]`) | Aligned | No pad needed |
| 14 | %44 | (0,0)→(8,1024) step(1,256) | fill + Final_LN_var(`[1,256,4]`) | Aligned | No pad needed |
| 15 | %45 | in(1) | LN_normalize (no config) | **SINGLE BLOCK BUG** | N/A |
| 16 | %47 | (0,0)→(8192,50304) step(32,128) | fill + **matmul**(`[32,128,32]`, promo=[0,1]) | K=384, 384%32=0 | No pad needed |

### Forward Pass — Softmax + Cross-Entropy Loss

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 17 | %48 | (0,0)→(8,1024) step(1,256) | fill + max_reduction(`[1,256,4]`) | 50304%4=0 | No pad needed |
| 18 | %49 | (0,0)→(8,1024) step(1,256) | fill + exp_sum_reduction(`[1,256,4]`) | 50304%4=0 | No pad needed |
| 19 | %51 | in(8,1024,8) | CE_loss(`[1,1,1,1024]`) | 1024%1024=0 | No pad needed |
| 20 | — | in(64) | scf.for partial reduce | N/A | N/A |
| 21 | — | in(1) | final reduce + divide | N/A | N/A |
| 22 | %54 | (0,0)→(8,1024) step(1,256) | exp(`[1,1,128]`) + fill + sum_reduction(`[1,256,4]`) | Aligned | No pad needed |
| 23 | %55 | (0,0,0)→(8,1024,50304) step(1,1,128) | exp + divide (`[1,1,128]`) | 50304%128=0 | No pad needed |

### Backward Pass — Index/Scatter + Output Projection

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 24 | %57 | in(8) | label_cast(`[1,1024]`) | Aligned | No pad needed |
| 25 | %59 | (0)→(8192) step(1024) | index_arith(`[1024]`) | Aligned | No pad needed |
| 26 | nested | in(8){in(1024)} | atomic scatter | N/A | N/A |
| 27 | %62 | (0,0,0)→(8,1024,50304) step(1,1,128) | grad_scale(`[1,1,128]`) | Aligned | No pad needed |
| 28 | %65 | (0,0)→(8192,384) step(32,128) | transpose + fill + **matmul**(dX, `[32,128,32]`) | K=50304, 50304%32=0 | No pad needed |
| 29 | %67 | (0,0)→(384,50304) step(32,128) | transpose + fill + **matmul**(dW, `[32,128,32]`) | K=8192, 8192%32=0 | No pad needed |

### Backward Pass — Final LN

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 30 | %68 | (0,0)→(8,1024) step(1,256) | LN_bwd_generic(no config) + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 31 | %69 | (0,0)→(8,1024) step(1,256) | LN_bwd_generic(no config) + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 32 | %71 | (0)→(384) step(128) | fill + d(gamma)(`[128,4,4]`) | 8%4=0, 1024%4=0 | No pad needed |
| 33 | %72 | (0)→(384) step(128) | fill + d(beta)(`[128,4,4]`) | Aligned | No pad needed |

### Backward Pass — FFN Block 3

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 34 | %75 | in(1) | transpose (no config) | **SINGLE BLOCK BUG** | N/A |
| 35 | %76:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | LN_bwd(no config) + W_bcast(no config) + fill + **batch_matmul**(dX_up, `[1,32,128,32]`) | K=384, aligned | No pad needed |
| 36 | %77 | (0,0,0)→(8,1536,384) step(1,32,128) | transpose + fill + **batch_matmul**(dW_down, `[1,32,128,32]`) | K=1024, aligned | No pad needed |
| 37 | %79 | (0)→(384) step(128) | fill + d(bias_down)(`[128,4,4]`) | Aligned | No pad needed |
| 38 | %81 | in(1) | transpose (no config) | **SINGLE BLOCK BUG** | N/A |
| 39 | %82:2 | (0,0,0)→(8,1024,384) step(1,32,128) | GELU_bwd(no config) + W_bcast(no config) + fill + **batch_matmul**(dX_down, `[1,32,128,32]`) | K=1536, aligned | No pad needed |
| 40 | %83 | (0,0,0)→(8,384,1536) step(1,32,128) | transpose + fill + **batch_matmul**(dW_up, `[1,32,128,32]`) | K=1024, aligned | No pad needed |
| 41 | %85 | (0)→(1536) step(512) | fill + d(bias_up)(`[512,4,4]`) | Aligned | No pad needed |

### Backward Pass — LN Block 3 + FFN Block 2

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 42 | %86 | (0,0)→(8,1024) step(1,256) | LN_bwd(no config) + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 43 | %87 | (0,0)→(8,1024) step(1,256) | LN_bwd(no config) + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 44 | %88 | in(1) | transpose (no config) | **SINGLE BLOCK** | N/A |
| 45 | %89:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | LN_bwd + W_bcast + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 46 | %90 | (0,0,0)→(8,1536,384) step(1,32,128) | transpose + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 47 | %91 | (0)→(384) step(128) | fill + d(bias_down)(`[128,4,4]`) | Aligned | No pad needed |
| 48 | %92 | in(1) | transpose (no config) | **SINGLE BLOCK** | N/A |
| 49 | %93:2 | (0,0,0)→(8,1024,384) step(1,32,128) | GELU_bwd + W_bcast + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 50 | %94 | (0,0,0)→(8,384,1536) step(1,32,128) | transpose + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 51 | %95 | (0)→(1536) step(512) | fill + d(bias_up)(`[512,4,4]`) | Aligned | No pad needed |

### Backward Pass — LN Block 2 + FFN Block 1

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 52 | %96 | (0,0)→(8,1024) step(1,256) | LN_bwd + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 53 | %97 | (0,0)→(8,1024) step(1,256) | LN_bwd + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 54 | %98 | in(1) | transpose (no config) | **SINGLE BLOCK** | N/A |
| 55 | %99:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | LN_bwd + W_bcast + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 56 | %100 | (0,0,0)→(8,1536,384) step(1,32,128) | transpose + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 57 | %101 | (0)→(384) step(128) | fill + d(bias_down)(`[128,4,4]`) | Aligned | No pad needed |
| 58 | %102 | in(1) | transpose (no config) | **SINGLE BLOCK** | N/A |
| 59 | %103:2 | (0,0,0)→(8,1024,384) step(1,32,128) | GELU_bwd + W_bcast + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 60 | %104 | (0,0,0)→(8,384,1536) step(1,32,128) | transpose + fill + **batch_matmul**(`[1,32,128,32]`) | Aligned | No pad needed |
| 61 | %105 | (0)→(1536) step(512) | fill + d(bias_up)(`[512,4,4]`) | Aligned | No pad needed |

### Backward Pass — LN Block 0 + Embedding Gradients

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 62 | %106 | (0,0)→(8,1024) step(1,256) | LN_bwd + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 63 | %107 | (0,0)→(8,1024) step(1,256) | LN_bwd + fill + reduction(`[1,256,4]`) | Aligned | No pad needed |
| 64 | %109 | (0,0)→(1024,384) step(1,128) | LN_bwd(no config) + fill + pos_embed_grad(`[1,128,4]`) | Aligned | No pad needed |
| 65 | nested | in(1024){in(384)} | atomic scatter (pos embed) | N/A | N/A |
| 66 | %112 | in(1) | fill (token embed grad init) | N/A | N/A |
| 67 | nested | in(8192){in(384)} | atomic scatter (token embed) | N/A | N/A |

### Optimizer — Weight/Bias Updates

| # | SSA | Bounds | Ops Inside | Padding Config | Padded? |
|---|-----|--------|------------|---------------|---------|
| 68 | %116 | (0,0)→(50304,384) step(1,128) | elem_add(`[1,128]`) | Aligned | No pad needed |
| 69 | %118 | (0,0)→(1024,384) step(1,128) | elem_add(`[1,128]`) | Aligned | No pad needed |
| 70 | %120 | (0)→(384) step(128) | fill + d(gamma) reduction(`[128,4,4]`) + elem_add(no config) | Aligned | No pad needed |
| 71 | %122 | (0)→(384) step(128) | fill + d(beta) reduction(`[128,4,4]`) + elem_add(no config) | Aligned | No pad needed |
| 72 | %124 | (0,0)→(384,1536) step(1,256) | reduction(`[1,256,4]`) + elem_add(no config) | Aligned | No pad needed |
| 73 | %126 | (0)→(1536) step(512) | elem_add(`[512]`) | Aligned | No pad needed |
| 74 | %128 | (0,0)→(1536,384) step(1,128) | reduction(`[1,128,4]`) + elem_add(no config) | Aligned | No pad needed |
| 75 | %130 | (0)→(384) step(128) | elem_add(`[128]`) | Aligned | No pad needed |
| 76–87 | %132–%158 | (varies) | Block 1 & 2 weight updates (same patterns) | Aligned | No pad needed |
| 88 | %160 | in(1) | W_vocab update (no config) | **SINGLE BLOCK** | N/A |

---

## 4. Distinct Padding Profiles

### Profile A: Batch Matmul — 18 ops
```
padding = [1, 32, 128, 32]      # batch=1, M=32, N=128, K=32
promoted_operands = [0, 1]
```

### Profile B: 2D Matmul — 3 ops
```
padding = [32, 128, 32]          # M=32, N=128, K=32
promoted_operands = [0, 1]
```

### Profile C: Row-Reduction (LN mean/var) — 22 ops
```
padding = [1, 256, 4]            # batch=1, rows=256, cols=4
promoted_operands = []
```

### Profile D: LN Backward d(gamma)/d(beta) — 11 ops
```
padding = [128, 4, 4]            # parallel=128, red1=4, red2=4
promoted_operands = []
```

### Profile E: Elementwise 3D — 7 ops
```
padding = [1, 1, 128]            # innermost dim vectorized
promoted_operands = []
```

### Profile F: Batch Reduction (weight grad) — 4 ops
```
padding = [1, 128, 4]            # batch=1, parallel=128, red=4
promoted_operands = []
```

### Profile G: Bias Backward 1536-dim — 3 ops
```
padding = [512, 4, 4]
promoted_operands = []
```

### Profile H-N: Various optimizer/update patterns
```
padding = [512], [128], [1, 128], [1, 256, 4], [1024], [1, 1024], [1, 1, 1, 1024]
```

---

## 5. Ops WITHOUT lowering_config (Never Padded)

These ops have NO `lowering_config` and are never candidates for padding. They are fused producers/consumers inside foralls:

| Op Type | Role | Count | Example |
|---------|------|-------|---------|
| `linalg.generic` (LN normalize) | Prologue (fused producer) | 4 | 5-input elementwise: gamma*(x-mean)/sqrt(var+eps)+beta |
| `linalg.generic` (W broadcast) | Prologue (fused producer) | 12 | 2D→3D weight expansion |
| `linalg.generic` (bias add) | Epilogue (fused consumer) | 3 | + b_up after batch_matmul |
| `linalg.generic` (GELU) | Epilogue (fused consumer) | 3 | GELU activation |
| `linalg.generic` (residual) | Epilogue (fused consumer) | 3 | out + x + b_down |
| `linalg.generic` (GELU bwd) | Prologue (fused producer) | 3 | grad * gelu'(x) |
| `linalg.generic` (LN bwd dx) | Prologue (fused producer) | 4 | 7/9-input backward LN |
| `linalg.transpose` | Prologue (fused producer) | 6+ | Weight transpose for backward |
| `linalg.fill` | Init (fused producer) | ~62 | Zero-init accumulators |

---

## 6. Issues and Observations

### 6.1 No Actual Padding Needed (All Dims Aligned)

The pass is a NO-OP for this model configuration. All problem dimensions (384, 1536, 1024, 8192, 50304) are divisible by all tile sizes (32, 128, 4, 256, 512, 1024). This is expected for well-designed transformer architectures.

**When padding WOULD be needed:** If hidden_dim were 400 instead of 384, then 400 % 32 = 16 (remainder), and K-tiles would need padding. Similarly, vocab_size=50257 (GPT-2 original) would need padding since 50257 % 128 = 97.

### 6.2 Single-Block Foralls Persist

The following ops from TileAndDistribute Phase 2 remain wrapped in `scf.forall(0 to 1)`:

| # | SSA | Op | Elements | Issue |
|---|-----|----|----------|-------|
| 15 | %45 | Final LN normalize | 3,145,728 | **CRITICAL** — will be fixed by `isContractionPrologue` fix |
| 34 | %75 | W_up transpose (block 3) | 589,824 | Will be fixed by same fix |
| 38 | %81 | W_down transpose (block 3) | 589,824 | Same |
| 44 | %88 | W_up transpose (block 2) | 589,824 | Same |
| 48 | %92 | W_down transpose (block 2) | 589,824 | Same |
| 54 | %98 | W_up transpose (block 1) | 589,824 | Same |
| 58 | %102 | W_down transpose (block 1) | 589,824 | Same |
| 88 | %160 | W_vocab update | 19,316,736 | Missing config for large 2D update |

**Total serialized elements: ~25.5M** — all running on 1 thread block each.

### 6.3 FoldFillIntoPad Pattern Unused

Since no `tensor.pad` ops were generated, the `FoldFillIntoPad` cleanup pattern had no matches. All `linalg.fill` ops retain their original (unpadded) shapes.

### 6.4 Lowering Config Coverage

| Category | Ops with config | Ops without config | Total |
|----------|----------------|-------------------|-------|
| Contractions (matmul, batch_matmul) | 21 | 0 | 21 |
| Reductions (generic with red iters) | ~50 | 0 | ~50 |
| Elementwise roots | ~15 | 0 | ~15 |
| Fused producers/epilogues | 0 | ~55 | ~55 |
| **Total** | ~86 | ~55 | ~141 |

100% of root ops have configs. 100% of fused producers/epilogues correctly have no config (they inherit tiling from the forall).

---

## 7. Summary

The PadOperands pass is functionally a no-op for this GPT-2 model because all dimensions are tile-aligned. The pass correctly:
- Reads padding specs from `lowering_config` attributes
- Checks each operand dimension against the pad-to values
- Finds no misalignment → inserts no `tensor.pad` ops
- Runs FoldFillIntoPad cleanup → no matches

The IR is unchanged between input and output of this pass. The next pass (PromoteMatmulOperands, Step 3) will insert `tensor.empty() + linalg.copy` for shared memory promotion on the 21 contraction ops that have `promoted_operands = [0, 1]`.
