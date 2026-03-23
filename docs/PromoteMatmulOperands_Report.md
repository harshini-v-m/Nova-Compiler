# NovaGPUPromoteMatmulOperandsPass — Detailed IR Analysis Report

**Pass:** `mlir::nova::NovaGPUPromoteMatmulOperandsPass` (`nova-gpu-promote-matmul-operands`)
**Source:** `lib/Transforms/LLVMGPU/NovaGPUPromoteMatmulOperands.cpp`
**IR dump location:** `debug2.log` lines 36187–38202
**Pipeline position:** Step 3 (after PadOperands, before K-tiling/reduction tiling)
**Date:** 2026-03-21

---

## 1. Pass Overview

The PromoteMatmulOperands pass inserts shared memory promotion copies for operands of contraction ops. For each op with `promoted_operands = [0, 1]` in its `lowering_config`, it inserts:

```
%empty = tensor.empty(sizes) : tensor<...xf32>
%copy  = linalg.copy ins(%operand) outs(%empty)
         {lowering_config = {thread = [1, 1, 4], workgroup = [0, 0, 0], ...}}
```

The copy has a **thread-only lowering_config** (zero workgroup/reduction tiles, non-zero thread tiles) so that:
1. **K-tiling (Step 4)** fuses the copy as a producer into the K-loop, shrinking it from `[wgM × K]` to `[wgM × kStep]`
2. **Thread tiling (Step 5)** treats the copy as an independent root (cooperative-loading `scf.forall`)
3. **InferMemorySpace (Step 8)** tags the `tensor.empty` as workgroup memory

### Skip rules:
- `linalg.fill` producers → SKIP (no benefit from promoting constants)
- `linalg.batch_matmul` / `linalg.matmul` producers → SKIP (contractions manage their own shared memory)
- All other producers (generics, transposes, extract_slices, block args) → PROMOTE

---

## 2. Summary Statistics

| Metric | Count |
|--------|-------|
| Total `linalg.copy` ops inserted | **42** |
| Total contraction ops (batch_matmul + matmul) | **21** (18 batch_matmul + 3 matmul) |
| Contractions with BOTH operands promoted | **21** (100%) |
| Contractions with only 1 operand promoted | **0** |
| Contractions with 0 operands promoted | **0** |
| Operands skipped (fill producers) | **0** (fills are DPS inits, not inputs) |
| Operands skipped (contraction producers) | **0** (no matmul-to-matmul chains in input position) |

---

## 3. Copy Lowering Config Profiles

### 3D Copy (batch_matmul operands) — 36 copies
```
lowering_config = {
  workgroup = [0, 0, 0],
  reduction = [0, 0, 0],
  thread    = [1, 1, 4],        # vectorize innermost dim: 128/32 = 4
  subgroup  = [0, 0, 0],
  mma_kind  = 0,
  promoted_operands = []
}
```
- Thread tiles: `[1, 1, 4]` → 4 f32 elements = 128 bits per thread (vector load width)
- Zero workgroup tiles → K-tiling fuses as producer
- Non-zero thread tiles → thread tiling creates independent cooperative-loading forall

### 2D Copy (matmul operands) — 6 copies
```
lowering_config = {
  workgroup = [0, 0],
  reduction = [0, 0],
  thread    = [1, 4],           # vectorize innermost dim
  subgroup  = [0, 0],
  mma_kind  = 0,
  promoted_operands = []
}
```

---

## 4. Complete Forall Inventory with Promotion Analysis

### Forward Pass — Embedding + LN0

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 1 | %28 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN reduction) |
| 2 | %29 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN reduction) |

### Forward Pass — Transformer Block 0

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 3 | %32:4 | (0,0,0)→(8,1024,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH promoted** |
| 4 | %34 | (0,0,0)→(8,1024,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH promoted** |

**Forall #3 detail (FFN-up mega-fusion):**
```
embedding_lookup → LN_normalize → [COPY op0] → tensor.empty
                   W_broadcast  → [COPY op1] → tensor.empty
                   fill →
                   batch_matmul(copy0, copy1, fill) → bias_add → GELU
```
- Copy 0: source = LN normalize (`linalg.generic`, 1x32x384xf32), thread=[1,1,4]
- Copy 1: source = W broadcast (`linalg.generic`, 1x384x128xf32), thread=[1,1,4]

**Forall #4 detail (FFN-down + residual):**
```
GELU_output (extract_slice) → [COPY op0] → tensor.empty
W_down_broadcast             → [COPY op1] → tensor.empty
fill →
batch_matmul(copy0, copy1, fill) → residual + bias_add
```
- Copy 0: source = `tensor.extract_slice` of GELU output (1x32x1536xf32)
- Copy 1: source = W_down broadcast (`linalg.generic`, 1x1536x128xf32)

### Forward Pass — Transformer Block 1

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 5 | %35 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN mean) |
| 6 | %36 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN var) |
| 7 | %37:3 | (0,0,0)→(8,1024,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH promoted** |
| 8 | %38 | (0,0,0)→(8,1024,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH promoted** |

### Forward Pass — Transformer Block 2

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 9 | %39 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN mean) |
| 10 | %40 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN var) |
| 11 | %41:3 | (0,0,0)→(8,1024,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH promoted** |
| 12 | %42 | (0,0,0)→(8,1024,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH promoted** |

### Forward Pass — Final LN + Output Projection

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 13 | %43 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN mean) |
| 14 | %44 | (0,0)→(8,1024) step(1,256) | 0 | 0 | N/A (LN var) |
| 15 | %45 | in(1) | 0 | 0 | N/A (single-block LN norm) |
| 16 | %47 | (0,0)→(8192,50304) step(32,128) | **2** | 1 matmul | **BOTH promoted** |

**Forall #16 detail (output projection matmul):**
- Copy 0: source = `tensor.extract_slice` of reshaped LN output (32x384xf32), thread=[1,4]
- Copy 1: source = `tensor.extract_slice` of W_vocab weight (384x128xf32), thread=[1,4]

### Forward Pass — Softmax + Loss

| # | SSA | Bounds | Copies | Contractions | Notes |
|---|-----|--------|--------|-------------|-------|
| 17–27 | %48–%62 | various | 0 | 0 | Softmax, CE loss, scatter — no contractions |

### Backward Pass — Output Projection

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 28 | %65 | (0,0)→(8192,384) step(32,128) | **2** | 1 matmul (dX) | **BOTH promoted** |
| 29 | %67 | (0,0)→(384,50304) step(32,128) | **2** | 1 matmul (dW) | **BOTH promoted** |

**Forall #28 detail (dX = dY × W_vocab^T):**
- Copy 0: source = `tensor.extract_slice` of grad (32x50304xf32), thread=[1,4]
- Copy 1: source = `linalg.transpose` result (50304x128xf32), thread=[1,4]

**Forall #29 detail (dW = X^T × dY):**
- Copy 0: source = `linalg.transpose` result (32x8192xf32), thread=[1,4]
- Copy 1: source = `tensor.extract_slice` of grad (8192x128xf32), thread=[1,4]

### Backward Pass — Final LN

| # | SSA | Bounds | Copies | Contractions | Notes |
|---|-----|--------|--------|-------------|-------|
| 30–33 | %68–%72 | various | 0 | 0 | LN backward reductions, d(gamma), d(beta) |

### Backward Pass — FFN Block 3

| # | SSA | Bounds | Copies | Contractions | Promotion Status |
|---|-----|--------|--------|-------------|-----------------|
| 34 | %75 | in(1) | 0 | 0 | Single-block transpose |
| 35 | %76:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | **2** | 1 batch_matmul (dX_up) | **BOTH promoted** |
| 36 | %77 | (0,0,0)→(8,1536,384) step(1,32,128) | **2** | 1 batch_matmul (dW_down) | **BOTH promoted** |
| 37 | %79 | (0)→(384) step(128) | 0 | 0 | d(bias_down) |
| 38 | %81 | in(1) | 0 | 0 | Single-block transpose |
| 39 | %82:2 | (0,0,0)→(8,1024,384) step(1,32,128) | **2** | 1 batch_matmul (dX_down) | **BOTH promoted** |
| 40 | %83 | (0,0,0)→(8,384,1536) step(1,32,128) | **2** | 1 batch_matmul (dW_up) | **BOTH promoted** |
| 41 | %85 | (0)→(1536) step(512) | 0 | 0 | d(bias_up) |

**Forall #35 detail (backward dX_up = grad × W_up):**
- Copy 0: source = LN backward normalize (`linalg.generic`, 1x32x384xf32), thread=[1,1,4]
- Copy 1: source = W_up broadcast (`linalg.generic`, 1x384x128xf32), thread=[1,1,4]

**Forall #36 detail (dW_down = X_up^T × grad):**
- Copy 0: source = `linalg.transpose` result (1x32x1024xf32), thread=[1,1,4]
- Copy 1: source = `tensor.extract_slice` of upstream result (1x1024x128xf32), thread=[1,1,4]

### Backward Pass — LN Block 3 + FFN Block 2

| # | SSA | Bounds | Copies | Contractions | Promotion |
|---|-----|--------|--------|-------------|-----------|
| 42–43 | %86–%87 | (0,0)→(8,1024) step(1,256) | 0 | 0 | LN backward |
| 44 | %88 | in(1) | 0 | 0 | Transpose |
| 45 | %89:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 46 | %90 | (0,0,0)→(8,1536,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 47 | %91 | (0)→(384) step(128) | 0 | 0 | d(bias_down) |
| 48 | %92 | in(1) | 0 | 0 | Transpose |
| 49 | %93:2 | (0,0,0)→(8,1024,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 50 | %94 | (0,0,0)→(8,384,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 51 | %95 | (0)→(1536) step(512) | 0 | 0 | d(bias_up) |

### Backward Pass — LN Block 2 + FFN Block 1

| # | SSA | Bounds | Copies | Contractions | Promotion |
|---|-----|--------|--------|-------------|-----------|
| 52–53 | %96–%97 | (0,0)→(8,1024) step(1,256) | 0 | 0 | LN backward |
| 54 | %98 | in(1) | 0 | 0 | Transpose |
| 55 | %99:2 | (0,0,0)→(8,1024,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 56 | %100 | (0,0,0)→(8,1536,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 57 | %101 | (0)→(384) step(128) | 0 | 0 | d(bias_down) |
| 58 | %102 | in(1) | 0 | 0 | Transpose |
| 59 | %103:2 | (0,0,0)→(8,1024,384) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 60 | %104 | (0,0,0)→(8,384,1536) step(1,32,128) | **2** | 1 batch_matmul | **BOTH** |
| 61 | %105 | (0)→(1536) step(512) | 0 | 0 | d(bias_up) |

### Backward Pass — LN Block 0 + Embedding Gradients + Optimizer

| # | SSA | Bounds | Copies | Contractions | Notes |
|---|-----|--------|--------|-------------|-------|
| 62–67 | %106–%112 | various | 0 | 0 | LN backward, scatter, fill |
| 68–88 | %116–%160 | various | 0 | 0 | Optimizer weight/bias updates |

---

## 5. Promoted Operand Source Classification

### All 42 promoted operand sources by producer type:

| Producer Type | Count | Example | Notes |
|---------------|-------|---------|-------|
| `linalg.generic` (LN normalize) | 6 | 1x32x384xf32 | Forward 3 blocks + backward 3 blocks (LHS of FFN-up) |
| `linalg.generic` (W broadcast) | 12 | 1x384x128xf32 | 2D→3D weight expansion (RHS of batch_matmul) |
| `linalg.generic` (GELU backward) | 3 | 1x32x1536xf32 | Backward FFN dX_down (LHS) |
| `linalg.generic` (LN bwd normalize) | 3 | 1x32x384xf32 | Backward FFN dX_up (LHS) |
| `linalg.transpose` | 8 | 1x32x1024xf32 | Backward dW matmuls + output proj backward |
| `tensor.extract_slice` | 10 | Various | Slices from prior forall results or function args |

### Skipped operands:

| Producer Type | Count | Reason |
|---------------|-------|--------|
| `linalg.fill` | 21 | Zero-init accumulators — DPS init operands, not inputs |

Note: `linalg.fill` ops are DPS init operands (operand index 2 for matmul), not input operands (index 0, 1). Since `promoted_operands = [0, 1]` only targets inputs, fills are never candidates for promotion. The `isFillProducer` skip is a safety guard for cases where a fill feeds an input.

---

## 6. Copy Shape Analysis

### Shapes of promoted copies (before K-tiling):

| Matmul Type | Op0 Copy Shape | Op1 Copy Shape | Total Shared per Op |
|-------------|---------------|---------------|-------------------|
| FFN-up (fwd) | 1×32×384 = 12K elems | 1×384×128 = 49K elems | 244 KB |
| FFN-down (fwd) | 1×32×1536 = 49K elems | 1×1536×128 = 197K elems | 983 KB |
| Output proj | 32×384 = 12K elems | 384×128 = 49K elems | 244 KB |
| dX (output proj bwd) | 32×50304 = 1.6M elems | 50304×128 = 6.4M elems | 32 MB |
| dW (output proj bwd) | 32×8192 = 262K elems | 8192×128 = 1M elems | 5 MB |
| dX_up (bwd FFN) | 1×32×384 = 12K elems | 1×384×128 = 49K elems | 244 KB |
| dW_down (bwd FFN) | 1×32×1024 = 33K elems | 1×1024×128 = 131K elems | 655 KB |
| dX_down (bwd FFN) | 1×32×1536 = 49K elems | 1×1536×128 = 197K elems | 983 KB |
| dW_up (bwd FFN) | 1×32×1024 = 33K elems | 1×1024×128 = 131K elems | 655 KB |

**These are PRE-K-tiling sizes.** After K-tiling (Step 4) fuses the copy into the K-loop, the actual shared memory per K-iteration will be:
- Op0: `[wgM × kStep]` = `[32 × 32]` = 4 KB
- Op1: `[kStep × wgN]` = `[32 × 128]` = 16 KB
- **Total per K-step: ~20 KB** — well within 48 KB shared memory limit

---

## 7. Issues and Observations

### 7.1 All 21 Contractions Fully Promoted (Good)

Every contraction op has both operands promoted. No operand was skipped due to the producer type check. This confirms the promotion logic is working correctly for this model.

### 7.2 Pre-K-Tiling Copy Sizes Are Large

The copy shapes at this stage are full operand sizes (up to 6.4M elements for dX backward). This is by design — K-tiling (Step 4) will shrink these to `kStep`-sized slices. The `tensor.empty()` destination (not `alloc_tensor(workgroup)`) ensures no premature memory allocation.

### 7.3 Thread-Only Config Enables Cooperative Loading

All 42 copies have `workgroup = [0,0,0]` and `thread = [1,1,4]`. This means:
- **K-tiling** sees zero workgroup tiles → fuses copy as producer into K-loop
- **Thread tiling** sees non-zero thread tiles → creates separate `scf.forall` with thread mapping
- This is the IREE DerivedThreadConfigAttr pattern — the copy becomes a cooperative load

### 7.4 Vectorization Width

All copies use `thread[innermost] = 4` for f32 (4 × 32 bits = 128 bits = one vector load). This matches the NVIDIA LDG.128 instruction width.

### 7.5 Single-Block Foralls Unchanged

The 7 single-block foralls from TileAndDistribute Phase 2 (%45, %75, %81, %88, %92, %98, %102) are unaffected — they contain no contractions and thus no promotion.

---

## 8. Expected Downstream Behavior

After this pass, the pipeline continues:

| Step | Pass | Effect on Promoted Copies |
|------|------|--------------------------|
| 4 | K-tiling (reduction) | Fuses copy as producer → copy shrinks from `[wgM × K]` to `[wgM × kStep]` |
| 5 | Thread tiling | Copy becomes cooperative-loading `scf.forall{thread}` |
| 8 | InferMemorySpace | `tensor.empty` → `alloc_tensor(workgroup)` for copies in thread foralls |
| 9 | Bufferize | `alloc_tensor(workgroup)` → `memref.alloc(workgroup)` (shared memory) |
| 11 | InsertWorkgroupBarriers | `gpu.barrier` between cooperative store and matmul read |

---

## 9. Summary

The PromoteMatmulOperands pass successfully inserted **42 `linalg.copy` ops** (2 per contraction) across all **21 contraction ops**. Every contraction has both input operands promoted with thread-only lowering configs for cooperative loading. The promotion skip logic correctly:
- Preserves `linalg.fill` (zero-init) operands unchanged
- Would skip contraction-to-contraction chains (none exist in input position)
- Promotes all other producer types: `linalg.generic`, `linalg.transpose`, `tensor.extract_slice`

The copies are still at full operand size — K-tiling (Step 4) will shrink them to tile-sized shared memory allocations.
