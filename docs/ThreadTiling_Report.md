# Thread Tiling Pass Report — `NovaGPUApplyTilingLevelThreadPass`

**Pipeline Stage:** Step 5 — `nova-gpu-apply-tiling-level-thread`
**IR Range in debug2.log:** Lines 49712–53317
**Model:** GPT-2 (8×1024×384, 6 layers)
**Target:** sm_86 (RTX 3060)

---

## 1. Pass Overview

The Thread tiling pass tiles ops that have non-zero `thread` tile sizes in their `lowering_config`.
For each root op it:
1. Creates an `scf.forall` with `#gpu.thread<linear_dim_N>` mapping
2. Fuses non-payload producers into the thread forall
3. Uses `clampThreadTilesToMaxThreads()` to ensure ≤ 1024 threads per block
4. For reduction ops with thread tiles on reduction dims → uses `PartialReductionOuterParallel`

This pass transforms the K-tiled IR from Step 4 into per-thread work items.

---

## 2. Op Census After Thread Tiling

| Op Type | Count | Notes |
|---------|-------|-------|
| `linalg.generic` (with config) | 61 | Tiled into thread foralls |
| `linalg.generic` (fused, no config) | 63 | Fused as producers into thread/block foralls |
| `linalg.copy` | 42 | Promoted matmul operand copies — each in own thread forall |
| `linalg.batch_matmul` | 18 | 6 layers × 3 contractions = 18 |
| `linalg.matmul` | 3 | 1 final logit matmul + 2 weight grad matmuls |
| `linalg.fill` | 62 | Accumulator initialization for contractions + reductions |
| `scf.for` (K-loops) | 77 | Sequential reduction loops (from Step 4) |
| `scf.forall` (total) | 426 | Includes both block + thread foralls |
| `lowering_config` attrs | 124 | On all independently-tiled ops |

---

## 3. Thread Forall Mapping Patterns

| Mapping Pattern | Count | Used By |
|----------------|-------|---------|
| `[linear_dim_2, linear_dim_1, linear_dim_0]` (3D) | 54 | Contractions + copy ops (batch_matmul tiles) |
| `[linear_dim_0]` (1D) | 64 | Reduction ops, embeddings, softmax, elementwise |
| `[linear_dim_1, linear_dim_0]` (2D) | 9 | Final logit matmul copies + matmul itself |

---

## 4. Thread Tile Configuration Breakdown

| Thread Config | Workgroup Config | Count | Op Type | Thread Count |
|--------------|-----------------|-------|---------|-------------|
| `thread=[1,1,4]` | `workgroup=[0,0,0]` | 36 | `linalg.copy` (promoted operands) | 1×1×(32/4)=8 per K-step slice |
| `thread=[0,1,0]` | `workgroup=[1,256,0]` | 22 | Reduction generics (LN sum, var) | 256 threads |
| `thread=[1,1,16,0]` | `workgroup=[1,32,128,0]` | 18 | `batch_matmul` contractions | 1×32×8=256 threads |
| `thread=[2,0,0]` | `workgroup=[128,0,0]` | 11 | 1D elementwise (LN normalize fused) | 64 threads |
| `thread=[0,0,2]` | `workgroup=[1,1,128]` | 8 | Embedding lookup generics | 2 threads (K-step) |
| `thread=[1,4]` | `workgroup=[0,0]` | 6 | Matmul copy ops (2D) | 32×8=256 threads |
| `thread=[2]` | `workgroup=[128]` | 5 | 1D small elementwise | 64 threads |
| `thread=[2]` | `workgroup=[512]` | 3 | 1D medium elementwise | 256 threads |
| `thread=[2,0,0]` | `workgroup=[512,0,0]` | 3 | Large 1D ops | 256 threads |
| `thread=[1,16,0]` | `workgroup=[32,128,0]` | 3 | `linalg.matmul` contractions | 32×8=256 threads |
| `thread=[0,2]` | `workgroup=[1,128]` | 2 | 2D elementwise | 64 threads |
| `thread=[4]` | `workgroup=[1024]` | 1 | Token indexing | 256 threads |
| `thread=[0,4]` | `workgroup=[1,1024]` | 1 | Token label cast | 256 threads |
| `thread=[0,0,0,4]` | `workgroup=[1,1,1,1024]` | 1 | SCE loss 4D generic | 256 threads |

---

## 5. Block Forall Mapping Patterns

| Mapping Pattern | Count | Used For |
|----------------|-------|----------|
| `[#gpu.block<x>]` (1D) | 38 | Reductions, 1D ops, single-block wraps |
| `[#gpu.block<y>, #gpu.block<x>]` (2D) | 31 | Reduction ops (batch × seq tiles) |
| `[#gpu.block<z>, #gpu.block<y>, #gpu.block<x>]` (3D) | 21 | Contractions (batch × M × N tiles) |

**Total block foralls:** 90

---

## 6. Single-Block Foralls (`in (1)`)

10 ops are wrapped in single-block `scf.forall (%arg) in (1)`:

| # | Line (relative) | Op | Shape | Issue |
|---|----------------|----|-------|-------|
| 1 | 849 | Final LN normalize (layer 6→logits) | 8×1024×384 = 3.1M elements | **CRITICAL — serialized** |
| 2 | 1006 | `memref.store` (loss reduction phase 2) | 64→1 scalar | OK (tiny) |
| 3 | 1416 | Weight transpose (384×1536) | 590K elements | **CRITICAL — serialized** |
| 4 | 1608 | Weight transpose (1536×384) | 590K elements | **CRITICAL — serialized** |
| 5 | 1879 | Weight transpose (384×1536) | 590K elements | **CRITICAL — serialized** |
| 6 | 2076 | Weight transpose (1536×384) | 590K elements | **CRITICAL — serialized** |
| 7 | 2346 | Weight transpose (384×1536) | 590K elements | **CRITICAL — serialized** |
| 8 | 2543 | Weight transpose (1536×384) | 590K elements | **CRITICAL — serialized** |
| 9 | 2909 | Embedding grad scatter (50304×384) | 19.3M elements | **CRITICAL — serialized** |
| 10 | 3592 | Logit weight grad transpose (384×50304) | 19.3M elements | **CRITICAL — serialized** |

**Impact:** 8 of these are large ops that will be serialized to a single workgroup (1 thread block). These are the same ops identified in the `isContractionPrologue` fix — they lack proper `lowering_config` and default to single-block wrapping.

---

## 7. Contraction Thread Tiling Detail

### 7.1 batch_matmul (18 ops)
- **Config:** `thread=[1, 1, 16, 0], workgroup=[1, 32, 128, 0]`
- **Thread forall:** `(0,0,0) to (1, 32, 128) step (1, 1, 16)` → 1×32×8 = **256 threads**
- **Per-thread tile:** `[1, 1, 16]` — each thread computes 1 row × 16 columns
- **Mapping:** `[linear_dim_2, linear_dim_1, linear_dim_0]`
- **Inside K-loop:** Each K-step has 2 copy foralls (LHS/RHS) + 1 matmul forall

### 7.2 matmul (3 ops)
- **Config:** `thread=[1, 16, 0], workgroup=[32, 128, 0]`
- **Thread forall:** `(0,0) to (32, 128) step (1, 16)` → 32×8 = **256 threads**
- **Per-thread tile:** `[1, 16]` — each thread computes 1 row × 16 columns
- **Mapping:** `[linear_dim_1, linear_dim_0]`

---

## 8. Copy Thread Tiling Detail (Shared Memory Promotion)

### 8.1 3D Copies (batch_matmul operands — 36 ops)
- **Config:** `thread=[1, 1, 4], workgroup=[0, 0, 0]`
- **Thread forall:** `(0,0,0) to (1, 32, 32) step (1, 1, 4)` → 1×32×8 = **256 threads** (LHS)
- **Thread forall:** `(0,0,0) to (1, 32, 128) step (1, 1, 4)` → 1×32×32 = **1024 threads** (RHS)
- **Per-thread tile:** `[1, 1, 4]` — each thread copies 4 contiguous f32 values
- **Mapping:** `[linear_dim_2, linear_dim_1, linear_dim_0]`

### 8.2 2D Copies (matmul operands — 6 ops)
- **Config:** `thread=[1, 4], workgroup=[0, 0]`
- **Thread forall:** `(0,0) to (32, 32) step (1, 4)` → 32×8 = **256 threads** (LHS)
- **Thread forall:** `(0,0) to (32, 128) step (1, 4)` → 32×32 = **1024 threads** (RHS)
- **Per-thread tile:** `[1, 4]` — each thread copies 4 contiguous f32 values
- **Mapping:** `[linear_dim_1, linear_dim_0]`

---

## 9. Reduction Op Thread Tiling

### 9.1 LN Sum / Variance Reductions (22 ops)
- **Config:** `thread=[0, 1, 0], workgroup=[1, 256, 0]`
- **Thread forall:** `in (256)` → **256 threads**
- **Per-thread tile:** `[1, 1]` — each thread reduces one sequence position
- **Inside K-loop:** `scf.for %arg = 0 to 384 step 4` (kStep=4)
- **Per K-step:** Each thread reduces 4 values (1×1×4 slice)

### 9.2 Softmax Max / SumExp Reductions (4 ops)
- **Config:** `thread=[0, 1, 0], workgroup=[1, 256, 0]`
- **Same pattern as LN reductions but K=50304, kStep=4**
- **Thread forall:** `in (256)` → **256 threads**

---

## 10. Elementwise / Embedding Ops

### 10.1 Embedding Lookups (8 ops)
- **Config:** `thread=[0, 0, 2], workgroup=[1, 1, 128]`
- **Thread forall:** `(0) to (K) step (2)` → K/2 threads (e.g., 384/2=192 or 32/2=16)
- **Fused into K-loops** as producers for LN normalize

### 10.2 LN Normalize (fused into contraction blocks)
- Appears as `linalg.generic` with 6 inputs (data, mean, var, gamma, beta, out)
- **Fused into the contraction block forall** — no separate thread forall
- **Config:** NOT independently tiled — fused as a producer into copy thread foralls
- The LN normalize is tiled to `[1, 1, 4]` by being fused into the copy forall

### 10.3 Bias Add + GELU (fused, no config)
- 63 `linalg.generic` ops without `lowering_config` are fused producers
- These include: bias additions, GELU activations, residual additions
- They execute at the granularity of their consumer's thread tile

---

## 11. Nesting Structure

The final IR has a 4-level nesting for contractions:

```
scf.forall (block: batch × M_wg × N_wg)      [#gpu.block mapping]
  └─ scf.for (K-loop: step=32)                 [sequential reduction]
       ├─ scf.forall (thread: copy LHS)         [#gpu.thread, 256 threads]
       │    └─ linalg.copy [1,1,4]
       ├─ scf.forall (thread: copy RHS)         [#gpu.thread, 1024 threads]
       │    └─ linalg.copy [1,1,4]
       └─ scf.forall (thread: matmul)           [#gpu.thread, 256 threads]
            └─ linalg.batch_matmul [1,1,16]
```

For reductions:
```
scf.forall (block: batch × seq_wg)            [#gpu.block mapping]
  └─ scf.for (K-loop: step=4)                  [sequential reduction]
       └─ scf.forall (thread: 256 seq positions) [#gpu.thread, 256 threads]
            └─ linalg.generic (reduce [1,1,4])
```

---

## 12. Thread Count Summary

| Op Category | Thread Count | Count | Within 1024 Limit? |
|------------|-------------|-------|-------------------|
| batch_matmul | 256 | 18 | YES |
| matmul | 256 | 3 | YES |
| Copy LHS (3D) | 256 | 18 | YES |
| Copy RHS (3D) | 1024 | 18 | YES (at limit) |
| Copy LHS (2D) | 256 | 3 | YES |
| Copy RHS (2D) | 1024 | 3 | YES (at limit) |
| Reductions | 256 | 26 | YES |
| Embedding/softmax | 2–256 | 14 | YES |
| Small elementwise | 64–256 | ~25 | YES |

**No thread count violations.** RHS copies hit exactly 1024 threads (the hardware max), which is fine.

---

## 13. Key Observations

### 13.1 Thread Tiling is Complete for Configured Ops
All 124 ops with `lowering_config` are correctly tiled into thread foralls with appropriate GPU thread mapping.

### 13.2 Copy-Matmul Coordination
Inside each K-loop iteration, the pattern is: copy LHS → copy RHS → matmul. All three are separate `scf.forall` ops at the thread level. The copies load from global memory into `tensor.empty()` (will become shared memory after bufferization), and the matmul reads from the copies' results.

### 13.3 Fused Producers Work Correctly
63 ops without `lowering_config` are fused into their consumer's thread forall by the fusion control function. This includes LN normalize (fused into copy foralls), bias additions, GELU, and residual connections.

### 13.4 Single-Block Serialization (Pre-existing Issue)
The 10 single-block foralls from Step 1 persist. 8 of these contain large ops (590K–19.3M elements) that will be serialized to 1 thread block. The `isContractionPrologue` fix (applied but not yet tested) should assign proper configs to 7 of these (final LN normalize + 6 weight transposes), leaving only the 2 large grad ops (embedding scatter + logit weight transpose) in single-block wraps.

### 13.5 RHS Copies at Thread Limit
RHS matmul operand copies use 1024 threads (the CUDA maximum). This is because:
- Shape: `1×32×128`, tile: `1×1×4` → 1×32×32 = 1024 threads
- This means the `clampThreadTilesToMaxThreads()` function is not triggered (exactly at limit)
- If workgroup tiles were larger, clamping would activate

---

## 14. Comparison: Before vs After Thread Tiling

| Metric | Before (Step 4) | After (Step 5) |
|--------|-----------------|----------------|
| `scf.forall` count | ~90 (block only) | 426 (block + thread) |
| Thread foralls | 0 | ~336 |
| Ops inside thread foralls | 0 | 124 tiled + 63 fused |
| Nesting depth (contractions) | 2 (block + K-loop) | 4 (block + K-loop + thread + op) |
| GPU thread mapping attrs | 0 | 127 |

---

## 15. Status

- **Thread tiling: COMPLETE** — all configured ops correctly tiled
- **No thread count violations**
- **Fused producers correctly handled**
- **Single-block serialization persists** for 8 large unconfigured ops (pending `isContractionPrologue` fix verification)
