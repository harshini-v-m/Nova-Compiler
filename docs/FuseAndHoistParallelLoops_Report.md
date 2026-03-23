# FuseAndHoistParallelLoops Pass Report — `NovaGPUFuseAndHoistParallelLoopsPass`

**Pipeline Stage:** Step 6 — `nova-gpu-fuse-and-hoist-parallel-loops`
**IR Range in debug2.log:** Lines 60315–63765
**Model:** GPT-2 (8×1024×384, 6 layers)
**Target:** sm_86 (RTX 3060)
**Source:** [NovaGPUFuseAndHoistParallelLoops.cpp](../lib/Transforms/LLVMGPU/NovaGPUFuseAndHoistParallelLoops.cpp)

---

## 1. Pass Overview

This pass runs 3 rounds of greedy rewrite patterns to fuse and hoist thread-mapped `scf.forall` loops, aiming to produce a single flat thread forall per compute region. It mirrors IREE's `GPUFuseAndHoistParallelLoops.cpp`.

**Round 1 — Hoist + Fuse:**
- `FuseForalls` — merge producer/consumer thread foralls with matching trip counts
- `FuseTilableForallConsumers` — fuse DPS consumers into producer foralls
- `HoistForallFromFor` — hoist thread foralls out of K-loops (loop interchange)

**Round 2 — Revealed consumers/destinations:**
- `FuseTilableDestinationProducers` — fuse DPS init producers into foralls
- `FuseUnitLoopDestination` — eliminate unit-trip-count foralls
- `FuseTilableForallConsumers` — re-run for newly revealed consumers

**Round 3 — New producer fusions:**
- `FuseTilableSliceProducers` — fuse tilable producers of slice ops
- `FuseTilableDestinationProducers` — re-run
- `ExtractSliceOfPadTensorSwap` — swap extract_slice(pad) → pad(extract_slice)

---

## 2. Result: EFFECTIVELY A NO-OP

**The pass made no structural changes to the IR for this model.**

| Metric | Before (post-CSE Step 5) | After (Step 6) | Delta |
|--------|-------------------------|----------------|-------|
| `scf.forall` total | 426 | 426 | 0 |
| `#gpu.thread` mappings | 127 | 127 | 0 |
| `#gpu.block` mappings | 90 | 90 | 0 |
| `scf.for` (K-loops) | 77 | 77 | 0 |
| `linalg.generic` | 124 | 124 | 0 |
| `linalg.copy` | 42 | 42 | 0 |
| `linalg.batch_matmul` | 18 | 18 | 0 |
| `linalg.fill` | 62 | 63 | +1 (minor fold artifact) |
| `lowering_config` attrs | 124 | 124 | 0 |
| IR body lines | 3405 | 3405 | 0 |

The only difference is the affine map definitions being inlined (CSE output uses `#mapN` shorthand; the pass output inlines maps) and 1 extra `linalg.fill` from a pattern fold/unfold cycle.

---

## 3. Why Each Pattern Failed to Fire

### 3.1 FuseForalls — No Matching Trip Counts

This pattern merges a producer `scf.forall` into its consumer `scf.forall` when:
- Both are thread-mapped and normalized
- Both have identical dimension-wise upper bounds
- Producer has a single use

**Why it failed:** Inside K-loops, the thread foralls have **different trip counts**:

| Forall | Trip Count | Dims |
|--------|-----------|------|
| Embedding lookup | `(0) to (4) step (2)` | 2 |
| Reduction (LN sum) | `in (256)` | 256 |
| Copy LHS (batch_matmul) | `(0,0,0) to (1,32,32) step (1,1,4)` | 256 |
| Copy RHS (batch_matmul) | `(0,0,0) to (1,32,128) step (1,1,4)` | 1024 |
| batch_matmul | `(0,0,0) to (1,32,128) step (1,1,16)` | 256 |

No two consecutive producer→consumer foralls have matching bounds, so FuseForalls never fires.

### 3.2 HoistForallFromFor — Multiple Foralls Per K-loop

This pattern hoists a thread forall outside its enclosing `scf.for` (K-loop) via loop interchange. Requirements:
- `scf.for` must have exactly 1 result
- The yielded value must come from exactly 1 `scf.forall`
- No other foralls in the for body

**Why it failed:** Every K-loop contains **2–3 thread foralls**:

| K-loop Type | Foralls Inside | Pattern Fail Reason |
|------------|----------------|-------------------|
| Embedding K-loop | 2 (embedding + reduction) | "for body contains another scf.forall" |
| Contraction K-loop | 3 (copy LHS + copy RHS + matmul) | "for body contains another scf.forall" |
| Softmax K-loop | 2 (exp + reduction) | "for body contains another scf.forall" |

The pattern explicitly rejects `scf.for` bodies containing more than one `scf.forall` (line 756-758).

### 3.3 FuseTilableForallConsumers — No Epilogue Ops Outside Foralls

This pattern fuses DPS consumers (bias add, GELU, etc.) into producer foralls.

**Why it failed:** All epilogue ops were already fused during Step 5 (Thread tiling). The 63 unfused `linalg.generic` ops are already inside block foralls as fused producers — they don't have `scf.forall` producers to fuse into.

### 3.4 FuseTilableDestinationProducers — No Eligible Destination Producers

This pattern fuses TilingInterface producers of forall init values.

**Why it failed:** The init values for thread foralls are either:
- `tensor.empty()` (for copies) — not a TilingInterface op
- `linalg.fill` (for contractions) — already fused by tiling
- `scf.for` iter args — block arguments, not TilingInterface

### 3.5 FuseUnitLoopDestination — No Unit-Trip-Count Foralls (Inside K-loops)

Thread foralls all have trip counts > 1 (256, 1024, etc.). The single-block foralls (`in (1)`) are block-mapped, not thread-mapped, so this pattern doesn't apply.

### 3.6 FuseTilableSliceProducers — All Producers Already Inside

Any `tensor.extract_slice` inside a forall whose producer is outside was already handled during tiling. No new opportunities remain.

---

## 4. Op Census (Unchanged from Step 5)

| Op Type | Count | Notes |
|---------|-------|-------|
| `linalg.generic` (with config) | 61 | Thread-tiled |
| `linalg.generic` (fused, no config) | 63 | Inside block foralls |
| `linalg.copy` | 42 | Promoted operand copies |
| `linalg.batch_matmul` | 18 | 6 layers × 3 |
| `linalg.matmul` | 3 | Final logit + weight grads |
| `linalg.fill` | 63 | +1 from fold artifact |
| `scf.for` (K-loops) | 77 | Unchanged |
| `scf.forall` (total) | 426 | 90 block + 336 thread |

---

## 5. Nesting Structure (Unchanged)

The 4-level nesting from Step 5 persists:

```
scf.forall (block)           [#gpu.block]
  └─ scf.for (K-loop)        [sequential]
       ├─ scf.forall (thread: copy LHS)   [#gpu.thread]
       ├─ scf.forall (thread: copy RHS)   [#gpu.thread]
       └─ scf.forall (thread: matmul)     [#gpu.thread]
```

**Ideal (what IREE achieves):**
```
scf.forall (block)           [#gpu.block]
  └─ scf.forall (thread)     [#gpu.thread]  ← single flat forall
       └─ scf.for (K-loop)   [sequential]
            ├─ copy LHS
            ├─ copy RHS
            └─ matmul
```

The pass couldn't achieve this because:
1. Copy foralls (256 or 1024 threads) and matmul foralls (256 threads) have different trip counts → can't fuse
2. Multiple foralls inside K-loop → can't hoist

---

## 6. Root Cause Analysis

The fundamental issue is the **SIMT tiling strategy** (mma_kind=0). With SIMT:

- **Copy ops** get `thread=[1,1,4]` → each thread copies 4 elements → trip count = total_elements/4
- **Matmul ops** get `thread=[1,1,16]` → each thread computes a 1×16 tile → trip count = rows × (cols/16)

LHS copy: `(1,32,32)/(1,1,4)` = 256 threads
RHS copy: `(1,32,128)/(1,1,4)` = 1024 threads
Matmul: `(1,32,128)/(1,1,16)` = 256 threads

The trip count mismatch (256 vs 1024 vs 256) prevents FuseForalls from merging them.

In IREE's MMA (tensor core) path, all ops within a workgroup use the same flat thread count (typically 128 = 4 warps), so FuseForalls succeeds and HoistForallFromFor can then interchange the K-loop.

### 6.1 What Would Fix This

To enable the fuse+hoist path, all thread foralls within a K-loop must have the **same flat thread count**. Options:

1. **Normalize copy thread counts to match matmul** — e.g., make copies also use 256 threads by adjusting thread tiles. This would change `thread=[1,1,4]` to different tiles on copies.

2. **Add MMA/Tensor Core support** — with MMA, all ops share a single warp-cooperative thread count (128 threads for 4 warps). But this requires TF32 intrinsic support (deferred for sm_86).

3. **Generalize HoistForallFromFor** — support multi-forall bodies by fusing them first with a trip-count-normalizing wrapper. This is complex and not in IREE.

---

## 7. Impact Assessment

**Performance impact of the no-op:** MODERATE

The failure to hoist means:
- **No loop interchange** — the K-loop stays outside thread foralls, so barrier synchronization happens at the thread forall boundaries (between copy and matmul), not at K-loop iteration boundaries
- **No forall fusion** — each K-step creates 3 separate thread foralls with independent scheduling
- **Each K-iteration materializes intermediate tensors** between copy and matmul foralls

In IREE's MMA path, hoisting produces a single `scf.forall(thread)` containing an `scf.for(K)` with fused copies+matmul — enabling register-level accumulation and barrier-free K-stepping within a thread.

For the SIMT path, the current structure still works correctly — the copies and matmul execute as separate cooperative parallel regions within each K-step. The intermediate tensors (`tensor.empty`) become shared memory after bufferization, and the implicit barrier at forall boundaries ensures correctness.

---

## 8. Status

- **Pass result: NO-OP** — IR structure unchanged
- **Root cause: SIMT trip count mismatch** between copy (256/1024) and matmul (256) thread foralls
- **No bug** — the pass correctly identifies that fusion/hoisting preconditions are not met
- **Performance impact: MODERATE** — functional but suboptimal compared to MMA path
- **Fix: Deferred** — requires either MMA support or trip count normalization
