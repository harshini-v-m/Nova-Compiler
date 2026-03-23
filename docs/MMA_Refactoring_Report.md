# MMA Pipeline Refactoring Report: Nova vs IREE TileAndFuse

## Executive Summary

Nova's MMA tiling path produces **correct but unfusable** IR. The FuseAndHoistParallelLoops
pass is a NO-OP because copy and matmul thread foralls have **mismatched trip counts**.
The root cause is that Nova **hardcodes** copy thread tiles at promotion time (`[1,1,4]`),
while IREE **defers** copy tile derivation to tiling time using `DerivedThreadConfigAttr`,
guaranteeing all foralls in a K-loop body share the same thread count.

**Impact:** Every contraction kernel runs 3 separate thread foralls per K-iteration
(LHS copy, RHS copy, matmul) with 3 barriers, instead of 1 fused forall with 1 barrier.
For K=768, kStep=16 → 48 iterations × 2 extra barriers = **96 wasted barriers per kernel**.

---

## 1. Current Nova Pipeline (Steps 0–6)

```
Step 0  : SelectLoweringStrategy    → stamps lowering_config on ops
Step 1  : TileAndDistribute         → scf.forall {block}
Step 2  : PadOperands               → static tile sizes
Step 3  : PromoteMatmulOperands     → tensor.empty + linalg.copy (thread-only config)
Step 4  : ApplyTilingLevelReduction → K-dim → scf.for
Step 5  : ApplyTilingLevelThread    → per-thread M/N → scf.forall {thread}
Step 6  : FuseAndHoistParallelLoops → (currently NO-OP)
```

### Current MMA Config (after TF32 change)

| Op Type | workgroup | thread | reduction | Threads |
|---------|-----------|--------|-----------|---------|
| batch_matmul | [1, 128, 64] | [1, 8, 8] | [0, 0, 0, 16] | **128** |
| LHS copy (3D) | [0, 0, 0] | [1, 1, 4] | [0, 0, 0] | depends on shape |
| RHS copy (3D) | [0, 0, 0] | [1, 1, 4] | [0, 0, 0] | depends on shape |

### Copy Thread Counts After K-Tiling

After K-tiling fuses copies into the K-loop, copy shapes become:
- **LHS copy**: `[1, 128, 16]` (batch, wgM, kStep) → thread `[1,1,4]` → trip count = 1×128×4 = **512**
- **RHS copy**: `[1, 16, 64]` (batch, kStep, wgN) → thread `[1,1,4]` → trip count = 1×16×16 = **256**
- **Matmul forall**: trip count = (128/8)×(64/8) = 16×8 = **128**

**Result: 3 different trip counts (512, 256, 128) → FuseForalls cannot merge any pair.**

---

## 2. IREE TileAndFuse Pipeline

```
Step 0    : LoweringConfigInterpreter  → reads user annotations
Step 1    : PromoteMatmulOperands      → tensor.empty + copy (DerivedThreadConfigAttr)
Step 1.5  : ExpandUndistributedInnerTiles → shape expansion for MMA
Step 2    : ApplyTilingLevel(Reduction) → K-dim → scf.for
Step 3    : DecomposePackUnPackOps      → pack/unpack → linalg ops
Step 4    : ApplyTilingLevel(Thread)    → per-thread (fuseConsumers=FALSE)
Step 4.5  : ApplyTilingLevel(Subgroup)  → per-warp (fuseConsumers=FALSE)
Step 5    : FuseAndHoistParallelLoops   → merges foralls, hoists out of K-loop
Step 6    : Vectorize                   → linalg → vector dialect → MMA intrinsics
```

### Key Difference: DerivedThreadConfigAttr

IREE does NOT set copy thread tiles at promotion time. Instead:

**File:** `iree/.../GPU/IR/DerivedConfigUtils.cpp` lines 156-190

```cpp
SmallVector<int64_t> deriveThreadTileSizes(Operation *op) {
  // 1. Read workgroup size from parent function (set by contraction config)
  std::optional<SmallVector<int64_t>> workgroupSize =
      getWorkgroupSize(op->getParentOfType<FunctionOpInterface>());
  int64_t numThreads = llvm::product_of(*workgroupSize);  // e.g., 128

  // 2. Compute tile sizes so (product of loop ranges / product of tile sizes) = numThreads
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  int64_t vectorSize = 128 / elementBitWidth;  // e.g., 4 for f32
  return getVectorTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize);
}
```

**Algorithm in `getVectorTileSizesFromLoopRanges`:**
1. `flatNumTrips = product(loopRanges)` — total elements in the copy
2. `maxVectorSize = min(vectorSize, flatNumTrips / numThreads)` — elements per thread
3. Set innermost tile = `maxVectorSize`, all other tiles = 1
4. Result: trip count = `flatNumTrips / product(tileSizes)` = **numThreads** (guaranteed!)

### Example with MMA config (128 threads):

| Copy | Shape | deriveThreadTileSizes | Trip Count |
|------|-------|----------------------|------------|
| LHS | [1, 128, 16] | [1, 1, min(4, 2048/128)] = [1, 1, 4] → but adjusted | **128** |
| RHS | [1, 16, 64] | [1, 1, min(4, 1024/128)] = [1, 1, 4] → adjusted | **128** |
| Matmul | 128/8 × 64/8 | [8, 8] from explicit config | **128** |

All three = 128 threads → **FuseForalls succeeds!**

### FuseForalls Validation

**File:** `iree/.../GPU/GPUFuseAndHoistParallelLoops.cpp` lines 61-92

```cpp
// Check: forall trip count must match flatWorkgroupSize
bool forallTripCountMatchesWorkgroupSize(scf::ForallOp forall, int64_t flatWorkgroupSize) {
  // For thread-mapped foralls: product of trip counts == flatWorkgroupSize
  // For lane-mapped: parent warp × lane count == flatWorkgroupSize
}
```

---

## 3. Root Cause Analysis: Why Nova's Copies Don't Match

### Problem: Hardcoded Copy Thread Tiles

**File:** `Nova-Compiler/.../NovaGPUPromoteMatmulOperands.cpp` lines 155-169

```cpp
// Nova: HARDCODED at promotion time (Step 3), BEFORE K-tiling
int64_t vectorSize = std::max<int64_t>(1, 128 / elemBits);  // = 4 for f32
SmallVector<int64_t> threadTiles(numLoops, 1);
threadTiles[numLoops - 1] = vectorSize;  // Always [1, 1, 4]
```

This gives every copy `thread=[1,1,4]` regardless of the matmul's thread count.
After K-tiling shrinks the copy to `[1, 128, 16]`, thread tiling creates a forall
with trip count `1 × 128 × (16/4)` = **512** — not 128.

### Why IREE Doesn't Have This Problem

| Aspect | IREE | Nova |
|--------|------|------|
| Copy config timing | **Deferred** to tiling time via `DerivedThreadConfigAttr` | **Hardcoded** at promotion time |
| Thread count source | Reads `workgroupSize` from parent function | Independent of matmul config |
| Tile computation | `flatNumTrips / numThreads` → per-thread work | Fixed `vectorSize` only |
| Guarantees | Trip count == workgroupSize (always) | Trip count varies by shape |
| FuseForalls result | **Succeeds** — all foralls match | **Fails** — mismatch |

---

## 4. Op Lowering Config Comparison

### 4.1 Contractions

| Aspect | IREE TileAndFuse | Nova MMA |
|--------|-----------------|----------|
| Intrinsic | MMA intrinsic attr on config | `mma_kind` field in DictionaryAttr |
| Workgroup tiles | Computed from intrinsic shape × subgroup layout | Same (128×64 for TF32) |
| Thread tiles | `workgroupTile / workgroupSize` | Hardcoded [8, 8] |
| Reduction tiles | K-step from intrinsic | Same (kStep=16) |
| Subgroup tiles | Explicit from intrinsic layout | [16, 16] (hardcoded) |
| Promoted operands | [0, 1] | [0, 1] |
| **Workgroup size** | **Set on function attr** | **Not set** (implicit from thread tiles) |

### 4.2 Copy Ops (Promoted Operands)

| Aspect | IREE | Nova |
|--------|------|------|
| Config type | `DerivedThreadConfigAttr` (lazy) | Explicit `lowering_config` dict |
| Thread tiles | Derived at tiling time from `numThreads` | Hardcoded `[1, 1, vectorSize]` |
| Workgroup tiles | None (zero) | None (zero) |
| Reduction tiles | None (zero) | None (zero) |
| Thread count | **= matmul thread count** (guaranteed) | **≠ matmul** (shape-dependent) |

### 4.3 Elementwise / Reductions / Other Ops

| Aspect | IREE | Nova |
|--------|------|------|
| Config source | `setTileAndFuseLoweringConfig()` generic | `setDefaultConfig()` |
| Thread count | Uses workgroupSize from function | Independent (256 default) |
| Epilogue fusion | No config → fuses into contraction | No config → `isContractionEpilogue()` |
| Prologue fusion | No config → fuses as producer | `isContractionPrologue()` check |

### 4.4 Pipeline Ordering Difference

| Step | IREE | Nova | Impact |
|------|------|------|--------|
| Config | Interpreter reads annotations | `initNovaGPULaunchConfig()` | Same concept |
| **Promote** | **Step 1** (before reduction) | **Step 3** (before reduction) | Same relative position |
| Reduction | Step 2 | Step 4 | Same |
| Thread | Step 4 | Step 5 | Same |
| **Subgroup** | **Step 4.5** (ENABLED) | **Disabled** | IREE has warp-level tiling |
| Fuse+Hoist | Step 5 | Step 6 | Same |
| **Vectorize** | **Step 6** (PRESENT) | **Missing** | Nova has no vector→NVVM path |

---

## 5. What Needs to Be Refactored

### 5.1 CRITICAL: Derived Copy Thread Tiles (Fixes FuseForalls)

**Problem:** Copy foralls have shape-dependent trip counts that don't match matmul's 128 threads.

**Solution:** Replace hardcoded `thread=[1,1,vectorSize]` with a derived computation that
guarantees `trip_count = matmulThreadCount`.

**File:** `NovaGPUPromoteMatmulOperands.cpp` (lines 155-169)

**Approach A — IREE-style DerivedThreadConfigAttr:**
- Create a Nova `DerivedThreadConfigAttr` that defers tile computation to tiling time
- At tiling time, read the contraction's thread count and derive copy tiles
- **Pros:** Clean, always correct, handles dynamic shapes
- **Cons:** Requires new attr type, interface changes to tiling pass

**Approach B — Compute copy tiles from matmul config at promotion time:**
- Read the matmul's lowering_config to get thread count (e.g., 128)
- Read the K-tiled copy shape (after K-tiling fuses copy into loop)
- Compute: `vectorSize = min(4, flatCopySize / matmulThreads)`
- Set: `thread=[1, ..., vectorSize]` so `product(shape) / product(threadTiles) = matmulThreads`
- **Problem:** At promotion time (Step 3), K-tiling hasn't run yet, so we don't know the final copy shape.

**Approach C — Recompute copy tiles AFTER K-tiling (new pass between Step 4 and 5):**
- After K-tiling (Step 4), walk all promoted copies (they now have K-tiled shapes)
- Read the sibling matmul's thread count from its lowering_config
- Recompute copy thread tiles to match: `vectorSize = flatCopySize / matmulThreads`
- Update the copy's lowering_config with new thread tiles
- **Pros:** Simple, concrete shapes known, no new attr types
- **Cons:** Extra pass, needs to find sibling matmul from copy

**Recommended: Approach C** — simplest implementation, no architectural changes needed.

#### Concrete Implementation (Approach C):

```
New pass: NovaGPUDeriveCopyThreadTiles (insert between Step 4 and Step 5)

Algorithm:
1. Walk all linalg.copy ops with lowering_config
2. For each copy, find the sibling contraction in the same scf.for body
3. Read contraction's thread count: product(shape) / product(threadTiles)
   - For MMA TF32: (128/8) × (64/8) = 128 threads
4. Read copy's current shape (now K-tiled, e.g., [1, 128, 16])
5. Compute: flatSize = product(shape) = 2048
6. Compute: perThread = flatSize / targetThreads = 2048 / 128 = 16
7. Set innermost tile = min(perThread, vectorSize_128bit)
8. Distribute remaining across outer dims
9. Verify: product(shape) / product(threadTiles) == targetThreads
10. Update copy's lowering_config
```

**Expected result:**
- LHS copy `[1, 128, 16]`, target 128 threads: `thread=[1, 1, 16]` → 128×1×1 = **128** ✓
  - But wait: vectorSize for f32 should be ≤4 for 128-bit loads
  - So: `thread=[1, 8, 4]` → (1)×(128/8)×(16/4) = 1×16×4 = 64? No.
  - Need: `thread=[1, T_m, T_k]` where `(1/1)×(128/T_m)×(16/T_k) = 128`
  - If T_k=4: `128/T_m × 4 = 128` → `T_m = 4` → thread=[1,4,4], trips = 1×32×4 = 128 ✓
  - If T_k=2: `128/T_m × 8 = 128` → `T_m = 8` → thread=[1,8,2], trips = 1×16×8 = 128 ✓
- RHS copy `[1, 16, 64]`, target 128 threads:
  - flatSize = 1024, perThread = 1024/128 = 8
  - T_k=4: `16/T_m × 16 = 128` → `T_m = 2` → thread=[1,2,4], trips = 1×8×16 = 128 ✓

### 5.2 IMPORTANT: Workgroup Size on Function (Enables DerivedConfig)

**Problem:** Nova doesn't set a `workgroup_size` attribute on the function. IREE uses this
to let all ops in a kernel share the same thread count.

**Solution:** In `SelectLoweringStrategy` (Step 0), after configuring the root contraction,
compute `workgroupSize = [numThreads, 1, 1]` and set it on the function.

For MMA TF32: `workgroupSize = [128, 1, 1]`

This isn't strictly needed for Approach C, but enables future DerivedThreadConfigAttr adoption.

### 5.3 NICE-TO-HAVE: HoistForallFromFor Relaxation

**Problem:** Even if FuseForalls merges all three foralls into one, `HoistForallFromFor`
requires exactly 1 `scf.forall` in the `scf.for` body. After fusion, there should be
exactly 1, so this should work automatically.

**File:** `NovaGPUFuseAndHoistParallelLoops.cpp` lines 756-758

No changes needed if FuseForalls succeeds — the fused forall is the single forall in the K-body.

### 5.4 FUTURE: Vectorization Pass (Not Required for Fusion Fix)

IREE has a full vectorization pipeline (Step 6) that converts `linalg` ops to `vector` dialect
and then to NVVM MMA intrinsics. Nova lacks this entirely. Without it:
- Compute runs through scalar `scf` loops → LLVM IR → PTX (no tensor cores)
- The MMA tiling config still helps by providing better data locality and tile sizes

This is a large separate effort, not part of the fusion fix.

---

## 6. Refactoring Plan (Ordered by Impact)

### Phase 1: Fix Copy Thread Tiles (Enables FuseForalls)

| # | Task | File | Lines Changed |
|---|------|------|---------------|
| 1a | Create `NovaGPUDeriveCopyThreadTiles` pass | New file | ~100 lines |
| 1b | Register pass in `Passes.cpp` between Step 4 and 5 | Passes.cpp | ~3 lines |
| 1c | Add pass declaration in `Passes.h` | Passes.h | ~2 lines |

**Algorithm for the new pass:**
```
for each scf.for (K-loop) in function:
  find the contraction op inside
  read contraction's lowering_config → compute targetThreads
  for each linalg.copy with lowering_config inside the same scf.for:
    read copy shape (now K-tiled)
    compute new threadTiles where product(shape/threadTiles) == targetThreads
    update copy's lowering_config with new threadTiles
```

**Expected result:** All three foralls (LHS copy, RHS copy, matmul) produce 128-thread
`scf.forall` ops → `FuseForalls` matches bounds → merges into 1 forall → `HoistForallFromFor`
hoists the single forall above the K-loop.

### Phase 2: Set Workgroup Size on Function

| # | Task | File | Lines Changed |
|---|------|------|---------------|
| 2a | After root config, set `workgroup_size` attr on function | NovaKernelConfig.cpp | ~10 lines |

### Phase 3: (Future) Vectorization Pipeline

Not in scope for this refactoring. Requires:
- `linalg` → `vector` dialect lowering
- `vector` → NVVM MMA intrinsic lowering
- Shared memory layout optimization for bank-conflict avoidance

---

## 7. Verification After Phase 1

1. **Build:** `make build`
2. **Run GPT-2:** `make run-snippet FILE=Tests/IrTest/gpt2_test_jit.cpp`
3. **Check FuseForalls:** After Step 6, count `scf.forall` ops with `#gpu.thread` mapping.
   Should be ~1/3 of current count (3 foralls merged into 1 per K-body).
4. **Check HoistForallFromFor:** Thread foralls should appear OUTSIDE `scf.for` K-loops,
   not inside.
5. **Check barriers:** Fewer `gpu.barrier` ops per kernel (1 per K-iteration instead of 3).
6. **Performance:** Expect significant throughput improvement from barrier elimination.

---

## 8. Summary Table

| Component | Current Nova | IREE Reference | Refactoring Needed |
|-----------|-------------|----------------|-------------------|
| Contraction tiling | MMA TF32 128×64, 128 threads ✓ | Similar | None |
| Copy thread tiles | Hardcoded [1,1,4] ✗ | DerivedThreadConfigAttr ✓ | **Phase 1** |
| Copy trip count | 512 / 256 (mismatched) ✗ | 128 (matches matmul) ✓ | **Phase 1** |
| FuseForalls | NO-OP ✗ | Merges all foralls ✓ | Fixed by Phase 1 |
| HoistForallFromFor | NO-OP ✗ | Hoists above K-loop ✓ | Fixed by Phase 1 |
| Workgroup size attr | Not set ✗ | Set on function ✓ | Phase 2 |
| Vectorization | Missing ✗ | Full vector→NVVM ✓ | Future (Phase 3) |
| Subgroup tiling | Disabled | Enabled | Future |
