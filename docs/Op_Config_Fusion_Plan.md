# Nova GPU Pipeline: Op Config & Fusion Refactoring Plan

## Executive Summary

118 "large op serialized to 1 workgroup" warnings stem from **one root cause**: `initNovaGPULaunchConfig` only handles `linalg::GenericOp` for non-contractions, silently dropping `TransposeOp`, `FillOp`, `CopyOp`, and other named linalg ops. This causes two problems:
1. Named ops that should be prologues/epilogues aren't checked → no fusion → single-block fallback
2. Named ops that need standalone configs get none → single-block fallback

The fix is a 6-phase refactoring touching 2 primary files.

---

## 1. Op-by-Op Analysis

### Complete Lowering Chain

```
Nova ops (NovaIREmitter.cpp)
  → [NovaToArith]     nova.constant → arith.constant / linalg.fill
  → [NovaToTosa]      nova.gelu → mul/add/tanh chain
                       nova.layer_norm → reduce_mean/sub/mul/add
                       nova.sce → reduce/log/gather chain
  → [NovaElementwiseToLinalg]  nova.add/mul/sub → linalg.generic (parallel)
  → [NovaToLinalg]    nova.matmul → linalg.fill + optional broadcast + batch_matmul
                       nova.transpose → linalg.transpose
                       nova.linear → broadcast + batch_matmul + bias_add
  → [TosaToLinalg]    tosa.reduce_sum → linalg.generic (reduction)
  → [ElementwiseOpFusion]  fuses elementwise chains into single generic
  → [FoldUnitExtentDims]   collapses keepdim reductions
  → [Step 0: SelectLoweringStrategy]  stamps lowering_config on roots
  → [Step 1: TileAndDistribute]  creates scf.forall {block} per workgroup
  → ... (subsequent tiling, bufferize, GPU mapping)
```

### Forward Pass: Per-Op Breakdown

| Nova Op | Linalg Ops Produced | Gets Config? | Expected Behavior |
|---------|-------------------|--------------|-------------------|
| **nova.embedding** | 1× generic (gather, 3D parallel) [8×1024×384] | YES (default: wg=[1,1,128], th=[0,0,2]) | Standalone root |
| **nova.layer_norm** | 1× generic (mean reduction) [8×1024×384]→[8×1024] | YES (reduction config) | Standalone root |
| | 1× generic (variance reduction) | YES (reduction config) | Standalone root |
| | 1× generic (normalize, all-parallel) [8×1024×384] | **NO** — identified as prologue, skipped | Should fuse as producer into downstream batch_matmul |
| **nova.linear** | 1× generic (weight broadcast 2D→3D) | **NO** — identified as prologue | Should fuse as producer into batch_matmul |
| | 1× fill (zero init) | **NO** — falls through GenericOp guard | Should fuse as DPS init of batch_matmul |
| | 1× batch_matmul | YES (MMA: wg=[1,128,64,0], mma_kind=5) | Standalone root |
| | 1× generic (bias add, epilogue) | **NO** — identified as epilogue | Fuses as consumer of batch_matmul |
| **nova.gelu** | 1× generic (fused tanh chain, all-parallel) | **NO** — identified as epilogue (1-hop from contraction) | Fuses as consumer of linear's bias_add |
| **nova.add** (residual) | 1× generic (elementwise add) | Varies | Epilogue of preceding op, or standalone |
| **nova.matmul** | 1× fill + 1× matmul (or batch_matmul) | Matmul: YES. Fill: **NO** | Fill fuses as DPS init |
| **nova.sce** | N× generics (max reduction, exp, sum, log, gather, mean) | Reductions: YES. Elementwise: some skipped as epilogues | Complex chain |

### Backward Pass: Per-Op Breakdown

| Nova Op | Linalg Ops Produced | Gets Config? | Expected Behavior |
|---------|-------------------|--------------|-------------------|
| **nova.linear_backward** | 2× transpose [8192×384]↔[384×8192] | **NO** — falls through GenericOp guard | Should fuse as prologue of grad matmul, OR get standalone config |
| | 2× fill (zero init for dW, dX) | **NO** — falls through GenericOp guard | Fuse as DPS init |
| | 2× matmul (dX, dW gradients) | YES (MMA config) | Standalone roots |
| | 1× generic (bias grad reduction) | YES (reduction config) | Standalone root |
| **nova.layer_norm_backward** | 2× generic (reduction: dgamma/dbeta sums) | YES | Standalone roots |
| | N× generic (elementwise: dx computation) | **NO** — prologues/epilogues | Should fuse |
| **nova.gelu_backward** | 1× generic (fused backward chain) | **NO** — epilogue | Fuses as epilogue |
| **nova.sce_backward** | N× generics (softmax grad, scatter) | Varies | Complex chain |

---

## 2. Warning Root Cause Mapping

### 118 Warnings → 5 Root Causes

```
┌─────────────────────────────────────────────────────────┐
│ Root Cause 1: GenericOp guard (lines 789-802)           │
│ Affected: TransposeOp (7), FillOp (10)                  │
│ Impact: Named linalg ops silently get NO config,        │
│         NO prologue/epilogue check → Phase 2 fallback   │
│ Warnings: 17                                            │
├─────────────────────────────────────────────────────────┤
│ Root Cause 2: Prologue fusion failure                   │
│ Affected: LN normalize (24), weight broadcast (4)       │
│ Impact: Correctly identified as prologues, skipped by   │
│         config, but producer fusion in TileAndDistribute │
│         fails → no forall, Phase 2 fallback             │
│ Warnings: 28                                            │
├─────────────────────────────────────────────────────────┤
│ Root Cause 3: False positive (ops WITH config)          │
│ Affected: generics with all-zero wg tiles after         │
│         full-tile optimization (41)                      │
│ Impact: No real issue — warning is misleading            │
│ Warnings: 41                                            │
├─────────────────────────────────────────────────────────┤
│ Root Cause 4: False positive (contraction producers)    │
│ Affected: batch_matmul (26) + matmul (6) warnings       │
│         are actually about their unfused producer ops    │
│         sharing same source location                     │
│ Impact: No real issue — warning attribution misleading   │
│ Warnings: 32                                            │
├─────────────────────────────────────────────────────────┤
│ Total: 118 warnings                                     │
│   Real issues: 45 (Root Causes 1+2)                     │
│   False positives: 73 (Root Causes 3+4)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Fusion Chains (Per Transformer Layer)

### Forward Pass Fusion Graph

```
                    ┌──────────────┐
                    │  embedding   │  ← standalone root, gets config
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LN mean     │  ← standalone root (reduction), gets config
                    │  reduction   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LN variance │  ← standalone root (reduction), gets config
                    │  reduction   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ LN normalize │  ← PROLOGUE: should fuse into ▼
                    │ (all-parallel)│     currently: FAILS TO FUSE (28 warnings)
                    └──────┬───────┘
                           │
    ┌──────────────┐  ┌────▼──────┐  ┌──────────────┐
    │ wt broadcast │  │   fill    │  │              │
    │ (prologue)   │  │ (DPS init)│  │              │
    └──────┬───────┘  └────┬──────┘  │              │
           │               │         │              │
           └───────┬───────┘         │              │
                   │                 │              │
            ┌──────▼───────┐         │              │
            │ batch_matmul │ ◄───────┘              │
            │ (MMA root)   │  gets config           │
            └──────┬───────┘                        │
                   │                                │
            ┌──────▼───────┐                        │
            │  bias add    │  ← EPILOGUE: fuses as consumer of matmul
            │ (all-parallel)│
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │    GELU      │  ← EPILOGUE (1-hop): fuses as consumer
            │ (all-parallel)│
            └──────┬───────┘
                   │
           (repeat for 2nd linear: QKV→projection)
                   │
            ┌──────▼───────┐
            │ residual add │  ← EPILOGUE of 2nd matmul
            └──────────────┘
```

### Backward Pass Fusion Graph (per linear_backward)

```
            ┌──────────────┐
            │  grad input  │  (from upstream backward)
            └──────┬───────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌─────────┐   ┌──────────┐
│transpose│   │transpose│   │ bias grad│  ← standalone (reduction), gets config
│(act→T) │   │(wt→T)  │   │ reduction│
└───┬────┘   └───┬─────┘   └──────────┘
    │             │
    ▼             ▼
┌────────┐   ┌─────────┐
│fill(dX)│   │fill(dW) │  ← DPS init, should fuse
└───┬────┘   └───┬─────┘
    │             │
    ▼             ▼
┌────────┐   ┌─────────┐
│matmul  │   │matmul   │  ← standalone root, gets MMA config
│(dX)    │   │(dW)     │
└────────┘   └─────────┘

Current problem: transposes have NO config, run on 1 block
```

---

## 4. Implementation Phases

### Phase 1: Remove GenericOp Guard in NovaKernelConfig.cpp (HIGH PRIORITY)

**File**: `Nova-Compiler/lib/Transforms/LLVMGPU/NovaKernelConfig.cpp`
**Lines**: 786-802

**Current code** (broken):
```cpp
// Priority 2: Any linalg.generic → default config, unless epilogue/prologue
if (isa<linalg::GenericOp>(op.getOperation())) {  // ← PROBLEM: excludes TransposeOp, FillOp, CopyOp
    if (isContractionEpilogue(op.getOperation())) { ... return; }
    if (isContractionPrologue(op.getOperation())) { ... return; }
    (void)setDefaultConfig(op, target);
    return;
}
// Named linalg ops silently fall through here → NO CONFIG
```

**Refactored code**:
```cpp
// Priority 2: FillOp → always fuses as DPS init, skip entirely.
if (isa<linalg::FillOp>(op.getOperation()))
    return;

// Priority 3: Epilogue/prologue of contraction → skip (fuses during TileAndDistribute).
if (isContractionEpilogue(op.getOperation())) {
    LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping epilogue: "
                             << op->getName() << "\n");
    return;
}
if (isContractionPrologue(op.getOperation())) {
    LLVM_DEBUG(llvm::dbgs() << "[nova-kernel-config] Skipping prologue: "
                             << op->getName() << "\n");
    return;
}

// Priority 4: Any other linalg op (generic, transpose, copy, etc.) → default config.
(void)setDefaultConfig(op, target);
```

**Impact**:
- TransposeOp now gets prologue check → if it feeds a matmul via projected permutation, skipped (fuses). If not, gets standalone default config with multi-block tiling.
- FillOp explicitly skipped (DPS init fusion).
- CopyOp and other named ops get proper handling.
- Eliminates 17 warnings (7 transpose + 10 fill).

**Risk**: `isContractionPrologue` and `isContractionEpilogue` currently only check `linalg::GenericOp` for 1-hop chains (line 714: `isa<linalg::GenericOp>(defOp)`). Need to verify they work for TransposeOp too.

### Phase 2: Fix isContractionEpilogue/Prologue for Named Ops

**File**: `Nova-Compiler/lib/Transforms/LLVMGPU/NovaKernelConfig.cpp`
**Lines**: 686-761

**Current `isContractionEpilogue`** (line 714):
```cpp
if (auto genericDef = dyn_cast<linalg::GenericOp>(defOp)) {
    for (Value innerOp : genericDef->getOperands()) { ... }
}
```
This only walks 1-hop chains through `GenericOp`. TransposeOp as an intermediate in an epilogue chain would be missed.

**Fix**: Replace `isa<linalg::GenericOp>` with `isa<linalg::LinalgOp>` for the 1-hop chain walk:
```cpp
if (auto linalgDef = dyn_cast<linalg::LinalgOp>(defOp)) {
    for (Value innerOp : linalgDef->getOperands()) { ... }
}
```

**Current `isContractionPrologue`** (line 736-761):
Already works on any `Operation*` — checks `op->getResults()` users for contractions. The indexing map check (`map.isProjectedPermutation()`) works for any linalg op. **No changes needed** here.

### Phase 3: Suppress False-Positive Warnings

**File**: `Nova-Compiler/lib/Transforms/LLVMGPU/NovaGPUTileDispatchUsingForall.cpp`
**Lines**: 632-649

**Current code**:
```cpp
if (numElements > 1024) {
    op->emitWarning("large op (") << numElements << " elements) serialized to 1 workgroup — missing lowering_config?";
}
```

**Refactored code**:
```cpp
if (numElements > 1024) {
    // Skip warnings for ops that are expected to be in single-block foralls:
    // - Ops WITH lowering_config: intentionally configured (may have all-zero wg tiles)
    // - Fill ops: always fuse as DPS init; single-block is expected fallback
    if (!getLoweringConfig(op) && !isa<linalg::FillOp>(op)) {
        op->emitWarning("large op (") << numElements
            << " elements) serialized to 1 workgroup — missing lowering_config?";
    }
}
```

**Impact**: Eliminates 73 false-positive warnings (41 generic-with-config + 26 batch_matmul-producers + 6 matmul-producers).

### Phase 4: Fix Prologue Fusion in TileAndDistribute

**File**: `Nova-Compiler/lib/Transforms/LLVMGPU/NovaGPUTileDispatchUsingForall.cpp`

The 28 LN-normalize and weight-broadcast generics are correctly identified as prologues (no config) but fail to fuse during Phase 1a's producer fusion. Root cause: the fusion control function (line 350-355) blocks producers with reduction iterators. The LN normalize is all-parallel, so it should fuse — but its OWN producers (mean/variance reductions) are blocked.

**Diagnostic**: Add LLVM_DEBUG output to the fusion control function:
```cpp
LLVM_DEBUG(llvm::dbgs() << "[nova-tile-dispatch] Fusion decision for "
                         << producerOp->getName() << ": ");
```

**Potential fixes**:
1. **Allow partial producer chains**: Fuse the all-parallel normalize even though its reduction producers can't fuse. The normalize reads from the reduction's result (a scalar per row), which is computed before the forall. The extract_slice of the normalize should produce a valid slice that reads from the already-computed reduction result.
2. **Fallback config for failed prologues**: After Phase 1, walk remaining ops. If they were identified as prologues but are still outside foralls, attach a default config and re-run Phase 1b.

### Phase 5: Add setTransposeConfig for Non-Prologue Transposes

**File**: `Nova-Compiler/lib/Transforms/LLVMGPU/NovaKernelConfig.cpp`

For transposes that are NOT prologues (e.g., standalone data reshuffles in backward pass), `setDefaultConfig` may not produce optimal tiles since transpose is pure data movement with no computation.

**Add specialized config** (optional, `setDefaultConfig` works as fallback):
```cpp
static LogicalResult setTransposeConfig(linalg::LinalgOp op,
                                         const NVIDIATargetInfo &target) {
    auto transposeOp = dyn_cast<linalg::TransposeOp>(op.getOperation());
    if (!transposeOp)
        return failure();
    // Tile outer dims for parallelism, inner dim for coalescing
    // Shape [8x1024x1536]: wg=[1,1,512], thread=[0,0,4] → 8×1024×3=24576 blocks
    // Shape [8192x384]:    wg=[1,128],    thread=[0,2]   → 8192×3=24576 blocks
    return setDefaultConfig(op, target);  // Default is good enough for now
}
```

### Phase 6: Verify Fusion Chains End-to-End

After implementing Phases 1-4, verify by reading `debug2.log`:

1. **Check Step 0 output**: All ops should either HAVE a config OR be identified as prologue/epilogue/fill.
2. **Check Phase 1 output**: All prologues should be inside contraction foralls.
3. **Check Phase 2 output**: Only small ops (<1024 elements) or explicitly-single-block ops should be wrapped.
4. **Warning count**: Should be 0 or near-0.

**Verification commands**:
```bash
# After build + run:
grep -c "serialized to 1 workgroup" debug2.log  # Target: 0
grep "Skipping prologue" debug2.log | wc -l      # Should match prologue count
grep "Skipping epilogue" debug2.log | wc -l      # Should match epilogue count
```

---

## 5. Expected Warning Reduction

| Phase | Warnings Before | Warnings After | What's Fixed |
|-------|----------------|---------------|-------------|
| Phase 1: Remove GenericOp guard | 118 | 101 | TransposeOp (7) + FillOp (10) get proper handling |
| Phase 3: Suppress false positives | 101 | 28 | Config-with-zero (41) + producer warnings (32) silenced |
| Phase 4: Fix prologue fusion | 28 | 0 | LN normalize (24) + weight broadcast (4) properly fuse |
| **Total** | **118** | **0** | |

---

## 6. Performance Impact

### Currently Serialized Ops (single-block execution)

| Op Type | Count | Elements per Op | Total Elements Serialized |
|---------|-------|----------------|--------------------------|
| LN normalize | 24 | 3,145,728 | 75.5M |
| Weight broadcast | 4 | 4,718,592 | 18.9M |
| Transpose (backward) | 7 | 589,824–3,145,728 | ~15M |
| Fill | 10 | varies | negligible (memset) |

**After fix**: All 24 LN normalize + 4 weight broadcast ops fuse into contraction foralls → zero additional kernels. 7 transposes get multi-block execution (e.g., 96-1536 blocks each).

### Kernel Count Impact

Current: ~86 kernels
Expected after fix: ~86 kernels (prologues fuse, transposes become multi-block but same kernel count)
Performance improvement: transpose ops run ~100-1500× faster (multi-block vs single-block)

---

## 7. File Change Summary

| File | Lines Changed | Description |
|------|--------------|-------------|
| `NovaKernelConfig.cpp:786-802` | ~15 lines | Remove GenericOp guard, add FillOp skip |
| `NovaKernelConfig.cpp:714` | 1 line | GenericOp → LinalgOp in epilogue chain walk |
| `NovaGPUTileDispatchUsingForall.cpp:632-649` | ~5 lines | Config-aware + FillOp-aware warning suppression |
| `NovaGPUTileDispatchUsingForall.cpp:350` | ~10 lines | Debug output for fusion decisions (diagnostic) |

**Total**: ~30 lines of changes across 2 files.

---

## 8. Implementation Order

```
Phase 1 + Phase 2 (NovaKernelConfig)  ──┐
                                         ├── Build & Test
Phase 3 (Warning suppression)          ──┘
         │
         ▼
    Read debug2.log → verify warning count
         │
         ▼
Phase 4 (Prologue fusion diagnostic)  ── Build & Test
         │
         ▼
    Read debug2.log → verify fusion chains
         │
         ▼
Phase 5 (Optional: transpose config)  ── Build & Test if needed
         │
         ▼
Phase 6 (End-to-end verification)
```
