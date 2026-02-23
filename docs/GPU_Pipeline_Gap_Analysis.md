# Nova vs IREE — Tile-and-Fuse Pipeline Gap Analysis

This document answers the question: **"Which IREE Tile-and-Fuse pipeline components do we have, and which are missing?"**

Reference: `iree/compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp → addGPUTileAndFusePassPipeline`

---

## Summary Table

| Component | IREE Pass | Nova Status |
|---|---|---|
| **Lowering Config Attribute** | `IREEGPU_LoweringConfigAttr` | ✅ `Nova_LoweringConfigAttr` in `NovaOps.td` |
| **Config Selection (Strategy)** | `LLVMGPUSelectLoweringStrategy` | ✅ `nova-gpu-select-lowering-strategy` |
| **Fusion Barrier** | `IREE::Codegen::FusionBarrierOp` | ✅ `nova.fusion_barrier` in `NovaOps.td` |
| **Tile and Distribute (Workgroup)** | `TileAndDistributeToWorkgroups` | ✅ `nova-tile-and-distribute` |
| **Pad Operands** | `GPUPadOperandsPass` | ✅ `nova-gpu-pad-operands` |
| **Promote Matmul Operands** | `GPUPromoteMatmulOperandsPass` | ✅ `nova-gpu-promote-matmul-operands` |
| **K-Reduction Tiling** | `GPUApplyTilingLevel(Reduction)` | ✅ `nova-gpu-apply-tiling-level-reduction` |
| **Thread Tiling** | `GPUApplyTilingLevel(Thread)` | ⚠️ Pass exists but **not wired into pipeline** |
| **Subgroup Tiling** | `GPUApplyTilingLevel(Subgroup)` | ⚠️ Pass exists but **not wired into pipeline** |
| **Config Propagation (Canonicalizer)** | `ConfigTrackingCanonicalizerPass` | ❌ **MISSING** |
| **Bufferization** | GPU-aware `IREEComprehensiveBufferize` | ⚠️ **Partial** — uses generic `OneShotBufferize` (no GPU memory spaces) |
| **Pack To Intrinsics** | `GPUPackToIntrinsicsPass` | ❌ **MISSING** |
| **Distribute Forall to Threads** | `GPUDistributeForallPass` | ❌ **MISSING** |
| **Unroll to Intrinsics** | `GPU::UnrollToIntrinsicsPass` | ❌ **MISSING** |
| **GPU Infer Memory Space** | `GPUInferMemorySpacePass` | ❌ **MISSING** |

---

## 1. ✅ What We Have

### 1.1 Lowering Config (`Nova_LoweringConfigAttr`)
A `DictionaryAttr` attached to linalg ops containing `workgroup`, `reduction`, `mma_kind`, and `promoted_operands` fields. Lives in `NovaOps.td`.

### 1.2 Config Selection (`nova-gpu-select-lowering-strategy`)
Reads `--cuda-arch=sm_XX`, queries `NVIDIATargetInfo`, runs MMA/SIMT heuristic, and stamps a `lowering_config` dict onto every `linalg.matmul`. Lives in `NovaGPUSelectLoweringStrategy.cpp`.

### 1.3 Fusion Barrier (`nova.fusion_barrier`)
A pure identity op in `NovaOps.td`. Inserted by `NovaGPUPromoteMatmulOperands` between the global→shared Stage-1 copy and the per-thread Stage-2 copy. Prevents loop-fusion from merging the two copy stages.

### 1.4 Tile-and-Distribute, Pad, Promote, K-Tiling
All wired in `Passes.cpp` and confirmed passing tests.

---

## 2. ⚠️ Partial — Bufferization

### IREE's Approach (`addGPUBufferizePasses`)
IREE uses a **GPU-aware** bufferization sequence:
1. `createEliminateEmptyTensorsPass()` — removes unnecessary `tensor.empty`
2. `createGPUInferMemorySpacePass()` — infers `#gpu.address_space<workgroup>` or `<private>` on tensors based on their usage context (inside workgroup forall = workgroup, inside thread forall = private)
3. `createIREEComprehensiveBufferizePass(gpuAllocationFn, gpuCopyFn)` — bufferizes using a custom allocation function that maps tensors to the correct GPU address spaces and inserts `gpu.barrier` barriers on copies to/from shared memory automatically

### Nova's Current State
Nova uses the simpler generic `bufferization::createOneShotBufferizePass()`. This works but:
- **Does not automatically insert `gpu.barrier`** around shared memory copies
- **Does not map new allocations to GPU address spaces** (workgroup/private) based on context
- **Will produce incorrect multi-threaded GPU code** if used without explicit barriers

**What needs to be added:** A custom `gpuAllocationFn` and `gpuCopyFn` pair (like IREE's), and the `GPUInferMemorySpacePass`.

---

## 3. ❌ Config Propagation (`ConfigTrackingCanonicalizerPass`)

### What It Does
In IREE, every call to `canonicalize` during the pipeline is replaced with `ConfigTrackingCanonicalizerPass`. This is a modified canonicalize pass that hooks a `ConfigTrackingListener` into the rewriter. Whenever a linalg op is **replaced** by a canonicalization pattern (e.g., during tiling, a `linalg.matmul` may be cloned into a new op), the listener detects the replacement and **copies the `lowering_config` attribute from the old op to the new op**.

### Why It Matters
After K-tiling, the original `linalg.matmul` is replaced by a new one inside the `scf.for` loop. Without config propagation, the new op **loses its `lowering_config` attribute** — so thread tiling, subgroup tiling, and MMA intrinsic selection would read nothing from it.

### Nova's Current State
Nova uses plain `createCanonicalizerPass()` which **does not propagate the config**. This means Nova's Thread and Subgroup tiling passes currently cannot reliably read config from ops that were tiled in a previous step.

**What needs to be added:** A `NovaConfigTrackingCanonicalizerPass` that wraps the standard canonicalizer with a rewriter listener that copies `lowering_config` on op replacement. This is approximately 60 lines of code (see `ConfigTrackingCanonicalizer.cpp` in IREE for the exact pattern).

---

## 4. ❌ Pack To Intrinsics (`GPUPackToIntrinsicsPass`)

### What It Does
After promoting operands to shared memory, ops need to be **repacked** so their inner dimensions (`M`, `N`, `K` of each tile) match the exact shape expected by the hardware MMA intrinsic (e.g., 16×16×16 for WMMA). This pass inserts `linalg.pack` and `linalg.unpack` ops to reshape the tiled tensors accordingly.

### Why It Matters
Without this pass, even if we selected the correct MMA intrinsic, the vectorized matmul tiles won't have the right shapes for `gpu.subgroup_mma_compute` / WMMA ops to accept them.

**What needs to be added:** A `NovaGPUPackToIntrinsicsPass` that reads the `mma_kind` from the `lowering_config` attr, looks up the MMA shape from `NVIDIATargetUtils`, and inserts the appropriate pack/unpack ops.

---

## 5. ❌ Distribute Forall to Threads (`GPUDistributeForallPass`)

### What It Does
After all tensor-level tiling, the pipeline has `scf.forall` ops with `gpu.thread` mapping attributes. This pass **lowers those `scf.forall` ops to `gpu.thread_id` index arithmetic**, effectively distributing the loop iterations to individual CUDA threads.

### Why It Matters
Without this, the generated MLIR has `scf.forall` loops with thread mappings that cannot be lowered to NVVM/PTX since NVVM doesn't understand `scf.forall`.

**What needs to be added:** Wire in the existing MLIR upstream `gpu::GPUDistributePass` or a Nova-specific wrapper that handles the `gpu.thread` and `gpu.warp` mapping attributes.

---

## 6. ❌ Unroll to Intrinsics (`GPU::UnrollToIntrinsicsPass`)

### What It Does
After vectorization, vector ops are still at the level of the workgroup tile (e.g., `vector<128x128xf16>`). This pass **unrolls them into the hardware intrinsic tile size** (e.g., `vector<16x16xf16>` for each WMMA call), generating the final sequence of `gpu.subgroup_mma_load_matrix` / `gpu.subgroup_mma_compute` ops.

### Why It Matters
This is the final lowering step that connects the high-level vector ops to actual hardware instructions. Without it, the IR remains at an abstraction level that cannot be compiled to PTX.

**What needs to be added:** Integrate `IREE::GPU::createUnrollToIntrinsicsPass()` or implement a simplified equivalent that applies `VectorUnrollPattern` with the MMA tile constraints.

---

## Priority Ordering

For the pipeline to produce **correct** GPU code:

1. **Config Propagation** (most urgent) — without it, Thread/Subgroup tiling is broken after K-tiling replaces ops
2. **GPU-aware Bufferization** — without barriers, shared memory is racey
3. **Distribute Forall** — without it, code cannot be lowered to NVVM

For the pipeline to produce **fast** GPU code (MMA utilization):

4. **Pack To Intrinsics** — reshape data to match MMA shape requirements
5. **Unroll To Intrinsics** — emit actual MMA calls
