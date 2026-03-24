# Nova GPU Memory Pipeline — Refactoring Plan

> **Date**: 2026-03-20
> **Status**: Planning (no code changes)
> **Reference**: IREE `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

---

## Table of Contents

1. [Current Nova Pipeline (Full Step Map)](#1-current-nova-pipeline-full-step-map)
2. [IREE Reference Pipeline (Memory-Focused)](#2-iree-reference-pipeline-memory-focused)
3. [Gap Analysis — What Nova Is Missing](#3-gap-analysis--what-nova-is-missing)
4. [Phase 1: Enable PromoteMatmulOperands](#4-phase-1-enable-promotematmuloperands)
5. [Phase 2: Add HoistStaticallyBoundAllocations](#5-phase-2-add-hoiststaticallyboundallocations)
6. [Phase 3: Add PadDynamicAlloc](#6-phase-3-add-paddynamicalloc)
7. [Phase 4: Fix Barrier Insertion Pipeline](#7-phase-4-fix-barrier-insertion-pipeline)
8. [Phase 5: Add GPUReuseSharedMemoryAllocs](#8-phase-5-add-gpureusesharedmemoryallocs)
9. [Phase 6: Add GPUCheckResourceUsage](#9-phase-6-add-gpucheckresourceusage)
10. [Phase 7: Simplify ConvertSharedMemAllocs](#10-phase-7-simplify-convertsharedmemallocs)
11. [Target Pipeline After Refactoring](#11-target-pipeline-after-refactoring)
12. [Execution Order & Dependencies](#12-execution-order--dependencies)

---

## 1. Current Nova Pipeline (Full Step Map)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NOVA DIALECT → LINALG                            │
│  NovaToArith → NovaToTosa → NovaElementwiseToLinalg → NovaToLinalg │
│  TosaToLinalgNamed → TosaToLinalg → TosaToArith/Tensor/SCF         │
│  Canonicalize                                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step -1   : LinalgElementwiseOpFusion                              │
│              Fuse elementwise chains (exp→log) before per-op tiling │
│  Step -0.5 : LinalgFoldUnitExtentDims                               │
│              Collapse keepdims softmax shapes for uniform tiling     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step 0  : SelectLoweringStrategy                                   │
│            Stamps #nova.lowering_config on each op:                  │
│              contractions: workgroup, reduction, thread, subgroup,   │
│                            mma_kind, promoted_operands=[0,1],       │
│                            padding                                   │
│              reductions:   workgroup, reduction, thread              │
│                            promoted_operands=[] (empty)              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                     TILING & DISTRIBUTION                            │
│                                                                      │
│  Step 1  : TileAndDistribute                                         │
│            → scf.forall {block} per workgroup                        │
│                                                                      │
│  Step 2  : PadOperands                                               │
│            Pad A/B/C to static multiples of tile sizes               │
│            Reads "padding" from lowering_config                      │
│                                                                      │
│  Step 3  : PromoteMatmulOperands  ← ──── DISABLED (commented out)   │
│            Two-stage copy: global→shared + fusion_barrier +          │
│            per-thread copy. Config-driven via promoted_operands.     │
│            MUST run BEFORE K-tiling (Step 4).                        │
│                                                                      │
│  Step 4  : ApplyTilingLevelReduction                                 │
│            K-dim → scf.for loop. If Step 3 were active, this tiles   │
│            the promoted copies along K automatically.                │
│                                                                      │
│  Step 5  : ApplyTilingLevelThread                                    │
│            Per-thread M/N register tiles                             │
│                                                                      │
│  Step 6  : FuseAndHoistParallelLoops                                 │
│  Step 7  : NormalizeLoopBounds (lb=0, step=1)                        │
│  Step 7.75: GeneralizeNamedOps + ElementwiseOpFusion                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                      BUFFERIZATION                                   │
│                                                                      │
│  Step 8  : NovaGPUBufferize  (tensor → memref)                       │
│            • Erases nova.fusion_barrier ops inline before buffering  │
│            • gpuAllocationFn routes:                                  │
│                workgroup → memref.alloc (shared)                     │
│                private   → memref.alloca (per-thread, inside forall) │
│                private at func scope → memref.alloc (global fallback)│
│                no space  → memref.alloc (global)                     │
│            • gpuCopyFn:                                               │
│                inside forall → linalg.copy (lower to loops later)    │
│                host side     → memref.copy                           │
│            • NOTE: does NOT insert gpu.barrier in copy function       │
│              (breaks OneShotBufferize analysis)                      │
│                                                                      │
│  Step 8.5: NormalizeLoopBounds (remove degenerate 1×1 foralls)       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                   GPU LOWERING & OUTLINING                            │
│                                                                      │
│  Step 9   : ForallToGPU (scf.forall → gpu.launch)                   │
│  Step 10  : LinalgToLoops (linalg.copy → scf.for + load/store)       │
│  Step 11  : InsertWorkgroupBarriers                                  │
│             Scans for workgroup memory write→read transitions        │
│             ⚠️ PROBLEM: only sees memrefs already tagged workgroup;  │
│             buffers promoted to shared LATER (step 12.5) get no      │
│             barriers                                                  │
│                                                                      │
│  Step 12  : GpuKernelOutlining → gpu.module + gpu.func               │
│  Step 12.5: ConvertSharedMemAllocs (inside gpu.module)               │
│             Three-tier strategy:                                      │
│               ≤1KB   → memref.alloca (per-thread stack)              │
│               1-48KB → memref.global("__global_memory__")            │
│               >48KB  → memref.global("__global_memory_large__")      │
│             ⚠️ PROBLEM: promotes per-thread allocs to SINGLE shared  │
│             buffer → data race across all 256 threads                │
│             ⚠️ PROBLEM: dynamic allocs needed ad-hoc fix to hoist    │
│             to entry block                                            │
│                                                                      │
│  Step 12.75: ConvertMemRefToGpu (host cross-kernel → gpu.alloc)      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                     LLVM/NVVM LOWERING                               │
│                                                                      │
│  Step 13: (inside gpu.module)                                        │
│    13.0  AttachNVVMTarget (sm_86, +ptx76, O3, fast-math, ftz)       │
│    13.1  GpuAsyncRegion                                              │
│    13.2  ExpandStridedMetadata → LowerAffine → LowerMemorySpace →   │
│           SCFToCF → GpuToNVVM → IndexToLLVM → ArithToLLVM →         │
│           MathToLLVM → ReconcileUnrealizedCasts →                    │
│           PromoteGlobalsToShared                                     │
│           (converts __global_memory__ AS 0 → AS 3 shared)           │
│    13.3  GpuModuleToBinary (PTX ISA)                                 │
│    13.4  LowerMemorySpace (host-side) → GpuToLLVM →                 │
│           ReconcileUnrealizedCasts                                   │
│    13.5  Host lowering: SCFToCF → CFToLLVM → ArithToLLVM →          │
│           ExpandStridedMetadata → FinalizeMemRefToLLVM →             │
│           FuncToLLVM → GenerateDynamicWrapper →                      │
│           ReconcileUnrealizedCasts                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Registered But Unused Passes

| Pass | File | Status |
|------|------|--------|
| `NovaGPUPromoteMatmulOperandsPass` | `NovaGPUPromoteMatmulOperands.cpp` | **Commented out** at Step 3 |
| `NovaGPUApplyTilingLevelSubgroupPass` | — | **Commented out** (warp tiling not tuned) |
| `NovaGPUEraseFusionBarriersPass` | `NovaGPUEraseFusionBarriers.cpp` | **Not in pipeline** (fusion barriers erased inline in `NovaGPUBufferize`) |
| `NovaGPUInferMemorySpacePass` | `NovaGPUInferMemorySpace.cpp` | **Registered but not in pipeline** (no forall with thread mappings at expected point) |
| `NovaEliminateEmptyTensorsPass` | `NovaEliminateEmptyTensors.cpp` | **Registered but not in pipeline** |

---

## 2. IREE Reference Pipeline (Memory-Focused)

IREE's `addGPUTileAndFusePassPipeline` handles memory in 6 distinct phases:

### Phase A: Pre-Bufferization Memory Tagging

```
GPUPadOperands
GPUPromoteMatmulOperands          ← inserts alloc_tensor(workgroup) + copy + barrier
GPUPackToIntrinsics
ReductionTiling                   ← K-tiling tiles the promoted copies
ThreadTiling
FuseAndHoist
GPUInferMemorySpacePass           ← tags remaining tensors with memory spaces
GPUTensorAllocPass                ← inserts alloc_tensor with spaces from inference
```

**Key insight**: Memory space decisions happen at the **tensor level**, before bufferization. The allocation function (`gpuAllocationFn`) then mechanically routes based on the already-decided memory space.

### Phase B: Bufferization

```
EliminateEmptyTensors             ← reduces unnecessary allocations
OneShotBufferize                  ← with gpuAllocationFn + gpuCopyFn
  gpuAllocationFn:
    workgroup → memref.alloc
    private   → memref.alloca
  gpuCopyFn:
    inserts gpu.barrier BEFORE copy into workgroup memory
```

### Phase C: Post-Bufferization Allocation Cleanup

```
HoistStaticallyBoundAllocations   ← runs FIRST time (hoists allocs out of loops)
PadDynamicAlloc                   ← DataFlowSolver + IntegerRangeAnalysis → static
HoistStaticallyBoundAllocations   ← runs AGAIN (hoists newly-static allocs)
```

**Key insight**: `HoistStaticallyBoundAllocations` runs **6+ times** throughout the IREE pipeline. It's cheap and catches allocs exposed by other transformations.

### Phase D: Shared Memory Optimization

```
LLVMGPUPackSharedMemoryAlloc      ← bank-conflict-free swizzling
GPUReuseSharedMemoryAllocs        ← liveness-based packing into single buffer
```

### Phase E: Barrier Management

```
gpu.barrier from gpuCopyFn        ← inserted DURING bufferization copies
createGpuEliminateBarriers()      ← removes redundant barriers (alias analysis)
```

### Phase F: Resource Validation

```
GPUCheckResourceUsage             ← sums shared memory, compares to device limit
```

---

## 3. Gap Analysis — What Nova Is Missing

| # | IREE Feature | Nova Status | Impact |
|---|---|---|---|
| **G1** | `GPUPromoteMatmulOperands` | **EXISTS but DISABLED** (`Passes.cpp:197`) | Root cause of shared memory race condition + alloca-in-loop crash |
| **G2** | `HoistStaticallyBoundAllocations` | **MISSING** — no equivalent pass | Allocs inside loops survive to PTX → stack overflow |
| **G3** | `PadDynamicAlloc` | **MISSING** — ad-hoc fix in `ConvertSharedMemAllocs` | Dynamic allocs need principled upper-bound inference |
| **G4** | `GPUInferMemorySpace` | **EXISTS but NOT IN PIPELINE** | No forall with thread mappings at expected insertion point |
| **G5** | `EliminateEmptyTensors` | **EXISTS but NOT IN PIPELINE** | Unnecessary buffer allocations persist |
| **G6** | `GPUReuseSharedMemoryAllocs` | **MISSING** | Each shared alloc gets separate `memref.global` |
| **G7** | `GPUCheckResourceUsage` | **MISSING** | Silent corruption when exceeding 48KB shared memory |
| **G8** | Barrier insertion during `gpuCopyFn` | **NOT DONE** (comment: "breaks OneShotBufferize") | Barriers inserted too late (Step 11), miss buffers promoted at Step 12.5 |
| **G9** | `createGpuEliminateBarriers()` | **MISSING** | Redundant barriers waste GPU cycles |
| **G10** | `EraseFusionBarriers` in pipeline | **DONE INLINE** in `NovaGPUBufferize` (line 6-8) | Works, but standalone pass exists and is unused |

### Root Cause Chain

```
G1 (Promote disabled) ──→ matmul operands stay in global memory
                        ──→ ConvertSharedMemAllocs retroactively promotes
                        ──→ per-thread alloc → single shared buffer = DATA RACE
                        ──→ dynamic alloc inside K-loop = ALLOCA STACK OVERFLOW
                        ──→ no barriers around shared loads = WRONG RESULTS
```

**Enabling G1 breaks this entire chain.** Most other gaps become minor cleanup.

---

## 4. Phase 1: Enable PromoteMatmulOperands

### What

Uncomment the three lines at `Passes.cpp:197-199`:

```cpp
// Step 3: currently disabled
pm.addNestedPass<func::FuncOp>(createNovaGPUPromoteMatmulOperandsPass());
pm.addNestedPass<func::FuncOp>(createNovaConfigTrackingCanonicalizerPass());
pm.addPass(createCSEPass());
```

### Prerequisites

1. **`SelectLoweringStrategy` already sets `promoted_operands = [0, 1]`** for all matmul contractions (`NovaKernelConfig.cpp:274`). No changes needed.

2. **`NovaGPUPromoteMatmulOperandsPass` is fully implemented** (`NovaGPUPromoteMatmulOperands.cpp`):
   - Reads `promoted_operands` from lowering config
   - For input operands: two-stage copy (global→shared alloc_tensor + fusion_barrier + per-thread copy)
   - For result operands: shared alloc + copy + barrier
   - Skips fill producers
   - Skips existing linalg producers (thread tiling handles them)

3. **`nova.fusion_barrier` is erased inline** at the start of `NovaGPUBufferize::runOnOperation()` (line 6-8 of the pass). No separate `EraseFusionBarriers` pass needed in the pipeline.

### What Changes in the IR

**Before (current — no promotion):**
```mlir
// K-loop body:
%slice_A = tensor.extract_slice %A[..., %k, ...] : tensor<128x32xf32>
%slice_B = tensor.extract_slice %B[%k, ..., ...] : tensor<32x128xf32>
%matmul = linalg.generic {contraction} ins(%slice_A, %slice_B) ...
```

**After (with promotion enabled):**
```mlir
// BEFORE K-tiling (Step 3), on the FULL workgroup slice:
%alloc_A = bufferization.alloc_tensor {memory_space = #gpu.address_space<workgroup>}
%shared_A = linalg.copy(%A_slice → %alloc_A)         // Stage 1: cooperative global→shared
%fence_A = nova.fusion_barrier %shared_A              // Prevent fusion
%local_A = linalg.copy(%fence_A → tensor.empty())     // Stage 2: per-thread shared→register

// Then K-tiling (Step 4) tiles the copies along K:
scf.for %k = 0 to K step kStep {
  %tile_A = tensor.extract_slice %local_A[..., %k, ...]  // tiled copy
  %tile_B = tensor.extract_slice %local_B[%k, ..., ...]
  %matmul = linalg.generic {contraction} ins(%tile_A, %tile_B) ...
}
```

### What This Fixes

| Bug | How Promotion Fixes It |
|-----|------------------------|
| **Shared memory race condition** | Operands get explicit `workgroup` memory space → `gpuAllocationFn` creates `memref.alloc` (not `alloca`) → `ConvertSharedMemAllocs` sees it's already tagged → cooperative copy semantics preserved |
| **Alloca-in-loop crash** | With promotion BEFORE K-tiling, the shared alloc is on the full slice; K-tiling only tiles the *copy*, not the *alloc*. The alloc stays outside the K-loop naturally. |
| **Missing barriers** | `gpuCopyFn` in bufferization wraps workgroup copies with `gpu.barrier` (currently commented — see Phase 4). Even without that, `InsertWorkgroupBarriers` (Step 11) can now see the workgroup-tagged memrefs. |

### Pipeline Placement

```
Step 2  : PadOperands
Step 3  : PromoteMatmulOperands  ← ENABLE HERE
Step 3a : ConfigTrackingCanonicalize + CSE
Step 4  : ApplyTilingLevelReduction (K-tiling)
```

### Risk & Validation

- **Risk**: The pass was disabled for a reason — likely an interaction with an incomplete lowering config or a missing downstream pass. The IR after promotion contains `bufferization.alloc_tensor` with workgroup space, which `NovaGPUBufferize` must handle correctly.
- **Validation**: Run `make run-snippet FILE=Tests/IrTest/atest_chain_forward.cpp` with `options.verbose = true` and inspect:
  1. Post-Step-3 IR: verify `alloc_tensor(workgroup)` + `linalg.copy` + `fusion_barrier` appear
  2. Post-Step-4 IR: verify the copies are tiled along K
  3. Post-Step-8 IR: verify `memref.alloc` with workgroup space (not `memref.alloca`)
  4. PTX: verify no `alloca.u64` inside loops, verify `.shared` buffers have cooperative stores

---

## 5. Phase 2: Add HoistStaticallyBoundAllocations

### What

A pass that walks every function/gpu.func, finds `memref.alloc`/`memref.alloca` where all sizes are statically known (or all dynamic sizes are defined outside the innermost loop), and moves them to the function entry block.

### Why

Even with Phase 1, non-matmul allocs (scratch buffers, reduction intermediates) may still be created inside loops by bufferization. Without hoisting, they become PTX `alloca` inside loops → stack overflow.

IREE runs this pass **6+ times** throughout its pipeline — it's cheap and catches allocs exposed by every transformation.

### IREE Reference

```cpp
// From IREE's Common/GPU/Passes.cpp
// ~50 lines: walk allocs, check dominance, move to entry block
void HoistStaticallyBoundAllocationsPass::runOnOperation() {
  auto funcOp = getOperation();
  Block &entryBlock = funcOp.getBody().front();
  for (auto allocOp : llvm::make_early_inc_range(funcOp.getOps<memref::AllocaOp>())) {
    if (allSizesDominateBlock(allocOp, entryBlock)) {
      allocOp->moveBefore(&entryBlock, entryBlock.begin());
    }
  }
  // Same for memref::AllocOp
}
```

### Pipeline Placement

Run in **3 locations**:

```
Step 8   : NovaGPUBufferize
Step 8.1 : HoistStaticallyBoundAllocations  ← NEW (first run, post-bufferize)
Step 8.5 : NormalizeLoopBounds
...
Step 12  : GpuKernelOutlining
Step 12.1: HoistStaticallyBoundAllocations  ← NEW (second run, post-outline)
Step 12.5: ConvertSharedMemAllocs
...
Step 13.2: (inside gpu.module, after LowerAffine)
           HoistStaticallyBoundAllocations  ← NEW (third run, pre-LLVM)
```

### Complexity

**Low** (~50-80 lines). Pure mechanical hoist based on dominance analysis. No dataflow solver needed.

### What This Fixes

- Any remaining allocs inside K-loops (non-matmul scratch buffers)
- Allocs exposed by canonicalization after outlining
- Safety net for any future pass that creates allocs inside loops

---

## 6. Phase 3: Add PadDynamicAlloc

### What

Replace dynamic-sized `memref.alloc(%d)` with static-sized `memref.alloc()` + `memref.subview`, using MLIR's `IntegerRangeAnalysis` to compute tight upper bounds on dynamic dimensions.

### Why

Currently, `ConvertSharedMemAllocs` has an **ad-hoc** upper-bound inference (our recent fix) that handles `arith.minsi`, `arith.minui`, and `arith.constant` patterns. This works for the current K-loop pattern but is brittle.

IREE's `PadDynamicAlloc` uses the **DataFlowSolver** with `IntegerRangeAnalysis`, which handles arbitrary chains of arith ops, loop induction variables, and affine expressions — far more robust.

### IREE Reference

```cpp
// From IREE's Common/GPU/GPUPadDynamicAlloc.cpp
void PadDynamicAllocPass::runOnOperation() {
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  solver.initializeAndRun(getOperation());

  funcOp.walk([&](memref::AllocOp allocOp) {
    SmallVector<int64_t> staticShape;
    for (auto [idx, dim] : enumerate(allocOp.getType().getShape())) {
      if (!ShapedType::isDynamic(dim)) {
        staticShape.push_back(dim);
        continue;
      }
      Value dynSize = allocOp.getDynamicSizes()[dynIdx++];
      auto *lattice = solver.lookupState<IntegerValueRangeLattice>(dynSize);
      if (lattice && !lattice->getValue().isUninitialized()) {
        APInt upper = lattice->getValue().getValue().smax();
        staticShape.push_back(upper.getSExtValue());
      } else {
        staticShape.push_back(kDefaultPadSize);  // 64 bytes
      }
    }
    // Replace with static alloc + subview
  });
}
```

### Pipeline Placement

```
Step 8   : NovaGPUBufferize
Step 8.1 : HoistStaticallyBoundAllocations  (Phase 2)
Step 8.2 : PadDynamicAlloc                   ← NEW
Step 8.3 : HoistStaticallyBoundAllocations   ← run again (newly-static allocs)
Step 8.5 : NormalizeLoopBounds
```

### Complexity

**Medium** (~100-150 lines). Requires setting up `DataFlowSolver` and `IntegerRangeAnalysis`, both provided by MLIR core. The pattern-matching and replacement is straightforward.

### What This Fixes

- Removes the ad-hoc upper-bound inference in `ConvertSharedMemAllocs`
- Handles arbitrary dynamic size expressions (not just `minsi`/`constant`)
- The `ConvertSharedMemAllocs` dynamic-alloc handler can be simplified (see Phase 7)

---

## 7. Phase 4: Fix Barrier Insertion Pipeline

### Problem

Nova's barrier insertion has **two timing problems**:

1. **`InsertWorkgroupBarriers` (Step 11) runs BEFORE `ConvertSharedMemAllocs` (Step 12.5)**
   - Buffers are still `memref.alloc` in default address space at Step 11
   - They get promoted to shared (`memref.global`) at Step 12.5
   - Result: **no barriers around shared memory accesses** for these buffers

2. **`gpuCopyFn` does NOT insert barriers** (comment in `NovaGPUBufferize.cpp:175-176`: "breaks OneShotBufferize analysis")
   - IREE inserts `gpu.barrier` in its `gpuCopyFn` — Nova skips this

### Solution: Three Sub-Steps

#### 4A: Tag Workgroup Memory Before Bufferization

With Phase 1 (PromoteMatmulOperands) enabled, matmul operands already get `workgroup` memory space via `bufferization.alloc_tensor(memory_space = workgroup)`. The bufferization allocation function routes these to `memref.alloc` with workgroup space.

For **non-matmul** workgroup buffers, re-enable `NovaGPUInferMemorySpacePass` at the right point:

```
Step 7.75 : GeneralizeNamedOps + ElementwiseOpFusion
Step 7.9  : NovaGPUInferMemorySpacePass   ← RE-ENABLE HERE
            (now runs AFTER ForallToGPU creates scf.forall with
             thread mappings — or after tiling creates the structure
             the pass needs to detect)
Step 8    : NovaGPUBufferize
```

**Alternative**: If the pass still can't detect thread-mapped foralls at this point, move it to run **after Step 9 (ForallToGPU)** and re-bufferize just the tagged tensors.

#### 4B: Re-run InsertWorkgroupBarriers After ConvertSharedMemAllocs

Add a second barrier insertion pass after shared memory promotion:

```
Step 12.5 : ConvertSharedMemAllocs
Step 12.6 : InsertWorkgroupBarriers  ← NEW (second run)
Step 12.75: ConvertMemRefToGpu
```

This catches barriers for buffers that were promoted to shared memory by `ConvertSharedMemAllocs`.

#### 4C: Add Barrier Elimination (Optional Optimization)

```
Step 12.7 : GpuEliminateBarriers  ← NEW (uses MLIR upstream gpu-eliminate-barriers)
```

Removes redundant barriers using alias analysis. Reduces barrier overhead without sacrificing correctness.

### What This Fixes

| Bug | Sub-Step |
|-----|----------|
| Weight tile buffer has no barriers (shared memory race) | 4A (memory space tagging) + 4B (second barrier pass) |
| dGamma/dBeta = 0 in LN backward (compiled outputs all zero) | 4B (barriers around reduction shared buffers) |
| Redundant barriers in simple kernels | 4C (elimination) |

---

## 8. Phase 5: Add GPUReuseSharedMemoryAllocs

### What

Perform liveness analysis on shared memory allocations within each `gpu.func`, then pack non-overlapping allocations into a single shared memory buffer using offset subviews.

### Why

The fused LN→GELU→Linear kernel currently allocates:
- 16KB accumulator (`__global_memory___0`)
- 2KB weight tile (`__global_memory__`)

These may not overlap in lifetime. With reuse analysis, they could share the same 16KB buffer, freeing shared memory budget for other uses.

For larger models (D=768, D=1024), shared memory pressure is the primary bottleneck — this pass is critical for scaling.

### IREE Reference

```cpp
// From IREE's Common/GPU/GPUReuseSharedMemoryAllocs.cpp
// Computes live ranges using block-level liveness
// Sorts allocations by size (largest first)
// Greedily assigns offsets, reusing space from expired allocations
// Replaces original allocs with subviews into a single large buffer
```

### Pipeline Placement

```
Step 12.5 : ConvertSharedMemAllocs
Step 12.55: GPUReuseSharedMemoryAllocs  ← NEW
Step 12.6 : InsertWorkgroupBarriers (second run from Phase 4)
```

### Complexity

**Medium-High** (~200-300 lines). Requires:
1. Block-level liveness computation for memref values
2. Interval-based allocation packing (greedy bin-packing)
3. SubView offset computation for packed buffers

### What This Fixes

- Reduces total shared memory usage by 20-50% for fused kernels
- Prevents exceeding 48KB shared memory limit on complex fused ops
- Required before scaling to larger model dimensions

---

## 9. Phase 6: Add GPUCheckResourceUsage

### What

A validation pass that sums all shared memory allocations in each `gpu.func` and compares against the device's `maxSharedMemPerBlock` (48KB for sm_86). Emits a **compilation error** if exceeded.

### Pipeline Placement

```
Step 12.6 : InsertWorkgroupBarriers (second run)
Step 12.7 : GPUCheckResourceUsage  ← NEW (before LLVM lowering)
```

### IREE Reference

```cpp
// From IREE's LLVMGPU/GPUCheckResourceUsage.cpp
funcOp.walk([&](memref::AllocOp allocOp) {
  if (hasSharedMemoryAddressSpace(allocOp.getType())) {
    totalSharedMem += getAllocationSize(allocOp);
  }
});
// Also counts memref.global with shared address space
if (totalSharedMem > target.maxSharedMemPerBlock) {
  funcOp.emitError() << "shared memory usage (" << totalSharedMem
                      << ") exceeds device limit (" << target.maxSharedMemPerBlock << ")";
  return signalPassFailure();
}
```

### Complexity

**Very Low** (~30-50 lines). Walk + sum + compare. High value as a safety net.

### What This Fixes

- Catches silent corruption from exceeding shared memory limits
- Provides clear error message instead of CUDA_ERROR_ILLEGAL_ADDRESS at runtime
- Essential for development velocity (fail fast at compile time)

---

## 10. Phase 7: Simplify ConvertSharedMemAllocs

### What

After Phases 1-6, the `ConvertSharedMemAllocs` pass (`NovaConvertSharedMemAllocs.cpp`) can be **dramatically simplified**:

### Current Responsibilities (and what absorbs them)

| Current Responsibility | Absorbed By |
|------------------------|-------------|
| Dynamic alloc → static alloca at entry (ad-hoc fix) | **Phase 3** (`PadDynamicAlloc` + `HoistStaticallyBoundAllocations`) |
| Per-thread alloc → shared `memref.global` (Tier 2) | **Phase 1** (PromoteMatmulOperands tags workgroup space correctly) |
| Three-tier size routing (≤1KB / 1-48KB / >48KB) | **Keep but simplify**: only handles explicitly-tagged workgroup allocs |

### Simplified Logic

After refactoring, `ConvertSharedMemAllocs` only needs to:

1. Convert `memref.alloc` with workgroup address space → `memref.global` + `memref.get_global` (for shared memory in NVVM)
2. Drop `memref.dealloc` for shared memory
3. Convert remaining non-workgroup `memref.alloc` inside `gpu.func` → `memref.alloca` (per-thread)

The three-tier heuristic based on byte size becomes unnecessary — the memory space decision was already made correctly upstream.

---

## 11. Target Pipeline After Refactoring

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NOVA DIALECT → LINALG                            │
│  (unchanged)                                                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step -1   : LinalgElementwiseOpFusion                              │
│  Step -0.5 : LinalgFoldUnitExtentDims                               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step 0   : SelectLoweringStrategy                                  │
│             (sets promoted_operands=[0,1] for contractions)          │
│                                                                      │
│  Step 1   : TileAndDistribute                                        │
│  Step 2   : PadOperands                                              │
│  Step 3   : PromoteMatmulOperands        ← ENABLED (Phase 1)        │
│  Step 4   : ApplyTilingLevelReduction                                │
│  Step 5   : ApplyTilingLevelThread                                   │
│  Step 6   : FuseAndHoistParallelLoops                                │
│  Step 7   : NormalizeLoopBounds                                      │
│  Step 7.75: GeneralizeNamedOps + ElementwiseOpFusion                 │
│  Step 7.9 : InferMemorySpace             ← RE-ENABLED (Phase 4A)    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step 7.95: EliminateEmptyTensors         ← ENABLED (from G5)       │
│  Step 8   : NovaGPUBufferize                                         │
│             (gpuAllocationFn correctly routes workgroup→alloc,       │
│              private→alloca based on memory spaces from Step 3+7.9)  │
│  Step 8.1 : HoistStaticallyBoundAllocs    ← NEW (Phase 2, run 1)    │
│  Step 8.2 : PadDynamicAlloc               ← NEW (Phase 3)           │
│  Step 8.3 : HoistStaticallyBoundAllocs    ← NEW (Phase 2, run 2)    │
│  Step 8.5 : NormalizeLoopBounds                                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step 9   : ForallToGPU                                              │
│  Step 10  : LinalgToLoops                                            │
│  Step 11  : InsertWorkgroupBarriers       (first run)                │
│  Step 12  : GpuKernelOutlining                                       │
│  Step 12.1: HoistStaticallyBoundAllocs    ← NEW (Phase 2, run 3)    │
│  Step 12.5: ConvertSharedMemAllocs        (SIMPLIFIED — Phase 7)     │
│  Step 12.55: GPUReuseSharedMemoryAllocs   ← NEW (Phase 5)           │
│  Step 12.6: InsertWorkgroupBarriers       ← NEW (Phase 4B, 2nd run) │
│  Step 12.65: GpuEliminateBarriers         ← NEW (Phase 4C)          │
│  Step 12.7: GPUCheckResourceUsage         ← NEW (Phase 6)           │
│  Step 12.75: ConvertMemRefToGpu                                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│  Step 13 : LLVM/NVVM Lowering (unchanged)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### New Passes Summary

| Pass | Phase | Placement | Lines of Code | Complexity |
|------|-------|-----------|---------------|------------|
| PromoteMatmulOperands (enable) | 1 | Step 3 | 0 (uncomment) | None |
| HoistStaticallyBoundAllocations | 2 | Steps 8.1, 8.3, 12.1 | ~60 | Low |
| PadDynamicAlloc | 3 | Step 8.2 | ~120 | Medium |
| InferMemorySpace (re-enable) | 4A | Step 7.9 | 0 (move) | Low |
| InsertWorkgroupBarriers (2nd) | 4B | Step 12.6 | 0 (reuse) | None |
| GpuEliminateBarriers | 4C | Step 12.65 | ~80 (or upstream) | Medium |
| GPUReuseSharedMemoryAllocs | 5 | Step 12.55 | ~250 | Medium-High |
| GPUCheckResourceUsage | 6 | Step 12.7 | ~40 | Very Low |
| ConvertSharedMemAllocs simplify | 7 | Step 12.5 | -100 (remove) | Low |

---

## 12. Execution Order & Dependencies

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 6 ──→ Phase 7
(enable)    (hoist)     (pad dyn)   (barriers)   (reuse)     (validate)  (simplify)
   │           │            │           │
   │           │            │           └── depends on Phase 1 (workgroup tags)
   │           │            └── depends on Phase 2 (hoist runs after pad)
   │           └── independent of Phase 1, but Phase 1 reduces its workload
   └── independent, but highest impact
```

### Phase Priorities by Bug Impact

| Priority | Phase | Fixes |
|----------|-------|-------|
| **P0 — Critical** | Phase 1 (PromoteMatmulOperands) | Shared memory race, alloca-in-loop crash, missing barriers |
| **P0 — Critical** | Phase 4B (2nd barrier pass) | dGamma/dBeta=0 in LN backward |
| **P1 — High** | Phase 2 (Hoist allocs) | Prevents future alloca-in-loop crashes |
| **P1 — High** | Phase 6 (Resource check) | Fail-fast instead of silent corruption |
| **P2 — Medium** | Phase 3 (PadDynamicAlloc) | Replaces ad-hoc fix with principled approach |
| **P2 — Medium** | Phase 4A (InferMemorySpace) | Correct space tagging for non-matmul ops |
| **P3 — Optimization** | Phase 5 (Reuse shared mem) | Reduces shared memory footprint 20-50% |
| **P3 — Optimization** | Phase 4C (Eliminate barriers) | Reduces barrier overhead |
| **P4 — Cleanup** | Phase 7 (Simplify ConvertSharedMemAllocs) | Technical debt reduction |

### Suggested Implementation Sprint

**Sprint 1 (Correctness)**: Phase 1 + Phase 4B + Phase 6
- Uncomment PromoteMatmulOperands
- Add second InsertWorkgroupBarriers after ConvertSharedMemAllocs
- Add GPUCheckResourceUsage
- **Expected outcome**: All chain tests pass with correct numerics

**Sprint 2 (Robustness)**: Phase 2 + Phase 3
- Implement HoistStaticallyBoundAllocations
- Implement PadDynamicAlloc
- **Expected outcome**: No dynamic allocs survive to PTX generation

**Sprint 3 (Optimization)**: Phase 4A + 4C + Phase 5 + Phase 7
- Re-enable InferMemorySpace at correct pipeline point
- Add GpuEliminateBarriers
- Implement GPUReuseSharedMemoryAllocs
- Simplify ConvertSharedMemAllocs
- **Expected outcome**: Lower shared memory usage, fewer barriers, cleaner code
