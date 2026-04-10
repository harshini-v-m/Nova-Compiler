# How a `linalg.batch_matmul` flows through the Nova pipeline

## The concrete example

Input operation — a batch matmul with these sizes:
```
linalg.batch_matmul
  ins(%A: tensor<8x1024x1536xf32>, %B: tensor<1536x50304xf32>)
  outs(%C: tensor<8x1024x50304xf32>)
```

| Parameter | Value |
|-----------|-------|
| Workgroup tile | `[1, 64, 8]` (Batch=1, M=64, N=8 per block) |
| MMA intrinsic | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` |
| GPU arch | `sm_86` |
| Warps per workgroup | 4 (in M dimension) |

---

## Stage 0 — Strategy selection (`createNovaGPUSelectLoweringStrategyPass`)

Reads the GPU arch string (`sm_86`), picks MMA kind `MMA_SYNC_TF32_16x8x8` and writes it
into a `lowering_config` dict attr on the matmul:

```mlir
lowering_config = {
  mma_kind     = 5,              // MMA_SYNC_TF32_16x8x8
  workgroup    = [1, 64, 8, 0],  // per-block tile sizes (Batch, M, N, K=0 means not tiled yet)
  reduction    = [0, 0, 0, 64],  // one K-step = 64
  subgroup     = [8, 64, 6288, 0],
  wg_subgroup  = [1, 4, 1, 0],   // 4 warps in M, 1 in N
  thread       = [1, 16, 8, 0]   // one warp tile = 16×8
}
```

---

## Stage 1 — Tile to workgroups (`createNovaTileAndDistributeToWorkgroupsPass`)

Wraps the matmul in an `scf.forall` over the output tensor. Each block processes one
`[1, 64, 8]` tile of the output:

```mlir
scf.forall (%b, %m, %n) in (8, 16, 6288) {
  // block (%b, %m, %n) computes C[b, m*64:(m+1)*64, n*8:(n+1)*8]
  linalg.batch_matmul
    ins(A[b, m*64:m*64+64, :], B[:, n*8:n*8+8])
    outs(C_tile[1, 64, 8])
}
```

After this: the matmul inside the block has iteration space `(1, 64, 8, 1536)`.

---

## Stage 3 — Promote to shared memory (`createNovaGPUPromoteMatmulOperandsPass`)

Copies A and B tiles into `__shared__` memory **before** K-tiling. A scratch
`tensor<1x64x64xf32>` (workgroup) is allocated for A and `tensor<1x64x8xf32>` for B.
This happens before K-tiling so the full workgroup tile is staged once per K-step,
not per-thread.

---

## Stage 4 — Tile K (`createNovaGPUApplyTilingLevelReductionPass`)

Wraps the promoted matmul in an `scf.for` over K with step 64:

```mlir
scf.for %k = 0 to 1536 step 64 {
  // load A[:, :, k:k+64]  → shared mem tile [1, 64, 64]
  // load B[k:k+64, :]     → shared mem tile [1, 64, 8]
  // compute matmul on shared tiles
}
```

Now the matmul inside has iteration space `(1, 64, 8, 64)`.

---

## Stage 5 — Tile threads (`createNovaGPUApplyTilingLevelThreadPass`)

Splits the 64-row M dimension across 4 warps using `scf.forall (1, 4, 1)`.
Each warp gets a `(1, 16, 8, 64)` slice:

```mlir
scf.forall (%_, %warp, %_) in (1, 4, 1) {
  // warp %warp computes rows [warp*16 : warp*16+16]
  linalg.batch_matmul
    ins(A_tile[1, 16, 64], B_tile[1, 64, 8])
    outs(C_tile[1, 16, 8])
    // iteration space: (1, 16, 8, 64)
}
```

---

## Stage 5.5 — Configure layouts (`createNovaGPUConfigureTensorLayoutsPass`)

**This is the layout pass.** It reads `lowering_config` and computes how the
`(1, 16, 8, 64)` iteration space maps to threads. Every calculation it does:

### Step 1 — MMA shape
```
mmaShape = [16, 8, 8]   (M=16, N=8, K=8)
```

### Step 2 — Per-WG subgroup counts
From `wg_subgroup = [1, 4, 1, 0]`:
```
perWgSgCounts = [1, 4, 1, 1]   (4 warps in M dim)
```

### Step 4 — Workgroup bounds
From config fields `workgroup` and `reduction`:
```
wgBounds = [1, 64, 8, 64]   (workgroup tile sizes per dimension)
```

### Step 5 — Per-subgroup bounds = wgBounds / perWgSgCounts
```
perSgBounds = [1, 64/4, 8/1, 64/1] = [1, 16, 8, 64]
```

### Step 6 — Identify M/N/K iteration dims
```
M = dim1,  N = dim2,  K = dim3
```

### Step 7 — Full-rank MMA shape
```
mmaFullShape = [1, 16, 8, 8]   (batch=1, M=16, N=8, K=8)
```

### Step 8 — Batch counts = ceil(perSgBounds / mmaFullShape)
```
batchCounts = [1, ceil(16/16), ceil(8/8), ceil(64/8)]
            = [1, 1, 1, 8]
```
This means **each warp does 8 MMA tiles along K** (K-batch = 8).

### Step 9 — Hardware thread layout (from `getHardwareLayout`, MMA_SYNC_TF32_16x8x8)

**LHS** (operand 0, M×K tile):
```
outerDim=M:  outer=1, thread=16, elem=1, tstride=1
innerDim=K:  outer=1, thread=2,  elem=4, tstride=16

full-rank threadCounts = [1, 16, ?,  2]
full-rank elemCounts   = [1,  1, ?,  4]
```

**RHS** (operand 1, K×N tile):
```
outerDim=K:  outer=1, thread=2,  elem=4, tstride=1
innerDim=N:  outer=1, thread=8,  elem=1, tstride=4

full-rank threadCounts = [1, ?, 8, 2]
full-rank elemCounts   = [1, ?, 1, 4]
```

**ACC** (operand 2, M×N tile):
```
outerDim=M:  outer=1, thread=16, elem=1, tstride=1
innerDim=N:  outer=1, thread=8,  elem=1, tstride=4

full-rank threadCounts = [1, 16, 8, ?]
full-rank elemCounts   = [1,  1, 1, ?]
```

> The hardware dictates these numbers from the PTX ISA.
> For `mma.sync.m16n8k8.tf32`: 32 threads per warp, A has 4 regs/thread,
> B has 4 regs/thread, C has 1 scalar/thread at this representation level.

### Step 10 — Project through operand indexing maps

**LHS map** `(d0,d1,d2,d3) → (d0,d1,d3)` — drops d2 (N):
```
proj_thread_counts  = [1, 16, 2]
proj_elem_counts    = [1,  1, 4]
proj_batch_counts   = [1,  1, 8]
proj_sg_counts      = [1,  4, 1]
proj_thread_strides = computeStrides([1,16,2]) = [0, 2, 1]
proj_sg_strides     = computeStrides([1, 4,1]) = [0, 1, 0]
```

**RHS map** `(d0,d1,d2,d3) → (d0,d3,d2)` — drops d1 (M), swaps K and N:
```
proj_thread_counts  = [1, 2, 8]
proj_elem_counts    = [1, 4, 1]
proj_batch_counts   = [1, 8, 1]
proj_sg_counts      = [1, 1, 1]
proj_thread_strides = computeStrides([1,2,8]) = [0, 8, 1]
proj_sg_strides     = computeStrides([1,1,1]) = [0, 0, 0]
```

**ACC map** `(d0,d1,d2,d3) → (d0,d1,d2)` — drops d3 (K):
```
proj_thread_counts  = [1, 16, 8]
proj_elem_counts    = [1,  1, 1]
proj_batch_counts   = [1,  1, 1]
proj_sg_counts      = [1,  4, 1]
proj_thread_strides = computeStrides([1,16,8]) = [0, 8, 1]
proj_sg_strides     = computeStrides([1, 4,1]) = [0, 1, 0]
```

These become the `nova.layout_0/1/2` attrs attached to the linalg op:
```mlir
nova.layout_0 = {batch_counts=[1,1,8], elem_counts=[1,1,4],
                 thread_counts=[1,16,2], thread_strides=[0,2,1],
                 sg_counts=[1,4,1], sg_strides=[0,1,0], shared_mem=true}

nova.layout_1 = {batch_counts=[1,8,1], elem_counts=[1,4,1],
                 thread_counts=[1,2,8], thread_strides=[0,8,1],
                 sg_counts=[1,1,1], sg_strides=[0,0,0], shared_mem=true}

nova.layout_2 = {batch_counts=[1,1,1], elem_counts=[1,1,1],
                 thread_counts=[1,16,8], thread_strides=[0,8,1],
                 sg_counts=[1,4,1], sg_strides=[0,1,0], shared_mem=true}
```

---

## Stage 18 — Vectorize (`createNovaGenericVectorizationPass`)

Converts `linalg.batch_matmul` → `vector.contract`. The linalg op's iteration space
`(1, 16, 8, 64)` becomes full-tile vector types:

```mlir
vector.contract
  {indexing_maps = [
    affine_map<(d0,d1,d2,d3) -> (d0,d1,d3)>,   // LHS
    affine_map<(d0,d1,d2,d3) -> (d0,d3,d2)>,   // RHS
    affine_map<(d0,d1,d2,d3) -> (d0,d1,d2)>],  // ACC
   iterator_types = ["parallel","parallel","parallel","reduction"]}
  %lhs : vector<1x16x64xf32>,
  %rhs : vector<1x64x8xf32>
  into vector<1x16x8xf32>
  {nova.layout_0 = ..., nova.layout_1 = ..., nova.layout_2 = ...}
```

The `nova.layout_*` attrs are **copied verbatim** from the linalg op onto the contract.

> Why 4D maps on 3D vectors?
> The 4th iteration dimension d3=K is the reduction dim — it appears in LHS and RHS
> maps but NOT in the ACC map, which is what makes it contract (reduce) away.
> This is the standard MLIR `vector.contract` encoding of batched GEMM.

---

## Stage 18.5 — Stage through shared memory (`createNovaGPUVectorAllocPass`)

All three operands have `shared_mem=true`. For each operand the pass emits:

```
(A) gpu.barrier              — wait for previous iteration's smem readers
(B) alloc_tensor(workgroup)  — allocate __shared__ scratch with same shape
(C) vector.transfer_write    — write full tile into __shared__ at [0,0,0]
(D) nova.value_barrier       — wait for all threads to finish writing
(E) vector.transfer_read     — read full tile back from __shared__ at [0,0,0]
(F) replace contract operand — contract now reads from smem-sourced vector
```

After this the contract still has **full-tile vectors** (`1x16x64`, `1x64x8`, `1x16x8`),
just sourced from shared memory. All indices on the new reads are `[0, 0, 0]`.

---

## Stage 18.6 — Combine barriers (`createNovaGPUCombineValueSemanticBarriersPass`)

VectorAlloc emits one `nova.value_barrier` per operand (3 total per contract).
This pass merges all barriers in the same block into one, so only a single
`gpu.barrier` is emitted after bufferization instead of three.

---

## Stage 8 — Bufferize (`addNovaGPUBufferizePasses` + `ConvertBufferizationToMemRefPass`)

```
alloc_tensor(workgroup)  →  memref.alloc() : memref<..., workgroup>
nova.value_barrier       →  gpu.barrier
vector.transfer_read/write on tensor  →  on memref
```

`ConvertBufferizationToMemRefPass` then runs liveness analysis and inserts
`memref.dealloc` at each allocation's last use point.

---

## `NovaGPUVectorDistributePass` (Passes.cpp line 222)

**This is the per-thread slicing pass.** For each `vector.contract` with `nova.layout_*`:

### Thread offset computation

For thread `tid` in `[0, 32)` (32 threads per warp):

**LHS offsets** (into `memref<1x16x64xf32>` in shared mem):
```
M offset = (tid / thread_strides[1]) % thread_counts[1] * elem_counts[1]
         + (warp_id / sg_strides[1])  % sg_counts[1]    * thread_counts[1] * elem_counts[1]
         = (tid/2) % 16 * 1  +  (tid/32) % 4 * 16

K offset = (tid / thread_strides[2]) % thread_counts[2] * elem_counts[2]
         = (tid/1) % 2 * 4     [for kb=0; increments by 8 per K-batch step]
```

**RHS offsets** (into `memref<1x64x8xf32>`):
```
K offset = (tid/8) % 2 * 4
N offset = (tid/1) % 8 * 1
```

**ACC offsets** (into `memref<1x16x8xf32>`):
```
M offset = (tid/8) % 16 * 1  +  (tid/32) % 4 * 16
N offset = (tid/1) % 8  * 1
```

### K-batch unroll (8 iterations, statically unrolled — no runtime loop)

```mlir
// Initial ACC read — one per thread
%acc = vector.transfer_read smem_C[0, M_off, N_off] : vector<1x1x1xf32>

// kb=0: K-range [K_off, K_off+4)
%lhs0 = vector.transfer_read smem_A[0, M_off, K_off+0]  : vector<1x1x4xf32>
%rhs0 = vector.transfer_read smem_B[0, K_off+0, N_off]  : vector<1x4x1xf32>
%r0   = vector.contract %lhs0, %rhs0, %acc               : vector<1x1x1xf32>

// kb=1: K-range [K_off+8, K_off+12)
%lhs1 = vector.transfer_read smem_A[0, M_off, K_off+8]  : vector<1x1x4xf32>
%rhs1 = vector.transfer_read smem_B[0, K_off+8, N_off]  : vector<1x4x1xf32>
%r1   = vector.contract %lhs1, %rhs1, %r0                : vector<1x1x1xf32>

// kb=2 ... kb=7 (same pattern, K offsets +16, +24, +32, +40, +48, +56)

// Final write-back
vector.transfer_write %r7, smem_C[0, M_off, N_off]
```

**Why no `scf.for`?** Static unrolling means the compiler sees all 8 independent
load pairs at once and can schedule them freely (hide latency). A dynamic loop would
prevent this.

**K coverage check:**
```
8 iterations × (thread_counts[K]=2 × elem_counts[K]=4) = 8 × 8 = 64 = total K ✓
```

Each `vector.contract<1x1x4> × <1x4x1> → <1x1x1>` is exactly **one**
`mma.sync.m16n8k8` instruction after the hardware mapping stage.

---

## Stage 35 — Hardware mapping (inside `gpu.GPUModuleOp`)

### Step 1 — `GpuHardwareMappingPass`

Three things happen:

1. **`arith.divui/remui` → `affine.apply`** for any non-indexing division/remainder.
   The NVVM legalizer requires affine expressions for address computations.

2. **Fold `extract_strided_slice(transfer_read)` → narrower `transfer_read`.**
   After distribution, some slicing ops remain as `extract_strided_slice` wrapping
   a `transfer_read`. This folds them into a single read with adjusted indices.

3. **`populatePrepareVectorToMMAPatterns`** reshapes the distributed vectors into
   the column-major fragment packing that `nvgpu.mma.sync` expects.
   For `m16n8k8 tf32`: A needs shape `vector<2x1xf32>` (2 rows × 1 elem), 
   B needs `vector<1x1xf32>`, C needs `vector<2x2xf32>` in physical register order.

### Step 2 — `createConvertVectorToGPUPass(useNvGpu=true)`

The single `vector.contract<1x1x4> × <1x4x1> → <1x1x1>` becomes:

```mlir
nvgpu.mma.sync (%lhs_frag, %rhs_frag, %acc_frag)
  [mmaShape = [16, 8, 8], tf32Enabled = true]
  : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
```

Fragment shapes follow PTX register packing (not the logical tile size):
- A: `2×1` — 2 tf32 registers per thread
- B: `1×1` — 1 tf32 register per thread  
- C: `2×2` — 4 f32 registers per thread (physical packing of the 16×8 accumulator)

### Step 3 — `FixMmaSyncTF32Pass`

Sets `tf32_enabled = true` on all f32 `nvgpu.mma.sync` ops. The conversion pass
occasionally misses this flag when the element type is `f32` (not explicitly `tf32`).

---

## Step 13 — NVVM/LLVM lowering (inside `gpu.GPUModuleOp`)

| Pass | What it converts |
|------|-----------------|
| `createConvertNVGPUToNVVMPass()` | `nvgpu.mma.sync` → `nvvm.mma.sync` LLVM intrinsic call |
| `createConvertGpuOpsToNVVMOps()` | `gpu.thread_id x` → `nvvm.read.ptx.sreg.tid.x` |
| `createLowerAffinePass()` | `affine.apply` → `arith.addi/muli` |
| `createConvertVectorToLLVMPass()` | vector ops → `llvm.insertelement/extractelement` |
| `createArithToLLVMConversionPass()` | `arith.*` → LLVM IR arithmetic |
| `createReconcileUnrealizedCastsPass()` | removes leftover `unrealized_conversion_cast` |

---

## `createGpuModuleToBinaryPass` — PTX compilation

MLIR calls the NVPTX LLVM backend on the `gpu.module`. Final PTX emitted:

```ptx
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    {%f0, %f1, %f2, %f3},   // C output (4 accumulators)
    {%f4, %f5, %f6, %f7},   // A input  (4 tf32 registers)
    {%f8},                   // B input  (1 tf32 register)
    {%f0, %f1, %f2, %f3};   // C input  (accumulate into previous)
```

This runs 8 times per thread (the K-batch unroll), accumulating the full K=64
reduction into 4 f32 registers.

---

## Summary — passes from Passes.cpp lines 186–222

| Line | Pass | Single-sentence description |
|------|------|-----------------------------|
| 187 | `NovaGenericVectorizationPass` | `linalg.batch_matmul(1×16×8, K=64)` → `vector.contract<1x16x64> × <1x64x8> → <1x16x8>` with `nova.layout_*` attrs propagated |
| 192 | `NovaGPUHoistVectorExtractInsertSlicePass` | Hoists loop-invariant `tensor.extract_slice` ops out of the K-loop to avoid redundant slicing per iteration |
| 203 | `NovaGPUVectorAllocPass` | Stages each full-tile vector (1×16×64, 1×64×8, 1×16×8) through `__shared__` memory with pre/post barriers |
| 213 | `NovaGPUCombineValueSemanticBarriersPass` | Merges the 3 `nova.value_barrier` ops into one per block so only one `gpu.barrier` is emitted after bufferization |
| 218 | `addNovaGPUBufferizePasses` | `alloc_tensor(workgroup)` → `memref.alloc(__shared__)`, `nova.value_barrier` → `gpu.barrier`, inserts `memref.dealloc` |
| 220 | `ConvertBufferizationToMemRefPass` | Lowers remaining `bufferization.*` ops to `memref.*` and inserts deallocations via liveness analysis |
| 222 | `NovaGPUVectorDistributePass` | Slices each full-tile read to `vector<1x1x4>` per thread and statically unrolls the K-batch into 8 sequential `vector.contract` ops each covering 8 of 64 K-elements |
