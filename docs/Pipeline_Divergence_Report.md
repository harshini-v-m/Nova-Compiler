# Nova vs IREE Pipeline Divergence Report

## Complete Gap Analysis Across All Pipeline Stages

---

## Stage 1: SelectLoweringStrategy

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **Root model** | Single-root per dispatch function | Multi-root: stamps ALL ops independently | Over-configuring creates too many independent kernels |
| **workgroup_size on function** | YES — `TranslationInfoAttr` with `[x,y,z]` on `FunctionOpInterface` | **NO** — no function-level workgroup size | FuseAndHoist can't validate forall trip counts against workgroup size |
| **Pipeline enum** | `TranslationInfoAttr` with pipeline type (TileAndFuse, VectorDistribute, etc.) | **NO** — no pipeline selection attribute | All ops use same pipeline; no specialization |
| **Config propagation** | `propagateLoweringConfig` copies root config to other ops | **NO** — each op configured independently | Redundant work; inconsistent configs possible |
| **Op type coverage** | Matmul, Convolution (direct+IGEMM), Transpose, Reduction, Pack/Unpack, Scatter, Argmax | Contraction only + generic default | Missing: convolution, transpose, pack/unpack, scatter configs |
| **Epilogue handling** | Not explicitly identified; single-root model means non-root ops fuse naturally | Explicit `isContractionEpilogue`/`isContractionPrologue` with rescue mechanism | Works but fragile; 1-hop chain walk misses deeper chains |
| **Verification** | `verifyEntryPoint` validates config + workgroup size consistency | **NO** verification step | Config errors are silent |

### What Nova Needs:
1. **Set `workgroup_size` function attribute** after configuring root contraction
2. **Verification pass** to catch config inconsistencies

---

## Stage 2: TileDispatchUsingForall

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **Configless ops** | Left untouched (fuse or stay outside forall) | **Wrapped in single-block forall** (Phase 2) | Creates unnecessary 1-workgroup kernels for fills, transposes |
| **Producer fusion blocking** | Only blocks `tensor::PadOp` | Blocks PadOp + contractions + ops with reductions | More conservative (correctness guard for multi-root) |
| **Consumer fusion** | Always attempted after tiling | **Conditional + two-phase** for contractions; deferred sibling handling | More sophisticated; handles multi-root correctly |
| **Cleanup patterns** | `SwapExtractWithExpand`, `FoldBroadcastSlice`, memref resolve, ForallOp canonicalization | **Missing**: SwapExtractWithExpand, FoldBroadcastSlice, memref resolve | Leftover IR artifacts may block later passes |
| **Workgroup reordering** | Supports `WorkgroupReorderingAttrInterface` for L2 cache optimization | **NO** | Can't optimize memory access patterns |
| **Mapping attributes** | Custom `IREE::Codegen::WorkgroupMappingAttr` | Upstream `gpu::GPUBlockMappingAttr` | Minor; functionally equivalent |

### What Nova Needs:
1. **Stop wrapping configless ops** in single-block foralls (let them fuse or stay)
2. **Add missing cleanup patterns** (SwapExtractWithExpand, FoldBroadcastSlice)

---

## Stage 3: Padding

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **Mechanism** | `TensorMaskingOpInterface::getMaskedImplementation()` | `linalg::rewriteAsPaddedOp()` (upstream MLIR) | Different APIs; IREE's is more general |
| **Level-awareness** | Reads tile sizes for specific tiling level | Flat `"padding"` key, not level-aware | Can't pad at different levels independently |
| **PadOp → DPS** | `makePadDPS`: PadOp → EmptyOp + CopyOp | **MISSING** | Non-DPS pad ops may cause dim reification failures |
| **Padding values** | Op-dependent identity elements | Always zero | Wrong for min/max reductions |
| **Non-linalg ops** | Pads any `TilingInterface + TensorMaskingOpInterface` | Only `linalg::LinalgOp` | Non-linalg ops silently skipped |

### What Nova Needs:
1. **Add `makePadDPS`** conversion (PadOp → CopyOp)
2. **Fix padding values** for non-additive reductions

---

## Stage 4: Promotion (Shared Memory Copies)

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **Config type** | `DerivedThreadConfigAttr` (zero-payload, implements `LoweringConfigAttrInterface`) | `DictionaryAttr` with `derived_thread=true` + `target_threads=N` markers | Functionally equivalent but different mechanism |
| **Thread count source** | `product(workgroup_size)` from function attribute at tiling time | `product(wgTiles/thTiles)` computed at promotion time | **Can diverge** if workgroup_size ≠ product of tile ratios |
| **Producer reuse** | Stamps config on existing tilable producer (no copy inserted) | **Always inserts new copy** | Extra copies for ops that could tile in-place |
| **Promotion types** | Per-operand: DerivedThread, GlobalLoadDMA, CacheSwizzle, SwizzleOperand | Single strategy for all operands | Missing DMA/swizzle optimizations |
| **Result promotion** | Per-thread copy gets `DerivedThreadConfigAttr` | Per-thread copy gets **no config** | Result copies may not tile correctly |

### What Nova Needs:
1. **Set workgroup_size** so derived thread count uses authoritative source
2. **Stamp config on result copies** during promotion

---

## Stage 5: Reduction (K-dim) Tiling

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **Config propagation post-tiling** | Copies config to first tiled op | **NOT DONE** | Tiled ops lose config; subsequent passes can't query it |
| **Consumer fusion** | Supported via `fuseConsumers` flag (inactive at reduction level) | **NOT IMPLEMENTED** at any level | Can't fuse consumers into forall loops |
| **Cleanup patterns** | `DimOp::getCanonicalizationPatterns`, `SwapExtractWithExpandPattern` | **MISSING** both | Leftover artifacts |
| **Error handling** | Hard failure on tiling error | Warning + continue | Silent failures mask bugs |
| **Partial reduction strategy** | `PartialReductionOuterReduction` | `PartialReductionOuterParallel` | Different loop structure for thread-level reduction |

### What Nova Needs:
1. **Propagate config to tiled ops** after tiling
2. **Add consumer fusion** support

---

## Stage 6: Thread Tiling

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **Thread count source** | `product(workgroup_size)` from function at tiling time | `target_threads` stored at promotion time | Same divergence risk as Stage 4 |
| **Tile derivation algorithm** | `getVectorTileSizesFromLoopRanges`: strict collapse check, `allowMultiDimCollapse` | `deriveThreadTileSizes`: greedy largest-divisor search | Different tiles for same inputs in edge cases |
| **Dynamic shape guard** | Falls back to `[1,...,1,vectorSize]` for dynamic ranges | **NO GUARD** — produces garbage with `kDynamic` | Bug: dynamic shapes will crash or produce wrong tiles |
| **Consumer fusion** | Supported when `fuseConsumers=true` | **NOT IMPLEMENTED** | Can't fuse epilogues (bias, activation) into thread foralls |
| **Config propagation** | Copies config to first tiled op | **NOT DONE** | Same gap as Stage 5 |

### What Nova Needs:
1. **Add dynamic shape guard** in `deriveThreadTileSizes`
2. **Implement consumer fusion** at thread level
3. **Propagate config** post-tiling

---

## Stage 7: FuseAndHoist (CRITICAL)

| Aspect | IREE | Nova | Gap |
|--------|------|------|-----|
| **FuseForalls matching** | Flat trip count == flatWorkgroupSize | Dimension-wise bounds match only | Can't fuse copy→matmul with different dims |
| **Fusion mechanism** | Shared memory alloc + barrier_region + linearize/delinearize + scf.for | Direct body inlining (IV remapping) | Only works for same-dimension foralls |
| **barrier_region op** | `IREE::GPU::BarrierRegionOp` with implicit synchronization | **MISSING entirely** | No tensor-level synchronization primitive |
| **Shared memory alloc** | `bufferization.alloc_tensor` with `#gpu.address_space<workgroup>` | **NOT DONE** during fusion | No shared memory intermediary for fused ops |
| **Dimension mismatch** | Linearize consumer IVs → delinearize to producer IVs | **REJECTS** (bail out) | 3 foralls remain unfused per K-body |
| **Lane/Warp fusion** | `FuseNestedLaneAndWarpForalls` merges warp+lane into thread | **MISSING** | No warp-level fusion support |
| **HoistForallFromFor** | Multi-result forall, backward slice safety analysis | Single-result only, no slice analysis | Less general hoisting |
| **FuseCollapseShape** | Supported | **MISSING** | Can't fuse through collapse_shape ops |
| **FuseExtractSlice** | Supported | Defined but **DISABLED** | Missing source clamping logic |
| **Workgroup size validation** | Validates trip count == workgroup size | **NO validation** | Risk of barriers inside conditioned regions |

### What Nova Needs (Priority Order):

#### CRITICAL — Enables copy→matmul fusion:
1. **BarrierRegion mechanism**: Either:
   - (a) Create Nova `barrier_tensor` op (tensor-level sync primitive), OR
   - (b) Post-bufferization fusion pass at memref level (simpler, no new ops), OR
   - (c) Make copy forall dims match matmul forall dims at thread tiling time

2. **Set `workgroup_size` on function** — single source of truth for all thread counts

#### IMPORTANT — Enables more fusion:
3. **Consumer fusion at thread level** — fuse epilogues into matmul forall
4. **Config propagation post-tiling** — tiled ops retain their config
5. **Multi-result HoistForallFromFor** — more hoisting opportunities

#### NICE-TO-HAVE:
6. FuseCollapseShapeConsumers
7. FuseExtractSliceConsumers (with proper clamping)
8. FuseNestedLaneAndWarpForalls (when subgroup tiling enabled)

---

## Summary: Root Causes of FuseAndHoist Failure

The copy→matmul fusion fails because of a **chain of dependencies**:

```
No workgroup_size on function
  → FuseForalls can't use IREE's flat trip count validation
    → Nova uses dimension-wise matching instead
      → Copy (1,32,4) ≠ Matmul (1,16,8) per-dim
        → Fusion rejected
          → 3 foralls per K-body
            → HoistForallFromFor can't hoist (needs 1 forall)
              → Thread launch overhead per K-iteration
```

**Three independent paths to fix this:**

**Path A — Match copy dims to matmul dims (simplest)**:
Change `deriveThreadTileSizes` to produce tiles that give the SAME per-dimension
trip counts as the matmul. For matmul bounds (1,16,8): LHS copy [1,128,16] gets
thread tiles [1,8,2] → bounds (1,16,8). Direct IV mapping works.

**Path B — Post-bufferization fusion (no new ops)**:
New pass after bufferization (Step 8.75). At memref level, fuses adjacent foralls:
linearize/delinearize IVs, inline producer body, insert `gpu.barrier`.
No tensor-SSA synchronization issue.

**Path C — Nova barrier_region op (IREE-equivalent)**:
Create a `nova.barrier_tensor` op that wraps producer computation with implicit
barrier semantics. Most general but requires new op + tablegen + dialect changes.
