# SelectLoweringStrategy Pass — Detailed IR Analysis Report

**Pass:** `mlir::nova::NovaGPUSelectLoweringStrategyPass` (`nova-gpu-select-lowering-strategy`)
**Source:** `lib/Transforms/LLVMGPU/NovaGPUSelectLoweringStrategy.cpp` + `NovaKernelConfig.cpp`
**IR dump location:** `debug2.log` lines 22478–23508
**Date:** 2026-03-21

---

## 1. Pass Overview

The SelectLoweringStrategy pass stamps `lowering_config` DictionaryAttr on linalg ops to control all downstream tiling passes. It dispatches through three paths:

1. **Contraction ops** (matmul, batch_matmul) → `setContractConfig()` → tries MMA first, falls back to SIMT table
2. **Generic ops** (not epilogue/prologue of contraction) → `setDefaultConfig()` → heuristic based on iterator types
3. **Epilogue/prologue ops** (bias add, GELU, weight broadcast, etc.) → **NO config** (fuse as consumer/producer)

### Decision tree in `initNovaGPULaunchConfig()`:
```
For each linalg op:
  if already has config → skip
  if isaContractionOpInterface → setContractConfig (MMA → SIMT fallback)
  if linalg.generic:
    if isContractionEpilogue → skip (fuse as consumer)
    if isContractionPrologue → skip (fuse as producer)
    else → setDefaultConfig
```

---

## 2. Summary Statistics

| Metric | Count |
|--------|-------|
| Total linalg ops | 141 |
| Ops WITH lowering_config (tiling roots) | 78 |
| Ops WITHOUT lowering_config (producers/epilogues) | 63 |

### By op type:
| Op Type | Total | Roots | Producers |
|---------|-------|-------|-----------|
| `linalg.generic` | 102 | 57 | 45 |
| `linalg.batch_matmul` | 18 | 18 | 0 |
| `linalg.matmul` | 3 | 3 | 0 |
| `linalg.fill` | 15 | 0 | 15 |
| `linalg.index` | 3 | 0 | 3 |

---

## 3. Distinct Lowering Config Profiles

### Profile A: Batch Matmul (SIMT) — 18 ops
```
workgroup = [1, 32, 128, 0]    # batch=1, M=32, N=128, K=untiled
reduction = [0, 0, 0, 32]      # K-tile = 32
thread    = [1, 1, 16, 0]      # M=1/thread, N=16/thread → 32×8=256 threads
subgroup  = [0, 0, 0, 0]       # no subgroup tiling
padding   = [1, 32, 128, 32]
promoted_operands = [0, 1]      # both LHS and RHS promoted
mma_kind  = 0 (NONE → SIMT path)
```
**Thread count:** wgM/thM × wgN/thN = 32/1 × 128/16 = 32×8 = **256 threads**
**Grid per op:** batch × ceil(M/32) × ceil(N/128) = 8 × 32 × 12 = 3072 blocks (for 1024×1536)
**SIMT table match:** Entry 0: {32, 128, 32} with workgroup {32, 8, 1}

**Used by:** All 18 `linalg.batch_matmul` ops:
- Forward: FFN up (#9, #20, #30) and FFN down (#14, #24, #34) × 3 blocks
- Backward: dX up (#64, #85, #101), dX down (#72, #91, #107), dW down (#66, #86, #102), dW up (#74, #92, #108) × 3 blocks

### Profile B: 2D Matmul (SIMT) — 3 ops
```
workgroup = [32, 128, 0]       # M=32, N=128, K=untiled
reduction = [0, 0, 32]         # K-tile = 32
thread    = [1, 16, 0]         # M=1/thread, N=16/thread → 32×8=256 threads
subgroup  = [0, 0, 0]
padding   = [32, 128, 32]
promoted_operands = [0, 1]
mma_kind  = 0 (NONE)
```
**Thread count:** 32×8 = **256 threads** (SIMT table entry 0 again)
**Used by:** Output projection matmuls:
- Forward: 8192×384 × 384×50304 (#40)
- Backward dX: 8192×50304 × 50304×384 (#54)
- Backward dW: 384×8192 × 8192×50304 (#56)

### Profile C: Row-Reduction (LN mean/variance) — 22 ops
```
workgroup = [1, 256, 0]        # batch=1, rows=256, cols=untiled
reduction = [0, 0, 4]          # sequential reduction tile = 4
thread    = [0, 1, 0]          # 1 row per thread → 256 threads
subgroup  = [0, 0, 0]
padding   = [1, 256, 4]
promoted_operands = []
mma_kind  = 0
```
**Thread count:** 256/1 = **256 threads**
**Used by:** All LayerNorm sum/sum-sq reductions, softmax reductions (#4,5,16,17,26,27,36,37,42,43,48,57,58,79,80,95,96,111,112)

### Profile D: LN Backward d(gamma)/d(beta) — 11 ops
```
workgroup = [128, 0, 0]        # parallel dim=128
reduction = [0, 4, 4]          # both reduction dims tile=4
thread    = [2, 0, 0]          # 2 elements/thread → 64 threads
subgroup  = [0, 0, 0]
padding   = [128, 4, 4]
promoted_operands = []
```
**Iterator types:** [parallel, reduction, reduction]
**Thread count:** 128/2 = **64 threads**
**Used by:** d(gamma), d(beta) computations (#61,62,69,81,82,88,97,98,104,113,114)

### Profile E: Elementwise 3D Broadcast — 4 ops
```
workgroup = [1, 1, 128]
reduction = [0, 0, 0]
thread    = [0, 0, 2]          # 2 elements/thread → 64 threads
padding   = [1, 1, 128]
promoted_operands = []
```
**Used by:** Embedding lookup (#1), softmax exp (#47), softmax normalize (#49), SCE grad (#52)

### Profile F: Batch Reduction (weight grad sum over batch) — 4 ops
```
workgroup = [1, 128, 0]
reduction = [0, 0, 4]
thread    = [0, 1, 0]          # 128 threads
padding   = [1, 128, 4]
promoted_operands = []
```
**Used by:** Weight grad batch reductions: dW_down (#68,87,103), pos_embed_grad (#117)

### Profile G: Bias Backward 1536-dim — 3 ops
```
workgroup = [512, 0, 0]
reduction = [0, 4, 4]
thread    = [2, 0, 0]          # 256 threads
padding   = [512, 4, 4]
promoted_operands = []
```
**Used by:** d(bias_up) 1536-dim reductions (#78, #94, #110)

### Profile H: Weight/Bias Update 1536-dim — 3 ops
```
workgroup = [512]
reduction = [0]
thread    = [2]                # 256 threads
padding   = [512]
promoted_operands = []
```
**Used by:** 1536-dim bias gradient accumulation (#124, #130, #136)

### Profile I: Weight/Bias Update 384-dim — 5 ops
```
workgroup = [128]
reduction = [0]
thread    = [2]                # 64 threads
padding   = [128]
promoted_operands = []
```
**Used by:** 384-dim parameter updates (#126, #132, #138, #139, #140)

### Profile J: 2D Elementwise (embedding grads) — 2 ops
```
workgroup = [1, 128]
reduction = [0, 0]
thread    = [0, 2]             # 64 threads
padding   = [1, 128]
promoted_operands = []
```
**Used by:** Token/position embedding grad accumulation (#119, #120)

### Profile K: SCE 4D Loss Tensor — 1 op
```
workgroup = [1, 1, 1, 1024]
reduction = [0, 0, 0, 0]
thread    = [0, 0, 0, 4]      # 256 threads
```
**Used by:** Cross-entropy loss computation (#44)

### Profile L: Label Cast (i16→i64) — 1 op
```
workgroup = [1, 1024]
thread    = [0, 4]             # 256 threads
```
**Used by:** Label index type conversion (#50)

### Profile M: Scatter Index — 1 op
```
workgroup = [1024]
thread    = [4]                # 256 threads
```
**Used by:** Scatter index computation (#51)

### Profile N: Weight Grad Batch Reduction (1536-dim output) — 3 ops
```
workgroup = [1, 256, 0]
reduction = [0, 0, 4]
thread    = [0, 1, 0]          # 256 threads
padding   = [1, 256, 4]
promoted_operands = []
```
**Used by:** dW_up batch reductions (#76, #93, #109)

---

## 4. Complete Op Inventory

### Forward Pass — Embedding + LN0

| # | Line | SSA | Op Type | Shapes (ins → outs) | Config | Semantic |
|---|------|-----|---------|---------------------|--------|----------|
| 1 | 22538 | %27 | generic | 8x1024xi16, 1024xi64 → 8x1024x384xf32 | Profile E | Embedding lookup (token + position) |
| 2 | 22540 | %221 | index | - | PRODUCER | Index for embedding gather |
| 3 | 22550 | %29 | fill | → 8x1024xf32 | PRODUCER | Zero-init for reduction |
| 4 | 22551 | %30 | generic | 8x1024x384xf32 → 8x1024xf32 | Profile C | LN0 mean: sum(x)/D |
| 5 | 22556 | %31 | generic | 8x1024x384xf32, 8x1024xf32 → 8x1024xf32 | Profile C | LN0 variance: sum((x-mean)^2)/D |

### Forward Pass — Transformer Block 1

| # | Line | SSA | Op Type | Shapes | Config | Semantic |
|---|------|-----|---------|--------|--------|----------|
| 6 | 22564 | %32 | generic | 5 inputs → 8x1024x384xf32 | PRODUCER (epilogue) | LN0 apply: gamma*(x-mean)/sqrt(var+eps)+beta |
| 7 | 22577 | %34 | generic | 384x1536xf32 → 8x384x1536xf32 | PRODUCER (prologue) | Weight broadcast: W_up [384,1536] → [8,384,1536] |
| 8 | 22582 | %36 | fill | → 8x1024x1536xf32 | PRODUCER | Zero-init for batch_matmul |
| 9 | 22583 | %37 | **batch_matmul** | 8x1024x384, 8x384x1536 → 8x1024x1536 | **Profile A** | **FFN up: X × W_up** |
| 10 | 22584 | %38 | generic | 8x1024x1536, 1536xf32 → 8x1024x1536 | PRODUCER (epilogue) | Bias add: + b_up |
| 11 | 22589 | %39 | generic | 8x1024x1536 → 8x1024x1536 | PRODUCER (epilogue) | GELU activation |
| 12 | 22603 | %41 | generic | 1536x384xf32 → 8x1536x384xf32 | PRODUCER (prologue) | Weight broadcast: W_down [1536,384] → [8,1536,384] |
| 13 | 22607 | %42 | fill | → 8x1024x384xf32 | PRODUCER | Zero-init |
| 14 | 22608 | %43 | **batch_matmul** | 8x1024x1536, 8x1536x384 → 8x1024x384 | **Profile A** | **FFN down: GELU(X×W_up+b) × W_down** |
| 15 | 22609 | %44 | generic | 8x1024x384, 8x1024x384, 384xf32 → 8x1024x384 | PRODUCER (epilogue) | Residual add + bias: out + x + b_down |

### Forward Pass — Transformer Block 2 (identical structure)

| # | Line | SSA | Op Type | Shapes | Config | Semantic |
|---|------|-----|---------|--------|--------|----------|
| 16 | 22615 | %45 | generic | 8x1024x384 → 8x1024xf32 | Profile C | LN1 mean |
| 17 | 22620 | %46 | generic | 8x1024x384, 8x1024xf32 → 8x1024xf32 | Profile C | LN1 variance |
| 18 | 22628 | %47 | generic | 5 inputs → 8x1024x384 | PRODUCER | LN1 apply |
| 19 | 22640 | %48 | generic | 384x1536 → 8x384x1536 | PRODUCER | W_up broadcast |
| 20 | 22644 | %49 | **batch_matmul** | 8x1024x384, 8x384x1536 → 8x1024x1536 | **Profile A** | FFN up block 2 |
| 21 | 22645 | %50 | generic | 8x1024x1536, 1536xf32 → 8x1024x1536 | PRODUCER | Bias add |
| 22 | 22650 | %51 | generic | 8x1024x1536 → 8x1024x1536 | PRODUCER | GELU |
| 23 | 22663 | %52 | generic | 1536x384 → 8x1536x384 | PRODUCER | W_down broadcast |
| 24 | 22667 | %53 | **batch_matmul** | 8x1024x1536, 8x1536x384 → 8x1024x384 | **Profile A** | FFN down block 2 |
| 25 | 22668 | %54 | generic | 8x1024x384, 8x1024x384, 384xf32 → 8x1024x384 | PRODUCER | Residual + bias |

### Forward Pass — Transformer Block 3 (identical structure)

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 26 | 22674 | %55 | 8x1024x384 → 8x1024xf32 | Profile C | LN2 mean |
| 27 | 22679 | %56 | 8x1024x384, 8x1024xf32 → 8x1024xf32 | Profile C | LN2 variance |
| 28 | 22687 | %57 | 5 inputs → 8x1024x384 | PRODUCER | LN2 apply |
| 29 | 22699 | %58 | 384x1536 → 8x384x1536 | PRODUCER | W_up broadcast |
| 30 | 22703 | %59 | **batch_matmul** | 8x1024x384, 8x384x1536 → 8x1024x1536 | **Profile A** | FFN up block 3 |
| 31 | 22704 | %60 | 8x1024x1536, 1536xf32 → 8x1024x1536 | PRODUCER | Bias add |
| 32 | 22709 | %61 | 8x1024x1536 → 8x1024x1536 | PRODUCER | GELU |
| 33 | 22722 | %62 | 1536x384 → 8x1536x384 | PRODUCER | W_down broadcast |
| 34 | 22726 | %63 | **batch_matmul** | 8x1024x1536, 8x1536x384 → 8x1024x384 | **Profile A** | FFN down block 3 |
| 35 | 22727 | %64 | 8x1024x384, 8x1024x384, 384xf32 → 8x1024x384 | PRODUCER | Residual + bias |

### Forward Pass — Final LN + Output Projection

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 36 | 22733 | %65 | 8x1024x384 → 8x1024xf32 | Profile C | Final LN mean |
| 37 | 22738 | %66 | 8x1024x384, 8x1024xf32 → 8x1024xf32 | Profile C | Final LN variance |
| 38 | 22746 | %67 | 5 inputs → 8x1024x384 | PRODUCER | Final LN apply |
| 39 | 22760 | %69 | fill → 8192x50304xf32 | PRODUCER | Zero-init |
| 40 | 22761 | %70 | **matmul** | 8192x384, 384x50304 → 8192x50304 | **Profile B** | **Output projection: flatten(LN(x)) × W_vocab** |

### Forward Pass — Softmax Cross-Entropy Loss

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 41 | 22763 | %71 | fill → 8x1024xf32 | PRODUCER | Zero-init |
| 42 | 22764 | %72 | 8x1024x50304 → 8x1024xf32 | Profile C | Softmax: max(logits) for numerical stability |
| 43 | 22769 | %73 | 8x1024x50304, 8x1024xf32 → 8x1024xf32 | Profile C | Softmax: sum(exp(logits - max)) |
| 44 | 22777 | %75 | 8x1024xf32, 8x1024xf32, 8x1024xi16 → 8x1024x8x1024xf32 | Profile K | CE loss: -log(softmax[label]) per position |
| 45-46 | 22779-80 | %221,%222 | index ops | PRODUCER | Index computation for gather |
| 47 | 22821 | %78 | 8x1024x50304, 8x1024xf32 → 8x1024x50304 | Profile E | Softmax: exp(logits - max) |
| 48 | 22827 | %79 | 8x1024x50304 → 8x1024xf32 | Profile C | Softmax: sum reduction |
| 49 | 22832 | %80 | 8x1024x50304, 8x1024xf32 → 8x1024x50304 | Profile E | Softmax: normalize = exp/sum |
| 50 | 22841 | %82 | 8x1024xi16 → 8x1024xi64 | Profile L | Label cast i16 → i64 |
| 51 | 22848 | %84 | 8192xi64, 8192xi64 → 8192xi64 | Profile M | Scatter index arithmetic |
| 52 | 22868 | %87 | 8x1024x50304 → 8x1024x50304 | Profile E | SCE gradient: softmax - one_hot |

### Backward Pass — Output Projection

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 53 | 22879 | %91 | fill → 8192x384xf32 | PRODUCER | Zero-init |
| 54 | 22880 | %92 | **matmul** | 8192x50304, 50304x384 → 8192x384 | **Profile B** | **dX = grad × W_vocab^T** |
| 55 | 22882 | %94 | fill → 384x50304xf32 | PRODUCER | Zero-init |
| 56 | 22883 | %95 | **matmul** | 384x8192, 8192x50304 → 384x50304 | **Profile B** | **dW_vocab = X^T × grad** |

### Backward Pass — Final LN

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 57 | 22897 | %97 | 8x1024x384 → 8x1024xf32 | Profile C | Bwd LN: sum(grad * (x-mean)) |
| 58 | 22902 | %98 | 8x1024x384 → 8x1024xf32 | Profile C | Bwd LN: sum(grad) |
| 59 | 22907 | %99 | 7 inputs → 8x1024x384 | PRODUCER | Bwd LN: dx computation |
| 60 | 22925 | %101 | fill → 384xf32 | PRODUCER | Zero-init |
| 61 | 22926 | %102 | 8x1024x384, ... → 384xf32 | Profile D | d(gamma): reduce over B×T |
| 62 | 22938 | %103 | 8x1024x384 → 384xf32 | Profile D | d(beta): reduce over B×T |

### Backward Pass — FFN Block 3

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 63 | 22948 | %106 | 384x1536 → 8x384x1536 | PRODUCER | W_up broadcast |
| 64 | 22952 | %107 | **batch_matmul** | 8x1024x384, 8x384x1536 → 8x1024x1536 | **Profile A** | dX_up: grad × W_up (for GELU backward) |
| 65 | 22955 | %109 | fill → 8x1536x384xf32 | PRODUCER | Zero-init |
| 66 | 22956 | %110 | **batch_matmul** | 8x1536x1024, 8x1024x384 → 8x1536x384 | **Profile A** | dW_down: GELU(x)^T × grad |
| 67 | 22958 | %112 | fill → 1536x384xf32 | PRODUCER | Zero-init |
| 68 | 22959 | %113 | 8x1536x384 → 1536x384xf32 | Profile F | dW_down: batch reduce [8,...] → [1536,384] |
| 69 | 22964 | %114 | 8x1024x384 → 384xf32 | Profile D | d(bias_down): reduce over B×T |
| 70 | 22969 | %115 | 8x1024x1536, 8x1024x1536 → 8x1024x1536 | PRODUCER (epilogue) | GELU backward: grad * gelu'(x) |
| 71 | 22995 | %117 | 1536x384 → 8x1536x384 | PRODUCER | W_down broadcast |
| 72 | 22999 | %118 | **batch_matmul** | 8x1024x1536, 8x1536x384 → 8x1024x384 | **Profile A** | dX_down: gelu_grad × W_down |
| 73 | 23002 | %120 | fill → 8x384x1536xf32 | PRODUCER | Zero-init |
| 74 | 23003 | %121 | **batch_matmul** | 8x384x1024, 8x1024x1536 → 8x384x1536 | **Profile A** | dW_up: X^T × gelu_grad |
| 75 | 23004 | %122 | fill → 384x1536xf32 | PRODUCER | Zero-init |
| 76 | 23005 | %123 | 8x384x1536 → 384x1536xf32 | Profile N | dW_up: batch reduce [8,...] → [384,1536] |
| 77 | 23011 | %125 | fill → 1536xf32 | PRODUCER | Zero-init |
| 78 | 23012 | %126 | 8x1024x1536 → 1536xf32 | Profile G | d(bias_up): reduce over B×T |

### Backward Pass — LN Block 3

| # | Line | SSA | Shapes | Config | Semantic |
|---|------|-----|--------|--------|----------|
| 79 | 23029 | %128 | 8x1024x384 → 8x1024xf32 | Profile C | Bwd LN: sum(grad * (x-mean)) |
| 80 | 23034 | %129 | 8x1024x384 → 8x1024xf32 | Profile C | Bwd LN: sum(grad) |
| 81 | 23039 | %130 | 8x1024x384, ... → 384xf32 | Profile D | d(gamma) |
| 82 | 23051 | %131 | 8x1024x384 → 384xf32 | Profile D | d(beta) |
| 83 | 23056 | %132 | 8 inputs → 8x1024x384 | PRODUCER | Bwd LN: dx + residual grad add |

### Backward Pass — FFN Block 2 (same structure as Block 3)

| # | Line | SSA | Config | Semantic |
|---|------|-----|--------|----------|
| 84 | 23077 | %133 | PRODUCER | W_up broadcast |
| 85 | 23081 | %134 | **Profile A** | dX_up: grad × W_up |
| 86 | 23083 | %135 | **Profile A** | dW_down: X^T × grad |
| 87 | 23084 | %136 | Profile F | dW_down: batch reduce |
| 88 | 23089 | %137 | Profile D | d(bias_down) |
| 89 | 23094 | %138 | PRODUCER | GELU backward |
| 90 | 23119 | %139 | PRODUCER | W_down broadcast |
| 91 | 23123 | %140 | **Profile A** | dX_down |
| 92 | 23125 | %141 | **Profile A** | dW_up |
| 93 | 23126 | %142 | Profile N | dW_up: batch reduce |
| 94 | 23131 | %143 | Profile G | d(bias_up) |

### Backward Pass — LN Block 2 + FFN Block 1 + LN Block 1

| # | Line | SSA | Config | Semantic |
|---|------|-----|--------|----------|
| 95 | 23148 | %145 | Profile C | Bwd LN: sum(grad * (x-mean)) |
| 96 | 23153 | %146 | Profile C | Bwd LN: sum(grad) |
| 97 | 23158 | %147 | Profile D | d(gamma) |
| 98 | 23170 | %148 | Profile D | d(beta) |
| 99 | 23175 | %149 | PRODUCER | Bwd LN: dx + residual |
| 100 | 23196 | %150 | PRODUCER | W_up broadcast |
| 101 | 23200 | %151 | **Profile A** | dX_up |
| 102 | 23202 | %152 | **Profile A** | dW_down |
| 103 | 23203 | %153 | Profile F | dW_down: batch reduce |
| 104 | 23208 | %154 | Profile D | d(bias_down) |
| 105 | 23213 | %155 | PRODUCER | GELU backward |
| 106 | 23238 | %156 | PRODUCER | W_down broadcast |
| 107 | 23242 | %157 | **Profile A** | dX_down |
| 108 | 23244 | %158 | **Profile A** | dW_up |
| 109 | 23245 | %159 | Profile N | dW_up: batch reduce |
| 110 | 23250 | %160 | Profile G | d(bias_up) |
| 111 | 23267 | %162 | Profile C | Bwd LN0: sum(grad * (x-mean)) |
| 112 | 23272 | %163 | Profile C | Bwd LN0: sum(grad) |
| 113 | 23277 | %164 | Profile D | d(gamma) |
| 114 | 23289 | %165 | Profile D | d(beta) |
| 115 | 23294 | %166 | PRODUCER | Bwd LN0: dx + residual |

### Backward Pass — Embedding Gradients

| # | Line | SSA | Config | Semantic |
|---|------|-----|--------|----------|
| 116 | 23313 | %168 | fill (PRODUCER) | Zero-init 1024x384 |
| 117 | 23314 | %169 | Profile F | Position embedding grad: batch reduce |
| 118 | 23333 | %172 | fill (PRODUCER) | Zero-init 50304x384 |

### Optimizer — Weight/Bias Updates

| # | Line | SSA | Config | Semantic |
|---|------|-----|--------|----------|
| 119 | 23346 | %176 | Profile J | Token embedding grad accumulation (50304×384) |
| 120 | 23353 | %178 | Profile J | Position embedding grad accumulation (1024×384) |
| 121-141 | 23360-23500 | various | Profiles H, I, or PRODUCER | Weight/bias updates for all 3 blocks + output proj |

---

## 5. Producer/Epilogue Classification Analysis

### Why certain ops are PRODUCERS (no config):

**Contraction Prologues** (detected by `isContractionPrologue()`):
- Weight broadcasts: 384x1536 → 8x384x1536 (#7, 12, 19, 23, 29, 33, 63, 71, 84, 90, 100, 106)
- These feed directly into batch_matmul → fused as producers

**Contraction Epilogues** (detected by `isContractionEpilogue()`):
- Bias adds after batch_matmul (#10, 21, 31)
- GELU after bias add (#11, 22, 32) — detected via 2-level deep check
- Residual connections (#15, 25, 35)
- GELU backward (#70, 89, 105)

**LN Apply ops** (#6, 18, 28, 38):
- All-parallel generics consuming reduction roots
- These are contraction prologues (their output feeds into batch_matmul)

**Fill ops** (#3, 8, 13, 39, 41, 53, 55, 60, 65, 67, 73, 75, 77, 116, 118):
- Always producers — they initialize accumulators for the consumer op

---

## 6. Key Observations and Issues

### 6.1 All Contractions Use SIMT (mma_kind = 0)

Despite targeting sm_80 (Ampere), ALL matmul/batch_matmul ops fall through to SIMT table lookup instead of MMA. This means `trySetMMAConfig()` is failing. Possible reasons:
- f32×f32→f32 may not have a matching MMA intrinsic in the target info
- The `selectMMAIntrinsic()` function may only support fp16/bf16/tf32 MMA
- **Impact:** No tensor core utilization. All matmuls use scalar FMA instructions.

### 6.2 promoted_operands Only Set for Contractions

Only the 21 contraction ops (18 batch_matmul + 3 matmul) have `promoted_operands = [0, 1]`. All other ops have `promoted_operands = []`. This is correct — only matmuls benefit from cooperative shared memory loading.

### 6.3 Fusion Chain Integrity

Each forward block forms a fusion chain:
```
[LN_mean → LN_var] → LN_apply → W_broadcast → batch_matmul_up → bias_add → GELU → W_broadcast → batch_matmul_down → residual
                      (prologue)  (prologue)     ROOT              (epilogue) (epilogue) (prologue)   ROOT              (epilogue)
```

This means each forward block produces exactly **2 root kernels** from the matmuls, plus **2 reduction kernels** from LN.

### 6.4 Thread Count Distribution

| Threads | Count | Used by |
|---------|-------|---------|
| 256 | ~40 ops | Matmuls, row-reductions, large elementwise |
| 64 | ~25 ops | LN backward grads, small elementwise |
| 128 | ~5 ops | Batch reductions |

### 6.5 Missing: No Config for Weight Update 2D Elementwise Producers

Ops #121-#141 include weight update patterns where some 2D elementwise ops (384x1536, 1536x384) are marked as PRODUCER with no config. These are: `old_weight + lr * grad` operations. Some get configs (Profile J/H/I) while others don't — the ones without configs are chained producers of the configured ones (e.g., `lr * grad` produces into `weight + scaled_grad`).

---

## 7. Kernel Formation Prediction

Based on the configs, the expected kernel count from tiling:

| Category | Roots | Expected Kernels |
|----------|-------|-----------------|
| Embedding lookup | 1 | 1 |
| LN reductions (fwd, 4 LNs × 2 reductions) | 8 | 8 |
| FFN up matmuls (fwd, 3 blocks) | 3 | 3 |
| FFN down matmuls (fwd, 3 blocks) | 3 | 3 |
| Output projection matmul | 1 | 1 |
| Softmax/CE forward | 9 | 9 |
| Output proj backward (dX + dW) | 2 | 2 |
| LN backward reductions (4 LNs × 2) | 8 | 8 |
| LN backward d(gamma)/d(beta) (4 LNs × 2) | 8 | 8 |
| FFN backward (3 blocks × 4 matmuls) | 12 | 12 |
| FFN backward batch reductions (3 × 2) | 6 | 6 |
| FFN backward bias grads (3 × 2) | 6 | 6 |
| Embedding grad reduction | 1 | 1 |
| Optimizer updates | ~10 | ~10 |
| Scalar ops | ~5 | ~5 |
| **Total** | **~78** | **~83** |

Actual kernel count from outlining: **90** (extra kernels from ops that get split during tiling or from LN backward dx computations).
