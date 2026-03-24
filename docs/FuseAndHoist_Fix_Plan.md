# FuseAndHoist Fix Plan — Root Cause & Implementation

## Diagnosis

### Bug 1: `flatWorkgroupSize` computed from wrong forall (CRITICAL)

**File:** `NovaGPUFuseAndHoistParallelLoops.cpp` lines 1172-1178

The pass walks the entire function and picks the **first** thread-mapped forall it finds:
```cpp
funcOp.walk([&](scf::ForallOp forall) {
  if (!maybeFlatWorkgroupSize && isThreadMappedForall(forall)) {
    maybeFlatWorkgroupSize = getStaticForallTripCount(forall);
  }
});
```

The first thread-mapped forall is a trivial **2-thread** embedding copy `scf.forall (%arg56) in (2)`.
This sets `flatWorkgroupSize = 2`. Every matmul/copy forall has 128 threads, so
`tripCountMatchesWorkgroupSize(128, 2)` → **false** → ALL fusion blocked.

**IREE doesn't have this problem** because it reads `workgroup_size` from a function attribute
set during SelectLoweringStrategy. Nova doesn't set this attribute.

### Bug 2: Single global workgroup size for multi-kernel function

Nova compiles the entire GPT-2 model as a **single function** with ~40 block-mapped foralls
(kernels), each with different thread counts (128, 256, 64, etc.). A single `flatWorkgroupSize`
can never be correct for all of them.

**IREE** processes one dispatch (kernel) per function, so one `flatWorkgroupSize` per function
is correct. Nova's single-function model requires per-kernel workgroup sizes.

### Bug 3: `tripCountMatchesWorkgroupSize` is an over-constraint

The FuseForalls check at line 276:
```cpp
!tripCountMatchesWorkgroupSize(consumerForall, flatWorkgroupSize)
```
This requires the consumer forall trip count == global workgroup size. But for fusion,
we only need **producer trip count == consumer trip count**. The `forallFlatTripCountsMatch`
check at line 287 already handles this. The workgroup size check is redundant and harmful.

---

## Fix Plan (3 changes, ordered by priority)

### Fix A: Remove global workgroup size gate from FuseForalls (IMMEDIATE)

**File:** `NovaGPUFuseAndHoistParallelLoops.cpp`

Remove the `tripCountMatchesWorkgroupSize(consumerForall, flatWorkgroupSize)` check.
The `forallFlatTripCountsMatch(producer, consumer)` check already ensures trip counts
match between the two foralls being fused. The global workgroup size is irrelevant.

**Change in FuseForalls::matchAndRewrite():**
```
REMOVE:
    auto consumerForall = currUser->getParentOfType<scf::ForallOp>();
    if (!consumerForall ||
        !tripCountMatchesWorkgroupSize(consumerForall, flatWorkgroupSize))

REPLACE WITH:
    auto consumerForall = currUser->getParentOfType<scf::ForallOp>();
    if (!consumerForall || !isThreadMappedForall(consumerForall))
```

This removes the dependency on `flatWorkgroupSize` entirely from `FuseForalls`.
The pattern no longer needs the constructor parameter.

**Also simplify `runOnOperation()`:**
- Remove the `flatWorkgroupSize` walk entirely
- Change `FuseForalls(ctx, *maybeFlatWorkgroupSize, 2)` → `FuseForalls(ctx, 2)`

**Expected result:** Copy foralls (128 threads) now match matmul foralls (128 threads)
via `forallFlatTripCountsMatch()`. The 8-thread embedding forall won't match 128-thread
foralls, correctly staying unfused.

### Fix B: Set `workgroup_size` function attribute in SelectLoweringStrategy (NEXT)

**File:** `NovaKernelConfig.cpp`

After the root contraction gets its MMA config, set a function attribute:
```cpp
// After configuring root op, set workgroup_size on the function.
// For MMA TF32: threads = product(wg) / product(thread) = 128
funcOp->setAttr("workgroup_size",
    builder.getI64ArrayAttr({numThreads, 1, 1}));
```

**Problem:** Nova has ~40 kernels in one function, each potentially with different thread
counts. The function attribute approach only works for single-kernel-per-function (IREE's model).

**Solution for Nova:** Instead of a function attribute, the `HoistForallFromFor` pattern
should read the thread count from the **block-mapped parent forall**'s body. Each block
forall corresponds to one kernel, so the thread foralls inside it share the same thread count.

This is a FUTURE improvement — Fix A alone unblocks fusion.

### Fix C: Handle mismatched thread counts in same K-loop (FUTURE)

After Fix A, the K-loop body will look like:
```
%177 = scf.forall in (8)         → 8 threads (embedding fetch)
%179 = scf.forall in (1,32,4)    → 128 threads (LN + copy LHS)
%181 = scf.forall in (1,8,16)    → 128 threads (copy RHS)
%182 = scf.forall in (1,16,8)    → 128 threads (matmul)
```

Fix A will fuse `%179→%182` and `%181→%182` (both 128==128), reducing to **2 foralls**:
```
%177 = scf.forall in (8)         → 8 threads (embedding fetch) [unfused]
%182 = scf.forall in (1,16,8)    → 128 threads (copy LHS + copy RHS + matmul) [fused]
```

The 8-thread embedding forall stays separate. This is actually correct — it can't fuse
into 128-thread forall without padding/masking. HoistForallFromFor requires exactly 1
forall in the K-body, so hoisting won't work yet. But the **critical 3→1 fusion** of
copy+matmul foralls IS achieved.

To also fuse the embedding forall:
- Either give it 128 threads (re-tile embedding copy to match matmul)
- Or hoist it above the K-loop (it's K-independent if it reads from the full embedding)
- This is a separate optimization.

---

## Verification After Fix A

1. Build: `make build`
2. Run GPT-2 with debug: check debug2.log
3. Count foralls per K-loop body: should be 1-2 (down from 3-4)
4. Check for linearize/delinearize ops in fused forall body
5. Check throughput improvement (fewer parallel regions → fewer barriers → faster)

---

## Summary

| Fix | File | Lines Changed | Impact |
|-----|------|---------------|--------|
| **A** | FuseAndHoistParallelLoops.cpp | ~15 lines | Unblocks ALL copy-matmul fusion |
| B | NovaKernelConfig.cpp | ~10 lines | Correct per-kernel workgroup size |
| C | Tiling/config passes | ~30 lines | Fuse embedding into matmul forall |
