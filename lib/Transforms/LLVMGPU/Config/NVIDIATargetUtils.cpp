//===- NVIDIATargetUtils.cpp - NVIDIA GPU target description table --------===//
//
// Implements the known NVIDIA SM architecture capability table and the
// MMA intrinsic selection helpers.
//
// Mirrors IREE's KnownTargets.cpp (getCUDATargetDetails) but is fully
// self-contained — no IREE HAL dependency.
//
// GPU / MEMORY SENSITIVE — The SM counts and shared-memory limits in this
// table are used to compute tile sizes for shared memory and to choose MMA
// intrinsics.  Wrong values cause silent performance regressions (occupancy
// drops) or incorrect kernel configurations (exceeding hardware limits).
//
// Element kind encoding used by NVMMAIntrinsicInfo fields:
//   0 = f16  (float16)
//   1 = bf16 (bfloat16)
//   2 = f32  (float32)
//
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::nova {

// Named constants make the MMA intrinsic table below self-documenting.
static constexpr int32_t kF16  = 0;
static constexpr int32_t kBF16 = 1;
static constexpr int32_t kF32  = 2;

//===----------------------------------------------------------------------===//
// Known NVIDIA MMA intrinsics
//
// Each entry encodes the (M, N, K) tile shape expected by the hardware
// instruction, the warp size, and the element types for LHS / RHS / accumulator.
//===----------------------------------------------------------------------===//

// WMMA (Volta+): 16×16×16, f16 inputs → f32 accumulator
static const NVMMAIntrinsicInfo kMmaSyncBf16_16x8x16 = {
    NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kBF16, /*rhsKind=*/kBF16, /*accKind=*/kF32,
    /*distribution=*/{/*threadsM=*/8, /*threadsN=*/4,
                      /*elemsPerThreadM=*/2, /*elemsPerThreadN=*/2}
};

static const NVMMAIntrinsicInfo kMmaSyncF16_16x8x16 = {
    NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF32,
    /*distribution=*/{/*threadsM=*/8, /*threadsN=*/4,   // same shape
                      /*elemsPerThreadM=*/2, /*elemsPerThreadN=*/2}
};

static const NVMMAIntrinsicInfo kMmaSyncTf32_16x8x8 = {
    NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/8,
    /*warpSize=*/32,
    /*lhsKind=*/kF32, /*rhsKind=*/kF32, /*accKind=*/kF32,
    /*distribution=*/{/*threadsM=*/8, /*threadsN=*/4,   // M/N same as bf16
                      /*elemsPerThreadM=*/2, /*elemsPerThreadN=*/2}
};

static const NVMMAIntrinsicInfo kWmmaF32_16x16x16 = {
    NVMMAIntrinsicValues::WMMA_F32_16x16x16,
    /*mSize=*/16, /*nSize=*/16, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF32,
    /*distribution=*/{/*threadsM=*/2, /*threadsN=*/16,
                      /*elemsPerThreadM=*/8, /*elemsPerThreadN=*/1}
};

static const NVMMAIntrinsicInfo kWmmaF16_16x16x16 = {
    NVMMAIntrinsicValues::WMMA_F16_16x16x16,
    /*mSize=*/16, /*nSize=*/16, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF16,
    /*distribution=*/{/*threadsM=*/2, /*threadsN=*/16, 
                      /*elemsPerThreadM=*/8, /*elemsPerThreadN=*/1}
};

std::optional<DistributionLayout> 
NVMMAIntrinsicInfo::getDistributionMappingKind() const {
    // Check if distribution was set (non-zero threads)
    if (distribution.threadsM == 0 || distribution.threadsN == 0)
        return std::nullopt;  // ← this is what causes the skip in your filter
    return distribution;
}

//===----------------------------------------------------------------------===//
// SM architecture capability table
//
// GPU / MEMORY SENSITIVE — maxWorkgroupMemBytes drives the shared-memory
// tile size decision. smCount is used to estimate wave occupancy. Getting
// either wrong silently degrades performance or produces kernel launch errors.
//===----------------------------------------------------------------------===//

NVIDIATargetInfo getNVIDIATargetInfo(llvm::StringRef smArch) {

  // ALGORITHM STEP: Normalize the architecture string to an integer SM level.
  // Accepts the canonical "sm_XX" prefix format as well as human-readable
  // codenames ("volta", "ampere", etc.) for ergonomic use in compiler flags.
  auto parseSM = [&]() -> int {
    if (smArch.starts_with("sm_")) {
      int val = 0;
      if (!smArch.drop_front(3).getAsInteger(10, val))
        return val;
    }
    return llvm::StringSwitch<int>(smArch)
        .Case("volta",     70)
        .Case("turing",    75)
        .Case("ampere",    80)
        .Case("ada",       89)
        .Case("hopper",    90)
        .Case("blackwell", 100)
        .Default(-1);
  };

  int sm = parseSM();
  if (sm < 0)
    return {}; // Unrecognized architecture — returns an invalid (empty) target.

  NVIDIATargetInfo info;
  // NVIDIA warps are always 32 threads on all SM generations.
  info.preferredSubgroupSize    = 32;
  info.maxThreadsPerWorkgroup   = 1024;

  // ALGORITHM STEP: Select capabilities by SM generation (newest-first).
  if (sm >= 100) {
    // Blackwell (B100 / B200 / RTX 5090 etc.)
    // sm_100 is the first Blackwell compute arch; consumer parts use sm_120.
    // Blackwell adds new UMMA (block-scaled, asynchronous) intrinsics that are
    // not yet modelled here. For now we expose the same mma.sync set as Hopper
    // so the compiler can generate functional (not peak-performance) code.
    // TODO: add UMMA / wgmma.mma_async intrinsics when Nova adds sm_100 support.
    info.archName = "sm_100";
    info.smCount  = 132; // B100 SXM5 has 132 SMs (same die count as H100 SXM)
    info.maxWorkgroupMemBytes = 228 * 1024; // same as Hopper shared-mem limit
    info.maxWorkgroupDynamicMemBytes = 228 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 90) {
    // Hopper (H100 SXM5 = 132 SMs, H100 PCIe = 114 SMs, H800 = 132 SMs)
    // 228 KB shared mem is achievable with dynamic smem carve-out.
    info.archName = "sm_90";
    info.smCount  = 132; // H100 SXM5
    info.maxWorkgroupMemBytes = 228 * 1024;
    info.maxWorkgroupDynamicMemBytes = 228 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 89) {
    // Ada Lovelace (RTX 4090 = 128 SMs, L40S = 142 SMs, L4 = 58 SMs)
    // smCount=128 targets RTX 4090 as the reference device.
    // Static cap is 48 KB; dynamic-SMEM opt-in raises it to 99 KB.
    info.archName = "sm_89";
    info.smCount  = 128;
    info.maxWorkgroupMemBytes = 48 * 1024;
    info.maxWorkgroupDynamicMemBytes = 99 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 87) {
    // Ampere GA10B — Jetson AGX Orin / Orin NX (16 SMs, 64 KB shared mem)
    // sm_87 is the embedded Ampere variant; shares mma.sync intrinsics with
    // sm_86 but has far fewer SMs and a smaller shared-mem limit per block.
    // Embedded Ampere has the same 163 KB dynamic carve-out as A100.
    info.archName = "sm_87";
    info.smCount  = 16; // Jetson AGX Orin has 16 SMs
    info.maxWorkgroupMemBytes = 48 * 1024;
    info.maxWorkgroupDynamicMemBytes = 163 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 86) {
    // Ampere GA106/GA107/GA104 desktop & professional:
    //   RTX 3060 = 28 SMs, RTX 3070 = 46 SMs, RTX 3080 = 68 SMs,
    //   RTX 3090 = 82 SMs, A10 = 72 SMs, A30 = 56 SMs.
    // We use 46 SMs as a middle-ground representative; the adjustSeedsForTarget
    // CU-fill pass will tune tiles for the actual SM count at runtime if a more
    // accurate count is supplied by the user.
    // Static cap is 48 KB; the dynamic-SMEM opt-in raises it to 99 KB. The
    // 3-stage matmul pipeline needs ~72 KB and only fits via the dynamic path
    // (see NovaGPUMapForallToGPU::maybeMaterializeDynamicSharedMemory and the
    // cuFuncSetAttribute call in mgpuLaunchKernel).
    info.archName = "sm_86";
    info.smCount  = 46;
    info.maxWorkgroupMemBytes = 48 * 1024;
    info.maxWorkgroupDynamicMemBytes = 99 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8};
  } else if (sm >= 80) {
    // Ampere GA100 (A100 SXM4/SXM5 = 108 SMs, A100 PCIe = 108 SMs,
    //               A30 = 56 SMs — use 108 as canonical data-centre value).
    // 164 KB shared mem is the maximum with dynamic smem carve-out.
    info.archName = "sm_80";
    info.smCount  = 108;
    info.maxWorkgroupMemBytes = 164 * 1024;
    info.maxWorkgroupDynamicMemBytes = 164 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 75) {
    // Turing (RTX 2080 Ti = 68 SMs, T4 = 40 SMs, RTX 2080 = 46 SMs)
    // Use 68 SMs (RTX 2080 Ti) as the reference; Turing has WMMA but NOT the
    // mma.sync instructions introduced with Ampere.
    info.archName = "sm_75";
    info.smCount  = 68;
    info.maxWorkgroupMemBytes = 64 * 1024;
    info.maxWorkgroupDynamicMemBytes = 64 * 1024;
    info.mmaIntrinsics = {kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 70) {
    // Volta (V100 SXM2 = 80 SMs, V100 PCIe = 80 SMs, 96 KB shared mem)
    // Only WMMA_F32 is available; WMMA_F16 accumulation requires Turing+.
    info.archName = "sm_70";
    info.smCount  = 80;
    info.maxWorkgroupMemBytes = 96 * 1024;
    info.maxWorkgroupDynamicMemBytes = 96 * 1024;
    info.mmaIntrinsics = {kWmmaF32_16x16x16};
  } else {
    // Pre-Volta (Pascal and older): no Tensor Core / MMA intrinsics.
    // KernelConfig falls back to a SIMT tiling path for these devices.
    info.archName = "sm_60";
    info.smCount  = 60;
    info.maxWorkgroupMemBytes = 48 * 1024;
    info.maxWorkgroupDynamicMemBytes = 48 * 1024;
  }
  return info;
}

//===----------------------------------------------------------------------===//
// MMA intrinsic selection
//
// Prefers an exact type match (lhs, rhs, acc).  Falls back to a relaxed
// match (lhs, rhs only, accepting a wider accumulator) when no exact match
// exists — e.g., requesting f16/f16/f16 on Ampere returns f16/f16/f32.
//===----------------------------------------------------------------------===//

NVMMAIntrinsicValues selectMMAIntrinsic(const NVIDIATargetInfo &target,
                                        int32_t lhsKind, int32_t rhsKind,
                                        int32_t accKind) {
  // ALGORITHM STEP: Try exact (lhs, rhs, acc) match first.
  for (const NVMMAIntrinsicInfo &info : target.mmaIntrinsics) {
    if (info.lhsKind == lhsKind && info.rhsKind == rhsKind &&
        info.accKind == accKind) {
      return info.intrinsic;
    }
  }

  // ALGORITHM STEP: Relaxed match — accept any accumulator type.
  // This handles the common case where the caller requests f16 accumulation
  // but the hardware can only offer f32 accumulation with f16 inputs.
  for (const NVMMAIntrinsicInfo &info : target.mmaIntrinsics) {
    if (info.lhsKind == lhsKind && info.rhsKind == rhsKind) {
      return info.intrinsic;
    }
  }
  return NVMMAIntrinsicValues::NONE;
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

int32_t typeToElementKind(mlir::Type t) {
  if (t.isF16())  return kF16;
  if (t.isBF16()) return kBF16;
  if (t.isF32())  return kF32;
  return -1; // Unsupported type — caller must handle.
}

} // namespace mlir::nova
