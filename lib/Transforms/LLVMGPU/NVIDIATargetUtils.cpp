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
static const NVMMAIntrinsicInfo kWmmaF32_16x16x16 = {
    NVMMAIntrinsicValues::WMMA_F32_16x16x16,
    /*mSize=*/16, /*nSize=*/16, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF32};

// WMMA (Volta+): 16×16×16, f16 inputs → f16 accumulator
static const NVMMAIntrinsicInfo kWmmaF16_16x16x16 = {
    NVMMAIntrinsicValues::WMMA_F16_16x16x16,
    /*mSize=*/16, /*nSize=*/16, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF16};

// mma.sync (Ampere+): 16×8×16, f16 inputs → f32 accumulator
static const NVMMAIntrinsicInfo kMmaSyncF16_16x8x16 = {
    NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF32};

// mma.sync (Ampere+): 16×8×16, bf16 inputs → f32 accumulator
static const NVMMAIntrinsicInfo kMmaSyncBf16_16x8x16 = {
    NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kBF16, /*rhsKind=*/kBF16, /*accKind=*/kF32};

// mma.sync (Ampere+): 16×8×8, f32 inputs (TF32 truncation in HW) → f32 accumulator
static const NVMMAIntrinsicInfo kMmaSyncTf32_16x8x8 = {
    NVMMAIntrinsicValues::MMA_SYNC_TF32_16x8x8,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/8,
    /*warpSize=*/32,
    /*lhsKind=*/kF32, /*rhsKind=*/kF32, /*accKind=*/kF32};

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
        .Case("volta",   70)
        .Case("turing",  75)
        .Case("ampere",  80)
        .Case("ada",     89)
        .Case("hopper",  90)
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
  if (sm >= 90) {
    // Hopper (H100 SXM, H100 PCIe, etc.)
    info.archName = "sm_90";
    info.smCount  = 132; // H100 SXM has 132 SMs
    info.maxWorkgroupMemBytes = 228 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 89) {
    // Ada Lovelace (RTX 4090, L40, L4, etc.)
    info.archName = "sm_89";
    info.smCount  = 128; // RTX 4090 has 128 SMs
    info.maxWorkgroupMemBytes = 100 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 86) {
    // Ampere GA106 / GA107 (RTX 3060 has 28 SMs, RTX 3090 has 84 SMs)
    info.archName = "sm_86";
    info.smCount  = 28; // Targeting RTX 3060 as the reference device
    info.maxWorkgroupMemBytes = 100 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 80) {
    // Ampere GA100 (A100 SXM has 108 SMs; 164 KB max shared mem)
    info.archName = "sm_80";
    info.smCount  = 108;
    info.maxWorkgroupMemBytes = 164 * 1024;
    info.mmaIntrinsics = {kMmaSyncTf32_16x8x8,
                          kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 75) {
    // Turing (T4, RTX 20xx series)
    info.archName = "sm_75";
    info.smCount  = 72; // RTX 2080 Ti has 68 SMs; use 72 as a round generic
    info.maxWorkgroupMemBytes = 64 * 1024;
    // Turing has WMMA but not the mma.sync Ampere instructions.
    info.mmaIntrinsics = {kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 70) {
    // Volta (V100 SXM2 has 80 SMs, 96 KB max shared mem)
    info.archName = "sm_70";
    info.smCount  = 80;
    info.maxWorkgroupMemBytes = 96 * 1024;
    info.mmaIntrinsics = {kWmmaF32_16x16x16};
  } else {
    // Pre-Volta: no Tensor Core / MMA intrinsics, SIMT-only path.
    info.archName = "sm_60";
    info.smCount  = 60;
    info.maxWorkgroupMemBytes = 48 * 1024;
    // mmaIntrinsics stays empty → KernelConfig falls back to SIMT tiling.
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
