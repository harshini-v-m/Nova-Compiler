//===- NVIDIATargetUtils.cpp - NVIDIA GPU target description table --------===//
//
// Implements the known NVIDIA SM architecture table.
// Mirrors IREE's KnownTargets.cpp (getCUDATargetDetails) but self-contained.
//
// Element kind encoding (matches NVMMAIntrinsicInfo fields):
//   0 = f16 (float 16)
//   1 = bf16 (bfloat 16)
//   2 = f32 (float 32)
//===----------------------------------------------------------------------===//

#include "Compiler/Transforms/LLVMGPU/NVIDIATargetUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::nova {

// Element kind constants for readability.
static constexpr int32_t kF16  = 0;
static constexpr int32_t kBF16 = 1;
static constexpr int32_t kF32  = 2;

//===----------------------------------------------------------------------===//
// Known NVIDIA MMA intrinsics
//===----------------------------------------------------------------------===//

// WMMA (Volta+): 16x16x16, f16 in -> f32 acc
static const NVMMAIntrinsicInfo kWmmaF32_16x16x16 = {
    NVMMAIntrinsicValues::WMMA_F32_16x16x16,
    /*mSize=*/16, /*nSize=*/16, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF32};

// WMMA (Volta+): 16x16x16, f16 in -> f16 acc
static const NVMMAIntrinsicInfo kWmmaF16_16x16x16 = {
    NVMMAIntrinsicValues::WMMA_F16_16x16x16,
    /*mSize=*/16, /*nSize=*/16, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF16};

// mma.sync (Ampere): 16x8x16, f16 in -> f32 acc
static const NVMMAIntrinsicInfo kMmaSyncF16_16x8x16 = {
    NVMMAIntrinsicValues::MMA_SYNC_F16_16x8x16,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kF16, /*rhsKind=*/kF16, /*accKind=*/kF32};

// mma.sync (Ampere): 16x8x16, bf16 in -> f32 acc
static const NVMMAIntrinsicInfo kMmaSyncBf16_16x8x16 = {
    NVMMAIntrinsicValues::MMA_SYNC_BF16_16x8x16,
    /*mSize=*/16, /*nSize=*/8, /*kSize=*/16,
    /*warpSize=*/32,
    /*lhsKind=*/kBF16, /*rhsKind=*/kBF16, /*accKind=*/kF32};

//===----------------------------------------------------------------------===//
// SM architecture table
//===----------------------------------------------------------------------===//

NVIDIATargetInfo getNVIDIATargetInfo(llvm::StringRef smArch) {
  // Normalize: accept "sm_80" or "80" or "ampere", etc.
  // We only look at the numeric part for the SM version.
  auto parseSM = [&]() -> int {
    // Try direct match on "sm_XX"
    if (smArch.starts_with("sm_")) {
      int val = 0;
      if (!smArch.drop_front(3).getAsInteger(10, val))
        return val;
    }
    // Try friendly names
    return llvm::StringSwitch<int>(smArch)
        .Case("volta",  70)
        .Case("turing",  75)
        .Case("ampere",  80)
        .Case("ada",     89)
        .Case("hopper",  90)
        .Default(-1);
  };

  int sm = parseSM();
  if (sm < 0)
    return {}; // Unrecognized — returns invalid target.

  NVIDIATargetInfo info;
  info.preferredSubgroupSize = 32; // NVIDIA warp is always 32 threads.
  info.maxThreadsPerWorkgroup = 1024;

  if (sm >= 90) {
    // Hopper (H100, etc.)
    info.archName = "sm_90";
    info.smCount  = 132; // H100 SXM has 132 SMs
    info.maxWorkgroupMemBytes = 228 * 1024;
    info.mmaIntrinsics = {kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 89) {
    // Ada Lovelace (RTX 4090, L40, etc.)
    info.archName = "sm_89";
    info.smCount  = 128; // RTX 4090 has 128 SMs
    info.maxWorkgroupMemBytes = 100 * 1024;
    info.mmaIntrinsics = {kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 86) {
    // Ampere GA102 (RTX 3090, A40, etc.)
    info.archName = "sm_86";
    info.smCount  = 84; // RTX 3090 has 84 SMs
    info.maxWorkgroupMemBytes = 100 * 1024;
    info.mmaIntrinsics = {kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 80) {
    // Ampere GA100 (A100)
    info.archName = "sm_80";
    info.smCount  = 108; // A100 SXM has 108 SMs
    info.maxWorkgroupMemBytes = 164 * 1024;
    info.mmaIntrinsics = {kMmaSyncF16_16x8x16, kMmaSyncBf16_16x8x16,
                          kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 75) {
    // Turing (T4, RTX 20xx)
    info.archName = "sm_75";
    info.smCount  = 72; // RTX 2080 Ti has 68 SMs; use 72 as generic.
    info.maxWorkgroupMemBytes = 64 * 1024;
    info.mmaIntrinsics = {kWmmaF32_16x16x16, kWmmaF16_16x16x16};
  } else if (sm >= 70) {
    // Volta (V100)
    info.archName = "sm_70";
    info.smCount  = 80; // V100 SXM2 has 80 SMs
    info.maxWorkgroupMemBytes = 96 * 1024;
    info.mmaIntrinsics = {kWmmaF32_16x16x16};
  } else {
    // Pre-Volta: no MMA intrinsics, SIMT only.
    info.archName = "sm_60";
    info.smCount  = 60;
    info.maxWorkgroupMemBytes = 48 * 1024;
    // mmaIntrinsics stays empty → KernelConfig falls back to SIMT tiling.
  }
  return info;
}

//===----------------------------------------------------------------------===//
// Intrinsic selection
//===----------------------------------------------------------------------===//

NVMMAIntrinsicValues selectMMAIntrinsic(const NVIDIATargetInfo &target,
                                        int32_t lhsKind, int32_t rhsKind,
                                        int32_t accKind) {
  for (const NVMMAIntrinsicInfo &info : target.mmaIntrinsics) {
    if (info.lhsKind == lhsKind && info.rhsKind == rhsKind &&
        info.accKind == accKind) {
      return info.intrinsic;
    }
  }
  // No exact match: try relaxed (any intrinsic that accepts these operand types
  // with a wider accumulator).
  for (const NVMMAIntrinsicInfo &info : target.mmaIntrinsics) {
    if (info.lhsKind == lhsKind && info.rhsKind == rhsKind) {
      return info.intrinsic; // Accept wider acc.
    }
  }
  return NVMMAIntrinsicValues::NONE;
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

int32_t typeToElementKind(mlir::Type t) {
  if (t.isF16())
    return kF16;
  if (t.isBF16())
    return kBF16;
  if (t.isF32())
    return kF32;
  return -1; // Unsupported.
}

} // namespace mlir::nova
