//===- NVIDIATargetUtils.h - NVIDIA GPU target description -------*- C++ -*-===//
//
// Describes NVIDIA GPU hardware limits and available MMA intrinsics per SM
// architecture.  Mirrors the role of IREE's KnownTargets.h (getCUDATargetDetails)
// but is self-contained within Nova.
//===----------------------------------------------------------------------===//

#ifndef NOVA_TRANSFORMS_LLVMGPU_NVIDIATARGETUTILS_H_
#define NOVA_TRANSFORMS_LLVMGPU_NVIDIATARGETUTILS_H_

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir::nova {

//===----------------------------------------------------------------------===//
// MMA Intrinsic values (must stay in sync with NVMMAIntrinsic enum in td file)
//===----------------------------------------------------------------------===//
enum class NVMMAIntrinsicValues : int32_t {
  NONE               = 0,
  WMMA_F32_16x16x16  = 1,
  WMMA_F16_16x16x16  = 2,
  MMA_SYNC_F16_16x8x16  = 3,
  MMA_SYNC_BF16_16x8x16 = 4,
};

//===----------------------------------------------------------------------===//
// Per-intrinsic shape info
//===----------------------------------------------------------------------===//

/// One MMA intrinsic available on a given SM.
struct NVMMAIntrinsicInfo {
  NVMMAIntrinsicValues intrinsic;
  int64_t mSize;        // Tile M dimension
  int64_t nSize;        // Tile N dimension
  int64_t kSize;        // Tile K dimension
  int32_t warpSize;     // Always 32 for NVIDIA
  // Input / accumulator element kinds: 0=f16, 1=bf16, 2=f32, 3=tf32
  int32_t lhsKind;
  int32_t rhsKind;
  int32_t accKind;
};

//===----------------------------------------------------------------------===//
// Per-SM target info
//===----------------------------------------------------------------------===//

/// Hardware capabilities of one NVIDIA SM architecture.
struct NVIDIATargetInfo {
  std::string archName;             // e.g. "sm_80"
  int32_t smCount;                  // Streaming multiprocessor count on chip
  int32_t maxThreadsPerWorkgroup;   // Max threads per block (1024 usually)
  int32_t maxWorkgroupMemBytes;     // Max shared memory per block in bytes
  int32_t preferredSubgroupSize;    // Warp size (32 for all NVIDIA GPUs)
  llvm::SmallVector<NVMMAIntrinsicInfo> mmaIntrinsics;

  bool isValid() const { return !archName.empty(); }
};

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

/// Returns hardware info for the given SM arch string.
/// Recognizes: "sm_70" (Volta), "sm_75" (Turing),
///             "sm_80" / "sm_86" (Ampere), "sm_89" (Ada).
/// Returns an invalid (empty archName) NVIDIATargetInfo if unrecognized.
NVIDIATargetInfo getNVIDIATargetInfo(llvm::StringRef smArch);

/// Selects the best MMA intrinsic from `target` for the given element types.
/// `lhsKind`, `rhsKind`, `accKind` use the same encoding as
/// NVMMAIntrinsicInfo: 0=f16, 1=bf16, 2=f32.
/// Returns NVMMAIntrinsicValues::NONE if no suitable intrinsic found.
NVMMAIntrinsicValues selectMMAIntrinsic(const NVIDIATargetInfo &target,
                                        int32_t lhsKind, int32_t rhsKind,
                                        int32_t accKind);

/// Convenience: map an mlir::Type to the element kind code used above.
/// Returns -1 for unsupported types.
int32_t typeToElementKind(mlir::Type t);

} // namespace mlir::nova

#endif // NOVA_TRANSFORMS_LLVMGPU_NVIDIATARGETUTILS_H_
