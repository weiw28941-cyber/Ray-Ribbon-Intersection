#pragma once

// VS Code's C/C++ IntelliSense does not get the custom NVCC commands used for
// these .cu files, so it may parse them as ordinary C++ files. Keep this shim
// IntelliSense-only; NVCC already defines the CUDA macros during real builds.
#if defined(__INTELLISENSE__)
#ifndef __CUDART__
#define __CUDART__
#endif
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 750
#endif
#ifndef __NV_NO_HOST_COMPILER_CHECK
#define __NV_NO_HOST_COMPILER_CHECK 1
#endif
#ifndef __location__
#define __location__(x)
#endif
#ifndef __annotate__
#define __annotate__(x)
#endif
#ifndef __device_builtin__
#define __device_builtin__
#endif
#ifndef __cudart_builtin__
#define __cudart_builtin__
#endif
#ifndef __no_init__
#define __no_init__
#endif
#ifndef __align__
#define __align__(n) alignas(n)
#endif
#ifndef __builtin_align__
#define __builtin_align__(n) alignas(n)
#endif

static inline float rsqrtf(float x) {
  return x;
}

template <typename T>
static inline T atomicAdd(T* address, T value) {
  return address ? (*address += value) : value;
}

static inline unsigned long long clock64() {
  return 0ull;
}

static inline unsigned int __float_as_uint(float value) {
  union {
    float f;
    unsigned int u;
  } bits = {value};
  return bits.u;
}

static inline float __uint_as_float(unsigned int value) {
  union {
    unsigned int u;
    float f;
  } bits = {value};
  return bits.f;
}
#endif
