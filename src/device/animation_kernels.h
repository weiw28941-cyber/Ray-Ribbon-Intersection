#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "shared/ribbon_types.h"

enum class DeviceWindMode : int {
  Grass = 0,
  StripIndependent = 1,
  StripConstrained = 2,
};

void launch_animate_primitives_kernel(
    const RibbonPrimitive* d_base_primitives,
    RibbonPrimitive* d_primitives,
    unsigned count,
    DeviceWindMode mode,
    float time,
    float strength,
    float speed,
    cudaStream_t stream);

void launch_compute_aabbs_kernel(
    const RibbonPrimitive* d_primitives,
    OptixAabb* d_aabbs,
    unsigned count,
    cudaStream_t stream);
