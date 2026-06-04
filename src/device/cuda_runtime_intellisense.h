#pragma once

#if defined(__INTELLISENSE__)

static const uint3 threadIdx = {0u, 0u, 0u};
static const uint3 blockIdx = {0u, 0u, 0u};
static const dim3 blockDim = dim3(1u, 1u, 1u);
static const dim3 gridDim = dim3(1u, 1u, 1u);

#endif
