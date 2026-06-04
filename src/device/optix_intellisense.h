#pragma once

#if defined(__INTELLISENSE__)

#include <optix_types.h>

static inline uint3 optixGetLaunchIndex() {
  return make_uint3(0u, 0u, 0u);
}

static inline uint3 optixGetLaunchDimensions() {
  return make_uint3(1u, 1u, 1u);
}

template <typename... Payload>
static inline void optixTrace(
    OptixTraversableHandle,
    float3,
    float3,
    float,
    float,
    float,
    OptixVisibilityMask,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    Payload&...) {}

template <typename... Attr>
static inline bool optixReportIntersection(float, unsigned int, Attr...) {
  return false;
}

static inline unsigned int optixGetPayload_0() {
  return 0u;
}

static inline unsigned int optixGetPayload_1() {
  return 0u;
}

static inline void optixSetPayload_0(unsigned int) {}

static inline unsigned int optixGetPrimitiveIndex() {
  return 0u;
}

static inline unsigned int optixGetAttribute_0() {
  return 0u;
}

static inline unsigned int optixGetAttribute_1() {
  return 0u;
}

static inline float3 optixGetWorldRayOrigin() {
  return make_float3(0.0f, 0.0f, 0.0f);
}

static inline float3 optixGetWorldRayDirection() {
  return make_float3(0.0f, 0.0f, 1.0f);
}

static inline float3 optixGetObjectRayOrigin() {
  return make_float3(0.0f, 0.0f, 0.0f);
}

static inline float3 optixGetObjectRayDirection() {
  return make_float3(0.0f, 0.0f, 1.0f);
}

static inline float optixGetRayTmin() {
  return 0.0f;
}

static inline float optixGetRayTmax() {
  return 0.0f;
}

#endif
