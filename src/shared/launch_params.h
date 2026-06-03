#pragma once

#include <optix.h>
#include <vector_types.h>
#include "shared/ribbon_types.h"
#include "shared/light_types.h"

struct CameraData {
  float3 origin;
  float3 lower_left;
  float3 horizontal;
  float3 vertical;
};

struct RenderCoreStats {
  unsigned long long primary_trace_cycles;
  unsigned long long primary_trace_count;
  unsigned long long shadow_trace_cycles;
  unsigned long long shadow_trace_count;
  unsigned long long closesthit_cycles;
  unsigned long long closesthit_count;
  unsigned long long intersection_cycles_r2;
  unsigned long long intersection_calls_r2;
  unsigned long long solver_cycles_r2;
  unsigned long long solver_calls_r2;
  unsigned long long intersection_cycles_r3;
  unsigned long long intersection_calls_r3;
  unsigned long long solver_cycles_r3;
  unsigned long long solver_calls_r3;
  unsigned long long intersection_cycles_r4a;
  unsigned long long intersection_calls_r4a;
  unsigned long long solver_cycles_r4a;
  unsigned long long solver_calls_r4a;
  unsigned long long intersection_cycles_r4b;
  unsigned long long intersection_calls_r4b;
  unsigned long long solver_cycles_r4b;
  unsigned long long solver_calls_r4b;
  unsigned long long intersection_cycles_r4c;
  unsigned long long intersection_calls_r4c;
  unsigned long long solver_cycles_r4c;
  unsigned long long solver_calls_r4c;
};

struct LaunchParams {
  uchar4* image;
  float4* aov_albedo;
  float4* aov_normal;
  float4* aov_depth;
  float4* aov_beauty_hdr;
  unsigned int width;
  unsigned int height;
  unsigned int spp;
  unsigned int max_depth;
  float exposure;
  float gamma;
  float firefly_clamp;
  OptixTraversableHandle gas;
  RibbonPrimitive* primitives;
  unsigned int primitive_count;
  LightData lights[8];
  unsigned int light_count;
  CameraData camera;
  RenderCoreStats* stats;
  unsigned int shader_wind_enabled;
  unsigned int shader_wind_mode;
  float shader_wind_time;
  float shader_wind_strength;
  float shader_wind_speed;
};
