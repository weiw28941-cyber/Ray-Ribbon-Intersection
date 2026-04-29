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
};
