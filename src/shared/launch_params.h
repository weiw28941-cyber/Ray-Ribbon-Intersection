#pragma once

#include <optix.h>
#include <vector_types.h>
#include "shared/ribbon_types.h"

struct CameraData {
  float3 origin;
  float3 lower_left;
  float3 horizontal;
  float3 vertical;
};

struct LaunchParams {
  uchar4* image;
  unsigned int width;
  unsigned int height;
  OptixTraversableHandle gas;
  RibbonPrimitive* primitives;
  unsigned int primitive_count;
  CameraData camera;
};
