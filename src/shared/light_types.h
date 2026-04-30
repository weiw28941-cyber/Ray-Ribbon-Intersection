#pragma once

#include <vector_types.h>

enum LightType : int {
  LIGHT_DIRECTIONAL = 0,
  LIGHT_POINT = 1,
  LIGHT_RECT = 2
};

struct LightData {
  int type;
  float3 direction;
  float3 position;
  float3 u;  // rect local axis/half-extent vector
  float3 v;  // rect local axis/half-extent vector
  float3 color;
  float intensity;
  float _pad0;
};
