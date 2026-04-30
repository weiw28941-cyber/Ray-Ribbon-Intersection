#pragma once

#include <vector_types.h>

enum RibbonType : int {
  RIBBON_TYPE_2 = 2,
  RIBBON_TYPE_3 = 3
};

struct Ribbon2Data {
  float3 q00;
  float3 q01;
  float3 q10;
  float3 q11;
  float w0;
  float w1;
  float w2;
  float _pad;
};

struct Ribbon3Data {
  float3 b0;
  float3 b1;
  float3 b2;
  float3 bt0;
  float3 bt1;
  float beta;
  float gamma;
  float alpha;
  float w0;
  float w1;
  float w2;
  float _pad;
};

struct RibbonPrimitive {
  int type;
  int material_id;
  int material_type;  // 0: diffuse, 1: metal, 2: dielectric
  int _pad0;
  float3 base_color;
  float emission;
  float metallic;
  float roughness;
  float ior;
  Ribbon2Data r2;
  Ribbon3Data r3;
};
