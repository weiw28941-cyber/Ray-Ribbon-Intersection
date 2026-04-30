#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "shared/ribbon_types.h"
#include "shared/light_types.h"

namespace rr {

struct CameraSettings {
  float3 look_from = {0.0f, 0.8f, 2.0f};
  float3 look_at = {0.0f, 0.0f, -4.0f};
  float3 up = {0.0f, 1.0f, 0.0f};
  float fov_deg = 35.0f;
};

struct RenderSettings {
  unsigned width = 1280;
  unsigned height = 720;
  unsigned spp = 8;
  unsigned max_depth = 4;
  float exposure = 1.0f;
  float gamma = 2.2f;
  float firefly_clamp = 10.0f;
  std::string output_path = "output.ppm";
};

struct Scene {
  CameraSettings camera;
  RenderSettings render;
  std::vector<RibbonPrimitive> primitives;
  std::vector<LightData> lights;
};

Scene make_default_scene();
bool load_scene_from_file(const std::string& path, Scene& out_scene, std::string& error);

}  // namespace rr
