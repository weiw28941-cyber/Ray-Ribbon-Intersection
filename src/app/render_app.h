#pragma once

#include <string>

#include "scene/scene.h"

namespace rr {

struct AppOptions {
  std::string ptx_path;
  std::string scene_path;
  std::string output_path;
  std::string aov_dir;
  unsigned width = 0;
  unsigned height = 0;
  unsigned spp = 0;
  unsigned max_depth = 0;
  float exposure = 0.0f;
  float gamma = 0.0f;
  float firefly_clamp = 0.0f;
  bool denoise = false;
  bool realtime = false;
  bool wind = false;
  float wind_strength = 0.18f;
  float wind_speed = 1.6f;
  std::string wind_mode = "independent";
  std::string wind_backend = "cpu";
  unsigned gas_rebuild_interval = 0;
  bool gas_rebuild_static_quality = false;
  bool wind_core_profile = false;
  unsigned benchmark_wind_frames = 0;
  unsigned benchmark_static_frames = 0;
  unsigned benchmark_core_frames = 0;
  unsigned benchmark_aabb_frames = 0;
};

class RenderApp {
public:
  int run(const AppOptions& options);
};

}  // namespace rr
