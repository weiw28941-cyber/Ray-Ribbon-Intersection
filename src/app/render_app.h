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
};

class RenderApp {
public:
  int run(const AppOptions& options);
};

}  // namespace rr
