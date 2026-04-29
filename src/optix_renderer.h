#pragma once

#include <string>
#include <vector>
#include "shared/ribbon_types.h"
#include "shared/light_types.h"
#include "scene/scene.h"

class OptixRibbonRenderer {
public:
  OptixRibbonRenderer() = default;
  ~OptixRibbonRenderer();

  void initialize(const std::string& ptx_path);
  void set_primitives(const std::vector<RibbonPrimitive>& primitives);
  void set_lights(const std::vector<LightData>& lights);
  void set_camera(const rr::CameraSettings& camera);
  void set_quality(unsigned spp, unsigned max_depth, float exposure, float gamma, float firefly_clamp);
  void render_to_ppm(
      const std::string& output_path,
      unsigned width,
      unsigned height,
      const std::string& aov_dir = "",
      bool denoise = false);

private:
  struct Impl;
  Impl* impl_ = nullptr;
};
