#pragma once

#include <string>
#include <vector>
#include "shared/ribbon_types.h"

class OptixRibbonRenderer {
public:
  OptixRibbonRenderer() = default;
  ~OptixRibbonRenderer();

  void initialize(const std::string& ptx_path);
  void set_primitives(const std::vector<RibbonPrimitive>& primitives);
  void render_to_ppm(const std::string& output_path, unsigned width, unsigned height);

private:
  struct Impl;
  Impl* impl_ = nullptr;
};

