#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

#include "shared/ribbon_types.h"
#include "shared/light_types.h"
#include "shared/launch_params.h"
#include "scene/scene.h"
#include "device/animation_kernels.h"

class OptixRibbonRenderer {
public:
  OptixRibbonRenderer() = default;
  ~OptixRibbonRenderer();

  void initialize(const std::string& ptx_path);
  void set_primitives(const std::vector<RibbonPrimitive>& primitives);
  void prepare_dynamic_primitives(const std::vector<RibbonPrimitive>& primitives);
  void set_primitives_with_grass_wind_bounds(const std::vector<RibbonPrimitive>& primitives, float strength);
  void update_dynamic_primitives_on_host(const std::vector<RibbonPrimitive>& primitives);
  void animate_primitives_on_gpu(DeviceWindMode mode, float time, float strength, float speed);
  void animate_primitives_on_gpu_without_aabb(DeviceWindMode mode, float time, float strength, float speed);
  void update_aabbs_on_gpu();
  void rebuild_gas_from_device_primitives(bool force_rebuild = false, bool allow_update_after_rebuild = true);
  void set_lights(const std::vector<LightData>& lights);
  void set_camera(const rr::CameraSettings& camera);
  void set_quality(unsigned spp, unsigned max_depth, float exposure, float gamma, float firefly_clamp);
  void set_render_core_profiling(bool enabled);
  void set_shader_wind(bool enabled, float time = 0.0f, float strength = 0.0f, float speed = 1.0f);
  void render_frame(unsigned width, unsigned height, std::vector<uchar4>& pixels);
  void render_frame_device(unsigned width, unsigned height);
  void render_frame_to_device_buffer(unsigned width, unsigned height, CUdeviceptr d_output_pixels);
  void download_frame(unsigned width, unsigned height, std::vector<uchar4>& pixels);
  void read_render_core_stats(RenderCoreStats& stats) const;
  void get_gpu_memory_info(size_t& free_bytes, size_t& total_bytes) const;
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
