#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace rr {

void save_rgb_ldr(const std::string& path, const std::vector<uchar4>& pixels, unsigned width, unsigned height);
void save_rgb_hdr_exr(const std::string& path, const std::vector<float4>& pixels, unsigned width, unsigned height);
void save_aov_from_float4_ppm(
    const std::string& path,
    const std::vector<float4>& pixels,
    unsigned width,
    unsigned height,
    bool remap_normal,
    bool depth_mode);

bool path_has_extension(const std::string& path, const char* ext_lowercase);

}  // namespace rr

