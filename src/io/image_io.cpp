#include "io/image_io.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

#define TINYEXR_USE_MINIZ 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace rr {

static float clamp01(float v) {
  return std::min(1.0f, std::max(0.0f, v));
}

bool path_has_extension(const std::string& path, const char* ext_lowercase) {
  if (!ext_lowercase) return false;
  std::string p = path;
  std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  std::string e = ext_lowercase;
  std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (p.size() < e.size()) return false;
  return p.rfind(e) == p.size() - e.size();
}

void save_rgb_ldr(const std::string& path, const std::vector<uchar4>& pixels, unsigned width, unsigned height) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open output image: " + path);
  out << "P6\n" << width << " " << height << "\n255\n";
  for (const auto& p : pixels) {
    out.put(static_cast<char>(p.x));
    out.put(static_cast<char>(p.y));
    out.put(static_cast<char>(p.z));
  }
}

void save_rgb_hdr_exr(const std::string& path, const std::vector<float4>& pixels, unsigned width, unsigned height) {
  std::vector<float> images[3];
  images[0].resize(width * height);
  images[1].resize(width * height);
  images[2].resize(width * height);
  for (size_t i = 0; i < pixels.size(); ++i) {
    images[0][i] = pixels[i].x;
    images[1][i] = pixels[i].y;
    images[2][i] = pixels[i].z;
  }

  EXRHeader header;
  InitEXRHeader(&header);
  EXRImage image;
  InitEXRImage(&image);
  image.num_channels = 3;

  std::vector<float*> image_ptrs = {images[2].data(), images[1].data(), images[0].data()};
  image.images = reinterpret_cast<unsigned char**>(image_ptrs.data());
  image.width = static_cast<int>(width);
  image.height = static_cast<int>(height);

  header.num_channels = 3;
  header.channels = static_cast<EXRChannelInfo*>(malloc(sizeof(EXRChannelInfo) * header.num_channels));
  strncpy(header.channels[0].name, "B", 255);
  header.channels[0].name[strlen("B")] = '\0';
  strncpy(header.channels[1].name, "G", 255);
  header.channels[1].name[strlen("G")] = '\0';
  strncpy(header.channels[2].name, "R", 255);
  header.channels[2].name[strlen("R")] = '\0';

  header.pixel_types = static_cast<int*>(malloc(sizeof(int) * header.num_channels));
  header.requested_pixel_types = static_cast<int*>(malloc(sizeof(int) * header.num_channels));
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
  }

  const char* err = nullptr;
  int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);
  if (ret != TINYEXR_SUCCESS) {
    std::string msg = "SaveEXRImageToFile failed";
    if (err) {
      msg += ": ";
      msg += err;
      FreeEXRErrorMessage(err);
    }
    throw std::runtime_error(msg);
  }
}

void save_aov_from_float4_ppm(
    const std::string& path,
    const std::vector<float4>& pixels,
    unsigned width,
    unsigned height,
    bool remap_normal,
    bool depth_mode) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open output image: " + path);
  out << "P6\n" << width << " " << height << "\n255\n";
  float max_depth = 1.0f;
  if (depth_mode) {
    max_depth = 0.0f;
    for (const auto& p : pixels) {
      if (p.w > 0.5f) max_depth = std::max(max_depth, p.x);
    }
    max_depth = std::max(max_depth, 1.0f);
  }
  for (const auto& p : pixels) {
    float r = p.x, g = p.y, b = p.z;
    if (depth_mode) {
      float d = (p.w > 0.5f) ? (p.x / max_depth) : 1.0f;
      d = clamp01(d);
      r = g = b = d;
    } else if (remap_normal) {
      r = 0.5f * (r + 1.0f);
      g = 0.5f * (g + 1.0f);
      b = 0.5f * (b + 1.0f);
    }
    unsigned char rc = static_cast<unsigned char>(255.0f * clamp01(r));
    unsigned char gc = static_cast<unsigned char>(255.0f * clamp01(g));
    unsigned char bc = static_cast<unsigned char>(255.0f * clamp01(b));
    out.put(static_cast<char>(rc));
    out.put(static_cast<char>(gc));
    out.put(static_cast<char>(bc));
  }
}

}  // namespace rr
