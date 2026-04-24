#include "optix_renderer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static float3 make3(float x, float y, float z) { return make_float3(x, y, z); }
static float3 operator-(const float3& a, const float3& b) { return make3(a.x - b.x, a.y - b.y, a.z - b.z); }
static float3 operator*(const float3& a, float s) { return make3(a.x * s, a.y * s, a.z * s); }

static float3 cross3(const float3& a, const float3& b) {
  return make_float3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
}

static float dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static float3 normalize3(const float3& v) {
  float len = std::sqrt(std::max(1e-20f, dot3(v, v)));
  return make3(v.x / len, v.y / len, v.z / len);
}

static RibbonPrimitive make_ribbon3(
    const float3& b0,
    const float3& b1,
    const float3& b2,
    const float3& normal0,
    const float3& normal1,
    float width) {
  RibbonPrimitive p = {};
  p.type = RIBBON_TYPE_3;
  p.r3.b0 = b0;
  p.r3.b1 = b1;
  p.r3.b2 = b2;
  const float3 t0 = normalize3((b1 - b0) * 2.0f);
  const float3 t1 = normalize3((b2 - b1) * 2.0f);
  p.r3.bt0 = normalize3(cross3(t0, normal0));
  p.r3.bt1 = normalize3(cross3(t1, normal1));
  if (dot3(p.r3.bt0, p.r3.bt1) < 0.0f) p.r3.bt1 = p.r3.bt1 * -1.0f;
  p.r3.alpha = 1.0f;
  p.r3.beta = 0.0f;
  p.r3.gamma = 0.0f;
  p.r3.w0 = width;
  p.r3.w1 = 0.0f;
  p.r3.w2 = 0.0f;
  return p;
}

static RibbonPrimitive make_ribbon2(const Ribbon2Data& r2) {
  RibbonPrimitive p = {};
  p.type = RIBBON_TYPE_2;
  p.r2 = r2;
  return p;
}

static std::string resolve_ptx_path(const char* argv0, int argc, char** argv) {
  namespace fs = std::filesystem;

  if (argc > 1) {
    return argv[1];
  }

  const fs::path exe_path = fs::absolute(fs::path(argv0));
  const fs::path exe_dir = exe_path.parent_path();
  const fs::path cwd = fs::current_path();

  const std::vector<fs::path> candidates = {
      cwd / "ribbon_kernels.ptx",
      cwd / "build" / "Release" / "ribbon_kernels.ptx",
      exe_dir / "ribbon_kernels.ptx",
      exe_dir / ".." / "ribbon_kernels.ptx"};

  for (const auto& p : candidates) {
    if (fs::exists(p)) return p.string();
  }

  throw std::runtime_error("Cannot find ribbon_kernels.ptx. Pass PTX path as argv[1].");
}

int main(int argc, char** argv) {
  try {
    const std::string ptx = resolve_ptx_path(argv[0], argc, argv);
    const std::string output = (argc > 2) ? argv[2] : "output.ppm";

    std::vector<RibbonPrimitive> primitives;

    Ribbon2Data r2 = {};
    r2.q00 = make3(-1.2f, -0.4f, -3.6f);
    r2.q01 = make3(-1.2f, 0.1f, -3.6f);
    r2.q10 = make3(1.0f, -0.4f, -5.2f);
    r2.q11 = make3(1.0f, 0.0f, -5.2f);
    r2.w0 = 0.10f;
    r2.w1 = 0.02f;
    r2.w2 = -0.01f;
    primitives.push_back(make_ribbon2(r2));

    primitives.push_back(make_ribbon3(
        make3(-0.8f, -0.1f, -3.0f),
        make3(0.0f, 1.2f, -4.2f),
        make3(0.9f, 0.0f, -5.0f),
        make3(0.0f, 1.0f, 0.0f),
        make3(0.0f, 1.0f, 0.0f),
        0.13f));

    OptixRibbonRenderer renderer;
    renderer.initialize(ptx);
    renderer.set_primitives(primitives);
    renderer.render_to_ppm(output, 1280, 720);

    std::cout << "Render complete: " << output << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
