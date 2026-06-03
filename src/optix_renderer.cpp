#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "optix_renderer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "shared/launch_params.h"
#include "io/image_io.h"
#include "device/animation_kernels.h"

static std::string cu_error_message(CUresult rc) {
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(rc, &name);
  cuGetErrorString(rc, &msg);
  return std::string(name ? name : "unknown") + ": " + (msg ? msg : "unknown");
}

static std::string optix_error_message(OptixResult rc) {
  return std::string(optixGetErrorName(rc)) + ": " + optixGetErrorString(rc);
}

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t rc = (call);                                                             \
    if (rc != cudaSuccess)                                                               \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(rc));   \
  } while (0)

#define CU_CHECK(call)                                                                          \
  do {                                                                                          \
    CUresult rc = (call);                                                                       \
    if (rc != CUDA_SUCCESS)                                                                     \
      throw std::runtime_error(std::string("CUDA Driver API error at line ") + std::to_string(__LINE__) + \
                               ": " + cu_error_message(rc));                                  \
  } while (0)

#define OPTIX_CHECK(call)                                                                          \
  do {                                                                                             \
    OptixResult rc = (call);                                                                       \
    if (rc != OPTIX_SUCCESS)                                                                       \
      throw std::runtime_error(std::string("OptiX error at line ") + std::to_string(__LINE__) +  \
                               ": " + optix_error_message(rc));                                  \
  } while (0)

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct RaygenData {};
struct MissData {
  float3 bg;
};
struct HitData {};
struct ShadowHitData {};

static float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static float3 operator*(const float3& a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
static float3 operator*(float s, const float3& a) { return a * s; }
static float dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float3 normalize3(const float3& a) {
  float l2 = fmaxf(1e-20f, dot3(a, a));
  float inv = 1.0f / std::sqrt(l2);
  return a * inv;
}
static float3 lerp3(const float3& a, const float3& b, float t) { return a * (1.0f - t) + b * t; }
static float eval_poly(const float* c, int degree, float u) {
  float v = c[degree];
  for (int i = degree - 1; i >= 0; --i) v = v * u + c[i];
  return v;
}

static float component3(const float3& v, int coord) {
  return coord == 0 ? v.x : (coord == 1 ? v.y : v.z);
}

static int solve_quadratic_roots(float a, float b, float c, float roots[2]) {
  constexpr float eps = 1e-12f;
  if (std::fabs(a) < eps) {
    if (std::fabs(b) < eps) return 0;
    roots[0] = -c / b;
    return 1;
  }
  const float disc = b * b - 4.0f * a * c;
  if (disc < 0.0f) return 0;
  const float s = std::sqrt(std::max(0.0f, disc));
  roots[0] = (-b - s) / (2.0f * a);
  roots[1] = (-b + s) / (2.0f * a);
  return std::fabs(roots[1] - roots[0]) > 1e-6f ? 2 : 1;
}

static void expand_poly_extrema(
    const float* c,
    int degree,
    const std::function<void(float)>& expand_coord) {
  expand_coord(eval_poly(c, degree, 0.0f));
  expand_coord(eval_poly(c, degree, 1.0f));

  if (degree == 2) {
    constexpr float eps = 1e-12f;
    if (std::fabs(c[2]) > eps) {
      const float u = -c[1] / (2.0f * c[2]);
      if (u > 0.0f && u < 1.0f) {
        expand_coord(eval_poly(c, degree, u));
      }
    }
    return;
  }

  if (degree == 3) {
    float roots[2];
    const int n = solve_quadratic_roots(3.0f * c[3], 2.0f * c[2], c[1], roots);
    for (int i = 0; i < n; ++i) {
      if (roots[i] > 0.0f && roots[i] < 1.0f) {
        expand_coord(eval_poly(c, degree, roots[i]));
      }
    }
    return;
  }

  if (degree == 4 || degree == 5) {
    float d[5] = {};
    for (int i = 0; i < degree; ++i) d[i] = static_cast<float>(i + 1) * c[i + 1];
    constexpr int kIntervals = 64;
    float prev_u = 0.0f;
    float prev_f = eval_poly(d, degree - 1, prev_u);
    for (int i = 1; i <= kIntervals; ++i) {
      const float u = static_cast<float>(i) / static_cast<float>(kIntervals);
      const float f = eval_poly(d, degree - 1, u);
      if (prev_f == 0.0f) {
        expand_coord(eval_poly(c, degree, prev_u));
      } else if ((prev_f < 0.0f) != (f < 0.0f)) {
        float lo = prev_u;
        float hi = u;
        for (int it = 0; it < 24; ++it) {
          const float mid = 0.5f * (lo + hi);
          const float fm = eval_poly(d, degree - 1, mid);
          if ((prev_f < 0.0f) != (fm < 0.0f)) {
            hi = mid;
          } else {
            lo = mid;
            prev_f = fm;
          }
        }
        expand_coord(eval_poly(c, degree, 0.5f * (lo + hi)));
      }
      prev_u = u;
      prev_f = f;
    }
    return;
  }
}

static void ribbon2_edge_poly_coord(const Ribbon2Data& r, int coord, float sign, float out[4]) {
  const float q00 = coord == 0 ? r.q00.x : (coord == 1 ? r.q00.y : r.q00.z);
  const float q01 = coord == 0 ? r.q01.x : (coord == 1 ? r.q01.y : r.q01.z);
  const float q10 = coord == 0 ? r.q10.x : (coord == 1 ? r.q10.y : r.q10.z);
  const float q11 = coord == 0 ? r.q11.x : (coord == 1 ? r.q11.y : r.q11.z);
  const float d0 = q00;
  const float d1 = q10 - q00;
  const float g0 = q01 - q00;
  const float g1 = q00 - q01 + q11 - q10;
  const float v0 = sign * r.w0;
  const float v1 = 1.0f + sign * r.w1;
  const float v2 = sign * r.w2;
  out[0] = d0 + v0 * g0;
  out[1] = d1 + v0 * g1 + v1 * g0;
  out[2] = v1 * g1 + v2 * g0;
  out[3] = v2 * g1;
}

static void ribbon3_generator_poly(const Ribbon3Data& r, float3 g[4]) {
  const float3 c1 = 2.0f * (r.b1 - r.b0);
  const float3 c2 = r.b0 - 2.0f * r.b1 + r.b2;
  const float3 a = r.beta * c2 + r.gamma * c1;
  g[0] = r.bt0;
  g[1] = r.alpha * r.bt1 - r.bt0 + a;
  g[2] = (-1.0f * a) + r.gamma * c2;
  g[3] = -r.gamma * c2;
}

static void ribbon3_edge_poly_coord(const Ribbon3Data& r, int coord, float sign, float out[6]) {
  const float3 c1 = 2.0f * (r.b1 - r.b0);
  const float3 c2 = r.b0 - 2.0f * r.b1 + r.b2;
  float3 g[4];
  ribbon3_generator_poly(r, g);
  const float c[3] = {
      coord == 0 ? r.b0.x : (coord == 1 ? r.b0.y : r.b0.z),
      coord == 0 ? c1.x : (coord == 1 ? c1.y : c1.z),
      coord == 0 ? c2.x : (coord == 1 ? c2.y : c2.z)};
  const float gc[4] = {
      coord == 0 ? g[0].x : (coord == 1 ? g[0].y : g[0].z),
      coord == 0 ? g[1].x : (coord == 1 ? g[1].y : g[1].z),
      coord == 0 ? g[2].x : (coord == 1 ? g[2].y : g[2].z),
      coord == 0 ? g[3].x : (coord == 1 ? g[3].y : g[3].z)};
  const float w[3] = {r.w0, r.w1, r.w2};
  for (int i = 0; i < 6; ++i) out[i] = 0.0f;
  out[0] = c[0];
  out[1] = c[1];
  out[2] = c[2];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i + j] += sign * w[i] * gc[j];
    }
  }
}

static void ribbon4_edge_poly_coord(const Ribbon4Data& r, int coord, float sign, int degree, float out[5]) {
  const float d[4] = {
      component3(r.d0, coord),
      component3(r.d1, coord),
      component3(r.d2, coord),
      component3(r.d3, coord)};
  const float g[3] = {
      component3(r.g0, coord),
      component3(r.g1, coord),
      component3(r.g2, coord)};
  const float w[3] = {r.w0, r.w1, r.w2};
  for (int i = 0; i < 5; ++i) out[i] = 0.0f;
  for (int i = 0; i <= degree && i < 4; ++i) out[i] = d[i];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i + j <= degree) out[i + j] += sign * w[i] * g[j];
    }
  }
}

static void ribbon4c_edge_poly_coord(const Ribbon4Data& r, int coord, float sign, float out[3]) {
  out[0] = component3(r.d0, coord) + sign * component3(r.d1, coord);
  out[1] = component3(r.g0, coord) + sign * component3(r.g1, coord);
  out[2] = component3(r.d2, coord) + sign * component3(r.d3, coord);
}

static OptixAabb primitive_aabb(const RibbonPrimitive& p) {
  float3 mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  auto expand_coord = [&](int coord, float value) {
    if (coord == 0) {
      mn.x = fminf(mn.x, value);
      mx.x = fmaxf(mx.x, value);
    } else if (coord == 1) {
      mn.y = fminf(mn.y, value);
      mx.y = fmaxf(mx.y, value);
    } else {
      mn.z = fminf(mn.z, value);
      mx.z = fmaxf(mx.z, value);
    }
  };

  if (p.type == RIBBON_TYPE_2) {
    for (float sign : {-1.0f, 1.0f}) {
      for (int coord = 0; coord < 3; ++coord) {
        float c[4];
        ribbon2_edge_poly_coord(p.r2, coord, sign, c);
        expand_poly_extrema(c, 3, [&](float v) { expand_coord(coord, v); });
      }
    }
  } else if (p.type == RIBBON_TYPE_3) {
    for (float sign : {-1.0f, 1.0f}) {
      for (int coord = 0; coord < 3; ++coord) {
        float c[6];
        ribbon3_edge_poly_coord(p.r3, coord, sign, c);
        expand_poly_extrema(c, 5, [&](float v) { expand_coord(coord, v); });
      }
    }
  } else if (p.type == RIBBON_TYPE_4A || p.type == RIBBON_TYPE_4B) {
    const int degree = p.type == RIBBON_TYPE_4A ? 3 : 4;
    for (float sign : {-1.0f, 1.0f}) {
      for (int coord = 0; coord < 3; ++coord) {
        float c[5];
        ribbon4_edge_poly_coord(p.r4, coord, sign, degree, c);
        expand_poly_extrema(c, degree, [&](float v) { expand_coord(coord, v); });
      }
    }
  } else if (p.type == RIBBON_TYPE_4C) {
    for (float sign : {-1.0f, 1.0f}) {
      for (int coord = 0; coord < 3; ++coord) {
        float c[3];
        ribbon4c_edge_poly_coord(p.r4, coord, sign, c);
        expand_poly_extrema(c, 2, [&](float v) { expand_coord(coord, v); });
      }
    }
  }

  constexpr float pad = 1e-3f;
  return OptixAabb{mn.x - pad, mn.y - pad, mn.z - pad, mx.x + pad, mx.y + pad, mx.z + pad};
}

static float3 ribbon4_directrix_host(const Ribbon4Data& r, float u) {
  return r.d0 + r.d1 * u + r.d2 * (u * u) + r.d3 * (u * u * u);
}

static float3 ribbon4c_center_host(const Ribbon4Data& r, float v) {
  return r.d0 + r.g0 * v + r.d2 * (v * v);
}

static float3 ribbon4_generator_host(const Ribbon4Data& r, float u) {
  return r.g0 + r.g1 * u + r.g2 * (u * u);
}

static void expand_aabb(OptixAabb& dst, const OptixAabb& src) {
  dst.minX = std::min(dst.minX, src.minX);
  dst.minY = std::min(dst.minY, src.minY);
  dst.minZ = std::min(dst.minZ, src.minZ);
  dst.maxX = std::max(dst.maxX, src.maxX);
  dst.maxY = std::max(dst.maxY, src.maxY);
  dst.maxZ = std::max(dst.maxZ, src.maxZ);
}

static void add_quadratic_directrix_delta_host(Ribbon4Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = dm - d0;
  const float3 e = d1 - d0;
  const float3 a = m * 4.0f - e;
  const float3 b = e * 2.0f - m * 4.0f;
  r.d0 = r.d0 + d0;
  r.d1 = r.d1 + a;
  r.d2 = r.d2 + b;
}

static void tilt_ribbon4_generator_host(Ribbon4Data& r, const float3& d, float amount) {
  const float3 g1 = ribbon4_generator_host(r, 1.0f);
  const float3 target = normalize3(g1 + d * amount);
  r.g1 = r.g1 + (target - g1);
}

static RibbonPrimitive grass_wind_displaced_primitive(const RibbonPrimitive& p, float strength, float signed_scale) {
  RibbonPrimitive out = p;
  if (std::fabs(strength) < 1e-8f) return out;
  if (!(p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A || p.type == RIBBON_TYPE_4B ||
        p.type == RIBBON_TYPE_4C)) return out;
  const float3 base = p.type == RIBBON_TYPE_3 ? p.r3.b0 :
      (p.type == RIBBON_TYPE_4C ? ribbon4c_center_host(p.r4, 0.0f) : ribbon4_directrix_host(p.r4, 0.0f));
  if (std::fabs(base.y) >= 0.08f) return out;
  const float3 tip = p.type == RIBBON_TYPE_3 ? p.r3.b2 :
      (p.type == RIBBON_TYPE_4C ? ribbon4c_center_host(p.r4, 1.0f) : ribbon4_directrix_host(p.r4, 1.0f));
  const float height = std::max(0.05f, tip.y - base.y);
  const float amp = strength * height * 1.20f * signed_scale;
  const float3 dir = normalize3(make_float3(0.82f, 0.0f, -0.38f));
  const float3 d = dir * amp;
  const float dlen = std::sqrt(std::max(0.0f, dot3(d, d)));
  if (p.type == RIBBON_TYPE_3) {
    out.r3.b1 = out.r3.b1 + d * 0.48f;
    out.r3.b2 = out.r3.b2 + d;
    out.r3.b1.y -= dlen * 0.03f;
    out.r3.b2.y -= dlen * 0.07f;
    out.r3.bt1 = normalize3(out.r3.bt1 + d * 0.10f);
  } else if (p.type == RIBBON_TYPE_4C) {
    const float3 mid = d * 0.48f + make_float3(0.0f, -dlen * 0.03f, 0.0f);
    const float3 end = d + make_float3(0.0f, -dlen * 0.07f, 0.0f);
    out.r4.g0 = out.r4.g0 + (mid * 4.0f - end);
    out.r4.d2 = out.r4.d2 + (end * 2.0f - mid * 4.0f);
  } else {
    const float3 mid = d * 0.48f + make_float3(0.0f, -dlen * 0.03f, 0.0f);
    const float3 end = d + make_float3(0.0f, -dlen * 0.07f, 0.0f);
    add_quadratic_directrix_delta_host(out.r4, make_float3(0.0f, 0.0f, 0.0f), mid, end);
    tilt_ribbon4_generator_host(out.r4, d, 0.10f);
  }
  return out;
}

static OptixAabb grass_wind_swept_aabb(const RibbonPrimitive& p, float strength) {
  if (std::fabs(strength) < 1e-8f) return primitive_aabb(p);
  OptixAabb box = primitive_aabb(p);
  expand_aabb(box, primitive_aabb(grass_wind_displaced_primitive(p, strength, 1.0f)));
  expand_aabb(box, primitive_aabb(grass_wind_displaced_primitive(p, strength, -1.0f)));
  return box;
}

static std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::string s;
  in.seekg(0, std::ios::end);
  s.resize(static_cast<size_t>(in.tellg()));
  in.seekg(0, std::ios::beg);
  in.read(s.data(), static_cast<std::streamsize>(s.size()));
  return s;
}

static std::vector<uchar4> tonemap_to_ldr(const std::vector<float4>& hdr) {
  std::vector<uchar4> out(hdr.size());
  for (size_t i = 0; i < hdr.size(); ++i) {
    const float r = hdr[i].x / (1.0f + hdr[i].x);
    const float g = hdr[i].y / (1.0f + hdr[i].y);
    const float b = hdr[i].z / (1.0f + hdr[i].z);
    out[i] = make_uchar4(
        static_cast<unsigned char>(255.0f * fminf(1.0f, fmaxf(0.0f, r))),
        static_cast<unsigned char>(255.0f * fminf(1.0f, fmaxf(0.0f, g))),
        static_cast<unsigned char>(255.0f * fminf(1.0f, fmaxf(0.0f, b))),
        255);
  }
  return out;
}

struct OptixRibbonRenderer::Impl {
  CUcontext cu_ctx = nullptr;
  CUstream stream = nullptr;
  OptixDeviceContext optix_ctx = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  OptixProgramGroup raygen_pg = nullptr;
  OptixProgramGroup miss_pg = nullptr;
  OptixProgramGroup miss_shadow_pg = nullptr;
  OptixProgramGroup hit_pg = nullptr;
  OptixProgramGroup hit_shadow_pg = nullptr;
  OptixShaderBindingTable sbt = {};
  CUdeviceptr d_raygen_record = 0;
  CUdeviceptr d_miss_record = 0;
  CUdeviceptr d_hit_record = 0;
  CUdeviceptr d_primitives = 0;
  CUdeviceptr d_base_primitives = 0;
  CUdeviceptr d_aabb = 0;
  CUdeviceptr d_gas = 0;
  size_t gas_buffer_size = 0;
  size_t gas_temp_build_size = 0;
  size_t gas_temp_update_size = 0;
  bool gas_allow_update = false;
  bool gas_is_built = false;
  CUdeviceptr d_launch_params = 0;
  CUdeviceptr d_render_stats = 0;
  CUdeviceptr d_frame_pixels = 0;
  CUdeviceptr d_frame_albedo = 0;
  CUdeviceptr d_frame_normal = 0;
  CUdeviceptr d_frame_depth = 0;
  CUdeviceptr d_frame_beauty_hdr = 0;
  unsigned frame_width = 0;
  unsigned frame_height = 0;
  OptixTraversableHandle gas = 0;
  unsigned primitive_count = 0;
  std::vector<LightData> lights;
  rr::CameraSettings camera = {};
  unsigned spp = 8;
  unsigned max_depth = 4;
  float exposure = 1.0f;
  float gamma = 2.2f;
  float firefly_clamp = 10.0f;
  bool render_core_profiling = false;
  bool shader_wind_enabled = false;
  unsigned shader_wind_mode = 0;
  float shader_wind_time = 0.0f;
  float shader_wind_strength = 0.0f;
  float shader_wind_speed = 1.0f;

  void release_geometry_buffers() {
    if (d_base_primitives) CU_CHECK(cuMemFree(d_base_primitives));
    if (d_primitives) CU_CHECK(cuMemFree(d_primitives));
    if (d_aabb) CU_CHECK(cuMemFree(d_aabb));
    if (d_gas) CU_CHECK(cuMemFree(d_gas));
    d_base_primitives = 0;
    d_primitives = 0;
    d_aabb = 0;
    d_gas = 0;
    gas = 0;
    primitive_count = 0;
    gas_buffer_size = 0;
    gas_temp_build_size = 0;
    gas_temp_update_size = 0;
    gas_allow_update = false;
    gas_is_built = false;
  }

  OptixBuildInput make_build_input() const {
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = const_cast<CUdeviceptr*>(&d_aabb);
    input.customPrimitiveArray.numPrimitives = primitive_count;
    static const uint32_t flag = OPTIX_GEOMETRY_FLAG_NONE;
    input.customPrimitiveArray.flags = &flag;
    input.customPrimitiveArray.numSbtRecords = 1;
    return input;
  }

  void build_gas_from_current_aabbs(bool allow_update) {
    if (!d_aabb || primitive_count == 0) throw std::runtime_error("AABBs were not prepared");
    if (d_gas) CU_CHECK(cuMemFree(d_gas));
    d_gas = 0;

    OptixBuildInput input = make_build_input();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = allow_update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_ctx, &accel_options, &input, 1, &sizes));
    gas_buffer_size = sizes.outputSizeInBytes;
    gas_temp_build_size = sizes.tempSizeInBytes;
    gas_temp_update_size = sizes.tempUpdateSizeInBytes;
    gas_allow_update = allow_update;

    CUdeviceptr d_temp = 0;
    if (gas_temp_build_size > 0) {
      CU_CHECK(cuMemAlloc(&d_temp, gas_temp_build_size));
    }
    CU_CHECK(cuMemAlloc(&d_gas, gas_buffer_size));

    OPTIX_CHECK(optixAccelBuild(
        optix_ctx,
        stream,
        &accel_options,
        &input,
        1,
        d_temp,
        gas_temp_build_size,
        d_gas,
        gas_buffer_size,
        &gas,
        nullptr,
        0));
    if (d_temp) CU_CHECK(cuMemFree(d_temp));
    CU_CHECK(cuStreamSynchronize(stream));
    gas_is_built = true;
  }

  void update_gas_from_current_aabbs() {
    if (!d_aabb || primitive_count == 0) throw std::runtime_error("AABBs were not prepared");
    if (!gas_allow_update || !gas_is_built || !d_gas) {
      build_gas_from_current_aabbs(true);
      return;
    }

    OptixBuildInput input = make_build_input();
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    CUdeviceptr d_temp = 0;
    if (gas_temp_update_size > 0) {
      CU_CHECK(cuMemAlloc(&d_temp, gas_temp_update_size));
    }
    OPTIX_CHECK(optixAccelBuild(
        optix_ctx,
        stream,
        &accel_options,
        &input,
        1,
        d_temp,
        gas_temp_update_size,
        d_gas,
        gas_buffer_size,
        &gas,
        nullptr,
        0));
    if (d_temp) CU_CHECK(cuMemFree(d_temp));
    CU_CHECK(cuStreamSynchronize(stream));
  }

  void release_frame_buffers() {
    if (d_frame_pixels) CU_CHECK(cuMemFree(d_frame_pixels));
    if (d_frame_albedo) CU_CHECK(cuMemFree(d_frame_albedo));
    if (d_frame_normal) CU_CHECK(cuMemFree(d_frame_normal));
    if (d_frame_depth) CU_CHECK(cuMemFree(d_frame_depth));
    if (d_frame_beauty_hdr) CU_CHECK(cuMemFree(d_frame_beauty_hdr));
    d_frame_pixels = 0;
    d_frame_albedo = 0;
    d_frame_normal = 0;
    d_frame_depth = 0;
    d_frame_beauty_hdr = 0;
    frame_width = 0;
    frame_height = 0;
  }

  void ensure_frame_buffers(unsigned width, unsigned height) {
    if (frame_width == width && frame_height == height && d_frame_pixels) {
      return;
    }
    release_frame_buffers();
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    CU_CHECK(cuMemAlloc(&d_frame_pixels, pixel_count * sizeof(uchar4)));
    CU_CHECK(cuMemAlloc(&d_frame_albedo, pixel_count * sizeof(float4)));
    CU_CHECK(cuMemAlloc(&d_frame_normal, pixel_count * sizeof(float4)));
    CU_CHECK(cuMemAlloc(&d_frame_depth, pixel_count * sizeof(float4)));
    CU_CHECK(cuMemAlloc(&d_frame_beauty_hdr, pixel_count * sizeof(float4)));
    frame_width = width;
    frame_height = height;
  }

  void fill_launch_params(
      LaunchParams& lp,
      CUdeviceptr d_pixels,
      CUdeviceptr d_albedo,
      CUdeviceptr d_normal,
      CUdeviceptr d_depth,
      CUdeviceptr d_beauty_hdr,
      unsigned width,
      unsigned height) {
    lp = {};
    lp.image = reinterpret_cast<uchar4*>(d_pixels);
    lp.aov_albedo = reinterpret_cast<float4*>(d_albedo);
    lp.aov_normal = reinterpret_cast<float4*>(d_normal);
    lp.aov_depth = reinterpret_cast<float4*>(d_depth);
    lp.aov_beauty_hdr = reinterpret_cast<float4*>(d_beauty_hdr);
    lp.width = width;
    lp.height = height;
    lp.spp = spp;
    lp.max_depth = max_depth;
    lp.exposure = exposure;
    lp.gamma = gamma;
    lp.firefly_clamp = firefly_clamp;
    lp.gas = gas;
    lp.primitives = reinterpret_cast<RibbonPrimitive*>(d_primitives);
    lp.primitive_count = primitive_count;
    lp.light_count = static_cast<unsigned>(std::min<size_t>(lights.size(), 8));
    for (unsigned i = 0; i < lp.light_count; ++i) lp.lights[i] = lights[i];
    lp.stats = render_core_profiling ? reinterpret_cast<RenderCoreStats*>(d_render_stats) : nullptr;
    lp.shader_wind_enabled = shader_wind_enabled ? 1u : 0u;
    lp.shader_wind_mode = shader_wind_mode;
    lp.shader_wind_time = shader_wind_time;
    lp.shader_wind_strength = shader_wind_strength;
    lp.shader_wind_speed = shader_wind_speed;

    const float3 look_from = camera.look_from;
    const float3 look_at = camera.look_at;
    const float3 up = camera.up;
    const float3 f = normalize3(look_at - look_from);
    const float3 r = normalize3(make_float3(
        up.y * f.z - up.z * f.y,
        up.z * f.x - up.x * f.z,
        up.x * f.y - up.y * f.x));
    const float3 u = make_float3(
        f.y * r.z - f.z * r.y,
        f.z * r.x - f.x * r.z,
        f.x * r.y - f.y * r.x);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float tan_half_fov = std::tan(camera.fov_deg * 0.5f * 3.14159265f / 180.0f);
    lp.camera.origin = look_from;
    lp.camera.horizontal = r * (2.0f * tan_half_fov * aspect);
    lp.camera.vertical = u * (2.0f * tan_half_fov);
    lp.camera.lower_left = f - 0.5f * lp.camera.horizontal - 0.5f * lp.camera.vertical;
  }

  void launch_frame(unsigned width, unsigned height, CUdeviceptr d_output_pixels) {
    ensure_frame_buffers(width, height);
    if (render_core_profiling) {
      if (!d_render_stats) {
        CU_CHECK(cuMemAlloc(&d_render_stats, sizeof(RenderCoreStats)));
      }
      CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<void*>(d_render_stats),
          0,
          sizeof(RenderCoreStats),
          reinterpret_cast<cudaStream_t>(stream)));
    }

    LaunchParams lp = {};
    fill_launch_params(
        lp,
        d_output_pixels ? d_output_pixels : d_frame_pixels,
        d_frame_albedo,
        d_frame_normal,
        d_frame_depth,
        d_frame_beauty_hdr,
        width,
        height);

    if (!d_launch_params) {
      CU_CHECK(cuMemAlloc(&d_launch_params, sizeof(LaunchParams)));
    }
    CU_CHECK(cuMemcpyHtoD(d_launch_params, &lp, sizeof(lp)));

    OPTIX_CHECK(optixLaunch(
        pipeline,
        stream,
        d_launch_params,
        sizeof(LaunchParams),
        &sbt,
        width,
        height,
        1));
    CU_CHECK(cuStreamSynchronize(stream));
  }
};

void OptixRibbonRenderer::initialize(const std::string& ptx_path) {
  if (impl_) return;
  impl_ = new Impl();

  CUDA_CHECK(cudaFree(nullptr));
  OPTIX_CHECK(optixInit());
  CU_CHECK(cuCtxGetCurrent(&impl_->cu_ctx));
  if (!impl_->cu_ctx) {
    CU_CHECK(cuCtxCreate(&impl_->cu_ctx, 0, 0));
  }
  CU_CHECK(cuStreamCreate(&impl_->stream, CU_STREAM_DEFAULT));

  OptixDeviceContextOptions options = {};
  OPTIX_CHECK(optixDeviceContextCreate(impl_->cu_ctx, &options, &impl_->optix_ctx));

  std::string ptx = read_text_file(ptx_path);

  OptixModuleCompileOptions mco = {};
  mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

  OptixPipelineCompileOptions pco = {};
  pco.usesMotionBlur = 0;
  pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pco.numPayloadValues = 2;
  pco.numAttributeValues = 2;
  pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pco.pipelineLaunchParamsVariableName = "params";
#if OPTIX_VERSION >= 90000
  pco.pipelineLaunchParamsSizeInBytes = sizeof(LaunchParams);
#endif

  char log[4096];
  size_t log_size = sizeof(log);
  OPTIX_CHECK(optixModuleCreate(
      impl_->optix_ctx, &mco, &pco, ptx.c_str(), ptx.size(), log, &log_size, &impl_->module));

  OptixProgramGroupOptions pgo = {};
  OptixProgramGroupDesc rg_desc = {};
  rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  rg_desc.raygen.module = impl_->module;
  rg_desc.raygen.entryFunctionName = "__raygen__rg";
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(
      impl_->optix_ctx, &rg_desc, 1, &pgo, log, &log_size, &impl_->raygen_pg));

  OptixProgramGroupDesc ms_desc = {};
  ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  ms_desc.miss.module = impl_->module;
  ms_desc.miss.entryFunctionName = "__miss__ms";
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(
      impl_->optix_ctx, &ms_desc, 1, &pgo, log, &log_size, &impl_->miss_pg));

  OptixProgramGroupDesc ms_shadow_desc = {};
  ms_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  ms_shadow_desc.miss.module = impl_->module;
  ms_shadow_desc.miss.entryFunctionName = "__miss__shadow";
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(
      impl_->optix_ctx, &ms_shadow_desc, 1, &pgo, log, &log_size, &impl_->miss_shadow_pg));

  OptixProgramGroupDesc hg_desc = {};
  hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hg_desc.hitgroup.moduleCH = impl_->module;
  hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  hg_desc.hitgroup.moduleIS = impl_->module;
  hg_desc.hitgroup.entryFunctionNameIS = "__intersection__ribbon";
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(
      impl_->optix_ctx, &hg_desc, 1, &pgo, log, &log_size, &impl_->hit_pg));

  OptixProgramGroupDesc hg_shadow_desc = {};
  hg_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hg_shadow_desc.hitgroup.moduleCH = impl_->module;
  hg_shadow_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
  hg_shadow_desc.hitgroup.moduleIS = impl_->module;
  hg_shadow_desc.hitgroup.entryFunctionNameIS = "__intersection__ribbon";
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(
      impl_->optix_ctx, &hg_shadow_desc, 1, &pgo, log, &log_size, &impl_->hit_shadow_pg));

  std::vector<OptixProgramGroup> pgs = {
      impl_->raygen_pg, impl_->miss_pg, impl_->miss_shadow_pg, impl_->hit_pg, impl_->hit_shadow_pg};
  OptixPipelineLinkOptions plo = {};
  plo.maxTraceDepth = 1;
#if OPTIX_VERSION >= 90000
  plo.maxContinuationCallableDepth = 0;
  plo.maxDirectCallableDepthFromState = 0;
  plo.maxDirectCallableDepthFromTraversal = 0;
  plo.maxTraversableGraphDepth = 1;
#endif
  log_size = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(
      impl_->optix_ctx, &pco, &plo, pgs.data(), static_cast<unsigned>(pgs.size()), log, &log_size, &impl_->pipeline));

  OPTIX_CHECK(optixPipelineSetStackSize(impl_->pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1));

  SbtRecord<RaygenData> rg = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(impl_->raygen_pg, &rg));
  CU_CHECK(cuMemAlloc(&impl_->d_raygen_record, sizeof(rg)));
  CU_CHECK(cuMemcpyHtoD(impl_->d_raygen_record, &rg, sizeof(rg)));

  SbtRecord<MissData> ms = {};
  ms.data.bg = make_float3(0.75f, 0.86f, 1.0f);
  OPTIX_CHECK(optixSbtRecordPackHeader(impl_->miss_pg, &ms));
  SbtRecord<MissData> ms_shadow = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(impl_->miss_shadow_pg, &ms_shadow));
  CU_CHECK(cuMemAlloc(&impl_->d_miss_record, sizeof(ms) * 2));
  CU_CHECK(cuMemcpyHtoD(impl_->d_miss_record, &ms, sizeof(ms)));
  CU_CHECK(cuMemcpyHtoD(impl_->d_miss_record + sizeof(ms), &ms_shadow, sizeof(ms_shadow)));

  SbtRecord<HitData> hg = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(impl_->hit_pg, &hg));
  SbtRecord<ShadowHitData> hg_shadow = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(impl_->hit_shadow_pg, &hg_shadow));
  CU_CHECK(cuMemAlloc(&impl_->d_hit_record, sizeof(hg) * 2));
  CU_CHECK(cuMemcpyHtoD(impl_->d_hit_record, &hg, sizeof(hg)));
  CU_CHECK(cuMemcpyHtoD(impl_->d_hit_record + sizeof(hg), &hg_shadow, sizeof(hg_shadow)));

  impl_->sbt.raygenRecord = impl_->d_raygen_record;
  impl_->sbt.missRecordBase = impl_->d_miss_record;
  impl_->sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
  impl_->sbt.missRecordCount = 2;
  impl_->sbt.hitgroupRecordBase = impl_->d_hit_record;
  impl_->sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
  impl_->sbt.hitgroupRecordCount = 2;
}

void OptixRibbonRenderer::set_primitives(const std::vector<RibbonPrimitive>& primitives) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");

  impl_->primitive_count = static_cast<unsigned>(primitives.size());
  if (impl_->primitive_count == 0) throw std::runtime_error("No primitives provided");

  impl_->release_geometry_buffers();
  impl_->primitive_count = static_cast<unsigned>(primitives.size());

  CU_CHECK(cuMemAlloc(&impl_->d_primitives, sizeof(RibbonPrimitive) * primitives.size()));
  CU_CHECK(cuMemcpyHtoD(
      impl_->d_primitives, primitives.data(), sizeof(RibbonPrimitive) * primitives.size()));

  std::vector<OptixAabb> aabbs(primitives.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    aabbs[i] = primitive_aabb(primitives[i]);
  }
  CU_CHECK(cuMemAlloc(&impl_->d_aabb, sizeof(OptixAabb) * aabbs.size()));
  CU_CHECK(cuMemcpyHtoD(impl_->d_aabb, aabbs.data(), sizeof(OptixAabb) * aabbs.size()));
  impl_->build_gas_from_current_aabbs(false);
}

void OptixRibbonRenderer::set_primitives_with_grass_wind_bounds(
    const std::vector<RibbonPrimitive>& primitives,
    float strength) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");

  impl_->primitive_count = static_cast<unsigned>(primitives.size());
  if (impl_->primitive_count == 0) throw std::runtime_error("No primitives provided");

  impl_->release_geometry_buffers();
  impl_->primitive_count = static_cast<unsigned>(primitives.size());

  CU_CHECK(cuMemAlloc(&impl_->d_primitives, sizeof(RibbonPrimitive) * primitives.size()));
  CU_CHECK(cuMemcpyHtoD(
      impl_->d_primitives, primitives.data(), sizeof(RibbonPrimitive) * primitives.size()));

  std::vector<OptixAabb> aabbs(primitives.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    aabbs[i] = grass_wind_swept_aabb(primitives[i], strength);
  }
  CU_CHECK(cuMemAlloc(&impl_->d_aabb, sizeof(OptixAabb) * aabbs.size()));
  CU_CHECK(cuMemcpyHtoD(impl_->d_aabb, aabbs.data(), sizeof(OptixAabb) * aabbs.size()));
  impl_->build_gas_from_current_aabbs(false);
}

void OptixRibbonRenderer::prepare_dynamic_primitives(const std::vector<RibbonPrimitive>& primitives) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (primitives.empty()) throw std::runtime_error("No primitives provided");

  impl_->release_geometry_buffers();
  impl_->primitive_count = static_cast<unsigned>(primitives.size());

  const size_t primitive_bytes = sizeof(RibbonPrimitive) * primitives.size();
  CU_CHECK(cuMemAlloc(&impl_->d_base_primitives, primitive_bytes));
  CU_CHECK(cuMemAlloc(&impl_->d_primitives, primitive_bytes));
  CU_CHECK(cuMemcpyHtoD(impl_->d_base_primitives, primitives.data(), primitive_bytes));
  CU_CHECK(cuMemcpyHtoD(impl_->d_primitives, primitives.data(), primitive_bytes));

  CU_CHECK(cuMemAlloc(&impl_->d_aabb, sizeof(OptixAabb) * primitives.size()));
  launch_compute_aabbs_kernel(
      reinterpret_cast<const RibbonPrimitive*>(impl_->d_primitives),
      reinterpret_cast<OptixAabb*>(impl_->d_aabb),
      impl_->primitive_count,
      reinterpret_cast<cudaStream_t>(impl_->stream));
  CUDA_CHECK(cudaGetLastError());
  CU_CHECK(cuStreamSynchronize(impl_->stream));
  impl_->build_gas_from_current_aabbs(true);
}

void OptixRibbonRenderer::update_dynamic_primitives_on_host(const std::vector<RibbonPrimitive>& primitives) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (!impl_->d_primitives || !impl_->d_aabb || !impl_->gas_allow_update) {
    throw std::runtime_error("Dynamic primitives were not prepared");
  }
  if (primitives.size() != impl_->primitive_count) {
    throw std::runtime_error("Dynamic primitive count changed");
  }

  const size_t primitive_bytes = sizeof(RibbonPrimitive) * primitives.size();
  CU_CHECK(cuMemcpyHtoD(impl_->d_primitives, primitives.data(), primitive_bytes));

  std::vector<OptixAabb> aabbs(primitives.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    aabbs[i] = primitive_aabb(primitives[i]);
  }
  CU_CHECK(cuMemcpyHtoD(impl_->d_aabb, aabbs.data(), sizeof(OptixAabb) * aabbs.size()));
  impl_->update_gas_from_current_aabbs();
}

void OptixRibbonRenderer::animate_primitives_on_gpu(DeviceWindMode mode, float time, float strength, float speed) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (!impl_->d_base_primitives || !impl_->d_primitives || !impl_->d_aabb) {
    throw std::runtime_error("Dynamic primitives were not prepared");
  }

  launch_animate_primitives_kernel(
      reinterpret_cast<const RibbonPrimitive*>(impl_->d_base_primitives),
      reinterpret_cast<RibbonPrimitive*>(impl_->d_primitives),
      impl_->primitive_count,
      mode,
      time,
      strength,
      speed,
      reinterpret_cast<cudaStream_t>(impl_->stream));
  launch_compute_aabbs_kernel(
      reinterpret_cast<const RibbonPrimitive*>(impl_->d_primitives),
      reinterpret_cast<OptixAabb*>(impl_->d_aabb),
      impl_->primitive_count,
      reinterpret_cast<cudaStream_t>(impl_->stream));
  CUDA_CHECK(cudaGetLastError());
  CU_CHECK(cuStreamSynchronize(impl_->stream));
}

void OptixRibbonRenderer::animate_primitives_on_gpu_without_aabb(
    DeviceWindMode mode, float time, float strength, float speed) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (!impl_->d_base_primitives || !impl_->d_primitives || !impl_->d_aabb) {
    throw std::runtime_error("Dynamic primitives were not prepared");
  }

  launch_animate_primitives_kernel(
      reinterpret_cast<const RibbonPrimitive*>(impl_->d_base_primitives),
      reinterpret_cast<RibbonPrimitive*>(impl_->d_primitives),
      impl_->primitive_count,
      mode,
      time,
      strength,
      speed,
      reinterpret_cast<cudaStream_t>(impl_->stream));
  CUDA_CHECK(cudaGetLastError());
  CU_CHECK(cuStreamSynchronize(impl_->stream));
}

void OptixRibbonRenderer::update_aabbs_on_gpu() {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (!impl_->d_primitives || !impl_->d_aabb) {
    throw std::runtime_error("Dynamic primitives were not prepared");
  }

  launch_compute_aabbs_kernel(
      reinterpret_cast<const RibbonPrimitive*>(impl_->d_primitives),
      reinterpret_cast<OptixAabb*>(impl_->d_aabb),
      impl_->primitive_count,
      reinterpret_cast<cudaStream_t>(impl_->stream));
  CUDA_CHECK(cudaGetLastError());
  CU_CHECK(cuStreamSynchronize(impl_->stream));
}

void OptixRibbonRenderer::rebuild_gas_from_device_primitives(bool force_rebuild, bool allow_update_after_rebuild) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (force_rebuild) {
    impl_->build_gas_from_current_aabbs(allow_update_after_rebuild);
  } else {
    impl_->update_gas_from_current_aabbs();
  }
}

void OptixRibbonRenderer::render_frame(unsigned width, unsigned height, std::vector<uchar4>& pixels) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (impl_->gas == 0) throw std::runtime_error("Geometry was not uploaded");
  if (width == 0 || height == 0) throw std::runtime_error("Invalid frame size");

  impl_->launch_frame(width, height, 0);
  download_frame(width, height, pixels);
}

void OptixRibbonRenderer::render_frame_device(unsigned width, unsigned height) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (impl_->gas == 0) throw std::runtime_error("Geometry was not uploaded");
  if (width == 0 || height == 0) throw std::runtime_error("Invalid frame size");

  impl_->launch_frame(width, height, 0);
}

void OptixRibbonRenderer::render_frame_to_device_buffer(
    unsigned width, unsigned height, CUdeviceptr d_output_pixels) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (impl_->gas == 0) throw std::runtime_error("Geometry was not uploaded");
  if (width == 0 || height == 0) throw std::runtime_error("Invalid frame size");
  if (!d_output_pixels) throw std::runtime_error("Invalid output device buffer");

  impl_->launch_frame(width, height, d_output_pixels);
}

void OptixRibbonRenderer::download_frame(unsigned width, unsigned height, std::vector<uchar4>& pixels) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (impl_->frame_width != width || impl_->frame_height != height || !impl_->d_frame_pixels) {
    throw std::runtime_error("No internal frame for the requested resolution");
  }
  pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  CU_CHECK(cuMemcpyDtoH(pixels.data(), impl_->d_frame_pixels, pixels.size() * sizeof(uchar4)));
}

void OptixRibbonRenderer::read_render_core_stats(RenderCoreStats& stats) const {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (!impl_->render_core_profiling || !impl_->d_render_stats) {
    throw std::runtime_error("Render core profiling is not enabled");
  }
  CU_CHECK(cuMemcpyDtoH(&stats, impl_->d_render_stats, sizeof(RenderCoreStats)));
}

void OptixRibbonRenderer::render_to_ppm(
    const std::string& output_path,
    unsigned width,
    unsigned height,
    const std::string& aov_dir,
    bool denoise) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (impl_->gas == 0) throw std::runtime_error("Geometry was not uploaded");

  CUdeviceptr d_pixels = 0;
  CUdeviceptr d_albedo = 0;
  CUdeviceptr d_normal = 0;
  CUdeviceptr d_depth = 0;
  CUdeviceptr d_beauty_hdr = 0;
  CUdeviceptr d_denoised_hdr = 0;
  CU_CHECK(cuMemAlloc(&d_pixels, width * height * sizeof(uchar4)));
  CU_CHECK(cuMemAlloc(&d_albedo, width * height * sizeof(float4)));
  CU_CHECK(cuMemAlloc(&d_normal, width * height * sizeof(float4)));
  CU_CHECK(cuMemAlloc(&d_depth, width * height * sizeof(float4)));
  CU_CHECK(cuMemAlloc(&d_beauty_hdr, width * height * sizeof(float4)));
  CU_CHECK(cuMemAlloc(&d_denoised_hdr, width * height * sizeof(float4)));

  LaunchParams lp = {};
  lp.image = reinterpret_cast<uchar4*>(d_pixels);
  lp.aov_albedo = reinterpret_cast<float4*>(d_albedo);
  lp.aov_normal = reinterpret_cast<float4*>(d_normal);
  lp.aov_depth = reinterpret_cast<float4*>(d_depth);
  lp.aov_beauty_hdr = reinterpret_cast<float4*>(d_beauty_hdr);
  lp.width = width;
  lp.height = height;
  lp.spp = impl_->spp;
  lp.max_depth = impl_->max_depth;
  lp.exposure = impl_->exposure;
  lp.gamma = impl_->gamma;
  lp.firefly_clamp = impl_->firefly_clamp;
  lp.gas = impl_->gas;
  lp.primitives = reinterpret_cast<RibbonPrimitive*>(impl_->d_primitives);
  lp.primitive_count = impl_->primitive_count;
  lp.light_count = static_cast<unsigned>(std::min<size_t>(impl_->lights.size(), 8));
  for (unsigned i = 0; i < lp.light_count; ++i) lp.lights[i] = impl_->lights[i];

  const float3 look_from = impl_->camera.look_from;
  const float3 look_at = impl_->camera.look_at;
  const float3 up = impl_->camera.up;
  const float3 f = normalize3(look_at - look_from);
  const float3 r = normalize3(make_float3(
      up.y * f.z - up.z * f.y,
      up.z * f.x - up.x * f.z,
      up.x * f.y - up.y * f.x));
  const float3 u = make_float3(
      f.y * r.z - f.z * r.y,
      f.z * r.x - f.x * r.z,
      f.x * r.y - f.y * r.x);
  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float tan_half_fov = std::tan(impl_->camera.fov_deg * 0.5f * 3.14159265f / 180.0f);
  lp.camera.origin = look_from;
  lp.camera.horizontal = r * (2.0f * tan_half_fov * aspect);
  lp.camera.vertical = u * (2.0f * tan_half_fov);
  lp.camera.lower_left = f - 0.5f * lp.camera.horizontal - 0.5f * lp.camera.vertical;

  if (!impl_->d_launch_params) {
    CU_CHECK(cuMemAlloc(&impl_->d_launch_params, sizeof(LaunchParams)));
  }
  CU_CHECK(cuMemcpyHtoD(impl_->d_launch_params, &lp, sizeof(lp)));

  OPTIX_CHECK(optixLaunch(
      impl_->pipeline,
      impl_->stream,
      impl_->d_launch_params,
      sizeof(LaunchParams),
      &impl_->sbt,
      width,
      height,
      1));
  CU_CHECK(cuStreamSynchronize(impl_->stream));

  std::vector<uchar4> pixels(width * height);
  std::vector<float4> albedo(width * height);
  std::vector<float4> normal(width * height);
  std::vector<float4> depth(width * height);
  std::vector<float4> beauty_hdr(width * height);
  std::vector<float4> denoised_hdr(width * height);
  CU_CHECK(cuMemcpyDtoH(pixels.data(), d_pixels, width * height * sizeof(uchar4)));
  CU_CHECK(cuMemcpyDtoH(albedo.data(), d_albedo, width * height * sizeof(float4)));
  CU_CHECK(cuMemcpyDtoH(normal.data(), d_normal, width * height * sizeof(float4)));
  CU_CHECK(cuMemcpyDtoH(depth.data(), d_depth, width * height * sizeof(float4)));
  CU_CHECK(cuMemcpyDtoH(beauty_hdr.data(), d_beauty_hdr, width * height * sizeof(float4)));

  if (denoise) {
    OptixDenoiserOptions denoiser_options = {};
    denoiser_options.guideAlbedo = 1;
    denoiser_options.guideNormal = 1;

    OptixDenoiser denoiser = nullptr;
    OPTIX_CHECK(optixDenoiserCreate(impl_->optix_ctx, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiser_options, &denoiser));

    OptixDenoiserSizes denoiser_sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiser_sizes));

    CUdeviceptr d_state = 0;
    CUdeviceptr d_scratch = 0;
    CUdeviceptr d_intensity = 0;
    CU_CHECK(cuMemAlloc(&d_state, denoiser_sizes.stateSizeInBytes));
    CU_CHECK(cuMemAlloc(&d_scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes));
    CU_CHECK(cuMemAlloc(&d_intensity, sizeof(float)));

    OPTIX_CHECK(optixDenoiserSetup(
        denoiser,
        impl_->stream,
        width,
        height,
        d_state,
        denoiser_sizes.stateSizeInBytes,
        d_scratch,
        denoiser_sizes.withoutOverlapScratchSizeInBytes));

    OptixImage2D input_layer = {};
    input_layer.data = d_beauty_hdr;
    input_layer.width = width;
    input_layer.height = height;
    input_layer.rowStrideInBytes = width * sizeof(float4);
    input_layer.pixelStrideInBytes = sizeof(float4);
    input_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D albedo_layer = {};
    albedo_layer.data = d_albedo;
    albedo_layer.width = width;
    albedo_layer.height = height;
    albedo_layer.rowStrideInBytes = width * sizeof(float4);
    albedo_layer.pixelStrideInBytes = sizeof(float4);
    albedo_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D normal_layer = {};
    normal_layer.data = d_normal;
    normal_layer.width = width;
    normal_layer.height = height;
    normal_layer.rowStrideInBytes = width * sizeof(float4);
    normal_layer.pixelStrideInBytes = sizeof(float4);
    normal_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D output_layer = {};
    output_layer.data = d_denoised_hdr;
    output_layer.width = width;
    output_layer.height = height;
    output_layer.rowStrideInBytes = width * sizeof(float4);
    output_layer.pixelStrideInBytes = sizeof(float4);
    output_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixDenoiserGuideLayer guide_layer = {};
    guide_layer.albedo = albedo_layer;
    guide_layer.normal = normal_layer;

    OptixDenoiserLayer denoiser_layer = {};
    denoiser_layer.input = input_layer;
    denoiser_layer.output = output_layer;

    OPTIX_CHECK(optixDenoiserComputeIntensity(
        denoiser, impl_->stream, &input_layer, d_intensity, d_scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes));

    OptixDenoiserParams denoiser_params = {};
    denoiser_params.hdrIntensity = d_intensity;
    denoiser_params.blendFactor = 0.0f;

    OPTIX_CHECK(optixDenoiserInvoke(
        denoiser,
        impl_->stream,
        &denoiser_params,
        d_state,
        denoiser_sizes.stateSizeInBytes,
        &guide_layer,
        &denoiser_layer,
        1,
        0,
        0,
        d_scratch,
        denoiser_sizes.withoutOverlapScratchSizeInBytes));
    CU_CHECK(cuStreamSynchronize(impl_->stream));
    CU_CHECK(cuMemcpyDtoH(denoised_hdr.data(), d_denoised_hdr, width * height * sizeof(float4)));

    CU_CHECK(cuMemFree(d_intensity));
    CU_CHECK(cuMemFree(d_scratch));
    CU_CHECK(cuMemFree(d_state));
    OPTIX_CHECK(optixDenoiserDestroy(denoiser));
  }
  CU_CHECK(cuMemFree(d_pixels));
  CU_CHECK(cuMemFree(d_albedo));
  CU_CHECK(cuMemFree(d_normal));
  CU_CHECK(cuMemFree(d_depth));
  CU_CHECK(cuMemFree(d_beauty_hdr));
  CU_CHECK(cuMemFree(d_denoised_hdr));

  if (rr::path_has_extension(output_path, ".exr")) {
    rr::save_rgb_hdr_exr(output_path, denoise ? denoised_hdr : beauty_hdr, width, height);
  } else {
    rr::save_rgb_ldr(output_path, denoise ? tonemap_to_ldr(denoised_hdr) : pixels, width, height);
  }
  if (!aov_dir.empty()) {
    std::filesystem::create_directories(aov_dir);
    rr::save_aov_from_float4_ppm((std::filesystem::path(aov_dir) / "albedo.ppm").string(), albedo, width, height, false, false);
    rr::save_aov_from_float4_ppm((std::filesystem::path(aov_dir) / "normal.ppm").string(), normal, width, height, true, false);
    rr::save_aov_from_float4_ppm((std::filesystem::path(aov_dir) / "depth.ppm").string(), depth, width, height, false, true);
    rr::save_rgb_hdr_exr((std::filesystem::path(aov_dir) / "beauty.exr").string(), beauty_hdr, width, height);
    rr::save_rgb_hdr_exr((std::filesystem::path(aov_dir) / "albedo.exr").string(), albedo, width, height);
    rr::save_rgb_hdr_exr((std::filesystem::path(aov_dir) / "normal.exr").string(), normal, width, height);
    rr::save_rgb_hdr_exr((std::filesystem::path(aov_dir) / "depth.exr").string(), depth, width, height);
    if (denoise) {
      rr::save_rgb_hdr_exr((std::filesystem::path(aov_dir) / "beauty_denoised.exr").string(), denoised_hdr, width, height);
      rr::save_rgb_ldr((std::filesystem::path(aov_dir) / "beauty_denoised.ppm").string(), tonemap_to_ldr(denoised_hdr), width, height);
    }
  }
}

void OptixRibbonRenderer::set_camera(const rr::CameraSettings& camera) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  impl_->camera = camera;
}

void OptixRibbonRenderer::set_lights(const std::vector<LightData>& lights) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  impl_->lights = lights;
}

void OptixRibbonRenderer::set_quality(
    unsigned spp,
    unsigned max_depth,
    float exposure,
    float gamma,
    float firefly_clamp) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  impl_->spp = std::max(1u, spp);
  impl_->max_depth = std::max(1u, max_depth);
  impl_->exposure = fmaxf(1e-4f, exposure);
  impl_->gamma = fmaxf(1e-4f, gamma);
  impl_->firefly_clamp = fmaxf(1e-4f, firefly_clamp);
}

void OptixRibbonRenderer::set_render_core_profiling(bool enabled) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  impl_->render_core_profiling = enabled;
}

void OptixRibbonRenderer::set_shader_wind(bool enabled, float time, float strength, float speed) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  impl_->shader_wind_enabled = enabled && std::fabs(strength) >= 1e-8f;
  impl_->shader_wind_mode = 0;
  impl_->shader_wind_time = time;
  impl_->shader_wind_strength = strength;
  impl_->shader_wind_speed = speed;
}

void OptixRibbonRenderer::get_gpu_memory_info(size_t& free_bytes, size_t& total_bytes) const {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  CU_CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
}

OptixRibbonRenderer::~OptixRibbonRenderer() {
  if (!impl_) return;
  impl_->release_frame_buffers();
  if (impl_->d_render_stats) cuMemFree(impl_->d_render_stats);
  if (impl_->d_launch_params) cuMemFree(impl_->d_launch_params);
  impl_->release_geometry_buffers();
  if (impl_->d_hit_record) cuMemFree(impl_->d_hit_record);
  if (impl_->d_miss_record) cuMemFree(impl_->d_miss_record);
  if (impl_->d_raygen_record) cuMemFree(impl_->d_raygen_record);
  if (impl_->pipeline) optixPipelineDestroy(impl_->pipeline);
  if (impl_->hit_shadow_pg) optixProgramGroupDestroy(impl_->hit_shadow_pg);
  if (impl_->hit_pg) optixProgramGroupDestroy(impl_->hit_pg);
  if (impl_->miss_shadow_pg) optixProgramGroupDestroy(impl_->miss_shadow_pg);
  if (impl_->miss_pg) optixProgramGroupDestroy(impl_->miss_pg);
  if (impl_->raygen_pg) optixProgramGroupDestroy(impl_->raygen_pg);
  if (impl_->module) optixModuleDestroy(impl_->module);
  if (impl_->optix_ctx) optixDeviceContextDestroy(impl_->optix_ctx);
  if (impl_->stream) cuStreamDestroy(impl_->stream);
  delete impl_;
  impl_ = nullptr;
}
