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
#include <stdexcept>
#include <string>
#include <vector>

#include "shared/launch_params.h"
#include "io/image_io.h"

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
      throw std::runtime_error(std::string("CUDA Driver API error at line ") + std::to_string(__LINE__)); \
  } while (0)

#define OPTIX_CHECK(call)                                                                          \
  do {                                                                                             \
    OptixResult rc = (call);                                                                       \
    if (rc != OPTIX_SUCCESS)                                                                       \
      throw std::runtime_error(std::string("OptiX error at line ") + std::to_string(__LINE__));  \
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
static float eval_quad(float w0, float w1, float w2, float u) { return w0 + w1 * u + w2 * u * u; }

static float3 eval_ribbon2(const Ribbon2Data& r, float u, float v) {
  const float omu = 1.0f - u;
  const float omv = 1.0f - v;
  return r.q00 * (omu * omv) + r.q01 * (omu * v) + r.q10 * (u * omv) + r.q11 * (u * v);
}

static float3 bezier2(const float3& b0, const float3& b1, const float3& b2, float u) {
  const float omu = 1.0f - u;
  return b0 * (omu * omu) + b1 * (2.0f * u * omu) + b2 * (u * u);
}

static float3 bitangent_ribbon3(const Ribbon3Data& r, float u) {
  const float3 c1 = 2.0f * (r.b1 - r.b0);
  const float3 c2 = r.b0 - 2.0f * r.b1 + r.b2;
  const float3 base = r.bt0 * (1.0f - u) + (r.alpha * r.bt1) * u;
  const float3 e = (r.beta * c2 + r.gamma * (c1 + c2 * u)) * (u * (1.0f - u));
  return base + e;
}

static OptixAabb primitive_aabb(const RibbonPrimitive& p) {
  float3 mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  auto expand = [&](const float3& q) {
    mn.x = fminf(mn.x, q.x);
    mn.y = fminf(mn.y, q.y);
    mn.z = fminf(mn.z, q.z);
    mx.x = fmaxf(mx.x, q.x);
    mx.y = fmaxf(mx.y, q.y);
    mx.z = fmaxf(mx.z, q.z);
  };

  constexpr int kSamples = 48;
  for (int i = 0; i <= kSamples; ++i) {
    float u = static_cast<float>(i) / static_cast<float>(kSamples);
    if (p.type == RIBBON_TYPE_2) {
      float w = std::fabs(eval_quad(p.r2.w0, p.r2.w1, p.r2.w2, u));
      expand(eval_ribbon2(p.r2, u, u - w));
      expand(eval_ribbon2(p.r2, u, u + w));
    } else {
      float w = std::fabs(eval_quad(p.r3.w0, p.r3.w1, p.r3.w2, u));
      float3 c = bezier2(p.r3.b0, p.r3.b1, p.r3.b2, u);
      float3 bn = normalize3(bitangent_ribbon3(p.r3, u));
      expand(c + w * bn);
      expand(c - w * bn);
    }
  }

  constexpr float pad = 1e-3f;
  return OptixAabb{mn.x - pad, mn.y - pad, mn.z - pad, mx.x + pad, mx.y + pad, mx.z + pad};
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
  CUdeviceptr d_aabb = 0;
  CUdeviceptr d_gas = 0;
  CUdeviceptr d_launch_params = 0;
  OptixTraversableHandle gas = 0;
  unsigned primitive_count = 0;
  std::vector<LightData> lights;
  rr::CameraSettings camera = {};
  unsigned spp = 8;
  unsigned max_depth = 4;
  float exposure = 1.0f;
  float gamma = 2.2f;
  float firefly_clamp = 10.0f;
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

  if (impl_->d_primitives) CU_CHECK(cuMemFree(impl_->d_primitives));
  if (impl_->d_aabb) CU_CHECK(cuMemFree(impl_->d_aabb));
  if (impl_->d_gas) CU_CHECK(cuMemFree(impl_->d_gas));

  CU_CHECK(cuMemAlloc(&impl_->d_primitives, sizeof(RibbonPrimitive) * primitives.size()));
  CU_CHECK(cuMemcpyHtoD(
      impl_->d_primitives, primitives.data(), sizeof(RibbonPrimitive) * primitives.size()));

  std::vector<OptixAabb> aabbs(primitives.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    aabbs[i] = primitive_aabb(primitives[i]);
  }
  CU_CHECK(cuMemAlloc(&impl_->d_aabb, sizeof(OptixAabb) * aabbs.size()));
  CU_CHECK(cuMemcpyHtoD(impl_->d_aabb, aabbs.data(), sizeof(OptixAabb) * aabbs.size()));

  OptixBuildInput input = {};
  input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  input.customPrimitiveArray.aabbBuffers = &impl_->d_aabb;
  input.customPrimitiveArray.numPrimitives = impl_->primitive_count;
  static const uint32_t flag = OPTIX_GEOMETRY_FLAG_NONE;
  input.customPrimitiveArray.flags = &flag;
  input.customPrimitiveArray.numSbtRecords = 1;

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes sizes = {};
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
      impl_->optix_ctx, &accel_options, &input, 1, &sizes));

  CUdeviceptr d_temp = 0;
  CU_CHECK(cuMemAlloc(&d_temp, sizes.tempSizeInBytes));
  CU_CHECK(cuMemAlloc(&impl_->d_gas, sizes.outputSizeInBytes));

  OPTIX_CHECK(optixAccelBuild(
      impl_->optix_ctx,
      impl_->stream,
      &accel_options,
      &input,
      1,
      d_temp,
      sizes.tempSizeInBytes,
      impl_->d_gas,
      sizes.outputSizeInBytes,
      &impl_->gas,
      nullptr,
      0));
  CU_CHECK(cuMemFree(d_temp));
  CU_CHECK(cuStreamSynchronize(impl_->stream));
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

OptixRibbonRenderer::~OptixRibbonRenderer() {
  if (!impl_) return;
  if (impl_->d_launch_params) cuMemFree(impl_->d_launch_params);
  if (impl_->d_primitives) cuMemFree(impl_->d_primitives);
  if (impl_->d_aabb) cuMemFree(impl_->d_aabb);
  if (impl_->d_gas) cuMemFree(impl_->d_gas);
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
