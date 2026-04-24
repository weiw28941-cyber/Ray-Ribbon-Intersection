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
#include <stdexcept>
#include <string>
#include <vector>

#include "shared/launch_params.h"

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

struct OptixRibbonRenderer::Impl {
  CUcontext cu_ctx = nullptr;
  CUstream stream = nullptr;
  OptixDeviceContext optix_ctx = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  OptixProgramGroup raygen_pg = nullptr;
  OptixProgramGroup miss_pg = nullptr;
  OptixProgramGroup hit_pg = nullptr;
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
  pco.numPayloadValues = 1;
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

  OptixProgramGroupDesc hg_desc = {};
  hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hg_desc.hitgroup.moduleCH = impl_->module;
  hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  hg_desc.hitgroup.moduleIS = impl_->module;
  hg_desc.hitgroup.entryFunctionNameIS = "__intersection__ribbon";
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(
      impl_->optix_ctx, &hg_desc, 1, &pgo, log, &log_size, &impl_->hit_pg));

  std::vector<OptixProgramGroup> pgs = {impl_->raygen_pg, impl_->miss_pg, impl_->hit_pg};
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
  CU_CHECK(cuMemAlloc(&impl_->d_miss_record, sizeof(ms)));
  CU_CHECK(cuMemcpyHtoD(impl_->d_miss_record, &ms, sizeof(ms)));

  SbtRecord<HitData> hg = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(impl_->hit_pg, &hg));
  CU_CHECK(cuMemAlloc(&impl_->d_hit_record, sizeof(hg)));
  CU_CHECK(cuMemcpyHtoD(impl_->d_hit_record, &hg, sizeof(hg)));

  impl_->sbt.raygenRecord = impl_->d_raygen_record;
  impl_->sbt.missRecordBase = impl_->d_miss_record;
  impl_->sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
  impl_->sbt.missRecordCount = 1;
  impl_->sbt.hitgroupRecordBase = impl_->d_hit_record;
  impl_->sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
  impl_->sbt.hitgroupRecordCount = 1;
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

void OptixRibbonRenderer::render_to_ppm(const std::string& output_path, unsigned width, unsigned height) {
  if (!impl_) throw std::runtime_error("Renderer is not initialized");
  if (impl_->gas == 0) throw std::runtime_error("Geometry was not uploaded");

  CUdeviceptr d_pixels = 0;
  CU_CHECK(cuMemAlloc(&d_pixels, width * height * sizeof(uchar4)));

  LaunchParams lp = {};
  lp.image = reinterpret_cast<uchar4*>(d_pixels);
  lp.width = width;
  lp.height = height;
  lp.gas = impl_->gas;
  lp.primitives = reinterpret_cast<RibbonPrimitive*>(impl_->d_primitives);
  lp.primitive_count = impl_->primitive_count;

  const float3 look_from = make_float3(0.0f, 0.8f, 2.0f);
  const float3 look_at = make_float3(0.0f, 0.0f, -4.0f);
  const float3 up = make_float3(0.0f, 1.0f, 0.0f);
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
  const float tan_half_fov = std::tan(35.0f * 0.5f * 3.14159265f / 180.0f);
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
  CU_CHECK(cuMemcpyDtoH(pixels.data(), d_pixels, width * height * sizeof(uchar4)));
  CU_CHECK(cuMemFree(d_pixels));

  std::ofstream out(output_path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open output image: " + output_path);
  out << "P6\n" << width << " " << height << "\n255\n";
  for (const auto& p : pixels) {
    out.put(static_cast<char>(p.x));
    out.put(static_cast<char>(p.y));
    out.put(static_cast<char>(p.z));
  }
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
  if (impl_->hit_pg) optixProgramGroupDestroy(impl_->hit_pg);
  if (impl_->miss_pg) optixProgramGroupDestroy(impl_->miss_pg);
  if (impl_->raygen_pg) optixProgramGroupDestroy(impl_->raygen_pg);
  if (impl_->module) optixModuleDestroy(impl_->module);
  if (impl_->optix_ctx) optixDeviceContextDestroy(impl_->optix_ctx);
  if (impl_->stream) cuStreamDestroy(impl_->stream);
  delete impl_;
  impl_ = nullptr;
}
