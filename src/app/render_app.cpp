#include "app/render_app.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/log.h"
#include "optix_renderer.h"
#include "app/wind_animation.h"

namespace rr {
namespace {

static DeviceWindMode parse_device_wind_mode(const std::string& wind_mode) {
  if (wind_mode == "strip-independent") return DeviceWindMode::StripIndependent;
  if (wind_mode == "constrained") return DeviceWindMode::StripConstrained;
  return DeviceWindMode::Grass;
}

static void apply_host_wind_mode(
    const AppOptions& options,
    const std::vector<RibbonPrimitive>& base_primitives,
    std::vector<RibbonPrimitive>& animated_primitives,
    float time) {
  if (options.wind_mode == "constrained") {
    apply_constrained_strip_wind_animation(
        base_primitives, animated_primitives, time, options.wind_strength, options.wind_speed);
  } else if (options.wind_mode == "strip-independent") {
    apply_independent_strip_wind_animation(
        base_primitives, animated_primitives, time, options.wind_strength, options.wind_speed);
  } else if (options.wind_mode == "grass-width") {
    apply_width_dynamic_grass_wind_animation(
        base_primitives, animated_primitives, time, options.wind_strength, options.wind_speed);
  } else if (options.wind_mode == "grass-rebuild") {
    apply_rebuilt_grass_wind_animation(
        base_primitives, animated_primitives, time, options.wind_strength, options.wind_speed);
  } else {
    apply_wind_animation(base_primitives, animated_primitives, time, options.wind_strength, options.wind_speed);
  }
}

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct TimingStats {
  double avg = 0.0;
  double avg_drop_max = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double max = 0.0;
};

static TimingStats make_stats(std::vector<double> values) {
  if (values.empty()) return {};
  std::sort(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double avg = sum / static_cast<double>(values.size());
  const double avg_drop_max = values.size() > 1
      ? (sum - values.back()) / static_cast<double>(values.size() - 1)
      : avg;
  const size_t median_index = values.size() / 2;
  const size_t p95_index = std::min(values.size() - 1, static_cast<size_t>(values.size() * 95 / 100));
  return {avg, avg_drop_max, values[median_index], values[p95_index], values.back()};
}

static double mib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static void add_core_stats(RenderCoreStats& dst, const RenderCoreStats& src) {
  dst.primary_trace_cycles += src.primary_trace_cycles;
  dst.primary_trace_count += src.primary_trace_count;
  dst.shadow_trace_cycles += src.shadow_trace_cycles;
  dst.shadow_trace_count += src.shadow_trace_count;
  dst.closesthit_cycles += src.closesthit_cycles;
  dst.closesthit_count += src.closesthit_count;
  dst.intersection_cycles_r2 += src.intersection_cycles_r2;
  dst.intersection_calls_r2 += src.intersection_calls_r2;
  dst.solver_cycles_r2 += src.solver_cycles_r2;
  dst.solver_calls_r2 += src.solver_calls_r2;
  dst.intersection_cycles_r3 += src.intersection_cycles_r3;
  dst.intersection_calls_r3 += src.intersection_calls_r3;
  dst.solver_cycles_r3 += src.solver_cycles_r3;
  dst.solver_calls_r3 += src.solver_calls_r3;
  dst.intersection_cycles_r4a += src.intersection_cycles_r4a;
  dst.intersection_calls_r4a += src.intersection_calls_r4a;
  dst.solver_cycles_r4a += src.solver_cycles_r4a;
  dst.solver_calls_r4a += src.solver_calls_r4a;
  dst.intersection_cycles_r4b += src.intersection_cycles_r4b;
  dst.intersection_calls_r4b += src.intersection_calls_r4b;
  dst.solver_cycles_r4b += src.solver_cycles_r4b;
  dst.solver_calls_r4b += src.solver_calls_r4b;
  dst.intersection_cycles_r4c += src.intersection_cycles_r4c;
  dst.intersection_calls_r4c += src.intersection_calls_r4c;
  dst.solver_cycles_r4c += src.solver_cycles_r4c;
  dst.solver_calls_r4c += src.solver_calls_r4c;
}

static double cycles_to_ms(unsigned long long cycles, double cycles_per_ms) {
  return cycles_per_ms > 0.0 ? static_cast<double>(cycles) / cycles_per_ms : 0.0;
}

static double avg_cycles(unsigned long long cycles, unsigned long long count) {
  return count > 0 ? static_cast<double>(cycles) / static_cast<double>(count) : 0.0;
}

static double avg_ns(unsigned long long cycles, unsigned long long count, double cycles_per_ms) {
  return cycles_per_ms > 0.0 ? avg_cycles(cycles, count) * 1000000.0 / cycles_per_ms : 0.0;
}

static int run_aabb_benchmark(const AppOptions& options, Scene& scene) {
  const unsigned frames = std::max(1u, options.benchmark_aabb_frames);
  const unsigned warmup = std::min(20u, frames / 5u);

  OptixRibbonRenderer renderer;
  renderer.initialize(options.ptx_path);
  renderer.prepare_dynamic_primitives(scene.primitives);

  std::vector<double> update_ms;
  std::vector<double> aabb_ms;
  update_ms.reserve(frames);
  aabb_ms.reserve(frames);

  for (unsigned frame = 0; frame < frames + warmup; ++frame) {
    const float time = static_cast<float>(frame) / 60.0f;

    const double update_start = now_ms();
    renderer.animate_primitives_on_gpu_without_aabb(
        parse_device_wind_mode(options.wind_mode), time, options.wind_strength, options.wind_speed);
    const double update_end = now_ms();

    const double aabb_start = now_ms();
    renderer.update_aabbs_on_gpu();
    const double aabb_end = now_ms();

    if (frame >= warmup) {
      update_ms.push_back(update_end - update_start);
      aabb_ms.push_back(aabb_end - aabb_start);
    }
  }

  const TimingStats update = make_stats(update_ms);
  const TimingStats aabb = make_stats(aabb_ms);
  const size_t primitive_bytes = scene.primitives.size() * sizeof(RibbonPrimitive);
  const size_t aabb_bytes = scene.primitives.size() * sizeof(float) * 6u;

  std::cout << std::fixed << std::setprecision(6)
            << "AABB benchmark\n"
            << "  scene: " << (options.scene_path.empty() ? "built-in" : options.scene_path) << "\n"
            << "  primitives: " << scene.primitives.size() << "\n"
            << "  frames: " << frames << " (warmup " << warmup << ")\n"
            << "  wind: strength " << options.wind_strength
            << ", speed " << options.wind_speed
            << ", mode " << options.wind_mode
            << ", backend gpu, gas none, render none\n\n"
            << "Timing ms/frame:\n"
            << "  update_gpu      avg " << update.avg
            << "  drop_max " << update.avg_drop_max
            << "  median " << update.median
            << "  p95 " << update.p95
            << "  max " << update.max << "\n"
            << "  aabb_gpu        avg " << aabb.avg
            << "  drop_max " << aabb.avg_drop_max
            << "  median " << aabb.median
            << "  p95 " << aabb.p95
            << "  max " << aabb.max << "\n\n"
            << "Memory MiB:\n"
            << "  ribbon_buffer   " << mib(primitive_bytes) << "\n"
            << "  aabb_buffer     " << mib(aabb_bytes) << "\n";
  return 0;
}

static int run_wind_benchmark(const AppOptions& options, Scene& scene) {
  const unsigned frames = std::max(1u, options.benchmark_wind_frames);
  const unsigned warmup = std::min(10u, frames / 4u);

  OptixRibbonRenderer renderer;
  renderer.initialize(options.ptx_path);
  renderer.set_camera(scene.camera);
  renderer.set_lights(scene.lights);
  renderer.set_quality(
      scene.render.spp,
      scene.render.max_depth,
      scene.render.exposure,
      scene.render.gamma,
      scene.render.firefly_clamp);
  renderer.set_render_core_profiling(options.wind_core_profile);

  int device = 0;
  cudaDeviceProp prop = {};
  if (options.wind_core_profile) {
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
  }
  const double cycles_per_ms = static_cast<double>(prop.clockRate);

  size_t free_before = 0;
  size_t total_before = 0;
  renderer.get_gpu_memory_info(free_before, total_before);

  std::vector<RibbonPrimitive> base_primitives = scene.primitives;
  std::vector<RibbonPrimitive> animated_primitives = base_primitives;
  const bool host_special_mode = options.wind_mode == "grass-rebuild" || options.wind_mode == "grass-width";
  const bool gpu_backend = options.wind_backend == "gpu" && !host_special_mode;
  const bool shader_backend = options.wind_backend == "shader" && !host_special_mode;
  const bool hybrid_gas_rebuild = gpu_backend && options.gas_rebuild_interval > 0;
  const bool static_quality_rebuild = hybrid_gas_rebuild && options.gas_rebuild_static_quality;
  if (shader_backend) {
    renderer.set_primitives_with_grass_wind_bounds(base_primitives, options.wind_strength);
  } else {
    renderer.prepare_dynamic_primitives(base_primitives);
  }
  std::vector<double> update_ms;
  std::vector<double> aabb_ms;
  std::vector<double> gas_ms;
  std::vector<double> render_ms;
  std::vector<double> frame_ms;
  std::vector<float> smoothness;
  std::vector<float> normal_avg_deg;
  std::vector<float> normal_p95_deg;
  std::vector<float> normal_max_deg;
  std::vector<float> joint_gap_avg;
  std::vector<float> joint_gap_p95;
  std::vector<float> joint_gap_max;
  std::vector<float> joint_normal_avg_deg;
  std::vector<float> joint_normal_p95_deg;
  std::vector<float> joint_normal_max_deg;
  std::vector<float> freedom_loss_norm;
  std::vector<float> freedom_variance_retention;
  std::vector<float> freedom_peak_response;
  size_t min_free = free_before;
  size_t total_mem = total_before;
  RenderCoreStats total_core = {};

  update_ms.reserve(frames);
  aabb_ms.reserve(frames);
  gas_ms.reserve(frames);
  render_ms.reserve(frames);
  frame_ms.reserve(frames);
  smoothness.reserve(frames);
  normal_avg_deg.reserve(frames);
  normal_p95_deg.reserve(frames);
  normal_max_deg.reserve(frames);
  joint_gap_avg.reserve(frames);
  joint_gap_p95.reserve(frames);
  joint_gap_max.reserve(frames);
  joint_normal_avg_deg.reserve(frames);
  joint_normal_p95_deg.reserve(frames);
  joint_normal_max_deg.reserve(frames);
  freedom_loss_norm.reserve(frames);
  freedom_variance_retention.reserve(frames);
  freedom_peak_response.reserve(frames);

  for (unsigned frame = 0; frame < frames + warmup; ++frame) {
    const float time = static_cast<float>(frame) / 60.0f;
    const double frame_start = now_ms();

    const double update_start = now_ms();
    if (shader_backend) {
      renderer.set_shader_wind(true, time, options.wind_strength, options.wind_speed);
    } else if (gpu_backend) {
      renderer.animate_primitives_on_gpu_without_aabb(
          parse_device_wind_mode(options.wind_mode), time, options.wind_strength, options.wind_speed);
    } else {
      apply_host_wind_mode(options, base_primitives, animated_primitives, time);
    }
    const double update_end = now_ms();

    const double aabb_start = now_ms();
    if (gpu_backend) {
      renderer.update_aabbs_on_gpu();
    }
    const double aabb_end = now_ms();

    const double gas_start = now_ms();
    if (shader_backend) {
      // Static GAS path: no per-frame GAS maintenance.
    } else if (gpu_backend) {
      const bool force_rebuild =
          hybrid_gas_rebuild && frame > 0 && (frame % options.gas_rebuild_interval) == 0;
      const bool allow_update_after_rebuild = !(static_quality_rebuild && options.gas_rebuild_interval == 1);
      renderer.rebuild_gas_from_device_primitives(force_rebuild, allow_update_after_rebuild);
    } else if (options.gas_rebuild_static_quality) {
      renderer.set_primitives(animated_primitives);
    } else {
      renderer.update_dynamic_primitives_on_host(animated_primitives);
    }
    const double gas_end = now_ms();

    const double render_start = now_ms();
    renderer.render_frame_device(scene.render.width, scene.render.height);
    const double render_end = now_ms();

    RenderCoreStats frame_core = {};
    if (options.wind_core_profile) {
      renderer.read_render_core_stats(frame_core);
    }

    size_t free_now = 0;
    size_t total_now = 0;
    renderer.get_gpu_memory_info(free_now, total_now);
    min_free = std::min(min_free, free_now);
    total_mem = total_now;

    if (frame >= warmup) {
      update_ms.push_back(update_end - update_start);
      aabb_ms.push_back(aabb_end - aabb_start);
      gas_ms.push_back(gas_end - gas_start);
      render_ms.push_back(render_end - render_start);
      frame_ms.push_back(render_end - frame_start);

      const bool collect_quality_metrics = frame + 1 == frames + warmup;
      if (collect_quality_metrics) {
        if (gpu_backend || shader_backend) {
          apply_host_wind_mode(options, base_primitives, animated_primitives, time);
        }
        smoothness.push_back(wind_smoothness_metric(base_primitives, animated_primitives, options.wind_strength));
        const NormalContinuityStats normal_stats = wind_normal_continuity_metric(animated_primitives);
        normal_avg_deg.push_back(normal_stats.avg_deg);
        normal_p95_deg.push_back(normal_stats.p95_deg);
        normal_max_deg.push_back(normal_stats.max_deg);
        const JointContinuityStats joint_stats = strip_joint_continuity_metric(base_primitives, animated_primitives);
        joint_gap_avg.push_back(joint_stats.position_avg);
        joint_gap_p95.push_back(joint_stats.position_p95);
        joint_gap_max.push_back(joint_stats.position_max);
        joint_normal_avg_deg.push_back(joint_stats.normal_avg_deg);
        joint_normal_p95_deg.push_back(joint_stats.normal_p95_deg);
        joint_normal_max_deg.push_back(joint_stats.normal_max_deg);
        if (options.wind_mode == "constrained") {
          std::vector<RibbonPrimitive> free_reference;
          apply_independent_strip_wind_animation(
              base_primitives, free_reference, time, options.wind_strength, options.wind_speed);
          const MotionFreedomStats freedom_stats =
              motion_freedom_metric(base_primitives, free_reference, animated_primitives);
          freedom_loss_norm.push_back(freedom_stats.loss_norm);
          freedom_variance_retention.push_back(freedom_stats.variance_retention);
          freedom_peak_response.push_back(freedom_stats.peak_response_ratio);
        }
      }
      if (options.wind_core_profile) {
        add_core_stats(total_core, frame_core);
      }
    }
  }

  size_t free_after = 0;
  size_t total_after = 0;
  renderer.get_gpu_memory_info(free_after, total_after);

  const TimingStats update = make_stats(update_ms);
  const TimingStats aabb = make_stats(aabb_ms);
  const TimingStats gas = make_stats(gas_ms);
  const TimingStats render = make_stats(render_ms);
  const TimingStats frame = make_stats(frame_ms);
  const double smooth_avg = smoothness.empty()
      ? 0.0
      : std::accumulate(smoothness.begin(), smoothness.end(), 0.0) / static_cast<double>(smoothness.size());
  const double smooth_max = smoothness.empty() ? 0.0 : *std::max_element(smoothness.begin(), smoothness.end());
  const double normal_avg = normal_avg_deg.empty()
      ? 0.0
      : std::accumulate(normal_avg_deg.begin(), normal_avg_deg.end(), 0.0) /
            static_cast<double>(normal_avg_deg.size());
  const double normal_p95 = normal_p95_deg.empty()
      ? 0.0
      : std::accumulate(normal_p95_deg.begin(), normal_p95_deg.end(), 0.0) /
            static_cast<double>(normal_p95_deg.size());
  const double normal_max = normal_max_deg.empty() ? 0.0 : *std::max_element(normal_max_deg.begin(), normal_max_deg.end());
  const double joint_gap_avg_value = joint_gap_avg.empty()
      ? 0.0
      : std::accumulate(joint_gap_avg.begin(), joint_gap_avg.end(), 0.0) / static_cast<double>(joint_gap_avg.size());
  const double joint_gap_p95_value = joint_gap_p95.empty()
      ? 0.0
      : std::accumulate(joint_gap_p95.begin(), joint_gap_p95.end(), 0.0) / static_cast<double>(joint_gap_p95.size());
  const double joint_gap_max_value = joint_gap_max.empty() ? 0.0 : *std::max_element(joint_gap_max.begin(), joint_gap_max.end());
  const double joint_normal_avg = joint_normal_avg_deg.empty()
      ? 0.0
      : std::accumulate(joint_normal_avg_deg.begin(), joint_normal_avg_deg.end(), 0.0) /
            static_cast<double>(joint_normal_avg_deg.size());
  const double joint_normal_p95 = joint_normal_p95_deg.empty()
      ? 0.0
      : std::accumulate(joint_normal_p95_deg.begin(), joint_normal_p95_deg.end(), 0.0) /
            static_cast<double>(joint_normal_p95_deg.size());
  const double joint_normal_max = joint_normal_max_deg.empty()
      ? 0.0
      : *std::max_element(joint_normal_max_deg.begin(), joint_normal_max_deg.end());
  const double freedom_loss = freedom_loss_norm.empty()
      ? 0.0
      : std::accumulate(freedom_loss_norm.begin(), freedom_loss_norm.end(), 0.0) /
            static_cast<double>(freedom_loss_norm.size());
  const double freedom_variance = freedom_variance_retention.empty()
      ? 0.0
      : std::accumulate(freedom_variance_retention.begin(), freedom_variance_retention.end(), 0.0) /
            static_cast<double>(freedom_variance_retention.size());
  const double freedom_peak = freedom_peak_response.empty()
      ? 0.0
      : std::accumulate(freedom_peak_response.begin(), freedom_peak_response.end(), 0.0) /
            static_cast<double>(freedom_peak_response.size());

  const size_t primitive_bytes = scene.primitives.size() * sizeof(RibbonPrimitive);
  const size_t aabb_bytes = scene.primitives.size() * sizeof(float) * 6u;
  const size_t framebuffer_bytes =
      static_cast<size_t>(scene.render.width) * static_cast<size_t>(scene.render.height) *
      (sizeof(uchar4) + 4u * sizeof(float4));
  const double peak_used = mib(total_mem - min_free);
  const double initial_used = mib(total_before - free_before);
  const double final_used = mib(total_after - free_after);

  std::cout << std::fixed << std::setprecision(3)
            << "Wind benchmark\n"
            << "  scene: " << (options.scene_path.empty() ? "built-in" : options.scene_path) << "\n"
            << "  primitives: " << scene.primitives.size() << "\n"
            << "  frames: " << frames << " (warmup " << warmup << ")\n"
            << "  resolution: " << scene.render.width << "x" << scene.render.height
            << ", spp " << scene.render.spp << ", depth " << scene.render.max_depth << "\n"
            << "  wind: strength " << options.wind_strength << ", speed " << options.wind_speed
            << ", mode " << options.wind_mode << ", backend " << (host_special_mode ? "cpu-special" : options.wind_backend)
            << ", gas "
            << (hybrid_gas_rebuild
                    ? "hybrid-rebuild"
                    : (shader_backend
                           ? "static-swept"
                           : (gpu_backend ? "update" : (options.gas_rebuild_static_quality ? "host-full-build" : "host-upload"))))
            << ", rebuild_interval " << options.gas_rebuild_interval
            << ", static_quality " << (static_quality_rebuild ? "yes" : "no") << "\n\n"
            << "Timing ms/frame:\n"
            << "  update_" << (shader_backend ? "shader" : (gpu_backend ? "gpu" : "cpu")) << "      avg " << update.avg
            << "  p95 " << update.p95 << "  max " << update.max << "\n"
            << "  " << (shader_backend ? "aabb_static" : (gpu_backend ? "aabb_gpu" : "aabb_cpu")) << "        avg " << aabb.avg
            << "  p95 " << aabb.p95 << "  max " << aabb.max << "\n"
            << "  " << (shader_backend ? "gas_static" : (gpu_backend ? "gas_update" : "upload_update")) << "    avg " << gas.avg
            << "  p95 " << gas.p95 << "  max " << gas.max << "\n"
            << "  render_launch  avg " << render.avg << "  p95 " << render.p95 << "  max " << render.max << "\n"
            << "  full_frame      avg " << frame.avg << "  p95 " << frame.p95 << "  max " << frame.max << "\n\n"
            << "Memory MiB:\n"
            << "  initial_used    " << initial_used << "\n"
            << "  peak_used       " << peak_used << "\n"
            << "  final_used      " << final_used << "\n"
            << "  ribbon_buffer   " << mib(primitive_bytes) << "\n"
            << "  aabb_buffer     " << mib(aabb_bytes) << "\n"
            << "  frame_buffers   " << mib(framebuffer_bytes) << "\n\n"
            << "Smoothness:\n"
            << "  avg " << smooth_avg << "  max " << smooth_max
            << "  (neighbor tip-displacement discontinuity / wind strength; lower is smoother)\n\n"
            << "Normal continuity deg:\n"
            << "  avg " << normal_avg << "  p95 " << normal_p95 << "  max " << normal_max
            << "  (nearest-neighbor unsigned normal angle at u=0.7; lower is smoother)\n";
  std::cout << "\nStrip joint continuity:\n"
            << "  position gap avg " << joint_gap_avg_value << "  p95 " << joint_gap_p95_value
            << "  max " << joint_gap_max_value << "\n"
            << "  normal deg   avg " << joint_normal_avg << "  p95 " << joint_normal_p95
            << "  max " << joint_normal_max
            << "  (consecutive connected segment joints; lower is smoother)\n";
  if (options.wind_mode == "constrained") {
    std::cout << "\nMotion freedom vs independent dynamic:\n"
              << "  MFL_norm " << freedom_loss
              << "  MVR " << freedom_variance
              << "  LRA " << freedom_peak
              << "  (lower MFL, MVR/LRA closer to 1 preserve more free motion)\n";
  }

  if (options.wind_core_profile) {
    const double frame_count = static_cast<double>(frames);
    const unsigned long long trace_cycles = total_core.primary_trace_cycles + total_core.shadow_trace_cycles;
    const unsigned long long trace_count = total_core.primary_trace_count + total_core.shadow_trace_count;
    const unsigned long long intersection_cycles = total_core.intersection_cycles_r2 +
        total_core.intersection_cycles_r3 + total_core.intersection_cycles_r4a + total_core.intersection_cycles_r4b +
        total_core.intersection_cycles_r4c;
    const unsigned long long intersection_calls = total_core.intersection_calls_r2 +
        total_core.intersection_calls_r3 + total_core.intersection_calls_r4a + total_core.intersection_calls_r4b +
        total_core.intersection_calls_r4c;
    const unsigned long long solver_cycles = total_core.solver_cycles_r2 + total_core.solver_cycles_r3 +
        total_core.solver_cycles_r4a + total_core.solver_cycles_r4b + total_core.solver_cycles_r4c;
    const unsigned long long solver_calls = total_core.solver_calls_r2 + total_core.solver_calls_r3 +
        total_core.solver_calls_r4a + total_core.solver_calls_r4b + total_core.solver_calls_r4c;
    const unsigned long long visible_cycles = intersection_cycles + total_core.closesthit_cycles;
    const unsigned long long traversal_est_cycles =
        trace_cycles > visible_cycles ? trace_cycles - visible_cycles : 0ull;
    const double trace_cycles_d = std::max(1.0, static_cast<double>(trace_cycles));

    std::cout << "\nWind render core profile per frame, aggregate SM time:\n"
              << "  all trace        " << cycles_to_ms(trace_cycles, cycles_per_ms) / frame_count << " ms"
              << "  calls/frame " << static_cast<double>(trace_count) / frame_count
              << "  avg/call " << avg_ns(trace_cycles, trace_count, cycles_per_ms) << " ns\n"
              << "  traversal_est    " << cycles_to_ms(traversal_est_cycles, cycles_per_ms) / frame_count
              << " ms  share " << (100.0 * static_cast<double>(traversal_est_cycles) / trace_cycles_d) << "%\n"
              << "  intersection     " << cycles_to_ms(intersection_cycles, cycles_per_ms) / frame_count
              << " ms  share " << (100.0 * static_cast<double>(intersection_cycles) / trace_cycles_d)
              << "%  calls/frame " << static_cast<double>(intersection_calls) / frame_count
              << "  avg/call " << avg_ns(intersection_cycles, intersection_calls, cycles_per_ms) << " ns\n"
              << "  solver           " << cycles_to_ms(solver_cycles, cycles_per_ms) / frame_count
              << " ms  share " << (100.0 * static_cast<double>(solver_cycles) / trace_cycles_d)
              << "%  calls/frame " << static_cast<double>(solver_calls) / frame_count
              << "  avg/call " << avg_ns(solver_cycles, solver_calls, cycles_per_ms) << " ns\n"
              << "  closesthit       " << cycles_to_ms(total_core.closesthit_cycles, cycles_per_ms) / frame_count
              << " ms  share " << (100.0 * static_cast<double>(total_core.closesthit_cycles) / trace_cycles_d)
              << "%  calls/frame " << static_cast<double>(total_core.closesthit_count) / frame_count
              << "  avg/call " << avg_ns(total_core.closesthit_cycles, total_core.closesthit_count, cycles_per_ms)
              << " ns\n"
              << "  r2 calls/frame   " << static_cast<double>(total_core.intersection_calls_r2) / frame_count
              << "  r3 calls/frame " << static_cast<double>(total_core.intersection_calls_r3) / frame_count
              << "  r4a calls/frame " << static_cast<double>(total_core.intersection_calls_r4a) / frame_count
              << "  r4b calls/frame " << static_cast<double>(total_core.intersection_calls_r4b) / frame_count
              << "  r4c calls/frame " << static_cast<double>(total_core.intersection_calls_r4c) / frame_count
              << "\n";
  }

  return 0;
}

static int run_core_benchmark(const AppOptions& options, Scene& scene) {
  const unsigned frames = std::max(1u, options.benchmark_core_frames);
  const unsigned warmup = std::min(10u, frames / 4u);

  OptixRibbonRenderer renderer;
  renderer.initialize(options.ptx_path);
  renderer.set_camera(scene.camera);
  renderer.set_lights(scene.lights);
  renderer.set_quality(
      scene.render.spp,
      scene.render.max_depth,
      scene.render.exposure,
      scene.render.gamma,
      scene.render.firefly_clamp);
  renderer.set_primitives(scene.primitives);
  renderer.set_render_core_profiling(true);

  int device = 0;
  cudaDeviceProp prop = {};
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  const double cycles_per_ms = static_cast<double>(prop.clockRate);

  std::vector<double> render_ms;
  render_ms.reserve(frames);
  RenderCoreStats total = {};

  for (unsigned frame = 0; frame < frames + warmup; ++frame) {
    const double render_start = now_ms();
    renderer.render_frame_device(scene.render.width, scene.render.height);
    const double render_end = now_ms();

    RenderCoreStats stats = {};
    renderer.read_render_core_stats(stats);
    if (frame >= warmup) {
      render_ms.push_back(render_end - render_start);
      add_core_stats(total, stats);
    }
  }

  const TimingStats render = make_stats(render_ms);
  const double frame_count = static_cast<double>(frames);
  const unsigned long long trace_cycles = total.primary_trace_cycles + total.shadow_trace_cycles;
  const unsigned long long trace_count = total.primary_trace_count + total.shadow_trace_count;
  const unsigned long long intersection_cycles = total.intersection_cycles_r2 + total.intersection_cycles_r3 +
      total.intersection_cycles_r4a + total.intersection_cycles_r4b + total.intersection_cycles_r4c;
  const unsigned long long intersection_calls = total.intersection_calls_r2 + total.intersection_calls_r3 +
      total.intersection_calls_r4a + total.intersection_calls_r4b + total.intersection_calls_r4c;
  const unsigned long long solver_cycles =
      total.solver_cycles_r2 + total.solver_cycles_r3 + total.solver_cycles_r4a + total.solver_cycles_r4b +
      total.solver_cycles_r4c;
  const unsigned long long solver_calls =
      total.solver_calls_r2 + total.solver_calls_r3 + total.solver_calls_r4a + total.solver_calls_r4b +
      total.solver_calls_r4c;
  const unsigned long long visible_cycles = intersection_cycles + total.closesthit_cycles;
  const unsigned long long traversal_est_cycles =
      trace_cycles > visible_cycles ? trace_cycles - visible_cycles : 0ull;
  const double trace_ms_per_frame = cycles_to_ms(trace_cycles, cycles_per_ms) / frame_count;
  const double traversal_ms_per_frame = cycles_to_ms(traversal_est_cycles, cycles_per_ms) / frame_count;
  const double intersection_ms_per_frame = cycles_to_ms(intersection_cycles, cycles_per_ms) / frame_count;
  const double solver_ms_per_frame = cycles_to_ms(solver_cycles, cycles_per_ms) / frame_count;
  const double closesthit_ms_per_frame = cycles_to_ms(total.closesthit_cycles, cycles_per_ms) / frame_count;
  const double trace_cycles_d = std::max(1.0, static_cast<double>(trace_cycles));

  std::cout << std::fixed << std::setprecision(3)
            << "Render core benchmark\n"
            << "  scene: " << (options.scene_path.empty() ? "built-in" : options.scene_path) << "\n"
            << "  primitives: " << scene.primitives.size() << "\n"
            << "  frames: " << frames << " (warmup " << warmup << ")\n"
            << "  resolution: " << scene.render.width << "x" << scene.render.height
            << ", spp " << scene.render.spp << ", depth " << scene.render.max_depth << "\n"
            << "  device clock: " << cycles_per_ms << " cycles/ms\n\n"
            << "Wall time:\n"
            << "  render_launch avg " << render.avg << "  p95 " << render.p95 << "  max " << render.max << " ms\n\n"
            << "Trace totals per frame, aggregate SM time:\n"
            << "  all trace        " << trace_ms_per_frame << " ms"
            << "  calls/frame " << static_cast<double>(trace_count) / frame_count
            << "  avg/call " << avg_ns(trace_cycles, trace_count, cycles_per_ms) << " ns\n"
            << "  primary trace    " << cycles_to_ms(total.primary_trace_cycles, cycles_per_ms) / frame_count << " ms"
            << "  calls/frame " << static_cast<double>(total.primary_trace_count) / frame_count
            << "  avg/call " << avg_ns(total.primary_trace_cycles, total.primary_trace_count, cycles_per_ms) << " ns\n"
            << "  shadow trace     " << cycles_to_ms(total.shadow_trace_cycles, cycles_per_ms) / frame_count << " ms"
            << "  calls/frame " << static_cast<double>(total.shadow_trace_count) / frame_count
            << "  avg/call " << avg_ns(total.shadow_trace_cycles, total.shadow_trace_count, cycles_per_ms) << " ns\n\n"
            << "Visible program cost per frame, aggregate SM time:\n"
            << "  traversal_est    " << traversal_ms_per_frame
            << " ms  share " << (100.0 * static_cast<double>(traversal_est_cycles) / trace_cycles_d) << "%\n"
            << "  intersection     " << intersection_ms_per_frame
            << " ms  share " << (100.0 * static_cast<double>(intersection_cycles) / trace_cycles_d)
            << "%  calls/frame " << static_cast<double>(intersection_calls) / frame_count
            << "  avg/call " << avg_ns(intersection_cycles, intersection_calls, cycles_per_ms) << " ns\n"
            << "  solver           " << solver_ms_per_frame
            << " ms  share " << (100.0 * static_cast<double>(solver_cycles) / trace_cycles_d)
            << "%  calls/frame " << static_cast<double>(solver_calls) / frame_count
            << "  avg/call " << avg_ns(solver_cycles, solver_calls, cycles_per_ms) << " ns\n"
            << "  closesthit       " << closesthit_ms_per_frame
            << " ms  share " << (100.0 * static_cast<double>(total.closesthit_cycles) / trace_cycles_d)
            << "%  calls/frame " << static_cast<double>(total.closesthit_count) / frame_count
            << "  avg/call " << avg_ns(total.closesthit_cycles, total.closesthit_count, cycles_per_ms) << " ns\n\n"
            << "Ribbon breakdown:\n"
            << "  r2 intersection  calls/frame " << static_cast<double>(total.intersection_calls_r2) / frame_count
            << "  avg/call " << avg_ns(total.intersection_cycles_r2, total.intersection_calls_r2, cycles_per_ms)
            << " ns  solver avg/call " << avg_ns(total.solver_cycles_r2, total.solver_calls_r2, cycles_per_ms) << " ns\n"
            << "  r3 intersection  calls/frame " << static_cast<double>(total.intersection_calls_r3) / frame_count
            << "  avg/call " << avg_ns(total.intersection_cycles_r3, total.intersection_calls_r3, cycles_per_ms)
            << " ns  solver avg/call " << avg_ns(total.solver_cycles_r3, total.solver_calls_r3, cycles_per_ms) << " ns\n"
            << "  r4a intersection calls/frame " << static_cast<double>(total.intersection_calls_r4a) / frame_count
            << "  avg/call " << avg_ns(total.intersection_cycles_r4a, total.intersection_calls_r4a, cycles_per_ms)
            << " ns  solver avg/call " << avg_ns(total.solver_cycles_r4a, total.solver_calls_r4a, cycles_per_ms) << " ns\n"
            << "  r4b intersection calls/frame " << static_cast<double>(total.intersection_calls_r4b) / frame_count
            << "  avg/call " << avg_ns(total.intersection_cycles_r4b, total.intersection_calls_r4b, cycles_per_ms)
            << " ns  solver avg/call " << avg_ns(total.solver_cycles_r4b, total.solver_calls_r4b, cycles_per_ms) << " ns\n"
            << "  r4c intersection calls/frame " << static_cast<double>(total.intersection_calls_r4c) / frame_count
            << "  avg/call " << avg_ns(total.intersection_cycles_r4c, total.intersection_calls_r4c, cycles_per_ms)
            << " ns  solver avg/call " << avg_ns(total.solver_cycles_r4c, total.solver_calls_r4c, cycles_per_ms) << " ns\n\n"
            << "Note:\n"
            << "  traversal_est = trace cycles - intersection cycles - closest-hit cycles.\n"
            << "  It includes OptiX BVH/AABB traversal, scheduling, miss handling, and trace overhead.\n";

  return 0;
}

static int run_static_benchmark(const AppOptions& options, Scene& scene) {
  const unsigned frames = std::max(1u, options.benchmark_static_frames);
  const unsigned warmup = std::min(10u, frames / 4u);

  OptixRibbonRenderer renderer;
  renderer.initialize(options.ptx_path);
  renderer.set_camera(scene.camera);
  renderer.set_lights(scene.lights);
  renderer.set_quality(
      scene.render.spp,
      scene.render.max_depth,
      scene.render.exposure,
      scene.render.gamma,
      scene.render.firefly_clamp);

  size_t free_before = 0;
  size_t total_before = 0;
  renderer.get_gpu_memory_info(free_before, total_before);

  renderer.set_primitives(scene.primitives);

  std::vector<double> render_ms;
  size_t min_free = free_before;
  size_t total_mem = total_before;
  render_ms.reserve(frames);

  for (unsigned frame = 0; frame < frames + warmup; ++frame) {
    const double render_start = now_ms();
    renderer.render_frame_device(scene.render.width, scene.render.height);
    const double render_end = now_ms();

    size_t free_now = 0;
    size_t total_now = 0;
    renderer.get_gpu_memory_info(free_now, total_now);
    min_free = std::min(min_free, free_now);
    total_mem = total_now;

    if (frame >= warmup) {
      render_ms.push_back(render_end - render_start);
    }
  }

  size_t free_after = 0;
  size_t total_after = 0;
  renderer.get_gpu_memory_info(free_after, total_after);

  const TimingStats render = make_stats(render_ms);
  const NormalContinuityStats normal_stats = wind_normal_continuity_metric(scene.primitives);
  const JointContinuityStats joint_stats = strip_joint_continuity_metric(scene.primitives, scene.primitives);
  const size_t primitive_bytes = scene.primitives.size() * sizeof(RibbonPrimitive);
  const size_t aabb_bytes = scene.primitives.size() * sizeof(float) * 6u;
  const size_t framebuffer_bytes =
      static_cast<size_t>(scene.render.width) * static_cast<size_t>(scene.render.height) *
      (sizeof(uchar4) + 4u * sizeof(float4));

  std::cout << std::fixed << std::setprecision(3)
            << "Static benchmark\n"
            << "  scene: " << (options.scene_path.empty() ? "built-in" : options.scene_path) << "\n"
            << "  primitives: " << scene.primitives.size() << "\n"
            << "  frames: " << frames << " (warmup " << warmup << ")\n"
            << "  resolution: " << scene.render.width << "x" << scene.render.height
            << ", spp " << scene.render.spp << ", depth " << scene.render.max_depth << "\n\n"
            << "Timing ms/frame:\n"
            << "  render_launch  avg " << render.avg << "  p95 " << render.p95 << "  max " << render.max << "\n\n"
            << "Memory MiB:\n"
            << "  initial_used    " << mib(total_before - free_before) << "\n"
            << "  peak_used       " << mib(total_mem - min_free) << "\n"
            << "  final_used      " << mib(total_after - free_after) << "\n"
            << "  ribbon_buffer   " << mib(primitive_bytes) << "\n"
            << "  aabb_buffer     " << mib(aabb_bytes) << "\n"
            << "  frame_buffers   " << mib(framebuffer_bytes) << "\n";
  std::cout << "\nNormal continuity deg:\n"
            << "  avg " << normal_stats.avg_deg << "  p95 " << normal_stats.p95_deg
            << "  max " << normal_stats.max_deg
            << "  (nearest-neighbor unsigned normal angle at u=0.7; lower is smoother)\n";
  std::cout << "\nStrip joint continuity:\n"
            << "  position gap avg " << joint_stats.position_avg << "  p95 " << joint_stats.position_p95
            << "  max " << joint_stats.position_max << "\n"
            << "  normal deg   avg " << joint_stats.normal_avg_deg << "  p95 " << joint_stats.normal_p95_deg
            << "  max " << joint_stats.normal_max_deg
            << "  (consecutive connected segment joints; lower is smoother)\n";

  return 0;
}

}  // namespace

int RenderApp::run(const AppOptions& options) {
  try {
    Scene scene;
    if (!options.scene_path.empty()) {
      std::string error;
      if (!load_scene_from_file(options.scene_path, scene, error)) {
        log(LogLevel::Error, error);
        return 2;
      }
      log(LogLevel::Info, "Loaded scene: " + options.scene_path);
    } else {
      scene = make_default_scene();
      log(LogLevel::Info, "Using built-in default scene.");
    }

    if (options.width > 0) scene.render.width = options.width;
    if (options.height > 0) scene.render.height = options.height;
    if (options.spp > 0) scene.render.spp = options.spp;
    if (options.max_depth > 0) scene.render.max_depth = options.max_depth;
    if (options.exposure > 0.0f) scene.render.exposure = options.exposure;
    if (options.gamma > 0.0f) scene.render.gamma = options.gamma;
    if (options.firefly_clamp > 0.0f) scene.render.firefly_clamp = options.firefly_clamp;
    if (!options.output_path.empty()) scene.render.output_path = options.output_path;

    if (options.benchmark_aabb_frames > 0) {
      return run_aabb_benchmark(options, scene);
    }
    if (options.benchmark_wind_frames > 0) {
      return run_wind_benchmark(options, scene);
    }
    if (options.benchmark_core_frames > 0) {
      return run_core_benchmark(options, scene);
    }
    if (options.benchmark_static_frames > 0) {
      return run_static_benchmark(options, scene);
    }

    OptixRibbonRenderer renderer;
    renderer.initialize(options.ptx_path);
    renderer.set_camera(scene.camera);
    renderer.set_lights(scene.lights);
    renderer.set_quality(
        scene.render.spp,
        scene.render.max_depth,
        scene.render.exposure,
        scene.render.gamma,
        scene.render.firefly_clamp);
    renderer.set_primitives(scene.primitives);
    renderer.render_to_ppm(
        scene.render.output_path, scene.render.width, scene.render.height, options.aov_dir, options.denoise);
    log(LogLevel::Info, "Render complete: " + scene.render.output_path);
    return 0;
  } catch (const std::exception& e) {
    log(LogLevel::Error, e.what());
    return 1;
  }
}

}  // namespace rr
