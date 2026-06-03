#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "shared/ribbon_types.h"

namespace rr {

inline float3 wind_add3(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 wind_sub3(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 wind_mul3(const float3& a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

inline float wind_dot3(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float wind_length3(const float3& a) {
  return std::sqrt(std::max(0.0f, wind_dot3(a, a)));
}

inline float3 wind_normalize3(const float3& a) {
  const float len2 = std::max(1e-20f, wind_dot3(a, a));
  return wind_mul3(a, 1.0f / std::sqrt(len2));
}

inline float3 wind_cross3(const float3& a, const float3& b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float3 ribbon3_c1_host(const Ribbon3Data& r) {
  return wind_mul3(wind_sub3(r.b1, r.b0), 2.0f);
}

inline float3 ribbon3_c2_host(const Ribbon3Data& r) {
  return wind_add3(wind_sub3(r.b0, wind_mul3(r.b1, 2.0f)), r.b2);
}

inline float3 ribbon3_velocity_host(const Ribbon3Data& r, float u) {
  return wind_add3(ribbon3_c1_host(r), wind_mul3(ribbon3_c2_host(r), 2.0f * u));
}

inline float3 ribbon3_curve_host(const Ribbon3Data& r, float u) {
  return wind_add3(wind_add3(r.b0, wind_mul3(ribbon3_c1_host(r), u)), wind_mul3(ribbon3_c2_host(r), u * u));
}

inline float3 ribbon3_bitangent_host(const Ribbon3Data& r, float u) {
  const float3 c1 = ribbon3_c1_host(r);
  const float3 c2 = ribbon3_c2_host(r);
  const float3 base = wind_add3(wind_mul3(r.bt0, 1.0f - u), wind_mul3(r.bt1, r.alpha * u));
  const float3 e = wind_mul3(
      wind_add3(wind_mul3(c2, r.beta), wind_mul3(wind_add3(c1, wind_mul3(c2, u)), r.gamma)),
      u * (1.0f - u));
  return wind_add3(base, e);
}

inline float3 ribbon3_normal_host(const Ribbon3Data& r, float u) {
  const float3 vel = ribbon3_velocity_host(r, u);
  const float3 bitangent = wind_normalize3(ribbon3_bitangent_host(r, u));
  return wind_normalize3(wind_cross3(vel, bitangent));
}

inline float3 ribbon2_point_host(const Ribbon2Data& r, float u, float v) {
  const float omu = 1.0f - u;
  const float omv = 1.0f - v;
  return wind_add3(
      wind_add3(wind_mul3(r.q00, omu * omv), wind_mul3(r.q01, omu * v)),
      wind_add3(wind_mul3(r.q10, u * omv), wind_mul3(r.q11, u * v)));
}

inline float3 ribbon2_normal_host(const Ribbon2Data& r, float u, float v) {
  const float3 dQdu = wind_add3(wind_mul3(wind_sub3(r.q10, r.q00), 1.0f - v), wind_mul3(wind_sub3(r.q11, r.q01), v));
  const float3 dQdv = wind_add3(wind_mul3(wind_sub3(r.q01, r.q00), 1.0f - u), wind_mul3(wind_sub3(r.q11, r.q10), u));
  return wind_normalize3(wind_cross3(dQdu, dQdv));
}

inline float3 ribbon4_directrix_host(const Ribbon4Data& r, float u) {
  return wind_add3(wind_add3(wind_mul3(r.d3, u * u * u), wind_mul3(r.d2, u * u)), wind_add3(wind_mul3(r.d1, u), r.d0));
}

inline float3 ribbon4_directrix_derivative_host(const Ribbon4Data& r, float u) {
  return wind_add3(wind_add3(wind_mul3(r.d3, 3.0f * u * u), wind_mul3(r.d2, 2.0f * u)), r.d1);
}

inline float3 ribbon4_generator_host(const Ribbon4Data& r, float u) {
  return wind_add3(wind_add3(wind_mul3(r.g2, u * u), wind_mul3(r.g1, u)), r.g0);
}

inline float3 ribbon4_generator_derivative_host(const Ribbon4Data& r, float u) {
  return wind_add3(wind_mul3(r.g2, 2.0f * u), r.g1);
}

inline float3 ribbon4c_center_host(const Ribbon4Data& r, float v) {
  return wind_add3(wind_add3(r.d0, wind_mul3(r.g0, v)), wind_mul3(r.d2, v * v));
}

inline float3 ribbon4c_width_axis_host(const Ribbon4Data& r, float v) {
  return wind_add3(wind_add3(r.d1, wind_mul3(r.g1, v)), wind_mul3(r.d3, v * v));
}

inline float3 ribbon4c_center_derivative_host(const Ribbon4Data& r, float v) {
  return wind_add3(r.g0, wind_mul3(r.d2, 2.0f * v));
}

inline float3 ribbon4c_width_axis_derivative_host(const Ribbon4Data& r, float v) {
  return wind_add3(r.g1, wind_mul3(r.d3, 2.0f * v));
}

inline float3 ribbon4_normal_host(const Ribbon4Data& r, float u, float v = 0.0f) {
  const float3 du = wind_add3(ribbon4_directrix_derivative_host(r, u), wind_mul3(ribbon4_generator_derivative_host(r, u), v));
  const float3 dv = ribbon4_generator_host(r, u);
  return wind_normalize3(wind_cross3(du, dv));
}

inline float3 ribbon4c_normal_host(const Ribbon4Data& r, float v = 0.7f) {
  const float3 du = ribbon4c_width_axis_host(r, v);
  const float3 dv = ribbon4c_center_derivative_host(r, v);
  return wind_normalize3(wind_cross3(du, dv));
}

inline bool is_benchmark_ribbon(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A || p.type == RIBBON_TYPE_4B ||
      p.type == RIBBON_TYPE_4C) return true;
  return p.type == RIBBON_TYPE_2 && p.r2.q00.y > -0.02f;
}

inline bool is_motion_ribbon(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A || p.type == RIBBON_TYPE_4B ||
      p.type == RIBBON_TYPE_4C) return true;
  return p.type == RIBBON_TYPE_2 && p.r2.q00.y > -0.02f;
}

inline float3 segment_start_host(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_2) return p.r2.q00;
  if (p.type == RIBBON_TYPE_3) return p.r3.b0;
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_center_host(p.r4, 0.0f);
  return ribbon4_directrix_host(p.r4, 0.0f);
}

inline float3 segment_mid_host(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_2) return ribbon2_point_host(p.r2, 0.5f, 0.5f);
  if (p.type == RIBBON_TYPE_3) {
    return wind_add3(wind_mul3(p.r3.b0, 0.25f), wind_add3(wind_mul3(p.r3.b1, 0.5f), wind_mul3(p.r3.b2, 0.25f)));
  }
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_center_host(p.r4, 0.5f);
  return ribbon4_directrix_host(p.r4, 0.5f);
}

inline float3 segment_end_host(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_2) return p.r2.q11;
  if (p.type == RIBBON_TYPE_3) return p.r3.b2;
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_center_host(p.r4, 1.0f);
  return ribbon4_directrix_host(p.r4, 1.0f);
}

inline float3 segment_normal_host(const RibbonPrimitive& p, float u) {
  if (p.type == RIBBON_TYPE_2) return ribbon2_normal_host(p.r2, u, u);
  if (p.type == RIBBON_TYPE_3) return ribbon3_normal_host(p.r3, u);
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_normal_host(p.r4, u);
  return ribbon4_normal_host(p.r4, u);
}

inline bool is_wind_ribbon(const RibbonPrimitive& p) {
  if (!(p.type == RIBBON_TYPE_2 || p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A ||
        p.type == RIBBON_TYPE_4B || p.type == RIBBON_TYPE_4C)) return false;
  return segment_end_host(p).y - segment_start_host(p).y > 0.01f;
}

inline float3 wind_displacement(const RibbonPrimitive& p, float time, float strength, float speed) {
  const float3 base = segment_start_host(p);
  const float height = std::max(0.05f, segment_end_host(p).y);
  const float phase = base.x * 2.2f + base.z * 3.1f + time * speed;
  const float gust = 0.55f + 0.45f * std::sin(phase) + 0.20f * std::sin(phase * 2.37f + 1.1f);
  const float amp = strength * height * gust;
  const float3 dir = wind_normalize3(make_float3(0.82f, 0.0f, -0.38f));
  return wind_mul3(dir, amp);
}

inline float3 grass_point_wind_displacement(const float3& p, float time, float strength, float speed) {
  const float height = std::max(0.0f, p.y);
  const float phase = p.x * 2.2f + p.z * 3.1f + p.y * 0.35f + time * speed;
  const float gust = 0.55f + 0.45f * std::sin(phase) + 0.20f * std::sin(phase * 2.37f + 1.1f);
  const float3 dir = wind_normalize3(make_float3(0.82f, 0.0f, -0.38f));
  float3 d = wind_mul3(dir, strength * height * gust);
  d.y = -wind_length3(d) * 0.055f;
  return d;
}

inline void add_quadratic_directrix_delta(Ribbon4Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = wind_sub3(dm, d0);
  const float3 e = wind_sub3(d1, d0);
  const float3 a = wind_sub3(wind_mul3(m, 4.0f), e);
  const float3 b = wind_sub3(wind_mul3(e, 2.0f), wind_mul3(m, 4.0f));
  r.d0 = wind_add3(r.d0, d0);
  r.d1 = wind_add3(r.d1, a);
  r.d2 = wind_add3(r.d2, b);
}

inline float3 quadratic_bezier_control_delta(const float3& d0, const float3& dm, const float3& d1) {
  return wind_sub3(wind_mul3(dm, 2.0f), wind_mul3(wind_add3(d0, d1), 0.5f));
}

inline void add_quadratic_ribbon2_center_delta(Ribbon2Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = wind_sub3(dm, d0);
  const float3 e = wind_sub3(d1, d0);
  const float3 a = wind_sub3(wind_mul3(m, 4.0f), e);
  const float3 b = wind_sub3(wind_mul3(e, 2.0f), wind_mul3(m, 4.0f));
  r.q00 = wind_add3(r.q00, d0);
  r.q01 = wind_add3(r.q01, d0);
  r.q10 = wind_add3(r.q10, wind_add3(d0, a));
  r.q11 = wind_add3(r.q11, wind_add3(d0, wind_add3(a, b)));
}

inline void add_quadratic_ribbon4c_center_delta(Ribbon4Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = wind_sub3(dm, d0);
  const float3 e = wind_sub3(d1, d0);
  r.d0 = wind_add3(r.d0, d0);
  r.g0 = wind_add3(r.g0, wind_sub3(wind_mul3(m, 4.0f), e));
  r.d2 = wind_add3(r.d2, wind_sub3(wind_mul3(e, 2.0f), wind_mul3(m, 4.0f)));
}

inline void add_quadratic_ribbon4c_width_delta(Ribbon4Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = wind_sub3(dm, d0);
  const float3 e = wind_sub3(d1, d0);
  r.d1 = wind_add3(r.d1, d0);
  r.g1 = wind_add3(r.g1, wind_sub3(wind_mul3(m, 4.0f), e));
  r.d3 = wind_add3(r.d3, wind_sub3(wind_mul3(e, 2.0f), wind_mul3(m, 4.0f)));
}

inline void tilt_ribbon4_generator(Ribbon4Data& r, const float3& d, float amount) {
  const float3 g1 = ribbon4_generator_host(r, 1.0f);
  const float3 target = wind_normalize3(wind_add3(g1, wind_mul3(d, amount)));
  r.g1 = wind_add3(r.g1, wind_sub3(target, g1));
}

inline float3 strip_point_wind_displacement(const float3& p, float time, float strength, float speed) {
  const float height_factor = std::max(0.0f, p.y);
  const float phase = p.x * 1.6f + p.z * 2.1f + p.y * 0.75f + time * speed;
  const float gust = 0.60f + 0.40f * std::sin(phase) + 0.16f * std::sin(phase * 1.91f + 0.7f);
  const float3 dir = wind_normalize3(make_float3(0.78f, 0.0f, -0.44f));
  float3 d = wind_mul3(dir, strength * height_factor * gust);
  d.y = -wind_length3(d) * 0.06f;
  return d;
}

inline void apply_wind_animation(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time,
    float strength,
    float speed) {
  animated = base;
  if (std::abs(strength) < 1e-8f) return;
  for (size_t i = 0; i < animated.size(); ++i) {
    if (!is_wind_ribbon(base[i])) continue;
    const float3 p0 = segment_start_host(base[i]);
    const float3 pm = segment_mid_host(base[i]);
    const float3 p1 = segment_end_host(base[i]);
    const float3 d0 = grass_point_wind_displacement(p0, time, strength, speed);
    const float3 dm = grass_point_wind_displacement(pm, time, strength, speed);
    const float3 d1 = grass_point_wind_displacement(p1, time, strength, speed);
    if (base[i].type == RIBBON_TYPE_2) {
      add_quadratic_ribbon2_center_delta(animated[i].r2, d0, dm, d1);
    } else if (base[i].type == RIBBON_TYPE_3) {
      animated[i].r3.b0 = wind_add3(base[i].r3.b0, d0);
      animated[i].r3.b1 = wind_add3(base[i].r3.b1, quadratic_bezier_control_delta(d0, dm, d1));
      animated[i].r3.b2 = wind_add3(base[i].r3.b2, d1);
      animated[i].r3.bt1 = wind_normalize3(wind_add3(base[i].r3.bt1, wind_mul3(d1, 0.10f)));
    } else if (base[i].type == RIBBON_TYPE_4C) {
      add_quadratic_ribbon4c_center_delta(animated[i].r4, d0, dm, d1);
    } else {
      add_quadratic_directrix_delta(animated[i].r4, d0, dm, d1);
      tilt_ribbon4_generator(animated[i].r4, d1, 0.10f);
    }
  }
}

inline float eval_width_host(float w0, float w1, float w2, float u) {
  return w0 + w1 * u + w2 * u * u;
}

inline void apply_ribbon3_edge_sampled_wind(
    const RibbonPrimitive& base,
    RibbonPrimitive& animated,
    float time,
    float strength,
    float speed) {
  const float3 p0 = segment_start_host(base);
  const float3 pm = segment_mid_host(base);
  const float3 p1 = segment_end_host(base);
  const float3 d0 = grass_point_wind_displacement(p0, time, strength, speed);
  const float3 dm = grass_point_wind_displacement(pm, time, strength, speed);
  const float3 d1 = grass_point_wind_displacement(p1, time, strength, speed);

  animated.r3.b0 = wind_add3(base.r3.b0, d0);
  animated.r3.b1 = wind_add3(base.r3.b1, quadratic_bezier_control_delta(d0, dm, d1));
  animated.r3.b2 = wind_add3(base.r3.b2, d1);

  const auto deformed_bitangent = [&](float u) {
    const float3 center = ribbon3_curve_host(base.r3, u);
    const float3 bitangent = wind_normalize3(ribbon3_bitangent_host(base.r3, u));
    const float width = std::fabs(eval_width_host(base.r3.w0, base.r3.w1, base.r3.w2, u));
    const float3 left = wind_sub3(center, wind_mul3(bitangent, width));
    const float3 right = wind_add3(center, wind_mul3(bitangent, width));
    const float3 dl = grass_point_wind_displacement(left, time, strength, speed);
    const float3 dr = grass_point_wind_displacement(right, time, strength, speed);
    return wind_normalize3(wind_add3(wind_mul3(bitangent, 2.0f * width), wind_sub3(dr, dl)));
  };

  animated.r3.bt0 = deformed_bitangent(0.0f);
  animated.r3.bt1 = deformed_bitangent(1.0f);
}

inline void apply_ribbon4c_edge_sampled_wind(
    const RibbonPrimitive& base,
    RibbonPrimitive& animated,
    float time,
    float strength,
    float speed) {
  float3 center_delta[3];
  float3 width_delta[3];
  const float samples[3] = {0.0f, 0.5f, 1.0f};
  for (int i = 0; i < 3; ++i) {
    const float v = samples[i];
    const float3 c = ribbon4c_center_host(base.r4, v);
    const float3 w = ribbon4c_width_axis_host(base.r4, v);
    const float3 left = wind_sub3(c, w);
    const float3 right = wind_add3(c, w);
    const float3 dl = grass_point_wind_displacement(left, time, strength, speed);
    const float3 dr = grass_point_wind_displacement(right, time, strength, speed);
    center_delta[i] = wind_mul3(wind_add3(dl, dr), 0.5f);
    width_delta[i] = wind_mul3(wind_sub3(dr, dl), 0.5f);
  }
  add_quadratic_ribbon4c_center_delta(animated.r4, center_delta[0], center_delta[1], center_delta[2]);
  add_quadratic_ribbon4c_width_delta(animated.r4, width_delta[0], width_delta[1], width_delta[2]);
}

inline void apply_width_dynamic_grass_wind_animation(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time,
    float strength,
    float speed) {
  animated = base;
  if (std::abs(strength) < 1e-8f) return;
  for (size_t i = 0; i < animated.size(); ++i) {
    if (!is_wind_ribbon(base[i])) continue;
    if (base[i].type == RIBBON_TYPE_4C) {
      apply_ribbon4c_edge_sampled_wind(base[i], animated[i], time, strength, speed);
    } else {
      std::vector<RibbonPrimitive> one_base{base[i]};
      std::vector<RibbonPrimitive> one_animated;
      apply_wind_animation(one_base, one_animated, time, strength, speed);
      animated[i] = one_animated[0];
    }
  }
}

inline void enforce_ribbon3_joint_constraints(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated) {
  const size_t n = std::min(base.size(), animated.size());
  for (size_t i = 0; i + 1 < n; ++i) {
    if (base[i].type != RIBBON_TYPE_3 || base[i + 1].type != RIBBON_TYPE_3) continue;
    if (wind_length3(wind_sub3(segment_end_host(base[i]), segment_start_host(base[i + 1]))) > 1e-3f) continue;

    const float3 joint = animated[i].r3.b2;
    animated[i + 1].r3.b0 = joint;
    animated[i + 1].r3.b1 = wind_add3(joint, wind_sub3(joint, animated[i].r3.b1));

    float3 end_bitangent = wind_mul3(animated[i].r3.bt1, animated[i].r3.alpha);
    float3 start_bitangent = animated[i + 1].r3.bt0;
    float3 joint_bitangent = wind_add3(end_bitangent, start_bitangent);
    if (wind_length3(joint_bitangent) < 1e-6f) joint_bitangent = end_bitangent;
    joint_bitangent = wind_normalize3(joint_bitangent);
    const float alpha = std::fabs(animated[i].r3.alpha) > 1e-6f ? animated[i].r3.alpha : 1.0f;
    animated[i].r3.bt1 = wind_mul3(joint_bitangent, 1.0f / alpha);
    animated[i + 1].r3.bt0 = joint_bitangent;
  }
}

inline void enforce_ribbon4c_joint_constraints(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated) {
  const size_t n = std::min(base.size(), animated.size());
  for (size_t i = 0; i + 1 < n; ++i) {
    if (base[i].type != RIBBON_TYPE_4C || base[i + 1].type != RIBBON_TYPE_4C) continue;
    if (wind_length3(wind_sub3(segment_end_host(base[i]), segment_start_host(base[i + 1]))) > 1e-3f) continue;

    const float3 center_joint = ribbon4c_center_host(animated[i].r4, 1.0f);
    const float3 width_joint = ribbon4c_width_axis_host(animated[i].r4, 1.0f);
    const float3 center_end = ribbon4c_center_host(animated[i + 1].r4, 1.0f);
    const float3 width_end = ribbon4c_width_axis_host(animated[i + 1].r4, 1.0f);
    const float3 center_tangent = ribbon4c_center_derivative_host(animated[i].r4, 1.0f);
    const float3 width_tangent = ribbon4c_width_axis_derivative_host(animated[i].r4, 1.0f);

    animated[i + 1].r4.d0 = center_joint;
    animated[i + 1].r4.g0 = center_tangent;
    animated[i + 1].r4.d2 = wind_sub3(wind_sub3(center_end, center_joint), center_tangent);

    animated[i + 1].r4.d1 = width_joint;
    animated[i + 1].r4.g1 = width_tangent;
    animated[i + 1].r4.d3 = wind_sub3(wind_sub3(width_end, width_joint), width_tangent);
  }
}

inline void apply_rebuilt_grass_wind_animation(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time,
    float strength,
    float speed) {
  animated = base;
  if (std::abs(strength) < 1e-8f) return;
  for (size_t i = 0; i < animated.size(); ++i) {
    if (!is_wind_ribbon(base[i])) continue;
    if (base[i].type == RIBBON_TYPE_3) {
      apply_ribbon3_edge_sampled_wind(base[i], animated[i], time, strength, speed);
    } else if (base[i].type == RIBBON_TYPE_4C) {
      apply_ribbon4c_edge_sampled_wind(base[i], animated[i], time, strength, speed);
    } else {
      std::vector<RibbonPrimitive> one_base{base[i]};
      std::vector<RibbonPrimitive> one_animated;
      apply_wind_animation(one_base, one_animated, time, strength, speed);
      animated[i] = one_animated[0];
    }
  }
  enforce_ribbon3_joint_constraints(base, animated);
  enforce_ribbon4c_joint_constraints(base, animated);
}

inline void apply_independent_strip_wind_animation(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time,
    float strength,
    float speed) {
  animated = base;
  if (std::abs(strength) < 1e-8f) return;
  for (size_t i = 0; i < animated.size(); ++i) {
    if (!is_motion_ribbon(base[i])) continue;
    const float3 d = strip_point_wind_displacement(segment_start_host(base[i]), time, strength, speed);
    if (base[i].type == RIBBON_TYPE_2) {
      animated[i].r2.q01 = wind_add3(base[i].r2.q01, wind_mul3(d, 0.50f));
      animated[i].r2.q10 = wind_add3(base[i].r2.q10, wind_mul3(d, 0.50f));
      animated[i].r2.q11 = wind_add3(base[i].r2.q11, d);
    } else if (base[i].type == RIBBON_TYPE_3) {
      animated[i].r3.b1 = wind_add3(base[i].r3.b1, wind_mul3(d, 0.50f));
      animated[i].r3.b2 = wind_add3(base[i].r3.b2, d);
      animated[i].r3.bt1 = wind_normalize3(wind_add3(base[i].r3.bt1, wind_mul3(d, 0.08f)));
    } else if (base[i].type == RIBBON_TYPE_4C) {
      animated[i].r4.g0 = wind_add3(animated[i].r4.g0, wind_mul3(d, 0.50f));
      animated[i].r4.d2 = wind_add3(animated[i].r4.d2, d);
    } else {
      add_quadratic_directrix_delta(
          animated[i].r4,
          make_float3(0.0f, 0.0f, 0.0f),
          wind_mul3(d, 0.50f),
          d);
      tilt_ribbon4_generator(animated[i].r4, d, 0.08f);
    }
  }
}

inline void apply_constrained_strip_wind_animation(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time,
    float strength,
    float speed) {
  animated = base;
  if (std::abs(strength) < 1e-8f) return;
  for (size_t i = 0; i < animated.size(); ++i) {
    if (!is_motion_ribbon(base[i])) continue;
    if (base[i].type == RIBBON_TYPE_2) {
      animated[i].r2.q00 = wind_add3(base[i].r2.q00, strip_point_wind_displacement(base[i].r2.q00, time, strength, speed));
      animated[i].r2.q01 = wind_add3(base[i].r2.q01, strip_point_wind_displacement(base[i].r2.q01, time, strength, speed));
      animated[i].r2.q10 = wind_add3(base[i].r2.q10, strip_point_wind_displacement(base[i].r2.q10, time, strength, speed));
      animated[i].r2.q11 = wind_add3(base[i].r2.q11, strip_point_wind_displacement(base[i].r2.q11, time, strength, speed));
    } else if (base[i].type == RIBBON_TYPE_3) {
      animated[i].r3.b0 = wind_add3(base[i].r3.b0, strip_point_wind_displacement(base[i].r3.b0, time, strength, speed));
      animated[i].r3.b1 = wind_add3(base[i].r3.b1, strip_point_wind_displacement(base[i].r3.b1, time, strength, speed));
      animated[i].r3.b2 = wind_add3(base[i].r3.b2, strip_point_wind_displacement(base[i].r3.b2, time, strength, speed));
      animated[i].r3.bt0 = base[i].r3.bt0;
      animated[i].r3.bt1 = base[i].r3.bt1;
    } else if (base[i].type == RIBBON_TYPE_4C) {
      const float3 p0 = ribbon4c_center_host(base[i].r4, 0.0f);
      const float3 pm = ribbon4c_center_host(base[i].r4, 0.5f);
      const float3 p1 = ribbon4c_center_host(base[i].r4, 1.0f);
      const float3 d0 = strip_point_wind_displacement(p0, time, strength, speed);
      const float3 dm = strip_point_wind_displacement(pm, time, strength, speed);
      const float3 d1 = strip_point_wind_displacement(p1, time, strength, speed);
      const float3 m = wind_sub3(dm, d0);
      const float3 e = wind_sub3(d1, d0);
      animated[i].r4.d0 = wind_add3(animated[i].r4.d0, d0);
      animated[i].r4.g0 = wind_add3(animated[i].r4.g0, wind_sub3(wind_mul3(m, 4.0f), e));
      animated[i].r4.d2 = wind_add3(animated[i].r4.d2, wind_sub3(wind_mul3(e, 2.0f), wind_mul3(m, 4.0f)));
    } else {
      const float3 p0 = ribbon4_directrix_host(base[i].r4, 0.0f);
      const float3 pm = ribbon4_directrix_host(base[i].r4, 0.5f);
      const float3 p1 = ribbon4_directrix_host(base[i].r4, 1.0f);
      add_quadratic_directrix_delta(
          animated[i].r4,
          strip_point_wind_displacement(p0, time, strength, speed),
          strip_point_wind_displacement(pm, time, strength, speed),
          strip_point_wind_displacement(p1, time, strength, speed));
    }
  }
}

inline float wind_smoothness_metric(
    const std::vector<RibbonPrimitive>& base,
    const std::vector<RibbonPrimitive>& animated,
    float strength) {
  const float radius2 = 0.35f * 0.35f;
  float sum = 0.0f;
  unsigned count = 0;
  const size_t n = std::min(base.size(), animated.size());
  for (size_t i = 0; i < n; ++i) {
    if (!is_wind_ribbon(base[i])) continue;
    float best_d2 = radius2;
    int best = -1;
    const float3 pi = segment_start_host(base[i]);
    for (size_t j = 0; j < n; ++j) {
      if (i == j || !is_wind_ribbon(base[j])) continue;
      const float3 pj = segment_start_host(base[j]);
      const float dx = pi.x - pj.x;
      const float dz = pi.z - pj.z;
      const float d2 = dx * dx + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = static_cast<int>(j);
      }
    }
    if (best >= 0) {
      const float3 di = wind_sub3(segment_end_host(animated[i]), segment_end_host(base[i]));
      const float3 dj = wind_sub3(segment_end_host(animated[best]), segment_end_host(base[best]));
      sum += wind_length3(wind_sub3(di, dj)) / std::max(1e-4f, strength);
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

struct NormalContinuityStats {
  float avg_deg = 0.0f;
  float p95_deg = 0.0f;
  float max_deg = 0.0f;
  unsigned pairs = 0;
};

struct JointContinuityStats {
  float position_avg = 0.0f;
  float position_p95 = 0.0f;
  float position_max = 0.0f;
  float normal_avg_deg = 0.0f;
  float normal_p95_deg = 0.0f;
  float normal_max_deg = 0.0f;
  unsigned joints = 0;
};

struct MotionFreedomStats {
  float loss_norm = 0.0f;
  float variance_retention = 0.0f;
  float peak_response_ratio = 0.0f;
  unsigned samples = 0;
};

inline MotionFreedomStats motion_freedom_metric(
    const std::vector<RibbonPrimitive>& base,
    const std::vector<RibbonPrimitive>& free_motion,
    const std::vector<RibbonPrimitive>& constrained_motion) {
  const size_t n = std::min(base.size(), std::min(free_motion.size(), constrained_motion.size()));
  std::vector<float3> df;
  std::vector<float3> dc;
  df.reserve(n);
  dc.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (!is_motion_ribbon(base[i])) continue;
    df.push_back(wind_sub3(segment_end_host(free_motion[i]), segment_end_host(base[i])));
    dc.push_back(wind_sub3(segment_end_host(constrained_motion[i]), segment_end_host(base[i])));
  }
  if (df.empty()) return {};

  float mean_free_mag = 0.0f;
  float mean_loss = 0.0f;
  float max_free = 0.0f;
  float max_constrained = 0.0f;
  float3 mean_free = make_float3(0.0f, 0.0f, 0.0f);
  float3 mean_constrained = make_float3(0.0f, 0.0f, 0.0f);
  for (size_t i = 0; i < df.size(); ++i) {
    mean_free_mag += wind_length3(df[i]);
    mean_loss += wind_length3(wind_sub3(df[i], dc[i]));
    max_free = std::max(max_free, wind_length3(df[i]));
    max_constrained = std::max(max_constrained, wind_length3(dc[i]));
    mean_free = wind_add3(mean_free, df[i]);
    mean_constrained = wind_add3(mean_constrained, dc[i]);
  }
  const float inv_n = 1.0f / static_cast<float>(df.size());
  mean_free_mag *= inv_n;
  mean_loss *= inv_n;
  mean_free = wind_mul3(mean_free, inv_n);
  mean_constrained = wind_mul3(mean_constrained, inv_n);

  float var_free = 0.0f;
  float var_constrained = 0.0f;
  for (size_t i = 0; i < df.size(); ++i) {
    const float3 vf = wind_sub3(df[i], mean_free);
    const float3 vc = wind_sub3(dc[i], mean_constrained);
    var_free += wind_dot3(vf, vf);
    var_constrained += wind_dot3(vc, vc);
  }
  var_free *= inv_n;
  var_constrained *= inv_n;

  return {
      mean_loss / std::max(1e-6f, mean_free_mag),
      var_constrained / std::max(1e-8f, var_free),
      max_constrained / std::max(1e-6f, max_free),
      static_cast<unsigned>(df.size())};
}

inline JointContinuityStats strip_joint_continuity_metric(
    const std::vector<RibbonPrimitive>& base,
    const std::vector<RibbonPrimitive>& primitives) {
  std::vector<float> gaps;
  std::vector<float> angles;
  const size_t n = std::min(base.size(), primitives.size());
  for (size_t i = 0; i + 1 < n; ++i) {
    if (!is_benchmark_ribbon(base[i]) || !is_benchmark_ribbon(base[i + 1])) continue;
    if (base[i].type != base[i + 1].type) continue;
    if (wind_length3(wind_sub3(segment_end_host(base[i]), segment_start_host(base[i + 1]))) > 1e-3f) continue;

    gaps.push_back(wind_length3(wind_sub3(segment_end_host(primitives[i]), segment_start_host(primitives[i + 1]))));

    const float3 ni = segment_normal_host(primitives[i], 1.0f);
    const float3 nj = segment_normal_host(primitives[i + 1], 0.0f);
    const float d = std::min(1.0f, std::max(0.0f, std::fabs(wind_dot3(ni, nj))));
    angles.push_back(std::acos(d) * 57.2957795f);
  }

  if (gaps.empty()) return {};
  std::sort(gaps.begin(), gaps.end());
  std::sort(angles.begin(), angles.end());
  const size_t p95_index = std::min(gaps.size() - 1, static_cast<size_t>(gaps.size() * 95 / 100));
  const float gap_sum = std::accumulate(gaps.begin(), gaps.end(), 0.0f);
  const float angle_sum = std::accumulate(angles.begin(), angles.end(), 0.0f);
  return {
      gap_sum / static_cast<float>(gaps.size()),
      gaps[p95_index],
      gaps.back(),
      angle_sum / static_cast<float>(angles.size()),
      angles[p95_index],
      angles.back(),
      static_cast<unsigned>(gaps.size())};
}

inline NormalContinuityStats wind_normal_continuity_metric(
    const std::vector<RibbonPrimitive>& primitives,
    float u = 0.7f) {
  const float radius2 = 0.35f * 0.35f;
  std::vector<float> angles;
  const size_t n = primitives.size();
  for (size_t i = 0; i < n; ++i) {
    if (!is_benchmark_ribbon(primitives[i])) continue;
    float best_d2 = radius2;
    int best = -1;
    const float3 pi = segment_mid_host(primitives[i]);
    for (size_t j = 0; j < n; ++j) {
      if (i == j || !is_benchmark_ribbon(primitives[j]) || primitives[i].type != primitives[j].type) continue;
      const float3 pj = segment_mid_host(primitives[j]);
      const float dx = pi.x - pj.x;
      const float dz = pi.z - pj.z;
      const float d2 = dx * dx + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = static_cast<int>(j);
      }
    }
    if (best >= 0) {
      const float3 ni = segment_normal_host(primitives[i], u);
      const float3 nj = segment_normal_host(primitives[best], u);
      const float d = std::min(1.0f, std::max(0.0f, std::fabs(wind_dot3(ni, nj))));
      angles.push_back(std::acos(d) * 57.2957795f);
    }
  }

  if (angles.empty()) return {};
  std::sort(angles.begin(), angles.end());
  const float sum = std::accumulate(angles.begin(), angles.end(), 0.0f);
  const size_t p95_index = std::min(angles.size() - 1, static_cast<size_t>(angles.size() * 95 / 100));
  return {
      sum / static_cast<float>(angles.size()),
      angles[p95_index],
      angles.back(),
      static_cast<unsigned>(angles.size())};
}

}  // namespace rr
