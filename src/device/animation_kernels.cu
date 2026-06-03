#include "device/animation_kernels.h"

#include <cmath>

namespace {

__device__ float3 add3(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 sub3(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 mul3(const float3& a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float dot3(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize3(const float3& a) {
  const float len2 = fmaxf(1e-20f, dot3(a, a));
  return mul3(a, rsqrtf(len2));
}

__device__ float length3(const float3& a) {
  return sqrtf(fmaxf(0.0f, dot3(a, a)));
}

__device__ float eval_poly(const float* c, int degree, float u) {
  float v = c[degree];
  for (int i = degree - 1; i >= 0; --i) v = v * u + c[i];
  return v;
}

__device__ float component3(const float3& v, int coord) {
  return coord == 0 ? v.x : (coord == 1 ? v.y : v.z);
}

__device__ int solve_quadratic_roots(float a, float b, float c, float roots[2]) {
  constexpr float eps = 1e-12f;
  if (fabsf(a) < eps) {
    if (fabsf(b) < eps) return 0;
    roots[0] = -c / b;
    return 1;
  }
  const float disc = b * b - 4.0f * a * c;
  if (disc < 0.0f) return 0;
  const float s = sqrtf(fmaxf(0.0f, disc));
  roots[0] = (-b - s) / (2.0f * a);
  roots[1] = (-b + s) / (2.0f * a);
  return fabsf(roots[1] - roots[0]) > 1e-6f ? 2 : 1;
}

__device__ void expand_coord(float& mn, float& mx, float v) {
  mn = fminf(mn, v);
  mx = fmaxf(mx, v);
}

__device__ void expand_poly_extrema(float& mn, float& mx, const float* c, int degree) {
  expand_coord(mn, mx, eval_poly(c, degree, 0.0f));
  expand_coord(mn, mx, eval_poly(c, degree, 1.0f));

  if (degree == 2) {
    constexpr float eps = 1e-12f;
    if (fabsf(c[2]) > eps) {
      const float u = -c[1] / (2.0f * c[2]);
      if (u > 0.0f && u < 1.0f) {
        expand_coord(mn, mx, eval_poly(c, degree, u));
      }
    }
    return;
  }

  if (degree == 3) {
    float roots[2];
    const int n = solve_quadratic_roots(3.0f * c[3], 2.0f * c[2], c[1], roots);
    for (int i = 0; i < n; ++i) {
      if (roots[i] > 0.0f && roots[i] < 1.0f) {
        expand_coord(mn, mx, eval_poly(c, degree, roots[i]));
      }
    }
    return;
  }

  if (degree == 4 || degree == 5) {
    float d[5] = {};
    for (int j = 0; j < degree; ++j) d[j] = static_cast<float>(j + 1) * c[j + 1];
    constexpr int kIntervals = 64;
    float prev_u = 0.0f;
    float prev_f = eval_poly(d, degree - 1, prev_u);
    for (int i = 1; i <= kIntervals; ++i) {
      const float u = static_cast<float>(i) / static_cast<float>(kIntervals);
      const float f = eval_poly(d, degree - 1, u);
      if (prev_f == 0.0f) {
        expand_coord(mn, mx, eval_poly(c, degree, prev_u));
      } else if ((prev_f < 0.0f) != (f < 0.0f)) {
        float lo = prev_u;
        float hi = u;
        float flo = prev_f;
        for (int it = 0; it < 24; ++it) {
          const float mid = 0.5f * (lo + hi);
          const float fm = eval_poly(d, degree - 1, mid);
          if ((flo < 0.0f) != (fm < 0.0f)) {
            hi = mid;
          } else {
            lo = mid;
            flo = fm;
          }
        }
        expand_coord(mn, mx, eval_poly(c, degree, 0.5f * (lo + hi)));
      }
      prev_u = u;
      prev_f = f;
    }
    return;
  }
}

__device__ float3 ribbon3_c1(const Ribbon3Data& r) {
  return mul3(sub3(r.b1, r.b0), 2.0f);
}

__device__ float3 ribbon3_c2(const Ribbon3Data& r) {
  return add3(sub3(r.b0, mul3(r.b1, 2.0f)), r.b2);
}

__device__ void ribbon2_edge_poly_coord(const Ribbon2Data& r, int coord, float sign, float out[4]) {
  const float q00 = component3(r.q00, coord);
  const float q01 = component3(r.q01, coord);
  const float q10 = component3(r.q10, coord);
  const float q11 = component3(r.q11, coord);
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

__device__ void ribbon3_generator_poly(const Ribbon3Data& r, float3 g[4]) {
  const float3 c1 = ribbon3_c1(r);
  const float3 c2 = ribbon3_c2(r);
  const float3 a = add3(mul3(c2, r.beta), mul3(c1, r.gamma));
  g[0] = r.bt0;
  g[1] = add3(sub3(mul3(r.bt1, r.alpha), r.bt0), a);
  g[2] = add3(mul3(a, -1.0f), mul3(c2, r.gamma));
  g[3] = mul3(c2, -r.gamma);
}

__device__ void ribbon3_edge_poly_coord(const Ribbon3Data& r, int coord, float sign, float out[6]) {
  const float3 c1 = ribbon3_c1(r);
  const float3 c2 = ribbon3_c2(r);
  float3 g[4];
  ribbon3_generator_poly(r, g);
  const float c[3] = {component3(r.b0, coord), component3(c1, coord), component3(c2, coord)};
  const float gc[4] = {
      component3(g[0], coord),
      component3(g[1], coord),
      component3(g[2], coord),
      component3(g[3], coord)};
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

__device__ void ribbon4_edge_poly_coord(const Ribbon4Data& r, int coord, float sign, int degree, float out[5]) {
  const float d[4] = {component3(r.d0, coord), component3(r.d1, coord), component3(r.d2, coord), component3(r.d3, coord)};
  const float g[3] = {component3(r.g0, coord), component3(r.g1, coord), component3(r.g2, coord)};
  const float w[3] = {r.w0, r.w1, r.w2};
  for (int i = 0; i < 5; ++i) out[i] = 0.0f;
  for (int i = 0; i <= degree && i < 4; ++i) out[i] = d[i];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i + j <= degree) out[i + j] += sign * w[i] * g[j];
    }
  }
}

__device__ void ribbon4c_edge_poly_coord(const Ribbon4Data& r, int coord, float sign, float out[3]) {
  out[0] = component3(r.d0, coord) + sign * component3(r.d1, coord);
  out[1] = component3(r.g0, coord) + sign * component3(r.g1, coord);
  out[2] = component3(r.d2, coord) + sign * component3(r.d3, coord);
}

__device__ float3 ribbon4_directrix(const Ribbon4Data& r, float u) {
  return add3(add3(mul3(r.d3, u * u * u), mul3(r.d2, u * u)), add3(mul3(r.d1, u), r.d0));
}

__device__ float3 ribbon4c_center(const Ribbon4Data& r, float v) {
  return add3(add3(r.d0, mul3(r.g0, v)), mul3(r.d2, v * v));
}

__device__ float3 ribbon4_generator(const Ribbon4Data& r, float u) {
  return add3(add3(mul3(r.g2, u * u), mul3(r.g1, u)), r.g0);
}

__device__ float3 ribbon2_point(const Ribbon2Data& r, float u, float v) {
  const float omu = 1.0f - u;
  const float omv = 1.0f - v;
  return add3(
      add3(mul3(r.q00, omu * omv), mul3(r.q01, omu * v)),
      add3(mul3(r.q10, u * omv), mul3(r.q11, u * v)));
}

__device__ float3 segment_start(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_2) return p.r2.q00;
  if (p.type == RIBBON_TYPE_3) return p.r3.b0;
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_center(p.r4, 0.0f);
  return ribbon4_directrix(p.r4, 0.0f);
}

__device__ float3 segment_end(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_2) return p.r2.q11;
  if (p.type == RIBBON_TYPE_3) return p.r3.b2;
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_center(p.r4, 1.0f);
  return ribbon4_directrix(p.r4, 1.0f);
}

__device__ float3 segment_mid(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_2) return ribbon2_point(p.r2, 0.5f, 0.5f);
  if (p.type == RIBBON_TYPE_3) return add3(mul3(p.r3.b0, 0.25f), add3(mul3(p.r3.b1, 0.5f), mul3(p.r3.b2, 0.25f)));
  if (p.type == RIBBON_TYPE_4C) return ribbon4c_center(p.r4, 0.5f);
  return ribbon4_directrix(p.r4, 0.5f);
}

__device__ bool is_device_motion_ribbon(const RibbonPrimitive& p) {
  if (p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A || p.type == RIBBON_TYPE_4B ||
      p.type == RIBBON_TYPE_4C) return true;
  return p.type == RIBBON_TYPE_2 && p.r2.q00.y > -0.02f;
}

__device__ bool is_device_grass_ribbon(const RibbonPrimitive& p) {
  if (!(p.type == RIBBON_TYPE_2 || p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A ||
        p.type == RIBBON_TYPE_4B || p.type == RIBBON_TYPE_4C)) return false;
  return segment_end(p).y - segment_start(p).y > 0.01f;
}

__device__ void add_quadratic_directrix_delta(Ribbon4Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = sub3(dm, d0);
  const float3 e = sub3(d1, d0);
  const float3 a = sub3(mul3(m, 4.0f), e);
  const float3 b = sub3(mul3(e, 2.0f), mul3(m, 4.0f));
  r.d0 = add3(r.d0, d0);
  r.d1 = add3(r.d1, a);
  r.d2 = add3(r.d2, b);
}

__device__ float3 quadratic_bezier_control_delta(const float3& d0, const float3& dm, const float3& d1) {
  return sub3(mul3(dm, 2.0f), mul3(add3(d0, d1), 0.5f));
}

__device__ void add_quadratic_ribbon2_center_delta(Ribbon2Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = sub3(dm, d0);
  const float3 e = sub3(d1, d0);
  const float3 a = sub3(mul3(m, 4.0f), e);
  const float3 b = sub3(mul3(e, 2.0f), mul3(m, 4.0f));
  r.q00 = add3(r.q00, d0);
  r.q01 = add3(r.q01, d0);
  r.q10 = add3(r.q10, add3(d0, a));
  r.q11 = add3(r.q11, add3(d0, add3(a, b)));
}

__device__ void add_quadratic_ribbon4c_center_delta(Ribbon4Data& r, const float3& d0, const float3& dm, const float3& d1) {
  const float3 m = sub3(dm, d0);
  const float3 e = sub3(d1, d0);
  r.d0 = add3(r.d0, d0);
  r.g0 = add3(r.g0, sub3(mul3(m, 4.0f), e));
  r.d2 = add3(r.d2, sub3(mul3(e, 2.0f), mul3(m, 4.0f)));
}

__device__ void tilt_ribbon4_generator(Ribbon4Data& r, const float3& d, float amount) {
  const float3 g1 = ribbon4_generator(r, 1.0f);
  const float3 target = normalize3(add3(g1, mul3(d, amount)));
  r.g1 = add3(r.g1, sub3(target, g1));
}

__device__ float3 grass_wind_displacement(const RibbonPrimitive& p, float time, float strength, float speed) {
  const float3 base = segment_start(p);
  const float height = fmaxf(0.05f, segment_end(p).y);
  const float phase = base.x * 2.2f + base.z * 3.1f + time * speed;
  const float gust = 0.55f + 0.45f * sinf(phase) + 0.20f * sinf(phase * 2.37f + 1.1f);
  const float amp = strength * height * gust;
  const float3 dir = normalize3(make_float3(0.82f, 0.0f, -0.38f));
  return mul3(dir, amp);
}

__device__ float3 grass_point_wind_displacement(const float3& p, float time, float strength, float speed) {
  const float height = fmaxf(0.0f, p.y);
  const float phase = p.x * 2.2f + p.z * 3.1f + p.y * 0.35f + time * speed;
  const float gust = 0.55f + 0.45f * sinf(phase) + 0.20f * sinf(phase * 2.37f + 1.1f);
  const float3 dir = normalize3(make_float3(0.82f, 0.0f, -0.38f));
  float3 d = mul3(dir, strength * height * gust);
  d.y = -length3(d) * 0.055f;
  return d;
}

__device__ float3 strip_point_wind_displacement(const float3& p, float time, float strength, float speed) {
  const float height_factor = fmaxf(0.0f, p.y);
  const float phase = p.x * 1.6f + p.z * 2.1f + p.y * 0.75f + time * speed;
  const float gust = 0.60f + 0.40f * sinf(phase) + 0.16f * sinf(phase * 1.91f + 0.7f);
  const float3 dir = normalize3(make_float3(0.78f, 0.0f, -0.44f));
  float3 d = mul3(dir, strength * height_factor * gust);
  d.y = -length3(d) * 0.06f;
  return d;
}

__global__ void animate_primitives_kernel(
    const RibbonPrimitive* base_primitives,
    RibbonPrimitive* primitives,
    unsigned count,
    DeviceWindMode mode,
    float time,
    float strength,
    float speed) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  RibbonPrimitive p = base_primitives[idx];
  if (fabsf(strength) < 1e-8f) {
    primitives[idx] = p;
    return;
  }
  if (mode == DeviceWindMode::Grass) {
    if (is_device_grass_ribbon(p)) {
      const float3 p0 = segment_start(p);
      const float3 pm = segment_mid(p);
      const float3 p1 = segment_end(p);
      const float3 d0 = grass_point_wind_displacement(p0, time, strength, speed);
      const float3 dm = grass_point_wind_displacement(pm, time, strength, speed);
      const float3 d1 = grass_point_wind_displacement(p1, time, strength, speed);
      if (p.type == RIBBON_TYPE_2) {
        add_quadratic_ribbon2_center_delta(p.r2, d0, dm, d1);
      } else if (p.type == RIBBON_TYPE_3) {
        p.r3.b0 = add3(p.r3.b0, d0);
        p.r3.b1 = add3(p.r3.b1, quadratic_bezier_control_delta(d0, dm, d1));
        p.r3.b2 = add3(p.r3.b2, d1);
        p.r3.bt1 = normalize3(add3(p.r3.bt1, mul3(d1, 0.10f)));
      } else if (p.type == RIBBON_TYPE_4C) {
        add_quadratic_ribbon4c_center_delta(p.r4, d0, dm, d1);
      } else {
        add_quadratic_directrix_delta(p.r4, d0, dm, d1);
        tilt_ribbon4_generator(p.r4, d1, 0.10f);
      }
    }
  } else if (mode == DeviceWindMode::StripIndependent) {
    if (!is_device_motion_ribbon(p)) {
    } else if (p.type == RIBBON_TYPE_2) {
      const float3 d = strip_point_wind_displacement(p.r2.q00, time, strength, speed);
      p.r2.q01 = add3(p.r2.q01, mul3(d, 0.50f));
      p.r2.q10 = add3(p.r2.q10, mul3(d, 0.50f));
      p.r2.q11 = add3(p.r2.q11, d);
    } else if (p.type == RIBBON_TYPE_3) {
      const float3 d = strip_point_wind_displacement(p.r3.b0, time, strength, speed);
      p.r3.b1 = add3(p.r3.b1, mul3(d, 0.50f));
      p.r3.b2 = add3(p.r3.b2, d);
      p.r3.bt1 = normalize3(add3(p.r3.bt1, mul3(d, 0.08f)));
    } else if (p.type == RIBBON_TYPE_4C) {
      const float3 d = strip_point_wind_displacement(segment_start(p), time, strength, speed);
      p.r4.g0 = add3(p.r4.g0, mul3(d, 0.50f));
      p.r4.d2 = add3(p.r4.d2, d);
    } else {
      const float3 d = strip_point_wind_displacement(segment_start(p), time, strength, speed);
      add_quadratic_directrix_delta(p.r4, make_float3(0.0f, 0.0f, 0.0f), mul3(d, 0.50f), d);
      tilt_ribbon4_generator(p.r4, d, 0.08f);
    }
  } else {
    if (!is_device_motion_ribbon(p)) {
    } else if (p.type == RIBBON_TYPE_2) {
      p.r2.q00 = add3(p.r2.q00, strip_point_wind_displacement(p.r2.q00, time, strength, speed));
      p.r2.q01 = add3(p.r2.q01, strip_point_wind_displacement(p.r2.q01, time, strength, speed));
      p.r2.q10 = add3(p.r2.q10, strip_point_wind_displacement(p.r2.q10, time, strength, speed));
      p.r2.q11 = add3(p.r2.q11, strip_point_wind_displacement(p.r2.q11, time, strength, speed));
    } else if (p.type == RIBBON_TYPE_3) {
      p.r3.b0 = add3(p.r3.b0, strip_point_wind_displacement(p.r3.b0, time, strength, speed));
      p.r3.b1 = add3(p.r3.b1, strip_point_wind_displacement(p.r3.b1, time, strength, speed));
      p.r3.b2 = add3(p.r3.b2, strip_point_wind_displacement(p.r3.b2, time, strength, speed));
    } else if (p.type == RIBBON_TYPE_4C) {
      const float3 p0 = ribbon4c_center(p.r4, 0.0f);
      const float3 pm = ribbon4c_center(p.r4, 0.5f);
      const float3 p1 = ribbon4c_center(p.r4, 1.0f);
      const float3 d0 = strip_point_wind_displacement(p0, time, strength, speed);
      const float3 dm = strip_point_wind_displacement(pm, time, strength, speed);
      const float3 d1 = strip_point_wind_displacement(p1, time, strength, speed);
      const float3 m = sub3(dm, d0);
      const float3 e = sub3(d1, d0);
      p.r4.d0 = add3(p.r4.d0, d0);
      p.r4.g0 = add3(p.r4.g0, sub3(mul3(m, 4.0f), e));
      p.r4.d2 = add3(p.r4.d2, sub3(mul3(e, 2.0f), mul3(m, 4.0f)));
    } else {
      const float3 p0 = ribbon4_directrix(p.r4, 0.0f);
      const float3 pm = ribbon4_directrix(p.r4, 0.5f);
      const float3 p1 = ribbon4_directrix(p.r4, 1.0f);
      add_quadratic_directrix_delta(
          p.r4,
          strip_point_wind_displacement(p0, time, strength, speed),
          strip_point_wind_displacement(pm, time, strength, speed),
          strip_point_wind_displacement(p1, time, strength, speed));
    }
  }

  primitives[idx] = p;
}

__global__ void compute_aabbs_kernel(const RibbonPrimitive* primitives, OptixAabb* aabbs, unsigned count) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  const RibbonPrimitive p = primitives[idx];
  float3 mn = make_float3(1e30f, 1e30f, 1e30f);
  float3 mx = make_float3(-1e30f, -1e30f, -1e30f);

  auto expand_axis = [&](int coord, float lo, float hi) {
    if (coord == 0) {
      mn.x = fminf(mn.x, lo);
      mx.x = fmaxf(mx.x, hi);
    } else if (coord == 1) {
      mn.y = fminf(mn.y, lo);
      mx.y = fmaxf(mx.y, hi);
    } else {
      mn.z = fminf(mn.z, lo);
      mx.z = fmaxf(mx.z, hi);
    }
  };

  if (p.type == RIBBON_TYPE_2) {
    for (int side = 0; side < 2; ++side) {
      const float sign = side == 0 ? -1.0f : 1.0f;
      for (int coord = 0; coord < 3; ++coord) {
        float c[4];
        ribbon2_edge_poly_coord(p.r2, coord, sign, c);
        float lo = 1e30f;
        float hi = -1e30f;
        expand_poly_extrema(lo, hi, c, 3);
        expand_axis(coord, lo, hi);
      }
    }
  } else if (p.type == RIBBON_TYPE_3) {
    for (int side = 0; side < 2; ++side) {
      const float sign = side == 0 ? -1.0f : 1.0f;
      for (int coord = 0; coord < 3; ++coord) {
        float c[6];
        ribbon3_edge_poly_coord(p.r3, coord, sign, c);
        float lo = 1e30f;
        float hi = -1e30f;
        expand_poly_extrema(lo, hi, c, 5);
        expand_axis(coord, lo, hi);
      }
    }
  } else if (p.type == RIBBON_TYPE_4A || p.type == RIBBON_TYPE_4B) {
    const int degree = p.type == RIBBON_TYPE_4A ? 3 : 4;
    for (int side = 0; side < 2; ++side) {
      const float sign = side == 0 ? -1.0f : 1.0f;
      for (int coord = 0; coord < 3; ++coord) {
        float c[5];
        ribbon4_edge_poly_coord(p.r4, coord, sign, degree, c);
        float lo = 1e30f;
        float hi = -1e30f;
        expand_poly_extrema(lo, hi, c, degree);
        expand_axis(coord, lo, hi);
      }
    }
  } else if (p.type == RIBBON_TYPE_4C) {
    for (int side = 0; side < 2; ++side) {
      const float sign = side == 0 ? -1.0f : 1.0f;
      for (int coord = 0; coord < 3; ++coord) {
        float c[3];
        ribbon4c_edge_poly_coord(p.r4, coord, sign, c);
        float lo = 1e30f;
        float hi = -1e30f;
        expand_poly_extrema(lo, hi, c, 2);
        expand_axis(coord, lo, hi);
      }
    }
  }

  constexpr float pad = 1e-3f;
  aabbs[idx] = OptixAabb{
      mn.x - pad, mn.y - pad, mn.z - pad,
      mx.x + pad, mx.y + pad, mx.z + pad};
}

}  // namespace

void launch_animate_primitives_kernel(
    const RibbonPrimitive* d_base_primitives,
    RibbonPrimitive* d_primitives,
    unsigned count,
    DeviceWindMode mode,
    float time,
    float strength,
    float speed,
    cudaStream_t stream) {
  const unsigned block_size = 128;
  const unsigned block_count = (count + block_size - 1) / block_size;
  animate_primitives_kernel<<<block_count, block_size, 0, stream>>>(
      d_base_primitives, d_primitives, count, mode, time, strength, speed);
}

void launch_compute_aabbs_kernel(
    const RibbonPrimitive* d_primitives,
    OptixAabb* d_aabbs,
    unsigned count,
    cudaStream_t stream) {
  const unsigned block_size = 128;
  const unsigned block_count = (count + block_size - 1) / block_size;
  compute_aabbs_kernel<<<block_count, block_size, 0, stream>>>(d_primitives, d_aabbs, count);
}
