#include <optix.h>
#include <optix_device.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "shared/launch_params.h"

extern "C" {
__constant__ LaunchParams params;
}

static __forceinline__ __device__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __forceinline__ __device__ float3 operator-(const float3& a, const float3& b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __forceinline__ __device__ float3 operator*(const float3& a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
static __forceinline__ __device__ float3 operator*(float s, const float3& a) { return a * s; }
static __forceinline__ __device__ float3 operator/(const float3& a, float s) { return f3(a.x / s, a.y / s, a.z / s); }
static __forceinline__ __device__ float dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __forceinline__ __device__ float3 cross3(const float3& a, const float3& b) {
  return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static __forceinline__ __device__ float3 normalize3(const float3& a) {
  float l2 = fmaxf(1e-20f, dot3(a, a));
  return a * rsqrtf(l2);
}

static __forceinline__ __device__ float eval_quad(float w0, float w1, float w2, float u) {
  return w0 + w1 * u + w2 * u * u;
}

static __forceinline__ __device__ float eval_poly2_asc(const float q[3], float u) {
  return q[0] + q[1] * u + q[2] * u * u;
}

static __forceinline__ __device__ float eval_poly3_asc(const float q[4], float u) {
  return ((q[3] * u + q[2]) * u + q[1]) * u + q[0];
}

static __forceinline__ __device__ float eval_poly3_derivative_asc(const float q[4], float u) {
  return (3.0f * q[3] * u + 2.0f * q[2]) * u + q[1];
}

static __forceinline__ __device__ int solve_quadratic_interval(const float q[3], float out[2], float t0, float t1) {
  const float a = q[2];
  const float b = q[1];
  const float c = q[0];
  const float eps = 1e-10f;

  if (fabsf(a) < eps) {
    if (fabsf(b) < eps) return 0;
    float r = -c / b;
    if (r >= t0 && r <= t1) {
      out[0] = r;
      return 1;
    }
    return 0;
  }

  float disc = b * b - 4.0f * a * c;
  if (disc < 0.0f) return 0;
  float s = sqrtf(fmaxf(disc, 0.0f));
  float qv = -0.5f * (b + copysignf(s, b));
  float r0 = qv / a;
  float r1 = c / qv;
  if (r0 > r1) {
    float tmp = r0;
    r0 = r1;
    r1 = tmp;
  }
  int n = 0;
  if (r0 >= t0 && r0 <= t1) out[n++] = r0;
  if (r1 >= t0 && r1 <= t1 && fabsf(r1 - r0) > 1e-6f) out[n++] = r1;
  return n;
}

static __forceinline__ __device__ int solve_cubic_cardano(float a, float b, float c, float d, float out[3]) {
  const float eps = 1e-10f;
  if (fabsf(a) < eps) {
    float q[3] = {d, c, b};
    return solve_quadratic_interval(q, out, -1e30f, 1e30f);
  }

  float inva = 1.0f / a;
  float A = b * inva;
  float B = c * inva;
  float C = d * inva;
  float sq_A = A * A;
  float p = (1.0f / 3.0f) * (-1.0f / 3.0f * sq_A + B);
  float q = (1.0f / 2.0f) * ((2.0f / 27.0f) * A * sq_A - (1.0f / 3.0f) * A * B + C);
  float D = q * q + p * p * p;

  int n = 0;
  if (D > eps) {
    float sqrtD = sqrtf(D);
    float u = cbrtf(-q + sqrtD);
    float v = cbrtf(-q - sqrtD);
    out[n++] = u + v - A / 3.0f;
  } else if (fabsf(D) <= eps) {
    float u = cbrtf(-q);
    out[n++] = 2.0f * u - A / 3.0f;
    out[n++] = -u - A / 3.0f;
  } else {
    float phi = acosf(fmaxf(-1.0f, fminf(1.0f, -q / sqrtf(-(p * p * p)))));
    float t = 2.0f * sqrtf(-p);
    out[n++] = t * cosf(phi / 3.0f) - A / 3.0f;
    out[n++] = t * cosf((phi + 2.0f * 3.1415926535f) / 3.0f) - A / 3.0f;
    out[n++] = t * cosf((phi + 4.0f * 3.1415926535f) / 3.0f) - A / 3.0f;
  }

  for (int i = 0; i < n; ++i) {
    for (int it = 0; it < 2; ++it) {
      float f = ((a * out[i] + b) * out[i] + c) * out[i] + d;
      float fp = (3.0f * a * out[i] + 2.0f * b) * out[i] + c;
      if (fabsf(fp) > 1e-12f) out[i] -= f / fp;
    }
  }
  return n;
}

static __forceinline__ __device__ int solve_cubic_interval_asc(const float q[4], float out[3], float t0, float t1) {
  const float eps = 1e-6f;
  float roots_direct[3];
  int n_direct = solve_cubic_cardano(q[3], q[2], q[1], q[0], roots_direct);

  float direct_cauchy = 1.0f + fmaxf(fabsf(q[2] / q[3]), fmaxf(fabsf(q[1] / q[3]), fabsf(q[0] / q[3])));
  float inverse_cauchy = 1e30f;
  float roots_inv_u[3];
  int n_inv = 0;
  if (fabsf(q[0]) > 1e-12f) {
    float inv_poly[4] = {q[3], q[2], q[1], q[0]};
    float roots_inv[3];
    n_inv = solve_cubic_cardano(inv_poly[3], inv_poly[2], inv_poly[1], inv_poly[0], roots_inv);
    for (int i = 0; i < n_inv; ++i) {
      roots_inv_u[i] = (fabsf(roots_inv[i]) > 1e-12f) ? 1.0f / roots_inv[i] : 1e30f;
    }
    inverse_cauchy = 1.0f + fmaxf(fabsf(q[1] / q[0]), fmaxf(fabsf(q[2] / q[0]), fabsf(q[3] / q[0])));
  }

  const float* candidates = roots_direct;
  int n = n_direct;
  if (n_inv > 0 && inverse_cauchy < direct_cauchy) {
    candidates = roots_inv_u;
    n = n_inv;
  }

  int out_n = 0;
  for (int i = 0; i < n; ++i) {
    float r = candidates[i];
    if (!(r >= t0 - eps && r <= t1 + eps)) continue;
    float clamped = fminf(t1, fmaxf(t0, r));
    float f = eval_poly3_asc(q, clamped);
    float fp = eval_poly3_derivative_asc(q, clamped);
    if (fabsf(fp) > 1e-12f) {
      float refined = clamped - f / fp;
      if (refined >= t0 - eps && refined <= t1 + eps) clamped = fminf(t1, fmaxf(t0, refined));
    }
    if (fabsf(eval_poly3_asc(q, clamped)) > 1e-4f) continue;
    bool unique = true;
    for (int j = 0; j < out_n; ++j) {
      if (fabsf(out[j] - clamped) < 1e-4f) {
        unique = false;
        break;
      }
    }
    if (unique) out[out_n++] = clamped;
  }
  return out_n;
}

static __forceinline__ __device__ float3 eval_ribbon2(const Ribbon2Data& r, float u, float v) {
  float omu = 1.0f - u;
  float omv = 1.0f - v;
  return r.q00 * (omu * omv) + r.q01 * (omu * v) + r.q10 * (u * omv) + r.q11 * (u * v);
}

static __forceinline__ __device__ float3 ribbon3_curve(const Ribbon3Data& r, float u) {
  float omu = 1.0f - u;
  return r.b0 * (omu * omu) + r.b1 * (2.0f * u * omu) + r.b2 * (u * u);
}

static __forceinline__ __device__ float3 ribbon3_c1(const Ribbon3Data& r) {
  return 2.0f * (r.b1 - r.b0);
}

static __forceinline__ __device__ float3 ribbon3_c2(const Ribbon3Data& r) {
  return r.b0 - 2.0f * r.b1 + r.b2;
}

static __forceinline__ __device__ float3 ribbon3_velocity(const Ribbon3Data& r, float u) {
  return ribbon3_c1(r) + 2.0f * ribbon3_c2(r) * u;
}

static __forceinline__ __device__ float3 ribbon3_bitangent(const Ribbon3Data& r, float u) {
  float3 c1 = ribbon3_c1(r);
  float3 c2 = ribbon3_c2(r);
  float3 base = r.bt0 * (1.0f - u) + (r.alpha * r.bt1) * u;
  float3 e = (r.beta * c2 + r.gamma * (c1 + c2 * u)) * (u * (1.0f - u));
  return base + e;
}

static __forceinline__ __device__ bool intersect_ribbon2(
    const Ribbon2Data& r,
    const float3& ro,
    const float3& rd,
    float tmin) {
  float3 D0 = r.q00 - ro;
  float3 D1 = r.q10 - r.q00;
  float3 G0 = r.q01 - r.q00;
  float3 G1 = r.q00 - r.q01 + r.q11 - r.q10;

  float3 X0 = cross3(G0, rd);
  float3 X1 = cross3(G1, rd);

  float q[3] = {
      dot3(D0, X0),
      dot3(D0, X1) + dot3(D1, X0),
      dot3(D1, X1)};

  float roots[2];
  int n_roots = solve_quadratic_interval(q, roots, 0.0f, 1.0f);
  bool hit = false;

  for (int i = 0; i < n_roots; ++i) {
    float u = roots[i];
    float3 D = r.q00 + (r.q10 - r.q00) * u;
    float3 G = G0 + G1 * u;
    float3 p = D - ro;
    float a = dot3(rd, rd);
    float b = dot3(rd, G);
    float c = dot3(G, G);
    float d = dot3(rd, p);
    float e = dot3(G, p);
    float den = a * c - b * b;
    if (fabsf(den) < 1e-10f) continue;
    float v = (b * d - a * e) / den;
    float t = (d + b * v) / a;
    if (t <= tmin) continue;
    float w = fabsf(eval_quad(r.w0, r.w1, r.w2, u));
    if (fabsf(u - v) > w) continue;
    if (optixReportIntersection(t, 0, __float_as_uint(u), __float_as_uint(v))) hit = true;
  }
  return hit;
}

static __forceinline__ __device__ bool intersect_ribbon3(
    const Ribbon3Data& r,
    const float3& ro,
    const float3& rd,
    float tmin) {
  const float3 C0 = r.b0 - ro;
  const float3 C1 = ribbon3_c1(r);
  const float3 C2 = ribbon3_c2(r);

  const float3 bt1s = r.alpha * r.bt1;
  float3 xn0 = cross3(rd, r.bt0);
  float3 xn1 = cross3(rd, bt1s - r.bt0);
  float3 rc2 = cross3(rd, C2);
  float3 rc0 = cross3(C0, rd);
  float gb = dot3(C1, rc0) * r.gamma + dot3(C0, rc2) * r.beta;
  float bg = dot3(C1, rc2) * r.beta + dot3(C0, rc2) * r.gamma;

  float q[4] = {
      dot3(C0, xn0),
      dot3(C0, xn1) + dot3(C1, xn0) + gb,
      dot3(C1, xn1) + dot3(C2, xn0) + bg - gb,
      dot3(C2, xn1) - bg};

  float roots[3];
  int n_roots = solve_cubic_interval_asc(q, roots, 0.0f, 1.0f);
  const float epsilon = 1e-5f;
  bool hit = false;

  for (int i = 0; i < n_roots; ++i) {
    float u = roots[i];
    float y0 = eval_poly3_asc(q, u);
    if (fabsf(y0) > epsilon) continue;

    float w = fabsf(eval_quad(r.w0, r.w1, r.w2, u));
    float3 bn = normalize3(ribbon3_bitangent(r, u));
    float3 C = ribbon3_curve(r, u);
    float3 sn = normalize3(cross3(ribbon3_velocity(r, u), bn));
    float ws = (dot3(bn, rd) > 0.0f) ? -1.0f : 1.0f;
    float3 pn = ((dot3(sn, rd) > 0.0f) ? -1.0f : 1.0f) * sn + ws * bn;

    float3 A = C + w * bn;
    float3 B = C - w * bn;
    float den = dot3(pn, rd);
    if (fabsf(den) < 1e-10f) continue;
    float num = dot3(pn, C);
    float tn = (num - ws * w) / den;
    float tp = (num + ws * w) / den;
    float dn = dot3((ro + tn * rd) - B, sn);
    float dp = dot3((ro + tp * rd) - A, sn);
    if ((dn > 0.0f) == (dp > 0.0f)) continue;
    float vi = dn / (dn - dp);
    float ti = tn + (tp - tn) * vi;
    if (ti <= tmin) continue;
    float v = 2.0f * vi - 1.0f;
    if (optixReportIntersection(ti - epsilon * 0.5f, 0, __float_as_uint(u), __float_as_uint(v))) hit = true;
  }
  return hit;
}

static __forceinline__ __device__ unsigned int pack_color(const float3& c) {
  unsigned int r = static_cast<unsigned int>(fminf(255.0f, fmaxf(0.0f, c.x * 255.0f)));
  unsigned int g = static_cast<unsigned int>(fminf(255.0f, fmaxf(0.0f, c.y * 255.0f)));
  unsigned int b = static_cast<unsigned int>(fminf(255.0f, fmaxf(0.0f, c.z * 255.0f)));
  return (255u << 24) | (b << 16) | (g << 8) | r;
}

static __forceinline__ __device__ float3 unpack_color(unsigned int p) {
  float r = static_cast<float>(p & 0xFFu) / 255.0f;
  float g = static_cast<float>((p >> 8) & 0xFFu) / 255.0f;
  float b = static_cast<float>((p >> 16) & 0xFFu) / 255.0f;
  return f3(r, g, b);
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  if (idx.x >= params.width || idx.y >= params.height) return;

  const float2 d = make_float2(
      (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x),
      (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y));
  float3 ro = params.camera.origin;
  float3 rd = normalize3(
      params.camera.lower_left + d.x * params.camera.horizontal + d.y * params.camera.vertical);

  unsigned int p0 = pack_color(f3(0.0f, 0.0f, 0.0f));
  optixTrace(
      params.gas,
      ro,
      rd,
      0.0f,
      1e16f,
      0.0f,
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,
      0,
      1,
      0,
      p0);

  float3 c = unpack_color(p0);
  const unsigned int pixel = idx.y * params.width + idx.x;
  params.image[pixel] = make_uchar4(
      static_cast<unsigned char>(255.0f * c.x),
      static_cast<unsigned char>(255.0f * c.y),
      static_cast<unsigned char>(255.0f * c.z),
      255);
}

extern "C" __global__ void __miss__ms() {
  const float3 d = normalize3(optixGetWorldRayDirection());
  const float t = 0.5f * (d.y + 1.0f);
  const float3 c = (1.0f - t) * f3(1.0f, 1.0f, 1.0f) + t * f3(0.6f, 0.8f, 1.0f);
  optixSetPayload_0(pack_color(c));
}

extern "C" __global__ void __closesthit__ch() {
  unsigned int prim_idx = optixGetPrimitiveIndex();
  if (prim_idx >= params.primitive_count) {
    optixSetPayload_0(pack_color(f3(1.0f, 0.0f, 1.0f)));
    return;
  }
  RibbonPrimitive p = params.primitives[prim_idx];
  float u = __uint_as_float(optixGetAttribute_0());
  float v = __uint_as_float(optixGetAttribute_1());

  float3 n = f3(0.0f, 1.0f, 0.0f);
  float3 albedo = f3(0.9f, 0.9f, 0.9f);
  if (p.type == RIBBON_TYPE_2) {
    const Ribbon2Data& r = p.r2;
    float3 dQdu = (1.0f - v) * (r.q10 - r.q00) + v * (r.q11 - r.q01);
    float3 dQdv = (1.0f - u) * (r.q01 - r.q00) + u * (r.q11 - r.q10);
    n = normalize3(cross3(dQdu, dQdv));
    albedo = f3(0.95f, 0.72f, 0.35f);
  } else {
    const Ribbon3Data& r = p.r3;
    float3 vel = ribbon3_velocity(r, u);
    float3 bn = normalize3(ribbon3_bitangent(r, u));
    n = normalize3(cross3(vel, bn));
    albedo = f3(0.35f, 0.72f, 0.95f);
  }

  float3 rd = normalize3(optixGetWorldRayDirection());
  if (dot3(n, rd) > 0.0f) n = n * -1.0f;
  float3 light_dir = normalize3(f3(0.6f, 1.0f, 0.4f));
  float nl = fmaxf(0.0f, dot3(n, light_dir));
  float3 c = albedo * (0.2f + 0.8f * nl);
  optixSetPayload_0(pack_color(c));
}

extern "C" __global__ void __intersection__ribbon() {
  unsigned int prim_idx = optixGetPrimitiveIndex();
  if (prim_idx >= params.primitive_count) return;
  RibbonPrimitive p = params.primitives[prim_idx];
  float3 ro = optixGetObjectRayOrigin();
  float3 rd = optixGetObjectRayDirection();
  float tmin = optixGetRayTmin();

  if (p.type == RIBBON_TYPE_2) {
    intersect_ribbon2(p.r2, ro, rd, tmin);
  } else if (p.type == RIBBON_TYPE_3) {
    intersect_ribbon3(p.r3, ro, rd, tmin);
  }
}

