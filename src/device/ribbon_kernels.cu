#include <optix.h>
#include <optix_device.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "shared/launch_params.h"

extern "C" {
__constant__ LaunchParams params;
}

struct HitInfo {
  int hit;
  float3 position;
  float3 normal;
  float3 albedo;
  float depth;
  float emission;
  float metallic;
  float roughness;
  int material_type;
  float ior;
  float3 env;
};

struct BsdfSample {
  float3 wi;
  float3 f_over_pdf;
  float pdf;
  int is_delta;
};

static __forceinline__ __device__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __forceinline__ __device__ float3 operator-(const float3& a, const float3& b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __forceinline__ __device__ float3 operator-(const float3& a) { return f3(-a.x, -a.y, -a.z); }
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
static __forceinline__ __device__ float3 lerp3(const float3& a, const float3& b, float t) { return a * (1.0f - t) + b * t; }
static __forceinline__ __device__ float3 mul3(const float3& a, const float3& b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __forceinline__ __device__ float clamp01(float v) { return fminf(1.0f, fmaxf(0.0f, v)); }
static __forceinline__ __device__ float length3(const float3& a) { return sqrtf(dot3(a, a)); }
static __forceinline__ __device__ float3 clamp3max(const float3& a, float mx) {
  return f3(fminf(a.x, mx), fminf(a.y, mx), fminf(a.z, mx));
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

static __forceinline__ __device__ void* unpack_ptr(unsigned int i0, unsigned int i1) {
  const uint64_t uptr = (static_cast<uint64_t>(i0) << 32) | static_cast<uint64_t>(i1);
  return reinterpret_cast<void*>(uptr);
}

static __forceinline__ __device__ void pack_ptr(void* ptr, unsigned int& i0, unsigned int& i1) {
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = static_cast<unsigned int>(uptr >> 32);
  i1 = static_cast<unsigned int>(uptr & 0x00000000ffffffff);
}

static __forceinline__ __device__ unsigned int lcg(unsigned int& state) {
  state = 1664525u * state + 1013904223u;
  return state;
}

static __forceinline__ __device__ float randf(unsigned int& state) {
  return static_cast<float>(lcg(state) & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

static __forceinline__ __device__ float3 sample_cosine_hemisphere(unsigned int& seed) {
  const float u1 = randf(seed);
  const float u2 = randf(seed);
  const float r = sqrtf(u1);
  const float phi = 2.0f * 3.1415926535f * u2;
  return f3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

static __forceinline__ __device__ void make_onb(const float3& n, float3& b1, float3& b2);
static __forceinline__ __device__ float3 to_world(const float3& local, const float3& n);

static __forceinline__ __device__ float lambert_pdf(const float3& n, const float3& wi) {
  const float c = fmaxf(0.0f, dot3(n, wi));
  return c * (1.0f / 3.1415926535f);
}

static __forceinline__ __device__ float3 lambert_f(const float3& albedo) {
  return albedo * (1.0f / 3.1415926535f);
}

static __forceinline__ __device__ float3 metal_spec_f(
    const float3& albedo,
    const float3& n,
    const float3& wo,
    const float3& wi,
    float roughness) {
  // Lightweight normalized Phong lobe as a practical direct-light term.
  const float r = clamp01(roughness);
  const float shininess = fmaxf(1.0f, 2.0f / fmaxf(1e-3f, r * r) - 2.0f);
  const float3 refl = normalize3(-wo + 2.0f * dot3(wo, n) * n);
  const float cos_rl = fmaxf(0.0f, dot3(refl, wi));
  const float norm = (shininess + 2.0f) * (1.0f / (2.0f * 3.1415926535f));
  return albedo * (norm * powf(cos_rl, shininess));
}

static __forceinline__ __device__ float metal_shininess(float roughness) {
  const float r = clamp01(roughness);
  return fmaxf(1.0f, 2.0f / fmaxf(1e-3f, r * r) - 2.0f);
}

static __forceinline__ __device__ float metal_spec_pdf(
    const float3& n,
    const float3& wo,
    const float3& wi,
    float roughness) {
  const float3 refl = normalize3(-wo + 2.0f * dot3(wo, n) * n);
  const float cos_rl = fmaxf(0.0f, dot3(refl, wi));
  const float s = metal_shininess(roughness);
  return (s + 1.0f) * (1.0f / (2.0f * 3.1415926535f)) * powf(cos_rl, s);
}

static __forceinline__ __device__ float balance_heuristic(float pdf_a, float pdf_b) {
  const float a = fmaxf(0.0f, pdf_a);
  const float b = fmaxf(0.0f, pdf_b);
  return a / fmaxf(1e-8f, a + b);
}

static __forceinline__ __device__ float schlick(float cos_theta, float eta_i, float eta_t) {
  float r0 = (eta_i - eta_t) / (eta_i + eta_t);
  r0 = r0 * r0;
  float m = 1.0f - cos_theta;
  return r0 + (1.0f - r0) * m * m * m * m * m;
}

static __forceinline__ __device__ bool refract_dir(
    const float3& wi, const float3& n, float eta, float3& wt) {
  const float cosi = fmaxf(-1.0f, fminf(1.0f, dot3(wi, n)));
  const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
  if (k < 0.0f) return false;
  wt = normalize3(eta * (-wi) + (eta * cosi - sqrtf(k)) * n);
  return true;
}

static __forceinline__ __device__ BsdfSample sample_bsdf(
    const HitInfo& hit,
    const float3& n,
    const float3& wo,
    unsigned int& seed) {
  BsdfSample s = {};
  s.pdf = 1.0f;
  s.is_delta = 0;
  const int mtype = hit.material_type;

  if (mtype == 2) {
    // Dielectric (delta)
    float3 nn = n;
    float eta_i = 1.0f;
    float eta_t = fmaxf(1.0001f, hit.ior);
    float cosi = dot3(wo, nn);
    if (cosi < 0.0f) {
      cosi = -cosi;
      nn = -nn;
      float tmp = eta_i;
      eta_i = eta_t;
      eta_t = tmp;
    }
    const float eta = eta_i / eta_t;
    float3 refr_d = f3(0.0f, 0.0f, 0.0f);
    const bool ok = refract_dir(wo, nn, eta, refr_d);
    const float Fr = ok ? schlick(cosi, eta_i, eta_t) : 1.0f;
    if (randf(seed) < Fr || !ok) {
      s.wi = normalize3(-wo + 2.0f * dot3(wo, nn) * nn);
    } else {
      s.wi = refr_d;
    }
    s.f_over_pdf = hit.albedo;
    s.is_delta = 1;
    return s;
  }

  if (mtype == 1) {
    // Metal
    float3 refl = normalize3(-wo + 2.0f * dot3(wo, n) * n);
    float rough = clamp01(hit.roughness);
    float shininess = metal_shininess(rough);
    if (rough < 1e-3f) {
      s.wi = refl;
      s.is_delta = 1;
      s.pdf = 1.0f;
      s.f_over_pdf = hit.albedo;
      return s;
    }
    // Sample normalized Phong lobe around perfect reflection direction.
    const float u1 = randf(seed);
    const float u2 = randf(seed);
    const float cos_theta = powf(u1, 1.0f / (shininess + 1.0f));
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = 2.0f * 3.1415926535f * u2;
    float3 t, b;
    make_onb(refl, t, b);
    s.wi = normalize3(t * (sin_theta * cosf(phi)) + b * (sin_theta * sinf(phi)) + refl * cos_theta);
    if (dot3(n, s.wi) <= 0.0f) {
      // Fallback keeps transport stable when lobe samples below the surface.
      s.wi = refl;
      s.is_delta = 1;
      s.pdf = 1.0f;
      s.f_over_pdf = hit.albedo;
      return s;
    }
    const float pdf = metal_spec_pdf(n, wo, s.wi, rough);
    const float3 f = metal_spec_f(hit.albedo, n, wo, s.wi, rough);
    const float cos_i = fmaxf(0.0f, dot3(n, s.wi));
    s.pdf = fmaxf(1e-6f, pdf);
    s.f_over_pdf = f * (cos_i / s.pdf);
    s.is_delta = 0;
    return s;
  }

  // Diffuse
  s.wi = to_world(sample_cosine_hemisphere(seed), n);
  s.pdf = fmaxf(1e-6f, lambert_pdf(n, s.wi));
  s.f_over_pdf = hit.albedo;  // for cosine-weighted diffuse sampling
  s.is_delta = 0;
  return s;
}

static __forceinline__ __device__ void make_onb(const float3& n, float3& b1, float3& b2) {
  if (fabsf(n.z) < 0.999f) {
    b1 = normalize3(cross3(n, f3(0.0f, 0.0f, 1.0f)));
  } else {
    b1 = normalize3(cross3(n, f3(0.0f, 1.0f, 0.0f)));
  }
  b2 = cross3(n, b1);
}

static __forceinline__ __device__ float3 to_world(const float3& local, const float3& n) {
  float3 t, b;
  make_onb(n, t, b);
  return normalize3(t * local.x + b * local.y + n * local.z);
}

static __forceinline__ __device__ float3 environment_color(const float3& rd) {
  const float t = 0.5f * (rd.y + 1.0f);
  return (1.0f - t) * f3(1.0f, 1.0f, 1.0f) + t * f3(0.6f, 0.8f, 1.0f);
}

static __forceinline__ __device__ bool trace_shadow(const float3& ro, const float3& rd, float tmax) {
  unsigned int occluded = 0;
  unsigned int dummy = 0;
  optixTrace(
      params.gas,
      ro,
      rd,
      1e-3f,
      tmax,
      0.0f,
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      1,
      2,
      1,
      occluded,
      dummy);
  return occluded != 0;
}

static __forceinline__ __device__ float3 sample_direct_lighting(
    const HitInfo& hit, const float3& n, const float3& wo, unsigned int& seed) {
  float3 direct = f3(0.0f, 0.0f, 0.0f);
  for (unsigned i = 0; i < params.light_count; ++i) {
    const LightData l = params.lights[i];
    float3 L = f3(0.0f, 0.0f, 0.0f);
    float3 Li = f3(0.0f, 0.0f, 0.0f);
    float tmax = 1e16f;
    if (l.type == LIGHT_DIRECTIONAL) {
      L = normalize3(l.direction * -1.0f);
      Li = l.color * l.intensity;
    } else if (l.type == LIGHT_POINT) {
      float3 toL = l.position - hit.position;
      float dist = length3(toL);
      if (dist <= 1e-6f) continue;
      L = toL * (1.0f / dist);
      tmax = dist - 1e-3f;
      Li = l.color * (l.intensity / fmaxf(1e-6f, dist * dist));
    } else if (l.type == LIGHT_RECT) {
      const float su = randf(seed) * 2.0f - 1.0f;
      const float sv = randf(seed) * 2.0f - 1.0f;
      const float3 lp = l.position + su * l.u + sv * l.v;
      const float3 toL = lp - hit.position;
      const float dist = length3(toL);
      if (dist <= 1e-6f) continue;
      L = toL * (1.0f / dist);
      tmax = dist - 1e-3f;
      const float3 ln = normalize3(cross3(l.u, l.v));
      const float cos_l = fmaxf(0.0f, dot3(ln, -L));
      if (cos_l <= 1e-6f) continue;
      const float area = fmaxf(1e-6f, 4.0f * length3(cross3(l.u, l.v)));
      const float light_pdf = (dist * dist) / (cos_l * area);
      float bsdf_pdf = 0.0f;
      if (hit.material_type == 1) {
        bsdf_pdf = metal_spec_pdf(n, wo, L, hit.roughness);
      } else if (hit.material_type == 0) {
        bsdf_pdf = lambert_pdf(n, L);
      }
      const float w = balance_heuristic(light_pdf, bsdf_pdf);
      Li = l.color * l.intensity * w;
    } else {
      continue;
    }
    float ndotl = fmaxf(0.0f, dot3(n, L));
    if (ndotl <= 0.0f) continue;
    bool occ = trace_shadow(hit.position + 1e-3f * n, L, tmax);
    if (!occ) {
      if (hit.material_type == 1) {
        const float3 f = metal_spec_f(hit.albedo, n, wo, L, hit.roughness);
        direct = direct + mul3(f, Li * ndotl);
      } else if (hit.material_type == 0) {
        direct = direct + mul3(lambert_f(hit.albedo), Li * ndotl);
      }
    }
  }
  return direct;
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  if (idx.x >= params.width || idx.y >= params.height) return;

  const float2 d = make_float2(
      (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x),
      (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y));
  unsigned int seed = 9781u * (idx.x + 1u) + 6271u * (idx.y + 1u);
  float3 final_c = f3(0.0f, 0.0f, 0.0f);
  float3 aov_albedo_acc = f3(0.0f, 0.0f, 0.0f);
  float3 aov_normal_acc = f3(0.0f, 0.0f, 0.0f);
  float aov_depth_acc = 0.0f;
  float aov_depth_valid = 0.0f;
  const unsigned spp = params.spp > 0 ? params.spp : 1;
  const unsigned max_depth = params.max_depth > 0 ? params.max_depth : 1;

  for (unsigned s = 0; s < spp; ++s) {
    float jitter_x = randf(seed) - 0.5f;
    float jitter_y = randf(seed) - 0.5f;
    const float2 sd = make_float2(
        (static_cast<float>(idx.x) + 0.5f + jitter_x) / static_cast<float>(dim.x),
        (static_cast<float>(idx.y) + 0.5f + jitter_y) / static_cast<float>(dim.y));

    float3 ro = params.camera.origin;
    float3 rd = normalize3(
        params.camera.lower_left + sd.x * params.camera.horizontal + sd.y * params.camera.vertical);

    float3 throughput = f3(1.0f, 1.0f, 1.0f);
    float3 radiance = f3(0.0f, 0.0f, 0.0f);

    for (unsigned bounce = 0; bounce < max_depth; ++bounce) {
      HitInfo hit = {};
      hit.hit = 0;
      unsigned int p0, p1;
      pack_ptr(&hit, p0, p1);
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
          2,
          0,
          p0,
          p1);

      if (!hit.hit) {
        radiance = radiance + mul3(throughput, hit.env);
        break;
      }
      if (bounce == 0) {
        aov_albedo_acc = aov_albedo_acc + hit.albedo;
        aov_normal_acc = aov_normal_acc + hit.normal;
        aov_depth_acc += hit.depth;
        aov_depth_valid += 1.0f;
      }

      radiance = radiance + mul3(throughput, hit.emission * hit.albedo);

      float3 n = hit.normal;
      if (dot3(n, rd) > 0.0f) n = n * -1.0f;
      radiance = radiance + mul3(throughput, sample_direct_lighting(hit, n, -rd, seed));

      BsdfSample bs = sample_bsdf(hit, n, -rd, seed);
      rd = bs.wi;
      throughput = mul3(throughput, bs.f_over_pdf);
      ro = hit.position + 1e-3f * rd;
      if (bounce >= 2) {
        float q = fmaxf(0.05f, 1.0f - fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)));
        if (randf(seed) < q) break;
        throughput = throughput * (1.0f / (1.0f - q));
      }
    }
    final_c = final_c + clamp3max(radiance, params.firefly_clamp);
  }

  float3 c = final_c * (1.0f / static_cast<float>(spp));
  const unsigned int pixel = idx.y * params.width + idx.x;
  params.aov_beauty_hdr[pixel] = make_float4(c.x, c.y, c.z, 1.0f);
  const float inv_gamma = 1.0f / fmaxf(1e-4f, params.gamma);
  float3 mapped = c * params.exposure;
  mapped = f3(
      mapped.x / (1.0f + mapped.x),
      mapped.y / (1.0f + mapped.y),
      mapped.z / (1.0f + mapped.z));
  mapped = f3(
      powf(fmaxf(0.0f, mapped.x), inv_gamma),
      powf(fmaxf(0.0f, mapped.y), inv_gamma),
      powf(fmaxf(0.0f, mapped.z), inv_gamma));
  params.image[pixel] = make_uchar4(
      static_cast<unsigned char>(255.0f * clamp01(mapped.x)),
      static_cast<unsigned char>(255.0f * clamp01(mapped.y)),
      static_cast<unsigned char>(255.0f * clamp01(mapped.z)),
      255);
  const float inv_spp = 1.0f / static_cast<float>(spp);
  const float3 albedo = aov_albedo_acc * inv_spp;
  const float3 normal = normalize3(aov_normal_acc * inv_spp + f3(1e-6f, 0.0f, 0.0f));
  const float depth = (aov_depth_valid > 0.0f) ? (aov_depth_acc / aov_depth_valid) : 0.0f;
  const float valid = aov_depth_valid > 0.0f ? 1.0f : 0.0f;
  params.aov_albedo[pixel] = make_float4(albedo.x, albedo.y, albedo.z, valid);
  params.aov_normal[pixel] = make_float4(normal.x, normal.y, normal.z, valid);
  params.aov_depth[pixel] = make_float4(depth, 0.0f, 0.0f, valid);
}

extern "C" __global__ void __miss__ms() {
  HitInfo* hit = reinterpret_cast<HitInfo*>(unpack_ptr(optixGetPayload_0(), optixGetPayload_1()));
  hit->hit = 0;
  hit->env = environment_color(normalize3(optixGetWorldRayDirection()));
}

extern "C" __global__ void __miss__shadow() {
  optixSetPayload_0(0u);
}

extern "C" __global__ void __closesthit__ch() {
  unsigned int prim_idx = optixGetPrimitiveIndex();
  HitInfo* hit = reinterpret_cast<HitInfo*>(unpack_ptr(optixGetPayload_0(), optixGetPayload_1()));
  if (prim_idx >= params.primitive_count) return;
  RibbonPrimitive p = params.primitives[prim_idx];
  float u = __uint_as_float(optixGetAttribute_0());
  float v = __uint_as_float(optixGetAttribute_1());

  float3 n = f3(0.0f, 1.0f, 0.0f);
  if (p.type == RIBBON_TYPE_2) {
    const Ribbon2Data& r = p.r2;
    float3 dQdu = (1.0f - v) * (r.q10 - r.q00) + v * (r.q11 - r.q01);
    float3 dQdv = (1.0f - u) * (r.q01 - r.q00) + u * (r.q11 - r.q10);
    n = normalize3(cross3(dQdu, dQdv));
  } else {
    const Ribbon3Data& r = p.r3;
    float3 vel = ribbon3_velocity(r, u);
    float3 bn = normalize3(ribbon3_bitangent(r, u));
    n = normalize3(cross3(vel, bn));
  }

  float3 rd = normalize3(optixGetWorldRayDirection());
  if (dot3(n, rd) > 0.0f) n = n * -1.0f;
  const float t = optixGetRayTmax();
  const float3 ro = optixGetWorldRayOrigin();
  hit->hit = 1;
  hit->position = ro + t * rd;
  hit->depth = t;
  hit->normal = n;
  hit->albedo = p.base_color;
  hit->emission = p.emission;
  hit->metallic = p.metallic;
  hit->roughness = p.roughness;
  hit->material_type = p.material_type;
  hit->ior = p.ior;
}

extern "C" __global__ void __closesthit__shadow() {
  optixSetPayload_0(1u);
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
