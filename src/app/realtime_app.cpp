#include "app/realtime_app.h"

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "core/log.h"
#include "io/image_io.h"
#include "optix_renderer.h"
#include "app/wind_animation.h"

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif

#ifndef GL_PIXEL_UNPACK_BUFFER
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#endif

#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif

#ifndef GLsizeiptr
using GLsizeiptr = ptrdiff_t;
#endif

namespace rr {
namespace {

struct RealtimeState {
  double last_x = 0.0;
  double last_y = 0.0;
  bool dragging = false;
  float scroll_delta = 0.0f;
  std::string pending_scene_path;
};

using GlGenBuffersFn = void (APIENTRY*)(GLsizei, GLuint*);
using GlDeleteBuffersFn = void (APIENTRY*)(GLsizei, const GLuint*);
using GlBindBufferFn = void (APIENTRY*)(GLenum, GLuint);
using GlBufferDataFn = void (APIENTRY*)(GLenum, GLsizeiptr, const void*, GLenum);

static GlGenBuffersFn rr_glGenBuffers = nullptr;
static GlDeleteBuffersFn rr_glDeleteBuffers = nullptr;
static GlBindBufferFn rr_glBindBuffer = nullptr;
static GlBufferDataFn rr_glBufferData = nullptr;

static void cuda_rt_check(cudaError_t rc, const char* what) {
  if (rc != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
  }
}

static void ensure_gl_buffer_api() {
  if (rr_glGenBuffers && rr_glDeleteBuffers && rr_glBindBuffer && rr_glBufferData) {
    return;
  }
  rr_glGenBuffers = reinterpret_cast<GlGenBuffersFn>(glfwGetProcAddress("glGenBuffers"));
  rr_glDeleteBuffers = reinterpret_cast<GlDeleteBuffersFn>(glfwGetProcAddress("glDeleteBuffers"));
  rr_glBindBuffer = reinterpret_cast<GlBindBufferFn>(glfwGetProcAddress("glBindBuffer"));
  rr_glBufferData = reinterpret_cast<GlBufferDataFn>(glfwGetProcAddress("glBufferData"));
  if (!rr_glGenBuffers || !rr_glDeleteBuffers || !rr_glBindBuffer || !rr_glBufferData) {
    throw std::runtime_error("Required OpenGL buffer functions are unavailable.");
  }
}

static float3 add3(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static float3 sub3(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static float3 mul3(const float3& a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

static float dot3h(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float3 cross3h(const float3& a, const float3& b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static float3 normalize3h(const float3& a) {
  const float len2 = std::max(1e-20f, dot3h(a, a));
  return mul3(a, 1.0f / std::sqrt(len2));
}

static float length3h(const float3& a) {
  return std::sqrt(std::max(0.0f, dot3h(a, a)));
}

static DeviceWindMode parse_device_wind_mode(const std::string& wind_mode) {
  if (wind_mode == "strip-independent") return DeviceWindMode::StripIndependent;
  if (wind_mode == "constrained") return DeviceWindMode::StripConstrained;
  return DeviceWindMode::Grass;
}

static void scroll_callback(GLFWwindow* window, double, double yoffset) {
  auto* state = static_cast<RealtimeState*>(glfwGetWindowUserPointer(window));
  if (state) {
    state->scroll_delta += static_cast<float>(yoffset);
  }
}

static void drop_callback(GLFWwindow* window, int count, const char** paths) {
  auto* state = static_cast<RealtimeState*>(glfwGetWindowUserPointer(window));
  if (state && count > 0 && paths && paths[0]) {
    state->pending_scene_path = paths[0];
  }
}

static bool key_pressed(GLFWwindow* window, int key, bool& was_down) {
  const bool down = glfwGetKey(window, key) == GLFW_PRESS;
  const bool pressed = down && !was_down;
  was_down = down;
  return pressed;
}

static std::string scene_label(const std::string& path) {
  if (path.empty()) {
    return "built-in";
  }
  return std::filesystem::path(path).filename().string();
}

static bool load_scene_source(const std::string& path, Scene& scene) {
  if (path.empty()) {
    scene = make_default_scene();
    return true;
  }
  std::string error;
  if (!load_scene_from_file(path, scene, error)) {
    log(LogLevel::Error, error);
    return false;
  }
  log(LogLevel::Info, "Loaded scene: " + path);
  return true;
}

static void apply_cli_overrides(const AppOptions& options, Scene& scene) {
  if (options.width > 0) scene.render.width = options.width;
  if (options.height > 0) scene.render.height = options.height;
  if (options.spp > 0) scene.render.spp = options.spp;
  if (options.max_depth > 0) scene.render.max_depth = options.max_depth;
  if (options.exposure > 0.0f) scene.render.exposure = options.exposure;
  if (options.gamma > 0.0f) scene.render.gamma = options.gamma;
  if (options.firefly_clamp > 0.0f) scene.render.firefly_clamp = options.firefly_clamp;
}

static void reset_orbit(
    const CameraSettings& camera,
    float3& target,
    float& distance,
    float& yaw,
    float& pitch) {
  target = camera.look_at;
  const float3 offset = sub3(camera.look_from, camera.look_at);
  distance = std::sqrt(std::max(1e-20f, dot3h(offset, offset)));
  yaw = std::atan2(offset.x, offset.z);
  pitch = std::asin(std::max(-1.0f, std::min(1.0f, offset.y / distance)));
}

static std::string save_screenshot(const std::vector<uchar4>& pixels, unsigned width, unsigned height, int index) {
  std::ostringstream name;
  name << "captures/realtime_" << std::setw(4) << std::setfill('0') << index << ".ppm";
  save_rgb_ldr(name.str(), pixels, width, height);
  return name.str();
}

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static bool is_wind_ribbon(const RibbonPrimitive& p) {
  if (!(p.type == RIBBON_TYPE_2 || p.type == RIBBON_TYPE_3 || p.type == RIBBON_TYPE_4A ||
        p.type == RIBBON_TYPE_4B || p.type == RIBBON_TYPE_4C)) return false;
  return segment_end_host(p).y - segment_start_host(p).y > 0.01f;
}

static float3 wind_displacement(const RibbonPrimitive& p, float time, float strength, float speed) {
  const float3 base = segment_start_host(p);
  const float height = std::max(0.05f, segment_end_host(p).y);
  const float phase = base.x * 2.2f + base.z * 3.1f + time * speed;
  const float gust = 0.55f + 0.45f * std::sin(phase) + 0.20f * std::sin(phase * 2.37f + 1.1f);
  const float amp = strength * height * gust;
  const float3 dir = normalize3h(make_float3(0.82f, 0.0f, -0.38f));
  return mul3(dir, amp);
}

static float3 grass_point_wind_displacement_local(const float3& p, float time, float strength, float speed) {
  const float height = std::max(0.0f, p.y);
  const float phase = p.x * 2.2f + p.z * 3.1f + p.y * 0.35f + time * speed;
  const float gust = 0.55f + 0.45f * std::sin(phase) + 0.20f * std::sin(phase * 2.37f + 1.1f);
  const float3 dir = normalize3h(make_float3(0.82f, 0.0f, -0.38f));
  float3 d = mul3(dir, strength * height * gust);
  d.y = -length3h(d) * 0.055f;
  return d;
}

static void apply_wind_animation(
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time,
    float strength,
    float speed) {
  animated = base;
  for (size_t i = 0; i < animated.size(); ++i) {
    if (!is_wind_ribbon(base[i])) continue;
    const float3 p0 = segment_start_host(base[i]);
    const float3 pm = segment_mid_host(base[i]);
    const float3 p1 = segment_end_host(base[i]);
    const float3 d0 = grass_point_wind_displacement_local(p0, time, strength, speed);
    const float3 dm = grass_point_wind_displacement_local(pm, time, strength, speed);
    const float3 d1 = grass_point_wind_displacement_local(p1, time, strength, speed);
    if (base[i].type == RIBBON_TYPE_2) {
      add_quadratic_ribbon2_center_delta(animated[i].r2, d0, dm, d1);
    } else if (base[i].type == RIBBON_TYPE_3) {
      animated[i].r3.b0 = add3(base[i].r3.b0, d0);
      animated[i].r3.b1 = add3(base[i].r3.b1, quadratic_bezier_control_delta(d0, dm, d1));
      animated[i].r3.b2 = add3(base[i].r3.b2, d1);
      animated[i].r3.bt1 = normalize3h(add3(base[i].r3.bt1, mul3(d1, 0.10f)));
    } else if (base[i].type == RIBBON_TYPE_4C) {
      add_quadratic_ribbon4c_center_delta(animated[i].r4, d0, dm, d1);
    } else {
      add_quadratic_directrix_delta(animated[i].r4, d0, dm, d1);
      tilt_ribbon4_generator(animated[i].r4, d1, 0.10f);
    }
  }
}

static void apply_host_wind_mode(
    const AppOptions& options,
    const std::vector<RibbonPrimitive>& base,
    std::vector<RibbonPrimitive>& animated,
    float time) {
  if (options.wind_mode == "constrained") {
    apply_constrained_strip_wind_animation(base, animated, time, options.wind_strength, options.wind_speed);
  } else if (options.wind_mode == "strip-independent") {
    apply_independent_strip_wind_animation(base, animated, time, options.wind_strength, options.wind_speed);
  } else if (options.wind_mode == "grass-width") {
    apply_width_dynamic_grass_wind_animation(base, animated, time, options.wind_strength, options.wind_speed);
  } else if (options.wind_mode == "grass-rebuild") {
    apply_rebuilt_grass_wind_animation(base, animated, time, options.wind_strength, options.wind_speed);
  } else {
    apply_wind_animation(base, animated, time, options.wind_strength, options.wind_speed);
  }
}

static float grass_wind_smoothness_metric(
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
    const float3 pi = base[i].r3.b0;
    for (size_t j = 0; j < n; ++j) {
      if (i == j || !is_wind_ribbon(base[j])) continue;
      const float3 pj = base[j].r3.b0;
      const float dx = pi.x - pj.x;
      const float dz = pi.z - pj.z;
      const float d2 = dx * dx + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = static_cast<int>(j);
      }
    }
    if (best >= 0) {
      const float3 di = sub3(animated[i].r3.b2, base[i].r3.b2);
      const float3 dj = sub3(animated[best].r3.b2, base[best].r3.b2);
      sum += length3h(sub3(di, dj)) / std::max(1e-4f, strength);
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

struct CudaGlInterop {
  GLuint pbo = 0;
  cudaGraphicsResource* resource = nullptr;
  int width = 0;
  int height = 0;
};

static GLuint create_texture() {
  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  return texture;
}

static void ensure_texture_storage(GLuint texture, int width, int height, int& tex_w, int& tex_h) {
  glBindTexture(GL_TEXTURE_2D, texture);
  if (tex_w != width || tex_h != height) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    tex_w = width;
    tex_h = height;
  }
}

static void release_interop(CudaGlInterop& interop) {
  if (interop.resource) {
    cudaGraphicsUnregisterResource(interop.resource);
    interop.resource = nullptr;
  }
  if (interop.pbo) {
    rr_glDeleteBuffers(1, &interop.pbo);
    interop.pbo = 0;
  }
  interop.width = 0;
  interop.height = 0;
}

static void ensure_interop_buffer(
    CudaGlInterop& interop, GLuint texture, int width, int height, int& tex_w, int& tex_h) {
  if (interop.width == width && interop.height == height && interop.resource && interop.pbo) {
    return;
  }

  release_interop(interop);
  ensure_texture_storage(texture, width, height, tex_w, tex_h);
  ensure_gl_buffer_api();

  rr_glGenBuffers(1, &interop.pbo);
  rr_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, interop.pbo);
  rr_glBufferData(
      GL_PIXEL_UNPACK_BUFFER,
      static_cast<GLsizeiptr>(static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar4)),
      nullptr,
      GL_STREAM_DRAW);
  rr_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  cuda_rt_check(
      cudaGraphicsGLRegisterBuffer(&interop.resource, interop.pbo, cudaGraphicsRegisterFlagsWriteDiscard),
      "cudaGraphicsGLRegisterBuffer");
  interop.width = width;
  interop.height = height;
}

static void render_texture_from_interop(GLuint texture, CudaGlInterop& interop, int width, int height) {
  glBindTexture(GL_TEXTURE_2D, texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  rr_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, interop.pbo);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  rr_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void draw_fullscreen_texture(GLuint texture) {
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBegin(GL_TRIANGLE_STRIP);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glEnd();
  glBindTexture(GL_TEXTURE_2D, 0);
}

}  // namespace

int RealtimeApp::run(const AppOptions& options) {
  try {
    Scene scene;
    if (!load_scene_source(options.scene_path, scene)) {
      return 2;
    }
    if (options.scene_path.empty()) {
      log(LogLevel::Info, "Using built-in default scene.");
    }

    apply_cli_overrides(options, scene);
    scene.render.spp = options.spp > 0 ? options.spp : 1;
    scene.render.max_depth = options.max_depth > 0 ? options.max_depth : 2;

    if (!glfwInit()) {
      log(LogLevel::Error, "Failed to initialize GLFW.");
      return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(scene.render.width),
        static_cast<int>(scene.render.height),
        "Ray Ribbon Intersection - Realtime",
        nullptr,
        nullptr);
    if (!window) {
      glfwTerminate();
      log(LogLevel::Error, "Failed to create GLFW window.");
      return 1;
    }

    RealtimeState state;
    glfwSetWindowUserPointer(window, &state);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetDropCallback(window, drop_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    OptixRibbonRenderer renderer;
    renderer.initialize(options.ptx_path);
    renderer.set_lights(scene.lights);
    renderer.set_quality(
        scene.render.spp,
        scene.render.max_depth,
        scene.render.exposure,
        scene.render.gamma,
        scene.render.firefly_clamp);
    std::vector<RibbonPrimitive> base_primitives = scene.primitives;
    std::vector<RibbonPrimitive> animated_primitives = scene.primitives;
    const bool gpu_backend = options.wind_backend == "gpu" &&
        options.wind_mode != "grass-rebuild" && options.wind_mode != "grass-width";
    if (options.wind) {
      renderer.prepare_dynamic_primitives(base_primitives);
    } else {
      renderer.set_primitives(scene.primitives);
    }

    CameraSettings camera = scene.camera;
    float3 target = camera.look_at;
    float3 offset = sub3(camera.look_from, camera.look_at);
    float distance = std::sqrt(std::max(1e-20f, dot3h(offset, offset)));
    float yaw = std::atan2(offset.x, offset.z);
    float pitch = std::asin(std::max(-1.0f, std::min(1.0f, offset.y / distance)));

    GLuint texture = create_texture();
    CudaGlInterop interop;
    int tex_w = 0;
    int tex_h = 0;
    std::vector<uchar4> screenshot_pixels;
    double last_title_time = glfwGetTime();
    double last_frame_time = last_title_time;
    int frame_count = 0;
    int screenshot_index = 1;
    std::string current_scene_path = options.scene_path;
    std::string current_scene_name = scene_label(current_scene_path);
    double last_update_ms = 0.0;
    double last_gas_ms = 0.0;
    double last_render_ms = 0.0;
    double last_frame_ms = 0.0;
    float last_smoothness = 0.0f;
    size_t last_free_mem = 0;
    size_t last_total_mem = 0;
    bool reload_was_down = false;
    bool reset_was_down = false;
    bool screenshot_was_down = false;
    bool q1_was_down = false;
    bool q2_was_down = false;
    bool q3_was_down = false;

    log(LogLevel::Info, "Realtime controls: drag left mouse to orbit, wheel to zoom, W/A/S/D/Q/E to pan.");
    log(LogLevel::Info, "Drop a .rrs file to import a scene. R reloads, Space resets camera, P saves a screenshot.");
    log(LogLevel::Info, "Quality presets: 1 = fast, 2 = balanced, 3 = cleaner.");
    if (options.wind) {
      log(LogLevel::Info, "Wind benchmark enabled: rebuilding custom-primitive GAS every frame.");
    }

    while (!glfwWindowShouldClose(window)) {
      const double frame_begin_ms = now_ms();
      const double frame_time = glfwGetTime();
      const float dt = static_cast<float>(std::max(0.001, frame_time - last_frame_time));
      last_frame_time = frame_time;

      glfwPollEvents();
      if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }

      if (!state.pending_scene_path.empty()) {
        Scene imported;
        const std::string path = state.pending_scene_path;
        state.pending_scene_path.clear();
        if (load_scene_source(path, imported)) {
          scene = imported;
          current_scene_path = path;
          current_scene_name = scene_label(path);
          renderer.set_lights(scene.lights);
          renderer.set_quality(
              scene.render.spp,
              scene.render.max_depth,
              scene.render.exposure,
              scene.render.gamma,
              scene.render.firefly_clamp);
          base_primitives = scene.primitives;
          animated_primitives = scene.primitives;
          if (options.wind) {
            renderer.prepare_dynamic_primitives(base_primitives);
          } else {
            renderer.set_primitives(scene.primitives);
          }
          camera = scene.camera;
          reset_orbit(camera, target, distance, yaw, pitch);
        }
      }

      if (key_pressed(window, GLFW_KEY_R, reload_was_down)) {
        Scene reloaded;
        if (load_scene_source(current_scene_path, reloaded)) {
          scene = reloaded;
          renderer.set_lights(scene.lights);
          renderer.set_quality(
              scene.render.spp,
              scene.render.max_depth,
              scene.render.exposure,
              scene.render.gamma,
              scene.render.firefly_clamp);
          base_primitives = scene.primitives;
          animated_primitives = scene.primitives;
          if (options.wind) {
            renderer.prepare_dynamic_primitives(base_primitives);
          } else {
            renderer.set_primitives(scene.primitives);
          }
          camera = scene.camera;
          reset_orbit(camera, target, distance, yaw, pitch);
          log(LogLevel::Info, "Reloaded scene: " + current_scene_name);
        }
      }

      if (key_pressed(window, GLFW_KEY_SPACE, reset_was_down)) {
        camera = scene.camera;
        reset_orbit(camera, target, distance, yaw, pitch);
      }

      if (key_pressed(window, GLFW_KEY_1, q1_was_down)) {
        scene.render.spp = 1;
        scene.render.max_depth = 1;
        renderer.set_quality(scene.render.spp, scene.render.max_depth, scene.render.exposure, scene.render.gamma, scene.render.firefly_clamp);
      }
      if (key_pressed(window, GLFW_KEY_2, q2_was_down)) {
        scene.render.spp = 1;
        scene.render.max_depth = 2;
        renderer.set_quality(scene.render.spp, scene.render.max_depth, scene.render.exposure, scene.render.gamma, scene.render.firefly_clamp);
      }
      if (key_pressed(window, GLFW_KEY_3, q3_was_down)) {
        scene.render.spp = 4;
        scene.render.max_depth = 4;
        renderer.set_quality(scene.render.spp, scene.render.max_depth, scene.render.exposure, scene.render.gamma, scene.render.firefly_clamp);
      }

      double mouse_x = 0.0;
      double mouse_y = 0.0;
      glfwGetCursorPos(window, &mouse_x, &mouse_y);
      const bool mouse_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      if (mouse_down && state.dragging) {
        const double dx = mouse_x - state.last_x;
        const double dy = mouse_y - state.last_y;
        yaw -= static_cast<float>(dx) * 0.006f;
        pitch -= static_cast<float>(dy) * 0.006f;
        pitch = std::max(-1.45f, std::min(1.45f, pitch));
      }
      state.dragging = mouse_down;
      state.last_x = mouse_x;
      state.last_y = mouse_y;

      if (std::fabs(state.scroll_delta) > 0.0f) {
        distance *= std::pow(0.88f, state.scroll_delta);
        distance = std::max(0.1f, distance);
        state.scroll_delta = 0.0f;
      }

      const float cp = std::cos(pitch);
      const float3 view_offset = make_float3(
          distance * cp * std::sin(yaw),
          distance * std::sin(pitch),
          distance * cp * std::cos(yaw));
      camera.look_from = add3(target, view_offset);
      camera.look_at = target;

      const float3 forward = normalize3h(sub3(camera.look_at, camera.look_from));
      const float3 right = normalize3h(cross3h(forward, camera.up));
      const float3 up = normalize3h(cross3h(right, forward));
      const float move_speed = 2.0f * dt;
      float3 pan = make_float3(0.0f, 0.0f, 0.0f);
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) pan = add3(pan, mul3(forward, move_speed));
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) pan = add3(pan, mul3(forward, -move_speed));
      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) pan = add3(pan, mul3(right, -move_speed));
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) pan = add3(pan, mul3(right, move_speed));
      if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) pan = add3(pan, mul3(up, -move_speed));
      if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) pan = add3(pan, mul3(up, move_speed));
      target = add3(target, pan);
      camera.look_from = add3(target, view_offset);
      camera.look_at = target;

      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      if (width > 0 && height > 0) {
        glViewport(0, 0, width, height);
        ensure_interop_buffer(interop, texture, width, height, tex_w, tex_h);
        renderer.set_camera(camera);
        last_update_ms = 0.0;
        last_gas_ms = 0.0;
        if (options.wind) {
          const double update_begin = now_ms();
          if (gpu_backend) {
            renderer.animate_primitives_on_gpu(
                parse_device_wind_mode(options.wind_mode),
                static_cast<float>(frame_time),
                options.wind_strength,
                options.wind_speed);
          } else {
            apply_host_wind_mode(options, base_primitives, animated_primitives, static_cast<float>(frame_time));
          }
          last_update_ms = now_ms() - update_begin;

          const double gas_begin = now_ms();
          if (gpu_backend) {
            renderer.rebuild_gas_from_device_primitives();
          } else {
            renderer.update_dynamic_primitives_on_host(animated_primitives);
          }
          last_gas_ms = now_ms() - gas_begin;

          if ((frame_count & 7) == 0) {
            if (gpu_backend) {
              apply_host_wind_mode(options, base_primitives, animated_primitives, static_cast<float>(frame_time));
            }
            last_smoothness = grass_wind_smoothness_metric(base_primitives, animated_primitives, options.wind_strength);
          }
        }
        const double render_begin = now_ms();
        cuda_rt_check(cudaGraphicsMapResources(1, &interop.resource, 0), "cudaGraphicsMapResources");
        uchar4* d_pixels = nullptr;
        size_t mapped_bytes = 0;
        cuda_rt_check(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pixels), &mapped_bytes, interop.resource),
            "cudaGraphicsResourceGetMappedPointer");
        renderer.render_frame_to_device_buffer(
            static_cast<unsigned>(width),
            static_cast<unsigned>(height),
            reinterpret_cast<CUdeviceptr>(d_pixels));
        cuda_rt_check(cudaGraphicsUnmapResources(1, &interop.resource, 0), "cudaGraphicsUnmapResources");
        last_render_ms = now_ms() - render_begin;
        if (key_pressed(window, GLFW_KEY_P, screenshot_was_down)) {
          renderer.render_frame(static_cast<unsigned>(width), static_cast<unsigned>(height), screenshot_pixels);
          const std::string path = save_screenshot(
              screenshot_pixels, static_cast<unsigned>(width), static_cast<unsigned>(height), screenshot_index++);
          log(LogLevel::Info, "Saved screenshot: " + path);
        }
        render_texture_from_interop(texture, interop, width, height);
        draw_fullscreen_texture(texture);
        glfwSwapBuffers(window);
      }
      last_frame_ms = now_ms() - frame_begin_ms;

      ++frame_count;
      const double now = glfwGetTime();
      if (now - last_title_time >= 1.0) {
        std::ostringstream title;
        title << "Ray Ribbon Intersection - " << frame_count << " FPS"
              << " | " << current_scene_name
              << " | spp " << scene.render.spp
              << " depth " << scene.render.max_depth;
        if (options.wind) {
          renderer.get_gpu_memory_info(last_free_mem, last_total_mem);
          const double used_mb = static_cast<double>(last_total_mem - last_free_mem) / (1024.0 * 1024.0);
          title << " | wind " << (gpu_backend ? "gpu" : "cpu")
                << " upd " << std::fixed << std::setprecision(1) << last_update_ms
                << " gas " << last_gas_ms
                << " render " << last_render_ms
                << " frame " << last_frame_ms << " ms"
                << " | mem " << std::setprecision(0) << used_mb << " MB"
                << " | smooth " << std::setprecision(3) << last_smoothness;
        } else {
          title << " | drop .rrs to import";
        }
        glfwSetWindowTitle(window, title.str().c_str());
        frame_count = 0;
        last_title_time = now;
      }
    }

    release_interop(interop);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
  } catch (const std::exception& e) {
    log(LogLevel::Error, e.what());
    glfwTerminate();
    return 1;
  }
}

}  // namespace rr
