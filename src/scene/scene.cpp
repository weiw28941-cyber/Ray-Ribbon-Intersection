#include "scene/scene.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace rr {

static float3 make3(float x, float y, float z) {
  return float3{x, y, z};
}

static RibbonPrimitive make_ribbon2(const Ribbon2Data& r2) {
  RibbonPrimitive p = {};
  p.type = RIBBON_TYPE_2;
  p.material_type = 0;
  p.base_color = make3(0.95f, 0.72f, 0.35f);
  p.emission = 0.0f;
  p.metallic = 0.0f;
  p.roughness = 1.0f;
  p.ior = 1.5f;
  p.r2 = r2;
  return p;
}

static RibbonPrimitive make_ribbon3(
    const float3& b0, const float3& b1, const float3& b2, const float3& bt0, const float3& bt1, float width) {
  RibbonPrimitive p = {};
  p.type = RIBBON_TYPE_3;
  p.material_type = 0;
  p.base_color = make3(0.35f, 0.72f, 0.95f);
  p.emission = 0.0f;
  p.metallic = 0.0f;
  p.roughness = 1.0f;
  p.ior = 1.5f;
  p.r3.b0 = b0;
  p.r3.b1 = b1;
  p.r3.b2 = b2;
  p.r3.bt0 = bt0;
  p.r3.bt1 = bt1;
  p.r3.alpha = 1.0f;
  p.r3.beta = 0.0f;
  p.r3.gamma = 0.0f;
  p.r3.w0 = width;
  p.r3.w1 = 0.0f;
  p.r3.w2 = 0.0f;
  return p;
}

Scene make_default_scene() {
  Scene scene;
  Ribbon2Data r2 = {};
  r2.q00 = make3(-1.2f, -0.4f, -3.6f);
  r2.q01 = make3(-1.2f, 0.1f, -3.6f);
  r2.q10 = make3(1.0f, -0.4f, -5.2f);
  r2.q11 = make3(1.0f, 0.0f, -5.2f);
  r2.w0 = 0.10f;
  r2.w1 = 0.02f;
  r2.w2 = -0.01f;
  scene.primitives.push_back(make_ribbon2(r2));

  scene.primitives.push_back(make_ribbon3(
      make3(-0.8f, -0.1f, -3.0f),
      make3(0.0f, 1.2f, -4.2f),
      make3(0.9f, 0.0f, -5.0f),
      make3(0.0f, 0.0f, 1.0f),
      make3(0.0f, 0.0f, 1.0f),
      0.13f));

  LightData dir = {};
  dir.type = LIGHT_DIRECTIONAL;
  dir.direction = make3(-0.6f, -1.0f, -0.4f);
  dir.color = make3(1.0f, 0.98f, 0.92f);
  dir.intensity = 1.5f;
  scene.lights.push_back(dir);

  LightData point = {};
  point.type = LIGHT_POINT;
  point.position = make3(0.0f, 1.8f, -2.5f);
  point.color = make3(1.0f, 0.95f, 0.85f);
  point.intensity = 16.0f;
  scene.lights.push_back(point);

  LightData rect = {};
  rect.type = LIGHT_RECT;
  rect.position = make3(0.0f, 2.2f, -3.8f);
  rect.u = make3(0.9f, 0.0f, 0.0f);
  rect.v = make3(0.0f, 0.0f, 0.9f);
  rect.color = make3(1.0f, 0.98f, 0.95f);
  rect.intensity = 8.0f;
  scene.lights.push_back(rect);
  return scene;
}

static bool parse_numbers(const std::vector<std::string>& toks, std::vector<float>& out) {
  out.clear();
  out.reserve(toks.size());
  try {
    for (const auto& t : toks) out.push_back(std::stof(t));
  } catch (...) {
    return false;
  }
  return true;
}

bool load_scene_from_file(const std::string& path, Scene& out_scene, std::string& error) {
  std::ifstream in(path);
  if (!in) {
    error = "Cannot open scene file: " + path;
    return false;
  }

  Scene scene = make_default_scene();
  scene.primitives.clear();

  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind.empty()) continue;

    std::vector<std::string> toks;
    std::string tok;
    while (ss >> tok) toks.push_back(tok);
    std::vector<float> nums;
    if (!parse_numbers(toks, nums)) {
      error = "Parse error at line " + std::to_string(line_no);
      return false;
    }

    if (kind == "render" && nums.size() >= 2) {
      scene.render.width = static_cast<unsigned>(nums[0]);
      scene.render.height = static_cast<unsigned>(nums[1]);
      if (nums.size() >= 3) scene.render.spp = static_cast<unsigned>(nums[2]);
      if (nums.size() >= 4) scene.render.max_depth = static_cast<unsigned>(nums[3]);
      if (nums.size() >= 5) scene.render.exposure = nums[4];
      if (nums.size() >= 6) scene.render.gamma = nums[5];
      if (nums.size() >= 7) scene.render.firefly_clamp = nums[6];
      continue;
    }
    if (kind == "camera_from" && nums.size() >= 3) {
      scene.camera.look_from = make3(nums[0], nums[1], nums[2]);
      continue;
    }
    if (kind == "camera_at" && nums.size() >= 3) {
      scene.camera.look_at = make3(nums[0], nums[1], nums[2]);
      continue;
    }
    if (kind == "camera_fov" && nums.size() >= 1) {
      scene.camera.fov_deg = nums[0];
      continue;
    }
    if (kind == "light_dir" && nums.size() >= 7) {
      LightData l = {};
      l.type = LIGHT_DIRECTIONAL;
      l.direction = make3(nums[0], nums[1], nums[2]);
      l.color = make3(nums[3], nums[4], nums[5]);
      l.intensity = nums[6];
      scene.lights.push_back(l);
      continue;
    }
    if (kind == "light_point" && nums.size() >= 7) {
      LightData l = {};
      l.type = LIGHT_POINT;
      l.position = make3(nums[0], nums[1], nums[2]);
      l.color = make3(nums[3], nums[4], nums[5]);
      l.intensity = nums[6];
      scene.lights.push_back(l);
      continue;
    }
    if (kind == "light_rect" && nums.size() >= 13) {
      LightData l = {};
      l.type = LIGHT_RECT;
      l.position = make3(nums[0], nums[1], nums[2]);
      l.u = make3(nums[3], nums[4], nums[5]);
      l.v = make3(nums[6], nums[7], nums[8]);
      l.color = make3(nums[9], nums[10], nums[11]);
      l.intensity = nums[12];
      scene.lights.push_back(l);
      continue;
    }
    if (kind == "ribbon2" && nums.size() >= 15) {
      Ribbon2Data r2 = {};
      r2.q00 = make3(nums[0], nums[1], nums[2]);
      r2.q01 = make3(nums[3], nums[4], nums[5]);
      r2.q10 = make3(nums[6], nums[7], nums[8]);
      r2.q11 = make3(nums[9], nums[10], nums[11]);
      r2.w0 = nums[12];
      r2.w1 = nums[13];
      r2.w2 = nums[14];
      RibbonPrimitive p = make_ribbon2(r2);
      if (nums.size() >= 18) {
        p.base_color = make3(nums[15], nums[16], nums[17]);
      }
      if (nums.size() >= 19) p.emission = nums[18];
      if (nums.size() >= 20) p.metallic = nums[19];
      if (nums.size() >= 21) p.roughness = nums[20];
      if (nums.size() >= 22) p.material_type = static_cast<int>(nums[21]);
      if (nums.size() >= 23) p.ior = nums[22];
      scene.primitives.push_back(p);
      continue;
    }
    if (kind == "ribbon3" && nums.size() >= 16) {
      RibbonPrimitive p = make_ribbon3(
          make3(nums[0], nums[1], nums[2]),
          make3(nums[3], nums[4], nums[5]),
          make3(nums[6], nums[7], nums[8]),
          make3(nums[9], nums[10], nums[11]),
          make3(nums[12], nums[13], nums[14]),
          nums[15]);
      if (nums.size() >= 19) {
        p.base_color = make3(nums[16], nums[17], nums[18]);
      }
      if (nums.size() >= 20) p.emission = nums[19];
      if (nums.size() >= 21) p.metallic = nums[20];
      if (nums.size() >= 22) p.roughness = nums[21];
      if (nums.size() >= 23) p.material_type = static_cast<int>(nums[22]);
      if (nums.size() >= 24) p.ior = nums[23];
      scene.primitives.push_back(p);
      continue;
    }
  }

  if (scene.primitives.empty()) {
    error = "Scene contains no ribbons";
    return false;
  }
  if (scene.lights.empty()) {
    LightData dir = {};
    dir.type = LIGHT_DIRECTIONAL;
    dir.direction = make3(-0.6f, -1.0f, -0.4f);
    dir.color = make3(1.0f, 0.98f, 0.92f);
    dir.intensity = 1.5f;
    scene.lights.push_back(dir);
  }
  out_scene = scene;
  return true;
}

}  // namespace rr
