#include "app/render_app.h"

#include <exception>

#include "core/log.h"
#include "optix_renderer.h"

namespace rr {

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
