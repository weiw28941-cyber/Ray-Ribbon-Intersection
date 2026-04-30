#include "app/render_app.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static std::string resolve_ptx_path(const char* argv0, int argc, char** argv) {
  namespace fs = std::filesystem;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--ptx") {
      return argv[i + 1];
    }
  }

  const fs::path exe_path = fs::absolute(fs::path(argv0));
  const fs::path exe_dir = exe_path.parent_path();
  const fs::path cwd = fs::current_path();
  const std::vector<fs::path> candidates = {
      cwd / "ribbon_kernels.ptx",
      cwd / "build" / "Release" / "ribbon_kernels.ptx",
      exe_dir / "ribbon_kernels.ptx",
      exe_dir / ".." / "ribbon_kernels.ptx"};
  for (const auto& p : candidates) {
    if (fs::exists(p)) return p.string();
  }
  throw std::runtime_error("Cannot find ribbon_kernels.ptx. Use --ptx <path>.");
}

int main(int argc, char** argv) {
  rr::AppOptions options = {};
  options.ptx_path = resolve_ptx_path(argv[0], argc, argv);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--scene" && i + 1 < argc) {
      options.scene_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      options.output_path = argv[++i];
    } else if (arg == "--aov-dir" && i + 1 < argc) {
      options.aov_dir = argv[++i];
    } else if (arg == "--width" && i + 1 < argc) {
      options.width = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg == "--height" && i + 1 < argc) {
      options.height = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg == "--spp" && i + 1 < argc) {
      options.spp = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg == "--max-depth" && i + 1 < argc) {
      options.max_depth = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg == "--exposure" && i + 1 < argc) {
      options.exposure = std::stof(argv[++i]);
    } else if (arg == "--gamma" && i + 1 < argc) {
      options.gamma = std::stof(argv[++i]);
    } else if (arg == "--firefly-clamp" && i + 1 < argc) {
      options.firefly_clamp = std::stof(argv[++i]);
    } else if (arg == "--denoise") {
      options.denoise = true;
    } else if (arg == "--ptx" && i + 1 < argc) {
      ++i;
    }
  }

  rr::RenderApp app;
  return app.run(options);
}
