# Ray-Ribbon-Intersection

OptiX implementation of:
`Ray/Ribbon Intersections` (Alexander Reshetov, 2022)
https://dl.acm.org/doi/abs/10.1145/3543862

Implemented in this repo:

- `ribbon2` data structure (bilinear patch, quadratic intersection in `u`)
- `ribbon3` data structure (quadratic directrix + linear/nonlinear generator, cubic intersection in `u`)
- OptiX custom primitive flow:
  - AABB generation
  - intersection program (`__intersection__ribbon`)
  - closest-hit shading (`__closesthit__ch`)
- Cubic + quadratic root solving used by the intersectors
- Demo scene rendering to `PPM`
- Production-oriented app skeleton:
  - `scene/` scene data + loader
  - `app/` runtime orchestration
  - CLI arguments for scene/output/resolution/PTX

## 1) Prerequisites

- NVIDIA GPU + driver
- CUDA Toolkit (tested with CUDA 12.5+)
- Visual Studio 2022 (MSVC toolchain)
- CMake 3.24+
- OpenGL-capable display driver
- GLFW 3.4 is fetched by CMake if no system package is found

## 2) Download + Configure OptiX SDK

Install the NVIDIA OptiX SDK, then point CMake at the SDK root. On Windows
PowerShell this can be done with:

```powershell
$env:OPTIX_ROOT = "C:\Path\To\optix-sdk-8.1.0"
```

The SDK root must contain `include\optix.h`.

For per-machine settings, copy `CMakeUserPresets.json.example` to
`CMakeUserPresets.json` and edit the local `OPTIX_ROOT` and CUDA paths. The
user preset file is ignored by Git. The VS Code build task automatically uses
the `local-release` preset when this file exists.

## 3) Build

Run from a Visual Studio Developer PowerShell or another shell with the MSVC
toolchain available:

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
```

If you copied the user preset template, build with:

```powershell
cmake --preset local-release
cmake --build --preset local-release
```

## 4) Run

Realtime GLFW/OpenGL viewer:

```powershell
.\build\ray_ribbon.exe --realtime --scene scenes\default.rrs --width 960 --height 540 --spp 1 --max-depth 2
```

Dynamic wind can be enabled on compatible ribbon scenes:

```powershell
.\build\ray_ribbon.exe --realtime --wind --scene scenes\default.rrs --width 960 --height 540 --spp 1 --max-depth 1 --wind-strength 0.20 --wind-speed 1.8
```

When `--wind` is enabled, the window title reports dynamic update costs:

- `upd`: CPU ribbon deformation time
- `gas`: primitive upload, AABB generation, and OptiX GAS rebuild time
- `render`: OptiX launch plus framebuffer readback time
- `frame`: full frame time, including display upload
- `mem`: currently used GPU memory
- `smooth`: neighboring-ribbon displacement discontinuity; lower is smoother

Headless benchmark:

```powershell
.\build\ray_ribbon.exe --benchmark-static 180 --scene scenes\default.rrs --width 960 --height 540 --spp 1 --max-depth 1
.\build\ray_ribbon.exe --benchmark-wind 180 --scene scenes\default.rrs --width 960 --height 540 --spp 1 --max-depth 1 --wind-strength 0.20 --wind-speed 1.8
```

Controls:

- Drag left mouse button to orbit the camera
- Mouse wheel zooms
- `W/A/S/D` pans horizontally/forward
- `Q/E` pans down/up
- Drop a `.rrs` file onto the window to import a scene
- `R` reloads the current scene
- `Space` resets the camera
- `P` saves a screenshot to `captures/`
- `1/2/3` switch realtime quality presets
- `Esc` closes the window

Offline image render:

```powershell
.\build\ray_ribbon.exe
```

or:

```powershell
.\build\ray_ribbon.exe --scene scenes\default.rrs --out output.exr --aov-dir aovs --width 1280 --height 720 --spp 16 --max-depth 4 --denoise
```

## Notes

- Device program entry is in `src/device/ribbon_kernels.cu`.
- Host-side OptiX setup and GAS build are in `src/optix_renderer.cpp`.
- Shared primitive/layout structs are in `src/shared/ribbon_types.h`.
- Scene parsing is in `src/scene/scene.cpp`.
- Scene light lines:
  - `light_dir dx dy dz r g b intensity`
  - `light_point px py pz r g b intensity`
  - `light_rect cx cy cz ux uy uz vx vy vz r g b intensity`
- Scene ribbon lines:
  - `ribbon2 q00 q01 q10 q11 w0 w1 w2 [color emission metallic roughness material_type ior]`
  - `ribbon3 b0 b1 b2 bt0 bt1 width [color emission metallic roughness material_type ior]`
  - `ribbon3_full b0 b1 b2 bt0 bt1 w0 w1 w2 alpha beta gamma [color emission metallic roughness material_type ior]`
- Material extension in ribbon lines(optional tail values):
  - `material_type`: `0=diffuse`, `1=metal`, `2=dielectric`
  - `ior`: index of refraction(e.g. 1.45)
- CLI options:
  - `--scene <path>`
  - `--out <path>`
  - `--aov-dir <path>` outputs `albedo.ppm`, `normal.ppm`, `depth.ppm`
  - `--width <int>`
  - `--height <int>`
  - `--spp <int>`
  - `--max-depth <int>`
  - `--exposure <float>`
  - `--gamma <float>`
  - `--firefly-clamp <float>`
  - `--denoise` enable OptiX denoiser(beauty guided by albedo+normal)
  - `--realtime` launch the GLFW/OpenGL realtime viewer
  - `--wind` enable realtime grass wind animation and benchmark overlay
  - `--wind-strength <float>`
  - `--wind-speed <float>`
  - `--benchmark-static [frames]`
  - `--benchmark-wind [frames]`
  - `--ptx <path>`
