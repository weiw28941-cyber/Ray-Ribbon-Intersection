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

## 2) Download + Configure OptiX SDK

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_optix.ps1
```

This script (default `v8.1.0`):

- downloads official OptiX SDK source package (default `v8.1.0`)
- extracts it to `D:\SDKs\optix-sdk-8.1.0`
- sets user environment variable `OPTIX_ROOT`

## 3) Build

```powershell
mkdir build
cd build
"C:\Program Files\CMake\bin\cmake.exe" -G "Visual Studio 17 2022" -A x64 -DOPTIX_ROOT="$env:OPTIX_ROOT" ..
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release
```

## 4) Run

```powershell
.\Release\ray_ribbon.exe
```

or:

```powershell
.\Release\ray_ribbon.exe --scene ..\..\scenes\default.rrs --out output.exr --aov-dir aovs --width 1280 --height 720 --spp 16 --max-depth 4 --denoise
```

## 5) One-Command Demo (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1 -Denoise
```

Demo output:

- `demo_output/demo_beauty.exr`
- `demo_output/aovs/` (albedo/normal/depth/beauty and denoised beauty when enabled)

Output image:

- `build\Release\output.ppm`

## Notes

- Device program entry is in `src/device/ribbon_kernels.cu`.
- Host-side OptiX setup and GAS build are in `src/optix_renderer.cpp`.
- Shared primitive/layout structs are in `src/shared/ribbon_types.h`.
- Scene parsing is in `src/scene/scene.cpp`.
- Scene light lines:
  - `light_dir dx dy dz r g b intensity`
  - `light_point px py pz r g b intensity`
  - `light_rect cx cy cz ux uy uz vx vy vz r g b intensity`
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
  - `--ptx <path>`
