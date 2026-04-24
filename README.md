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

Output image:

- `build\Release\output.ppm`

## Notes

- Device program entry is in `src/device/ribbon_kernels.cu`.
- Host-side OptiX setup and GAS build are in `src/optix_renderer.cpp`.
- Shared primitive/layout structs are in `src/shared/ribbon_types.h`.
