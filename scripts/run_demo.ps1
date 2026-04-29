param(
  [string]$Config = "Release",
  [int]$Width = 1280,
  [int]$Height = 720,
  [int]$Spp = 16,
  [int]$MaxDepth = 5,
  [switch]$Denoise
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $repoRoot "build"
$scene = Join-Path $repoRoot "scenes/default.rrs"
$demoDir = Join-Path $repoRoot "demo_output"
$aovDir = Join-Path $demoDir "aovs"
$outFile = Join-Path $demoDir "demo_beauty.exr"

if ([string]::IsNullOrWhiteSpace($env:OPTIX_ROOT) -or -not (Test-Path $env:OPTIX_ROOT)) {
  Write-Error "OPTIX_ROOT is not set or invalid. Please run scripts/setup_optix.ps1 first."
}

New-Item -ItemType Directory -Force -Path $demoDir | Out-Null
New-Item -ItemType Directory -Force -Path $aovDir | Out-Null

if (-not (Test-Path $buildDir)) {
  & cmake -S $repoRoot -B $buildDir -G "Visual Studio 17 2022" -A x64 -DOPTIX_ROOT="$env:OPTIX_ROOT"
}

& cmake --build $buildDir --config $Config

$exe = Join-Path $buildDir "$Config/ray_ribbon.exe"
if (-not (Test-Path $exe)) {
  Write-Error "Executable not found: $exe"
}

$args = @(
  "--scene", $scene,
  "--out", $outFile,
  "--aov-dir", $aovDir,
  "--width", $Width,
  "--height", $Height,
  "--spp", $Spp,
  "--max-depth", $MaxDepth,
  "--exposure", "1.1",
  "--gamma", "2.2",
  "--firefly-clamp", "6.0"
)
if ($Denoise) {
  $args += "--denoise"
}

& $exe @args

Write-Host "Demo complete."
Write-Host "Beauty: $outFile"
Write-Host "AOVs:   $aovDir"
