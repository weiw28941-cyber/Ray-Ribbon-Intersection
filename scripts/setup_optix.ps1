param(
  [string]$Version = "v8.1.0",
  [string]$InstallRoot = "D:\SDKs"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  throw "curl.exe not found. Please install curl first."
}

if (-not (Test-Path $InstallRoot)) {
  New-Item -ItemType Directory -Path $InstallRoot | Out-Null
}

$zipPath = Join-Path $InstallRoot "optix-sdk-$Version.zip"
$extractPath = Join-Path $InstallRoot "optix-sdk-$($Version.TrimStart('v'))"
$url = "https://codeload.github.com/NVIDIA/optix-sdk/zip/refs/tags/$Version"

Write-Host "Downloading OptiX SDK $Version from $url"
& curl.exe -L --retry 8 --retry-all-errors --connect-timeout 20 -o $zipPath $url

if (Test-Path $extractPath) {
  Remove-Item -Recurse -Force $extractPath
}

Expand-Archive -Path $zipPath -DestinationPath $InstallRoot -Force

$expanded = Join-Path $InstallRoot "optix-sdk-$($Version.TrimStart('v'))"
if (-not (Test-Path (Join-Path $expanded "include\optix.h"))) {
  throw "OptiX SDK extraction failed. include\\optix.h not found."
}

[Environment]::SetEnvironmentVariable("OPTIX_ROOT", $expanded, "User")
Write-Host "OPTIX_ROOT set to $expanded (User scope)."
Write-Host "Done."
