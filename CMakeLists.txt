cmake_minimum_required(VERSION 3.24)
project(RayRibbonIntersection LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(OPTIX_ROOT_DEFAULT "D:/SDKs/optix-sdk-8.1.0")
set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "Path to OptiX SDK root")
if(NOT OPTIX_ROOT)
  set(OPTIX_ROOT "${OPTIX_ROOT_DEFAULT}")
endif()
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")

if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
  message(FATAL_ERROR "OptiX include not found at ${OPTIX_INCLUDE_DIR}. Set OPTIX_ROOT to the SDK root.")
endif()

add_executable(ray_ribbon
  src/main.cpp
  src/optix_renderer.cpp
  src/app/render_app.cpp
  src/scene/scene.cpp
  src/io/image_io.cpp
  third_party/tinyexr-release/deps/miniz/miniz.c
)

target_include_directories(ray_ribbon PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OPTIX_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexr-release
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexr-release/deps/miniz
)

target_link_libraries(ray_ribbon PRIVATE CUDA::cuda_driver CUDA::cudart)

set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/ribbon_kernels.ptx")
set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/device/ribbon_kernels.cu")

add_custom_command(
  OUTPUT "${PTX_OUTPUT}"
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    --ptx
    --std=c++17
    --use_fast_math
    -lineinfo
    -I${OPTIX_INCLUDE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -o "${PTX_OUTPUT}"
    "${DEVICE_SRC}"
  DEPENDS "${DEVICE_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/shared/launch_params.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/shared/ribbon_types.h"
  COMMENT "Compiling OptiX device program to PTX"
  VERBATIM
)

add_custom_target(device_ptx ALL DEPENDS "${PTX_OUTPUT}")
add_dependencies(ray_ribbon device_ptx)

add_custom_command(
  TARGET ray_ribbon POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PTX_OUTPUT}" "$<TARGET_FILE_DIR:ray_ribbon>/ribbon_kernels.ptx"
)
