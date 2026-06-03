cmake_minimum_required(VERSION 3.24)
project(RayRibbonIntersection LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

find_package(glfw3 CONFIG QUIET)
if(NOT glfw3_FOUND)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glfw
    URL https://codeload.github.com/glfw/glfw/zip/refs/tags/3.4
  )
  FetchContent_MakeAvailable(glfw)
endif()

set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "Path to OptiX SDK root")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")

if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
  message(FATAL_ERROR "OptiX include not found. Set OPTIX_ROOT or pass -DOPTIX_ROOT=<OptiX SDK root>.")
endif()

add_executable(ray_ribbon
  src/main.cpp
  src/optix_renderer.cpp
  src/app/render_app.cpp
  src/app/realtime_app.cpp
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

target_link_libraries(ray_ribbon PRIVATE CUDA::cuda_driver CUDA::cudart OpenGL::GL)
target_link_libraries(ray_ribbon PRIVATE glfw)

set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/ribbon_kernels.ptx")
set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/device/ribbon_kernels.cu")
set(ANIMATION_OBJ "${CMAKE_BINARY_DIR}/animation_kernels.obj")
set(ANIMATION_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/device/animation_kernels.cu")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ANIMATION_HOST_FLAGS "/EHsc,/MDd")
  set(ANIMATION_DEFINES -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=2)
else()
  set(ANIMATION_HOST_FLAGS "/EHsc,/MD")
  set(ANIMATION_DEFINES -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0)
endif()

set_source_files_properties("${ANIMATION_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_sources(ray_ribbon PRIVATE "${ANIMATION_OBJ}")

add_custom_command(
  OUTPUT "${PTX_OUTPUT}"
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    --ptx
    --std=c++17
    $<$<BOOL:${MSVC}>:-allow-unsupported-compiler>
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

add_custom_command(
  OUTPUT "${ANIMATION_OBJ}"
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    -c
    --std=c++17
    $<$<BOOL:${MSVC}>:-allow-unsupported-compiler>
    -Xcompiler=${ANIMATION_HOST_FLAGS}
    ${ANIMATION_DEFINES}
    -I${OPTIX_INCLUDE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -o "${ANIMATION_OBJ}"
    "${ANIMATION_SRC}"
  DEPENDS "${ANIMATION_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/device/animation_kernels.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/shared/ribbon_types.h"
  COMMENT "Compiling CUDA animation kernels"
  VERBATIM
)

add_custom_target(device_kernels ALL DEPENDS "${PTX_OUTPUT}" "${ANIMATION_OBJ}")
add_dependencies(ray_ribbon device_kernels)

add_custom_command(
  TARGET ray_ribbon POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PTX_OUTPUT}" "$<TARGET_FILE_DIR:ray_ribbon>/ribbon_kernels.ptx"
)
