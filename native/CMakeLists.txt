cmake_minimum_required(VERSION 3.25)

project(dart_mlx_ffi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
set(MLX_USE_CCACHE OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF ON CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(MLX_DART_ENABLE_X64_MAC OFF)

if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
  set(MLX_ENABLE_X64_MAC ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx
                 ${CMAKE_CURRENT_BINARY_DIR}/mlx)

set(MLXC_SRC
    ${CMAKE_CURRENT_LIST_DIR}/bridge_core.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_array.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_conv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_extra.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_math.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_memio.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_mm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_misc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_more.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_rand.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_sys.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bridge_fast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/array.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/closure.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/compile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/cuda.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/device.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/distributed.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/distributed_group.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/error.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/fast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/fft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/io_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/linalg.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/map.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/metal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/random.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/string.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/transforms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/transforms_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/vector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c/mlx/c/version.cpp)

add_library(${PROJECT_NAME} SHARED ${MLXC_SRC})

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx-c
          ${CMAKE_CURRENT_LIST_DIR}/../third_party/mlx)

target_link_libraries(${PROJECT_NAME} PRIVATE mlx)

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME "dart_mlx_ffi"
             BUILD_WITH_INSTALL_RPATH ON)
