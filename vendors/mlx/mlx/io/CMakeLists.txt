target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp)

if(MLX_BUILD_SAFETENSORS)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/safetensors.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_safetensors.cpp)
endif()

if(MLX_BUILD_GGUF)
  set(MLX_LOCAL_GGUF_DIR "${PROJECT_SOURCE_DIR}/../gguflib")
  if(EXISTS "${MLX_LOCAL_GGUF_DIR}/gguflib.c")
    message(STATUS "Using vendored gguflib from ${MLX_LOCAL_GGUF_DIR}")
    set(gguflib_SOURCE_DIR ${MLX_LOCAL_GGUF_DIR})
  else()
    message(STATUS "Downloading gguflib")
    FetchContent_Declare(
      gguflib
      GIT_REPOSITORY https://github.com/antirez/gguf-tools/
      GIT_TAG 8fa6eb65236618e28fd7710a0fba565f7faa1848)
    FetchContent_MakeAvailable(gguflib)
  endif()
  target_include_directories(mlx
                             PRIVATE $<BUILD_INTERFACE:${gguflib_SOURCE_DIR}>)
  add_library(gguflib STATIC ${gguflib_SOURCE_DIR}/fp16.c
                             ${gguflib_SOURCE_DIR}/gguflib.c)
  target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:gguflib>)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gguf.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/gguf_quants.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_gguf.cpp)
endif()
