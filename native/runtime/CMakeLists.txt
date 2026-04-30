cmake_minimum_required(VERSION 3.25)

project(dart_mlx_ffi_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DMF_ENABLE_ORT "Enable ONNX Runtime backend" OFF)
option(DMF_BUILD_CLI "Build standalone native runtime benchmark CLI" OFF)

set(RUNTIME_SRC
    ${CMAKE_CURRENT_LIST_DIR}/runtime_bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/memory_bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/litert_backend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ort_backend.cpp)

if(APPLE)
  enable_language(OBJCXX)
  list(APPEND RUNTIME_SRC ${CMAKE_CURRENT_LIST_DIR}/coreml_backend.mm)
else()
  list(APPEND RUNTIME_SRC ${CMAKE_CURRENT_LIST_DIR}/coreml_stub.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${RUNTIME_SRC})

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}
          ${CMAKE_CURRENT_LIST_DIR}/../../third_party/json/single_include)

if(DMF_ENABLE_ORT)
  if(NOT DEFINED DMF_ORT_INCLUDE_DIR OR NOT DEFINED DMF_ORT_LIBRARY)
    message(FATAL_ERROR
            "DMF_ENABLE_ORT requires DMF_ORT_INCLUDE_DIR and DMF_ORT_LIBRARY")
  endif()
  target_include_directories(${PROJECT_NAME} PRIVATE ${DMF_ORT_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${DMF_ORT_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PRIVATE DMF_ENABLE_ORT=1)
else()
  target_compile_definitions(${PROJECT_NAME} PRIVATE DMF_ENABLE_ORT=0)
endif()

if(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Foundation"
                                                "-framework CoreML")
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/coreml_backend.mm
                              PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set(_dmf_runtime_rpath "@loader_path")
elseif(WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE psapi)
  set(_dmf_runtime_rpath "")
elseif(UNIX)
  target_link_libraries(${PROJECT_NAME} PRIVATE dl)
  set(_dmf_runtime_rpath "$ORIGIN")
endif()

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME "dart_mlx_ffi_runtime"
             BUILD_WITH_INSTALL_RPATH ON
             INSTALL_RPATH "${_dmf_runtime_rpath}")

if(DMF_BUILD_CLI)
  add_executable(dart_mlx_ffi_runtime_runner
                 ${CMAKE_CURRENT_LIST_DIR}/runtime_cli.cpp)
  target_include_directories(
    dart_mlx_ffi_runtime_runner
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
            ${CMAKE_CURRENT_LIST_DIR}/../../third_party/json/single_include)
  target_link_libraries(dart_mlx_ffi_runtime_runner PRIVATE ${PROJECT_NAME})
  set_target_properties(
    dart_mlx_ffi_runtime_runner
    PROPERTIES BUILD_WITH_INSTALL_RPATH ON
               INSTALL_RPATH "${_dmf_runtime_rpath}")
endif()
