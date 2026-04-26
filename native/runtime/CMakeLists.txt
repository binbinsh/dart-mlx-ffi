cmake_minimum_required(VERSION 3.25)

project(dart_inference_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DINF_ENABLE_ORT "Enable ONNX Runtime backend" OFF)
option(DINF_BUILD_CLI "Build standalone native runtime benchmark CLI" OFF)

set(RUNTIME_SRC
    ${CMAKE_CURRENT_LIST_DIR}/runtime_bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/memory_bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/options.cpp
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
          ${CMAKE_CURRENT_LIST_DIR}/../../vendors/json/single_include)

if(DINF_ENABLE_ORT)
  if(NOT DEFINED DINF_ORT_INCLUDE_DIR OR NOT DEFINED DINF_ORT_LIBRARY)
    message(FATAL_ERROR
            "DINF_ENABLE_ORT requires DINF_ORT_INCLUDE_DIR and DINF_ORT_LIBRARY")
  endif()
  target_include_directories(${PROJECT_NAME} PRIVATE ${DINF_ORT_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${DINF_ORT_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PRIVATE DINF_ENABLE_ORT=1)
else()
  target_compile_definitions(${PROJECT_NAME} PRIVATE DINF_ENABLE_ORT=0)
endif()

if(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Foundation"
                                                "-framework CoreML")
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/coreml_backend.mm
                              PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set(_dart_inference_runtime_rpath "@loader_path")
  set(_dart_inference_exports
      "${CMAKE_CURRENT_BINARY_DIR}/dart_inference_runtime_adapter.exports")
  file(
    WRITE "${_dart_inference_exports}"
    "_dinf_cpp_runtime_backend_json\n"
    "_dinf_cpp_runtime_create\n"
    "_dinf_cpp_runtime_diagnostics_json\n"
    "_dinf_cpp_runtime_free\n"
    "_dinf_cpp_runtime_free_string\n"
    "_dinf_cpp_runtime_free_tensors\n"
    "_dinf_cpp_runtime_memory_info_json\n"
    "_dinf_cpp_runtime_run\n")
  target_link_options(
    ${PROJECT_NAME} PRIVATE
    "LINKER:-exported_symbols_list,${_dart_inference_exports}")
elseif(WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE psapi)
  set(_dart_inference_runtime_rpath "")
elseif(UNIX)
  target_link_libraries(${PROJECT_NAME} PRIVATE dl)
  set(_dart_inference_runtime_rpath "$ORIGIN")
  set(_dart_inference_exports
      "${CMAKE_CURRENT_BINARY_DIR}/dart_inference_runtime_adapter.map")
  file(
    WRITE "${_dart_inference_exports}"
    "{ global: dinf_cpp_runtime_*; local: *; };\n")
  target_link_options(
    ${PROJECT_NAME} PRIVATE
    "LINKER:--version-script=${_dart_inference_exports}")
endif()

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME "dart_inference_runtime_adapter"
             C_VISIBILITY_PRESET hidden
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             BUILD_WITH_INSTALL_RPATH ON
             INSTALL_RPATH "${_dart_inference_runtime_rpath}")

if(DINF_BUILD_CLI)
  add_executable(dart_inference_runtime_runner
                 ${CMAKE_CURRENT_LIST_DIR}/runtime_cli.cpp)
  target_include_directories(
    dart_inference_runtime_runner
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
            ${CMAKE_CURRENT_LIST_DIR}/../../vendors/json/single_include)
  target_link_libraries(dart_inference_runtime_runner PRIVATE ${PROJECT_NAME})
  set_target_properties(
    dart_inference_runtime_runner
    PROPERTIES BUILD_WITH_INSTALL_RPATH ON
               INSTALL_RPATH "${_dart_inference_runtime_rpath}")
endif()
