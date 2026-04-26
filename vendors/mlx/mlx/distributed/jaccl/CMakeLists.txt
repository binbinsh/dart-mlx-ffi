if(MLX_BUILD_CPU
   AND APPLE
   AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS"
   AND MLX_APPLE_SDK_VERSION GREATER_EQUAL 26.2)
  target_sources(
    mlx
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jaccl.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mesh.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ring.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_jaccl.cpp)
endif()
