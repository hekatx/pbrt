cmake_minimum_required(VERSION 3.10)

project(pbrt VERSION 1.0)

set(CORE_SOURCE
  src/core/geometry.cpp
  )

set(CORE_HEADERS
  src/core/geometry.hpp
  )

include_directories(src)
include_directories(src/core)

add_library(pbrt STATIC ${CORE_SOURCE} ${CORE_HEADERS})

set(ALL_PBRT_LIBS
  pbrt
)

add_executable(pbrt_exe src/main.cpp)

target_link_libraries(pbrt_exe ${ALL_PBRT_LIBS})

