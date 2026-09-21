cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# All headers in include/
include_directories(include)

# ── SINGLE executable — only main.cpp compiled
#    All modules are header-only (.hpp) — no multiple definitions possible
add_executable(logos src/main.cpp)

# Platform-aware flags — macOS ARM64 pe -march=x86-64 nahi chalta
if(APPLE)
  target_compile_options(logos PRIVATE -O3)
else()
  target_compile_options(logos PRIVATE -O3 -march=x86-64)
endif()

install(TARGETS logos DESTINATION bin)

message(STATUS "LOGOS — header-only build (no multiple-definition errors)")
message(STATUS "cmake --build build && ./build/logos --train dataset.txt")
