cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags — -O3 + native SIMD (auto-vectorizes VedicGEMM)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# Include paths
include_directories(include)

# Main executable
add_executable(logos src/main.cpp)
target_compile_options(logos PRIVATE -O3 -march=native)

# Individual test executables
add_executable(test_vedic src/VedicGEMM.cpp)
target_compile_definitions(test_vedic PRIVATE LOGOS_TEST_VEDIC)
target_compile_options(test_vedic PRIVATE -O3 -march=native)

add_executable(test_tokenizer src/Tokenizer.cpp)
target_compile_definitions(test_tokenizer PRIVATE LOGOS_TEST_TOKENIZER)

# Install
install(TARGETS logos DESTINATION bin)

message(STATUS "LOGOS LLM — C++20 build configured")
message(STATUS "Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")
