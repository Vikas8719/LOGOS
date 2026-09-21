cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")

include_directories(include)

# ── Main executable (test + forward + benchmark + train + eval + generate)
add_executable(logos src/main.cpp)
target_compile_options(logos PRIVATE -O3 -march=x86-64)

# ── Standalone unit tests
add_executable(test_vedic src/VedicGEMM.cpp)
target_compile_definitions(test_vedic PRIVATE LOGOS_TEST_VEDIC)
target_compile_options(test_vedic PRIVATE -O3 -march=x86-64)

add_executable(test_tokenizer src/Tokenizer.cpp)
target_compile_definitions(test_tokenizer PRIVATE LOGOS_TEST_TOKENIZER)

install(TARGETS logos DESTINATION bin)

message(STATUS "LOGOS LLM — C++20 build configured")
message(STATUS "Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")
message(STATUS "Train: ./build/logos --train dataset.txt")
message(STATUS "Eval:  ./build/logos --eval dataset.txt checkpoint.bin")
