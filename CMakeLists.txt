cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

# ── CPU executable ────────────────────────────────────────────
add_executable(logos src/main.cpp)
if(APPLE)
    target_compile_options(logos PRIVATE -O3)
else()
    target_compile_options(logos PRIVATE -O3 -march=x86-64)
endif()

# ── CUDA detection ────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "✅ CUDA found — building GPU targets")

    # ── CUDA Architecture ─────────────────────────────────────
    set(CUDA_ARCH_MODE "native" CACHE STRING
        "CUDA arch mode: 'native' (auto-detect) or 'release' (all GPUs)")

    if(CUDA_ARCH_MODE STREQUAL "release")
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch: RELEASE — ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch: NATIVE — auto-detecting")
    endif()

    # ── Shared CUDA compile options ───────────────────────────
    set(LOGOS_CUDA_FLAGS
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xcompiler=-O3
            --ptxas-options=-v
            --extended-lambda
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )

    # ── GPU Training executable ───────────────────────────────
    add_executable(logos_gpu
        cuda/train_gpu.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
    )
    target_include_directories(logos_gpu PRIVATE include cuda)
    target_compile_options(logos_gpu PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(logos_gpu PRIVATE cudart)

    # ── TEST 1: Syntax & Build Test ───────────────────────────
    # Tests: CUDA device, GPUTensor RAII, all kernels, full pipeline
    add_executable(test_syntax_build
        tests/test_syntax_build.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
        src/main.cpp        # pulls in LOGOSModel (CPU)
    )
    target_include_directories(test_syntax_build PRIVATE include cuda tests)
    target_compile_options(test_syntax_build PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_syntax_build PRIVATE cudart)
    # Exclude main.cpp's main() — test has its own
    target_compile_definitions(test_syntax_build PRIVATE LOGOS_TEST_BUILD=1)

    # ── TEST 2: Mathematical Unit Test ────────────────────────
    # Tests: CPU math only — no CUDA dependency, pure math correctness
    add_executable(test_math_unit
        tests/test_math_unit.cpp
    )
    target_include_directories(test_math_unit PRIVATE include tests)
    if(APPLE)
        target_compile_options(test_math_unit PRIVATE -O2 -std=c++20)
    else()
        target_compile_options(test_math_unit PRIVATE -O2 -march=x86-64 -std=c++20)
    endif()
    # No CUDA needed for math unit tests — pure CPU

    # ── TEST 3: Memory & Bounds Stress Test ───────────────────
    # Tests: VRAM limits, 1M+ data, boundary reads/writes, leak detection
    add_executable(test_memory_stress
        tests/test_memory_stress.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
    )
    target_include_directories(test_memory_stress PRIVATE include cuda tests)
    target_compile_options(test_memory_stress PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_memory_stress PRIVATE cudart)

    # ── CTest integration ─────────────────────────────────────
    enable_testing()

    add_test(NAME "T2_Math_Unit"
             COMMAND test_math_unit)

    add_test(NAME "T1_Syntax_Build"
             COMMAND test_syntax_build)

    add_test(NAME "T3_Memory_Stress"
             COMMAND test_memory_stress)

    # Timeouts: math=60s, syntax=120s, memory=300s
    set_tests_properties("T2_Math_Unit"    PROPERTIES TIMEOUT 60)
    set_tests_properties("T1_Syntax_Build" PROPERTIES TIMEOUT 120)
    set_tests_properties("T3_Memory_Stress" PROPERTIES TIMEOUT 300)

    message(STATUS "Tests: test_syntax_build, test_math_unit, test_memory_stress")
    message(STATUS "Run:   ctest --output-on-failure")
    message(STATUS "Or:    ./build/test_math_unit && ./build/test_syntax_build && ./build/test_memory_stress")

else()
    message(STATUS "⚠️  CUDA not found — CPU-only build")

    # Math unit test still works without CUDA
    add_executable(test_math_unit
        tests/test_math_unit.cpp
    )
    target_include_directories(test_math_unit PRIVATE include tests)
    if(APPLE)
        target_compile_options(test_math_unit PRIVATE -O2 -std=c++20)
    else()
        target_compile_options(test_math_unit PRIVATE -O2 -march=x86-64 -std=c++20)
    endif()

    enable_testing()
    add_test(NAME "T2_Math_Unit" COMMAND test_math_unit)
    set_tests_properties("T2_Math_Unit" PROPERTIES TIMEOUT 60)
endif()

install(TARGETS logos DESTINATION bin)
if(CMAKE_CUDA_COMPILER)
    install(TARGETS logos_gpu DESTINATION bin)
endif()

message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  logos              — CPU training")
if(CMAKE_CUDA_COMPILER)
message(STATUS "  logos_gpu          — GPU training (Phase 1+2)")
message(STATUS "  test_syntax_build  — Test 1: CUDA syntax + GPU kernels")
message(STATUS "  test_math_unit     — Test 2: Vedic math formulas vs modern")
message(STATUS "  test_memory_stress — Test 3: 1M+ data, VRAM leak, bounds")
endif()
message(STATUS "")
