cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.10.0 LANGUAGES CXX)
# v0.10.0 — 6 Physics Components + M11-M16 Tests
# Build fixes:
#   test_syntax_build: src/main.cpp REMOVED (duplicate main), Checkpoint.cpp ADDED
#   logos_gpu:         src/Checkpoint.cpp ADDED (save_checkpoint/load_checkpoint)
#   test_math_unit:    now includes Attention.hpp + LayerNorm.hpp + FeedForward.hpp

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

# ── CPU executable ────────────────────────────────────────────
add_executable(logos
    src/main.cpp
    src/Checkpoint.cpp
)
if(APPLE)
    target_compile_options(logos PRIVATE -O3)
elseif(MSVC)
    target_compile_options(logos PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(logos PRIVATE -O3 -march=x86-64)
endif()

function(logos_set_cpu_test_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
    elseif(APPLE)
        target_compile_options(${target} PRIVATE -O2 -std=c++20)
    else()
        target_compile_options(${target} PRIVATE -O2 -march=x86-64 -std=c++20)
    endif()
endfunction()

# ── CUDA detection ────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "✅ CUDA found — building GPU targets")

    option(LOGOS_USE_CUBLAS "Use cuBLAS as the primary CUDA GEMM backend" ON)
    option(LOGOS_CUBLAS_TF32 "Use NVIDIA Tensor Core TF32 math for float GEMM" OFF)

    set(LOGOS_CUDA_LIBRARIES CUDA::cudart)
    if(LOGOS_USE_CUBLAS)
        list(APPEND LOGOS_CUDA_LIBRARIES CUDA::cublas)
    endif()

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
        src/Checkpoint.cpp      # FIX: save_checkpoint / load_checkpoint
    )
    target_include_directories(logos_gpu PRIVATE include cuda)
    target_compile_options(logos_gpu PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(logos_gpu PRIVATE ${LOGOS_CUDA_LIBRARIES})
    target_compile_definitions(logos_gpu PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 1: Syntax & Build Test ───────────────────────────
    # Tests: CUDA device, GPUTensor RAII, all kernels, full pipeline
    # FIX: src/main.cpp REMOVED (duplicate main) — test has its own main()
    # FIX: src/Checkpoint.cpp ADDED — provides save_checkpoint / load_checkpoint
    add_executable(test_syntax_build
        tests/test_syntax_build.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
        src/Checkpoint.cpp
    )
    target_include_directories(test_syntax_build PRIVATE include cuda tests)
    target_compile_options(test_syntax_build PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_syntax_build PRIVATE ${LOGOS_CUDA_LIBRARIES})
    target_compile_definitions(test_syntax_build PRIVATE
        LOGOS_TEST_BUILD=1
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 2: Mathematical Unit Test ────────────────────────
    # Tests: CPU math only — no CUDA dependency, pure math correctness
    add_executable(test_math_unit
        tests/test_math_unit.cpp
    )
    target_include_directories(test_math_unit PRIVATE include tests)
    logos_set_cpu_test_options(test_math_unit)
    # No CUDA needed for math unit tests — pure CPU

    add_executable(test_checkpoint
        tests/test_checkpoint.cpp
        src/Checkpoint.cpp
    )
    target_include_directories(test_checkpoint PRIVATE include)
    logos_set_cpu_test_options(test_checkpoint)

    # ── TEST 3: Memory & Bounds Stress Test ───────────────────
    # Tests: VRAM limits, 1M+ data, boundary reads/writes, leak detection
    add_executable(test_memory_stress
        tests/test_memory_stress.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
    )
    target_include_directories(test_memory_stress PRIVATE include cuda tests)
    target_compile_options(test_memory_stress PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_memory_stress PRIVATE ${LOGOS_CUDA_LIBRARIES})
    target_compile_definitions(test_memory_stress PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 4: Large Data Scale & Production Test ────────────
    # Tests: 1TB int64_t, chunked pipeline, grad accum, multi-shard,
    #        smoke test, GPU large vocab — CI/CD production readiness
    add_executable(test_large_data_scale
        tests/test_large_data_scale.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
        src/Checkpoint.cpp
    )
    target_include_directories(test_large_data_scale PRIVATE include cuda tests)
    target_compile_options(test_large_data_scale PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_large_data_scale PRIVATE ${LOGOS_CUDA_LIBRARIES})
    target_compile_definitions(test_large_data_scale PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── CTest integration ─────────────────────────────────────
    enable_testing()

    add_test(NAME "T2_Math_Unit"
             COMMAND test_math_unit)
    add_test(NAME "Checkpoint_Roundtrip" COMMAND test_checkpoint)

    add_test(NAME "T1_Syntax_Build"
             COMMAND test_syntax_build)

    add_test(NAME "T3_Memory_Stress"
             COMMAND test_memory_stress)

    add_test(NAME "T4_Large_Data_Scale"
             COMMAND test_large_data_scale)

    # Timeouts: math=60s, syntax=120s, memory=300s, large_data=600s
    set_tests_properties("T2_Math_Unit"        PROPERTIES TIMEOUT 60)
    set_tests_properties("Checkpoint_Roundtrip" PROPERTIES TIMEOUT 60)
    set_tests_properties("T1_Syntax_Build"     PROPERTIES TIMEOUT 120)
    set_tests_properties("T3_Memory_Stress"    PROPERTIES TIMEOUT 300)
    set_tests_properties("T4_Large_Data_Scale" PROPERTIES TIMEOUT 600)

    message(STATUS "Tests: test_syntax_build, test_math_unit, test_memory_stress, test_large_data_scale")
    message(STATUS "Run:   ctest --output-on-failure")
    message(STATUS "Or:    ./build/test_math_unit && ./build/test_syntax_build && ./build/test_memory_stress && ./build/test_large_data_scale")

else()
    message(STATUS "⚠️  CUDA not found — CPU-only build")

    # Math unit test still works without CUDA
    add_executable(test_math_unit
        tests/test_math_unit.cpp
    )
    target_include_directories(test_math_unit PRIVATE include tests)
    logos_set_cpu_test_options(test_math_unit)

    add_executable(test_checkpoint
        tests/test_checkpoint.cpp
        src/Checkpoint.cpp
    )
    target_include_directories(test_checkpoint PRIVATE include)
    logos_set_cpu_test_options(test_checkpoint)

    # ── TEST 4 (CPU-only build) ───────────────────────────────
    # test_large_data_scale.cu compiles as CXX without CUDA
    # HAS_CUDA=0 disables GPU sections automatically
    add_executable(test_large_data_scale
        tests/test_large_data_scale.cu
        src/Checkpoint.cpp
    )
    target_include_directories(test_large_data_scale PRIVATE include tests)
    # Rename .cu to allow CXX compiler (no nvcc needed)
    set_source_files_properties(
        tests/test_large_data_scale.cu
        PROPERTIES LANGUAGE CXX
    )
    logos_set_cpu_test_options(test_large_data_scale)

    enable_testing()
    add_test(NAME "T2_Math_Unit"        COMMAND test_math_unit)
    add_test(NAME "Checkpoint_Roundtrip" COMMAND test_checkpoint)
    add_test(NAME "T4_Large_Data_Scale" COMMAND test_large_data_scale)
    set_tests_properties("T2_Math_Unit"        PROPERTIES TIMEOUT 60)
    set_tests_properties("Checkpoint_Roundtrip" PROPERTIES TIMEOUT 60)
    set_tests_properties("T4_Large_Data_Scale" PROPERTIES TIMEOUT 600)

    message(STATUS "CPU-only tests: test_math_unit, test_checkpoint, test_large_data_scale")
endif()

install(TARGETS logos DESTINATION bin)
if(CMAKE_CUDA_COMPILER)
    install(TARGETS logos_gpu DESTINATION bin)
endif()

message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  logos                  — CPU training")
if(CMAKE_CUDA_COMPILER)
message(STATUS "  logos_gpu              — GPU training (Phase 1+2)")
message(STATUS "  test_syntax_build      — Test 1: CUDA syntax + GPU kernels")
message(STATUS "  test_math_unit         — Test 2: Vedic math formulas vs modern")
message(STATUS "  test_memory_stress     — Test 3: 1M+ data, VRAM leak, bounds")
message(STATUS "  test_large_data_scale  — Test 4: 1TB scale, production CI/CD")
endif()
message(STATUS "")
