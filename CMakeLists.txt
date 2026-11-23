cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.14.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

# ── Options ───────────────────────────────────────────────────
option(LOGOS_SANITIZE "Enable ASan + UBSan on CPU test targets" OFF)
option(LOGOS_USE_COMPUTE_SANITIZER "Wrap GPU CTest tests with compute-sanitizer" OFF)
option(LOGOS_RAPIDCHECK "Build T25 RapidCheck property tests (fetches rapidcheck via FetchContent)" OFF)
set(LOGOS_RAPIDCHECK_TAG "master" CACHE STRING "rapidcheck git tag/branch/commit used by LOGOS_RAPIDCHECK")
# Real Mull mutation testing (Linux + clang only). Path of the Mull IR-frontend plugin,
# e.g. /usr/lib/mull-ir-frontend-14 (must match the clang + mull-NN package version).
# When set, every CPU test target is built -O0 -g with the Mull plugin so that
# `mull-runner-NN <test exe>` can mutate include/*.hpp and src/*.cpp.
set(LOGOS_MULL_PLUGIN "" CACHE FILEPATH "Mull IR frontend plugin (enables Mull instrumentation of CPU tests)")
if(LOGOS_MULL_PLUGIN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR APPLE OR WIN32)
        message(FATAL_ERROR "LOGOS_MULL_PLUGIN needs clang on Linux (got ${CMAKE_CXX_COMPILER_ID})")
    endif()
    if(NOT EXISTS "${LOGOS_MULL_PLUGIN}")
        message(FATAL_ERROR "LOGOS_MULL_PLUGIN not found: ${LOGOS_MULL_PLUGIN}")
    endif()
    message(STATUS "Mull instrumentation ON: ${LOGOS_MULL_PLUGIN}")
endif()

function(logos_set_cpu_test_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
        if(LOGOS_SANITIZE)
            target_compile_options(${target} PRIVATE /fsanitize=address)
            message(STATUS "  ${target}: MSVC ASan ON")
        endif()
    elseif(APPLE)
        target_compile_options(${target} PRIVATE -O1 -g -std=c++20)
        if(LOGOS_SANITIZE)
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
                -fno-optimize-sibling-calls)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
            message(STATUS "  ${target}: ASan+UBSan ON (Apple clang)")
        endif()
    else()
        target_compile_options(${target} PRIVATE -O1 -g -march=x86-64 -std=c++20
            -fno-omit-frame-pointer)
        if(LOGOS_SANITIZE)
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
                -fsanitize-address-use-after-scope
                -fno-sanitize-recover=undefined
                -fsanitize=signed-integer-overflow
                -fsanitize=null
                -fsanitize=alignment
                -fsanitize=bounds)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
            message(STATUS "  ${target}: ASan+UBSan ON (GCC/Clang Linux)")
        endif()
        if(LOGOS_MULL_PLUGIN)
            # appended after -O1 above, so -O0 wins (Mull needs unoptimised IR + debug info)
            target_compile_options(${target} PRIVATE
                -O0 -g -grecord-command-line -fpass-plugin=${LOGOS_MULL_PLUGIN})
            message(STATUS "  ${target}: Mull plugin ON")
        endif()
    endif()
endfunction()

set(ASAN_ENV
    "ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:print_stats=1:check_initialization_order=1:detect_stack_use_after_return=1"
    "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
)

# ── CPU test: checkpoint roundtrip (host-side weights container) ──
add_executable(test_checkpoint
    tests/test_checkpoint.cpp
    src/Checkpoint.cpp
)
target_include_directories(test_checkpoint PRIVATE include)
logos_set_cpu_test_options(test_checkpoint)

enable_testing()
add_test(NAME "T2b_Checkpoint" COMMAND test_checkpoint)
set_tests_properties("T2b_Checkpoint" PROPERTIES TIMEOUT 120 ENVIRONMENT "${ASAN_ENV}")

# ── CPU test suite (no CUDA needed) ───────────────────────────
#   T20 tokenizer unit | T21 physics (CPU) | T22 fuzz | T23 integration | T24 bench
#   ctest -L UNIT | -L FUZZ | -L INTEGRATION | -L PHYSICS | -L BENCH
find_package(Threads REQUIRED)
option(LOGOS_TSAN "ThreadSanitizer on CPU test targets (Linux only; do not combine with LOGOS_SANITIZE)" OFF)
set(LOGOS_FUZZ_ITERS "3000" CACHE STRING "Iterations for T22 fuzz tests")
set(LOGOS_FUZZ_SEED "1337" CACHE STRING "RNG seed for T22 fuzz tests (nightly CI randomises it)")
set(CPU_TEST_ENV ${ASAN_ENV} "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1"
    "LOGOS_FUZZ_ITERS=${LOGOS_FUZZ_ITERS}" "LOGOS_FUZZ_SEED=${LOGOS_FUZZ_SEED}")

function(logos_add_cpu_test target src test_name timeout labels)
    add_executable(${target} ${src})
    target_include_directories(${target} PRIVATE include tests/cpu)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    logos_set_cpu_test_options(${target})
    if(LOGOS_TSAN AND NOT MSVC AND NOT APPLE AND NOT WIN32)
        target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=thread)
    endif()
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT ${timeout} LABELS "${labels}" ENVIRONMENT "${CPU_TEST_ENV}")
endfunction()

logos_add_cpu_test(test_tokenizer   tests/cpu/test_tokenizer.cpp   "T20_Tokenizer_Unit" 300 "CPU;UNIT")
logos_add_cpu_test(test_physics_cpu tests/cpu/test_physics_cpu.cpp "T21_Physics_CPU"    120 "CPU;UNIT;PHYSICS")
logos_add_cpu_test(test_fuzz        tests/cpu/test_fuzz.cpp        "T22_Fuzz"           900 "CPU;FUZZ")
logos_add_cpu_test(test_integration tests/cpu/test_integration.cpp "T23_Integration"    600 "CPU;INTEGRATION")
logos_add_cpu_test(test_bench       tests/cpu/bench_tokenizer.cpp  "T24_Bench"           900 "CPU;BENCH")
if(NOT MSVC)
    target_compile_options(test_bench PRIVATE -O2)   # later flag wins over -O1: bench Release-like
endif()

# T25 RapidCheck (property-based; opt-in so normal configures never touch the network)
if(LOGOS_RAPIDCHECK)
    include(FetchContent)
    set(RC_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(RC_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(RC_INSTALL_ALL_EXTRAS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG        ${LOGOS_RAPIDCHECK_TAG}
        GIT_SHALLOW    ON)
    FetchContent_MakeAvailable(rapidcheck)
    logos_add_cpu_test(test_rapidcheck tests/cpu/test_rapidcheck.cpp "T25_RapidCheck" 900 "CPU;RAPIDCHECK")
    target_link_libraries(test_rapidcheck PRIVATE rapidcheck)
endif()

# libFuzzer (LLVM's open-source C++ coverage-guided fuzzer) + ASan + UBSan. Clang only, opt-in.
# Targets are run by hand / CI (./fuzz_tokenizer corpus/ -max_total_time=N), not via ctest.
option(LOGOS_LIBFUZZER "Build libFuzzer targets (fuzz_tokenizer, fuzz_vocab, fuzz_tokenbin, fuzz_checkpoint)" OFF)
if(LOGOS_LIBFUZZER)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR MSVC)
        message(FATAL_ERROR "LOGOS_LIBFUZZER needs clang (got ${CMAKE_CXX_COMPILER_ID})")
    endif()
    function(logos_add_libfuzzer target src)
        add_executable(${target} ${src} ${ARGN})
        target_include_directories(${target} PRIVATE include tests/cpu tests/fuzz)
        target_compile_options(${target} PRIVATE -std=c++20 -O1 -g -fno-omit-frame-pointer
            -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined)
        target_link_options(${target} PRIVATE -fsanitize=fuzzer,address,undefined)
        target_link_libraries(${target} PRIVATE Threads::Threads)
    endfunction()
    logos_add_libfuzzer(fuzz_tokenizer  tests/fuzz/fuzz_tokenizer.cpp)
    logos_add_libfuzzer(fuzz_vocab      tests/fuzz/fuzz_vocab.cpp)
    logos_add_libfuzzer(fuzz_tokenbin   tests/fuzz/fuzz_tokenbin.cpp)
    logos_add_libfuzzer(fuzz_checkpoint tests/fuzz/fuzz_checkpoint.cpp src/Checkpoint.cpp)
endif()

# ── CUDA detection ────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "✅ CUDA found — building GPU targets")

    # cuBLAS is MANDATORY — every forward/backward GEMM/GEMV goes through cuda_gemm_rm().
    # (LOGOS_USE_CUBLAS=OFF stays ON.)
    set(LOGOS_USE_CUBLAS ON CACHE BOOL "cuBLAS is the only CUDA GEMM backend" FORCE)
    option(LOGOS_CUBLAS_TF32 "Use NVIDIA Tensor Core TF32 math for float GEMM (Ampere+ only)" OFF)

    set(LOGOS_CUDA_LIBRARIES CUDA::cudart CUDA::cublas)

    # ── CUDA Architecture ─────────────────────────────────────
    set(CUDA_ARCH_MODE "native" CACHE STRING
        "CUDA arch mode: 'native' (auto-detect) or 'release' (all GPUs)")

    if(CUDA_ARCH_MODE MATCHES "^[0-9]+(;[0-9]+)*$")
        # explicit list, e.g. -DCUDA_ARCH_MODE=75 or "75;86" (compile-only CI runners have no GPU to 'native'-detect)
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_MODE}"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch: EXPLICIT — ${CMAKE_CUDA_ARCHITECTURES}")
    elseif(CUDA_ARCH_MODE STREQUAL "release")
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch: RELEASE — ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch: NATIVE — auto-detecting")
    endif()

    # ── CUDA flags: PRODUCTION (logos_gpu, logos_core) ───────
    set(LOGOS_CUDA_FLAGS_PROD
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xcompiler=-O3
            --ptxas-options=-v
            --extended-lambda
            --expt-relaxed-constexpr
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )

    # ── CUDA flags: TEST targets (no --use_fast_math) ─────────
    set(LOGOS_CUDA_FLAGS_TEST
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O1
            -g
            --generate-line-info
            --ptxas-options=-v
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-O1,-g,-fno-omit-frame-pointer
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O1 -g>
    )

    # ── logos_core STATIC LIBRARY (ODR fix) ───────────────────
    add_library(logos_core STATIC
        cuda/CublasGemm.cu
        cuda/ModelGPU.cu
        src/Checkpoint.cpp
    )
    target_include_directories(logos_core PUBLIC include cuda)
    target_compile_options(logos_core PRIVATE ${LOGOS_CUDA_FLAGS_PROD})
    target_link_libraries(logos_core PUBLIC ${LOGOS_CUDA_LIBRARIES})
    target_compile_definitions(logos_core PUBLIC
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)
    set_target_properties(logos_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

    # ── GPU Training executable ───────────────────────────────
    add_executable(logos_gpu cuda/train_gpu.cu)
    target_include_directories(logos_gpu PRIVATE include cuda)
    target_compile_options(logos_gpu PRIVATE ${LOGOS_CUDA_FLAGS_PROD})
    target_link_libraries(logos_gpu PRIVATE logos_core)
    target_compile_definitions(logos_gpu PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── GPU test targets ──────────────────────────────────────
    function(logos_add_gpu_test target src)
        add_executable(${target} ${src})
        target_include_directories(${target} PRIVATE include cuda tests)
        target_compile_options(${target} PRIVATE ${LOGOS_CUDA_FLAGS_TEST})
        target_link_libraries(${target} PRIVATE logos_core)
        target_compile_definitions(${target} PRIVATE
            LOGOS_TEST_BUILD=1
            LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
            LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)
    endfunction()

    logos_add_gpu_test(test_syntax_build    tests/test_syntax_build.cu)     # T1
    logos_add_gpu_test(test_math_units_gpu  tests/test_math_units.cu)       # T2
    logos_add_gpu_test(test_memory_stress   tests/test_memory_stress.cu)    # T3
    logos_add_gpu_test(test_large_data_scale tests/test_large_data_scale.cu)# T4
    logos_add_gpu_test(test_cuda_kernels    tests/test_cuda_kernels.cu)     # T5
    logos_add_gpu_test(test_cublas_gemm     tests/test_cublas_gemm.cu)      # T10

    # ── GPU CTest registration ────────────────────────────────
    set(LOGOS_GPU_TESTS
        "T1_Syntax_Build:test_syntax_build:300"
        "T2_Math_Unit_GPU:test_math_units_gpu:300"
        "T3_Memory_Stress:test_memory_stress:600"
        "T4_Large_Data_Scale:test_large_data_scale:900"
        "T5_CUDA_Kernels:test_cuda_kernels:300"
        "T10_CuBLAS_GEMM:test_cublas_gemm:300"
    )

    set(_SANITIZER_OK OFF)
    if(LOGOS_USE_COMPUTE_SANITIZER)
        find_program(COMPUTE_SANITIZER_EXE
            NAMES compute-sanitizer
            PATHS /usr/local/cuda/bin $ENV{CUDA_PATH}/bin)
        if(COMPUTE_SANITIZER_EXE)
            set(_SANITIZER_OK ON)
            message(STATUS "  compute-sanitizer: ${COMPUTE_SANITIZER_EXE}")
        else()
            message(WARNING "  compute-sanitizer not found — GPU tests will run unwrapped")
        endif()
    endif()

    foreach(entry ${LOGOS_GPU_TESTS})
        string(REPLACE ":" ";" parts "${entry}")
        list(GET parts 0 tname)
        list(GET parts 1 texe)
        list(GET parts 2 ttimeout)
        if(_SANITIZER_OK)
            add_test(NAME ${tname}
                COMMAND ${COMPUTE_SANITIZER_EXE} --tool memcheck --error-exitcode 1
                        --log-file sanitizer_${texe}.log $<TARGET_FILE:${texe}>)
        else()
            add_test(NAME ${tname} COMMAND ${texe})
        endif()
        set_tests_properties(${tname} PROPERTIES
            TIMEOUT ${ttimeout} LABELS "GPU" RESOURCE_LOCK "GPU_DEVICE_0")
    endforeach()

    message(STATUS "")
    message(STATUS "CTest targets:")
    message(STATUS "  CPU+sanitizer: ctest -LE GPU --output-on-failure (cmake -DLOGOS_SANITIZE=ON)")
    message(STATUS "  GPU normal:    ctest -L GPU --output-on-failure")
    message(STATUS "  GPU+memcheck:  cmake -DLOGOS_USE_COMPUTE_SANITIZER=ON .. && ctest -L GPU")
    message(STATUS "")
else()
    message(STATUS "⚠️  CUDA not found — only the CPU checkpoint test is built")
endif()

if(CMAKE_CUDA_COMPILER)
    install(TARGETS logos_gpu DESTINATION bin)
endif()

message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  test_checkpoint        — T2b: checkpoint roundtrip (CPU)")
if(CMAKE_CUDA_COMPILER)
message(STATUS "  logos_gpu              — GPU training/inference (cuBLAS)")
message(STATUS "  test_syntax_build      — T1: CUDA syntax + GPU kernels")
message(STATUS "  test_math_units_gpu    — T2: physics math (GPU vs CPU reference)")
message(STATUS "  test_memory_stress     — T3: 1M+ data, VRAM leak, bounds")
message(STATUS "  test_large_data_scale  — T4: 1TB scale, production CI/CD")
message(STATUS "  test_cuda_kernels      — T5: kernel correctness + cross-file")
message(STATUS "  test_cublas_gemm       — T10: cuBLAS wrapper vs CPU")
endif()
message(STATUS "")
