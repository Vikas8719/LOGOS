cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.12.0 LANGUAGES CXX)
# v0.12.0 — Sanitizers wired
#   ASan+UBSan: CPU tests (T2_Math_Unit, T2b_Checkpoint) mein ON
#     cmake -DLOGOS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug ..
#     ctest -LE GPU   ← CPU tests run honge sanitizers ke saath
#   CUDA GPU tests: compute-sanitizer via CTest environment
#     LOGOS_USE_COMPUTE_SANITIZER=ON set karo
#     ctest -L GPU   ← compute-sanitizer --tool memcheck se wrap hoga
#   --use_fast_math sirf production (logos_gpu) mein hai
#   test targets mein --use_fast_math NAHI — warna sanitizer results galat aate hain

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

# ── Sanitizer option ─────────────────────────────────────────
# cmake -DLOGOS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug ..
# Sirf CPU tests pe kaam karta hai (ASan CUDA ke saath incompatible hai).
# Debug build zaruri hai — Release mein inlining se stack traces galat aate hain.
option(LOGOS_SANITIZE "Enable ASan + UBSan on CPU test targets" OFF)

# Compute-sanitizer option (GPU memory checker — cuda-memcheck successor)
# cmake -DLOGOS_USE_COMPUTE_SANITIZER=ON ..
# Yeh flag CTest GPU tests ko compute-sanitizer --tool memcheck se wrap karta hai.
# Requirement: compute-sanitizer PATH mein hona chahiye (CUDA Toolkit 11.1+ ke saath aata hai).
option(LOGOS_USE_COMPUTE_SANITIZER "Wrap GPU CTest tests with compute-sanitizer" OFF)

function(logos_set_cpu_test_options target)
    if(MSVC)
        # MSVC: ASan support hai lekin /fsanitize=address sirf VS 2019 16.9+ mein
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
        if(LOGOS_SANITIZE)
            target_compile_options(${target} PRIVATE /fsanitize=address)
            message(STATUS "  ${target}: MSVC ASan ON")
        endif()
    elseif(APPLE)
        target_compile_options(${target} PRIVATE -O1 -g -std=c++20)
        if(LOGOS_SANITIZE)
            # Apple clang: ASan + UBSan
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
                -fno-optimize-sibling-calls)
            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined)
            message(STATUS "  ${target}: ASan+UBSan ON (Apple clang)")
        endif()
    else()
        # Linux GCC/Clang
        # -O1 (not -O2): sanitizers need some optimization but not full
        # -g: debug symbols for stack traces
        # -fno-omit-frame-pointer: readable stack traces
        target_compile_options(${target} PRIVATE -O1 -g -march=x86-64 -std=c++20
            -fno-omit-frame-pointer)
        if(LOGOS_SANITIZE)
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
                -fsanitize-address-use-after-scope
                -fno-sanitize-recover=undefined)
            # UBSan extra checks: integer overflow, null deref, alignment
            target_compile_options(${target} PRIVATE
                -fsanitize=signed-integer-overflow
                -fsanitize=null
                -fsanitize=alignment
                -fsanitize=bounds)
            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined)
            message(STATUS "  ${target}: ASan+UBSan ON (GCC/Clang Linux)")
        endif()
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
        message(STATUS "  sm_90 = H100 SXM/PCIe (Hopper)")
        message(STATUS "  sm_89 = RTX 4090 (Ada Lovelace)")
        message(STATUS "  sm_86 = A10/A30/RTX 3090 (Ampere)")
        message(STATUS "  sm_80 = A100 (Ampere)")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch: NATIVE — auto-detecting")
    endif()

    # ── CUDA flags: PRODUCTION (logos_gpu, logos_core) ───────
    # --use_fast_math: training speed ke liye zaroori hai
    #   lekin issse NaN/Inf quiet ho jaate hain aur tests galat pass karte hain.
    #   ISLIYE: sirf logos_gpu aur logos_core mein. Test targets mein NAHI.
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

    # ── CUDA flags: TEST targets ──────────────────────────────
    # --use_fast_math NAHI: warna inf/nan tests bypass ho jaate hain
    # -O1 -g: stack traces readable honge compute-sanitizer output mein
    # --generate-line-info: CUDA kernel line numbers compute-sanitizer mein
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

    # ── [v27-ODR-FIX] logos_core STATIC LIBRARY ──────────────
    # ROOT CAUSE of "multiple definition" linker errors:
    #   VedicGEMM.cu + ModelGPU.cu har executable mein alag compile ho rahe the.
    #   Anonymous namespace kernels (MixedPrecision.cuh) ODR se safe hain,
    #   lekin NVCC __device_stub__ wrappers (cast_fp32_to_fp16_kernel etc.)
    #   tab bhi global scope mein emit hote hain jab VedicGEMM.cu + ModelGPU.cu
    #   dono ek hi executable mein linked hote hain — ek TU se aaya stub
    #   doosre TU ka stub se collide karta hai.
    # FIX: VedicGEMM.cu + ModelGPU.cu ek baar compile karo (static library),
    #   phir har executable sirf apni test .cu file + logos_core link kare.
    #   Ek hi .o file = zero duplicate symbols = no linker error.
    add_library(logos_core STATIC
        cuda/VedicGEMM.cu
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
    add_executable(logos_gpu
        cuda/train_gpu.cu
    )
    target_include_directories(logos_gpu PRIVATE include cuda)
    target_compile_options(logos_gpu PRIVATE ${LOGOS_CUDA_FLAGS_PROD})
    target_link_libraries(logos_gpu PRIVATE logos_core)
    target_compile_definitions(logos_gpu PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 1: Syntax & Build Test ───────────────────────────
    add_executable(test_syntax_build
        tests/test_syntax_build.cu
    )
    target_include_directories(test_syntax_build PRIVATE include cuda tests)
    target_compile_options(test_syntax_build PRIVATE ${LOGOS_CUDA_FLAGS_TEST})
    target_link_libraries(test_syntax_build PRIVATE logos_core)
    target_compile_definitions(test_syntax_build PRIVATE
        LOGOS_TEST_BUILD=1
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 2: Mathematical Unit Test (CPU-only) ─────────────
    add_executable(test_math_unit
        tests/test_math_unit.cpp
    )
    target_include_directories(test_math_unit PRIVATE include tests)
    logos_set_cpu_test_options(test_math_unit)

    # ── TEST 2b: Checkpoint roundtrip ────────────────────────
    add_executable(test_checkpoint
        tests/test_checkpoint.cpp
        src/Checkpoint.cpp
    )
    target_include_directories(test_checkpoint PRIVATE include)
    logos_set_cpu_test_options(test_checkpoint)

    # ── TEST 3: Memory & Bounds Stress Test ───────────────────
    add_executable(test_memory_stress
        tests/test_memory_stress.cu
    )
    target_include_directories(test_memory_stress PRIVATE include cuda tests)
    target_compile_options(test_memory_stress PRIVATE ${LOGOS_CUDA_FLAGS_TEST})
    target_link_libraries(test_memory_stress PRIVATE logos_core)
    target_compile_definitions(test_memory_stress PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 4: Large Data Scale & Production Test ────────────
    add_executable(test_large_data_scale
        tests/test_large_data_scale.cu
    )
    target_include_directories(test_large_data_scale PRIVATE include cuda tests)
    target_compile_options(test_large_data_scale PRIVATE ${LOGOS_CUDA_FLAGS_TEST})
    target_link_libraries(test_large_data_scale PRIVATE logos_core)
    target_compile_definitions(test_large_data_scale PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 5: CUDA Kernel Correctness ──────────────────────
    # add_bias_kernel cross-file launch verify (the VedicGEMM.cuh fix),
    # AMP bias path, training/validation forward modes, SHM optimizer,
    # adjoint gradient checks, FP16 cast, free_energy grad, E2E step.
    add_executable(test_cuda_kernels
        tests/test_cuda_kernels.cu
    )
    target_include_directories(test_cuda_kernels PRIVATE include cuda tests)
    target_compile_options(test_cuda_kernels PRIVATE ${LOGOS_CUDA_FLAGS_TEST})
    target_link_libraries(test_cuda_kernels PRIVATE logos_core)
    target_compile_definitions(test_cuda_kernels PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── CTest integration ─────────────────────────────────────
    enable_testing()

    # ── CPU tests: ASan+UBSan environment ────────────────────
    # ASAN_OPTIONS / UBSAN_OPTIONS runtime behavior set karo:
    #   halt_on_error=1    → pehli error pe abort (warna continue karta hai silently)
    #   detect_leaks=1     → LeakSanitizer ON (ASan ke saath bundled)
    #   print_stats=1      → end mein memory stats print ho
    #   suppressions       → false positives filter karne ke liye (future use)
    # NOTE: LOGOS_SANITIZE=OFF hone par bhi yeh env vars harmless hain
    #       (runtime sanitizer absent hoga, vars ignore honge)
    set(ASAN_ENV
        "ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:print_stats=1:check_initialization_order=1:detect_stack_use_after_return=1"
        "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
    )

    add_test(NAME "T2_Math_Unit"   COMMAND test_math_unit)
    add_test(NAME "T2b_Checkpoint" COMMAND test_checkpoint)
    set_tests_properties("T2_Math_Unit"   PROPERTIES TIMEOUT 120 ENVIRONMENT "${ASAN_ENV}")
    set_tests_properties("T2b_Checkpoint" PROPERTIES TIMEOUT 120 ENVIRONMENT "${ASAN_ENV}")

    if(LOGOS_SANITIZE)
        message(STATUS "  ASan+UBSan: CPU tests T2/T2b mein ACTIVE")
        message(STATUS "  Build karo: cmake -DLOGOS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug ..")
        message(STATUS "  Run:        ctest -LE GPU --output-on-failure")
    endif()

    # ── GPU tests ─────────────────────────────────────────────
    # compute-sanitizer wrap: LOGOS_USE_COMPUTE_SANITIZER=ON
    #   cmake -DLOGOS_USE_COMPUTE_SANITIZER=ON ..
    #   Requires: compute-sanitizer in PATH (CUDA Toolkit 11.1+)
    #   Kaggle: /usr/local/cuda/bin/compute-sanitizer available hai
    #
    # compute-sanitizer --tool memcheck:
    #   - Out-of-bounds GPU memory read/write detect karta hai
    #   - Uninitialized memory read detect karta hai
    #   - Device malloc/free errors detect karta hai
    #
    # Bina sanitizer ke normal GPU test (default):
    #   add_test(COMMAND test_memory_stress) — seedha binary chalti hai
    # Sanitizer ke saath:
    #   add_test(COMMAND compute-sanitizer --tool memcheck test_memory_stress)

    if(LOGOS_USE_COMPUTE_SANITIZER)
        find_program(COMPUTE_SANITIZER_EXE
            NAMES compute-sanitizer
            PATHS /usr/local/cuda/bin $ENV{CUDA_PATH}/bin)

        if(COMPUTE_SANITIZER_EXE)
            message(STATUS "  compute-sanitizer: ${COMPUTE_SANITIZER_EXE}")
            message(STATUS "  GPU tests compute-sanitizer --tool memcheck se wrapped honge")

            add_test(NAME "T1_Syntax_Build"
                COMMAND ${COMPUTE_SANITIZER_EXE}
                    --tool memcheck
                    --error-exitcode 1
                    --log-file sanitizer_t1.log
                    $<TARGET_FILE:test_syntax_build>)

            add_test(NAME "T3_Memory_Stress"
                COMMAND ${COMPUTE_SANITIZER_EXE}
                    --tool memcheck
                    --error-exitcode 1
                    --log-file sanitizer_t3.log
                    $<TARGET_FILE:test_memory_stress>)

            add_test(NAME "T4_Large_Data_Scale"
                COMMAND ${COMPUTE_SANITIZER_EXE}
                    --tool memcheck
                    --error-exitcode 1
                    --log-file sanitizer_t4.log
                    $<TARGET_FILE:test_large_data_scale>)

            add_test(NAME "T5_CUDA_Kernels"
                COMMAND ${COMPUTE_SANITIZER_EXE}
                    --tool memcheck
                    --error-exitcode 1
                    --log-file sanitizer_t5.log
                    $<TARGET_FILE:test_cuda_kernels>)
        else()
            message(WARNING "  compute-sanitizer not found — GPU tests bina sanitizer ke chalenge")
            message(WARNING "  PATH mein add karo: export PATH=/usr/local/cuda/bin:$PATH")
            # Fallback: bina sanitizer ke
            add_test(NAME "T1_Syntax_Build"     COMMAND test_syntax_build)
            add_test(NAME "T3_Memory_Stress"    COMMAND test_memory_stress)
            add_test(NAME "T4_Large_Data_Scale" COMMAND test_large_data_scale)
            add_test(NAME "T5_CUDA_Kernels"     COMMAND test_cuda_kernels)
        endif()
    else()
        # Default: bina compute-sanitizer (fast, CI default)
        add_test(NAME "T1_Syntax_Build"     COMMAND test_syntax_build)
        add_test(NAME "T3_Memory_Stress"    COMMAND test_memory_stress)
        add_test(NAME "T4_Large_Data_Scale" COMMAND test_large_data_scale)
        add_test(NAME "T5_CUDA_Kernels"     COMMAND test_cuda_kernels)
    endif()

    set_tests_properties("T1_Syntax_Build"     PROPERTIES TIMEOUT 300  LABELS "GPU")
    set_tests_properties("T3_Memory_Stress"    PROPERTIES TIMEOUT 600  LABELS "GPU")
    set_tests_properties("T4_Large_Data_Scale" PROPERTIES TIMEOUT 900  LABELS "GPU")
    set_tests_properties("T5_CUDA_Kernels"     PROPERTIES TIMEOUT 300  LABELS "GPU")

    # ── CTest resource guard: GPU tests serialize on device 0 ──
    set_tests_properties(
        "T1_Syntax_Build"
        "T3_Memory_Stress"
        "T4_Large_Data_Scale"
        "T5_CUDA_Kernels"
        PROPERTIES RESOURCE_LOCK "GPU_DEVICE_0"
    )

    message(STATUS "")
    message(STATUS "CTest targets:")
    message(STATUS "  CPU+sanitizer: ctest -LE GPU -DLOGOS_SANITIZE=ON --output-on-failure")
    message(STATUS "  GPU normal:    ctest -L GPU --output-on-failure")
    message(STATUS "  GPU+memcheck:  cmake -DLOGOS_USE_COMPUTE_SANITIZER=ON .. && ctest -L GPU")
    message(STATUS "  All:           ctest --output-on-failure")
    message(STATUS "")

else()
    message(STATUS "⚠️  CUDA not found — CPU-only build")

    # CPU-only targets
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

    # test_large_data_scale compiles as CXX in CPU-only mode
    add_executable(test_large_data_scale
        tests/test_large_data_scale.cu
        src/Checkpoint.cpp
    )
    target_include_directories(test_large_data_scale PRIVATE include tests)
    set_source_files_properties(
        tests/test_large_data_scale.cu
        PROPERTIES LANGUAGE CXX
    )
    logos_set_cpu_test_options(test_large_data_scale)

    enable_testing()
    add_test(NAME "T2_Math_Unit"        COMMAND test_math_unit)
    add_test(NAME "T2b_Checkpoint"      COMMAND test_checkpoint)
    add_test(NAME "T4_Large_Data_Scale" COMMAND test_large_data_scale)
    set_tests_properties("T2_Math_Unit"        PROPERTIES TIMEOUT 60)
    set_tests_properties("T2b_Checkpoint"      PROPERTIES TIMEOUT 60)
    set_tests_properties("T4_Large_Data_Scale" PROPERTIES TIMEOUT 600)

    message(STATUS "CPU-only tests: T2_Math_Unit, T2b_Checkpoint, T4_Large_Data_Scale")
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
message(STATUS "  test_syntax_build      — T1: CUDA syntax + GPU kernels")
message(STATUS "  test_math_unit         — T2: Vedic math formulas (CPU)")
message(STATUS "  test_checkpoint        — T2b: checkpoint roundtrip (CPU)")
message(STATUS "  test_memory_stress     — T3: 1M+ data, VRAM leak, bounds")
message(STATUS "  test_large_data_scale  — T4: 1TB scale, production CI/CD")
message(STATUS "  test_cuda_kernels      — T5: kernel correctness + cross-file")
endif()
message(STATUS "")
