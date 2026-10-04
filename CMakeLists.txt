cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.11.0 LANGUAGES CXX)
# v0.11.0 — CTest fully wired: T5 cuda_kernels added
#   T5 covers: add_bias_kernel cross-file launch, AMP path,
#   training/validation forward modes, SHM optimizer, adjoint
#   gradient checks (ns_advect, ns_diffuse), FP16 cast clamp,
#   free_energy_loss numerical gradient, expmap0 ball constraint,
#   Gunitasamuchayah large-matrix, E2E forward+loss+step.

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
        message(STATUS "  sm_90 = H100 SXM/PCIe (Hopper)")
        message(STATUS "  sm_89 = RTX 4090 (Ada Lovelace)")
        message(STATUS "  sm_86 = A10/A30/RTX 3090 (Ampere)")
        message(STATUS "  sm_80 = A100 (Ampere)")
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
            --expt-relaxed-constexpr
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
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
    target_compile_options(logos_core PRIVATE ${LOGOS_CUDA_FLAGS})
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
    target_compile_options(logos_gpu PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(logos_gpu PRIVATE logos_core)
    target_compile_definitions(logos_gpu PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 1: Syntax & Build Test ───────────────────────────
    add_executable(test_syntax_build
        tests/test_syntax_build.cu
    )
    target_include_directories(test_syntax_build PRIVATE include cuda tests)
    target_compile_options(test_syntax_build PRIVATE ${LOGOS_CUDA_FLAGS})
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
    target_compile_options(test_memory_stress PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_memory_stress PRIVATE logos_core)
    target_compile_definitions(test_memory_stress PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── TEST 4: Large Data Scale & Production Test ────────────
    add_executable(test_large_data_scale
        tests/test_large_data_scale.cu
    )
    target_include_directories(test_large_data_scale PRIVATE include cuda tests)
    target_compile_options(test_large_data_scale PRIVATE ${LOGOS_CUDA_FLAGS})
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
    target_compile_options(test_cuda_kernels PRIVATE ${LOGOS_CUDA_FLAGS})
    target_link_libraries(test_cuda_kernels PRIVATE logos_core)
    target_compile_definitions(test_cuda_kernels PRIVATE
        LOGOS_USE_CUBLAS=$<BOOL:${LOGOS_USE_CUBLAS}>
        LOGOS_CUBLAS_TF32=$<BOOL:${LOGOS_CUBLAS_TF32}>)

    # ── CTest integration ─────────────────────────────────────
    enable_testing()

    # ── CPU tests (always run, no GPU needed) ─────────────────
    add_test(NAME "T2_Math_Unit"        COMMAND test_math_unit)
    add_test(NAME "T2b_Checkpoint"      COMMAND test_checkpoint)
    set_tests_properties("T2_Math_Unit"   PROPERTIES TIMEOUT 60)
    set_tests_properties("T2b_Checkpoint" PROPERTIES TIMEOUT 60)

    # ── GPU tests (require CUDA device) ──────────────────────
    # LABELS used by CI to run GPU-only subset:
    #   ctest -L GPU --output-on-failure
    add_test(NAME "T1_Syntax_Build"     COMMAND test_syntax_build)
    add_test(NAME "T3_Memory_Stress"    COMMAND test_memory_stress)
    add_test(NAME "T4_Large_Data_Scale" COMMAND test_large_data_scale)
    add_test(NAME "T5_CUDA_Kernels"     COMMAND test_cuda_kernels)

    set_tests_properties("T1_Syntax_Build"     PROPERTIES TIMEOUT 120  LABELS "GPU")
    set_tests_properties("T3_Memory_Stress"    PROPERTIES TIMEOUT 300  LABELS "GPU")
    set_tests_properties("T4_Large_Data_Scale" PROPERTIES TIMEOUT 600  LABELS "GPU")
    set_tests_properties("T5_CUDA_Kernels"     PROPERTIES TIMEOUT 180  LABELS "GPU")

    # ── CTest resource guard: GPU tests serialize on device 0 ──
    # Prevents two GPU tests from fighting over VRAM on single-GPU runners
    set_tests_properties(
        "T1_Syntax_Build"
        "T3_Memory_Stress"
        "T4_Large_Data_Scale"
        "T5_CUDA_Kernels"
        PROPERTIES RESOURCE_LOCK "GPU_DEVICE_0"
    )

    message(STATUS "")
    message(STATUS "CTest targets:")
    message(STATUS "  CPU (always): T2_Math_Unit, T2b_Checkpoint")
    message(STATUS "  GPU (need CUDA device): T1 T3 T4 T5")
    message(STATUS "  Run all:      ctest --output-on-failure")
    message(STATUS "  Run GPU only: ctest -L GPU --output-on-failure")
    message(STATUS "  Run CPU only: ctest -LE GPU --output-on-failure")
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
