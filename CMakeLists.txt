cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.5.0 LANGUAGES CXX)

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

# ── GPU executable (CUDA) ────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "✅ CUDA found — building logos_gpu")

    # BUG 5 FIX: CUDA Architecture Strategy
    #
    # Option A — "native" (LOCAL / KAGGLE BUILD — RECOMMENDED):
    #   Detects the exact GPU on this machine and builds only for it.
    #   Fastest compile, smallest binary, best performance.
    #   Use: cmake -DCUDA_ARCH_MODE=native ..
    #
    # Option B — Full explicit list (RELEASE / DISTRIBUTION BUILD):
    #   Covers P100 (60), V100 (70), T4/RTX2080 (75), A100 (80),
    #   RTX3090 (86), RTX4090 (89), H100 (90).
    #   Slower compile, larger binary, runs on all GPUs.
    #   Use: cmake -DCUDA_ARCH_MODE=release ..
    #
    # Pehle: "75;80;86" hardcoded
    #   → P100 (cc60) miss, H100 (cc90) miss, Ada (cc89) miss
    # Ab: auto-detect OR full list based on CUDA_ARCH_MODE variable

    set(CUDA_ARCH_MODE "native" CACHE STRING
        "CUDA architecture mode: 'native' (auto-detect) or 'release' (all GPUs)")

    if(CUDA_ARCH_MODE STREQUAL "release")
        # Release: supports P100, V100, T4, A100, RTX 30/40, H100
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch mode: RELEASE — compiling for: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Native: auto-detect current GPU (local/Kaggle default)
        set(CMAKE_CUDA_ARCHITECTURES "native"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch mode: NATIVE — auto-detecting GPU architecture")
    endif()

    add_executable(logos_gpu
        cuda/train_gpu.cu
        cuda/VedicGEMM.cu
        cuda/ModelGPU.cu
    )

    target_include_directories(logos_gpu PRIVATE include cuda)

    target_compile_options(logos_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xcompiler=-O3
            --ptxas-options=-v
            # BUG 3 FIX: Enable device-side exception support
            # Required for proper error propagation from CUDA kernels
            --extended-lambda
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
        >
    )

    # C++ exceptions needed for our CUDA_CHECK throw pattern
    target_compile_definitions(logos_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:>
    )

    target_link_libraries(logos_gpu PRIVATE cudart)

    message(STATUS "GPU build: ./build/logos_gpu --train dataset.txt")
    message(STATUS "To build for all GPUs: cmake -DCUDA_ARCH_MODE=release ..")
else()
    message(STATUS "⚠️  CUDA not found — only CPU build (logos)")
    message(STATUS "   On Kaggle: nvcc should be in /usr/local/cuda/bin/")
endif()

install(TARGETS logos DESTINATION bin)
if(CMAKE_CUDA_COMPILER)
    install(TARGETS logos_gpu DESTINATION bin)
endif()

message(STATUS "CPU build:  ./build/logos      --train dataset.txt")
message(STATUS "GPU build:  ./build/logos_gpu  --train dataset.txt")
