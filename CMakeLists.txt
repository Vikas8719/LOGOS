cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.4.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

# ── CPU executable (existing) ────────────────────────────────
add_executable(logos src/main.cpp)
if(APPLE)
    target_compile_options(logos PRIVATE -O3)
else()
    target_compile_options(logos PRIVATE -O3 -march=x86-64)
endif()

# ── GPU executable (CUDA) ────────────────────────────────────
# Only build if CUDA is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "✅ CUDA found — building logos_gpu")

    # Detect GPU architecture automatically
    # T4 = sm_75, A100 = sm_80, V100 = sm_70
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")

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
            -Xcompiler -O3
            --ptxas-options=-v
        >
    )

    target_link_libraries(logos_gpu PRIVATE cudart)

    message(STATUS "GPU build: ./build/logos_gpu --train dataset.txt")
else()
    message(STATUS "⚠️  CUDA not found — only CPU build (logos)")
    message(STATUS "   On Kaggle: nvcc should be available")
endif()

install(TARGETS logos DESTINATION bin)
if(CMAKE_CUDA_COMPILER)
    install(TARGETS logos_gpu DESTINATION bin)
endif()

message(STATUS "CPU build:  ./build/logos      --train dataset.txt")
