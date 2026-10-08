cmake_minimum_required(VERSION 3.20)
project(LOGOS VERSION 0.13.0 LANGUAGES CXX)
# v0.13.0 — MULL Mutation Testing wired
#
#   ASan+UBSan: CPU tests (T2_Math_Unit, T2b_Checkpoint) mein ON
#     cmake -DLOGOS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug ..
#     ctest -LE GPU   ← CPU tests run honge sanitizers ke saath
#
#   CUDA GPU tests: compute-sanitizer via CTest environment
#     LOGOS_USE_COMPUTE_SANITIZER=ON set karo
#     ctest -L GPU   ← compute-sanitizer --tool memcheck se wrap hoga
#
#   MULL Mutation Testing: test suite ki weakness dhundta hai
#     cmake -DLOGOS_MUTATION_TEST=ON -DCMAKE_BUILD_TYPE=Debug ..
#     cmake --build build --target mull-run
#     ← Automatically mutants banata hai + test_math_unit + test_checkpoint chalata hai
#     ← Report: build/mull-report/index.html
#
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

# ═══════════════════════════════════════════════════════════════
# MULL — Mutation Testing
# ═══════════════════════════════════════════════════════════════
#
# Kya karta hai:
#   test_math_unit.cpp aur test_checkpoint.cpp pe MULL mutations chalata hai.
#   Target source files jo mutate hote hain:
#     src/VedicGEMM.cpp      ← Vedic GEMM formulas (M1, M2, M9)
#     src/Checkpoint.cpp     ← checkpoint save/load roundtrip (T2b)
#     include/VedicGEMM.hpp  ← tiling logic
#     include/Tensor.hpp     ← tensor ops (free energy, softmax)
#     include/PhysicsOpt.hpp ← optimizer physics (M11, M14, M15, M16)
#     include/LayerNorm.hpp  ← LayerNorm stats (M8, M13)
#     include/Attention.hpp  ← NS attention (M12)
#
# Kaise use karein:
#   # Step 1: MULL install karo (Linux/macOS)
#   #   Ubuntu:  sudo apt-get install mull  (LLVM apt repo se)
#   #   macOS:   brew install mull
#   #   Manual:  https://github.com/mull-project/mull/releases
#   #
#   # Step 2: Build karo
#   cmake -B build -DLOGOS_MUTATION_TEST=ON -DCMAKE_BUILD_TYPE=Debug \
#         -DCMAKE_CXX_COMPILER=clang++ ..
#   cmake --build build --target mull-run
#   #
#   # Step 3: Report dekho
#   open build/mull-report/index.html   (macOS)
#   xdg-open build/mull-report/index.html  (Linux)
#   #
#   # Expected output:
#   #   Mutation score: XX%  (target: > 80%)
#   #   Killed:   N   ← good (tests ne pakad liya)
#   #   Survived: M   ← bad  (test gap hai)
#   #   Timeout:  K
#   #
#   # NOTE: MULL ko clang++ chahiye (GCC se kaam nahi karta)
#   #       Debug build zaruri hai (Release mein LLVM IR alag hota hai)
#
# Mutation operators (MULL default set):
#   cxx_add_to_sub     : a + b  → a - b  (e.g., CE + T*S badal CE - T*S)
#   cxx_mul_to_div     : a * b  → a / b  (e.g., T*S → T/S)
#   cxx_sub_to_add     : a - b  → a + b
#   cxx_gt_to_ge       : >  → >=         (loop boundary off-by-one)
#   cxx_lt_to_le       : <  → <=         (tiling loop ka edge case)
#   cxx_logical_and_to_or: &&  → ||      (condition flip)
#   cxx_remove_void_call : function call hata deta hai
#   cxx_negate_mutator : !cond → cond    (condition negate)
#
# ── MULL option ─────────────────────────────────────────────────
option(LOGOS_MUTATION_TEST "Enable MULL mutation testing (requires clang++ + mull-runner)" OFF)

if(LOGOS_MUTATION_TEST)
    # ── Requirement checks ──────────────────────────────────────
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "[MULL] clang++ chahiye, ${CMAKE_CXX_COMPILER_ID} nahi.\n"
            "       Dobara configure karo:\n"
            "       cmake -DLOGOS_MUTATION_TEST=ON "
            "-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..")
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING
            "[MULL] Debug build recommended (current: ${CMAKE_BUILD_TYPE}).\n"
            "       Release mein LLVM IR optimized hota hai aur mutations miss ho sakte hain.\n"
            "       Sahi command:\n"
            "       cmake -DLOGOS_MUTATION_TEST=ON -DCMAKE_BUILD_TYPE=Debug ..")
    endif()

    find_program(MULL_RUNNER_EXE
        NAMES mull-runner mull-runner-15 mull-runner-14 mull-runner-16 mull-runner-17
        PATHS
            /usr/bin
            /usr/local/bin
            /opt/homebrew/bin
            /usr/lib/mull
            $ENV{MULL_PATH}
    )

    if(NOT MULL_RUNNER_EXE)
        message(FATAL_ERROR
            "[MULL] mull-runner nahi mila.\n"
            "       Install karo:\n"
            "         Ubuntu 22.04:\n"
            "           wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -\n"
            "           sudo add-apt-repository 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main'\n"
            "           sudo apt-get update && sudo apt-get install mull-17\n"
            "         macOS:\n"
            "           brew install mull\n"
            "         Manual:\n"
            "           https://github.com/mull-project/mull/releases\n"
            "       Ya MULL_PATH env var set karo.")
    endif()

    message(STATUS "✅ MULL found: ${MULL_RUNNER_EXE}")

    # ── Mutation target: test_math_unit (M1-M16 sab) ───────────
    # -femit-all-decls    : dead code eliminate na ho (mutations miss ho sakte hain)
    # -fprofile-instr-generate: LLVM instrumentation for mull
    # -fcoverage-mapping  : line coverage data
    # -g -O0              : debug symbols, no optimization (mutations readable hote hain)
    # ── GCC-12 toolchain path detect karo (clang ABI fix ke liye) ──
    # Problem: ubuntu-22.04 pe clang-15 default GCC-14 ka libstdc++ use karta hai.
    # GCC-14 mein std::__shared_ptr<_Dir>::operator bool() inline ho gaya —
    # yeh symbol .so mein nahi hota, isliye undefined reference linker error aata hai.
    # Fix: clang ko explicitly GCC-12 ka sysroot/toolchain use karwao.
    # --gcc-toolchain=<path>: clang GCC-12 ke headers + libstdc++ dhundta hai.
    find_program(GCC12_EXE NAMES g++-12 PATHS /usr/bin)
    if(GCC12_EXE)
        execute_process(
            COMMAND ${GCC12_EXE} -print-file-name=libstdc++.a
            OUTPUT_VARIABLE _GCC12_LIBPATH OUTPUT_STRIP_TRAILING_WHITESPACE)
        get_filename_component(_GCC12_TOOLCHAIN "${_GCC12_LIBPATH}" DIRECTORY)
        # toolchain root = lib dir se 3 levels upar (lib/gcc/x86_64-linux-gnu/12 → /)
        get_filename_component(_GCC12_TOOLCHAIN "${_GCC12_TOOLCHAIN}" DIRECTORY)
        get_filename_component(_GCC12_TOOLCHAIN "${_GCC12_TOOLCHAIN}" DIRECTORY)
        get_filename_component(_GCC12_TOOLCHAIN "${_GCC12_TOOLCHAIN}" DIRECTORY)
        message(STATUS "[MULL] GCC-12 toolchain: ${_GCC12_TOOLCHAIN}")
        set(MULL_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=${_GCC12_TOOLCHAIN}")
    else()
        message(STATUS "[MULL] g++-12 nahi mila — system default GCC use hoga")
        set(MULL_GCC_TOOLCHAIN_FLAG "")
    endif()

    add_executable(mull_math_unit
        tests/test_math_unit.cpp
        src/VedicGEMM.cpp
        src/Checkpoint.cpp
    )
    target_include_directories(mull_math_unit PRIVATE include tests)
    target_compile_options(mull_math_unit PRIVATE
        -O0
        -g
        -grecord-gcc-switches
        -fprofile-instr-generate
        -fcoverage-mapping
        -femit-all-decls
        -fno-omit-frame-pointer
        -std=c++20
        -fexperimental-new-pass-manager   # LLVM new pass manager (mull needs this)
        ${MULL_GCC_TOOLCHAIN_FLAG}        # clang ko GCC-12 libstdc++ use karwao
    )
    target_link_options(mull_math_unit PRIVATE
        -fprofile-instr-generate
        -Wl,--no-as-needed
        ${MULL_GCC_TOOLCHAIN_FLAG}        # link time bhi GCC-12 toolchain
    )
    # stdc++fs pehle, phir stdc++ (dependency order correct rakhna zaroori hai)
    target_link_libraries(mull_math_unit PRIVATE stdc++fs stdc++)

    # ── Mutation target: test_checkpoint (T2b roundtrip) ────────
    add_executable(mull_checkpoint
        tests/test_checkpoint.cpp
        src/Checkpoint.cpp
    )
    target_include_directories(mull_checkpoint PRIVATE include)
    target_compile_options(mull_checkpoint PRIVATE
        -O0
        -g
        -grecord-gcc-switches
        -fprofile-instr-generate
        -fcoverage-mapping
        -femit-all-decls
        -fno-omit-frame-pointer
        -std=c++20
        -fexperimental-new-pass-manager
        ${MULL_GCC_TOOLCHAIN_FLAG}        # clang ko GCC-12 libstdc++ use karwao
    )
    target_link_options(mull_checkpoint PRIVATE
        -fprofile-instr-generate
        # clang-15 + GCC-14 ABI mismatch fix:
        # std::__shared_ptr<_Dir>::operator bool() GCC-14 mein inline ho gaya.
        # GCC-12 toolchain force karo taaki yeh symbol available ho.
        -Wl,--no-as-needed
        ${MULL_GCC_TOOLCHAIN_FLAG}        # link time bhi GCC-12 toolchain
    )
    # stdc++fs pehle (filesystem), phir stdc++ (base runtime) — order matters.
    target_link_libraries(mull_checkpoint PRIVATE stdc++fs stdc++)

    # ── MULL config file ────────────────────────────────────────
    # mull.yml: kaunse operators chalein, kaunsi files mutate hon,
    #           timeout, parallel workers, report format
    set(MULL_CONFIG_PATH "${CMAKE_BINARY_DIR}/mull.yml")
    set(MULL_REPORT_DIR  "${CMAKE_BINARY_DIR}/mull-report")

    # Source files jo MULL mutate karega (sirf yeh files, test files nahi)
    # Relative to project source dir
    set(MULL_INCLUDE_SOURCES
        "${CMAKE_SOURCE_DIR}/src/VedicGEMM.cpp"
        "${CMAKE_SOURCE_DIR}/src/Checkpoint.cpp"
        "${CMAKE_SOURCE_DIR}/include/VedicGEMM.hpp"
        "${CMAKE_SOURCE_DIR}/include/Tensor.hpp"
        "${CMAKE_SOURCE_DIR}/include/PhysicsOpt.hpp"
        "${CMAKE_SOURCE_DIR}/include/LayerNorm.hpp"
        "${CMAKE_SOURCE_DIR}/include/Attention.hpp"
    )

    # mull.yml banao (configure_file se — build time pe generate hota hai)
    # Yeh file mull-runner ko batata hai:
    #   - kaunse mutation operators use karo
    #   - kaunsi source files mutate karo
    #   - timeout per mutant
    #   - output format
    file(WRITE "${MULL_CONFIG_PATH}" "# LOGOS MULL Mutation Testing Config
# Auto-generated by CMakeLists.txt (LOGOS_MUTATION_TEST=ON)
# Manually edit karne ki zarurat nahi — cmake se regenerate hota hai

mutators:
  # Arithmetic mutations (Free Energy, GEMM formulas)
  - cxx_add_to_sub          # a+b → a-b  : CE+T*S catch karega M3
  - cxx_sub_to_add          # a-b → a+b  : F=CE-T*S catch karega M3/M10
  - cxx_mul_to_div          # a*b → a/b  : scaling mutations
  - cxx_div_to_mul          # a/b → a*b  : inverse scaling
  # Relational mutations (loop boundaries, tiling)
  - cxx_gt_to_ge            # >  → >=    : off-by-one in GEMM loops
  - cxx_lt_to_le            # <  → <=    : tiling edge case M9
  - cxx_ge_to_gt            # >= → >
  - cxx_le_to_lt            # <= → <
  - cxx_eq_to_ne            # == → !=
  - cxx_ne_to_eq            # != → ==
  # Logical mutations (conditions)
  - cxx_logical_and_to_or   # && → ||    : guard condition flip
  - cxx_logical_or_to_and   # || → &&
  - cxx_negate_mutator      # !x → x     : negation removal
  # Call removal (void functions)
  - cxx_remove_void_call    # function() hata deta hai

timeout_per_test_ms: 10000    # 10s per mutant per test (M16 ke liye kaafi)
jobs: 4                        # parallel mutant runners
include_paths:
  # Sirf in files ke mutations chalenge (test files exclude hain)
  - \"${CMAKE_SOURCE_DIR}/src/VedicGEMM.cpp\"
  - \"${CMAKE_SOURCE_DIR}/src/Checkpoint.cpp\"
  - \"${CMAKE_SOURCE_DIR}/include/VedicGEMM.hpp\"
  - \"${CMAKE_SOURCE_DIR}/include/Tensor.hpp\"
  - \"${CMAKE_SOURCE_DIR}/include/PhysicsOpt.hpp\"
  - \"${CMAKE_SOURCE_DIR}/include/LayerNorm.hpp\"
  - \"${CMAKE_SOURCE_DIR}/include/Attention.hpp\"
exclude_paths:
  # Test files mutate mat karo (mutation mein test code change hone se wrong results)
  - \"${CMAKE_SOURCE_DIR}/tests/\"
  - \"${CMAKE_SOURCE_DIR}/cuda/\"
reporters:
  - IDE          # stdout: mutant location + result (killed/survived)
  - HTML         # build/mull-report/index.html (browser mein dekho)
report_dir: \"${MULL_REPORT_DIR}\"
")

    # ── mull-run custom target ──────────────────────────────────
    # cmake --build build --target mull-run
    # Do steps:
    #   1. mull_math_unit + mull_checkpoint build karo
    #   2. mull-runner dono binaries pe chalao
    #   3. HTML report generate karo

    add_custom_target(mull-run
        # Step 1: report dir banao
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MULL_REPORT_DIR}"

        # Step 2: mull_math_unit pe mull-runner chalao (M1-M16 tests)
        COMMAND ${MULL_RUNNER_EXE}
            --config "${MULL_CONFIG_PATH}"
            --report-dir "${MULL_REPORT_DIR}/math"
            --test-program "$<TARGET_FILE:mull_math_unit>"
            --ld-search-path "${CMAKE_BINARY_DIR}"
        COMMENT "[MULL 1/2] Mutation testing: test_math_unit (M1-M16)"

        # Step 3: mull_checkpoint pe mull-runner chalao (T2b checkpoint)
        COMMAND ${MULL_RUNNER_EXE}
            --config "${MULL_CONFIG_PATH}"
            --report-dir "${MULL_REPORT_DIR}/checkpoint"
            --test-program "$<TARGET_FILE:mull_checkpoint>"
            --ld-search-path "${CMAKE_BINARY_DIR}"
        COMMENT "[MULL 2/2] Mutation testing: test_checkpoint (T2b)"

        # Step 4: summary print karo
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "══════════════════════════════════════════"
        COMMAND ${CMAKE_COMMAND} -E echo "  MULL Mutation Testing Complete"
        COMMAND ${CMAKE_COMMAND} -E echo "  Reports:"
        COMMAND ${CMAKE_COMMAND} -E echo "    Math:       ${MULL_REPORT_DIR}/math/index.html"
        COMMAND ${CMAKE_COMMAND} -E echo "    Checkpoint: ${MULL_REPORT_DIR}/checkpoint/index.html"
        COMMAND ${CMAKE_COMMAND} -E echo "══════════════════════════════════════════"

        DEPENDS mull_math_unit mull_checkpoint
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        VERBATIM
    )

    # ── CTest integration: mull-run ko CTest mein bhi add karo ─
    # ctest -L MUTATION se chalega
    # Note: mull lamba chalta hai (sabhi mutants × tests) → timeout 1800s
    enable_testing()
    add_test(NAME "T6_Mutation_Math"
        COMMAND ${MULL_RUNNER_EXE}
            --config "${MULL_CONFIG_PATH}"
            --report-dir "${MULL_REPORT_DIR}/math"
            --test-program "$<TARGET_FILE:mull_math_unit>"
            --ld-search-path "${CMAKE_BINARY_DIR}"
    )
    add_test(NAME "T6_Mutation_Checkpoint"
        COMMAND ${MULL_RUNNER_EXE}
            --config "${MULL_CONFIG_PATH}"
            --report-dir "${MULL_REPORT_DIR}/checkpoint"
            --test-program "$<TARGET_FILE:mull_checkpoint>"
            --ld-search-path "${CMAKE_BINARY_DIR}"
    )
    set_tests_properties("T6_Mutation_Math"
        PROPERTIES
            TIMEOUT 1800
            LABELS "MUTATION"
            PASS_REGULAR_EXPRESSION "Mutation score: [89][0-9]\\.[0-9]+%|Mutation score: 100"
    )
    set_tests_properties("T6_Mutation_Checkpoint"
        PROPERTIES
            TIMEOUT 600
            LABELS "MUTATION"
            PASS_REGULAR_EXPRESSION "Mutation score: [89][0-9]\\.[0-9]+%|Mutation score: 100"
    )

    message(STATUS "")
    message(STATUS "✅ MULL Mutation Testing ENABLED")
    message(STATUS "   Runner  : ${MULL_RUNNER_EXE}")
    message(STATUS "   Targets : mull_math_unit, mull_checkpoint")
    message(STATUS "   Config  : ${MULL_CONFIG_PATH}")
    message(STATUS "   Report  : ${MULL_REPORT_DIR}/*/index.html")
    message(STATUS "")
    message(STATUS "   Run:    cmake --build build --target mull-run")
    message(STATUS "   CTest:  ctest -L MUTATION --output-on-failure")
    message(STATUS "   Goal:   Mutation score > 80% (90%+ excellent)")
    message(STATUS "")
    message(STATUS "   Operators (${CMAKE_SOURCE_DIR}/src + include):")
    message(STATUS "     cxx_add_to_sub / cxx_sub_to_add  ← Free Energy formula")
    message(STATUS "     cxx_mul_to_div / cxx_div_to_mul  ← GEMM scaling")
    message(STATUS "     cxx_gt_to_ge / cxx_lt_to_le      ← loop boundary")
    message(STATUS "     cxx_logical_and_to_or             ← guard conditions")
    message(STATUS "     cxx_remove_void_call              ← checkpoint write skip")
    message(STATUS "")

endif() # LOGOS_MUTATION_TEST

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
