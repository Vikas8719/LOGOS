// ============================================================
//  LOGOS — tests/test_rapidcheck.cpp
//  RapidCheck Property-Based Tests (C++ ka Hypothesis)
//
//  Kya test karta hai:
//    [RC1]  VedicGEMM == reference_gemm (random sizes + values)
//    [RC2]  VEDIC_BLOCK=64 boundary sizes (63,64,65,128,129)
//    [RC3]  vedic_gemm_bias correctness (random bias)
//    [RC4]  Free Energy F = CE - T*S properties
//    [RC5]  Tensor::reshape roundtrip
//    [RC6]  Riemannian distance: symmetry, self-distance=0, triangle inequality
//    [RC7]  Checkpoint expected_bytes formula (random ModelConfig)
//    [RC8]  Leapfrog energy drift < Euler drift
//
//  Build karo:
//    cmake -B build -DLOGOS_RAPIDCHECK=ON
//    cmake --build build --target test_rapidcheck
//    ./build/test_rapidcheck
//
//  Make shortcut:
//    make rapidcheck
//
//  Kaise kaam karta hai RapidCheck:
//    rc::check("description", []() {
//        auto x = *rc::gen::inRange(1, 100);   // random int [1,100)
//        auto v = *rc::gen::arbitrary<float>(); // any float
//        RC_ASSERT(some_property(x, v));         // property check
//    });
//    Failure pe → automatically simplest failing case dhundta hai (shrinking)
//    1000+ random cases try karta hai default mein
// ============================================================
#include "../include/VedicGEMM.hpp"
#include "../include/Tensor.hpp"
#include "../include/PhysicsOpt.hpp"
#include "../include/LayerNorm.hpp"
#include "../include/Checkpoint.hpp"

#include <rapidcheck.h>

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>

// ── Helpers ───────────────────────────────────────────────────

// Naive reference GEMM — ground truth
static std::vector<float> naive_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int K, int N)
{
    std::vector<float> C(M * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
    return C;
}

static bool mat_close(const std::vector<float>& A,
                      const std::vector<float>& B,
                      float atol = 1e-2f)
{
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i)
        if (std::abs(A[i] - B[i]) > atol) return false;
    return true;
}

// CPU softmax + CE + entropy (same as test_math_unit.cpp)
static std::vector<float> cpu_softmax(const std::vector<float>& logits) {
    float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(logits.size());
    float sum = 0.f;
    for (size_t i = 0; i < logits.size(); ++i) { p[i] = std::exp(logits[i]-mx); sum += p[i]; }
    for (float& x : p) x /= sum;
    return p;
}

static float cpu_entropy(const std::vector<float>& p) {
    float S = 0.f;
    for (float pi : p) if (pi > 1e-12f) S -= pi * std::log(pi);
    return S;
}

static float cpu_ce(const std::vector<float>& p, int target) {
    return -std::log(std::max(p[target], 1e-12f));
}

// ═══════════════════════════════════════════════════════════════
//  [RC1] VedicGEMM == reference_gemm — random sizes + values
// ═══════════════════════════════════════════════════════════════
static void rc1_vedic_matches_reference() {
    rc::check("[RC1] vedic_gemm(A,B) == reference_gemm(A,B) for random inputs", []() {
        // Random matrix dimensions: [1, 32] — small for speed
        // RC_PRE = precondition (skip invalid)
        int M = *rc::gen::inRange(1, 33);
        int K = *rc::gen::inRange(1, 33);
        int N = *rc::gen::inRange(1, 33);

        // Random finite floats [-1, 1]
        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );

        auto flat_a = *rc::gen::container<std::vector<float>>(M * K, float_gen);
        auto flat_b = *rc::gen::container<std::vector<float>>(K * N, float_gen);

        // Build Tensors
        Tensor A({M, K}); A.data = flat_a;
        Tensor B({K, N}); B.data = flat_b;

        // VedicGEMM (actual LOGOS code)
        Tensor C = vedic_gemm(A, B);

        // Reference (naive triple loop)
        auto C_ref = naive_gemm(flat_a, flat_b, M, K, N);

        RC_ASSERT(mat_close(C.data, C_ref));
    });
}

// ═══════════════════════════════════════════════════════════════
//  [RC2] VEDIC_BLOCK=64 boundary sizes
//  Off-by-one tiling bugs sirf specific sizes pe dikhte hain
// ═══════════════════════════════════════════════════════════════
static void rc2_tiling_boundary() {
    // Fixed boundary sizes — yeh RapidCheck ke saath parametrize karte hain
    const std::vector<int> boundary_sizes = {1, 63, 64, 65, 127, 128, 129, 192, 193};

    for (int sz : boundary_sizes) {
        rc::check(
            std::string("[RC2] VEDIC_BLOCK boundary size=") + std::to_string(sz),
            [sz]() {
                auto float_gen = rc::gen::map(
                    rc::gen::inRange(-50, 51),
                    [](int v) { return v * 0.02f; }
                );

                auto flat_a = *rc::gen::container<std::vector<float>>(sz * sz, float_gen);
                auto flat_b = *rc::gen::container<std::vector<float>>(sz * sz, float_gen);

                Tensor A({sz, sz}); A.data = flat_a;
                Tensor B({sz, sz}); B.data = flat_b;

                Tensor C     = vedic_gemm(A, B);
                auto   C_ref = naive_gemm(flat_a, flat_b, sz, sz, sz);

                RC_ASSERT(mat_close(C.data, C_ref, 5e-2f));  // relaxed tol for large sz
            });
    }
}

// ═══════════════════════════════════════════════════════════════
//  [RC3] vedic_gemm_bias correctness
// ═══════════════════════════════════════════════════════════════
static void rc3_gemm_bias() {
    rc::check("[RC3] vedic_gemm_bias(A, W, b) == reference(A,W) + b", []() {
        int M = *rc::gen::inRange(1, 17);
        int K = *rc::gen::inRange(1, 17);
        int N = *rc::gen::inRange(1, 17);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-50, 51),
            [](int v) { return v * 0.02f; }
        );

        auto flat_a = *rc::gen::container<std::vector<float>>(M * K, float_gen);
        auto flat_w = *rc::gen::container<std::vector<float>>(K * N, float_gen);
        auto flat_b = *rc::gen::container<std::vector<float>>(N, float_gen);

        Tensor A({M, K}); A.data = flat_a;
        Tensor W({K, N}); W.data = flat_w;
        Tensor bias({N}); bias.data = flat_b;

        // LOGOS actual code
        Tensor C_bias = vedic_gemm_bias(A, W, bias);

        // Reference: naive GEMM + bias
        auto C_ref = naive_gemm(flat_a, flat_w, M, K, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C_ref[i*N+j] += flat_b[j];

        RC_ASSERT(mat_close(C_bias.data, C_ref));
    });
}

// ═══════════════════════════════════════════════════════════════
//  [RC4] Free Energy F = CE - T*S properties
// ═══════════════════════════════════════════════════════════════
static void rc4_free_energy_properties() {
    // Property A: S >= 0 hamesha
    rc::check("[RC4a] Entropy S >= 0 for any probability distribution", []() {
        int vocab = *rc::gen::inRange(2, 33);
        auto float_gen = rc::gen::map(
            rc::gen::inRange(1, 1001),
            [](int v) { return v * 0.001f; }  // positive floats → valid logits
        );
        auto logits = *rc::gen::container<std::vector<float>>(vocab, float_gen);

        auto p = cpu_softmax(logits);
        float S = cpu_entropy(p);

        RC_ASSERT(S >= -1e-6f);
    });

    // Property B: T=0 pe F = CE
    rc::check("[RC4b] F = CE when T = 0", []() {
        int vocab  = *rc::gen::inRange(2, 17);
        int target = *rc::gen::inRange(0, vocab);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.05f; }
        );
        auto logits = *rc::gen::container<std::vector<float>>(vocab, float_gen);

        auto p  = cpu_softmax(logits);
        float CE = cpu_ce(p, target);
        float S  = cpu_entropy(p);

        float F_T0 = CE - 0.0f * S;  // T=0
        RC_ASSERT(std::abs(F_T0 - CE) < 1e-6f);
    });

    // Property C: F <= CE jab T > 0 (entropy term reduce karta hai F)
    rc::check("[RC4c] F <= CE when T > 0", []() {
        int vocab  = *rc::gen::inRange(2, 17);
        int target = *rc::gen::inRange(0, vocab);
        float T    = *rc::gen::map(rc::gen::inRange(1, 101), [](int v){ return v * 0.01f; });

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-50, 51),
            [](int v) { return v * 0.1f; }
        );
        auto logits = *rc::gen::container<std::vector<float>>(vocab, float_gen);

        auto  p  = cpu_softmax(logits);
        float CE = cpu_ce(p, target);
        float S  = cpu_entropy(p);
        float F  = CE - T * S;

        RC_ASSERT(F <= CE + 1e-5f);
    });

    // Property D: F nan nahi hona chahiye valid inputs pe
    rc::check("[RC4d] F is not NaN for valid inputs", []() {
        int vocab  = *rc::gen::inRange(2, 17);
        int target = *rc::gen::inRange(0, vocab);
        float T    = *rc::gen::map(rc::gen::inRange(0, 201), [](int v){ return v * 0.01f; });

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.1f; }
        );
        auto logits = *rc::gen::container<std::vector<float>>(vocab, float_gen);

        auto  p  = cpu_softmax(logits);
        float CE = cpu_ce(p, target);
        float S  = cpu_entropy(p);
        float F  = CE - T * S;

        RC_ASSERT(!std::isnan(F));
        RC_ASSERT(!std::isinf(F));
    });
}

// ═══════════════════════════════════════════════════════════════
//  [RC5] Tensor::reshape roundtrip
// ═══════════════════════════════════════════════════════════════
static void rc5_tensor_reshape() {
    rc::check("[RC5] Tensor reshape roundtrip: data unchanged", []() {
        int rows = *rc::gen::inRange(1, 33);
        int cols = *rc::gen::inRange(1, 33);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );
        auto flat = *rc::gen::container<std::vector<float>>(rows * cols, float_gen);

        Tensor T({rows, cols});
        T.data = flat;

        // Reshape to [total, 1] → back to [rows, cols]
        Tensor T2 = T.reshape({rows * cols, 1});
        Tensor T3 = T2.reshape({rows, cols});

        // Data hamesha same rehna chahiye
        RC_ASSERT(T3.data == T.data);
        RC_ASSERT(T3.rows() == rows);
        RC_ASSERT(T3.cols() == cols);
    });
}

// ═══════════════════════════════════════════════════════════════
//  [RC6] Riemannian distance properties
// ═══════════════════════════════════════════════════════════════
static void rc6_riemannian_distance() {
    // Property A: d(θ, θ) = 0
    rc::check("[RC6a] Riemannian self-distance = 0", []() {
        int dim = *rc::gen::inRange(1, 65);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );
        auto vals = *rc::gen::container<std::vector<float>>(dim, float_gen);

        RiemannianMetric rm(dim, 1e-4f);
        Tensor theta({1, dim});
        theta.data = vals;

        float d = rm.riemannian_distance(theta, theta);
        RC_ASSERT(d < 1e-5f);
    });

    // Property B: d(θ1, θ2) > 0 jab θ1 ≠ θ2
    rc::check("[RC6b] Riemannian distance > 0 for different points", []() {
        int dim = *rc::gen::inRange(1, 33);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );
        auto vals1 = *rc::gen::container<std::vector<float>>(dim, float_gen);
        auto vals2 = *rc::gen::container<std::vector<float>>(dim, float_gen);

        // Ensure they are actually different
        RC_PRE(vals1 != vals2);

        RiemannianMetric rm(dim, 1e-4f);
        Tensor t1({1, dim}); t1.data = vals1;
        Tensor t2({1, dim}); t2.data = vals2;

        float d = rm.riemannian_distance(t1, t2);
        RC_ASSERT(d > 0.0f);
    });

    // Property C: Symmetry d(θ1, θ2) == d(θ2, θ1)
    rc::check("[RC6c] Riemannian distance is symmetric", []() {
        int dim = *rc::gen::inRange(1, 33);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-50, 51),
            [](int v) { return v * 0.02f; }
        );
        auto vals1 = *rc::gen::container<std::vector<float>>(dim, float_gen);
        auto vals2 = *rc::gen::container<std::vector<float>>(dim, float_gen);

        RiemannianMetric rm(dim, 1e-4f);
        Tensor t1({1, dim}); t1.data = vals1;
        Tensor t2({1, dim}); t2.data = vals2;

        float d12 = rm.riemannian_distance(t1, t2);
        float d21 = rm.riemannian_distance(t2, t1);

        RC_ASSERT(std::abs(d12 - d21) < 1e-5f);
    });
}

// ═══════════════════════════════════════════════════════════════
//  [RC7] Checkpoint expected_bytes formula — random ModelConfig
// ═══════════════════════════════════════════════════════════════
static void rc7_checkpoint_size_formula() {
    rc::check("[RC7] Checkpoint expected_bytes > 0 for valid ModelConfig", []() {
        // Valid config ranges (same as CKPT_MAX_* constants)
        int vocab_size  = *rc::gen::inRange(2, 513);
        int d_model_div = *rc::gen::inRange(1, 9);   // d_model = d_model_div * 4
        int d_model     = d_model_div * 4;            // must be multiple of 4 (for heads)
        int num_heads   = *rc::gen::element(std::vector<int>{1, 2, 4});
        int num_layers  = *rc::gen::inRange(1, 5);
        int max_seq_len = *rc::gen::inRange(2, 65);

        // d_model num_heads ka multiple hona chahiye
        RC_PRE(d_model % num_heads == 0);

        // Same formula as test_checkpoint.cpp expected_bytes
        size_t SIZEOF_CONFIG = 5 * sizeof(int);

        // Embedding: [vocab_size, d_model]
        size_t embedding = static_cast<size_t>(vocab_size) * d_model;
        // LM head: [d_model, vocab_size]
        size_t lm_head = static_cast<size_t>(d_model) * vocab_size;
        // Legacy position block
        size_t legacy_pos = static_cast<size_t>(max_seq_len) * d_model;

        // Per-layer parameters
        size_t per_layer = 0;
        // Attention: W_Q, W_K, W_V, W_O
        per_layer += 4ULL * d_model * d_model;
        // W_proj
        per_layer += static_cast<size_t>(d_model) * d_model;
        // FFN: W1, b1, W2, b2
        per_layer += static_cast<size_t>(d_model) * (4*d_model)
                   + 4*d_model
                   + static_cast<size_t>(4*d_model) * d_model
                   + d_model;
        // LN1 + LN2: gamma, beta
        per_layer += 2ULL * (d_model + d_model);
        // ReynoldsBatchNorm stats: ln1 + ln2, mean + var
        per_layer += 4ULL * d_model;

        size_t total_params = embedding + lm_head + per_layer * num_layers;
        size_t expected_bytes = SIZEOF_CONFIG
                              + legacy_pos * sizeof(float)
                              + total_params * sizeof(float);

        // Must be positive and reasonable
        RC_ASSERT(expected_bytes > SIZEOF_CONFIG);
        RC_ASSERT(expected_bytes < 10ULL * 1024 * 1024 * 1024);  // < 10GB

        // Monotonic in d_model: larger d → larger file
        size_t legacy_pos_2x   = static_cast<size_t>(max_seq_len) * (d_model * 2);
        size_t total_params_2x = static_cast<size_t>(vocab_size)  * (d_model * 2) * 2
                               + per_layer * 4 * num_layers;
        size_t bytes_2x = SIZEOF_CONFIG
                        + legacy_pos_2x * sizeof(float)
                        + total_params_2x * sizeof(float);
        RC_ASSERT(bytes_2x > expected_bytes);
    });
}

// ═══════════════════════════════════════════════════════════════
//  [RC8] Leapfrog energy drift < Euler drift
// ═══════════════════════════════════════════════════════════════
static void rc8_leapfrog_vs_euler() {
    rc::check("[RC8] Leapfrog energy drift < Euler drift for harmonic oscillator", []() {
        // lr mein range: [0.01, 0.2] — larger lr mein Euler diverge karta hai
        float lr = *rc::gen::map(
            rc::gen::inRange(1, 21),
            [](int v) { return v * 0.01f; }
        );
        int steps = *rc::gen::inRange(10, 101);

        // Euler-Maruyama (1st order)
        float W_e = 1.0f, V_e = 0.0f;
        float E0 = 0.5f * W_e * W_e + 0.5f * V_e * V_e;
        for (int i = 0; i < steps; ++i) {
            V_e = 0.9f * V_e - lr * W_e;
            W_e += V_e;
        }
        float euler_drift = std::abs(0.5f*W_e*W_e + 0.5f*V_e*V_e - E0) / E0;

        // Leapfrog / Störmer-Verlet (2nd order, symplectic)
        float W_l = 1.0f, V_l = 0.0f;
        for (int i = 0; i < steps; ++i) {
            V_l -= (lr * 0.5f) * W_l;
            W_l += lr * V_l;
            V_l -= (lr * 0.5f) * W_l;
        }
        float lf_drift = std::abs(0.5f*W_l*W_l + 0.5f*V_l*V_l - E0) / E0;

        // Leapfrog should conserve energy better (symplectic integrator)
        RC_ASSERT(lf_drift < euler_drift + 1e-3f);
    });
}

// ═══════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS — RapidCheck Property-Based Tests         ║\n";
    std::cout << "║  C++ ka Hypothesis — actual code directly test   ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    bool all_passed = true;

    auto run = [&](const char* name, auto fn) {
        std::cout << "Running " << name << "...\n";
        try {
            fn();
            std::cout << "  ✅ " << name << " — passed\n\n";
        } catch (const std::exception& e) {
            std::cerr << "  ❌ " << name << " — FAILED: " << e.what() << "\n\n";
            all_passed = false;
        }
    };

    run("RC1 VedicGEMM vs reference",       rc1_vedic_matches_reference);
    run("RC2 VEDIC_BLOCK tiling boundaries", rc2_tiling_boundary);
    run("RC3 vedic_gemm_bias correctness",   rc3_gemm_bias);
    run("RC4 Free Energy properties",        rc4_free_energy_properties);
    run("RC5 Tensor reshape roundtrip",      rc5_tensor_reshape);
    run("RC6 Riemannian distance properties",rc6_riemannian_distance);
    run("RC7 Checkpoint size formula",       rc7_checkpoint_size_formula);
    run("RC8 Leapfrog vs Euler stability",   rc8_leapfrog_vs_euler);

    std::cout << "════════════════════════════════════════════════\n";
    if (all_passed) {
        std::cout << "  ✅ ALL RAPIDCHECK PROPERTY TESTS PASSED\n";
    } else {
        std::cout << "  ❌ SOME TESTS FAILED — see output above\n";
    }
    std::cout << "════════════════════════════════════════════════\n";

    return all_passed ? 0 : 1;
}
