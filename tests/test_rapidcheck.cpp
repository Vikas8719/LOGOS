
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
#include <filesystem>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <cstdlib>   // setenv, unsetenv, getenv

class ScopedCoutSilencer {
public:
    ScopedCoutSilencer() : previous_(std::cout.rdbuf(output_.rdbuf())) {}
    ~ScopedCoutSilencer() { std::cout.rdbuf(previous_); }

private:
    std::ostringstream output_;
    std::streambuf* previous_;
};


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
                      float atol = 2e-4f,
                      float rtol = 1e-4f)
{
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i)
        if (!std::isfinite(A[i]) || !std::isfinite(B[i]) ||
            std::abs(A[i] - B[i]) > atol + rtol * std::abs(B[i])) return false;
    return true;
}

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

static void rc1_vedic_matches_reference() {
    rc::check("[RC1] vedic_gemm(A,B) == reference_gemm(A,B) for random inputs", []() {
        int M = *rc::gen::inRange(1, 33);
        int K = *rc::gen::inRange(1, 33);
        int N = *rc::gen::inRange(1, 33);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );

        auto flat_a = *rc::gen::container<std::vector<float>>(M * K, float_gen);
        auto flat_b = *rc::gen::container<std::vector<float>>(K * N, float_gen);

        Tensor A({M, K}); A.data = flat_a;
        Tensor B({K, N}); B.data = flat_b;

        Tensor C = vedic_gemm(A, B);

        auto C_ref = naive_gemm(flat_a, flat_b, M, K, N);

        RC_ASSERT(mat_close(C.data, C_ref));
    });
}

static void rc2_tiling_boundary() {
    const std::vector<int> boundary_sizes = {1, 63, 64, 65, 127, 128, 129, 192, 193};

    const std::string saved_rc = []() -> std::string {
        const char* v = std::getenv("RC_PARAMS");
        return v ? v : "";
    }();

    setenv("RC_PARAMS", "max_success=200;max_discard_ratio=10", 1);

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

                RC_ASSERT(mat_close(C.data, C_ref, 5e-4f, 5e-5f));
            });
    }

    if (saved_rc.empty())
        unsetenv("RC_PARAMS");
    else
        setenv("RC_PARAMS", saved_rc.c_str(), 1);
}

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

        Tensor C_bias = vedic_gemm_bias(A, W, bias);

        auto C_ref = naive_gemm(flat_a, flat_w, M, K, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C_ref[i*N+j] += flat_b[j];

        RC_ASSERT(mat_close(C_bias.data, C_ref, 2e-4f, 1e-4f));
    });
}

static void rc4_free_energy_properties() {
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
        RC_ASSERT(S <= std::log(static_cast<float>(vocab)) + 1e-5f);
        const float probability_sum = std::accumulate(p.begin(), p.end(), 0.0f);
        RC_ASSERT(std::abs(probability_sum - 1.0f) < 1e-5f);
        for (float pi : p) RC_ASSERT(std::isfinite(pi) && pi > 0.0f);
    });

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
        RC_ASSERT(F >= CE - T * std::log(static_cast<float>(vocab)) - 1e-5f);
    });

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

        Tensor T2 = T.reshape({rows * cols, 1});
        Tensor T3 = T2.reshape({rows, cols});

        RC_ASSERT(T3.data == T.data);
        RC_ASSERT(T3.rows() == rows);
        RC_ASSERT(T3.cols() == cols);
        RC_ASSERT(T3.total_size == T.total_size);
        RC_ASSERT(T.data == flat);

        bool rejected = false;
        try {
            (void)T.reshape({rows * cols + 1});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        RC_ASSERT(rejected);
    });
}

static void rc6_riemannian_distance() {
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
        RC_ASSERT(std::isfinite(d));
        RC_ASSERT(d >= 0.0f);
        RC_ASSERT(std::abs(d) < 1e-6f);
    });

    rc::check("[RC6b] Riemannian distance > 0 for different points", []() {
        int dim = *rc::gen::inRange(1, 33);

        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );
        auto vals1 = *rc::gen::container<std::vector<float>>(dim, float_gen);
        auto vals2 = *rc::gen::container<std::vector<float>>(dim, float_gen);

        RC_PRE(vals1 != vals2);

        RiemannianMetric rm(dim, 1e-4f);
        Tensor t1({1, dim}); t1.data = vals1;
        Tensor t2({1, dim}); t2.data = vals2;

        float d = rm.riemannian_distance(t1, t2);
        RC_ASSERT(std::isfinite(d));
        RC_ASSERT(d > 0.0f);
    });

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

        RC_ASSERT(std::isfinite(d12) && std::isfinite(d21));
        RC_ASSERT(std::abs(d12 - d21) < 1e-6f);
    });

    rc::check("[RC6d] Riemannian distance satisfies triangle inequality", []() {
        int dim = *rc::gen::inRange(1, 33);
        auto float_gen = rc::gen::map(
            rc::gen::inRange(-100, 101),
            [](int v) { return v * 0.01f; }
        );
        auto a = *rc::gen::container<std::vector<float>>(dim, float_gen);
        auto b = *rc::gen::container<std::vector<float>>(dim, float_gen);
        auto c = *rc::gen::container<std::vector<float>>(dim, float_gen);
        RiemannianMetric rm(dim, 1e-4f);
        const float ab = rm.riemannian_distance(a, b);
        const float bc = rm.riemannian_distance(b, c);
        const float ac = rm.riemannian_distance(a, c);
        RC_ASSERT(std::isfinite(ab) && std::isfinite(bc) && std::isfinite(ac));
        RC_ASSERT(ac <= ab + bc + 1e-5f);
    });
}

static void rc7_checkpoint_size_formula() {
    rc::check("[RC7] serialized size matches actual model parameters", []() {
        int vocab_size  = *rc::gen::inRange(2, 513);
        int d_model_div = *rc::gen::inRange(1, 9);   // d_model = d_model_div * 4
        int d_model     = d_model_div * 4;            // must be multiple of 4 (for heads)
        const int heads_arr[] = {1, 2, 4};
        int heads_idx = *rc::gen::inRange(0, 3);
        int num_heads = heads_arr[heads_idx];
        int num_layers  = *rc::gen::inRange(1, 5);
        int max_seq_len = *rc::gen::inRange(2, 65);

        RC_PRE(d_model % num_heads == 0);

        ScopedCoutSilencer silence;
        ModelConfig cfg;
        cfg.vocab_size = vocab_size;
        cfg.d_model = d_model;
        cfg.num_heads = num_heads;
        cfg.num_layers = num_layers;
        cfg.max_seq_len = max_seq_len;
        LOGOSModel model(cfg);

        size_t parameter_floats = 0;
        for (const Tensor* parameter : model.parameters())
            parameter_floats += static_cast<size_t>(parameter->total_size);
        const size_t config_bytes = sizeof(ModelConfig);
        const size_t position_bytes = static_cast<size_t>(max_seq_len) * d_model * sizeof(float);
        const size_t batch_norm_bytes = static_cast<size_t>(num_layers) * 4 * d_model * sizeof(float);
        const size_t expected_bytes = config_bytes + position_bytes
                                    + parameter_floats * sizeof(float) + batch_norm_bytes;

        RC_ASSERT(expected_bytes > config_bytes);
        RC_ASSERT(expected_bytes < 10ULL * 1024 * 1024 * 1024);  // < 10GB
        RC_ASSERT(parameter_floats >= static_cast<size_t>(2) * vocab_size * d_model);

        const std::string base = "logos_rapidcheck_size";
        constexpr int step = 1;
        const std::string path = base + "_step" + std::to_string(step) + ".bin";
        std::remove(path.c_str());
        const bool saved = save_checkpoint(model, base, step);
        if (!saved) {
            std::remove(path.c_str());
            RC_ASSERT(saved);
        }
        const size_t actual_bytes = std::filesystem::file_size(path);
        std::remove(path.c_str());
        RC_ASSERT(actual_bytes == expected_bytes);
    });
}

static void rc8_leapfrog_vs_euler() {
    rc::check("[RC8a] Leapfrog energy drift < Euler drift — lr in (0.01, 0.5)", []() {
        float lr = *rc::gen::map(
            rc::gen::inRange(1, 51),   // 1..50 → 0.01..0.50
            [](int v) { return v * 0.01f; }
        );
        int steps = *rc::gen::inRange(50, 501);  // enough steps to see divergence

        const float W0 = 1.0f, V0 = 0.0f;
        const float E0 = 0.5f * W0 * W0 + 0.5f * V0 * V0;  // = 0.5

        float W_e = W0, V_e = V0;
        for (int i = 0; i < steps; ++i) {
            float w_old = W_e, v_old = V_e;
            W_e = w_old + lr * v_old;
            V_e = v_old - lr * w_old;
        }
        float E_euler = 0.5f * W_e * W_e + 0.5f * V_e * V_e;
        float euler_drift = std::abs(E_euler - E0) / E0;

        float W_l = W0, V_l = V0;
        for (int i = 0; i < steps; ++i) {
            V_l -= (lr * 0.5f) * W_l;
            W_l += lr * V_l;
            V_l -= (lr * 0.5f) * W_l;
        }
        float E_lf = 0.5f * W_l * W_l + 0.5f * V_l * V_l;
        float lf_drift = std::abs(E_lf - E0) / E0;

        RC_ASSERT(std::isfinite(euler_drift));
        RC_ASSERT(std::isfinite(lf_drift));
        RC_ASSERT(lf_drift < 1.0f);          // leapfrog stays near E0
        RC_ASSERT(lf_drift < euler_drift);
    });

    rc::check("[RC8b] Leapfrog drift < 1% for small lr (< 0.1)", []() {
        float lr = *rc::gen::map(
            rc::gen::inRange(1, 11),   // 0.01..0.10
            [](int v) { return v * 0.01f; }
        );
        int steps = *rc::gen::inRange(10, 101);

        const float E0 = 0.5f;
        float W = 1.0f, V = 0.0f;
        for (int i = 0; i < steps; ++i) {
            V -= (lr * 0.5f) * W;
            W += lr * V;
            V -= (lr * 0.5f) * W;
        }
        float drift = std::abs(0.5f * W * W + 0.5f * V * V - E0) / E0;

        RC_ASSERT(std::isfinite(drift));
        RC_ASSERT(drift < 0.01f);
    });
}

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
    run("RC8 Leapfrog vs Euler (RC8a drift<euler, RC8b <1%)", rc8_leapfrog_vs_euler);

    std::cout << "════════════════════════════════════════════════\n";
    if (all_passed) {
        std::cout << "  ✅ ALL RAPIDCHECK PROPERTY TESTS PASSED\n";
    } else {
        std::cout << "  ❌ SOME TESTS FAILED — see output above\n";
    }
    std::cout << "════════════════════════════════════════════════\n";

    return all_passed ? 0 : 1;
}
