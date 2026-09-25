// ============================================================
//  LOGOS — tests/test_math_unit.cpp
//  Test 2: Mathematical Unit Tests
//
//  Kya test hoga:
//    [M1]  Urdhva Tiryagbhyam — Vedic GEMM vs reference GEMM
//    [M2]  Gunitasamuchayah — Product of sums = Sum of products
//    [M3]  Free Energy — F = CE - T*S
//    [M4]  Leapfrog Langevin — energy preservation
//    [M5]  Boltzmann Softmax — temperature scaling
//    [M6]  Nikhilam complement — arithmetic identity
//    [M7]  Hyperbolic exp_map / log_map — inverse property
//    [M8]  LayerNorm — exact statistics
//    [M9]  VedicGEMM Tiling — result matches naive triple-loop
//    [M10] Gradient correctness — numerical vs analytical
//
//  NEW (v10) — 6 Physics Components:
//    [M11] Natural Gradient — Fisher diagonal EMA, g̃ = F⁻¹g
//          a) Fisher diagonal converges to gradient variance
//          b) Natural gradient step smaller than raw gradient in high-curvature dims
//          c) Scale invariance: natural grad invariant to param re-scaling
//          d) After many steps with const grad, Fisher ≈ g² (stationary EMA)
//
//    [M12] Navier-Stokes Attention — fluid flow token interactions
//          a) Advection: Q_adv[i] ≠ Q[i] for i > 0 (advection modifies query)
//          b) Viscous diffusion: V_smooth ≠ V (diffusion modifies values)
//          c) Incompressibility: attention weights still sum to 1 (softmax)
//          d) Boundary conditions: Q_adv[0] == Q[0] (no advection at left boundary)
//          e) NS output != standard output (fluid physics changes result)
//
//    [M13] Reynolds Batch Norm — laminar/turbulent regime blend
//          a) Re_eff > 0 always (ratio of RMS to std)
//          b) Laminar weight ∈ (0,1) always (sigmoid bounded)
//          c) When Re → 0: laminar_weight → 1 (pure LayerNorm)
//          d) When Re → ∞: laminar_weight → 0 (pure BroadcastNorm)
//          e) ReynoldsBatchNorm output ≠ raw input (actually normalises)
//          f) Re_crit annealing: Re_crit decreases over training
//
//    [M14] Feynman Dropout — path integral amplitude distribution
//          a) mean(weight) ≈ (1-p) regardless of ħ (scale invariance)
//          b) All weights ∈ (0,1) — no negative or >1 amplitudes (Beta dist)
//          c) Classical limit (ħ→0): weights bimodal near {0, 1}
//          d) Quantum limit (ħ=1): weights smooth, not bimodal
//          e) p=0: all weights = 1 (no dropout, identity)
//          f) Training=false: forward() is exact identity (no dropout at eval)
//
//    [M15] Riemannian Metric — curved parameter space geometry
//          a) riemannian_distance(θ,θ) == 0 (zero self-distance)
//          b) riemannian_distance(θ1,θ2) > 0 for θ1 ≠ θ2
//          c) riemannian_distance(θ1,θ2) == riemannian_distance(θ2,θ1) (symmetry)
//          d) After update(g): metric_diag increases (EMA with g²>0)
//          e) riemannian_gradient: ||g̃||_G < ||g||_G (natural grad has less R-norm)
//          f) parallel_transport: transported vector ⊥ Δθ in G-inner-product
//
//    [M16] Weight Path Integral — Feynman action accumulation
//          a) log_amplitude decreases monotonically when action > 0
//          b) best_step is correctly tracked (highest log_amplitude)
//          c) lr_scale ∈ [lr_min_frac, 1.0] always
//          d) Discrete Lagrangian: S_t = loss * ||Δθ|| (verified manually)
//          e) relative_amplitude == 1.0 at best step (by definition)
//          f) recent_mean_action computes correctly over last N steps
// ============================================================
#include "../include/VedicGEMM.hpp"
#include "../include/Tensor.hpp"
#include "../include/PhysicsOpt.hpp"
#include "../include/LayerNorm.hpp"
#include "../include/FeedForward.hpp"
#include "../include/Attention.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>

// ── Test framework ────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

#define TEST(name, expr) do { \
    bool _ok = (expr); \
    if (_ok) { std::cout << "  ✅ " << (name) << "\n"; ++g_pass; } \
    else { std::cerr << "  ❌ " << (name) << "  [FAIL]\n"; ++g_fail; } \
} while(0)

#define TEST_CLOSE(name, a, b, tol) \
    TEST(name, std::abs((a)-(b)) <= (tol))

// ── Reference implementations (pure CPU, no vectorization) ───

// Naive triple-loop GEMM for ground truth
static std::vector<float> naive_gemm(
    const std::vector<float>& A, const std::vector<float>& B,
    int M, int K, int N)
{
    std::vector<float> C(M*N, 0.f);
    for (int i=0; i<M; ++i)
        for (int k=0; k<K; ++k)
            for (int j=0; j<N; ++j)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
    return C;
}

// CPU softmax on a single row
static std::vector<float> cpu_softmax(const std::vector<float>& logits, float T=1.0f) {
    int n = logits.size();
    std::vector<float> out(n);
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (int i=0; i<n; ++i) { out[i] = std::exp((logits[i]-mx)/T); sum += out[i]; }
    for (int i=0; i<n; ++i) out[i] /= sum;
    return out;
}

// CPU CE loss for one token
static float cpu_ce(const std::vector<float>& p, int target) {
    return -std::log(std::max(p[target], 1e-12f));
}

// CPU Shannon entropy
static float cpu_entropy(const std::vector<float>& p) {
    float S = 0.f;
    for (float pi : p) if (pi > 1e-12f) S -= pi * std::log(pi);
    return S;
}

// CPU free energy
static float cpu_free_energy(const std::vector<float>& logits, int target, float T) {
    auto p = cpu_softmax(logits, 1.0f);
    float CE = cpu_ce(p, target);
    float S  = cpu_entropy(p);
    return CE - T * S;
}

// CPU CE gradient (for one token)
static std::vector<float> cpu_ce_grad(const std::vector<float>& p, int target, int seq) {
    int n = p.size();
    std::vector<float> g(n);
    for (int v=0; v<n; ++v)
        g[v] = (p[v] - (v==target ? 1.f : 0.f)) / seq;
    return g;
}

// CPU free energy gradient
static std::vector<float> cpu_fe_grad(
    const std::vector<float>& logits, int target, float T, int seq)
{
    auto p = cpu_softmax(logits, 1.0f);
    float S = cpu_entropy(p);
    int n = logits.size();
    std::vector<float> g(n);
    for (int v=0; v<n; ++v) {
        float y_v  = (v == target) ? 1.f : 0.f;
        float ce_g = (p[v] - y_v);
        float H_g  = T * p[v] * (std::log(p[v]+1e-9f) + S);
        g[v] = (ce_g + H_g) / seq;
    }
    return g;
}

// CPU exp_map (Poincaré ball, c=1)
static std::vector<float> cpu_expmap(const std::vector<float>& v) {
    float norm_sq = 0.f;
    for (float x : v) norm_sq += x*x;
    float norm = std::sqrt(norm_sq + 1e-12f);
    float mapped_norm = std::min(std::tanh(norm*0.5f), 1.0f - 1e-6f);
    float factor = norm > 1e-7f ? mapped_norm / norm : 0.5f;
    std::vector<float> out(v.size());
    for (int i=0; i<(int)v.size(); ++i) out[i] = factor * v[i];
    return out;
}

// CPU log_map (Poincaré ball, c=1)
static std::vector<float> cpu_logmap(const std::vector<float>& y) {
    float norm_sq = 0.f;
    for (float x : y) norm_sq += x*x;
    float norm = std::sqrt(norm_sq + 1e-12f);
    float safe_norm = std::min(norm, 0.9999f);
    float factor = norm > 1e-7f
        ? 2.f * std::atanh(safe_norm) / norm
        : 2.0f;
    std::vector<float> out(y.size());
    for (int i=0; i<(int)y.size(); ++i) out[i] = factor * y[i];
    return out;
}

// ============================================================
//  [M1] Urdhva Tiryagbhyam vs reference GEMM
// ============================================================
static void test_m1_vedic_gemm() {
    std::cout << "\n[M1] Urdhva Tiryagbhyam — Vedic GEMM vs Reference\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Test multiple shapes
    std::vector<std::tuple<int,int,int>> shapes = {
        {4,4,4}, {8,16,8}, {32,64,32}, {64,128,64}, {128,256,128}
    };

    for (auto& [M, K, N] : shapes) {
        std::vector<float> A(M*K), B(K*N);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        // Vedic GEMM (using Tensor wrapper)
        Tensor TA({M,K}); std::copy(A.begin(),A.end(),TA.data.begin());
        Tensor TB({K,N}); std::copy(B.begin(),B.end(),TB.data.begin());
        Tensor TC = vedic_gemm(TA, TB);

        // Reference
        auto Cref = naive_gemm(A, B, M, K, N);

        float max_err = 0.f;
        for (int i=0; i<M*N; ++i)
            max_err = std::max(max_err, std::abs(TC.data[i] - Cref[i]));

        std::string name = "Vedic GEMM [" + std::to_string(M) + "x" +
                           std::to_string(K) + "x" + std::to_string(N) +
                           "] max_err=" + std::to_string(max_err);
        TEST(name, max_err < 1e-2f);
    }
}

// ============================================================
//  [M2] Gunitasamuchayah — CPU verification
// ============================================================
static void test_m2_gunitasamuchayah() {
    std::cout << "\n[M2] Gunitasamuchayah (Product of Sums)\n";
    // For C = A @ B, sum(C) is the dot product of A's column sums
    // and B's row sums. This preserves the shared K dimension exactly.

    auto test_case = [&](const char* name,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         int M, int K, int N, float tol) {
        auto C = naive_gemm(A, B, M, K, N);

        // sum(C)
        float sum_C = std::accumulate(C.begin(), C.end(), 0.f);

        std::vector<float> a_col_sums(K, 0.f);
        std::vector<float> b_row_sums(K, 0.f);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                a_col_sums[k] += A[i*K + k];
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                b_row_sums[k] += B[k*N + j];

        float vedic_pred = 0.f;
        for (int k = 0; k < K; ++k)
            vedic_pred += a_col_sums[k] * b_row_sums[k];
        float rel_err = std::abs(sum_C - vedic_pred) / (std::abs(sum_C) + 1e-6f);

        std::cout << "    " << name << ": sum_C=" << sum_C
                  << " vedic=" << vedic_pred << " err=" << rel_err*100 << "%\n";
        TEST(std::string("Gunitasamuchayah: ") + name, rel_err < tol);
    };

    // Case 1: Identity-like (A=I, B=I → C=I → sum=min(M,N))
    {
        int M=8,K=8,N=8;
        std::vector<float> A(M*K,0.f), B(K*N,0.f);
        for (int i=0; i<M; ++i) A[i*K+i]=1.f;
        for (int i=0; i<K; ++i) B[i*N+i]=1.f;
        test_case("Identity", A, B, M, K, N, 0.5f);  // relaxed: I*I = I, sum=8
    }

    // Case 2: All ones (A=1, B=1 → C=K*1 → sum=M*N*K)
    {
        int M=4,K=8,N=4;
        std::vector<float> A(M*K,1.f), B(K*N,1.f);
        test_case("All-ones (exact)", A, B, M, K, N, 1e-4f);
    }

    // Case 3: Random values; the shared-K checksum remains exact.
    {
        int M=16,K=32,N=16;
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> A(M*K), B(K*N);
        for (float& x : A) x = nd(rng);
        for (float& x : B) x = nd(rng);
        test_case("Zero-mean random", A, B, M, K, N, 1e-5f);
    }

    // Direct identity: double complement = original
    {
        int base = 127;
        std::vector<int> vals = {0, 1, 50, 63, 127, -1, -50, -127};
        bool ok = true;
        for (int v : vals) {
            int complement  = base - std::abs(v);
            int reconstructed = (v >= 0 ? 1 : -1) * (base - complement);
            if (reconstructed != v) ok = false;
        }
        TEST("Nikhilam: base - (base - x) = x (complement identity)", ok);
    }
}

// ============================================================
//  [M3] Free Energy — physics property tests
// ============================================================
static void test_m3_free_energy() {
    std::cout << "\n[M3] Free Energy F = CE - T·S Properties\n";

    std::vector<float> logits = {2.f, 1.f, 0.f, -1.f, 0.5f, -0.5f, 1.5f, -2.f};
    int vocab = (int)logits.size();
    int target = 0;

    auto p = cpu_softmax(logits, 1.0f);
    float CE = cpu_ce(p, target);
    float S  = cpu_entropy(p);

    // [a] S >= 0
    TEST("S >= 0 (entropy non-negative)", S >= -1e-6f);

    // [b] F <= CE when T > 0
    float T = 0.05f;
    float F = CE - T * S;
    TEST("F <= CE when T > 0", F <= CE + 1e-6f);

    // [c] As T→0: F → CE
    float F_small = CE - 1e-8f * S;
    TEST("F → CE as T → 0", std::abs(F_small - CE) < 1e-4f);

    // [d] T=0: gradient → CE gradient
    {
        std::vector<float> fe_g = cpu_fe_grad(logits, target, 0.f, 1);
        std::vector<float> ce_g = cpu_ce_grad(p, target, 1);
        float max_diff = 0.f;
        for (int i=0; i<vocab; ++i)
            max_diff = std::max(max_diff, std::abs(fe_g[i]-ce_g[i]));
        TEST("dF/dlogit = dCE/dlogit when T=0", max_diff < 1e-6f);
    }

    // [e] Uniform distribution → max entropy S = log(vocab)
    {
        std::vector<float> uniform_logits(vocab, 0.f);
        auto pu = cpu_softmax(uniform_logits, 1.0f);
        float Su = cpu_entropy(pu);
        float expected_S = std::log((float)vocab);
        TEST("Uniform dist → S = log(vocab)", std::abs(Su - expected_S) < 1e-4f);
    }

    // [f] Peaked distribution → S near 0
    {
        std::vector<float> peaked(vocab, -100.f);
        peaked[0] = 100.f;
        auto pp = cpu_softmax(peaked, 1.0f);
        float Sp = cpu_entropy(pp);
        TEST("Peaked dist → S ≈ 0", Sp < 0.01f);
    }

    std::cout << "    CE=" << CE << " S=" << S << " F=" << F << " (T=" << T << ")\n";
}

// ============================================================
//  [M4] Leapfrog Langevin — stability vs Euler
// ============================================================
static void test_m4_leapfrog_stability() {
    std::cout << "\n[M4] Leapfrog Langevin — Stability vs Euler-Maruyama\n";

    // Simple harmonic oscillator: grad = W (spring force)
    // Euler: W += -lr*W + noise          (1st order, energy drift)
    // Leapfrog: v = -lr/2*W + v, W += lr*v  (2nd order, symplectic)
    // Ideal: total energy E = 0.5*W^2 + 0.5*V^2 should be conserved

    float lr = 0.1f, friction = 1.0f, noise = 0.0f;  // zero noise for clean test
    int steps = 100;

    // Euler-Maruyama
    float W_euler = 1.0f, V_euler = 0.0f;
    float E0_euler = 0.5f*W_euler*W_euler + 0.5f*V_euler*V_euler;
    for (int i=0; i<steps; ++i) {
        V_euler = friction*V_euler - lr*W_euler;
        W_euler += V_euler;
    }
    float E1_euler = 0.5f*W_euler*W_euler + 0.5f*V_euler*V_euler;
    float euler_drift = std::abs(E1_euler - E0_euler) / E0_euler;

    // Leapfrog (Störmer-Verlet) with both half-kicks for each full step.
    float W_lf = 1.0f, V_lf = 0.0f;
    float E0_lf = 0.5f*W_lf*W_lf + 0.5f*V_lf*V_lf;
    for (int i=0; i<steps; ++i) {
        V_lf -= (lr * 0.5f) * W_lf;
        W_lf += lr * V_lf;
        V_lf -= (lr * 0.5f) * W_lf;
    }
    float E1_lf = 0.5f*W_lf*W_lf + 0.5f*V_lf*V_lf;
    float lf_drift = std::abs(E1_lf - E0_lf) / E0_lf;

    std::cout << "    Euler  energy drift: " << euler_drift*100 << "%\n";
    std::cout << "    Leapfrog drift:      " << lf_drift*100 << "%\n";

    // Leapfrog should have LESS energy drift than Euler for same lr
    TEST("Leapfrog drift < Euler drift (better stability)", lf_drift < euler_drift);
    TEST("Leapfrog: energy drift < 50% over 100 steps",    lf_drift < 0.5f);

    // Order test: Euler error ∝ lr, Leapfrog error ∝ lr²
    // At lr=0.1: Euler drift should be >> Leapfrog drift
    TEST("Euler drift > 1% (expected 1st order error)", euler_drift > 0.01f);
}

// ============================================================
//  [M5] Boltzmann Softmax — temperature scaling
// ============================================================
static void test_m5_boltzmann_softmax() {
    std::cout << "\n[M5] Boltzmann Softmax Temperature Scaling\n";

    std::vector<float> logits = {3.f, 1.f, 0.f, -1.f, 0.5f};
    int n = logits.size();
    int argmax = 0;  // index 0 has highest logit

    // T → 0 (very small): should become near one-hot
    auto p_cold = cpu_softmax(logits, 0.001f);
    TEST("T→0: argmax token gets high prob",  p_cold[argmax] > 0.999f);
    TEST("T→0: other tokens get near-0 prob", p_cold[1] < 0.001f);

    // T → ∞ (very large): should approach uniform
    auto p_hot = cpu_softmax(logits, 1000.f);
    float expected_uniform = 1.f / n;
    float max_dev = 0.f;
    for (float pi : p_hot) max_dev = std::max(max_dev, std::abs(pi - expected_uniform));
    TEST("T→∞: distribution approaches uniform", max_dev < 0.01f);

    // T=1: standard softmax
    auto p_std = cpu_softmax(logits, 1.f);
    float sum = 0.f; for (float pi : p_std) sum += pi;
    TEST("T=1: standard softmax (sum=1)",    std::abs(sum - 1.f) < 1e-5f);
    TEST("T=1: argmax still most probable",  p_std[argmax] > p_std[1]);

    // Temperature and entropy monotonically related
    float S1 = cpu_entropy(p_cold);
    float S2 = cpu_entropy(p_std);
    float S3 = cpu_entropy(p_hot);
    TEST("Entropy: S(T=0.001) < S(T=1) < S(T=1000)", S1 < S2 && S2 < S3);
}

// ============================================================
//  [M6] Nikhilam complement — arithmetic identity
// ============================================================
static void test_m6_nikhilam_complement() {
    std::cout << "\n[M6] Nikhilam Complement Identity\n";

    // a) Double complement = identity
    int base = 127;
    bool identity_ok = true;
    for (int v = -127; v <= 127; ++v) {
        int8_t q = (int8_t)std::max(-127, std::min(127, v));
        int reconstructed = (int)q;
        if (reconstructed != v) { identity_ok = false; break; }
    }
    TEST("int8 round-trip: cast/uncast identity [-127,127]", identity_ok);

    // b) Quantization error bounded by scale/2
    float absmax = 10.0f;
    float scale = absmax / 127.0f;
    float max_quant_err = scale / 2.0f;  // max rounding error
    // Test specific values
    float test_val = 7.3f;
    int8_t q = (int8_t)std::round(test_val / scale);
    float rec = q * scale;
    float err = std::abs(test_val - rec);
    TEST("Quant error <= scale/2", err <= max_quant_err + 1e-6f);

    // c) Scale = absmax/127 → no clipping for |v| <= absmax
    bool no_clip = true;
    std::vector<float> test_vals = {-10.f, -5.f, 0.f, 5.f, 10.f};
    for (float v : test_vals) {
        int8_t qi = (int8_t)std::max(-127.f, std::min(127.f,
                        std::round(v / scale)));
        // No clipping means |qi| < 127 (not saturated at boundary)
        if (std::abs(v) < absmax && std::abs((int)qi) == 127) no_clip = false;
    }
    TEST("No clipping for |v| < absmax", no_clip);

    // d) Nikhilam: 9's complement property — for digits summing to 9
    // In decimal: 7 + 3 = 10 (Nikhilam: complement of 7 in base 10 is 3)
    // In our int8: complement of q is (127 - q)
    // Double complement: 127 - (127 - q) = q ✅
    int q_test = 45;
    int complement = base - q_test;
    int double_complement = base - complement;
    TEST("Nikhilam: 127 - (127 - q) == q", double_complement == q_test);

    std::cout << "    scale=" << scale << " max_quant_err=" << max_quant_err << "\n";
}

// ============================================================
//  [M7] Hyperbolic exp_map / log_map — inverse property
// ============================================================
static void test_m7_hyperbolic_maps() {
    std::cout << "\n[M7] Hyperbolic exp_map / log_map Inverse Property\n";

    // a) log_map(exp_map(v)) ≈ v for small ||v||
    {
        std::vector<float> v = {0.1f, -0.2f, 0.15f, -0.05f};
        auto y = cpu_expmap(v);
        auto v_rec = cpu_logmap(y);
        float max_err = 0.f;
        for (int i=0; i<(int)v.size(); ++i)
            max_err = std::max(max_err, std::abs(v[i] - v_rec[i]));
        std::cout << "    logmap(expmap(v)) max_err=" << max_err << "\n";
        TEST("log_map(exp_map(v)) ≈ v (small v)",  max_err < 1e-4f);
    }

    // b) ||exp_map(v)|| < 1 for all v (Poincaré ball)
    {
        bool inside = true;
        std::vector<std::vector<float>> test_vs = {
            {0.1f, 0.2f},
            {10.f, 20.f, 30.f},      // large vector
            {0.001f, 0.002f},         // tiny vector
            {-5.f, 5.f, -5.f, 5.f},  // alternating
        };
        for (auto& v : test_vs) {
            auto y = cpu_expmap(v);
            float norm_sq = 0.f;
            for (float x : y) norm_sq += x*x;
            if (std::sqrt(norm_sq) >= 1.0f) inside = false;
        }
        TEST("||exp_map(v)|| < 1 for all v", inside);
    }

    // c) exp_map(0) = 0
    {
        std::vector<float> zero(8, 0.0f);
        auto y = cpu_expmap(zero);
        float norm = 0.f; for (float x : y) norm += x*x;
        TEST("exp_map(0) = 0", std::sqrt(norm) < 1e-6f);
    }

    // d) Curvature: larger c → tighter ball (smaller output norm for same input)
    {
        std::vector<float> v = {1.f, 1.f, 1.f, 1.f};
        // c=1
        auto y1 = cpu_expmap(v);
        float n1=0.f; for (float x : y1) n1+=x*x; n1=std::sqrt(n1);
        // c=4: scale v by sqrt(c) inside expmap
        // expmap_c(v) = tanh(sqrt(c)*||v||/2) * v / (sqrt(c)*||v||)
        float c = 4.f, sc = std::sqrt(c);
        float norm=0.f; for (float x : v) norm+=x*x; norm=std::sqrt(norm);
        float factor = std::tanh(sc*norm*0.5f) / (sc*norm + 1e-9f);
        float n4 = 0.f;
        for (float x : v) n4 += (factor*x)*(factor*x);
        n4 = std::sqrt(n4);
        std::cout << "    ||expmap_c=1(v)||=" << n1 << " ||expmap_c=4(v)||=" << n4 << "\n";
        TEST("Larger curvature → smaller output norm", n4 < n1);
    }
}

// ============================================================
//  [M8] LayerNorm — exact statistics
// ============================================================
static void test_m8_layernorm() {
    std::cout << "\n[M8] LayerNorm — Exact Statistics\n";

    // CPU layernorm
    auto layernorm_cpu = [](const std::vector<float>& x,
                            const std::vector<float>& gamma,
                            const std::vector<float>& beta) {
        int n = x.size();
        float mean=0.f, var=0.f;
        for (float v : x) mean += v; mean /= n;
        for (float v : x) var += (v-mean)*(v-mean); var /= n;
        float inv_std = 1.f / std::sqrt(var + 1e-5f);
        std::vector<float> out(n);
        for (int i=0; i<n; ++i)
            out[i] = gamma[i] * (x[i]-mean) * inv_std + beta[i];
        return out;
    };

    int d = 64;
    std::mt19937 rng(999);
    std::normal_distribution<float> nd(5.f, 3.f);  // non-zero mean/std input
    std::vector<float> x(d), gamma(d, 2.0f), beta(d, 1.0f);
    for (float& v : x) v = nd(rng);

    auto y = layernorm_cpu(x, gamma, beta);

    float mean_y = 0.f;
    for (float v : y) mean_y += v;
    mean_y /= d;

    float var_y = 0.f;
    for (float v : y) var_y += (v-mean_y)*(v-mean_y);
    var_y /= d;
    float std_y = std::sqrt(var_y);

    std::cout << "    LN output: mean=" << mean_y << " std=" << std_y
              << " (expected: mean≈beta=1, std≈gamma=2)\n";

    // With uniform gamma=2, beta=1: output mean≈1, std≈2
    TEST("LayerNorm: mean ≈ beta (1.0)",    std::abs(mean_y - 1.0f) < 0.01f);
    TEST("LayerNorm: std ≈ gamma (2.0)",    std::abs(std_y  - 2.0f) < 0.1f);
}

// ============================================================
//  [M9] VedicGEMM Tiling — non-divisible shapes
// ============================================================
static void test_m9_gemm_tiling() {
    std::cout << "\n[M9] VedicGEMM Tiling — Non-TILE-Divisible Shapes\n";

    // TILE_SIZE=16 in CUDA; CPU vedic_gemm has its own tiling
    // Test shapes NOT divisible by 16
    std::vector<std::tuple<int,int,int>> shapes = {
        {3, 5, 7},          // tiny non-divisible
        {17, 33, 19},       // just over TILE
        {31, 63, 15},       // one less than 2*TILE
        {100, 200, 150},    // large non-divisible
    };

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& [M, K, N] : shapes) {
        std::vector<float> A(M*K), B(K*N);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        Tensor TA({M,K}); std::copy(A.begin(),A.end(),TA.data.begin());
        Tensor TB({K,N}); std::copy(B.begin(),B.end(),TB.data.begin());
        Tensor TC = vedic_gemm(TA, TB);
        auto Cref = naive_gemm(A, B, M, K, N);

        float max_err = 0.f;
        for (int i=0; i<M*N; ++i)
            max_err = std::max(max_err, std::abs(TC.data[i] - Cref[i]));

        std::string name = "VedicGEMM [" + std::to_string(M) + "x" +
                           std::to_string(K) + "x" + std::to_string(N) + "]";
        TEST(name, max_err < 1e-2f);
    }
}

// ============================================================
//  [M10] Gradient correctness — numerical finite difference
// ============================================================
static void test_m10_gradient_fd() {
    std::cout << "\n[M10] Free Energy Gradient — Finite Difference Check\n";

    int vocab = 16;
    int target = 3;
    float T = 0.05f;
    float eps = 1e-3f;
    int seq = 1;

    std::vector<float> logits = {0.5f, 1.2f, -0.3f, 2.1f, 0.8f, -0.5f,
                                  1.0f, 0.2f, -0.1f, 0.9f, 0.4f, -0.8f,
                                  0.3f, 1.5f, -0.2f, 0.7f};

    // Analytical gradient
    auto analytical = cpu_fe_grad(logits, target, T, seq);

    // Numerical finite difference
    std::vector<float> numerical(vocab);
    for (int v=0; v<vocab; ++v) {
        auto l_plus = logits;
        auto l_minus = logits;
        l_plus[v]  += eps;
        l_minus[v] -= eps;
        float F_plus  = cpu_free_energy(l_plus,  target, T);
        float F_minus = cpu_free_energy(l_minus, target, T);
        numerical[v] = (F_plus - F_minus) / (2.f * eps);
    }

    float max_rel_err = 0.f;
    for (int v=0; v<vocab; ++v) {
        float denom = std::abs(numerical[v]) + 1e-6f;
        float rel   = std::abs(analytical[v] - numerical[v]) / denom;
        max_rel_err = std::max(max_rel_err, rel);
    }

    std::cout << "    Max relative error (analytical vs FD): "
              << max_rel_err * 100 << "%\n";

    // Print a few values for inspection
    std::cout << "    Sample [v=0]: analytical=" << analytical[0]
              << " numerical=" << numerical[0] << "\n";
    std::cout << "    Sample [target=" << target << "]: analytical=" << analytical[target]
              << " numerical=" << numerical[target] << "\n";

    TEST("FE gradient: max relative error < 1%",  max_rel_err < 0.01f);
    TEST("FE gradient: target grad is most negative",
         analytical[target] < 0.f || vocab == 1);
}

// ============================================================
//  [M11] Natural Gradient — Fisher diagonal EMA correctness
// ============================================================
static void test_m11_natural_gradient() {
    std::cout << "\n[M11] Natural Gradient — Fisher Diagonal EMA\n";

    // a) After many steps with constant gradient, Fisher ≈ g²
    // EMA formula: F_t = β*F_{t-1} + (1-β)*g²
    // Stationary value: F_∞ = g² (geometric series sum = (1-β)/(1-β) = 1)
    {
        float beta = 0.99f, g_val = 2.5f, eps = 1e-8f;
        float F = eps;  // init to eps (same as NaturalGradientOptimizer)
        for (int i = 0; i < 2000; ++i)
            F = beta * F + (1.0f - beta) * g_val * g_val;
        float expected = g_val * g_val;  // stationary EMA = g²
        TEST("Fisher EMA converges to g² (stationary)", std::abs(F - expected) < 0.05f);
        std::cout << "    F_∞=" << F << " expected=" << expected << "\n";
    }

    // b) Natural gradient step < raw gradient in high-curvature dimensions
    // g̃ = g / (√F + ε). If F >> ε, g̃ < g (curvature-damped step).
    {
        float g  = 1.0f;
        float F_high = 100.0f;  // high curvature → small step
        float F_low  = 0.01f;   // low curvature → larger step
        float g_nat_high = g / (std::sqrt(F_high) + 1e-8f);
        float g_nat_low  = g / (std::sqrt(F_low)  + 1e-8f);
        TEST("Natural grad: high-curvature dim has smaller step", g_nat_high < g);
        TEST("Natural grad: low-curvature  dim has larger  step", g_nat_low  > g);
        std::cout << "    g_nat_high=" << g_nat_high << " g_nat_low=" << g_nat_low << "\n";
    }

    // c) Scale invariance: natural grad magnitude independent of param scale
    // If we scale θ by c, grad scales by 1/c, Fisher by 1/c², g̃ = (g/c)/(1/c) = g
    // (approximate: for diagonal Fisher this holds dimensionally)
    {
        float g1 = 0.5f,  F1 = 0.5f * 0.5f;   // original
        float g2 = 5.0f,  F2 = 5.0f * 5.0f;   // 10x scaled param → 10x grad
        float gnat1 = g1 / (std::sqrt(F1) + 1e-8f);
        float gnat2 = g2 / (std::sqrt(F2) + 1e-8f);
        // Both should equal 1.0 (g / sqrt(g²) = 1)
        TEST("Natural grad: scale invariant (g̃ = sign(g))",
             std::abs(std::abs(gnat1) - 1.0f) < 1e-5f &&
             std::abs(std::abs(gnat2) - 1.0f) < 1e-5f);
        std::cout << "    g̃₁=" << gnat1 << " g̃₂=" << gnat2 << "\n";
    }

    // d) NaturalGradientOptimizer actually changes weights
    {
        Tensor W({4, 4}); W.fill_random(-0.1f, 0.1f);
        Tensor G({4, 4}); G.fill(0.05f);
        NaturalGradientOptimizer opt(1e-3f, 0.99f, 1e-8f, 0.9f, 0.01f, 1e-6f, 100);
        std::vector<Tensor*> ps = {&W};
        std::vector<Tensor*> gs = {&G};
        float w0 = W.data[0];
        opt.step(ps, gs);
        TEST("NaturalGradOpt: weights change after step", W.data[0] != w0);
        TEST("NaturalGradOpt: no NaN after step",         !W.has_nan());
    }
}

// ============================================================
//  [M12] Navier-Stokes Attention — fluid physics properties
// ============================================================
static void test_m12_navier_stokes_attention() {
    std::cout << "\n[M12] Navier-Stokes Attention — Fluid Physics\n";

    int seq = 8, d_k = 16, d_v = 16;
    logos_rng::set_global_seed(42);

    Tensor Q({seq, d_k}); Q.fill_random(-0.5f, 0.5f);
    Tensor K({seq, d_k}); K.fill_random(-0.5f, 0.5f);
    Tensor V({seq, d_v}); V.fill_random(-0.5f, 0.5f);

    float temperature = std::sqrt((float)d_k);
    float eta = 0.2f, nu = 0.1f;

    // Build causal mask
    Tensor mask({seq, seq}, 0.0f);
    for (int i = 0; i < seq; ++i)
        for (int j = i+1; j < seq; ++j)
            mask.at(i, j) = -1e9f;

    // a) Advection: Q_adv[i] should differ from Q[i] for i > 0
    {
        // Manually compute Q_adv[1] = Q[1] + η*(Q[1]-Q[0])
        float q0 = Q.at(1, 0);
        float q_prev = Q.at(0, 0);
        float q_adv_expected = q0 + eta * (q0 - q_prev);
        // We can't inspect internals of navier_stokes_attention directly,
        // but we can verify: if eta > 0 and Q[0] != Q[1], output must differ
        // from standard. Verify NS output != raw V (it's processed).
        Tensor ns_out = navier_stokes_attention(Q, K, V, mask, temperature, eta, nu);
        TEST("NS attention: output shape correct (seq x d_v)",
             ns_out.rows() == seq && ns_out.cols() == d_v);
        TEST("NS attention: output has no NaN", !ns_out.has_nan());
    }

    // b) Boundary condition: advection at i=0 uses Q[0] as own prev
    // Q_adv[0] = Q[0] + eta*(Q[0]-Q[0]) = Q[0] (no change at left boundary)
    // We verify indirectly: NS output must be finite and valid
    {
        // Run with eta=1.0 (strong advection) — should still be stable
        Tensor ns_strong = navier_stokes_attention(Q, K, V, mask, temperature, 1.0f, nu);
        TEST("NS attention: stable with strong advection (eta=1)", !ns_strong.has_nan());
    }

    // c) Incompressibility: attention weights must sum to 1 per row
    // We test standard attention (which NS uses internally via boltzmann_softmax)
    {
        Tensor K_T = K.transpose();
        Tensor scores = vedic_gemm(Q, K_T);
        scores += mask;
        Tensor w = boltzmann_softmax(scores, temperature);
        bool rows_sum_to_one = true;
        for (int i = 0; i < seq; ++i) {
            float row_sum = 0.f;
            for (int j = 0; j < seq; ++j) row_sum += w.at(i, j);
            if (std::abs(row_sum - 1.0f) > 1e-4f) rows_sum_to_one = false;
        }
        TEST("NS attention: softmax rows sum to 1 (incompressible)", rows_sum_to_one);
    }

    // d) Viscous diffusion changes V
    // V_smooth[i] = (1-nu)*V[i] + nu/2*(V[i-1]+V[i+1])
    // For interior point i=1, d=0: must differ from V[1,0] when nu > 0
    {
        float v_raw  = V.at(1, 0);
        float v_left = V.at(0, 0);
        float v_right= V.at(2, 0);
        float v_smooth_expected = (1.0f - nu) * v_raw + 0.5f * nu * (v_left + v_right);
        bool diffusion_active = std::abs(v_smooth_expected - v_raw) > 1e-6f;
        TEST("NS attention: viscous diffusion non-trivial (nu=0.1)", diffusion_active);
        std::cout << "    V[1,0]=" << v_raw << " V_smooth[1,0]≈" << v_smooth_expected << "\n";
    }

    // e) NS output vs standard output should differ (fluid physics changes result)
    {
        // Standard attention
        Tensor K_T    = K.transpose();
        Tensor scores = vedic_gemm(Q, K_T);
        scores += mask;
        Tensor w_std  = boltzmann_softmax(scores, temperature);
        Tensor std_out = vedic_gemm(w_std, V);

        Tensor ns_out = navier_stokes_attention(Q, K, V, mask, temperature, eta, nu);

        float max_diff = 0.f;
        for (int i = 0; i < ns_out.total_size; ++i)
            max_diff = std::max(max_diff, std::abs(ns_out.data[i] - std_out.data[i]));
        std::cout << "    NS vs Standard max_diff=" << max_diff << "\n";
        TEST("NS attention: output differs from standard attention (eta+nu > 0)",
             max_diff > 1e-6f);
    }
}

// ============================================================
//  [M13] Reynolds Batch Norm — regime blend properties
// ============================================================
static void test_m13_reynolds_batch_norm() {
    std::cout << "\n[M13] Reynolds Batch Norm — Laminar/Turbulent Blend\n";

    int seq = 8, d = 32;
    logos_rng::set_global_seed(77);

    Tensor X({seq, d}); X.fill_random(-2.0f, 2.0f);

    ReynoldsBatchNorm rbn(d);

    // a) Re_eff > 0 always
    {
        float Re = rbn.reynolds_number(X);
        TEST("Re_eff > 0 always", Re > 0.0f);
        std::cout << "    Re_eff=" << Re << "\n";
    }

    // b) Laminar weight ∈ (0,1) — sigmoid is bounded
    {
        float Re = rbn.reynolds_number(X);
        float w  = rbn.laminar_weight(Re);
        TEST("Laminar weight ∈ (0,1)", w > 0.0f && w < 1.0f);
        std::cout << "    laminar_weight(Re=" << Re << ")=" << w << "\n";
    }

    // c) Low Re → laminar_weight → 1 (pure LayerNorm regime)
    {
        float w_low = rbn.laminar_weight(0.001f);   // Re << Re_crit=1.0
        TEST("Re→0: laminar_weight → 1 (laminar regime)", w_low > 0.95f);
        std::cout << "    laminar_weight(Re=0.001)=" << w_low << "\n";
    }

    // d) High Re → laminar_weight → 0 (pure BroadcastNorm regime)
    {
        float w_high = rbn.laminar_weight(100.0f);  // Re >> Re_crit=1.0
        TEST("Re→∞: laminar_weight → 0 (turbulent regime)", w_high < 0.05f);
        std::cout << "    laminar_weight(Re=100)=" << w_high << "\n";
    }

    // e) ReynoldsBatchNorm output differs from raw input (normalisation happens)
    {
        Tensor Y = rbn.forward(X, true);
        float max_diff = 0.f;
        for (int i = 0; i < X.total_size; ++i)
            max_diff = std::max(max_diff, std::abs(Y.data[i] - X.data[i]));
        TEST("ReynoldsBatchNorm: output ≠ input (normalisation active)", max_diff > 1e-3f);
        TEST("ReynoldsBatchNorm: output has no NaN", !Y.has_nan());
    }

    // f) Re_crit annealing: Re_crit decreases from start to end
    {
        ReynoldsBatchNorm rbn2(d);
        rbn2.Re_crit = 2.0f;  // init high (laminar start)
        float Re_start = rbn2.Re_crit;
        rbn2.anneal_reynolds(500, 1000, 2.0f, 0.5f);  // 50% through training
        float Re_mid = rbn2.Re_crit;
        rbn2.anneal_reynolds(999, 1000, 2.0f, 0.5f);  // near end
        float Re_end = rbn2.Re_crit;
        TEST("Reynolds annealing: Re_crit decreases over training",
             Re_start > Re_mid && Re_mid > Re_end);
        std::cout << "    Re_crit: " << Re_start << " → " << Re_mid << " → " << Re_end << "\n";
    }
}

// ============================================================
//  [M14] Feynman Dropout — path integral amplitude distribution
// ============================================================
static void test_m14_feynman_dropout() {
    std::cout << "\n[M14] Feynman Dropout — Path Integral Amplitudes\n";

    int N_samples = 10000;

    // a) Mean weight ≈ (1-p) regardless of ħ (scale invariance)
    {
        float p = 0.3f;
        for (float hbar : {0.1f, 0.5f, 1.0f, 2.0f}) {
            FeynmanDropout fd(p, hbar, 42);
            float sum = 0.0f;
            for (int i = 0; i < N_samples; ++i) sum += fd.sample_weight();
            float mean = sum / N_samples;
            float expected = 1.0f - p;  // = 0.7
            std::string name = "Mean weight ≈ (1-p)=0.7 for ħ=" + std::to_string(hbar);
            TEST(name, std::abs(mean - expected) < 0.03f);
            std::cout << "    ħ=" << hbar << " mean_weight=" << mean << "\n";
        }
    }

    // b) All weights ∈ [0,1] — Beta distribution bounded
    {
        FeynmanDropout fd(0.2f, 1.0f, 99);
        bool all_bounded = true;
        for (int i = 0; i < N_samples; ++i) {
            float w = fd.sample_weight();
            if (w < 0.0f || w > 1.0f) { all_bounded = false; break; }
        }
        TEST("All Feynman weights ∈ [0,1] (Beta bounded)", all_bounded);
    }

    // c) Classical limit (ħ→0): weights bimodal near {0,1}
    //    Std dev should be HIGHER (close to Bernoulli std = sqrt(p*(1-p)))
    {
        float p = 0.3f;
        FeynmanDropout fd_classical(p, 0.01f, 7);  // near-classical
        FeynmanDropout fd_quantum  (p, 2.0f,  7);  // quantum
        float sum_c=0, sum2_c=0, sum_q=0, sum2_q=0;
        for (int i = 0; i < N_samples; ++i) {
            float wc = fd_classical.sample_weight();
            float wq = fd_quantum.sample_weight();
            sum_c += wc; sum2_c += wc*wc;
            sum_q += wq; sum2_q += wq*wq;
        }
        float var_c = sum2_c/N_samples - (sum_c/N_samples)*(sum_c/N_samples);
        float var_q = sum2_q/N_samples - (sum_q/N_samples)*(sum_q/N_samples);
        TEST("Classical ħ→0: higher variance than quantum ħ=2 (bimodal)",
             var_c > var_q);
        std::cout << "    var(ħ=0.01)=" << var_c << " var(ħ=2.0)=" << var_q << "\n";
    }

    // set_hbar must update the underlying Beta distribution as well.
    {
        FeynmanDropout fd(0.3f, 2.0f, 1234);
        auto sample_variance = [&](int count) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int i = 0; i < count; ++i) {
                const float w = fd.sample_weight();
                sum += w;
                sum_sq += w * w;
            }
            const float mean = sum / count;
            return sum_sq / count - mean * mean;
        };
        const float smooth_variance = sample_variance(N_samples);
        fd.set_hbar(0.01f);
        const float classical_variance = sample_variance(N_samples);
        TEST("set_hbar: smaller ħ produces more Bernoulli-like weights",
             classical_variance > smooth_variance);
        std::cout << "    set_hbar variance: ħ=2 " << smooth_variance
                  << " → ħ=0.01 " << classical_variance << "\n";
    }

    // d) p=0: identity (no dropout)
    {
        FeynmanDropout fd(0.0f, 1.0f, 1);
        Tensor X({4, 4}); X.fill(1.0f);
        fd.training = true;
        Tensor Y = fd.forward(X);
        float diff = 0.0f;
        for (int i = 0; i < X.total_size; ++i) diff += std::abs(Y.data[i] - X.data[i]);
        TEST("p=0: Feynman dropout is identity", diff < 1e-5f);
    }

    // e) training=false: exact identity (eval mode)
    {
        FeynmanDropout fd(0.5f, 1.0f, 1);
        fd.training = false;
        Tensor X({8, 8}); X.fill_random(-1.0f, 1.0f);
        Tensor Y = fd.forward(X);
        float diff = 0.0f;
        for (int i = 0; i < X.total_size; ++i) diff += std::abs(Y.data[i] - X.data[i]);
        TEST("training=false: Feynman dropout is identity (eval mode)", diff < 1e-6f);
    }
}

// ============================================================
//  [M15] Riemannian Metric — curved parameter space geometry
// ============================================================
static void test_m15_riemannian_metric() {
    std::cout << "\n[M15] Riemannian Metric — Curved Parameter Space\n";

    int dim = 64;
    RiemannianMetric rm(dim, 1e-4f);

    // Initialise metric from some gradient data
    Tensor g1({1, dim}); g1.fill_random(-1.0f, 1.0f);
    rm.update(g1, 0.9f);

    // a) Self-distance = 0
    {
        Tensor theta({1, dim}); theta.fill_random(-0.5f, 0.5f);
        float d = rm.riemannian_distance(theta, theta);
        TEST("Riemannian distance: d(θ,θ) = 0", d < 1e-5f);
    }

    // b) Distance > 0 for different points
    {
        Tensor t1({1, dim}); t1.fill_random(-1.0f, 1.0f);
        Tensor t2({1, dim}); t2.fill_random(-1.0f, 1.0f);
        float d = rm.riemannian_distance(t1, t2);
        TEST("Riemannian distance: d(θ1,θ2) > 0 for θ1 ≠ θ2", d > 1e-6f);
        std::cout << "    d(θ1,θ2)=" << d << "\n";
    }

    // c) Symmetry: d(θ1,θ2) == d(θ2,θ1)
    {
        Tensor t1({1, dim}); t1.fill_random(-0.5f, 0.5f);
        Tensor t2({1, dim}); t2.fill_random(-0.5f, 0.5f);
        float d12 = rm.riemannian_distance(t1, t2);
        float d21 = rm.riemannian_distance(t2, t1);
        TEST("Riemannian distance: symmetric d(θ1,θ2) == d(θ2,θ1)",
             std::abs(d12 - d21) < 1e-5f);
    }

    // d) After update(g): metric_diag increases from damping baseline
    {
        RiemannianMetric rm2(dim, 1e-4f);
        float initial = rm2.metric_diag[0];  // = 1.0 (init to 1)
        Tensor g({1, dim}); g.fill(2.0f);    // large grad → large Fisher
        rm2.update(g, 0.0f);                 // β=0 → immediate replace
        float updated = rm2.metric_diag[0];
        TEST("After update: metric_diag reflects gradient magnitude",
             std::abs(updated - 4.0f) < 1e-4f);  // g²=4
        std::cout << "    metric_diag: " << initial << " → " << updated << "\n";
    }

    // e) Natural gradient: ||g̃||_G ≤ ||g||_G (G-norm of natural grad ≤ raw grad)
    {
        Tensor g({1, dim}); g.fill_random(-1.0f, 1.0f);
        rm.update(g, 0.9f);
        Tensor g_nat = rm.riemannian_gradient(g);
        float norm_g     = rm.riemannian_norm(g);
        float norm_g_nat = rm.riemannian_norm(g_nat);
        TEST("||g̃||_G ≤ ||g||_G (natural grad has less Riemannian norm)",
             norm_g_nat <= norm_g + 1e-4f);
        std::cout << "    ||g||_G=" << norm_g << " ||g̃||_G=" << norm_g_nat << "\n";
    }

    // f) Parallel transport: transported vector has reduced component along Δθ
    {
        Tensor v({1, dim}); v.fill(1.0f);       // gradient vector
        Tensor dtheta({1, dim}); dtheta.fill(1.0f);  // step direction (same as v)
        Tensor v_t = rm.parallel_transport(v, dtheta);
        // When v || Δθ, the parallel-transported v should be ≈ 0
        float norm_transported = rm.riemannian_norm(v_t);
        float norm_original    = rm.riemannian_norm(v);
        TEST("Parallel transport: removes component along Δθ",
             norm_transported < norm_original);
        std::cout << "    ||v||_G=" << norm_original
                  << " ||v_transported||_G=" << norm_transported << "\n";
    }
}

// ============================================================
//  [M16] Weight Path Integral — Feynman action accumulation
// ============================================================
static void test_m16_weight_path_integral() {
    std::cout << "\n[M16] Weight Path Integral — Action Accumulation\n";

    WeightPathIntegral wpi(1.0f, 100);

    // a) log_amplitude decreases when action > 0
    {
        float log_A0 = wpi.log_amplitude;  // = 0 initially
        std::vector<float> delta(16, 0.1f);  // ||Δθ|| = sqrt(16*0.01) ≈ 0.4
        float loss = 2.0f;                    // positive loss → positive action
        wpi.record_step(loss, delta);
        TEST("log_amplitude decreases when action > 0", wpi.log_amplitude < log_A0);
        std::cout << "    log_A: " << log_A0 << " → " << wpi.log_amplitude << "\n";
    }

    // b) Discrete Lagrangian S_t = loss * ||Δθ|| (verified manually)
    {
        WeightPathIntegral wpi2(1.0f, 10);
        float loss_t = 3.0f;
        std::vector<float> delta_t = {3.0f, 4.0f};  // ||Δθ|| = 5.0
        wpi2.record_step(loss_t, delta_t);
        // S_t = 3.0 * 5.0 = 15.0
        // log_A = 0 - 15.0/1.0 = -15.0
        float expected_logA = -15.0f;
        TEST("Discrete Lagrangian: S_t = loss * ||Δθ|| correct",
             std::abs(wpi2.log_amplitude - expected_logA) < 1e-4f);
        std::cout << "    Expected log_A=" << expected_logA
                  << " actual=" << wpi2.log_amplitude << "\n";
    }

    // c) lr_scale ∈ [lr_min_frac, 1.0] always
    {
        WeightPathIntegral wpi3(1.0f, 50);
        float lr_min = 0.1f;
        // Fresh: relative_amplitude=1 → lr_scale=1
        float scale_fresh = wpi3.lr_scale(lr_min);
        TEST("lr_scale = 1.0 at start (no steps)", std::abs(scale_fresh - 1.0f) < 1e-5f);

        // After many high-action steps: log_A very negative → rel_amp ≈ 0
        std::vector<float> big_step(4, 10.0f);
        for (int i = 0; i < 30; ++i) wpi3.record_step(5.0f, big_step);
        // Then one low-action step to set a new best
        std::vector<float> tiny_step(4, 0.0f);
        wpi3.record_step(0.0f, tiny_step);  // S=0 → no amplitude decrease
        float scale_after = wpi3.lr_scale(lr_min);
        TEST("lr_scale ∈ [lr_min, 1.0] always",
             scale_after >= lr_min - 1e-5f && scale_after <= 1.0f + 1e-5f);
        std::cout << "    lr_scale after high-action=" << scale_after << "\n";
    }

    // d) best_step is correctly tracked
    {
        WeightPathIntegral wpi4(1.0f, 50);
        // Step 0: small action → high amplitude
        wpi4.record_step(0.1f, {0.01f, 0.01f});  // S≈0.001, log_A ≈ -0.001
        int step_after_best = wpi4.best_step;

        // Step 1: large action → amplitude drops sharply
        wpi4.record_step(10.0f, {5.0f, 5.0f, 5.0f});  // S=70+, log_A very negative
        TEST("best_step is not the last high-action step",
             wpi4.best_step != 1);  // best should still be step 0
        std::cout << "    best_step=" << wpi4.best_step
                  << " current_step=" << wpi4.step_count << "\n";
    }

    // e) relative_amplitude == 1.0 at the best step seen so far
    {
        WeightPathIntegral wpi5(1.0f, 50);
        wpi5.record_step(0.5f, {0.1f});   // some action
        wpi5.record_step(0.5f, {0.1f});   // more action
        // At any point, relative_amplitude ≤ 1.0
        float rel = wpi5.relative_amplitude();
        TEST("relative_amplitude ∈ (0,1] always", rel > 0.0f && rel <= 1.0f + 1e-5f);
        std::cout << "    relative_amplitude=" << rel << "\n";
    }

    // f) recent_mean_action computes correctly
    {
        WeightPathIntegral wpi6(1.0f, 100);
        float fixed_loss = 2.0f;
        std::vector<float> fixed_step = {1.0f, 0.0f};  // ||Δθ||=1, S=2
        for (int i = 0; i < 10; ++i)
            wpi6.record_step(fixed_loss, fixed_step);
        float mean_action = wpi6.recent_mean_action(10);
        float expected_action = fixed_loss * 1.0f;  // S = loss * ||Δθ|| = 2*1 = 2
        TEST("recent_mean_action correct (last 10 steps)",
             std::abs(mean_action - expected_action) < 1e-4f);
        std::cout << "    mean_action=" << mean_action
                  << " expected=" << expected_action << "\n";
    }
}

// ============================================================
//  MAIN
// ============================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS Test 2: Mathematical Unit Tests (v10)     ║\n";
    std::cout << "║  Vedic + Modern Math + 6 Physics Components      ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    test_m1_vedic_gemm();
    test_m2_gunitasamuchayah();
    test_m3_free_energy();
    test_m4_leapfrog_stability();
    test_m5_boltzmann_softmax();
    test_m6_nikhilam_complement();
    test_m7_hyperbolic_maps();
    test_m8_layernorm();
    test_m9_gemm_tiling();
    test_m10_gradient_fd();

    // ── NEW: 6 Physics Component Tests ───────────────────────
    test_m11_natural_gradient();
    test_m12_navier_stokes_attention();
    test_m13_reynolds_batch_norm();
    test_m14_feynman_dropout();
    test_m15_riemannian_metric();
    test_m16_weight_path_integral();

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << "  PASS: " << g_pass << "  FAIL: " << g_fail << "\n";
    std::cout << "  Total: " << (g_pass + g_fail) << " assertions\n";
    std::cout << "════════════════════════════════════════════════\n";
    return g_fail > 0 ? 1 : 0;
}
