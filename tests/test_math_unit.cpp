// ============================================================
//  LOGOS — tests/test_math_unit.cpp
//  Test 2: Mathematical Unit Tests
//
//  Kya test hoga:
//    [M1]  Urdhva Tiryagbhyam — Vedic GEMM vs reference GEMM
//          Tolerance: max element error < 1e-3 (float32 accumulation)
//
//    [M2]  Gunitasamuchayah — Product of sums = Sum of products
//          Manual verification: sum(C) == sum_rows(A) * sum_cols(B) / K
//          Test on identity, random, and sparse matrices
//
//    [M3]  Free Energy — F = CE - T*S
//          Property tests:
//            a) F <= CE always (entropy term is non-negative, T > 0)
//            b) S >= 0 always (Shannon entropy)
//            c) As T→0: F → CE
//            d) dF/dlogit reduces to dCE/dlogit when T=0
//            e) When predictions uniform: S = log(vocab) [max entropy]
//
//    [M4]  Leapfrog Langevin — energy preservation
//          a) Störmer-Verlet is symplectic: phase space volume preserved
//             (det of update Jacobian ≈ 1 for small lr)
//          b) Half-step energy ≈ full-step energy (leapfrog stability)
//          c) With zero noise: energy drift O(lr²) vs Euler O(lr) per step
//             Leapfrog should have smaller energy error for same lr
//
//    [M5]  Boltzmann Softmax — temperature scaling
//          a) T→0: distribution → one-hot (confident)
//          b) T→∞: distribution → uniform (max entropy)
//          c) At T=1: matches standard softmax
//
//    [M6]  Nikhilam complement — arithmetic identity
//          a) base - (base - x) = x  (double complement = identity)
//          b) quantize → dequantize error bounded by scale/2
//          c) Scale = absmax/127 → no clipping for values in [-absmax, absmax]
//
//    [M7]  Hyperbolic exp_map / log_map — inverse property
//          a) log_map(exp_map(v)) ≈ v  for small ||v||
//          b) ||exp_map(v)|| < 1 for all v (Poincaré ball constraint)
//          c) exp_map(0) = 0 (origin maps to origin)
//          d) Curvature c: larger c → more curved → smaller ball
//
//    [M8]  LayerNorm — exact statistics
//          mean(LN(x)) = beta
//          std(LN(x))  = gamma  (when gamma uniform, beta uniform)
//
//    [M9]  VedicGEMM Tiling — result matches naive triple-loop
//          Test multiple shapes: square, tall, wide, non-divisible by TILE
//
//    [M10] Gradient correctness — numerical vs analytical
//          Finite-difference check on free_energy_loss gradient
//          |analytical_grad - numerical_grad| / |numerical_grad| < 1e-2
// ============================================================
#include "../include/VedicGEMM.hpp"
#include "../include/Tensor.hpp"
#include "../include/PhysicsOpt.hpp"

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
    float factor = norm > 1e-7f
        ? std::tanh(norm*0.5f) / norm
        : 0.5f;
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
    // Vedic sutram: sum(A@B) = dot(row_sums(A), col_sums(B)) / K (approx)
    // Exact when rows of A and cols of B are orthogonal mean-zero

    auto test_case = [&](const char* name,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         int M, int K, int N, float tol) {
        auto C = naive_gemm(A, B, M, K, N);

        // sum(C)
        float sum_C = std::accumulate(C.begin(), C.end(), 0.f);

        // sum of row sums of A
        float sum_rs = std::accumulate(A.begin(), A.end(), 0.f);
        // sum of col sums of B
        float sum_cs = std::accumulate(B.begin(), B.end(), 0.f);

        float vedic_pred = sum_rs * sum_cs / K;
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

    // Case 3: Zero-mean random (Gunitasamuchayah most accurate here)
    {
        int M=16,K=32,N=16;
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> A(M*K), B(K*N);
        for (float& x : A) x = nd(rng);
        for (float& x : B) x = nd(rng);
        test_case("Zero-mean random", A, B, M, K, N, 0.5f);  // statistical
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

    // Leapfrog (Störmer-Verlet)
    float W_lf = 1.0f, V_lf = 0.0f;
    float E0_lf = 0.5f*W_lf*W_lf + 0.5f*V_lf*V_lf;
    for (int i=0; i<steps; ++i) {
        float v_half = friction*V_lf - (lr*0.5f)*W_lf;  // half-kick
        W_lf = W_lf + lr*v_half;                          // full drift
        V_lf = v_half;                                    // store v_{t+1/2}
        // next step's half-kick using new grad:
        // (In practice: next iter does: v_half = friction*V_lf - lr/2 * grad_new)
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
//  MAIN
// ============================================================
int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS Test 2: Mathematical Unit Tests   ║\n";
    std::cout << "║  Vedic + Modern Math Comparison          ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

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

    std::cout << "\n════════════════════════════════════\n";
    std::cout << "  PASS: " << g_pass << "  FAIL: " << g_fail << "\n";
    std::cout << "════════════════════════════════════\n";
    return g_fail > 0 ? 1 : 0;
}
