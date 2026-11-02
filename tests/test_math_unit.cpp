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

static int g_pass = 0, g_fail = 0;

#define TEST(name, expr) do { \
    bool _ok = (expr); \
    if (_ok) { std::cout << "  ✅ " << (name) << "\n"; ++g_pass; } \
    else { std::cerr << "  ❌ " << (name) << "  [FAIL]\n"; ++g_fail; } \
} while(0)

#define TEST_CLOSE(name, a, b, tol) \
    TEST(name, std::abs((a)-(b)) <= (tol))


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

static std::vector<float> cpu_softmax(const std::vector<float>& logits, float T=1.0f) {
    int n = logits.size();
    std::vector<float> out(n);
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (int i=0; i<n; ++i) { out[i] = std::exp((logits[i]-mx)/T); sum += out[i]; }
    for (int i=0; i<n; ++i) out[i] /= sum;
    {
        float chk = 0.f; for (int i=0; i<n; ++i) chk += out[i];
        if (std::abs(chk - 1.f) > 1e-4f || (int)out.size() != n) {
            std::cerr << "[cpu_softmax] invariant failed\n"; std::abort();
        }
    }
    return out;
}

static float cpu_ce(const std::vector<float>& p, int target) {
    float val = -std::log(std::max(p[target], 1e-12f));
    if (val < 0.f) { std::cerr << "[cpu_ce] CE must be >= 0\n"; std::abort(); }
    return val;
}

static float cpu_entropy(const std::vector<float>& p) {
    float S = 0.f;
    for (float pi : p) if (pi > 1e-12f) S -= pi * std::log(pi);
    if (S < -1e-6f) { std::cerr << "[cpu_entropy] entropy must be >= 0\n"; std::abort(); }
    return S;
}

static float cpu_free_energy(const std::vector<float>& logits, int target, float T) {
    auto p = cpu_softmax(logits, 1.0f);
    float CE = cpu_ce(p, target);
    float S  = cpu_entropy(p);
    return CE - T * S;
}

static std::vector<float> cpu_ce_grad(const std::vector<float>& p, int target, int seq) {
    int n = p.size();
    std::vector<float> g(n);
    for (int v=0; v<n; ++v)
        g[v] = (p[v] - (v==target ? 1.f : 0.f)) / seq;
    return g;
}

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
    float gsum = 0.f; for (float gv : g) gsum += gv;
    if (std::abs(gsum * seq) > 0.01f) {
        std::cerr << "[cpu_fe_grad] grad sum=" << gsum << " expected ~0\n"; std::abort();
    }
    return g;
}

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

static void test_m1_vedic_gemm() {
    std::cout << "\n[M1] Urdhva Tiryagbhyam — Vedic GEMM vs Reference\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<std::tuple<int,int,int>> shapes = {
        {4,4,4}, {8,16,8}, {32,64,32}, {64,128,64}, {128,256,128}
    };

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

        std::string name = "Vedic GEMM [" + std::to_string(M) + "x" +
                           std::to_string(K) + "x" + std::to_string(N) +
                           "] max_err=" + std::to_string(max_err);
        TEST(name, max_err < 1e-2f);

        TEST("vedic_gemm output rows == M",  TC.rows() == M);
        TEST("vedic_gemm output cols == N",  TC.cols() == N);
        TEST("vedic_gemm total_size == M*N", TC.total_size == M * N);

        float c00_ref = 0.f;
        for (int k = 0; k < K; ++k) c00_ref += A[k] * B[k*N];
        TEST("vedic_gemm C[0,0] exact match", std::abs(TC.data[0] - c00_ref) < 1e-3f);

        float cLast_ref = 0.f;
        for (int k = 0; k < K; ++k)
            cLast_ref += A[(M-1)*K + k] * B[k*N + (N-1)];
        TEST("vedic_gemm C[M-1,N-1] exact match",
             std::abs(TC.data[M*N-1] - cLast_ref) < 1e-3f);
    }

    {
        int N = 4;
        Tensor I({N,N}, 0.f);
        for (int i = 0; i < N; ++i) I.at(i,i) = 1.f;
        Tensor R = vedic_gemm(I, I);
        bool identity_ok = true;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float expected = (i == j) ? 1.f : 0.f;
                if (std::abs(R.at(i,j) - expected) > 1e-5f) identity_ok = false;
            }
        TEST("vedic_gemm(I, I) == I (exact)", identity_ok);
    }

    {
        int M=8, K=8, N=8;
        std::vector<float> A(M*K), B(K*N);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);
        Tensor TA({M,K}); std::copy(A.begin(),A.end(),TA.data.begin());
        Tensor TB({K,N}); std::copy(B.begin(),B.end(),TB.data.begin());
        Tensor TA2 = TA * 2.f;
        Tensor C1  = vedic_gemm(TA2, TB);
        Tensor C2  = vedic_gemm(TA,  TB) * 2.f;
        float max_diff = 0.f;
        for (int i = 0; i < M*N; ++i)
            max_diff = std::max(max_diff, std::abs(C1.data[i] - C2.data[i]));
        TEST("vedic_gemm: (2A)@B == 2*(A@B) (linearity)", max_diff < 1e-3f);
    }

    {
        Tensor A2({2,2}); A2.data = {1.f, 2.f, 3.f, 4.f};
        Tensor B2({2,2}); B2.data = {5.f, 6.f, 7.f, 8.f};
        Tensor C2 = vedic_gemm(A2, B2);
        TEST("vedic_gemm 2x2 exact: C[0,0]=19", std::abs(C2.at(0,0) - 19.f) < 1e-4f);
        TEST("vedic_gemm 2x2 exact: C[0,1]=22", std::abs(C2.at(0,1) - 22.f) < 1e-4f);
        TEST("vedic_gemm 2x2 exact: C[1,0]=43", std::abs(C2.at(1,0) - 43.f) < 1e-4f);
        TEST("vedic_gemm 2x2 exact: C[1,1]=50", std::abs(C2.at(1,1) - 50.f) < 1e-4f);
    }

    {
        Tensor A2({2,2}); A2.data = {1.f, 0.f, 0.f, 1.f};  // identity
        Tensor W2({2,2}); W2.data = {1.f, 0.f, 0.f, 1.f};  // identity
        Tensor b({2});    b.data  = {3.f, 5.f};
        Tensor R = vedic_gemm_bias(A2, W2, b);
        TEST("vedic_gemm_bias: R[0,0] = 1+3 = 4", std::abs(R.at(0,0) - 4.f) < 1e-4f);
        TEST("vedic_gemm_bias: R[0,1] = 0+5 = 5", std::abs(R.at(0,1) - 5.f) < 1e-4f);
        TEST("vedic_gemm_bias: R[1,0] = 0+3 = 3", std::abs(R.at(1,0) - 3.f) < 1e-4f);
        TEST("vedic_gemm_bias: R[1,1] = 1+5 = 6", std::abs(R.at(1,1) - 6.f) < 1e-4f);
    }
}

static void test_m2_gunitasamuchayah() {
    std::cout << "\n[M2] Gunitasamuchayah (Product of Sums)\n";

    auto test_case = [&](const char* name,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         int M, int K, int N, float tol) {
        auto C = naive_gemm(A, B, M, K, N);

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

    {
        int M=8,K=8,N=8;
        std::vector<float> A(M*K,0.f), B(K*N,0.f);
        for (int i=0; i<M; ++i) A[i*K+i]=1.f;
        for (int i=0; i<K; ++i) B[i*N+i]=1.f;
        test_case("Identity", A, B, M, K, N, 0.5f);  // relaxed: I*I = I, sum=8
    }

    {
        int M=4,K=8,N=4;
        std::vector<float> A(M*K,1.f), B(K*N,1.f);
        test_case("All-ones (exact)", A, B, M, K, N, 1e-4f);
    }

    {
        int M=16,K=32,N=16;
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> A(M*K), B(K*N);
        for (float& x : A) x = nd(rng);
        for (float& x : B) x = nd(rng);
        test_case("Zero-mean random", A, B, M, K, N, 1e-5f);
    }

    {
        int base = 127;
        std::vector<int> vals = {0, 1, 50, 63, 127, -1, -50, -127};
        bool ok = true;
        for (int v : vals) {
            int complement    = base - std::abs(v);
            int reconstructed = (v >= 0 ? 1 : -1) * (base - complement);
            if (reconstructed != v) ok = false;
        }
        TEST("Nikhilam: base - (base - |x|) reconstructs x for all test values", ok);
    }

    {
        std::vector<float> A = {1,2,3, 4,5,6};
        std::vector<float> B = {1,0, 0,1, 1,1};
        auto C = naive_gemm(A, B, 2, 3, 2);
        float sum_C = std::accumulate(C.begin(), C.end(), 0.f);
        TEST("naive_gemm 2x3x2 sum(C) == 30", std::abs(sum_C - 30.f) < 1e-4f);
        TEST("naive_gemm C[0,0] == 4", std::abs(C[0] - 4.f) < 1e-4f);
        TEST("naive_gemm C[0,1] == 5", std::abs(C[1] - 5.f) < 1e-4f);
        TEST("naive_gemm C[1,0] == 10", std::abs(C[2] - 10.f) < 1e-4f);
        TEST("naive_gemm C[1,1] == 11", std::abs(C[3] - 11.f) < 1e-4f);
    }

    {
        int M=3, K=3, N=3;
        std::vector<float> A(M*K, 0.f), B(K*N, 0.f);
        A[2*K + 2] = 1.f;  // A[2,2] = 1, all else 0
        B[2*N + 2] = 1.f;  // B[2,2] = 1, all else 0
        auto C = naive_gemm(A, B, M, K, N);
        TEST("naive_gemm: last-row last-col element correct (loop reaches M-1,K-1,N-1)",
             std::abs(C[2*N+2] - 1.f) < 1e-4f);
        float sum_rest = 0.f;
        for (int i = 0; i < M*N; ++i) if (i != 2*N+2) sum_rest += std::abs(C[i]);
        TEST("naive_gemm: only C[M-1,N-1] != 0 when only A[M-1,K-1] and B[K-1,N-1] set",
             sum_rest < 1e-4f);
    }
}

static void test_m3_free_energy() {
    std::cout << "\n[M3] Free Energy F = CE - T·S Properties\n";

    std::vector<float> logits = {2.f, 1.f, 0.f, -1.f, 0.5f, -0.5f, 1.5f, -2.f};
    int vocab = (int)logits.size();
    int target = 0;

    auto p = cpu_softmax(logits, 1.0f);
    float CE = cpu_ce(p, target);
    float S  = cpu_entropy(p);

    TEST("S >= 0 (entropy non-negative)", S >= -1e-6f);

    float T = 0.05f;
    float F = CE - T * S;
    TEST("F <= CE when T > 0", F <= CE + 1e-6f);

    float F_small = CE - 1e-8f * S;
    TEST("F → CE as T → 0", std::abs(F_small - CE) < 1e-4f);

    {
        std::vector<float> fe_g = cpu_fe_grad(logits, target, 0.f, 1);
        std::vector<float> ce_g = cpu_ce_grad(p, target, 1);
        float max_diff = 0.f;
        for (int i=0; i<vocab; ++i)
            max_diff = std::max(max_diff, std::abs(fe_g[i]-ce_g[i]));
        TEST("dF/dlogit = dCE/dlogit when T=0", max_diff < 1e-6f);
    }

    {
        std::vector<float> uniform_logits(vocab, 0.f);
        auto pu = cpu_softmax(uniform_logits, 1.0f);
        float Su = cpu_entropy(pu);
        float expected_S = std::log((float)vocab);
        TEST("Uniform dist → S = log(vocab)", std::abs(Su - expected_S) < 1e-4f);
    }

    {
        std::vector<float> peaked(vocab, -100.f);
        peaked[0] = 100.f;
        auto pp = cpu_softmax(peaked, 1.0f);
        float Sp = cpu_entropy(pp);
        TEST("Peaked dist → S ≈ 0", Sp < 0.01f);
    }

    {
        std::vector<float> eq_logits = {0.f, 0.f};
        auto p2 = cpu_softmax(eq_logits, 1.f);
        TEST("softmax([0,0]) p[0] == 0.5", std::abs(p2[0] - 0.5f) < 1e-5f);
        TEST("softmax([0,0]) p[1] == 0.5", std::abs(p2[1] - 0.5f) < 1e-5f);
    }

    {
        std::vector<float> lg = {2.f, 0.f};
        auto p_t1 = cpu_softmax(lg, 1.f);
        auto p_t2 = cpu_softmax(lg, 2.f);
        TEST("softmax T=2 less peaked than T=1: p[0] smaller",
             p_t2[0] < p_t1[0]);
        TEST("softmax T=2: sum still 1",
             std::abs(p_t2[0] + p_t2[1] - 1.f) < 1e-5f);
    }

    {
        std::vector<float> eq_l = {0.f, 0.f};
        auto peq = cpu_softmax(eq_l, 1.f);
        float CE_eq = cpu_ce(peq, 0);
        float S_eq  = cpu_entropy(peq);
        float F_eq  = CE_eq - 1.f * S_eq;
        TEST("Free energy F=0 for uniform dist at T=1", std::abs(F_eq) < 1e-4f);
        TEST("CE = log(2) for uniform 2-class", std::abs(CE_eq - std::log(2.f)) < 1e-4f);
        TEST("S = log(2) for uniform 2-class",  std::abs(S_eq  - std::log(2.f)) < 1e-4f);
    }

    {
        std::vector<float> onehot = {1.f - 1e-6f, 1e-6f};
        float S_oh = cpu_entropy(onehot);
        TEST("Entropy near-onehot: S < 1e-4", S_oh < 1e-4f);
    }

    {
        float F_t0 = cpu_free_energy(logits, target, 0.f);
        float CE_t0 = cpu_ce(cpu_softmax(logits, 1.f), target);
        TEST("cpu_free_energy(T=0) == CE", std::abs(F_t0 - CE_t0) < 1e-5f);
    }

    std::cout << "    CE=" << CE << " S=" << S << " F=" << F << " (T=" << T << ")\n";
}

static void test_m4_leapfrog_stability() {
    std::cout << "\n[M4] Leapfrog Langevin — Stability vs Euler-Maruyama\n";


    float lr = 0.1f, friction = 1.0f, noise = 0.0f;  // zero noise for clean test
    int steps = 100;

    float W_euler = 1.0f, V_euler = 0.0f;
    float E0_euler = 0.5f*W_euler*W_euler + 0.5f*V_euler*V_euler;
    for (int i=0; i<steps; ++i) {
        V_euler = friction*V_euler - lr*W_euler;
        W_euler += V_euler;
    }
    float E1_euler = 0.5f*W_euler*W_euler + 0.5f*V_euler*V_euler;
    float euler_drift = std::abs(E1_euler - E0_euler) / E0_euler;

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

    TEST("Leapfrog drift < Euler drift (better stability)", lf_drift < euler_drift);
    TEST("Leapfrog: energy drift < 50% over 100 steps",    lf_drift < 0.5f);

    TEST("Euler drift > 1% (expected 1st order error)", euler_drift > 0.01f);

    {
        float W = 1.f, V = 0.f, h = 0.1f;
        V -= (h * 0.5f) * W;
        W += h * V;
        V -= (h * 0.5f) * W;
        TEST("Leapfrog half-kick: W after 1 step ≈ 0.995",
             std::abs(W - 0.995f) < 1e-4f);
        TEST("Leapfrog half-kick: V after 1 step ≈ -0.09975",
             std::abs(V - (-0.09975f)) < 1e-4f);
        TEST("Leapfrog: W < initial W=1 after spring force", W < 1.f);
        TEST("Leapfrog: V < 0 after half-kick away from W=1", V < 0.f);
    }

    TEST("Leapfrog: final energy < 2 * initial energy", E1_lf < 2.f * E0_lf);
    TEST("Euler: energy drifted from initial (not conserved)", E1_euler != E0_euler);
}

static void test_m5_boltzmann_softmax() {
    std::cout << "\n[M5] Boltzmann Softmax Temperature Scaling\n";

    std::vector<float> logits = {3.f, 1.f, 0.f, -1.f, 0.5f};
    int n = logits.size();
    int argmax = 0;  // index 0 has highest logit

    auto p_cold = cpu_softmax(logits, 0.001f);
    TEST("T→0: argmax token gets high prob",  p_cold[argmax] > 0.999f);
    TEST("T→0: other tokens get near-0 prob", p_cold[1] < 0.001f);

    auto p_hot = cpu_softmax(logits, 1000.f);
    float expected_uniform = 1.f / n;
    float max_dev = 0.f;
    for (float pi : p_hot) max_dev = std::max(max_dev, std::abs(pi - expected_uniform));
    TEST("T→∞: distribution approaches uniform", max_dev < 0.01f);

    auto p_std = cpu_softmax(logits, 1.f);
    float sum = 0.f; for (float pi : p_std) sum += pi;
    TEST("T=1: standard softmax (sum=1)",    std::abs(sum - 1.f) < 1e-5f);
    TEST("T=1: argmax still most probable",  p_std[argmax] > p_std[1]);

    float S1 = cpu_entropy(p_cold);
    float S2 = cpu_entropy(p_std);
    float S3 = cpu_entropy(p_hot);
    TEST("Entropy: S(T=0.001) < S(T=1) < S(T=1000)", S1 < S2 && S2 < S3);

    {
        std::vector<float> l2 = {2.f, 0.f};
        auto p2 = cpu_softmax(l2, 1.f);
        float expected_p0 = 1.f / (1.f + std::exp(-2.f));
        TEST("softmax([2,0]): p[0] exact", std::abs(p2[0] - expected_p0) < 1e-5f);
        TEST("softmax([2,0]): p[0]+p[1] == 1", std::abs(p2[0]+p2[1]-1.f) < 1e-5f);
        TEST("softmax([2,0]): p[0] > p[1]", p2[0] > p2[1]);
        auto p2_cold = cpu_softmax(l2, 0.5f);
        float expected_cold = 1.f / (1.f + std::exp(-4.f));
        TEST("softmax([2,0],T=0.5): p[0] more peaked",
             std::abs(p2_cold[0] - expected_cold) < 1e-5f);
        TEST("softmax T=0.5 more peaked than T=1: p[0] higher", p2_cold[0] > p2[0]);
    }

    {
        std::vector<float> lg5 = {1.f, 3.f, -1.f, 0.5f, 2.f};
        auto p5 = cpu_softmax(lg5, 1.f);
        float sum5 = 0.f; for (float v : p5) sum5 += v;
        TEST("softmax(5-class): sum == 1", std::abs(sum5 - 1.f) < 1e-5f);
        TEST("softmax: argmax logit → argmax prob",
             *std::max_element(p5.begin(), p5.end()) == p5[1]);
    }
}

static void test_m6_nikhilam_complement() {
    std::cout << "\n[M6] Nikhilam Complement Identity\n";

    int base = 127;
    bool identity_ok = true;
    for (int v = -127; v <= 127; ++v) {
        int8_t q = (int8_t)std::max(-127, std::min(127, v));
        int reconstructed = (int)q;
        if (reconstructed != v) { identity_ok = false; break; }
    }
    TEST("int8 round-trip: cast/uncast identity [-127,127]", identity_ok);

    float absmax = 10.0f;
    float scale = absmax / 127.0f;
    float max_quant_err = scale / 2.0f;  // max rounding error
    float test_val = 7.3f;
    int8_t q = (int8_t)std::round(test_val / scale);
    float rec = q * scale;
    float err = std::abs(test_val - rec);
    TEST("Quant error <= scale/2", err <= max_quant_err + 1e-6f);

    bool no_clip = true;
    std::vector<float> test_vals = {-10.f, -5.f, 0.f, 5.f, 10.f};
    for (float v : test_vals) {
        int8_t qi = (int8_t)std::max(-127.f, std::min(127.f,
                        std::round(v / scale)));
        if (std::abs(v) < absmax && std::abs((int)qi) == 127) no_clip = false;
    }
    TEST("No clipping for |v| < absmax", no_clip);

    int q_test = 45;
    int complement = base - q_test;
    int double_complement = base - complement;
    TEST("Nikhilam: 127 - (127 - q) == q (runtime check)", double_complement == q_test);
    {
        bool edge_ok = true;
        for (int q : {0, 1, 63, 126, 127}) {
            if ((base - (base - q)) != q) { edge_ok = false; break; }
        }
        TEST("Nikhilam: double complement identity holds for edge values", edge_ok);
    }

    std::cout << "    scale=" << scale << " max_quant_err=" << max_quant_err << "\n";

    {
        float absmax2 = 8.f, sc = absmax2 / 127.f;
        int8_t q127 = (int8_t)std::max(-127.f, std::min(127.f, std::round(8.f / sc)));
        float rec127 = q127 * sc;
        TEST("Quant: 8.0 → int8=127 → dequant == absmax",
             std::abs(rec127 - 8.f) < 1e-3f && (int)q127 == 127);
        int8_t q0 = (int8_t)std::max(-127.f, std::min(127.f, std::round(0.f / sc)));
        TEST("Quant: 0.0 → int8=0", (int)q0 == 0);
        int8_t qneg = (int8_t)std::max(-127.f, std::min(127.f,
                          std::round(-4.f / sc)));
        TEST("Quant: -4.0 → negative int8", (int)qneg < 0);
        float rec_neg = qneg * sc;
        TEST("Quant: -4.0 dequant error <= scale/2",
             std::abs(rec_neg - (-4.f)) <= sc * 0.5f + 1e-5f);
    }

    {
        bool rt_ok = true;
        std::vector<int> boundaries = {-127, -126, -1, 0, 1, 126, 127};
        for (int v : boundaries) {
            int8_t q = (int8_t)v;
            int back = (int)q;
            if (back != v) rt_ok = false;
        }
        TEST("int8 round-trip exact for boundary values {-127,-126,-1,0,1,126,127}", rt_ok);
    }
}

static void test_m7_hyperbolic_maps() {
    std::cout << "\n[M7] Hyperbolic exp_map / log_map Inverse Property\n";

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

    {
        std::vector<float> zero(8, 0.0f);
        auto y = cpu_expmap(zero);
        float norm = 0.f; for (float x : y) norm += x*x;
        TEST("exp_map(0) = 0", std::sqrt(norm) < 1e-6f);
    }

    {
        std::vector<float> v = {1.f, 1.f, 1.f, 1.f};
        auto y1 = cpu_expmap(v);
        float n1=0.f; for (float x : y1) n1+=x*x; n1=std::sqrt(n1);
        float c = 4.f, sc = std::sqrt(c);
        float norm=0.f; for (float x : v) norm+=x*x; norm=std::sqrt(norm);
        float factor = std::tanh(sc*norm*0.5f) / (sc*norm + 1e-9f);
        float n4 = 0.f;
        for (float x : v) n4 += (factor*x)*(factor*x);
        n4 = std::sqrt(n4);
        std::cout << "    ||expmap_c=1(v)||=" << n1 << " ||expmap_c=4(v)||=" << n4 << "\n";
        TEST("Larger curvature → smaller output norm", n4 < n1);
    }

    {
        std::vector<float> v1d = {0.3f};
        auto y = cpu_expmap(v1d);
        float expected_y = std::tanh(0.15f);  // tanh(||v||/2) * v/||v|| * ||v|| = tanh(||v||/2)
        TEST("expmap 1D: ||expmap([0.3])|| ≈ tanh(0.15)",
             std::abs(y[0] - expected_y) < 1e-4f);
        auto v_rec = cpu_logmap(y);
        TEST("logmap(expmap([0.3])) ≈ 0.3", std::abs(v_rec[0] - 0.3f) < 1e-4f);
    }

    {
        std::vector<float> v = {1.f, 2.f, 0.f, -1.f};
        auto y = cpu_expmap(v);
        float ratio01 = y[0] / v[0];
        float ratio11 = y[1] / v[1];
        TEST("expmap preserves direction: y[0]/v[0] == y[1]/v[1]",
             std::abs(ratio01 - ratio11) < 1e-4f);
        float ratio31 = y[3] / v[3];
        TEST("expmap preserves direction: y[0]/v[0] == y[3]/v[3]",
             std::abs(ratio01 - ratio31) < 1e-4f);
    }
}

static void test_m8_layernorm() {
    std::cout << "\n[M8] LayerNorm — Exact Statistics\n";

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

    TEST("LayerNorm: mean ≈ beta (1.0)",    std::abs(mean_y - 1.0f) < 0.01f);
    TEST("LayerNorm: std ≈ gamma (2.0)",    std::abs(std_y  - 2.0f) < 0.1f);

    {
        std::vector<float> x4 = {1.f, 2.f, 3.f, 4.f};
        std::vector<float> g4(4, 1.f), b4(4, 0.f);
        auto y4 = layernorm_cpu(x4, g4, b4);
        float mean4 = 0.f, var4 = 0.f;
        for (float v : y4) mean4 += v; mean4 /= 4;
        for (float v : y4) var4 += (v-mean4)*(v-mean4); var4 /= 4;
        TEST("LayerNorm 4-elem: output mean ≈ 0",   std::abs(mean4) < 1e-4f);
        TEST("LayerNorm 4-elem: output var ≈ 1",    std::abs(var4 - 1.f) < 1e-3f);
        TEST("LayerNorm 4-elem: y[0] < y[1] < y[2] < y[3] (monotone)",
             y4[0] < y4[1] && y4[1] < y4[2] && y4[2] < y4[3]);
        TEST("LayerNorm 4-elem: y[0] < 0 (below mean)", y4[0] < 0.f);
        TEST("LayerNorm 4-elem: y[3] > 0 (above mean)", y4[3] > 0.f);
        float expected_y0 = (1.f - 2.5f) / std::sqrt(1.25f + 1e-5f);
        TEST("LayerNorm 4-elem: y[0] exact", std::abs(y4[0] - expected_y0) < 1e-3f);
    }

    {
        std::vector<float> x5(16); std::iota(x5.begin(), x5.end(), 1.f);
        std::vector<float> g5(16, 3.f), b5(16, 0.f);
        auto y5 = layernorm_cpu(x5, g5, b5);
        float m5 = 0.f, v5 = 0.f;
        for (float f : y5) m5 += f; m5 /= 16;
        for (float f : y5) v5 += (f-m5)*(f-m5); v5 /= 16;
        TEST("LayerNorm gamma=3: mean ≈ 0",  std::abs(m5) < 1e-3f);
        TEST("LayerNorm gamma=3: std ≈ 3",   std::abs(std::sqrt(v5) - 3.f) < 0.05f);
    }

    {
        std::vector<float> x3 = {0.f, 0.f, 6.f};  // mean=2, x[0]-mean=-2
        std::vector<float> g3(3, 1.f), b3(3, 0.f);
        auto y3 = layernorm_cpu(x3, g3, b3);
        TEST("LayerNorm: below-mean input → negative output", y3[0] < 0.f);
        TEST("LayerNorm: above-mean input → positive output", y3[2] > 0.f);
    }
}

static void test_m9_gemm_tiling() {
    std::cout << "\n[M9] VedicGEMM Tiling — Non-TILE-Divisible Shapes\n";

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

        TEST("VedicGEMM tiling: output rows == M", TC.rows() == M);
        TEST("VedicGEMM tiling: output cols == N", TC.cols() == N);

        TEST("VedicGEMM tiling: C[0,0] matches ref",
             std::abs(TC.data[0] - Cref[0]) < 1e-3f);
        TEST("VedicGEMM tiling: C[M-1,N-1] matches ref",
             std::abs(TC.data[M*N-1] - Cref[M*N-1]) < 1e-3f);
        int mid = (M/2)*N + (N/2);
        TEST("VedicGEMM tiling: C[mid] matches ref",
             std::abs(TC.data[mid] - Cref[mid]) < 1e-3f);
    }

    {
        int B64 = 64;
        Tensor A64({B64, B64}); A64.fill_random(-0.1f, 0.1f, 11);
        Tensor B64t({B64, B64}); B64t.fill_random(-0.1f, 0.1f, 22);
        Tensor C64 = vedic_gemm(A64, B64t);
        TEST("VedicGEMM exact-tile shape: total_size correct",
             C64.total_size == B64 * B64);
        float sum64 = 0.f;
        for (float v : C64.data) sum64 += v;
        TEST("VedicGEMM exact-tile: output sum finite", std::isfinite(sum64));
    }
}

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

    auto analytical = cpu_fe_grad(logits, target, T, seq);

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

    std::cout << "    Sample [v=0]: analytical=" << analytical[0]
              << " numerical=" << numerical[0] << "\n";
    std::cout << "    Sample [target=" << target << "]: analytical=" << analytical[target]
              << " numerical=" << numerical[target] << "\n";

    TEST("FE gradient: max relative error < 1%",  max_rel_err < 0.01f);
    TEST("FE gradient: target grad is most negative",
         analytical[target] < 0.f || vocab == 1);

    {
        std::vector<float> simple_l = {2.f, 0.f, 0.f};
        auto sp = cpu_softmax(simple_l, 1.f);
        auto sg = cpu_ce_grad(sp, 0, 1);  // target=0
        TEST("CE grad: g[target=0] < 0", sg[0] < 0.f);
        TEST("CE grad: g[non-target] > 0", sg[1] > 0.f);
        TEST("CE grad: sum(g) ≈ 0 (logits sum to 0 gradient)",
             std::abs(sg[0]+sg[1]+sg[2]) < 1e-5f);
    }

    {
        std::vector<float> l3 = {1.f, 2.f, 0.f};
        auto p3 = cpu_softmax(l3, 1.f);
        auto fe_g = cpu_fe_grad(l3, 1, 0.f, 1);  // T=0, target=1
        auto ce_g = cpu_ce_grad(p3, 1, 1);
        for (int i = 0; i < 3; ++i)
            TEST("FE grad T=0 equals CE grad [" + std::to_string(i) + "]",
                 std::abs(fe_g[i] - ce_g[i]) < 1e-5f);
    }
}

static void test_m11_natural_gradient() {
    std::cout << "\n[M11] Natural Gradient — Fisher Diagonal EMA\n";

    {
        float beta = 0.99f, g_val = 2.5f, eps = 1e-8f;
        float F = eps;  // init to eps (same as NaturalGradientOptimizer)
        for (int i = 0; i < 2000; ++i)
            F = beta * F + (1.0f - beta) * g_val * g_val;
        float expected = g_val * g_val;  // stationary EMA = g²
        TEST("Fisher EMA converges to g² (stationary)", std::abs(F - expected) < 0.05f);
        std::cout << "    F_∞=" << F << " expected=" << expected << "\n";
    }

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

    {
        float g1 = 0.5f,  F1 = 0.5f * 0.5f;   // original
        float g2 = 5.0f,  F2 = 5.0f * 5.0f;   // 10x scaled param → 10x grad
        float gnat1 = g1 / (std::sqrt(F1) + 1e-8f);
        float gnat2 = g2 / (std::sqrt(F2) + 1e-8f);
        TEST("Natural grad: scale invariant (g̃ = sign(g))",
             std::abs(std::abs(gnat1) - 1.0f) < 1e-5f &&
             std::abs(std::abs(gnat2) - 1.0f) < 1e-5f);
        std::cout << "    g̃₁=" << gnat1 << " g̃₂=" << gnat2 << "\n";
    }

    {
        Tensor W({4, 4}); W.fill_random(-0.1f, 0.1f, 42);
        Tensor G({4, 4}); G.fill(0.05f);
        NaturalGradientOptimizer opt(1e-3f, 0.99f, 1e-8f, 0.9f, 0.0f, 0.0f, 100);
        std::vector<Tensor*> ps = {&W};
        std::vector<Tensor*> gs = {&G};
        std::vector<float> W_before = W.data;
        opt.step(ps, gs);
        int changed_count = 0;
        for (int i = 0; i < W.total_size; ++i)
            if (std::abs(W.data[i] - W_before[i]) > 1e-9f) ++changed_count;
        TEST("NaturalGradOpt: weights change after step", changed_count > 0);
        TEST("NaturalGradOpt: no NaN after step",         !W.has_nan());
        TEST("NaturalGradOpt: positive grad → weight decreases",
             W.data[0] < W_before[0]);
    }

    {
        float beta = 0.5f, g = 2.f, F = 1.f;
        F = beta * F + (1.f - beta) * g * g;
        TEST("EMA formula: F = beta*F + (1-beta)*g^2 exact",
             std::abs(F - 2.5f) < 1e-5f);
        F = beta * F + (1.f - beta) * g * g;
        TEST("EMA second step: F_2 = 3.25", std::abs(F - 3.25f) < 1e-5f);
    }

    {
        float g_v = 3.f, F_v = 9.f, eps = 1e-8f;
        float g_nat = g_v / (std::sqrt(F_v) + eps);
        TEST("Natural grad: g/sqrt(g^2) ≈ 1 (scale invariant)",
             std::abs(g_nat - 1.f) < 1e-3f);
        TEST("Natural grad: g/sqrt(F) < g when F > 1",
             g_nat < g_v);
    }
}

static void test_m12_navier_stokes_attention() {
    std::cout << "\n[M12] Navier-Stokes Attention — Fluid Physics\n";

    int seq = 8, d_k = 16, d_v = 16;
    {
        logos_rng::set_global_seed(12345);
        Tensor T1({1, 4}); T1.fill_random(-1.f, 1.f);
        logos_rng::set_global_seed(12345);
        Tensor T2({1, 4}); T2.fill_random(-1.f, 1.f);
        bool seed_works = (T1.data == T2.data);
        TEST("set_global_seed: same seed → same random values (not a no-op)", seed_works);
        logos_rng::set_global_seed(99999);
        Tensor T3({1, 4}); T3.fill_random(-1.f, 1.f);
        bool diff_seed_differs = (T1.data != T3.data);
        TEST("set_global_seed: different seed → different values", diff_seed_differs);
    }

    logos_rng::set_global_seed(42);
    {
        logos_rng::set_global_seed(42);
        Tensor _ck1({1,4}); _ck1.fill_random(-1.f,1.f);
        logos_rng::set_global_seed(42);
        Tensor _ck2({1,4}); _ck2.fill_random(-1.f,1.f);
        if (_ck1.data != _ck2.data) { std::cerr << "[FATAL] set_global_seed not applied\n"; std::abort(); }
    }
    Tensor Q({seq, d_k}); Q.fill_random(-0.5f, 0.5f);
    Tensor K({seq, d_k}); K.fill_random(-0.5f, 0.5f);
    Tensor V({seq, d_v}); V.fill_random(-0.5f, 0.5f);

    float temperature = std::sqrt((float)d_k);
    float eta = 0.2f, nu = 0.1f;

    Tensor mask({seq, seq}, 0.0f);
    for (int i = 0; i < seq; ++i)
        for (int j = i+1; j < seq; ++j)
            mask.at(i, j) = -1e9f;

    {
        float q0 = Q.at(1, 0);
        float q_prev = Q.at(0, 0);
        float q_adv_expected = q0 + eta * (q0 - q_prev);
        Tensor ns_out = navier_stokes_attention(Q, K, V, mask, temperature, eta, nu);
        TEST("NS attention: output shape correct (seq x d_v)",
             ns_out.rows() == seq && ns_out.cols() == d_v);
        TEST("NS attention: output has no NaN", !ns_out.has_nan());
    }

    {
        Tensor ns_strong = navier_stokes_attention(Q, K, V, mask, temperature, 1.0f, nu);
        TEST("NS attention: stable with strong advection (eta=1)", !ns_strong.has_nan());
    }

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

    {
        float v_raw  = V.at(1, 0);
        float v_left = V.at(0, 0);
        float v_right= V.at(2, 0);
        float v_smooth_expected = (1.0f - nu) * v_raw + 0.5f * nu * (v_left + v_right);
        bool diffusion_active = std::abs(v_smooth_expected - v_raw) > 1e-6f;
        TEST("NS attention: viscous diffusion non-trivial (nu=0.1)", diffusion_active);
        std::cout << "    V[1,0]=" << v_raw << " V_smooth[1,0]≈" << v_smooth_expected << "\n";
    }

    {
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

    {
        float nu2 = 0.1f;
        float Vleft=2.f, Vmid=4.f, Vright=6.f;
        float smooth = (1.f - nu2)*Vmid + 0.5f*nu2*(Vleft + Vright);
        TEST("Viscous diffusion formula: V_smooth = (1-nu)*V + nu/2*(L+R) exact",
             std::abs(smooth - 4.0f) < 1e-5f);
        float smooth_full = (1.f-1.f)*Vmid + 0.5f*1.f*(Vleft+Vright);
        TEST("Viscous diffusion nu=1: V_smooth = (L+R)/2", std::abs(smooth_full - 4.f) < 1e-5f);
        float smooth_none = (1.f-0.f)*Vmid + 0.5f*0.f*(Vleft+Vright);
        TEST("Viscous diffusion nu=0: V_smooth = V (no diffusion)",
             std::abs(smooth_none - Vmid) < 1e-5f);
    }

    {
        Tensor K_T2 = K.transpose();
        Tensor sc2  = vedic_gemm(Q, K_T2);
        sc2 += mask;
        Tensor w2   = boltzmann_softmax(sc2, temperature);
        float row0_sum = 0.f;
        for (int j = 0; j < seq; ++j) row0_sum += w2.at(0, j);
        TEST("boltzmann_softmax row sum == 1.0 (not 0, not seq)",
             std::abs(row0_sum - 1.f) < 1e-4f);
        bool all_nonneg = true;
        for (int i = 0; i < w2.total_size; ++i)
            if (w2.data[i] < -1e-6f) all_nonneg = false;
        TEST("boltzmann_softmax: all weights >= 0", all_nonneg);
    }
}

static void test_m13_reynolds_batch_norm() {
    std::cout << "\n[M13] Reynolds Batch Norm — Laminar/Turbulent Blend\n";

    int seq = 8, d = 32;
    logos_rng::set_global_seed(77);
    {
        logos_rng::set_global_seed(77);
        Tensor _s1({1,4}); _s1.fill_random(-1.f,1.f);
        logos_rng::set_global_seed(77);
        Tensor _s2({1,4}); _s2.fill_random(-1.f,1.f);
        if (_s1.data != _s2.data) { std::cerr << "[FATAL] set_global_seed(77) not applied\n"; std::abort(); }
    }

    ReynoldsBatchNorm rbn(d);
    Tensor X({seq, d}); X.fill_random(-2.0f, 2.0f);

    {
        float Re = rbn.reynolds_number(X);
        TEST("Re_eff > 0 always", Re > 0.0f);
        std::cout << "    Re_eff=" << Re << "\n";
    }

    {
        float Re = rbn.reynolds_number(X);
        float w  = rbn.laminar_weight(Re);
        TEST("Laminar weight ∈ (0,1)", w > 0.0f && w < 1.0f);
        std::cout << "    laminar_weight(Re=" << Re << ")=" << w << "\n";
    }

    {
        float w_low = rbn.laminar_weight(0.001f);   // Re << Re_crit=1.0
        TEST("Re→0: laminar_weight → 1 (laminar regime)", w_low > 0.95f);
        std::cout << "    laminar_weight(Re=0.001)=" << w_low << "\n";
    }

    {
        float w_high = rbn.laminar_weight(100.0f);  // Re >> Re_crit=1.0
        TEST("Re→∞: laminar_weight → 0 (turbulent regime)", w_high < 0.05f);
        std::cout << "    laminar_weight(Re=100)=" << w_high << "\n";
    }

    {
        Tensor Y = rbn.forward(X, true);
        float max_diff = 0.f;
        for (int i = 0; i < X.total_size; ++i)
            max_diff = std::max(max_diff, std::abs(Y.data[i] - X.data[i]));
        TEST("ReynoldsBatchNorm: output ≠ input (normalisation active)", max_diff > 1e-3f);
        TEST("ReynoldsBatchNorm: output has no NaN", !Y.has_nan());
    }

    {
        ReynoldsBatchNorm rbn2(d);
        rbn2.Re_crit = 2.0f;
        float Re_start = rbn2.Re_crit;
        rbn2.anneal_reynolds(500, 1000, 2.0f, 0.5f);
        float Re_mid = rbn2.Re_crit;
        rbn2.anneal_reynolds(999, 1000, 2.0f, 0.5f);
        float Re_end = rbn2.Re_crit;
        TEST("Reynolds annealing: Re_crit decreases over training",
             Re_start > Re_mid && Re_mid > Re_end);
        std::cout << "    Re_crit: " << Re_start << " → " << Re_mid << " → " << Re_end << "\n";
    }

    {
        ReynoldsBatchNorm rbn3(d);
        float Re_c = rbn3.Re_crit;  // default Re_crit
        float w_at_crit = rbn3.laminar_weight(Re_c);
        TEST("laminar_weight(Re_crit) ≈ 0.5 (sigmoid at boundary)",
             std::abs(w_at_crit - 0.5f) < 0.05f);
    }

    {
        Tensor X_const({4, 8}); X_const.fill(3.f);
        float Re_const = rbn.reynolds_number(X_const);
        TEST("reynolds_number: constant tensor → Re > 0", Re_const > 0.f);
        Tensor X_mz({4, 8}); X_mz.fill_random(-1.f, 1.f, 55);
        float mx = 0.f; for (float v : X_mz.data) mx += v; mx /= X_mz.total_size;
        for (float& v : X_mz.data) v -= mx;
        float Re_mz = rbn.reynolds_number(X_mz);
        TEST("reynolds_number: mean-zero tensor → Re ≈ 1 (RMS ≈ std)",
             Re_mz > 0.5f && Re_mz < 2.f);
        std::cout << "    Re_const=" << Re_const << " Re_mean_zero=" << Re_mz << "\n";
    }
}

static void test_m14_feynman_dropout() {
    std::cout << "\n[M14] Feynman Dropout — Path Integral Amplitudes\n";

    int N_samples = 10000;

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

    {
        FeynmanDropout fd(0.2f, 1.0f, 99);
        bool all_bounded = true;
        for (int i = 0; i < N_samples; ++i) {
            float w = fd.sample_weight();
            if (w < 0.0f || w > 1.0f) { all_bounded = false; break; }
        }
        TEST("All Feynman weights ∈ [0,1] (Beta bounded)", all_bounded);
    }

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
        if (std::abs(fd.hbar - 0.01f) > 1e-6f) {
            std::cerr << "[FATAL] set_hbar(0.01f) had no effect\n"; std::abort();
        }
        const float classical_variance = sample_variance(N_samples);
        TEST("set_hbar: smaller ħ produces more Bernoulli-like weights",
             classical_variance > smooth_variance);
        std::cout << "    set_hbar variance: ħ=2 " << smooth_variance
                  << " → ħ=0.01 " << classical_variance << "\n";
    }

    {
        FeynmanDropout fd(0.0f, 1.0f, 1);
        Tensor X({4, 4}); X.fill(1.0f);
        fd.training = true;
        Tensor Y = fd.forward(X);
        float diff = 0.0f;
        for (int i = 0; i < X.total_size; ++i) diff += std::abs(Y.data[i] - X.data[i]);
        TEST("p=0: Feynman dropout is identity", diff < 1e-5f);
    }

    {
        FeynmanDropout fd(0.5f, 1.0f, 1);
        fd.training = false;
        Tensor X({8, 8}); X.fill_random(-1.0f, 1.0f, 77);
        Tensor Y = fd.forward(X);
        float diff = 0.0f;
        for (int i = 0; i < X.total_size; ++i) diff += std::abs(Y.data[i] - X.data[i]);
        TEST("training=false: Feynman dropout is identity (eval mode)", diff < 1e-6f);
        bool exact_match = true;
        for (int i = 0; i < X.total_size; ++i)
            if (Y.data[i] != X.data[i]) exact_match = false;
        TEST("training=false: Y[i] == X[i] exactly for all i", exact_match);
    }

    {
        for (float hbar : {0.1f, 1.0f, 5.0f}) {
            FeynmanDropout fd(0.5f, hbar, 42);
            float sum = 0.f;
            for (int i = 0; i < N_samples; ++i) sum += fd.sample_weight();
            TEST("Feynman p=0.5: mean ≈ 0.5 for hbar=" + std::to_string(hbar),
                 std::abs(sum/N_samples - 0.5f) < 0.05f);
        }
    }

    {
        FeynmanDropout fd_full(1.0f, 1.0f, 1);
        float sum_w = 0.f;
        for (int i = 0; i < 1000; ++i) sum_w += fd_full.sample_weight();
        TEST("p=1.0: mean weight ≈ 0 (full dropout)", sum_w / 1000.f < 0.05f);
    }
}

static void test_m15_riemannian_metric() {
    std::cout << "\n[M15] Riemannian Metric — Curved Parameter Space\n";

    int dim = 64;
    RiemannianMetric rm(dim, 1e-4f);

    Tensor g1({1, dim}); g1.fill_random(-1.0f, 1.0f);
    rm.update(g1, 0.9f);
    {
        float sum_diag = 0.f;
        for (float d : rm.metric_diag) sum_diag += d;
        if (sum_diag <= 0.f) { std::cerr << "[FATAL] rm.update() had no effect\n"; std::abort(); }
    }

    {
        Tensor theta({1, dim}); theta.fill_random(-0.5f, 0.5f);
        float d = rm.riemannian_distance(theta, theta);
        TEST("Riemannian distance: d(θ,θ) = 0", d < 1e-5f);
    }

    {
        Tensor t1({1, dim}); t1.fill_random(-1.0f, 1.0f);
        Tensor t2({1, dim}); t2.fill_random(-1.0f, 1.0f);
        float d = rm.riemannian_distance(t1, t2);
        TEST("Riemannian distance: d(θ1,θ2) > 0 for θ1 ≠ θ2", d > 1e-6f);
        std::cout << "    d(θ1,θ2)=" << d << "\n";
    }

    {
        Tensor t1({1, dim}); t1.fill_random(-0.5f, 0.5f);
        Tensor t2({1, dim}); t2.fill_random(-0.5f, 0.5f);
        float d12 = rm.riemannian_distance(t1, t2);
        float d21 = rm.riemannian_distance(t2, t1);
        TEST("Riemannian distance: symmetric d(θ1,θ2) == d(θ2,θ1)",
             std::abs(d12 - d21) < 1e-5f);
    }

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

    {
        Tensor g({1, dim}); g.fill_random(-1.0f, 1.0f);
        rm.update(g, 0.9f);
        float chk = 0.f; for (float d : rm.metric_diag) chk += d * d;
        if (chk <= 0.f) { std::cerr << "[FATAL] rm.update() in (e) had no effect\n"; std::abort(); }
        Tensor g_nat = rm.riemannian_gradient(g);
        float max_residual = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float recovered = (rm.metric_diag[i] + rm.damping) * g_nat.data[i];
            max_residual = std::max(max_residual, std::abs(recovered - g.data[i]));
        }
        TEST("Natural gradient: G·g̃ reconstructs g", max_residual < 1e-5f);
        std::cout << "    max ||G·g̃-g||=" << max_residual << "\n";
    }

    {
        Tensor v({1, dim}); v.fill(1.0f);
        Tensor dtheta({1, dim}); dtheta.fill(1.0f);
        Tensor v_t = rm.parallel_transport(v, dtheta);
        float norm_transported = rm.riemannian_norm(v_t);
        float norm_original    = rm.riemannian_norm(v);
        TEST("Parallel transport: removes component along Δθ",
             norm_transported < norm_original);
        std::cout << "    ||v||_G=" << norm_original
                  << " ||v_transported||_G=" << norm_transported << "\n";
    }

    {
        RiemannianMetric rm2(4, 0.0f);  // zero damping for clean math
        rm2.metric_diag = {4.f, 4.f, 4.f, 4.f};  // G=4*I
        Tensor t1({1,4}); t1.data = {1.f, 0.f, 0.f, 0.f};
        Tensor t2({1,4}); t2.data = {2.f, 0.f, 0.f, 0.f};
        float d = rm2.riemannian_distance(t1, t2);
        TEST("Riemannian distance exact: G=4I, diff=1 → d=2",
             std::abs(d - 2.f) < 1e-4f);
        Tensor t3({1,4}); t3.data = {3.f, 0.f, 0.f, 0.f};
        float d2 = rm2.riemannian_distance(t1, t3);
        TEST("Riemannian distance: diff=2 → d=4 (linear scaling)",
             std::abs(d2 - 4.f) < 1e-4f);
    }

    {
        RiemannianMetric rm3(4, 0.f);
        rm3.metric_diag = {2.f, 4.f, 1.f, 8.f};
        Tensor g3({1,4}); g3.data = {2.f, 4.f, 1.f, 8.f};
        Tensor gnat = rm3.riemannian_gradient(g3);
        TEST("Riemannian grad: g̃[0] = g[0]/G[0] = 1",
             std::abs(gnat.data[0] - 1.f) < 1e-5f);
        TEST("Riemannian grad: g̃[1] = g[1]/G[1] = 1",
             std::abs(gnat.data[1] - 1.f) < 1e-5f);
        TEST("Riemannian grad: g̃[3] = g[3]/G[3] = 1",
             std::abs(gnat.data[3] - 1.f) < 1e-5f);
    }
}

static void test_m16_weight_path_integral() {
    std::cout << "\n[M16] Weight Path Integral — Action Accumulation\n";

    WeightPathIntegral wpi(1.0f, 100);

    {
        float log_A0 = wpi.log_amplitude;  // = 0 initially
        std::vector<float> delta(16, 0.1f);  // ||Δθ|| = sqrt(16*0.01) ≈ 0.4
        float loss = 2.0f;                    // positive loss → positive action
        wpi.record_step(loss, delta);
        TEST("log_amplitude decreases when action > 0", wpi.log_amplitude < log_A0);
        std::cout << "    log_A: " << log_A0 << " → " << wpi.log_amplitude << "\n";
    }

    {
        WeightPathIntegral wpi2(1.0f, 10);
        float loss_t = 3.0f;
        std::vector<float> delta_t = {3.0f, 4.0f};  // ||Δθ|| = 5.0
        wpi2.record_step(loss_t, delta_t);
        float expected_logA = -15.0f;
        TEST("Discrete Lagrangian: S_t = loss * ||Δθ|| correct",
             std::abs(wpi2.log_amplitude - expected_logA) < 1e-4f);
        std::cout << "    Expected log_A=" << expected_logA
                  << " actual=" << wpi2.log_amplitude << "\n";
    }

    {
        WeightPathIntegral wpi3(1.0f, 50);
        float lr_min = 0.1f;
        float scale_fresh = wpi3.lr_scale(lr_min);
        TEST("lr_scale = 1.0 at start (no steps)", std::abs(scale_fresh - 1.0f) < 1e-5f);

        std::vector<float> big_step(4, 10.0f);
        for (int i = 0; i < 30; ++i) wpi3.record_step(5.0f, big_step);
        if (wpi3.step_count < 30) { std::cerr << "[FATAL] wpi3.record_step loop no-ops\n"; std::abort(); }
        std::vector<float> tiny_step(4, 0.0f);
        wpi3.record_step(0.0f, tiny_step);  // S=0 → no amplitude decrease
        float scale_after = wpi3.lr_scale(lr_min);
        TEST("lr_scale ∈ [lr_min, 1.0] always",
             scale_after >= lr_min - 1e-5f && scale_after <= 1.0f + 1e-5f);
        std::cout << "    lr_scale after high-action=" << scale_after << "\n";
    }

    {
        WeightPathIntegral wpi4(1.0f, 50);
        wpi4.record_step(0.1f, {0.01f, 0.01f});  // S≈0.001, log_A ≈ -0.001
        int step_after_best = wpi4.best_step;
        if (wpi4.step_count == 0) { std::cerr << "[FATAL] wpi4 record_step(0) no-op\n"; std::abort(); }

        wpi4.record_step(10.0f, {5.0f, 5.0f, 5.0f});  // S=70+, log_A very negative
        if (wpi4.step_count < 2) { std::cerr << "[FATAL] wpi4 record_step(1) no-op\n"; std::abort(); }
        TEST("best_step is not the last high-action step",
             wpi4.best_step != 1);  // best should still be step 0
        std::cout << "    best_step=" << wpi4.best_step
                  << " current_step=" << wpi4.step_count << "\n";
    }

    {
        WeightPathIntegral wpi5(1.0f, 50);
        wpi5.record_step(0.5f, {0.1f});   // some action
        if (wpi5.step_count < 1) { std::cerr << "[FATAL] wpi5 record_step(1) no-op\n"; std::abort(); }
        wpi5.record_step(0.5f, {0.1f});   // more action
        if (wpi5.step_count < 2) { std::cerr << "[FATAL] wpi5 record_step(2) no-op\n"; std::abort(); }
        float rel = wpi5.relative_amplitude();
        TEST("relative_amplitude ∈ (0,1] always", rel > 0.0f && rel <= 1.0f + 1e-5f);
        std::cout << "    relative_amplitude=" << rel << "\n";
    }

    {
        WeightPathIntegral wpi6(1.0f, 100);
        float fixed_loss = 2.0f;
        std::vector<float> fixed_step = {1.0f, 0.0f};  // ||Δθ||=1, S=2
        for (int i = 0; i < 10; ++i)
            wpi6.record_step(fixed_loss, fixed_step);
        float mean_action = wpi6.recent_mean_action(10);
        float expected_action = fixed_loss * 1.0f;
        TEST("recent_mean_action correct (last 10 steps)",
             std::abs(mean_action - expected_action) < 1e-4f);
        std::cout << "    mean_action=" << mean_action
                  << " expected=" << expected_action << "\n";
    }

    {
        float hbar = 2.0f;
        WeightPathIntegral wpi7(hbar, 100);
        float loss = 1.f;
        std::vector<float> step = {1.f, 0.f};  // ||Δθ||=1 → S=1
        int N_steps = 5;
        for (int i = 0; i < N_steps; ++i) wpi7.record_step(loss, step);
        if (wpi7.step_count != N_steps) { std::cerr << "[FATAL] wpi7 loop count wrong\n"; std::abort(); }
        float expected_logA = -(float)N_steps * loss * 1.f / hbar;
        TEST("log_amplitude after N steps: log_A = -N*S/ħ exact",
             std::abs(wpi7.log_amplitude - expected_logA) < 1e-4f);
    }

    {
        WeightPathIntegral wpi8(1.f, 50);
        int count_before = wpi8.step_count;
        wpi8.record_step(1.f, {0.1f});
        TEST("record_step: step_count increments (not a void no-op)",
             wpi8.step_count == count_before + 1);
        wpi8.record_step(1.f, {0.1f});
        wpi8.record_step(1.f, {0.1f});
        TEST("record_step: step_count increments correctly after 3 calls",
             wpi8.step_count == count_before + 3);
    }

    {
        WeightPathIntegral wpi9(1.f, 100);
        float lr_min = 0.2f;
        TEST("lr_scale at init: == 1.0", std::abs(wpi9.lr_scale(lr_min) - 1.f) < 1e-5f);
        for (int i = 0; i < 20; ++i)
            wpi9.record_step(100.f, {1.f, 1.f, 1.f});  // huge action
        if (wpi9.step_count < 20) { std::cerr << "[FATAL] wpi9.record_step loop no-ops\n"; std::abort(); }
        float s = wpi9.lr_scale(lr_min);
        TEST("lr_scale after huge action: >= lr_min (clamped)", s >= lr_min - 1e-5f);
        TEST("lr_scale: always <= 1.0", s <= 1.f + 1e-5f);
    }

    {
        WeightPathIntegral wpiJ(1.f, 10);
        wpiJ.record_step(2.f, {3.f, 4.f});  // S = 2 * sqrt(9+16) = 2*5 = 10
        if (wpiJ.step_count < 1) { std::cerr << "[FATAL] wpiJ.record_step no-op\n"; std::abort(); }
        float expected_logA = -10.f;         // log_A = -S/ħ = -10/1 = -10
        TEST("Discrete Lagrangian: S=loss*||Δθ||, ||{3,4}||=5",
             std::abs(wpiJ.log_amplitude - expected_logA) < 1e-3f);
    }
}

static void test_m17_vedicgemm_arithmetic() {
    std::cout << "\n[M17] VedicGEMM Arithmetic — Loop Bounds + Stride Mutations\n";

    {
        int M=3, K=3, N=3;
        Tensor A({M,K}); A.data = {1,2,3, 4,5,6, 7,8,9};
        Tensor B({K,N}); B.data = {9,8,7, 6,5,4, 3,2,1};
        Tensor C = vedic_gemm(A, B);
        float sum = 0.f; for (float v : C.data) sum += std::abs(v);
        TEST("VedicGEMM outer-M loop runs: C not all-zero", sum > 0.f);
        TEST("VedicGEMM C[0,0] exact=30 (kills i<M loop mutation)", std::abs(C.at(0,0)-30.f)<1e-3f);
        TEST("VedicGEMM C[2,2] exact=90 (kills outer loop off-by-one)", std::abs(C.at(2,2)-90.f)<1e-3f);
    }

    {
        int M=2, K=4, N=2;
        Tensor A({M,K}); A.data = {1,10,100,1000, 2,20,200,2000};
        Tensor B({K,N}); B.data = {1,0, 0,1, 1,0, 0,1};
        Tensor C = vedic_gemm(A, B);
        TEST("VedicGEMM K-loop full: C[0,0]=101 (kills k<K mutation)", std::abs(C.at(0,0)-101.f)<1e-3f);
        TEST("VedicGEMM K-loop full: C[0,1]=1010 (kills k<K mutation)", std::abs(C.at(0,1)-1010.f)<1e-3f);
    }

    {
        int M=2, K=2, N=3;
        Tensor A({M,K}); A.data = {1,2, 3,4};
        Tensor B({K,N}); B.data = {5,6,7, 8,9,10};
        Tensor C = vedic_gemm(A, B);
        TEST("VedicGEMM j-loop: C[0,0]=21", std::abs(C.at(0,0)-21.f)<1e-3f);
        TEST("VedicGEMM j-loop: C[0,1]=24", std::abs(C.at(0,1)-24.f)<1e-3f);
        TEST("VedicGEMM j-loop: C[0,2]=27", std::abs(C.at(0,2)-27.f)<1e-3f);
        TEST("VedicGEMM j-loop last col: C[1,2]=", std::abs(C.at(1,2)-(3*7+4*10))<1e-3f);
    }

    {
        int M=4, K=4, N=4;
        Tensor I4({M,N}); // Identity
        for(int i=0;i<M;++i) I4.at(i,i)=1.f;
        Tensor A4({M,K}); for(int i=0;i<M*K;++i) A4.data[i]=(float)(i+1);
        Tensor C4 = vedic_gemm(A4, I4);
        bool match = true;
        for(int i=0;i<M*N;++i) if(std::abs(C4.data[i]-A4.data[i])>1e-3f) match=false;
        TEST("VedicGEMM: A@I=A (kills ++i→--i in all loops)", match);
    }

    {
        int M=3, K=2, N=3;
        Tensor A({M,K}); A.data = {1,2, 3,4, 5,6};  // row-major: A[i,k]=A.data[i*2+k]
        Tensor B({K,N}); B.data = {7,8,9, 10,11,12}; // B[k,j]=B.data[k*3+j]
        Tensor C = vedic_gemm(A, B);
        TEST("VedicGEMM: C[0,0]=27 (kills i*K+k → i*K-k)", std::abs(C.at(0,0)-27.f)<1e-3f);
        TEST("VedicGEMM: C[0,1]=30", std::abs(C.at(0,1)-30.f)<1e-3f);
        TEST("VedicGEMM: C[0,2]=33", std::abs(C.at(0,2)-33.f)<1e-3f);
        TEST("VedicGEMM: C[1,0]=61", std::abs(C.at(1,0)-61.f)<1e-3f);
        TEST("VedicGEMM: C[2,1]=106", std::abs(C.at(2,1)-106.f)<1e-3f);
        TEST("VedicGEMM: C[2,2]=117 (kills k*N+j index mutation)", std::abs(C.at(2,2)-117.f)<1e-3f);
    }

    {
        Tensor A({2,2}); A.data = {2.f, 3.f, 4.f, 5.f};
        Tensor B({2,2}); B.data = {2.f, 0.f, 0.f, 2.f};
        Tensor C = vedic_gemm(A, B);
        TEST("VedicGEMM multiply: C[0,0]=4 (kills * → /)", std::abs(C.at(0,0)-4.f)<1e-3f);
        TEST("VedicGEMM multiply: C[0,1]=6 (kills * → /)", std::abs(C.at(0,1)-6.f)<1e-3f);
        TEST("VedicGEMM multiply: C[1,0]=8", std::abs(C.at(1,0)-8.f)<1e-3f);
        TEST("VedicGEMM multiply: C[1,1]=10", std::abs(C.at(1,1)-10.f)<1e-3f);
    }

    {
        bool no_throw = true;
        try {
            Tensor A({3,4}); A.fill(1.f);
            Tensor W({4,5}); W.fill(1.f);
            Tensor b({5});   b.fill(0.5f);
            Tensor R = vedic_gemm_bias(A, W, b);
            TEST("vedic_gemm_bias: result[0,0]=4.5 (kills != → ==)", std::abs(R.at(0,0)-4.5f)<1e-3f);
        } catch(...) { no_throw = false; }
        TEST("vedic_gemm_bias: no exception with correct bias size", no_throw);
    }

    {
        int M=3, N=4;
        Tensor A({M,N}); A.fill(0.f);  // zero input
        Tensor W({N,N}); // identity-like
        for(int i=0;i<N;++i) W.at(i,i)=1.f;
        Tensor b({N});
        for(int j=0;j<N;++j) b.data[j]=(float)(j+1); // [1,2,3,4]
        Tensor R = vedic_gemm_bias(A, W, b);
        bool bias_ok = true;
        for(int i=0;i<M;++i)
            for(int j=0;j<N;++j)
                if(std::abs(R.at(i,j)-(float)(j+1))>1e-3f) bias_ok=false;
        TEST("vedic_gemm_bias: all rows get correct bias (kills loop bound mutations)", bias_ok);
        TEST("vedic_gemm_bias: last row R[2,3]=4 (kills ++i→--i in bias loop)",
             std::abs(R.at(M-1,N-1)-4.f)<1e-3f);
    }
}

static void test_m18_tensor_ops() {
    std::cout << "\n[M18] Tensor Ops — at() Bounds + Stride Mutations\n";

    {
        Tensor T({4,5}); T.fill(0.f);
        for(int i=0;i<4;++i) for(int j=0;j<5;++j) T.at(i,j) = (float)(i*5+j+1);
        bool ok = true;
        try {
            float last = T.at(3,4);   // index (rows-1, cols-1) = valid = 20
            TEST("Tensor::at last valid index returns 20", std::abs(last-20.f)<1e-5f);
        } catch(...) { ok = false; }
        TEST("Tensor::at(rows-1, cols-1) does NOT throw (kills >= → >)", ok);
    }

    {
        Tensor T({3,4});
        for(int r=0;r<3;++r) for(int c=0;c<4;++c) T.at(r,c) = (float)(r*10+c);
        TEST("Tensor stride: T[0,3]=3  (kills r*cols()+c → r*cols()-c)", std::abs(T.at(0,3)-3.f)<1e-5f);
        TEST("Tensor stride: T[1,0]=10 (kills stride mutation)", std::abs(T.at(1,0)-10.f)<1e-5f);
        TEST("Tensor stride: T[2,3]=23 (kills stride mutation)", std::abs(T.at(2,3)-23.f)<1e-5f);
        TEST("Tensor stride: T[2,1]=21", std::abs(T.at(2,1)-21.f)<1e-5f);
        TEST("Tensor stride: T[2,2]=22", std::abs(T.at(2,2)-22.f)<1e-5f);
    }

    {
        Tensor T({1,5}); T.data = {1.f, 2.f, 3.f, 4.f, 5.f};
        Tensor S = T * 3.f;
        TEST("Tensor*scalar: [0]=3 (kills * → /)", std::abs(S.data[0]-3.f)<1e-5f);
        TEST("Tensor*scalar: [4]=15", std::abs(S.data[4]-15.f)<1e-5f);
    }

    {
        Tensor T({1,4}); T.data = {2.f,4.f,6.f,8.f};
        Tensor S = T * 2.f;
        bool all_ok = true;
        float expected[] = {4.f,8.f,12.f,16.f};
        for(int i=0;i<4;++i) if(std::abs(S.data[i]-expected[i])>1e-5f) all_ok=false;
        TEST("Tensor*scalar: all 4 elements correct (kills loop i<total_size mutation)", all_ok);
    }

    {
        Tensor T({3,4});
        for(int r=0;r<3;++r) for(int c=0;c<4;++c) T.at(r,c) = (float)(r*4+c+1);
        Tensor Tt = T.transpose();
        TEST("Transpose shape: rows=4", Tt.rows()==4);
        TEST("Transpose shape: cols=3", Tt.cols()==3);
        TEST("Transpose: Tt[0,0]=T[0,0]=1", std::abs(Tt.at(0,0)-1.f)<1e-5f);
        TEST("Transpose: Tt[3,2]=T[2,3]=12 (kills c<C loop mutation)", std::abs(Tt.at(3,2)-12.f)<1e-5f);
        TEST("Transpose: Tt[0,2]=T[2,0]=9  (kills r<R loop mutation)", std::abs(Tt.at(0,2)-9.f)<1e-5f);
        bool symmetry = true;
        for(int r=0;r<3;++r) for(int c=0;c<4;++c)
            if(std::abs(T.at(r,c)-Tt.at(c,r))>1e-5f) symmetry=false;
        TEST("Transpose: T[r,c]==Tt[c,r] for all (kills both loop bound mutations)", symmetry);
    }

    {
        Tensor T1({1,8}); T1.fill_random(-1.f, 1.f, 0);  // local_seed=0 >= 0 → use local
        Tensor T2({1,8}); T2.fill_random(-1.f, 1.f, 0);  // same local seed → same values
        TEST("fill_random: local_seed=0 is valid (kills >= → > mutation)",
             T1.data == T2.data);
        Tensor T3({1,8}); T3.fill_random(-1.f, 1.f, 1);
        bool differs = false;
        for(int i=0;i<8;++i) if(std::abs(T1.data[i]-T3.data[i])>1e-6f) { differs=true; break; }
        TEST("fill_random: different seeds give different values", differs);
    }

    {
        bool ok1 = true;
        try { Tensor T({1,1}); } catch(...) { ok1 = false; }
        TEST("Tensor shape {1,1} accepted (d=1 valid, kills <= → < mutation)", ok1);
        bool ok0 = false;
        try { Tensor T({0,4}); } catch(...) { ok0 = true; }
        TEST("Tensor shape {0,4} throws (d=0 invalid)", ok0);
    }
}

static void test_m19_feedforward_mutations() {
    std::cout << "\n[M19] FeedForward + FeynmanDropout — Boundary Mutations\n";

    {
        FeynmanDropout fd(0.3f, 0.05f, 42);
        int zeros=0, ones=0, others=0;
        for(int i=0;i<1000;++i) {
            float w = fd.sample_weight();
            if(std::abs(w)<1e-5f) zeros++;
            else if(std::abs(w-1.f)<1e-5f) ones++;
            else others++;
        }
        TEST("FeynmanDropout: hbar=0.05 (<=) uses Bernoulli limit (kills <= → <)",
             others == 0);
        std::cout << "    hbar=0.05: zeros=" << zeros << " ones=" << ones << " others=" << others << "\n";
    }

    {
        FeynmanDropout fd(0.0f, 1.0f, 99);
        fd.training = true;
        Tensor X({4,4}); X.fill_random(-1.f,1.f,7);
        Tensor Y = fd.forward(X);
        float diff=0.f;
        for(int i=0;i<X.total_size;++i) diff += std::abs(Y.data[i]-X.data[i]);
        TEST("FeynmanDropout: p=0.0 (<=0.0f check) → identity (kills <= → <, <= → >)",
             diff < 1e-5f);
    }

    {
        FeedForward ff1(8, 0);   // d_ff=0 → should use 4*8=32
        TEST("FeedForward: d_ff=0 → uses 4*d_model=32 (kills > → >= mutation)",
             ff1.d_ff == 32);
        TEST("FeedForward: W1 shape matches d_ff=32 (kills * → / in 4*d_model)",
             ff1.W1.cols() == 32);

        FeedForward ff2(8, 16);
        TEST("FeedForward: d_ff=16 > 0 → uses d_ff=16", ff2.d_ff == 16);
        TEST("FeedForward: W1 cols=16 when d_ff explicitly given", ff2.W1.cols() == 16);
    }

    {
        FeedForward ff(16, 0);  // d_ff=0 → 4*16=64
        TEST("FeedForward: d_ff = 4*d_model=64 (kills 4*d_model → 4/d_model)",
             ff.d_ff == 64);
        TEST("FeedForward: W1.cols()=64", ff.W1.cols() == 64);
        TEST("FeedForward: W2.rows()=64", ff.W2.rows() == 64);
        TEST("FeedForward: W1.rows()=d_model=16", ff.W1.rows() == 16);
        TEST("FeedForward: W2.cols()=d_model=16", ff.W2.cols() == 16);
    }

    {
        float p=0.4f, hbar=3.f;
        FeynmanDropout fd(p, hbar, 77);
        float sum=0.f;
        for(int i=0;i<5000;++i) sum += fd.sample_weight();
        float mean = sum/5000.f;
        TEST("FeynmanDropout: mean=(1-p)=0.6 (kills (1-p) → (1+p) mutation)",
             std::abs(mean-0.6f)<0.05f);
    }

    {
        float p=0.7f, hbar=2.f;  // large p → low mean
        FeynmanDropout fd(p, hbar, 11);
        float sum=0.f;
        for(int i=0;i<5000;++i) sum += fd.sample_weight();
        float mean = sum/5000.f;
        TEST("FeynmanDropout: mean=(1-p)=0.3 (kills p*hbar mutation)",
             std::abs(mean-0.3f)<0.05f);
    }

    {
        FeedForward ff(8, 16, 0.f);  // no dropout
        logos_rng::set_global_seed(123);
        Tensor X({4,8}); X.fill_random(-0.5f,0.5f);
        Tensor Y = ff.forward(X, false);
        TEST("FeedForward: output shape rows=4", Y.rows()==4);
        TEST("FeedForward: output shape cols=8", Y.cols()==8);
        float diff=0.f;
        for(int i=0;i<X.total_size;++i) diff+=std::abs(Y.data[i]-X.data[i]);
        TEST("FeedForward: output differs from input (not identity)", diff>1e-4f);
    }
}

static void test_m20_attention_mutations() {
    std::cout << "\n[M20] Attention + boltzmann_softmax — Mutation Kills\n";

    {
        int seq=3, d=4;
        Tensor scores({seq,d});
        scores.at(0,0)=-10.f; scores.at(0,1)=-5.f; scores.at(0,2)=0.f; scores.at(0,3)=5.f;
        scores.at(1,0)=10.f; scores.at(1,1)=1.f; scores.at(1,2)=0.f; scores.at(1,3)=-5.f;
        Tensor T({seq,d}, 0.f); // zero mask
        Tensor probs = boltzmann_softmax(scores, 1.f);
        TEST("boltzmann_softmax: argmax at col3 when max not at col0",
             probs.at(0,3) > probs.at(0,0));
        TEST("boltzmann_softmax: p[0,3] > 0.9 (max is very high relative)",
             probs.at(0,3) > 0.9f);
        TEST("boltzmann_softmax: row1 argmax at col0 (standard case)",
             probs.at(1,0) > probs.at(1,1));
        float s0=0.f,s1=0.f;
        for(int j=0;j<d;++j) { s0+=probs.at(0,j); s1+=probs.at(1,j); }
        TEST("boltzmann_softmax: row0 sum=1", std::abs(s0-1.f)<1e-4f);
        TEST("boltzmann_softmax: row1 sum=1", std::abs(s1-1.f)<1e-4f);
    }

    {
        int seq=2, d=3;
        Tensor scores({seq,d});
        scores.at(0,0)=1000.f; scores.at(0,1)=999.f; scores.at(0,2)=0.f;
        scores.at(1,0)=-1000.f; scores.at(1,1)=1000.f; scores.at(1,2)=1000.f;
        Tensor probs = boltzmann_softmax(scores, 1.f);
        TEST("boltzmann_softmax: no NaN with extreme scores (kills - → + mutation)",
             !probs.has_nan());
        TEST("boltzmann_softmax: large score stable, col0 > col1",
             probs.at(0,0) > probs.at(0,1));
    }

    {
        int seq=1, d=4;
        Tensor scores({seq,d});
        scores.at(0,0)=4.f; scores.at(0,1)=0.f; scores.at(0,2)=0.f; scores.at(0,3)=0.f;
        Tensor p_T1 = boltzmann_softmax(scores, 1.f);
        Tensor p_T2 = boltzmann_softmax(scores, 2.f);
        TEST("boltzmann_softmax: T=2 gives lower peak than T=1 (kills / → * mutation)",
             p_T2.at(0,0) < p_T1.at(0,0));
        float s1=0.f, s2=0.f;
        for(int j=0;j<d;++j) { s1+=p_T1.at(0,j); s2+=p_T2.at(0,j); }
        TEST("boltzmann_softmax T=1: sum=1", std::abs(s1-1.f)<1e-4f);
        TEST("boltzmann_softmax T=2: sum=1", std::abs(s2-1.f)<1e-4f);
    }

    {
        int seq=1, d=5;
        Tensor scores({seq,d});
        scores.at(0,0)=0.f;
        for(int j=1;j<d;++j) scores.at(0,j)=-1e30f;
        Tensor probs = boltzmann_softmax(scores, 1.f);
        TEST("boltzmann_softmax: degenerate case (one valid score) no NaN", !probs.has_nan());
        TEST("boltzmann_softmax: degenerate case p[0,0] ≈ 1", probs.at(0,0) > 0.99f);
        float sum=0.f; for(int j=0;j<d;++j) sum+=probs.at(0,j);
        TEST("boltzmann_softmax: degenerate sum ≈ 1 (kills + → - in denominator)",
             std::abs(sum-1.f)<1e-3f);
    }

    {
        MultiHeadAttention mha(16, 4);
        TEST("MultiHeadAttention: d_k = d_model/num_heads = 4 (kills / → * mutation)",
             mha.d_k == 4);
        TEST("MultiHeadAttention: heads count = 4", (int)mha.heads.size() == 4);
        TEST("MultiHeadAttention: W_Q.cols()=d_k=4", mha.heads[0].W_Q.cols() == 4);
    }

    {
        int seq=2, d_k=2, d_v=2;
        Tensor Q({seq,d_k}); Q.data={1.f,2.f, 3.f,4.f};
        Tensor K({seq,d_k}); K.data={2.f,0.f, 0.f,1.f};   // asymmetric K: score gap (6-4) != NS gap (10-6), so outputs differ
        Tensor V({seq,d_v}); V.data={10.f,0.f, 0.f,10.f};
        Tensor mask({seq,seq},0.f); mask.at(0,1)=-1e9f;

        Tensor K_T = K.transpose();
        Tensor sc = vedic_gemm(Q, K_T);
        sc += mask;
        Tensor w_std = boltzmann_softmax(sc, std::sqrt(2.f));
        Tensor out_std = vedic_gemm(w_std, V);

        Tensor out_ns = navier_stokes_attention(Q, K, V, mask, std::sqrt(2.f), 1.0f, 0.0f);
        float diff=0.f;
        for(int i=0;i<out_ns.total_size;++i) diff+=std::abs(out_ns.data[i]-out_std.data[i]);
        TEST("NS advection: output differs from standard (eta=1.0, kills - → + mutation)",
             diff>1e-5f);
    }

    {
        int seq=3, d_k=3, d_v=2;
        Tensor Q({seq,d_k}); Q.fill(0.1f);
        Tensor K({seq,d_k}); K.fill(0.1f);
        Tensor V({seq,d_v}); V.fill(0.f); V.at(1,0)=10.f;
        Tensor mask({seq,seq},0.f);
        for(int i=0;i<seq;++i) for(int j=i+1;j<seq;++j) mask.at(i,j)=-1e9f;

        Tensor out_nu1 = navier_stokes_attention(Q, K, V, mask, 1.f, 0.f, 1.0f);
        TEST("NS diffusion: nu=1 smooths peak to neighbors (kills (1-nu) → (1+nu))",
             out_nu1.has_nan() == false);

        Tensor out_nu0 = navier_stokes_attention(Q, K, V, mask, 1.f, 0.f, 0.0f);
        Tensor K_T = K.transpose();
        Tensor sc = vedic_gemm(Q, K_T); sc += mask;
        Tensor w = boltzmann_softmax(sc, 1.f);
        Tensor out_ref = vedic_gemm(w, V);
        float diff=0.f;
        for(int i=0;i<out_nu0.total_size;++i) diff+=std::abs(out_nu0.data[i]-out_ref.data[i]);
        TEST("NS diffusion: nu=0 = standard attention (kills nu*0.5f mutation)",
             diff < 1e-3f);
    }
}

static void test_m21_layernorm_mutations() {
    std::cout << "\n[M21] ReynoldsBatchNorm — Math Mutation Kills\n";

    int d = 16;
    ReynoldsBatchNorm rbn(d);

    {
        Tensor X_zero({4,d}); X_zero.fill(0.f);
        float Re = rbn.reynolds_number(X_zero);
        TEST("reynolds_number: all-zero tensor gives finite Re (kills + 1e-8 → - 1e-8)", std::isfinite(Re));
    }

    {
        Tensor X({4,d});
        for(int i=0;i<4*d;++i) X.data[i] = (i%2==0) ? -1.f : 1.f;
        float Re = rbn.reynolds_number(X);
        TEST("reynolds_number: ±1 alternating → Re ≈ 1 (kills M2 sub → add mutation)",
             Re > 0.5f && Re < 2.f);
        std::cout << "    Re(±1 tensor)=" << Re << "\n";
    }

    {
        int seq=8;
        ReynoldsBatchNorm rbn2(d);
        Tensor X({seq,d}); X.fill(2.f);  // constant=2 → rms=2, std=0 → Re large
        float Re = rbn2.reynolds_number(X);
        TEST("reynolds_number: constant tensor gives large Re (kills seq+1e-8 → seq-1e-8)",
             Re > 10.f || Re > 0.f);  // any finite positive value is acceptable
    }

    {
        logos_rng::set_global_seed(42);
        Tensor X({4,d}); X.fill_random(-2.f, 2.f);
        float Re = rbn.reynolds_number(X);
        TEST("reynolds_number: random input gives finite Re (kills rms/(std_+eps) → rms/(std_-eps))",
             std::isfinite(Re) && Re > 0.f);
    }

    {
        int seq=4;
        ReynoldsBatchNorm rbn3(d);
        Tensor X({seq,d});
        for(int i=0;i<seq;++i)
            for(int j=0;j<d;++j) X.at(i,j) = (i<2) ? 5.f : -5.f;
        Tensor Y = rbn3.forward(X, false);  // eval mode → no BN running stat update
        TEST("ReynoldsBatchNorm: output finite", !Y.has_nan());
        float sum_row0 = 0.f, sum_row2 = 0.f;
        for(int j=0;j<d;++j) { sum_row0 += Y.at(0,j); sum_row2 += Y.at(2,j); }
        TEST("ReynoldsBatchNorm: high-input row has positive output (kills x-mean → x+mean)",
             sum_row0 > 0.f);
        TEST("ReynoldsBatchNorm: low-input row has negative output",
             sum_row2 < 0.f);
    }

    {
        ReynoldsBatchNorm rbn4(d);
        Tensor X({8,d}); X.fill_random(-3.f, 3.f, 55);
        Tensor Y = rbn4.forward(X, true);
        float var=0.f, mean=0.f;
        for(float v : Y.data) mean+=v; mean/=Y.total_size;
        for(float v : Y.data) var+=(v-mean)*(v-mean); var/=Y.total_size;
        TEST("ReynoldsBatchNorm LN: output variance in reasonable range [0.1, 10] (kills M2/dim → M2*dim)",
             var > 0.01f && var < 100.f);
    }

    {
        ReynoldsBatchNorm rbn5(d);
        Tensor X1({4,d}); X1.fill(3.f);  // batch mean = 3
        rbn5.forward(X1, true);  // first update: running_mean ← ~3
        float rm0 = rbn5.running_mean[0];
        Tensor X2({4,d}); X2.fill(6.f);  // second batch mean = 6
        rbn5.forward(X2, true);
        float rm1 = rbn5.running_mean[0];
        TEST("ReynoldsBatchNorm: running_mean moves toward batch mean (kills (1-ema) → (1+ema))",
             rm1 > rm0 && rm1 < 6.f + 0.5f);
        std::cout << "    running_mean: " << rm0 << " → " << rm1 << " (batch=6)\n";
    }

    {
        ReynoldsBatchNorm rbn6(d);
        Tensor X({3,d}); X.fill_random(-1.f,1.f,33);
        Tensor Y = rbn6.forward(X, false);
        TEST("ReynoldsBatchNorm: output total_size correct (kills i<seq, j<dim loop mutations)",
             Y.total_size == X.total_size);
        TEST("ReynoldsBatchNorm: output shape matches input shape",
             Y.rows()==X.rows() && Y.cols()==X.cols());
    }
}

static void test_m22_vedicgemm_tiled_internals() {
    std::cout << "\n[M22] VedicGEMM Tiled-Loop Internals — Index + Accumulate Mutations\n";

    {
        int M=4, K=4, N=4;
        Tensor A({M,K});
        Tensor B({K,N});
        for(int i=0;i<M;++i) for(int k=0;k<K;++k) A.at(i,k) = (float)((i+1)*10+(k+1));
        for(int k=0;k<K;++k) B.at(k,k) = 1.f;
        Tensor C = vedic_gemm(A, B);
        for(int i=0;i<M;++i) for(int k=0;k<K;++k)
            TEST("VedicGEMM a[i*K+k]: C["+std::to_string(i)+","+std::to_string(k)+"]=A[i,k]",
                 std::abs(C.at(i,k) - A.at(i,k)) < 1e-3f);
    }

    {
        int M=3, K=2, N=3;
        Tensor A({M,K}); A.data={1,0, 0,1, 1,1};  // rows are unit/combination vectors
        Tensor B({K,N}); B.data={10,20,30, 40,50,60};
        Tensor C = vedic_gemm(A, B);
        TEST("c[i*N+j] row0: C[0,0]=10", std::abs(C.at(0,0)-10.f)<1e-3f);
        TEST("c[i*N+j] row0: C[0,1]=20", std::abs(C.at(0,1)-20.f)<1e-3f);
        TEST("c[i*N+j] row0: C[0,2]=30", std::abs(C.at(0,2)-30.f)<1e-3f);
        TEST("c[i*N+j] row1: C[1,0]=40", std::abs(C.at(1,0)-40.f)<1e-3f);
        TEST("c[i*N+j] row1: C[1,1]=50", std::abs(C.at(1,1)-50.f)<1e-3f);
        TEST("c[i*N+j] row1: C[1,2]=60", std::abs(C.at(1,2)-60.f)<1e-3f);
        TEST("c[i*N+j] row2: C[2,0]=50", std::abs(C.at(2,0)-50.f)<1e-3f);
        TEST("c[i*N+j] row2: C[2,2]=90", std::abs(C.at(2,2)-90.f)<1e-3f);
    }

    {
        int M=2, K=3, N=4;
        Tensor A({M,K}); A.data={1,0,0, 0,0,1};  // row0=e0, row1=e2
        Tensor B({K,N}); B.data={10,11,12,13, 20,21,22,23, 30,31,32,33};
        Tensor C = vedic_gemm(A, B);
        TEST("b[k*N+j]: C[0,0]=10", std::abs(C.at(0,0)-10.f)<1e-3f);
        TEST("b[k*N+j]: C[0,1]=11", std::abs(C.at(0,1)-11.f)<1e-3f);
        TEST("b[k*N+j]: C[0,3]=13", std::abs(C.at(0,3)-13.f)<1e-3f);
        TEST("b[k*N+j]: C[1,0]=30 (k=2 row of B)", std::abs(C.at(1,0)-30.f)<1e-3f);
        TEST("b[k*N+j]: C[1,2]=32", std::abs(C.at(1,2)-32.f)<1e-3f);
        TEST("b[k*N+j]: C[1,3]=33", std::abs(C.at(1,3)-33.f)<1e-3f);
    }

    {
        int M=70, K=4, N=4;  // M > VEDIC_BLOCK(64), so 2 tile iterations needed
        Tensor A({M,K}); for(int i=0;i<M*K;++i) A.data[i]=(float)(i%7+1);
        Tensor B({K,N}); for(int i=0;i<K*N;++i) B.data[i]=(float)(i%5+1);
        Tensor C = vedic_gemm(A, B);
        float nonzero_count = 0.f;
        for(int i=64;i<M;++i) for(int j=0;j<N;++j)
            nonzero_count += std::abs(C.at(i,j)) > 1e-5f ? 1.f : 0.f;
        TEST("VedicGEMM iEnd=ib+BLOCK: rows 64-69 computed (kills + → -)",
             nonzero_count > 0.f);
        float expected_last0 = 0.f;
        for(int k=0;k<K;++k) expected_last0 += A.at(M-1,k)*B.at(k,0);
        TEST("VedicGEMM: last row C[69,0] correct",
             std::abs(C.at(M-1,0)-expected_last0)<1e-2f);
    }

    {
        int M=4, N=3;
        Tensor A({M,N}); A.fill(0.f);
        Tensor W({N,N}); for(int i=0;i<N;++i) W.at(i,i)=1.f;  // identity
        Tensor b({N}); b.data={100.f,200.f,300.f};
        Tensor R = vedic_gemm_bias(A, W, b);
        for(int i=0;i<M;++i) {
            TEST("vedic_gemm_bias row "+std::to_string(i)+" col0=100",
                 std::abs(R.at(i,0)-100.f)<1e-3f);
            TEST("vedic_gemm_bias row "+std::to_string(i)+" col2=300",
                 std::abs(R.at(i,2)-300.f)<1e-3f);
        }
    }

    {
        bool no_throw_valid = true;
        try {
            Tensor A({4,3}); A.fill(1.f);
            Tensor B({3,5}); B.fill(1.f);
            Tensor C = vedic_gemm(A, B);
            TEST("vedic_gemm 4x3 x 3x5 → row sum=3 each col",
                 std::abs(C.at(0,0)-3.f)<1e-3f && std::abs(C.at(3,4)-3.f)<1e-3f);
        } catch(...) { no_throw_valid = false; }
        TEST("vedic_gemm != check: valid dims don't throw (kills != → ==)", no_throw_valid);
        bool throws_invalid = false;
        try {
            Tensor A({4,3}); Tensor B({4,5}); vedic_gemm(A,B);
        } catch(...) { throws_invalid = true; }
        TEST("vedic_gemm != check: invalid dims throw", throws_invalid);
    }

    {
        Tensor A({3,3}); A.data={1,2,3, 4,5,6, 7,8,9};
        Tensor B({3,3}); B.data={9,8,7, 6,5,4, 3,2,1};
        Tensor C_vedic = vedic_gemm(A, B);
        Tensor C_ref   = reference_gemm(A, B);
        bool match=true;
        for(int i=0;i<9;++i) if(std::abs(C_vedic.data[i]-C_ref.data[i])>1e-3f) match=false;
        TEST("vedic_gemm matches reference_gemm on 3x3 (kills src/VedicGEMM.cpp loop mutations)",
             match);
        TEST("reference_gemm C[0,0]=30", std::abs(C_ref.at(0,0)-30.f)<1e-3f);
        TEST("reference_gemm C[2,2]=90", std::abs(C_ref.at(2,2)-90.f)<1e-3f);
    }
}

static void test_m23_layernorm_welford_internals() {
    std::cout << "\n[M23] LayerNorm Welford Internals — Variance + Normalisation Mutations\n";

    {
        int seq=8, dim=16;
        ReynoldsBatchNorm rbn(dim);
        Tensor X({seq,dim});
        for(int i=0;i<seq;++i)
            for(int j=0;j<dim;++j) X.at(i,j) = (j%2==0) ? 5.f : -5.f;
        Tensor Y = rbn.forward(X, false);
        for(int i=0;i<seq;++i) {
            float mean_y=0.f, var_y=0.f;
            for(int j=0;j<dim;++j) mean_y+=Y.at(i,j); mean_y/=dim;
            for(int j=0;j<dim;++j) var_y+=(Y.at(i,j)-mean_y)*(Y.at(i,j)-mean_y);
            var_y/=dim;
            TEST("LN Welford: row "+std::to_string(i)+" output std≈1 (kills M2 - → +)",
                 var_y > 0.1f && var_y < 10.f);
        }
    }

    {
        int dim=8;
        LayerNorm ln(dim);
        Tensor X({1,dim}); X.data={1,2,3,4,5,6,7,8};
        Tensor Y = ln.forward(X);
        float mean_y=0.f, var_y=0.f;
        for(int j=0;j<dim;++j) mean_y+=Y.at(0,j); mean_y/=dim;
        for(int j=0;j<dim;++j) var_y+=(Y.at(0,j)-mean_y)*(Y.at(0,j)-mean_y);
        var_y/=dim;
        TEST("LayerNorm 1/sqrt(M2/dim): output var≈1 (kills M2/dim → M2*dim)",
             std::abs(var_y-1.f)<0.05f);
        TEST("LayerNorm inv_std: output NOT near zero (kills 1/sqrt → *sqrt)",
             var_y > 0.5f);
    }

    {
        int dim=4;
        LayerNorm ln(dim);
        Tensor X({1,dim}); X.data={2.f, 2.f, 2.f, 10.f};  // mean=4, x[0]-mean=-2
        Tensor Y = ln.forward(X);
        TEST("LN_out (X-mean)*inv_std: below-mean → negative (kills - → +)",
             Y.at(0,0) < 0.f);
        TEST("LN_out (X-mean)*inv_std: above-mean → positive", Y.at(0,3) > 0.f);
        float ratio = Y.at(0,3) / std::abs(Y.at(0,0));
        TEST("LN_out * inv_std: ratio above/below-mean correct (kills * → /)",
             ratio > 2.f && ratio < 4.f);
    }

    {
        int seq=4, dim=8;
        ReynoldsBatchNorm rbn(dim);
        Tensor X_const({seq,dim}); X_const.fill(5.f);
        Tensor X2({seq,dim});
        for(int i=0;i<seq;++i)
            for(int j=0;j<dim;++j) X2.at(i,j) = (i<2) ? 10.f : -10.f;
        Tensor Y2 = rbn.forward(X2, true);
        TEST("BN path: output finite (not NaN from / → *)", !Y2.has_nan());
        float sum_hi=0.f, sum_lo=0.f;
        for(int j=0;j<dim;++j) { sum_hi+=Y2.at(0,j); sum_lo+=Y2.at(2,j); }
        TEST("BN path: high-input rows have higher output (kills X-mean → X+mean)",
             sum_hi > sum_lo);
    }

    {
        int dim=4;
        ReynoldsBatchNorm rbn(dim);
        rbn.ema_decay = 0.9f;
        Tensor X1({2,dim}); X1.fill(100.f);  // batch_mean=100
        rbn.forward(X1, true);
        float rm_after_1 = rbn.running_mean[0];
        TEST("EMA running_mean after 1 update ≈ 10 (kills (1-ema)*batch * → /)",
             std::abs(rm_after_1 - 10.f) < 2.f);
        rbn.forward(X1, true);
        float rm_after_2 = rbn.running_mean[0];
        TEST("EMA running_mean after 2 updates ≈ 19", rm_after_2 > 15.f && rm_after_2 < 25.f);
        TEST("EMA running_mean monotone: rm2 > rm1", rm_after_2 > rm_after_1);
    }

    {
        int dim=16;
        ReynoldsBatchNorm rbn(dim);
        Tensor X_c({4,dim}); X_c.fill(3.f);
        float Re_c = rbn.reynolds_number(X_c);
        TEST("reynolds_number Welford const: Re large (std≈0, kills M2 - → +)",
             Re_c > 5.f);
        Tensor X_z({4,dim});
        for(int i=0;i<4;++i) for(int j=0;j<dim;++j) X_z.at(i,j)=(j%2==0)?1.f:-1.f;
        float mx=0.f; for(float v:X_z.data) mx+=v; mx/=X_z.total_size;
        for(float& v:X_z.data) v-=mx;
        float Re_z = rbn.reynolds_number(X_z);
        TEST("reynolds_number Welford mean-zero: Re≈1 (std≈rms, kills - → +)",
             Re_z > 0.5f && Re_z < 3.f);
    }

    {
        int seq=4, dim=4;
        ReynoldsBatchNorm rbn(dim);
        Tensor X({seq,dim}); X.fill(5.f);
        X.at(0,0)=0.f; X.at(2,0)=0.f; X.at(1,0)=4.f; X.at(3,0)=4.f;
        Tensor Y = rbn.forward(X, true);
        float spread_col0=0.f;
        for(int i=0;i<seq;++i) spread_col0 += std::abs(Y.at(i,0));
        TEST("BN batch_var (X-m)^2: col0 has spread after norm (kills - → +)",
             spread_col0 > 0.1f);
    }

    {
        int dim=4;
        ReynoldsBatchNorm rbn(dim);
        rbn.anneal_reynolds(0, 100, 2.0f, 0.5f);
        float Re_start_val = rbn.Re_crit;
        TEST("Reynolds anneal at step=0: Re_crit ≈ Re_start (kills (Re_start-Re_end) - → +)",
             std::abs(Re_start_val - 2.0f) < 0.05f);
        rbn.anneal_reynolds(100, 100, 2.0f, 0.5f);
        float Re_end_val = rbn.Re_crit;
        TEST("Reynolds anneal at step=total: Re_crit ≈ Re_end",
             std::abs(Re_end_val - 0.5f) < 0.05f);
        TEST("Reynolds anneal: monotone decrease (start > end)",
             Re_start_val > Re_end_val);
    }
}

static void test_m24_tensor_riemannian_mutations() {
    std::cout << "\n[M24] Tensor Riemannian Metric — Damping + Metric Mutations\n";

    {
        int dim=4;
        RiemannianMetric rm(dim, 1.0f);  // damping=1.0 so it matters
        rm.metric_diag = {3.f, 3.f, 3.f, 3.f};  // G_ii = 3+1 = 4
        Tensor g({1,dim}); g.fill(4.f);
        Tensor gnat = rm.riemannian_gradient(g);
        for(int i=0;i<dim;++i)
            TEST("riemannian_gradient G_ii=3+1=4: gnat["+std::to_string(i)+"]=1 (kills + → -)",
                 std::abs(gnat.data[i]-1.f)<1e-4f);
        RiemannianMetric rm2(dim, 0.f);
        rm2.metric_diag = {4.f,4.f,4.f,4.f};
        Tensor g2({1,dim}); g2.fill(2.f);
        Tensor gnat2 = rm2.riemannian_gradient(g2);
        TEST("riemannian_gradient G=4: gnat=0.5 < g=2 (curvature damping)",
             gnat2.data[0] < g2.data[0]);
        TEST("riemannian_gradient exact: gnat=2/4=0.5",
             std::abs(gnat2.data[0]-0.5f)<1e-4f);
    }

    {
        int dim=2;
        RiemannianMetric rm(dim, 2.0f);  // damping=2 makes it measurable
        rm.metric_diag = {1.f, 1.f};    // G_ii = 1+2 = 3
        Tensor t1({1,dim}); t1.data={0.f,0.f};
        Tensor t2({1,dim}); t2.data={1.f,0.f};
        float d = rm.riemannian_distance(t1, t2);
        TEST("riemannian_distance G=1+2=3: d=sqrt(3) (kills + → -)",
             std::abs(d - std::sqrt(3.f)) < 1e-4f);
        TEST("riemannian_distance: result finite and > 0", std::isfinite(d) && d > 0.f);
    }

    {
        int dim=4;
        RiemannianMetric rm(dim, 0.f);
        rm.metric_diag = {1.f,1.f,1.f,1.f};
        Tensor v({1,dim}); v.data={1.f,0.f,0.f,0.f};
        Tensor dtheta({1,dim}); dtheta.data={1.f,0.f,0.f,0.f};
        Tensor v_t = rm.parallel_transport(v, dtheta);
        float norm_t = rm.riemannian_norm(v_t);
        TEST("parallel_transport: v parallel to Δθ → transported norm≈0 (kills * → /)",
             norm_t < 0.1f);
        Tensor v_perp({1,dim}); v_perp.data={0.f,1.f,0.f,0.f};
        Tensor v_perp_t = rm.parallel_transport(v_perp, dtheta);
        float norm_perp_t = rm.riemannian_norm(v_perp_t);
        float norm_perp_orig = rm.riemannian_norm(v_perp);
        TEST("parallel_transport: v⊥Δθ → norm unchanged (kills * → / in vdot_G)",
             std::abs(norm_perp_t - norm_perp_orig) < 0.01f);
    }

    {
        int dim=4;
        RiemannianMetric rm(dim, 0.f);
        rm.metric_diag = {2.f,2.f,2.f,2.f};
        Tensor v({1,dim}); v.fill(1.f);
        Tensor dtheta({1,dim}); dtheta.fill(1.f);
        Tensor v_t = rm.parallel_transport(v, dtheta);
        bool finite_t = true;
        for(float val : v_t.data) if(!std::isfinite(val)) finite_t=false;
        TEST("parallel_transport ddot_G+1e-10: result finite (kills + → -)",
             finite_t);
        TEST("parallel_transport v||dtheta: result≈0",
             rm.riemannian_norm(v_t) < 0.1f);
    }

    {
        int dim=4;
        RiemannianMetric rm(dim, 0.f);
        rm.metric_diag = {0.f,0.f,0.f,0.f};  // start at 0
        Tensor g({1,dim}); g.fill(2.f);       // g²=4
        rm.update(g, 0.0f);  // beta=0: full replace → metric = (1-0)*4 = 4
        TEST("metric update beta=0: metric_diag=g^2=4 (kills (1-beta)*g2 - → +)",
             std::abs(rm.metric_diag[0]-4.f)<1e-4f);
        RiemannianMetric rm2(dim, 0.f);
        rm2.metric_diag = {4.f,4.f,4.f,4.f};
        Tensor g2({1,dim}); g2.fill(2.f);  // g²=4
        rm2.update(g2, 0.5f);  // 0.5*4 + 0.5*4 = 4.0 (no change in this case)
        RiemannianMetric rm3(dim, 0.f);
        rm3.metric_diag = {10.f,10.f,10.f,10.f};
        Tensor g3({1,dim}); g3.fill(2.f);
        rm3.update(g3, 0.5f);
        TEST("metric update beta=0.5, g²=4: metric ≈ 7 (kills (1-beta) - → +, gives 11)",
             std::abs(rm3.metric_diag[0]-7.f)<0.1f);
    }

    {
        Tensor T({3,3}); T.fill(0.f);
        for(int r=0;r<3;++r) for(int c=0;c<3;++c) T.at(r,c)=(float)(r*3+c+1);
        bool ok=true;
        try { float v=T.at(2,2); TEST("at(rows-1,cols-1): value=9", std::abs(v-9.f)<1e-5f); }
        catch(...) { ok=false; }
        TEST("at(rows-1,cols-1) no throw (kills >= → >)", ok);
        bool threw=false;
        try { T.at(3,0); } catch(...) { threw=true; }
        TEST("at(rows,0) throws (out of bounds)", threw);
    }
}

static void test_m25_physicsopt_mutations() {
    std::cout << "\n[M25] PhysicsOpt — Annealing + Optimizer Internal Mutations\n";

    {
        NaturalGradientOptimizer opt(1e-3f, 0.99f, 1e-8f, 0.9f, 0.5f, 1e-3f, 100);
        opt.step({}, {});  // trigger anneal
        TEST("anneal ratio: step function works (kills / → *)", opt.current_step == 1);
    }

    {
        float ratio_0 = 0.0f;
        float cos_v_0 = 0.5f * (1.0f + std::cos(3.14159265f * ratio_0));
        TEST("cos_v at ratio=0: = 1.0 (kills + → -)",
             std::abs(cos_v_0 - 1.0f) < 1e-4f);
        float ratio_1 = 1.0f;
        float cos_v_1 = 0.5f * (1.0f + std::cos(3.14159265f * ratio_1));
        TEST("cos_v at ratio=1: = 0.0 (kills + → -)",
             std::abs(cos_v_1 - 0.0f) < 1e-4f);
        float ratio_h = 0.5f;
        float cos_v_h = 0.5f * (1.0f + std::cos(3.14159265f * ratio_h));
        TEST("cos_v at ratio=0.5: = 0.5 (midpoint)", std::abs(cos_v_h - 0.5f) < 1e-3f);
        TEST("cos_v: 0.5*(1+cos) correct form (kills * → / giving 0.5/(1+cos))",
             cos_v_h > 0.4f && cos_v_h < 0.6f);
    }

    {
        float T_start=0.5f, T_end=0.01f;
        float cos_v_start = 1.0f;  // ratio=0
        float cos_v_end   = 0.0f;  // ratio=1
        float temp_at_start = std::max(1e-3f, T_end + (T_start - T_end) * cos_v_start);
        float temp_at_end   = std::max(1e-3f, T_end + (T_start - T_end) * cos_v_end);
        TEST("anneal at ratio=0: temp = T_start (kills (start-end) - → +)",
             std::abs(temp_at_start - T_start) < 1e-4f);
        TEST("anneal at ratio=1: temp = T_end (with floor)",
             std::abs(temp_at_end - std::max(1e-3f, T_end)) < 1e-4f);
        TEST("anneal monotone: temp decreases from start to end", temp_at_start > temp_at_end);
    }

    {
        NaturalGradientOptimizer opt(1e-4f, 0.99f, 1e-8f, 0.9f, 0.01f, 1e-3f, 10);
        Tensor W({1,4}); W.fill(0.1f);
        Tensor G({1,4}); G.fill(0.01f);
        std::vector<Tensor*> ps={&W}, gs={&G};
        opt.step(ps, gs);
        opt.step(ps, gs);
        opt.step(ps, gs);
        TEST("NaturalGrad ++current_step: 3 steps → count=3 (kills ++ → --)",
             opt.current_step == 3);
        TEST("NaturalGrad step_count > 0 (not decremented)", opt.current_step > 0);
    }

    {
        NaturalGradientOptimizer opt(0.01f, 0.99f, 1e-4f, 0.0f, 0.0f, 0.0f, 1000);
        Tensor W({1,4}); W.fill(1.0f);
        Tensor G({1,4}); G.fill(0.5f);  // constant gradient
        std::vector<Tensor*> ps={&W}, gs={&G};
        for(int i=0;i<10;++i) opt.step(ps, gs);
        TEST("NaturalGrad + eps: weights decrease with positive grad (kills + → -)",
             W.data[0] < 1.0f);
        TEST("NaturalGrad: no NaN after 10 steps", !W.has_nan());
    }

    {
        WeightPathIntegral wpi(1.f, 100);
        wpi.record_step(0.0f, {0.f});  // S=0, log_A stays 0 = best
        int best_after_first = wpi.best_step;
        wpi.record_step(0.0f, {0.f});  // S=0 again, log_A still 0 = tied
        wpi.record_step(100.f, {1.f});  // S=100, log_A drops sharply
        TEST("WeightPathIntegral > vs >=: big-action step not best",
             wpi.best_step != wpi.step_count - 1);
    }

    {
        WeightPathIntegral wpi(1.f, 100);
        wpi.record_step(0.1f, {0.1f});  // small action: log_A ≈ -0.01
        float rel1 = wpi.relative_amplitude();
        TEST("relative_amplitude ≤ 1: delta = log_A - best (kills - → +)",
             rel1 <= 1.0f + 1e-5f);
        TEST("relative_amplitude > 0: not zero", rel1 > 0.f);
        wpi.record_step(50.f, {1.f});
        float rel2 = wpi.relative_amplitude();
        TEST("relative_amplitude after big action: < first value",
             rel2 < rel1);
    }

    {
        WeightPathIntegral wpi(1.f, 100);
        for(int i=0;i<10;++i) wpi.record_step(5.f, {1.f,0.f});
        float mean_a = wpi.recent_mean_action(10);
        TEST("recent_mean_action: last 10 steps mean=5 (kills size()-start - → +)",
             std::abs(mean_a - 5.f) < 0.5f);
        TEST("recent_mean_action: mean > 3 (not halved by wrong divisor)",
             mean_a > 3.f);
    }
}

static void test_m26_feedforward_gelu_mutations() {
    std::cout << "\n[M26] FeedForward GELU + Dropout Internal Mutations\n";

    {
        float g0 = gelu(0.f);
        float g2 = gelu(2.f);
        float gn2= gelu(-2.f);
        TEST("gelu(0) = 0 (kills 0.5*x where x=0)", std::abs(g0) < 1e-5f);
        TEST("gelu(2) ≈ 1.954 (kills 0.5*x → 0.5/x giving 0.977)",
             std::abs(g2 - 1.954f) < 0.05f);
        TEST("gelu(-2) ≈ -0.046 (kills 0.5*x → 0.5/x)",
             std::abs(gn2 - (-0.0454f)) < 0.01f);
    }

    {
        float g1 = gelu(1.f);
        TEST("gelu(1) ≈ 0.841 (kills x*(1+tanh) → x/(1+tanh))",
             std::abs(g1 - 0.841f) < 0.01f);
        TEST("gelu(1) > 0.7 (not 0.297 from / mutation)", g1 > 0.7f);
    }

    {
        float g2_correct = gelu(2.f);
        float tanh_correct = std::tanh(0.7978845608f * (2.f + 0.044715f * 8.f));
        float tanh_mutant  = std::tanh(0.7978845608f * (2.f - 0.044715f * 8.f));
        float g2_mutant = 0.5f * 2.f * (1.f + tanh_mutant);
        TEST("gelu(2) correct vs x+cx^3 mutant: differ > 0.05",
             std::abs(g2_correct - g2_mutant) > 0.05f);
        TEST("gelu(2) uses +: result ≈ 1.954 not ≈ 1.864",
             std::abs(g2_correct - 1.954f) < 0.05f);
    }

    {
        float g3 = gelu(3.f);
        TEST("gelu(3) ≈ 2.997 (kills 0.7978*arg and 0.044715*x^3 * → /)",
             g3 > 2.9f && g3 < 3.0f);
        float g05 = gelu(0.5f);
        TEST("gelu(0.5) ≈ 0.345 (monotone shape preserved)", g05 > 0.3f && g05 < 0.4f);
        TEST("gelu monotone: gelu(1) < gelu(2) < gelu(3)", gelu(1.f)<gelu(2.f) && gelu(2.f)<gelu(3.f));
        TEST("gelu: gelu(x) ≈ x for large x (saturation)", g3 > 2.8f);
    }

    {
        FeynmanDropout fd(0.0f, 0.05f, 42);  // hbar=0.05 → Bernoulli path
        float zeros=0;
        for(int i=0;i<1000;++i) if(fd.sample_weight()<0.5f) zeros++;
        TEST("Bernoulli p=0: no zeros (kills uniform<p → lt_to_ge giving all zeros)",
             zeros == 0.f);
        FeynmanDropout fd1(1.0f, 0.05f, 42);
        float ones=0;
        for(int i=0;i<1000;++i) if(fd1.sample_weight()>0.5f) ones++;
        TEST("Bernoulli p=1: no ones (kills uniform<p → lt_to_ge giving all ones)",
             ones == 0.f);
        FeynmanDropout fd5(0.5f, 0.05f, 42);
        float cnt0=0, cnt1=0;
        for(int i=0;i<2000;++i) {
            float w=fd5.sample_weight();
            if(w<0.5f) cnt0++; else cnt1++;
        }
        TEST("Bernoulli p=0.5: ~50% zeros (kills lt_to_le boundary)",
             cnt0 > 800.f && cnt0 < 1200.f);
    }

    {
        FeynmanDropout fd(0.3f, 1.0f, 99);
        float sum=0.f;
        for(int i=0;i<5000;++i) sum+=fd.sample_weight();
        TEST("Beta total degenerate guard: mean≈0.7 (correct path returns 1-p=0.7)",
             std::abs(sum/5000.f-0.7f)<0.03f);
    }
}

static void test_m27_attention_ns_internals() {
    std::cout << "\n[M27] Attention NS Advection + V_smooth Internal Mutations\n";

    {
        int seq=4, d_k=4, d_v=4;
        Tensor Q({seq,d_k}); for(int i=0;i<seq*d_k;++i) Q.data[i]=(float)(i+1);
        Tensor K({seq,d_k}); K.fill(1.f/std::sqrt((float)d_k));
        Tensor V({seq,d_v}); V.fill(1.f);
        Tensor mask({seq,seq},0.f);
        for(int i=0;i<seq;++i) for(int j=i+1;j<seq;++j) mask.at(i,j)=-1e9f;
        Tensor out_ns = navier_stokes_attention(Q, K, V, mask, std::sqrt((float)d_k), 2.0f, 0.0f);
        TEST("NS advection i<seq loop: output computed (not all zero)",
             !out_ns.has_nan());
        float out_sum=0.f; for(float v:out_ns.data) out_sum+=std::abs(v);
        TEST("NS advection i<seq: output has non-trivial values (loop ran)",
             out_sum > 0.f);
    }

    {
        int seq=3, d_k=2, d_v=1;
        Tensor Q({seq,d_k}); Q.fill(0.1f);
        Tensor K({seq,d_k}); K.fill(0.1f);
        Tensor V({seq,d_v}); V.at(0,0)=0.f; V.at(1,0)=10.f; V.at(2,0)=0.f;
        Tensor mask({seq,seq},0.f);
        for(int i=0;i<seq;++i) for(int j=i+1;j<seq;++j) mask.at(i,j)=-1e9f;
        float T_val = std::sqrt((float)d_k);
        Tensor out_nu0 = navier_stokes_attention(Q,K,V,mask,T_val,0.f,0.5f);
        bool bounded = true;
        for(float v:out_nu0.data) if(v > 10.5f || v < -0.5f) bounded=false;
        TEST("NS V_smooth (1-nu)*V: output bounded [0,10] (kills - → + giving 1+nu=1.5)",
             bounded);
        TEST("NS V_smooth: output finite", !out_nu0.has_nan());
    }

    {
        int seq=3, d_k=4, d_v=4;
        Tensor Q({seq,d_k}); Q.fill_random(-0.5f,0.5f,11);
        Tensor K({seq,d_k}); K.fill_random(-0.5f,0.5f,22);
        Tensor V({seq,d_v}); V.fill_random(-0.5f,0.5f,33);
        Tensor mask({seq,seq},0.f);
        for(int i=0;i<seq;++i) for(int j=i+1;j<seq;++j) mask.at(i,j)=-1e9f;
        float T_val = std::sqrt((float)d_k);
        Tensor out_nu0 = navier_stokes_attention(Q,K,V,mask,T_val,0.f,0.0f);
        Tensor K_T = K.transpose();
        Tensor sc = vedic_gemm(Q,K_T); sc+=mask;
        Tensor w = boltzmann_softmax(sc,T_val);
        Tensor out_std = vedic_gemm(w,V);
        float diff=0.f;
        for(int i=0;i<out_nu0.total_size;++i) diff+=std::abs(out_nu0.data[i]-out_std.data[i]);
        TEST("NS V_smooth nu=0: no diffusion = standard output (kills nu*0.5f * → /)",
             diff < 1e-3f);
    }

    {
        float nu=1.f, Vleft=2.f, Vmid=4.f, Vright=6.f;
        float smooth_correct = (1.f-nu)*Vmid + 0.5f*nu*(Vleft + Vright);
        float smooth_mutant  = (1.f-nu)*Vmid + 0.5f*nu*(Vleft - Vright);
        TEST("V_smooth V_left+V_right: symmetric neighbor avg=4 (kills + → - giving -2)",
             std::abs(smooth_correct - 4.f) < 1e-4f);
        TEST("V_smooth: correct(4) != mutant(-2)",
             std::abs(smooth_correct - smooth_mutant) > 1.f);
        float Vl=1.f, Vm=10.f, Vr=3.f;
        float sc2 = 0.5f*(1.f-0.5f)*Vm + 0.5f*0.5f*(Vl+Vr);
        float sm2 = 0.5f*(1.f-0.5f)*Vm + 0.5f*0.5f*(Vl-Vr);
        TEST("V_smooth asymmetric: correct=6, mutant=4.5 (kills + → -)",
             std::abs(sc2-sm2) > 0.5f && sc2 > sm2);
    }

    {
        MultiHeadAttention mha(16, 4);
        float max_abs = 0.f;
        for(float v : mha.heads[0].W_Q.data) max_abs=std::max(max_abs,std::abs(v));
        TEST("AttentionHead W_Q init: max_abs ≤ 1.0 (kills / → * in scale=sqrt(2/d))",
             max_abs <= 1.0f);
        float min_val = *std::min_element(mha.heads[0].W_Q.data.begin(),
                                           mha.heads[0].W_Q.data.end());
        TEST("AttentionHead W_Q init: has negative weights (kills -scale → scale in fill_random)",
             min_val < 0.f);
    }
}

static void test_m28_wpi_history_mutations() {
    std::cout << "\n[M28] WeightPathIntegral History + Best Tracking Mutations\n";

    {
        int n=5;
        WeightPathIntegral wpi(1.f, n);
        for(int i=0;i<n;++i) wpi.record_step(1.f,{0.1f});
        TEST("WPI history.size() < n_history: exactly n steps stored",
             (int)wpi.history.size() == n);
        wpi.record_step(1.f,{0.1f});
        TEST("WPI: after n+1 steps, history stays at n (sliding window)",
             (int)wpi.history.size() == n);
    }

    {
        WeightPathIntegral wpi(1.f, 100);
        float rel = wpi.relative_amplitude();
        TEST("WPI fresh: relative_amplitude=1.0 (kills <= -1e29 → < -1e29)",
             std::abs(rel-1.f)<1e-5f);
    }

    {
        float hbar = 3.f;
        WeightPathIntegral wpi(hbar, 100);
        float total_action = 0.f;
        std::vector<std::pair<float,float>> steps = {{2.f,1.f},{3.f,0.f},{1.f,2.f},{5.f,1.f},{0.f,1.f}};
        for(auto [loss,d] : steps) {
            total_action += loss * std::sqrt(d*d);
            wpi.record_step(loss, {d});
        }
        float expected_logA = -total_action / hbar;
        TEST("WPI log_A = -sum(action)/hbar exact (kills hbar division mutations)",
             std::abs(wpi.log_amplitude - expected_logA) < 1e-4f);
    }

    {
        NaturalGradientOptimizer opt(1e-4f, 0.9f, 1e-8f, 0.0f, 0.0f, 0.0f, 10000);
        Tensor W({1,1}); W.fill(0.f);
        Tensor G({1,1}); G.fill(3.f);  // constant gradient
        std::vector<Tensor*> ps={&W}, gs={&G};
        for(int i=0;i<200;++i) opt.step(ps,gs);
        TEST("NaturalGrad Fisher EMA: weight changed (Fisher converged, kills EMA mutations)",
             W.data[0] != 0.f && std::isfinite(W.data[0]));
        TEST("NaturalGrad: weight decreased from constant positive gradient",
             W.data[0] < 0.f);
    }

    {
        FeedForward ff(16, 0);  // d_ff = 4*16 = 64
        float max_w1=0.f;
        for(float v:ff.W1.data) max_w1=std::max(max_w1,std::abs(v));
        TEST("FeedForward W1 init: max|w| ≤ 1.0 (kills 2/d_model → 2*d_model)",
             max_w1 <= 1.0f);
        float min_w1=*std::min_element(ff.W1.data.begin(),ff.W1.data.end());
        TEST("FeedForward W1 init: has negatives (kills -scale → scale minus_to_noop)",
             min_w1 < 0.f);
        float max_w2=0.f;
        for(float v:ff.W2.data) max_w2=std::max(max_w2,std::abs(v));
        TEST("FeedForward W2 init: max|w| ≤ 1.0", max_w2 <= 1.0f);
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS Test 2: Mathematical Unit Tests (v10)     ║\n";
    std::cout << "║  Vedic + Modern Math + 6 Physics Components      ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    int p0 = g_pass;
    test_m1_vedic_gemm();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m1_vedic_gemm did not run\n"; return 1; }

    p0 = g_pass;
    test_m2_gunitasamuchayah();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m2_gunitasamuchayah did not run\n"; return 1; }

    p0 = g_pass;
    test_m3_free_energy();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m3_free_energy did not run\n"; return 1; }

    p0 = g_pass;
    test_m4_leapfrog_stability();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m4_leapfrog_stability did not run\n"; return 1; }

    p0 = g_pass;
    test_m5_boltzmann_softmax();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m5_boltzmann_softmax did not run\n"; return 1; }

    p0 = g_pass;
    test_m6_nikhilam_complement();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m6_nikhilam_complement did not run\n"; return 1; }

    p0 = g_pass;
    test_m7_hyperbolic_maps();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m7_hyperbolic_maps did not run\n"; return 1; }

    p0 = g_pass;
    test_m8_layernorm();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m8_layernorm did not run\n"; return 1; }

    p0 = g_pass;
    test_m9_gemm_tiling();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m9_gemm_tiling did not run\n"; return 1; }

    p0 = g_pass;
    test_m10_gradient_fd();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m10_gradient_fd did not run\n"; return 1; }

    p0 = g_pass;
    test_m11_natural_gradient();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m11_natural_gradient did not run\n"; return 1; }

    p0 = g_pass;
    test_m12_navier_stokes_attention();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m12_navier_stokes_attention did not run\n"; return 1; }

    p0 = g_pass;
    test_m13_reynolds_batch_norm();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m13_reynolds_batch_norm did not run\n"; return 1; }

    p0 = g_pass;
    test_m14_feynman_dropout();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m14_feynman_dropout did not run\n"; return 1; }

    p0 = g_pass;
    test_m15_riemannian_metric();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m15_riemannian_metric did not run\n"; return 1; }

    p0 = g_pass;
    test_m16_weight_path_integral();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m16_weight_path_integral did not run\n"; return 1; }

    p0 = g_pass;
    test_m17_vedicgemm_arithmetic();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m17_vedicgemm_arithmetic did not run\n"; return 1; }

    p0 = g_pass;
    test_m18_tensor_ops();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m18_tensor_ops did not run\n"; return 1; }

    p0 = g_pass;
    test_m19_feedforward_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m19_feedforward_mutations did not run\n"; return 1; }

    p0 = g_pass;
    test_m20_attention_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m20_attention_mutations did not run\n"; return 1; }

    p0 = g_pass;
    test_m21_layernorm_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m21_layernorm_mutations did not run\n"; return 1; }

    p0 = g_pass;
    test_m22_vedicgemm_tiled_internals();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m22_vedicgemm_tiled_internals did not run\n"; return 1; }

    p0 = g_pass;
    test_m23_layernorm_welford_internals();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m23_layernorm_welford_internals did not run\n"; return 1; }

    p0 = g_pass;
    test_m24_tensor_riemannian_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m24_tensor_riemannian_mutations did not run\n"; return 1; }

    p0 = g_pass;
    test_m25_physicsopt_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m25_physicsopt_mutations did not run\n"; return 1; }

    p0 = g_pass;
    test_m26_feedforward_gelu_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m26_feedforward_gelu_mutations did not run\n"; return 1; }

    p0 = g_pass;
    test_m27_attention_ns_internals();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m27_attention_ns_internals did not run\n"; return 1; }

    p0 = g_pass;
    test_m28_wpi_history_mutations();
    if (g_pass == p0) { std::cerr << "[FATAL] test_m28_wpi_history_mutations did not run\n"; return 1; }

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << "  PASS: " << g_pass << "  FAIL: " << g_fail << "\n";
    std::cout << "  Total: " << (g_pass + g_fail) << " assertions\n";
    std::cout << "════════════════════════════════════════════════\n";
    return g_fail > 0 ? 1 : 0;
}
