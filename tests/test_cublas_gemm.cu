// ═══════════════════════════════════════════════════════════════════════════

#include "../cuda/CublasGemm.cuh"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#ifndef LOGOS_CUBLAS_TF32
#define LOGOS_CUBLAS_TF32 0
#endif

#if LOGOS_CUBLAS_TF32
static constexpr double kTol = 2e-2;   // TF32: ~10-bit mantissa
#else
static constexpr double kTol = 1e-4;   // strict FP32 (CUBLAS_COMPUTE_32F)
#endif

static int g_fail = 0;
static int g_cases = 0;

static bool run_gemm_case(bool tA, bool tB, int M, int N, int K,
                          float alpha, float beta, int pad, std::mt19937& rng)
{
    // Row-major stored shapes
    const int Ar = tA ? K : M, Ac = tA ? M : K;
    const int Br = tB ? N : K, Bc = tB ? K : N;
    const int lda = Ac + pad, ldb = Bc + pad, ldc = N + pad;

    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> A((size_t)Ar * lda), B((size_t)Br * ldb), C((size_t)M * ldc);
    for (auto& v : A) v = U(rng);
    for (auto& v : B) v = U(rng);
    for (auto& v : C) v = U(rng);
    std::vector<float> Cref = C;

    // CPU reference
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const float a = tA ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k];
                const float b = tB ? B[(size_t)j * ldb + k] : B[(size_t)k * ldb + j];
                acc += (double)a * (double)b;
            }
            const double prev = (beta != 0.f) ? (double)beta * C[(size_t)i * ldc + j] : 0.0;
            Cref[(size_t)i * ldc + j] = (float)((double)alpha * acc + prev);
        }
    }

    // GPU
    CudaPtr<float> dA((int)A.size(), false), dB((int)B.size(), false), dC((int)C.size(), false);
    dA.from_host(A.data(), (int)A.size());
    dB.from_host(B.data(), (int)B.size());
    dC.from_host(C.data(), (int)C.size());
    cuda_gemm_rm(tA, tB, M, N, K, alpha, dA.get(), lda, dB.get(), ldb, beta, dC.get(), ldc);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> G(C.size());
    dC.to_host(G.data(), (int)G.size());

    // Compare: valid region (tolerance) + padding (exactly unchanged)
    bool ok = true;
    double worst = 0.0;
    for (int i = 0; i < M && ok; ++i) {
        for (int j = 0; j < ldc; ++j) {
            const size_t idx = (size_t)i * ldc + j;
            if (j < N) {
                const double err = std::fabs((double)G[idx] - (double)Cref[idx]) /
                                   (1.0 + std::fabs((double)Cref[idx]));
                worst = std::fmax(worst, err);
                if (!(err <= kTol)) { ok = false; break; }
            } else if (G[idx] != C[idx]) {
                ok = false; break;   // padding column chhua gaya
            }
        }
    }
    ++g_cases;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 8)
            printf("  FAIL  tA=%d tB=%d M=%d N=%d K=%d alpha=%.2f beta=%.2f pad=%d (worst rel err=%.3e)\n",
                   (int)tA, (int)tB, M, N, K, alpha, beta, pad, worst);
    }
    return ok;
}

static bool run_colsum_case(int rows, int cols, std::mt19937& rng)
{
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> dY((size_t)rows * cols), db(cols), ref(cols);
    for (auto& v : dY) v = U(rng);
    for (auto& v : db) v = U(rng);
    for (int j = 0; j < cols; ++j) {
        double s = 0.0;
        for (int r = 0; r < rows; ++r) s += dY[(size_t)r * cols + j];
        ref[j] = (float)((double)db[j] + s);
    }

    CudaPtr<float> dY_d((int)dY.size(), false), db_d(cols, false);
    dY_d.from_host(dY.data(), (int)dY.size());
    db_d.from_host(db.data(), cols);
    cuda_colsum_accumulate(dY_d.get(), db_d.get(), rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(cols);
    db_d.to_host(out.data(), cols);

    bool ok = true;
    double worst = 0.0;
    for (int j = 0; j < cols; ++j) {
        const double err = std::fabs((double)out[j] - (double)ref[j]) / (1.0 + std::fabs((double)ref[j]));
        worst = std::fmax(worst, err);
        if (!(err <= kTol)) ok = false;
    }
    ++g_cases;
    if (!ok) {
        ++g_fail;
        printf("  FAIL  colsum rows=%d cols=%d (worst rel err=%.3e)\n", rows, cols, worst);
    }
    return ok;
}

int main()
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("T10 cuBLAS GEMM: no CUDA device found\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("T10 cuBLAS GEMM wrapper | GPU: %s (sm_%d%d) | tol=%.0e (%s)\n",
           prop.name, prop.major, prop.minor, kTol, LOGOS_CUBLAS_TF32 ? "TF32" : "strict FP32");

    std::mt19937 rng(1234);

    struct Shape { int M, N, K; };
    const Shape shapes[] = {
        {37, 53, 29},     // odd, non-multiple-of-anything
        {64, 64, 128},
        {1, 17, 9},       // M=1 edge
        {130, 70, 33},
        {16, 64, 4096},   // long-K (seq reduction jaisa)
        {256, 64, 256},   // DH=64 attention jaisa
    };
    const float ab[][2] = { {1.f, 0.f}, {1.f, 1.f}, {0.5f, 0.75f} };   // {alpha, beta}
    const int pads[] = { 0, 3 };

    for (const auto& s : shapes)
        for (int tA = 0; tA < 2; ++tA)
            for (int tB = 0; tB < 2; ++tB)
                for (const auto& p : ab)
                    for (int pad : pads)
                        run_gemm_case(tA, tB, s.M, s.N, s.K, p[0], p[1], pad, rng);

    // Bias-gradient column sum (ones-buffer grow path bhi: chhota rows pehle, phir bada)
    run_colsum_case(37, 53, rng);
    run_colsum_case(2048, 1024, rng);
    run_colsum_case(4096, 4096, rng);

    // Convenience wrapper: cuda_gemm == cuda_gemm_rm(NN)
    {
        const int M = 48, K = 40, N = 56;
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N), ref((size_t)M * N);
        for (auto& v : hA) v = U(rng);
        for (auto& v : hB) v = U(rng);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double)hA[(size_t)i * K + k] * hB[(size_t)k * N + j];
                ref[(size_t)i * N + j] = (float)acc;
            }
        GPUTensor A = gpu_alloc(M, K), B = gpu_alloc(K, N), C = gpu_alloc(M, N);
        h2d(A, hA.data(), M * K);
        h2d(B, hB.data(), K * N);
        cuda_gemm(A, B, C);
        CUDA_CHECK(cudaDeviceSynchronize());
        d2h(hC.data(), C, M * N);
        bool ok = true;
        for (size_t i = 0; i < hC.size(); ++i)
            if (!(std::fabs((double)hC[i] - (double)ref[i]) / (1.0 + std::fabs((double)ref[i])) <= kTol)) ok = false;
        ++g_cases;
        if (!ok) { ++g_fail; printf("  FAIL  cuda_gemm (NN wrapper)\n"); }
    }

    printf("T10 result: %d/%d cases passed\n", g_cases - g_fail, g_cases);
    printf(g_fail == 0 ? "T10 cuBLAS GEMM: PASS\n" : "T10 cuBLAS GEMM: FAIL\n");
    return g_fail == 0 ? 0 : 1;
}
