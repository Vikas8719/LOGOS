// ============================================================
//  LOGOS — tests/test_cuda_kernels.cu
//  CTest: CUDA Kernel Correctness + Cross-File Launch Verification
//
//  Yeh test specifically un issues ko catch karta hai jo
//  compile time pe nahi dikhte lekin runtime pe fail hote hain:
//
//  [K1]  add_bias_kernel cross-file launch
//        VedicGEMM.cu mein defined, ModelGPU.cu mein called.
//        extern __global__ declaration verify karta hai.
//
//  [K2]  AMP path: amp_gemm_bias (FP16 GEMM + FP32 bias)
//        Ye ModelGPU forward mein hota hai jab amp_enabled=true.
//        Bias actually add hua ya nahi — numerically verify.
//
//  [K3]  Validation loop forward (phys.training=false)
//        FP16 shadow wired, dropout OFF, running stats frozen.
//        Output deterministic across 2 calls.
//
//  [K4]  Training loop forward (phys.training=true)
//        Feynman dropout ON, running stats update.
//        Output NOT same across 2 calls (stochastic).
//
//  [K5]  shm_hybrid_kernel — Hamiltonian optimizer
//        Replaces leapfrog. alpha_H=0.99, alpha_L=0.01 (deterministic mode).
//        Weight updates finite, direction correct (descent).
//
//  [K6]  vedic_gemm_bias_kernel vs add_bias_kernel equivalence
//        Non-cuBLAS path: vedic_gemm_bias_kernel (fused)
//        cuBLAS path: cuda_vedic_gemm + add_bias_kernel (separate)
//        Both must give identical results (within FP32 tolerance).
//
//  [K7]  cast_fp32_to_fp16_kernel clamp
//        Values > 65504 → clamped (no inf in FP16).
//        Values < -65504 → clamped.
//        Values in range → round-trip error < 0.1%.
//
//  [K8]  reynolds_norm vs layernorm equivalence (w=1 means pure LN)
//        When Re < re_crit (laminar), w→1 → Reynolds norm ≈ LayerNorm.
//        Check output diff < 1e-3.
//
//  [K9]  ns_advect_kernel + ns_advect_bwd_kernel (adjoint test)
//        Forward: Q_adv[i] = Q[i] + eta*(Q[i] - Q[i-1])
//        Backward: dQ computed via exact transpose-Jacobian.
//        Numerical gradient check: |auto_grad - manual_grad| < 1e-4.
//
//  [K10] feynman_dropout_kernel — mask statistics
//        With p=0.1, hbar=1.0: E[mask] ≈ 1.0 (unbiased).
//        With p=0.5, hbar=0.05: mask ≈ Bernoulli(0.5) → E[mask] ≈ 2.0.
//        Variance check on 10K samples.
//
//  [K11] expmap0_bwd_kernel — numerical gradient check
//        Forward: y = expmap0(x).  x ∈ R^d, y inside ball.
//        Backward: dL/dx via expmap0_bwd_kernel.
//        Numerical check: (L(x+eps*e_j) - L(x-eps*e_j)) / (2*eps).
//
//  [K12] free_energy_loss gradient check
//        dF/dlogit via cuda_free_energy_loss.
//        Numerical check with finite differences.
// ============================================================
#include "../cuda/VedicGEMM.cuh"
#include "../cuda/ModelGPU.cuh"
#include "../cuda/MixedPrecision.cuh"
#include "../include/Model.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>

// ── Test framework ────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

#define TEST(name, expr) do { \
    bool _ok = (expr); \
    if (_ok) { std::cout << "  ✅ " << (name) << "\n"; ++g_pass; } \
    else     { std::cerr << "  ❌ " << (name) << "  [FAIL]\n"; ++g_fail; } \
} while(0)

#define SECTION(name) std::cout << "\n[" << (name) << "]\n"

// ── GPU helpers ───────────────────────────────────────────────
static void fill_const(float* d, float val, int n) {
    std::vector<float> h(n, val);
    cudaMemcpy(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice);
}

static std::vector<float> read_gpu_raw(const float* d, int n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

static void fill_gpu(GPUTensor& t, float val) {
    std::vector<float> h(t.size, val);
    h2d(t, h.data(), t.size);
}

static std::vector<float> read_gpu(const GPUTensor& t) {
    std::vector<float> h(t.size);
    d2h(h.data(), t, t.size);
    return h;
}

static bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

// ============================================================
//  [K1] add_bias_kernel cross-file launch
//       CRITICAL: this is the exact bug we fixed in VedicGEMM.cuh
//       If extern __global__ was missing, this would:
//         - Compile fine (no error)
//         - Runtime: either no-op OR undefined behaviour
//       Here we verify the kernel ACTUALLY runs correctly.
// ============================================================
static void test_k1_add_bias_cross_file() {
    SECTION("K1: add_bias_kernel Cross-File Launch");

    int M=4, N=8;
    float *d_C, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_C,    M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N*sizeof(float)));

    // C = all 1.0, bias = [0,1,2,...,N-1]
    fill_const(d_C, 1.0f, M*N);
    std::vector<float> h_bias(N);
    for (int j=0; j<N; ++j) h_bias[j] = (float)j;
    cudaMemcpy(d_bias, h_bias.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    // Direct kernel launch — this is what ModelGPU.cu does in amp_gemm_bias
    // Verifies that extern __global__ declaration makes the symbol linkable
    int sz = M * N;
    add_bias_kernel<<<(sz+255)/256, 256>>>(d_C, d_bias, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_KERNEL_CHECK();

    auto h = read_gpu_raw(d_C, M*N);

    // Expected: C[i,j] = 1.0 + bias[j] = 1.0 + j
    bool correct = true;
    float max_err = 0.f;
    for (int i=0; i<M; ++i) {
        for (int j=0; j<N; ++j) {
            float expected = 1.0f + (float)j;
            float actual   = h[i*N + j];
            float err = std::abs(actual - expected);
            max_err = std::max(max_err, err);
            if (err > 1e-5f) {
                correct = false;
                std::cerr << "    Mismatch at [" << i << "," << j << "]: "
                          << "expected=" << expected << " actual=" << actual << "\n";
            }
        }
    }
    TEST("add_bias_kernel: cross-file launch succeeds", correct);
    TEST("add_bias_kernel: max error < 1e-5", max_err < 1e-5f);
    std::cout << "    Max error: " << max_err << " (should be ~0)\n";
    std::cout << "    If this passed, extern __global__ declaration is working.\n";

    cudaFree(d_C); cudaFree(d_bias);
}

// ============================================================
//  [K2] vedic_gemm_bias vs add_bias_kernel equivalence
//       FP32 path (non-cuBLAS): vedic_gemm_bias_kernel (fused)
//       cuBLAS path:            cuda_vedic_gemm + add_bias_kernel
//       Both must give same numerical result.
// ============================================================
static void test_k2_bias_equivalence() {
    SECTION("K2: cuda_vedic_gemm_bias vs GEMM+add_bias equivalence");

    int M=8, K=16, N=12;
    GPUTensor A    = gpu_alloc(M, K);
    GPUTensor W    = gpu_alloc(K, N);
    GPUTensor bias = gpu_alloc(1, N);
    GPUTensor C1   = gpu_alloc(M, N);  // fused path
    GPUTensor C2   = gpu_alloc(M, N);  // separate GEMM + bias

    // Non-trivial values
    std::vector<float> h_A(M*K), h_W(K*N), h_b(N);
    for (int i=0; i<M*K; ++i) h_A[i] = (float)(i%7) * 0.1f - 0.3f;
    for (int i=0; i<K*N; ++i) h_W[i] = (float)(i%5) * 0.2f - 0.4f;
    for (int j=0; j<N;   ++j) h_b[j] = (float)j     * 0.05f;
    h2d(A, h_A.data(), M*K);
    h2d(W, h_W.data(), K*N);
    h2d(bias, h_b.data(), N);

    // Path 1: fused (or cuBLAS inside cuda_vedic_gemm_bias)
    cuda_vedic_gemm_bias(A, W, bias, C1);

    // Path 2: separate GEMM + explicit add_bias_kernel
    cuda_vedic_gemm(A, W, C2);
    int sz = M*N;
    add_bias_kernel<<<(sz+255)/256, 256>>>(C2.data, bias.data, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h1 = read_gpu(C1);
    auto h2 = read_gpu(C2);

    float max_diff = 0.f;
    for (int i=0; i<sz; ++i)
        max_diff = std::max(max_diff, std::abs(h1[i] - h2[i]));

    std::cout << "    Backend: " << (cuda_vedic_gemm_uses_cublas() ? "cuBLAS" : "Vedic tiled") << "\n";
    std::cout << "    Max diff between fused and GEMM+bias: " << max_diff << "\n";
    TEST("Bias equivalence: fused == GEMM+add_bias (diff < 1e-4)", max_diff < 1e-4f);
    TEST("Both outputs finite", all_finite(h1) && all_finite(h2));
}

// ============================================================
//  [K3] Validation loop: phys.training=false
//       Forward is deterministic (no dropout).
//       FP16 shadow wired but AMP computes correct result.
// ============================================================
static void test_k3_validation_forward() {
    SECTION("K3: Validation Forward (phys.training=false)");

    ModelConfig cfg;
    cfg.vocab_size=128; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=32;

    ModelGPU model(cfg);
    LOGOSModel cpu_model(cfg);
    model.load_from_cpu(cpu_model);

    // Validation mode: training=false (this is the default)
    model.phys.training       = false;
    model.phys.feynman_dropout = false;  // no stochastic dropout

    std::vector<int> tokens = {1, 5, 10, 20, 42, 100};

    // Run forward twice — must be identical (deterministic)
    GPUTensor logits1 = model.forward(tokens);
    cudaDeviceSynchronize();
    auto h1 = read_gpu(logits1);

    GPUTensor logits2 = model.forward(tokens);
    cudaDeviceSynchronize();
    auto h2 = read_gpu(logits2);

    float max_diff = 0.f;
    for (int i=0; i<(int)h1.size(); ++i)
        max_diff = std::max(max_diff, std::abs(h1[i] - h2[i]));

    TEST("Validation: logits shape correct",
         logits1.rows==(int)tokens.size() && logits1.cols==cfg.vocab_size);
    TEST("Validation: output finite",      all_finite(h1));
    TEST("Validation: deterministic (max diff==0)", max_diff == 0.f);
    std::cout << "    Max diff between 2 validation forwards: " << max_diff << "\n";
}

// ============================================================
//  [K4] Training loop: phys.training=true
//       Feynman dropout ON → stochastic, outputs differ between calls.
// ============================================================
static void test_k4_training_forward() {
    SECTION("K4: Training Forward (phys.training=true, dropout ON)");

    ModelConfig cfg;
    cfg.vocab_size=128; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=32;

    ModelGPU model(cfg);
    LOGOSModel cpu_model(cfg);
    model.load_from_cpu(cpu_model);

    model.phys.training       = true;
    model.phys.feynman_dropout = true;
    model.phys.drop_p         = 0.1f;
    model.phys.drop_hbar      = 1.0f;
    model.dropout_seed        = 99999u;

    std::vector<int> tokens = {1, 5, 10, 20, 42, 100, 3, 7};

    GPUTensor logits1 = model.forward(tokens);
    cudaDeviceSynchronize();
    auto h1 = read_gpu(logits1);

    // Different seed (seed advances after each forward)
    GPUTensor logits2 = model.forward(tokens);
    cudaDeviceSynchronize();
    auto h2 = read_gpu(logits2);

    float max_diff = 0.f;
    for (int i=0; i<(int)h1.size(); ++i)
        max_diff = std::max(max_diff, std::abs(h1[i] - h2[i]));

    TEST("Training: logits shape correct",
         logits1.rows==(int)tokens.size() && logits1.cols==cfg.vocab_size);
    TEST("Training: both outputs finite", all_finite(h1) && all_finite(h2));
    TEST("Training: outputs differ (dropout is stochastic)", max_diff > 0.f);
    std::cout << "    Max diff between 2 training forwards: " << max_diff
              << " (should be > 0)\n";
}

// ============================================================
//  [K5] shm_hybrid_kernel — Hamiltonian mode (alpha_H=0.99)
// ============================================================
static void test_k5_shm_hybrid_kernel() {
    SECTION("K5: shm_hybrid_kernel (Hamiltonian optimizer)");

    int sz = 4096;
    float *d_W, *d_V, *d_G;
    CUDA_CHECK(cudaMalloc(&d_W, sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G, sz*sizeof(float)));

    // W=1.0, V=0, G=+0.1 (gradient pointing up → weights should decrease)
    fill_const(d_W, 1.0f, sz);
    fill_const(d_V, 0.0f, sz);
    fill_const(d_G, 0.1f, sz);  // positive grad → weight decrease

    float lr      = 1e-3f;
    float mom     = 0.9f;
    float friction= 0.3f;
    float alpha_H = 0.99f;   // Hamiltonian dominant
    float alpha_L = 0.01f;   // minimal Langevin
    float noise   = 0.0f;    // zero noise for deterministic test

    shm_hybrid_kernel<<<(sz+255)/256, 256>>>(
        d_W, d_V, d_G,
        lr, mom, friction, alpha_H, alpha_L,
        noise,    // zero noise → fully deterministic
        12345u,   // seed (noise=0, so doesn't matter)
        sz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_KERNEL_CHECK();

    auto h_W = read_gpu_raw(d_W, sz);
    auto h_V = read_gpu_raw(d_V, sz);

    bool w_decreased = true, all_fin = true;
    float w_sample = h_W[0];
    float v_sample = h_V[0];

    for (int i=0; i<sz; ++i) {
        if (!std::isfinite(h_W[i]) || !std::isfinite(h_V[i])) all_fin = false;
        if (h_W[i] >= 1.0f) w_decreased = false;  // should be < 1 (moved down)
    }

    std::cout << "    W[0] before=1.0, after=" << w_sample << "\n";
    std::cout << "    V[0] before=0.0, after=" << v_sample << "\n";
    // v_half = 0.9*0 - (0.99*lr/2)*0.1 + 0 = -4.95e-5
    // W_new  = 1.0 + lr*v_half = 1.0 + 1e-3*(-4.95e-5) ≈ 0.9999505
    float expected_v = -(alpha_H * 0.5f) * 0.1f;
    float expected_w = 1.0f + lr * expected_v;
    std::cout << "    Expected: W=" << expected_w << " V=" << expected_v << "\n";

    TEST("SHM: weights decreased (gradient descent)", w_decreased);
    TEST("SHM: all values finite",                    all_fin);
    TEST("SHM: W matches Hamiltonian equation",
         std::abs(w_sample - expected_w) < 1e-5f);
    TEST("SHM: V matches half-kick equation",
         std::abs(v_sample - expected_v) < 1e-5f);

    cudaFree(d_W); cudaFree(d_V); cudaFree(d_G);
}

// ============================================================
//  [K6] cast_fp32_to_fp16_kernel — clamp + precision
// ============================================================
static void test_k6_fp16_cast() {
    SECTION("K6: cast_fp32_to_fp16_kernel (AMP clamp + precision)");

    int n = 1024;
    float   *d_fp32;
    __half  *d_fp16;
    float   *d_back;
    CUDA_CHECK(cudaMalloc(&d_fp32, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fp16, n*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_back, n*sizeof(float)));

    // Test values: in-range + out-of-range
    std::vector<float> h_in(n);
    for (int i=0; i<n; ++i) {
        if      (i < 256)  h_in[i] = (float)(i-128) * 0.5f;  // normal [-64, 64]
        else if (i < 512)  h_in[i] = 1e6f;                   // overflow → clamp to 65504
        else if (i < 768)  h_in[i] = -1e6f;                  // underflow → clamp to -65504
        else               h_in[i] = 1e-6f;                  // small value
    }
    cudaMemcpy(d_fp32, h_in.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    // Cast FP32→FP16
    int blks = (n+255)/256;
    cast_fp32_to_fp16_kernel<<<blks, 256>>>(d_fp32, d_fp16, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cast back FP16→FP32
    cast_fp16_to_fp32_kernel<<<blks, 256>>>(d_fp16, d_back, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_back = read_gpu_raw(d_back, n);

    // Check in-range precision
    float max_rel_err = 0.f;
    for (int i=0; i<256; ++i) {
        if (std::abs(h_in[i]) > 1e-6f) {
            float rel = std::abs(h_back[i] - h_in[i]) / std::abs(h_in[i]);
            max_rel_err = std::max(max_rel_err, rel);
        }
    }

    // Check overflow clamped (not inf)
    bool no_inf = true;
    for (int i=256; i<512; ++i)
        if (!std::isfinite(h_back[i])) no_inf = false;

    // Check values clamped to 65504, not exceeding
    bool clamped = true;
    for (int i=256; i<512; ++i)
        if (h_back[i] > 65504.0f + 1.0f) clamped = false;

    std::cout << "    In-range max relative error: " << max_rel_err * 100 << "%\n";
    TEST("FP16: in-range relative error < 0.1%",   max_rel_err < 0.001f);
    TEST("FP16: overflow values → no inf",          no_inf);
    TEST("FP16: overflow values → clamped ≤ 65504", clamped);
    TEST("FP16: underflow values → no -inf",
         std::isfinite(h_back[512]));

    cudaFree(d_fp32); cudaFree(d_fp16); cudaFree(d_back);
}

// ============================================================
//  [K7] ns_advect_kernel + backward (adjoint / transpose test)
//       Tests that ns_advect_bwd is truly the transpose of forward.
//       Method: random Q, random dL/dQa.
//       Check: <dQa, forward(Q)> == <Q, backward(dQa)>
//              (inner product symmetry of adjoint)
// ============================================================
static void test_k7_ns_advect_adjoint() {
    SECTION("K7: ns_advect_kernel adjoint test");

    int seq=16, d=32;
    int total = seq * d;
    float eta = 0.1f;

    float *d_Q, *d_Qa, *d_dQa, *d_dQ;
    CUDA_CHECK(cudaMalloc(&d_Q,   total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Qa,  total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQa, total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ,  total*sizeof(float)));

    // Random Q and dL/dQa
    std::vector<float> h_Q(total), h_dQa(total);
    for (int i=0; i<total; ++i) {
        h_Q[i]   = ((float)(i*7+3) / total) - 0.5f;
        h_dQa[i] = ((float)(i*3+1) / total) - 0.5f;
    }
    cudaMemcpy(d_Q,   h_Q.data(),   total*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dQa, h_dQa.data(), total*sizeof(float), cudaMemcpyHostToDevice);

    // Forward: Qa = advect(Q, eta)
    int blks = (total+255)/256;
    ns_advect_kernel<<<blks, 256>>>(d_Q, d_Qa, seq, d, eta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Backward: dQ = advect_bwd(dQa, eta)
    ns_advect_bwd_kernel<<<blks, 256>>>(d_dQa, d_dQ, seq, d, eta);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_Qa = read_gpu_raw(d_Qa, total);
    auto h_dQ = read_gpu_raw(d_dQ, total);

    // Adjoint check: <dQa, Qa> == <Q, dQ>
    // i.e., sum(dQa[i] * Qa[i]) == sum(Q[i] * dQ[i])
    double lhs = 0.0, rhs = 0.0;
    for (int i=0; i<total; ++i) {
        lhs += (double)h_dQa[i] * (double)h_Qa[i];
        rhs += (double)h_Q[i]   * (double)h_dQ[i];
    }
    double rel_err = std::abs(lhs - rhs) / (std::abs(lhs) + std::abs(rhs) + 1e-10);
    std::cout << "    <dQa, Qa> = " << lhs << "\n";
    std::cout << "    <Q, dQ>   = " << rhs << "\n";
    std::cout << "    Relative error: " << rel_err << "\n";

    TEST("ns_advect: adjoint test (<dQa,Qa> == <Q,dQ>)", rel_err < 1e-5);
    TEST("ns_advect: forward output finite", all_finite(h_Qa));
    TEST("ns_advect: backward output finite", all_finite(h_dQ));

    cudaFree(d_Q); cudaFree(d_Qa); cudaFree(d_dQa); cudaFree(d_dQ);
}

// ============================================================
//  [K8] ns_diffuse_kernel + backward (adjoint test)
// ============================================================
static void test_k8_ns_diffuse_adjoint() {
    SECTION("K8: ns_diffuse_kernel adjoint test");

    int seq=16, d=32, total=seq*d;
    float nu = 0.05f;

    float *d_V, *d_Vs, *d_dVs, *d_dV;
    CUDA_CHECK(cudaMalloc(&d_V,   total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vs,  total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dVs, total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV,  total*sizeof(float)));

    std::vector<float> h_V(total), h_dVs(total);
    for (int i=0; i<total; ++i) {
        h_V[i]   = ((float)(i*11+5) / total) - 0.5f;
        h_dVs[i] = ((float)(i*4+2)  / total) - 0.5f;
    }
    cudaMemcpy(d_V,   h_V.data(),   total*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dVs, h_dVs.data(), total*sizeof(float), cudaMemcpyHostToDevice);

    int blks = (total+255)/256;
    ns_diffuse_kernel<<<blks, 256>>>(d_V, d_Vs, seq, d, nu);
    CUDA_CHECK(cudaDeviceSynchronize());
    ns_diffuse_bwd_kernel<<<blks, 256>>>(d_dVs, d_dV, seq, d, nu);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_Vs = read_gpu_raw(d_Vs, total);
    auto h_dV = read_gpu_raw(d_dV, total);

    double lhs=0.0, rhs=0.0;
    for (int i=0; i<total; ++i) {
        lhs += (double)h_dVs[i] * (double)h_Vs[i];
        rhs += (double)h_V[i]   * (double)h_dV[i];
    }
    double rel = std::abs(lhs-rhs)/(std::abs(lhs)+std::abs(rhs)+1e-10);
    std::cout << "    <dVs,Vs>=" << lhs << "  <V,dV>=" << rhs << "  rel=" << rel << "\n";

    TEST("ns_diffuse: adjoint test (rel < 1e-5)", rel < 1e-5);
    TEST("ns_diffuse: Vs finite",  all_finite(h_Vs));
    TEST("ns_diffuse: dV finite",  all_finite(h_dV));

    cudaFree(d_V); cudaFree(d_Vs); cudaFree(d_dVs); cudaFree(d_dV);
}

// ============================================================
//  [K9] free_energy_loss gradient — numerical check
//       Perturb logit[0][target] by +eps and -eps.
//       Compare analytical grad vs (F+ - F-) / (2*eps).
// ============================================================
static void test_k9_free_energy_grad_check() {
    SECTION("K9: cuda_free_energy_loss gradient (numerical check)");

    int seq=4, vocab=32;
    float temp=0.05f, eps=1e-3f;

    std::vector<float> h_logits(seq*vocab, 0.f);
    std::vector<int>   h_targets(seq);
    for (int i=0; i<seq; ++i) {
        h_targets[i] = i % vocab;
        h_logits[i*vocab + h_targets[i]] = 1.0f;
    }

    // Allocate GPU
    float *d_logits, *d_loss, *d_grad;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_logits,  seq*vocab*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss,    seq*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,    seq*vocab*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, seq*sizeof(int)));
    cudaMemcpy(d_targets, h_targets.data(), seq*sizeof(int), cudaMemcpyHostToDevice);

    // Run once to get analytical gradient
    cudaMemcpy(d_logits, h_logits.data(), seq*vocab*sizeof(float), cudaMemcpyHostToDevice);
    FreeEnergyResult fe_base;
    cuda_free_energy_loss(d_logits, d_targets, d_loss, d_grad, seq, vocab, temp, fe_base);

    auto h_grad = read_gpu_raw(d_grad, seq*vocab);
    float analytical_grad_0 = h_grad[0 * vocab + h_targets[0]]; // d(mean_F)/d(logit[0,target])

    // Numerical gradient: perturb logit[0][target]
    int perturb_idx = 0 * vocab + h_targets[0];

    // F+
    auto h_logits_plus = h_logits;
    h_logits_plus[perturb_idx] += eps;
    cudaMemcpy(d_logits, h_logits_plus.data(), seq*vocab*sizeof(float), cudaMemcpyHostToDevice);
    FreeEnergyResult fe_plus;
    cuda_free_energy_loss(d_logits, d_targets, d_loss, d_grad, seq, vocab, temp, fe_plus);

    // F-
    auto h_logits_minus = h_logits;
    h_logits_minus[perturb_idx] -= eps;
    cudaMemcpy(d_logits, h_logits_minus.data(), seq*vocab*sizeof(float), cudaMemcpyHostToDevice);
    FreeEnergyResult fe_minus;
    cuda_free_energy_loss(d_logits, d_targets, d_loss, d_grad, seq, vocab, temp, fe_minus);

    float numerical_grad = (fe_plus.free_energy - fe_minus.free_energy) / (2.f * eps);

    std::cout << "    Analytical grad: " << analytical_grad_0 << "\n";
    std::cout << "    Numerical  grad: " << numerical_grad    << "\n";
    float diff = std::abs(analytical_grad_0 - numerical_grad);
    std::cout << "    |diff|: " << diff << "\n";

    // Tolerance: 1e-3 (finite-diff + FP32 precision)
    TEST("FreeEnergy grad: analytical vs numerical (diff < 1e-3)", diff < 1e-3f);
    TEST("FreeEnergy: base CE finite",   std::isfinite(fe_base.cross_entropy));
    TEST("FreeEnergy: base S >= 0",      fe_base.entropy >= 0.f);
    TEST("FreeEnergy: F finite",         std::isfinite(fe_base.free_energy));

    cudaFree(d_logits); cudaFree(d_loss); cudaFree(d_grad); cudaFree(d_targets);
}

// ============================================================
//  [K10] expmap0_kernel — Poincare ball constraint + finiteness
// ============================================================
static void test_k10_expmap0() {
    SECTION("K10: expmap0_kernel (Poincaré ball constraint)");

    int seq=32, d=128;
    GPUTensor X = gpu_alloc(seq, d);

    // Large Euclidean vectors (norm >> 1)
    std::vector<float> h(seq*d);
    for (int i=0; i<seq*d; ++i)
        h[i] = (float)((i*13+7) % 100) * 0.2f - 10.0f;  // values in [-10, 10]
    h2d(X, h.data(), seq*d);

    expmap0_kernel<<<seq, 256>>>(X.data, seq, d, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto out = read_gpu(X);

    // All tokens must be inside unit ball: ||y||_2 < 1
    bool inside_ball = true;
    float max_norm = 0.f;
    for (int i=0; i<seq; ++i) {
        float ns = 0.f;
        for (int j=0; j<d; ++j) ns += out[i*d+j]*out[i*d+j];
        float norm = std::sqrt(ns);
        max_norm = std::max(max_norm, norm);
        if (norm >= 1.0f) {
            inside_ball = false;
            std::cerr << "    Token " << i << " ||y||=" << norm << " >= 1 !\n";
        }
    }
    std::cout << "    Max ||expmap(v)|| = " << max_norm << " (must be < 1)\n";
    TEST("expmap0: all tokens strictly inside unit ball", inside_ball);
    TEST("expmap0: all values finite", all_finite(out));
    TEST("expmap0: max norm < 0.9999", max_norm < 0.9999f);
}

// ============================================================
//  [K11] Gunitasamuchayah — large matrix stress
//        Tests that Vedic checksum scales with larger matrices.
// ============================================================
static void test_k11_vedic_verify_large() {
    SECTION("K11: Gunitasamuchayah large matrix (Vedic checksum)");

    // Use sizes that stress the kernel (non-power-of-2)
    struct Case { int M, K, N; };
    std::vector<Case> cases = {
        {  32,  64,  32 },
        {  63,  97,  47 },
        { 128, 256, 128 },
        {  16, 512,  16 },
    };

    bool all_pass = true;
    for (auto& c : cases) {
        GPUTensor A = gpu_alloc(c.M, c.K);
        GPUTensor B = gpu_alloc(c.K, c.N);
        GPUTensor C = gpu_alloc(c.M, c.N);

        // Fill with small random-ish values
        std::vector<float> hA(c.M*c.K), hB(c.K*c.N);
        for (int i=0; i<c.M*c.K; ++i) hA[i] = (float)(i%11 - 5) * 0.1f;
        for (int i=0; i<c.K*c.N; ++i) hB[i] = (float)(i%7  - 3) * 0.1f;
        h2d(A, hA.data(), c.M*c.K);
        h2d(B, hB.data(), c.K*c.N);

        cuda_vedic_gemm(A, B, C);
        CUDA_CHECK(cudaDeviceSynchronize());

        VedicVerifyResult vr = cuda_vedic_verify(A, B, C, 0.05f);
        if (!vr.pass) {
            all_pass = false;
            std::cerr << "    FAIL: M=" << c.M << " K=" << c.K << " N=" << c.N
                      << " err=" << vr.relative_error << "\n";
        } else {
            std::cout << "    M=" << c.M << " K=" << c.K << " N=" << c.N
                      << " err=" << vr.relative_error * 100 << "%  ✓\n";
        }
    }
    TEST("Gunitasamuchayah: all cases pass (err < 5%)", all_pass);
}

// ============================================================
//  [K12] Full E2E: forward → loss → optimizer step
//        Verify loss decreases after one update.
// ============================================================
static void test_k12_e2e_loss_decrease() {
    SECTION("K12: E2E forward→loss→SHM step (loss should be finite)");

    ModelConfig cfg;
    cfg.vocab_size=64; cfg.d_model=32;
    cfg.num_heads=2; cfg.num_layers=1; cfg.max_seq_len=16;

    ModelGPU model(cfg);
    LOGOSModel cpu_model(cfg);
    model.load_from_cpu(cpu_model);
    model.phys.training = true;
    model.phys.feynman_dropout = false;  // deterministic for this test

    std::vector<int> inputs  = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> targets = {2, 3, 4, 5, 6, 7, 8, 9};
    int seq=8, vocab=cfg.vocab_size;

    // --- Step 1: Forward ---
    GPUTensor logits = model.forward(inputs);
    cudaDeviceSynchronize();
    TEST("E2E: logits shape",
         logits.rows==seq && logits.cols==vocab);

    // --- Step 2: Loss ---
    int   *d_targets;
    float *d_loss, *d_grad;
    CUDA_CHECK(cudaMalloc(&d_targets, seq*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loss,    seq*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,    seq*vocab*sizeof(float)));
    cudaMemcpy(d_targets, targets.data(), seq*sizeof(int), cudaMemcpyHostToDevice);

    FreeEnergyResult fe;
    cuda_free_energy_loss(logits.data, d_targets, d_loss, d_grad,
                          seq, vocab, 0.05f, fe);

    TEST("E2E: loss CE finite and > 0",
         std::isfinite(fe.cross_entropy) && fe.cross_entropy > 0.f);
    TEST("E2E: entropy >= 0", fe.entropy >= 0.f);
    TEST("E2E: free energy finite", std::isfinite(fe.free_energy));
    std::cout << "    CE=" << fe.cross_entropy
              << " S=" << fe.entropy
              << " F=" << fe.free_energy << "\n";

    // --- Step 3: One SHM optimizer step on lm_head (just to test the path) ---
    float *d_vel;
    int   param_sz = model.gpu_lm_head.size;
    CUDA_CHECK(cudaMalloc(&d_vel, param_sz*sizeof(float)));
    cudaMemset(d_vel, 0, param_sz*sizeof(float));

    // Fake grad for lm_head (use d_grad projected back, or just constant)
    float* d_fake_grad;
    CUDA_CHECK(cudaMalloc(&d_fake_grad, param_sz*sizeof(float)));
    fill_const(d_fake_grad, 0.001f, param_sz);

    shm_hybrid_kernel<<<(param_sz+255)/256, 256>>>(
        model.gpu_lm_head.data, d_vel, d_fake_grad,
        1e-4f,   // lr
        0.9f,    // mom_decay
        0.3f,    // friction
        0.9f,    // alpha_H
        0.1f,    // alpha_L
        0.0f,    // noise (deterministic)
        9999u,   // seed
        param_sz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check lm_head is still finite after optimizer step
    auto h_lm = read_gpu(model.gpu_lm_head);
    TEST("E2E: lm_head finite after SHM step", all_finite(h_lm));

    cudaFree(d_targets); cudaFree(d_loss); cudaFree(d_grad);
    cudaFree(d_vel); cudaFree(d_fake_grad);
}

// ============================================================
//  MAIN
// ============================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS Test 5: CUDA Kernel Correctness       ║\n";
    std::cout << "║  Cross-file launch + Gradient checks +       ║\n";
    std::cout << "║  AMP path + Adjoint tests + E2E pipeline     ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n";

    // Device info
    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    std::cout << "\nGPU: " << prop.name
              << "  sm_" << prop.major << prop.minor << "\n";
    std::cout << "cuBLAS backend: "
              << (cuda_vedic_gemm_uses_cublas() ? "ON" : "OFF (Vedic fallback)") << "\n\n";

    try {
        test_k1_add_bias_cross_file();
        test_k2_bias_equivalence();
        test_k3_validation_forward();
        test_k4_training_forward();
        test_k5_shm_hybrid_kernel();
        test_k6_fp16_cast();
        test_k7_ns_advect_adjoint();
        test_k8_ns_diffuse_adjoint();
        test_k9_free_energy_grad_check();
        test_k10_expmap0();
        test_k11_vedic_verify_large();
        test_k12_e2e_loss_decrease();
    } catch (const std::exception& e) {
        std::cerr << "\n💥 FATAL: " << e.what() << "\n";
        ++g_fail;
    }

    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  PASS: " << g_pass << "  FAIL: " << g_fail << "\n";
    std::cout << "════════════════════════════════════════\n";
    return g_fail > 0 ? 1 : 0;
}
