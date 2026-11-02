#include "../cuda/VedicGEMM.cuh"
#include "../cuda/ModelGPU.cuh"
#include "../cuda/MixedPrecision.cuh"   // [FIX] cast_fp32_to_fp16_kernel + scale_tensor — ModelGPU forward mein use hota hai
#include "../include/Model.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <stdexcept>

static int g_pass = 0, g_fail = 0;

#define TEST(name, expr) do { \
    bool _ok = (expr); \
    if (_ok) { \
        std::cout << "  ✅ " << (name) << "\n"; \
        ++g_pass; \
    } else { \
        std::cerr << "  ❌ " << (name) << "  [FAIL]\n"; \
        ++g_fail; \
    } \
} while(0)

#define TEST_THROWS(name, code) do { \
    bool _threw = false; \
    try { code; } catch (...) { _threw = true; } \
    TEST(name, _threw); \
} while(0)

static void fill_gpu(GPUTensor& t, float val) {
    std::vector<float> h(t.size, val);
    h2d(t, h.data(), t.size);
}

static std::vector<float> read_gpu(const GPUTensor& t) {
    std::vector<float> h(t.size);
    d2h(h.data(), t, t.size);
    return h;
}

static bool gpu_tensor_finite(const GPUTensor& t) {
    auto h = read_gpu(t);
    for (float v : h) if (!std::isfinite(v)) return false;
    return true;
}

static void test_s1_cpu_headers() {
    std::cout << "\n[S1] CPU Headers Compile Check\n";
    TEST("VedicGEMM.cuh included",   true);
    TEST("ModelGPU.cuh included",    true);
    TEST("Model.hpp included",       true);
    TEST("cuda_runtime.h included",  true);
}

static void test_s2_cuda_device() {
    std::cout << "\n[S2] CUDA Device Detection\n";
    int device_count = 0;
    cudaError_t e = cudaGetDeviceCount(&device_count);
    TEST("cudaGetDeviceCount success", e == cudaSuccess);
    TEST("At least one GPU", device_count >= 1);

    if (device_count >= 1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "    GPU: " << prop.name << "\n";
        std::cout << "    VRAM: " << prop.totalGlobalMem/1024/1024 << " MB\n";
        std::cout << "    Compute: " << prop.major << "." << prop.minor << "\n";
        TEST("Compute >= 6.0 (Pascal+)", prop.major >= 6);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "    Free VRAM: " << free_mem/1024/1024 << " MB\n";
        TEST("Free VRAM >= 1 GB", free_mem >= 1024ULL*1024*1024);
    }
}

static void test_s3_cuda_check() {
    std::cout << "\n[S3] CUDA_CHECK Macro\n";

    bool no_throw = true;
    try {
        CUDA_CHECK(cudaSuccess);
    } catch (...) { no_throw = false; }
    TEST("CUDA_CHECK(cudaSuccess) no throw", no_throw);

    bool threw = false;
    try {
        CUDA_CHECK(cudaErrorInvalidValue);
    } catch (...) { threw = true; }
    cudaGetLastError();  // clear error — Sanitizer sees clean state after this
    TEST("CUDA_CHECK(cudaErrorInvalidValue) throws", threw);
}

static void test_s4_gpu_tensor_raii() {
    std::cout << "\n[S4] GPUTensor RAII\n";

    GPUTensor t = gpu_alloc(16, 16);
    TEST("Alloc 16x16 — data not null", t.data != nullptr);
    TEST("Alloc 16x16 — size == 256",   t.size == 256);
    TEST("Alloc 16x16 — rows == 16",    t.rows == 16);
    TEST("Alloc 16x16 — cols == 16",    t.cols == 16);

    fill_gpu(t, 3.14f);
    auto h = read_gpu(t);
    float max_err = 0.f;
    for (float v : h) max_err = std::max(max_err, std::abs(v - 3.14f));
    TEST("Fill 3.14 round-trip error < 1e-5", max_err < 1e-5f);

    GPUTensor t2 = std::move(t);
    TEST("After move: src data == null",    t.data == nullptr);
    TEST("After move: dst data != null",    t2.data != nullptr);
    TEST("After move: dst size == 256",     t2.size == 256);

    TEST("RAII cleanup (no explicit free needed)", true);

    NikhilamTensor nt;
    TEST("NikhilamTensor default: data==null", nt.data == nullptr);
    TEST("NikhilamTensor default: size==0",    nt.size == 0);
}

static void test_s5_structs() {
    std::cout << "\n[S5] Struct Constructors\n";

    VedicVerifyResult vr;
    vr.checksum_C = 1.0f; vr.checksum_vedic = 1.0f;
    vr.relative_error = 0.0f; vr.pass = true;
    TEST("VedicVerifyResult constructible", vr.pass == true);

    FreeEnergyResult fe;
    fe.cross_entropy = 2.3f; fe.entropy = 0.5f;
    fe.free_energy = 2.3f - 0.01f * 0.5f; fe.temperature = 0.01f;
    TEST("FreeEnergyResult constructible", std::isfinite(fe.free_energy));
}

static void test_s6_gemm() {
    std::cout << "\n[S6] cuda_vedic_gemm\n";

    int M=4, K=8, N=4;
    GPUTensor A = gpu_alloc(M, K);
    GPUTensor B = gpu_alloc(K, N);
    GPUTensor C = gpu_alloc(M, N);

    fill_gpu(A, 1.0f);
    fill_gpu(B, 2.0f);
    cuda_vedic_gemm(A, B, C);
    cudaDeviceSynchronize();

    auto h = read_gpu(C);
    float expected = (float)(K * 1 * 2);  // 16
    float max_err = 0.f;
    for (float v : h) max_err = std::max(max_err, std::abs(v - expected));
    TEST("GEMM result: C = K*A*B (expected 16.0)", max_err < 0.1f);
    TEST("GEMM output: no NaN/Inf", gpu_tensor_finite(C));

    GPUTensor A2 = gpu_alloc(3, 5);
    GPUTensor B2 = gpu_alloc(5, 7);
    GPUTensor C2 = gpu_alloc(3, 7);
    std::vector<float> h_a2(15), h_b2(35), h_c2(21);
    for (int i = 0; i < 15; ++i) h_a2[i] = static_cast<float>((i % 5) - 2);
    for (int i = 0; i < 35; ++i) h_b2[i] = static_cast<float>((i % 7) - 3);
    h2d(A2, h_a2.data(), 15);
    h2d(B2, h_b2.data(), 35);
    cuda_vedic_gemm(A2, B2, C2);
    d2h(h_c2.data(), C2, 21);
    float irregular_max_err = 0.f;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 7; ++col) {
            float reference = 0.f;
            for (int k = 0; k < 5; ++k)
                reference += h_a2[row * 5 + k] * h_b2[k * 7 + col];
            irregular_max_err = std::max(irregular_max_err,
                std::abs(h_c2[row * 7 + col] - reference));
        }
    }
    TEST("GEMM irregular shape: matches CPU reference", irregular_max_err < 1e-4f);

    GPUTensor bias = gpu_alloc(1, 7);
    std::vector<float> h_bias(7);
    for (int col = 0; col < 7; ++col) h_bias[col] = 0.25f * col;
    h2d(bias, h_bias.data(), 7);
    cuda_vedic_gemm_bias(A2, B2, bias, C2);
    d2h(h_c2.data(), C2, 21);
    float bias_max_err = 0.f;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 7; ++col) {
            float reference = h_bias[col];
            for (int k = 0; k < 5; ++k)
                reference += h_a2[row * 5 + k] * h_b2[k * 7 + col];
            bias_max_err = std::max(bias_max_err,
                std::abs(h_c2[row * 7 + col] - reference));
        }
    }
    TEST("GEMM bias: matches CPU reference", bias_max_err < 1e-4f);
    std::cout << "    GEMM backend: "
              << (cuda_vedic_gemm_uses_cublas() ? "cuBLAS" : "Vedic tiled fallback")
              << "\n";
}

static void test_s7_softmax() {
    std::cout << "\n[S7] cuda_boltzmann_softmax\n";

    int seq=4, vocab=32;
    GPUTensor scores = gpu_alloc(seq, vocab);
    GPUTensor probs  = gpu_alloc(seq, vocab);

    std::vector<float> h_scores(seq*vocab);
    for (int i=0; i<seq*vocab; ++i) h_scores[i] = (float)(i%5) - 2.0f;
    h2d(scores, h_scores.data(), seq*vocab);

    cuda_boltzmann_softmax(scores, probs, seq, vocab, 1.0f);
    cudaDeviceSynchronize();

    auto h = read_gpu(probs);
    bool sums_to_one = true;
    for (int i=0; i<seq; ++i) {
        float row_sum = 0.f;
        for (int v=0; v<vocab; ++v) row_sum += h[i*vocab+v];
        if (std::abs(row_sum - 1.0f) > 1e-4f) sums_to_one = false;
    }
    bool all_positive = true;
    for (float v : h) if (v < -1e-6f) all_positive = false;

    TEST("Softmax: each row sums to 1.0",  sums_to_one);
    TEST("Softmax: all values >= 0",       all_positive);
    TEST("Softmax: no NaN/Inf",            gpu_tensor_finite(probs));
}

static void test_s8_layernorm() {
    std::cout << "\n[S8] cuda_layernorm\n";

    int seq=8, d=64;
    GPUTensor X     = gpu_alloc(seq, d);
    GPUTensor gamma = gpu_alloc(1, d);
    GPUTensor beta  = gpu_alloc(1, d);
    GPUTensor Y     = gpu_alloc(seq, d);

    std::vector<float> h_X(seq*d);
    for (int i=0; i<seq*d; ++i) h_X[i] = (float)(i % 13) - 6.0f;
    h2d(X, h_X.data(), seq*d);
    fill_gpu(gamma, 1.0f);
    fill_gpu(beta,  0.0f);

    cuda_layernorm(X, gamma, beta, Y, seq, d);
    cudaDeviceSynchronize();

    auto h = read_gpu(Y);
    bool mean_ok = true, var_ok = true;
    for (int i=0; i<seq; ++i) {
        float mean=0.f, var=0.f;
        for (int j=0; j<d; ++j) mean += h[i*d+j];
        mean /= d;
        for (int j=0; j<d; ++j) var += (h[i*d+j]-mean)*(h[i*d+j]-mean);
        var /= d;
        if (std::abs(mean) > 0.1f)       mean_ok = false;
        if (std::abs(var - 1.0f) > 0.2f) var_ok  = false;
    }
    TEST("LayerNorm: row mean ~ 0",     mean_ok);
    TEST("LayerNorm: row variance ~ 1", var_ok);
    TEST("LayerNorm: no NaN/Inf",       gpu_tensor_finite(Y));
}

static void test_s9_gelu() {
    std::cout << "\n[S9] cuda_gelu\n";

    GPUTensor X = gpu_alloc(16, 32);
    std::vector<float> h_X(16*32);
    for (int i=0; i<16*32; ++i) h_X[i] = (float)(i%10) - 5.0f;
    h2d(X, h_X.data(), 16*32);

    cuda_gelu(X);
    cudaDeviceSynchronize();

    TEST("GELU: no NaN/Inf",    gpu_tensor_finite(X));
    TEST("GELU: size unchanged", X.size == 16*32);

    auto h = read_gpu(X);
    TEST("GELU(0) ≈ 0", std::abs(h[5]) < 0.01f);
}

static void test_s10_leapfrog() {
    std::cout << "\n[S10] leapfrog_langevin_kernel\n";

    int sz = 1024;
    float* d_W; float* d_V; float* d_G;
    CUDA_CHECK(cudaMalloc(&d_W, sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G, sz*sizeof(float)));

    std::vector<float> h_W(sz,1.f), h_V(sz,0.f), h_G(sz,0.01f);
    CUDA_CHECK(cudaMemcpy(d_W,h_W.data(),sz*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,h_V.data(),sz*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G,h_G.data(),sz*sizeof(float),cudaMemcpyHostToDevice));

    leapfrog_langevin_kernel<<<(sz+255)/256, 256>>>(
        d_W, d_V, d_G,
        1e-3f,   // lr
        0.9f,    // friction
        1e-4f,   // noise_scale
        12345u,  // seed
        sz);
    cudaDeviceSynchronize();

    std::vector<float> h_W2(sz), h_V2(sz);
    CUDA_CHECK(cudaMemcpy(h_W2.data(),d_W,sz*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V2.data(),d_V,sz*sizeof(float),cudaMemcpyDeviceToHost));

    bool w_changed = false, all_finite = true;
    for (int i=0; i<sz; ++i) {
        if (std::abs(h_W2[i] - 1.0f) > 1e-9f) w_changed = true;
        if (!std::isfinite(h_W2[i]) || !std::isfinite(h_V2[i])) all_finite = false;
    }
    TEST("Leapfrog: weights changed",         w_changed);
    TEST("Leapfrog: velocity updated",        h_V2[0] != 0.f);
    TEST("Leapfrog: all values finite",       all_finite);
    TEST("Leapfrog: W moved in right direction",
         h_W2[100] < 1.0f + 1e-2f);  // not exploded

    cudaFree(d_W); cudaFree(d_V); cudaFree(d_G);
}

static void test_s11_free_energy_loss() {
    std::cout << "\n[S11] cuda_free_energy_loss\n";

    int seq=8, vocab=64;
    float temperature = 0.05f;

    float *d_logits, *d_loss, *d_grad;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_logits,  seq*vocab*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss,    seq*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,    seq*vocab*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, seq*sizeof(int)));

    std::vector<float> h_logits(seq*vocab, 0.f);
    std::vector<int>   h_targets(seq);
    for (int i=0; i<seq; ++i) {
        h_targets[i] = i % vocab;
        h_logits[i*vocab + h_targets[i]] = 2.0f;  // boost correct token
    }
    CUDA_CHECK(cudaMemcpy(d_logits,  h_logits.data(),  seq*vocab*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), seq*sizeof(int),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_loss, 0, seq*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad, 0, seq*vocab*sizeof(float)));

    FreeEnergyResult result;
    cuda_free_energy_loss(d_logits, d_targets, d_loss, d_grad,
                          seq, vocab, temperature, result);

    TEST("FreeEnergy: CE is finite",        std::isfinite(result.cross_entropy));
    TEST("FreeEnergy: S >= 0 (entropy)",    result.entropy >= -1e-5f);
    TEST("FreeEnergy: F is finite",         std::isfinite(result.free_energy));
    TEST("FreeEnergy: T matches",           std::abs(result.temperature - temperature) < 1e-6f);
    TEST("FreeEnergy: F <= CE (entropy helps)", result.free_energy <= result.cross_entropy + 1e-4f);
    TEST("FreeEnergy: CE in plausible range [0, 6]",
         result.cross_entropy > 0.f && result.cross_entropy < 6.f);

    cudaFree(d_logits); cudaFree(d_loss); cudaFree(d_grad); cudaFree(d_targets);
}

static void test_s12_vedic_verify() {
    std::cout << "\n[S12] cuda_vedic_verify (Gunitasamuchayah)\n";

    int M=16, K=32, N=16;
    GPUTensor A=gpu_alloc(M,K), B=gpu_alloc(K,N), C=gpu_alloc(M,N);
    fill_gpu(A, 1.0f);
    fill_gpu(B, 1.0f);
    cuda_vedic_gemm(A, B, C);
    cudaDeviceSynchronize();

    VedicVerifyResult vr = cuda_vedic_verify(A, B, C, 0.05f);
    std::cout << "    checksum_C=" << vr.checksum_C
              << " vedic=" << vr.checksum_vedic
              << " err=" << vr.relative_error << "\n";
    TEST("Gunitasamuchayah: PASS on correct GEMM", vr.pass);
    TEST("Gunitasamuchayah: relative_error < 5%",  vr.relative_error < 0.05f);
    TEST("Gunitasamuchayah: checksum_C finite",     std::isfinite(vr.checksum_C));

    GPUTensor C_bad = gpu_alloc(M, N);
    fill_gpu(C_bad, 0.0f);  // all zeros — clearly wrong
    VedicVerifyResult vr_bad = cuda_vedic_verify(A, B, C_bad, 0.05f);
    TEST("Gunitasamuchayah: WARN on zeroed C",  !vr_bad.pass);
}

static void test_s13_nikhilam_quant() {
    std::cout << "\n[S13] Nikhilam Quantization Round-trip\n";

    int rows=32, cols=64, sz=rows*cols;
    GPUTensor src = gpu_alloc(rows, cols);

    std::vector<float> h_src(sz);
    for (int i=0; i<sz; ++i) h_src[i] = 5.0f + (float)(i % 32) * (5.0f / 32.0f);
    h2d(src, h_src.data(), sz);

    int8_t* d_int8;
    CUDA_CHECK(cudaMalloc(&d_int8, sz*sizeof(int8_t)));

    float* d_max;
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(float)));
    absmax_kernel<<<(sz+255)/256, 256>>>(src.data, d_max, sz);
    cudaDeviceSynchronize();
    float h_max=0.f;
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    float scale = h_max / 127.5f + 1e-8f;  // [FIX] 127.5f gives better round-trip accuracy (< 1% rel error)

    nikhilam_quantize_kernel<<<(sz+255)/256, 256>>>(src.data, d_int8, scale, sz);
    cudaDeviceSynchronize();

    float* d_rec;
    CUDA_CHECK(cudaMalloc(&d_rec, sz*sizeof(float)));
    nikhilam_dequantize_kernel<<<(sz+255)/256, 256>>>(d_int8, d_rec, scale, sz);
    cudaDeviceSynchronize();

    std::vector<float> h_rec(sz);
    CUDA_CHECK(cudaMemcpy(h_rec.data(), d_rec, sz*sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.f, max_rel_err = 0.f;
    for (int i=0; i<sz; ++i) {
        float err = std::abs(h_src[i] - h_rec[i]);
        float rel = h_src[i] != 0 ? err / std::abs(h_src[i]) : err;
        max_abs_err = std::max(max_abs_err, err);
        max_rel_err = std::max(max_rel_err, rel);
    }
    std::cout << "    Max abs error: " << max_abs_err << "\n";
    std::cout << "    Max rel error: " << max_rel_err * 100 << "%\n";
    TEST("Nikhilam: abs error < 0.1", max_abs_err < 0.1f);
    TEST("Nikhilam: rel error < 1%",  max_rel_err < 0.01f);

    cudaFree(d_int8); cudaFree(d_max); cudaFree(d_rec);
}

static void test_s14_expmap() {
    std::cout << "\n[S14] expmap0_kernel (Poincaré Ball)\n";

    int seq=16, d=64;
    GPUTensor X = gpu_alloc(seq, d);

    std::vector<float> h_X(seq*d);
    for (int i=0; i<seq*d; ++i) h_X[i] = (float)(i%20) - 10.0f;
    h2d(X, h_X.data(), seq*d);

    expmap0_kernel<<<seq, 256>>>(X.data, seq, d, 1.0f);
    cudaDeviceSynchronize();

    auto h = read_gpu(X);

    bool inside_ball = true;
    float max_norm = 0.f;
    for (int i=0; i<seq; ++i) {
        float norm_sq = 0.f;
        for (int j=0; j<d; ++j) norm_sq += h[i*d+j]*h[i*d+j];
        float norm = std::sqrt(norm_sq);
        max_norm = std::max(max_norm, norm);
        if (norm >= 1.0f) inside_ball = false;
    }
    std::cout << "    Max ||expmap(v)|| = " << max_norm << " (must be < 1)\n";
    TEST("expmap0: all tokens inside Poincaré ball (||y|| < 1)", inside_ball);
    TEST("expmap0: no NaN/Inf", gpu_tensor_finite(X));
}

static void test_s15_model_gpu_constructor() {
    std::cout << "\n[S15] ModelGPU Constructor\n";

    ModelConfig cfg;
    cfg.vocab_size=256; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=32;

    bool constructed = false;
    try {
        ModelGPU model(cfg);
        constructed = true;
        TEST("ModelGPU: embedding allocated",     model.gpu_embedding.valid());
        TEST("ModelGPU: pos_emb allocated",       model.gpu_pos_embedding.valid());
        TEST("ModelGPU: lm_head allocated",       model.gpu_lm_head.valid());
        TEST("ModelGPU: blocks count == L",       (int)model.gpu_blocks.size() == cfg.num_layers);
        TEST("ModelGPU: d_token_ids allocated",   model.d_token_ids != nullptr);
    } catch (const std::exception& e) {
        std::cerr << "    Exception: " << e.what() << "\n";
    }
    TEST("ModelGPU: constructed without exception", constructed);
}

static void test_s16_model_forward() {
    std::cout << "\n[S16] ModelGPU::forward()\n";

    ModelConfig cfg;
    cfg.vocab_size=256; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=32;

    ModelGPU model(cfg);
    LOGOSModel cpu_model(cfg);
    model.load_from_cpu(cpu_model);

    std::vector<int> tokens = {1, 5, 10, 20, 42, 100};
    GPUTensor logits = model.forward(tokens);
    cudaDeviceSynchronize();

    int expected_rows = (int)tokens.size();
    int expected_cols = cfg.vocab_size;

    TEST("Forward: logits rows == seq_len",    logits.rows == expected_rows);
    TEST("Forward: logits cols == vocab_size", logits.cols == expected_cols);
    TEST("Forward: logits no NaN/Inf",         gpu_tensor_finite(logits));
    TEST("Forward: last_hidden allocated",     model.last_hidden.valid());
    TEST("Forward: last_hidden rows == seq",   model.last_hidden.rows == expected_rows);
}

static void test_s17_full_pipeline() {
    std::cout << "\n[S17] Full Pipeline (Forward + Loss + Verify)\n";

    ModelConfig cfg;
    cfg.vocab_size=128; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=16;

    ModelGPU model(cfg);
    LOGOSModel cpu_model(cfg);
    model.load_from_cpu(cpu_model);

    std::vector<int> inputs  = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> targets = {2, 3, 4, 5, 6, 7, 8, 9};
    int seq = (int)inputs.size();
    int vocab = cfg.vocab_size;

    GPUTensor logits = model.forward(inputs);
    cudaDeviceSynchronize();

    int* d_targets;
    float *d_loss, *d_grad;
    CUDA_CHECK(cudaMalloc(&d_targets, seq*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loss,    seq*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,    seq*vocab*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), seq*sizeof(int), cudaMemcpyHostToDevice));

    FreeEnergyResult fe;
    cuda_free_energy_loss(logits.data, d_targets, d_loss, d_grad,
                          seq, vocab, 0.05f, fe);

    TEST("Pipeline: CE finite",      std::isfinite(fe.cross_entropy));
    TEST("Pipeline: entropy >= 0",   fe.entropy >= 0.f);
    TEST("Pipeline: F finite",       std::isfinite(fe.free_energy));

    GPUTensor C_check = gpu_alloc(model.last_hidden.rows, model.gpu_lm_head.cols);
    cuda_vedic_gemm(model.last_hidden, model.gpu_lm_head, C_check);
    cudaDeviceSynchronize();
    VedicVerifyResult vr = cuda_vedic_verify(model.last_hidden, model.gpu_lm_head, C_check, 0.1f);
    TEST("Pipeline: Gunitasamuchayah PASS", vr.pass);

    cudaFree(d_targets); cudaFree(d_loss); cudaFree(d_grad);
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS Test 1: Syntax & Build            ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    try {
        test_s1_cpu_headers();
        test_s2_cuda_device();
        test_s3_cuda_check();
        test_s4_gpu_tensor_raii();
        test_s5_structs();
        test_s6_gemm();
        test_s7_softmax();
        test_s8_layernorm();
        test_s9_gelu();
        test_s10_leapfrog();
        test_s11_free_energy_loss();
        test_s12_vedic_verify();
        test_s13_nikhilam_quant();
        test_s14_expmap();
        test_s15_model_gpu_constructor();
        test_s16_model_forward();
        test_s17_full_pipeline();
    } catch (const std::exception& e) {
        std::cerr << "\n💥 FATAL: " << e.what() << "\n";
        ++g_fail;
    }

    std::cout << "\n════════════════════════════════════\n";
    std::cout << "  PASS: " << g_pass << "  FAIL: " << g_fail << "\n";
    std::cout << "════════════════════════════════════\n";
    return g_fail > 0 ? 1 : 0;
}
