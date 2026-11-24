#pragma once
// Shared helpers for the new GPU test executables (test_gradcheck, test_model_reference, test_bwd_kernels).
#include "../cuda/CublasGemm.cuh"
#include "../cuda/ModelGPU.cuh"
#include "../cuda/BackwardGPU.cuh"
#include "../include/Model.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace gt {

inline int g_pass = 0;
inline int g_fail = 0;

}  // namespace gt

#define GT_SECTION(name) std::printf("\n== %s ==\n", (name))

#define GT_CHECK(cond, msg)                                                              \
    do {                                                                                 \
        if (cond) { std::printf("  PASS: %s\n", (msg)); ++gt::g_pass; }                  \
        else { std::fprintf(stderr, "  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
               ++gt::g_fail; }                                                           \
    } while (0)

#define GT_CHECKF(cond, ...)                                                             \
    do {                                                                                 \
        char _gt_buf[512];                                                               \
        std::snprintf(_gt_buf, sizeof(_gt_buf), __VA_ARGS__);                            \
        GT_CHECK(cond, _gt_buf);                                                         \
    } while (0)

namespace gt {

// tiny deterministic RNG (no <random> state shared with the model code)
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed ? seed : 1u) {}
    uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    float uniform() { return (float)(next() & 0xFFFF) / 65536.0f; }              // [0,1)
    float uniform(float lo, float hi) { return lo + (hi - lo) * uniform(); }
    int   below(int n) { return (int)(next() % (uint32_t)n); }
};

inline std::vector<float> to_host(const GPUTensor& t) {
    std::vector<float> h((size_t)t.size);
    d2h(h.data(), t, t.size);
    return h;
}

inline std::vector<float> dev_to_host(const float* d, int n) {
    std::vector<float> h((size_t)n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

inline float* dev_from_host(const std::vector<float>& h) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

inline double max_abs_diff(const std::vector<float>& a, const std::vector<double>& b) {
    double w = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) w = std::max(w, std::fabs((double)a[i] - b[i]));
    return w;
}

inline double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double w = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) w = std::max(w, std::fabs((double)a[i] - (double)b[i]));
    return w;
}

inline bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

// Deterministic, non-trivial weights for EVERY tensor the GPU model uses (positional embedding, LN
// gamma/beta, biases, final LN) so that a gradient/forward check cannot pass by accident on zeros/ones.
inline void randomize_model(LOGOSModel& m, uint32_t seed, float scale = 0.2f) {
    int s = (int)(seed & 0x7FFFFFFF);
    for (Tensor* p : m.parameters()) p->fill_random(-scale, scale, s++);
    m.pos_embedding.fill_random(-0.1f, 0.1f, s++);
    m.final_ln.gamma.fill_random(0.8f, 1.2f, s++);
    m.final_ln.beta.fill_random(-0.1f, 0.1f, s++);
    for (auto& blk : m.layers) {
        blk.ln1.gamma.fill_random(0.8f, 1.2f, s++);
        blk.ln1.beta.fill_random(-0.1f, 0.1f, s++);
        blk.ln2.gamma.fill_random(0.8f, 1.2f, s++);
        blk.ln2.beta.fill_random(-0.1f, 0.1f, s++);
    }
}

// Name of entry `idx` of ModelGPU::all_parameters() (layout documented in ModelGPU.cu / BackwardGPU.cu).
inline std::string param_name(int idx, const ModelConfig& cfg) {
    const int H = cfg.num_heads, PPL = H * 4 + 9;
    if (idx == 0) return "embedding";
    if (idx == 1) return "pos_embedding";
    if (idx == 2) return "lm_head";
    const int last = 3 + cfg.num_layers * PPL;
    if (idx == last)     return "final_ln.gamma";
    if (idx == last + 1) return "final_ln.beta";
    const int l = (idx - 3) / PPL, off = (idx - 3) % PPL;
    static const char* hn[4] = {"W_Q", "W_K", "W_V", "W_O"};
    static const char* tail[9] = {"W_proj", "W1", "b1", "W2", "b2", "ln1.gamma", "ln1.beta", "ln2.gamma", "ln2.beta"};
    char buf[96];
    if (off < H * 4) std::snprintf(buf, sizeof(buf), "layer%d.head%d.%s", l, off / 4, hn[off % 4]);
    else             std::snprintf(buf, sizeof(buf), "layer%d.%s", l, tail[off - H * 4]);
    return buf;
}

// Mean free-energy loss F = CE - T*S of one sequence; optionally returns dF/dlogits (seq x V) as a GPUTensor.
inline float loss_F(ModelGPU& m, const std::vector<int>& in, const std::vector<int>& tg, float T,
                    GPUTensor* grad_out = nullptr) {
    const int seq = (int)in.size(), V = m.cfg.vocab_size;
    GPUTensor logits = m.forward(in);
    CudaPtr<int>   d_t(seq);
    CudaPtr<float> d_l(seq);
    CudaPtr<float> d_g(seq * V);
    CUDA_CHECK(cudaMemcpy(d_t.get(), tg.data(), (size_t)seq * sizeof(int), cudaMemcpyHostToDevice));
    FreeEnergyResult fe;
    cuda_free_energy_loss(logits.data, d_t.get(), d_l.get(), d_g.get(), seq, V, T, fe);
    if (grad_out) {
        *grad_out = gpu_alloc(seq, V);
        CUDA_CHECK(cudaMemcpy(grad_out->data, d_g.get(), (size_t)seq * V * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    return fe.free_energy;
}

inline int finish(const char* name) {
    std::printf("\n%s: %d passed, %d failed\n", name, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace gt
