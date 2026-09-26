//  [P1-A] Gunitasamuchayah — unchanged
//  [P1-B] FreeEnergy Loss  — unchanged
//  [P1-C] leapfrog_langevin_kernel — REPLACED by shm_hybrid_kernel
//  [NEW]  shm_hybrid_kernel — Hamiltonian + Langevin fused kernel
// ============================================================
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cmath>

// ── Error handling (v4, retained) ────────────────────────────
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "CUDA Error at %s:%d — %s", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
            fprintf(stderr, "%s\n", _msg); \
            throw std::runtime_error(_msg); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t _e = cudaGetLastError(); \
        if (_e != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "Kernel launch error at %s:%d — %s", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
            fprintf(stderr, "%s\n", _msg); \
            throw std::runtime_error(_msg); \
        } \
    } while(0)

// ── GPUTensor — RAII wrapper (v4, retained) ──────────────────
struct GPUTensor {
    float* data = nullptr;
    int rows = 0, cols = 0, size = 0;

    GPUTensor() = default;
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;

    GPUTensor(GPUTensor&& o) noexcept
        : data(o.data), rows(o.rows), cols(o.cols), size(o.size)
    { o.data = nullptr; o.rows = o.cols = o.size = 0; }

    GPUTensor& operator=(GPUTensor&& o) noexcept {
        if (this != &o) {
            if (data) cudaFree(data);
            data=o.data; rows=o.rows; cols=o.cols; size=o.size;
            o.data=nullptr; o.rows=o.cols=o.size=0;
        }
        return *this;
    }

    ~GPUTensor() { if (data) { cudaFree(data); data=nullptr; } }
    bool valid() const { return data != nullptr && size > 0; }
};

// ── [P1-A] Gunitasamuchayah Result struct ────────────────────
// Returned by cuda_vedic_verify() — caller checks pass/fail
struct VedicVerifyResult {
    float checksum_C;       // sum(C) from actual output
    float checksum_vedic;   // sum_rows(A) · sum_cols(B) = Vedic prediction
    float relative_error;   // |checksum_C - checksum_vedic| / |checksum_vedic|
    bool  pass;             // true if relative_error < tolerance
};

// ── [P1-B] Free Energy Loss result ───────────────────────────
// Returned per batch step — log for monitoring thermodynamic state
struct FreeEnergyResult {
    float cross_entropy;   // U  = standard CE loss (same as before)
    float entropy;         // S  = Shannon entropy of softmax probs
    float free_energy;     // F  = U - temperature * S
    float temperature;     // T  at this step (for logging)
};

// ── Memory management ─────────────────────────────────────────
GPUTensor gpu_alloc(int rows, int cols);
void      gpu_free(GPUTensor& t);
void      h2d(GPUTensor& dst, const float* src, int size);
void      d2h(float* dst, const GPUTensor& src, int size);

// ── Core GPU ops (v4, retained) ──────────────────────────────
void cuda_vedic_gemm(const GPUTensor& A, const GPUTensor& B, GPUTensor& C);
// True when this build routes GEMM through NVIDIA cuBLAS. The Vedic
// Gunitasamuchayah verification remains available for either backend.
bool cuda_vedic_gemm_uses_cublas();
void cuda_vedic_gemm_bias(const GPUTensor& A, const GPUTensor& W,
                           const GPUTensor& bias, GPUTensor& C);
void cuda_boltzmann_softmax(const GPUTensor& scores, GPUTensor& probs,
                             int seq_len, int vocab_size, float temperature);
void cuda_layernorm(const GPUTensor& X, const GPUTensor& gamma,
                    const GPUTensor& beta, GPUTensor& Y,
                    int seq_len, int d_model, float eps = 1e-5f);
void cuda_gelu(GPUTensor& data);
float cuda_clip_gradients(std::vector<GPUTensor*>& grads, float max_norm);

// ── [P1-A] Gunitasamuchayah GEMM Verification ────────────────
// Vedic sutram: sum(C) = dot(row_sums(A), col_sums(B))
// Complexity: O(M*K + K*N + M + N) instead of O(M*K*N)
// Call every ~1000 steps in training loop for cheap sanity check
// tolerance: 0.01f (1%) works well for float32 accumulation error
VedicVerifyResult cuda_vedic_verify(const GPUTensor& A,
                                    const GPUTensor& B,
                                    const GPUTensor& C,
                                    float tolerance = 0.01f);

// ── [P1-B] Free Energy Loss kernel (host wrapper) ────────────
// Replaces ce_loss_kernel_parallel in train_gpu.cu for P1 training
// Computes: F = CE - temperature * entropy(softmax(logits))
// Gradient: dF/dlogit[i] = (softmax[i] - label[i]) / seq
//                         + temperature * softmax[i] * (log(softmax[i]) + S) / seq
// temperature: current Langevin T from GPULangevinOpt::temperature
// out_result: host-side FreeEnergyResult (filled after kernel + sync)
void cuda_free_energy_loss(
    const float* d_logits,          // GPU: (seq × vocab)
    const int*   d_targets,         // GPU: (seq,)
    float*       d_loss_buf,        // GPU: (seq,)  — per-token F values
    float*       d_grad_out,        // GPU: (seq × vocab) — dF/dlogit
    int seq_len, int vocab_size,
    float temperature,              // current Langevin T
    FreeEnergyResult& out_result);  // host-side result (filled after sync)

// ── [P1-C / v9] leapfrog_langevin_kernel — KEPT for backward compat ──
// Deprecated in v9 — use shm_hybrid_kernel instead
// Still declared so older code that references it compiles
__global__ void leapfrog_langevin_kernel(
    float* __restrict__       weights,
    float* __restrict__       velocity,
    const float* __restrict__ gradients,
    float lr,
    float friction,
    float noise_scale,
    unsigned int seed,
    int size);

// ── [v9 NEW] shm_hybrid_kernel — Hybrid Stochastic Hamiltonian ───────
//
//  Fused kernel: Hamiltonian symplectic + Langevin stochastic in one pass
//
//  Update equations (per parameter i):
//    noise     = noise_scale * η_i            [thermal fluctuation, FDT]
//    v_{t+½}   = mom_decay · v_t              [Hamiltonian momentum carry]
//              - (lr · alpha_H / 2) · ∇L_i   [Hamiltonian gradient half-kick]
//              - (lr · alpha_L / 2) · γ · v_t [Langevin friction half-kick]
//              + noise                         [Langevin thermal noise]
//    W_{t+1}   = W_t + lr · v_{t+½}          [position full-step]
//
//  Parameters:
//    weights    [in/out] W_t → W_{t+1}
//    velocity   [in/out] v_t → v_{t+½}
//    gradients  [in]     ∇L at W_t (from backward pass)
//    lr         learning rate (full step size)
//    mom_decay  β = Hamiltonian momentum decay (≈0.9, like Adam β₁)
//    friction   γ = Langevin friction coefficient (≈0.1, lower than pure)
//    alpha_H    Hamiltonian weight ∈ [0,1] (annealed 0.3→0.9)
//    alpha_L    Langevin weight   ∈ [0,1] (annealed 0.7→0.1, = 1-alpha_H)
//    noise_scale √(γ·kT·lr·alpha_L) — FDT-consistent thermal amplitude
//    seed       per-step RNG seed (xorshift32, reproducible)
//    size       number of parameters in this tensor
__global__ void shm_hybrid_kernel(
    float* __restrict__       weights,
    float* __restrict__       velocity,
    const float* __restrict__ gradients,
    float lr,
    float mom_decay,
    float friction,
    float alpha_H,
    float alpha_L,
    float noise_scale,
    unsigned int seed,
    int size);

// ── Config validation (v4, retained) ─────────────────────────
std::string validate_model_config(int d_model, int num_heads, int num_layers,
                                   int vocab_size, int max_seq_len);
