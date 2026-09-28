#pragma once

//    Forward: compress after compute, decompress before attn_probs
//    Backward: use float32 K/V (already cached before compression)
//    VRAM: at seq=512, H=8, DH=64: saves 8×512×64×3B = 1.5MB per layer
//    On T4 with L=6: ~9MB saved — allows seq_len to scale from 512 → 640+
// ============================================================
#include "VedicGEMM.cuh"
#include "../include/Model.hpp"
#include <vector>

// ── GPU version of one Transformer block's weights ───────────
struct GPUBlock {
    std::vector<GPUTensor> W_Q, W_K, W_V, W_O;  // per head
    GPUTensor W_proj;
    GPUTensor W1, b1, W2, b2;
    GPUTensor ln1_gamma, ln1_beta;
    GPUTensor ln2_gamma, ln2_beta;
};

// ── [P2-B] Nikhilam INT8 compressed tensor ───────────────────
// Stores int8* on GPU + per-tensor float scale
// Used for K_int8, V_int8 in HeadCache
struct NikhilamTensor {
    int8_t* data  = nullptr;   // GPU int8 buffer
    float   scale = 1.0f;      // quantization scale: float = int8 * scale
    int     size  = 0;         // number of elements

    NikhilamTensor() = default;
    NikhilamTensor(const NikhilamTensor&) = delete;
    NikhilamTensor& operator=(const NikhilamTensor&) = delete;

    NikhilamTensor(NikhilamTensor&& o) noexcept
        : data(o.data), scale(o.scale), size(o.size)
    { o.data = nullptr; o.size = 0; }

    NikhilamTensor& operator=(NikhilamTensor&& o) noexcept {
        if (this != &o) {
            if (data) cudaFree(data);
            data=o.data; scale=o.scale; size=o.size;
            o.data=nullptr; o.size=0;
        }
        return *this;
    }

    ~NikhilamTensor() { if (data) { cudaFree(data); data=nullptr; } }
    bool valid() const { return data != nullptr && size > 0; }
};

// ── [P2-A+P2-B] Per-head attention cache ────────────────────
// v6: Q, K, V (float32), attn_probs, head_out
// v8: + K_int8, V_int8 (Nikhilam INT8 compressed)
//     K/V float32 kept for backward (gradients need full precision)
struct HeadCache {
    GPUTensor Q;            // (seq × DH) float32 — for dW_Q backward
    GPUTensor K;            // (seq × DH) float32 — for dK backward
    GPUTensor V;            // (seq × DH) float32 — for dV backward
    GPUTensor attn_probs;   // (seq × seq) float32 — for softmax_bwd
    GPUTensor head_out;     // (seq × DH) float32 — for dW_O backward

    // [P2-B] Nikhilam INT8 compressed K and V (for forward attention only)
    // These are 4x smaller than float32 K/V — reduce VRAM during long seqs
    NikhilamTensor K_int8;  // (seq × DH) int8 — Nikhilam complement encoded
    NikhilamTensor V_int8;  // (seq × DH) int8 — Nikhilam complement encoded
    GPUTensor Q_adv;        // advected Q actually used in scores (backward needs it)
    GPUTensor V_s;          // diffused/dequantised V actually used in head_out (backward needs it)
};

// ── Per-layer activation cache ───────────────────────────────
struct LayerCache {
    GPUTensor block_input;
    GPUTensor normed1;
    GPUTensor normed2;
    GPUTensor ffn_H;
    GPUTensor ffn_A;
    GPUTensor attn_out;
    GPUTensor concat;
    std::vector<HeadCache> heads;
    GPUTensor post_attn;    // X after attention residual = true input of LN2 (backward)
    GPUTensor ffn_mask;     // Feynman dropout mask (invalid when dropout off)
    GPUTensor ln1_w;        // Reynolds laminar weight per token (LN1)
    GPUTensor ln2_w;        // Reynolds laminar weight per token (LN2)
};

// ── [P2-A] Hyperbolic Embedding Config ───────────────────────
// Controls curvature of the Poincaré ball
struct HyperConfig {
    float curvature = 1.0f;   // c > 0: tighter ball; c < 1: flatter
    bool  enabled   = true;   // set false to fall back to Euclidean
};

// ── GPU physics/Vedic switches (each can be toggled to isolate issues) ──
struct GPUPhysicsConfig {
    bool  nikhilam_kv     = true;   // Nikhilam INT8 K/V in attention (straight-through backward)
    bool  shunyam         = true;   // Shunyam sparse (local window + global stride) attention mask
    int   window          = 32;
    int   stride          = 8;
    bool  navier_stokes   = true;   // causal advection on Q + viscous diffusion on V
    float ns_eta          = 0.1f;
    float ns_nu           = 0.05f;
    bool  feynman_dropout = true;   // Beta-amplitude path-integral dropout on FFN activations
    float drop_p          = 0.1f;
    float drop_hbar       = 1.0f;
    bool  reynolds        = true;   // Reynolds-adaptive blend of token LayerNorm and running-stat BatchNorm
    float re_crit         = 2.0f;
    float re_k            = 5.0f;
    bool  training        = false;  // enables dropout + running-stat updates (set by train loop only)
};

// ── Kernel declarations ───────────────────────────────────────
// (Legacy CE loss — kept for fallback; P1 uses free_energy_loss_kernel)
__global__ void ce_loss_kernel(const float* logits, const int* targets,
                               float* loss_out, float* grad_out,
                               int seq_len, int vocab_size);

// (Legacy Langevin — kept for CPU fallback; P1 uses leapfrog_langevin_kernel)
__global__ void langevin_step_kernel(float* weights, float* velocity,
                                     const float* gradients, float lr,
                                     float friction, float noise_scale,
                                     unsigned int seed, int size);

// [P2-A] Hyperbolic embedding kernels (defined in ModelGPU.cu)
// exp_map: Euclidean R^d → Poincaré ball  (forward: after emb lookup)
__global__ void expmap0_kernel(float* X, int seq, int d, float curvature);
// log_map: Poincaré ball → Euclidean R^d  (backward: before emb grad)
__global__ void logmap0_kernel(float* X, int seq, int d, float curvature);
// exp_map backward: gradient through expmap (chain rule)
__global__ void expmap0_bwd_kernel(const float* X_hyp, const float* dOut,
                                   float* dX, int seq, int d, float curvature);

// [P2-B] Nikhilam quantization kernels (defined in ModelGPU.cu)
// Compress float32 → int8 using Nikhilam complement encoding
__global__ void nikhilam_quantize_kernel(const float* src, int8_t* dst,
                                          float scale, int size);
// Decompress int8 → float32
__global__ void nikhilam_dequantize_kernel(const int8_t* src, float* dst,
                                            float scale, int size);
// Compute per-tensor absmax (for scale computation)
__global__ void absmax_kernel(const float* data, float* out, int size);

// Host helper: decompress NikhilamTensor → float32 GPUTensor
// (used in inference / checkpoint export; defined in ModelGPU.cu)
GPUTensor nikhilam_decompress(const NikhilamTensor& src, int rows, int cols);

// Shunyam: adds -1e9 to non-resonant (outside window/stride/causal) score entries
__global__ void shunyam_mask_kernel(float* scores, int seq, int window, int stride);
// Navier-Stokes: upwind advection of Q (causal) and its exact backward
__global__ void ns_advect_kernel(const float* Q, float* Qa, int seq, int d, float eta);
__global__ void ns_advect_bwd_kernel(const float* dQa, float* dQ, int seq, int d, float eta);
// Navier-Stokes: causal viscous diffusion of V and its exact backward
__global__ void ns_diffuse_kernel(const float* V, float* Vs, int seq, int d, float nu);
__global__ void ns_diffuse_bwd_kernel(const float* dVs, float* dV, int seq, int d, float nu);
// Reynolds norm: per-token blend of LayerNorm and running-stat BatchNorm (+ backward, stats EMA)
__global__ void reynolds_norm_kernel(const float* X, const float* gamma, const float* beta,
                                     const float* rmean, const float* rvar, float* Y, float* W,
                                     int seq, int d, float re_crit, float k, float eps);
__global__ void reynolds_bwd_kernel(const float* X, const float* gamma, const float* dY, const float* W,
                                    const float* rmean, const float* rvar, float* dX,
                                    float* dGamma, float* dBeta, int seq, int d, float eps);
__global__ void run_stats_update_kernel(const float* X, float* rmean, float* rvar,
                                        int seq, int d, float decay);
// Feynman dropout: Beta((1-p)h, p*h) amplitude per activation, mask saved for backward
__global__ void feynman_dropout_kernel(float* A, float* mask, int n, float p, float hbar, unsigned seed);

// ── GPU Model class ───────────────────────────────────────────
class ModelGPU {
public:
    ModelConfig cfg;
    HyperConfig hyper_cfg;   // [P2-A] hyperbolic embedding settings
    GPUPhysicsConfig phys;   // Vedic/physics switches for the GPU forward pass
    std::vector<GPUTensor> run_mean, run_var;   // Reynolds BN running stats, 2 slots per layer (LN1, LN2)
    unsigned dropout_seed = 12345u;

    GPUTensor gpu_embedding;
    GPUTensor gpu_pos_embedding;
    GPUTensor gpu_lm_head;
    std::vector<GPUBlock> gpu_blocks;

    int* d_token_ids = nullptr;
    GPUTensor last_hidden;

    // [P2-A] Cache X_euclidean (before exp_map) for backward
    GPUTensor X_euclidean;   // (seq × D) — embedding output before hyperbolic map

    std::vector<LayerCache> layer_cache;

    ModelGPU(const ModelConfig& cfg, const HyperConfig& hcfg = {});
    ~ModelGPU();

    void load_from_cpu(const LOGOSModel& cpu_model);

    // Forward: applies exp_map after embedding, Nikhilam compress K/V
    GPUTensor forward(const std::vector<int>& token_ids);

    std::vector<GPUTensor*> all_parameters();
    std::vector<GPUTensor*> alloc_grad_buffers() const;
    void sync_to_cpu(LOGOSModel& cpu_model) const;
    void free_layer_cache();

    // Reynolds/LayerNorm dispatch for one norm site (slot = 2*layer + {0:LN1,1:LN2})
    void apply_norm(const GPUTensor& X, const GPUTensor& gamma, const GPUTensor& beta,
                    GPUTensor& Y, GPUTensor& W, int slot, int seq);
    // EMA-update Reynolds running stats from cached inputs (call after backward)
    void update_norm_stats();

    // [P2-B] Statistics: print compression ratio and memory saved
    void print_kvcache_stats(int seq, int num_steps) const;
};
