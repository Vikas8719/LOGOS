#pragma once
// ============================================================
//  LOGOS — cuda/ModelGPU.cuh  (v8 — Phase 2 Physics)
//
//  v6 retained: Full attention backward (10 kernels), HeadCache,
//               LayerCache, all_parameters(), alloc_grad_buffers()
//
//  PHASE 2 ADDITIONS:
//
//  [P2-A] Hyperbolic Embedding Space (Poincaré Ball Model)
//    Applies AFTER token+positional embedding lookup, BEFORE Transformer.
//    exp_map: Euclidean vector → Poincaré ball surface
//      expmap0(v) = tanh(||v||/2) * v / (||v|| + ε)
//    log_map: Poincaré ball → Euclidean tangent space (for backward)
//      logmap0(y) = 2 * arctanh(||y||) * y / (||y|| + ε)
//
//    Why only embedding layer:
//      - All Transformer ops (GEMM, LayerNorm, Attention) assume flat R^d
//      - Embedding maps discrete tokens to geometry — hyperbolic is natural here
//      - Hierarchical language structure fits exponentially in hyperbolic space
//      - NO change to attention backward, FFN backward, or any other kernel
//
//    VRAM: zero extra (in-place on X buffer)
//    Speed: +2 kernels per forward pass (seq×D tanh/atanh ops — negligible)
//    Curvature: c = 1.0 (unit Poincaré ball, adjustable via HyperConfig)
//
//  [P2-B] Nikhilam KV Cache Compression (INT8 per-head)
//    Nikhilam Navatascharamam Dashatah: complement-from-base
//    Applied to K and V tensors in HeadCache after computation, before store.
//    Compression: float32 (4B) → int8 (1B) = 4x memory reduction
//    Method:
//      scale[h] = max(|K|) / 127.0f
//      K_int8[i] = clamp(round(K[i] / scale), -127, 127)
//      K_reconstructed[i] = K_int8[i] * scale   (for backward)
//    Nikhilam complement encoding:
//      For each 8-element group: store (base - value) where base = 127
//      Reconstruction: value = base - complement
//      This is Nikhilam's "nines' complement" adapted to int8 range
//
//    HeadCache change:
//      K, V: float32 GPUTensor → kept for backward (needed for dK, dV grads)
//      K_int8, V_int8: int8 compressed versions (for VRAM savings during fwd)
//      scale_K, scale_V: per-head float scalars
//
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
};

// ── [P2-A] Hyperbolic Embedding Config ───────────────────────
// Controls curvature of the Poincaré ball
struct HyperConfig {
    float curvature = 1.0f;   // c > 0: tighter ball; c < 1: flatter
    bool  enabled   = true;   // set false to fall back to Euclidean
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

// ── GPU Model class ───────────────────────────────────────────
class ModelGPU {
public:
    ModelConfig cfg;
    HyperConfig hyper_cfg;   // [P2-A] hyperbolic embedding settings

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

    // [P2-B] Statistics: print compression ratio and memory saved
    void print_kvcache_stats(int seq, int num_steps) const;
};
