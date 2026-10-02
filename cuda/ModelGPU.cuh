#pragma once
#include "VedicGEMM.cuh"
#include "../include/Model.hpp"
#include <vector>

struct GPUBlock {
    std::vector<GPUTensor> W_Q, W_K, W_V, W_O;
    GPUTensor W_proj;
    GPUTensor W1, b1, W2, b2;
    GPUTensor ln1_gamma, ln1_beta;
    GPUTensor ln2_gamma, ln2_beta;
};

// INT8 compressed tensor for Nikhilam KV cache
struct NikhilamTensor {
    int8_t* data  = nullptr;
    float   scale = 1.0f;
    int     size  = 0;

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

// Per-head forward cache (float32 K/V kept for backward; INT8 for stats)
struct HeadCache {
    GPUTensor Q, K, V;
    GPUTensor attn_probs;
    GPUTensor head_out;
    NikhilamTensor K_int8, V_int8;
    GPUTensor Q_adv;  // advected Q used in scores
    GPUTensor V_s;    // diffused V used in head_out
};

struct LayerCache {
    GPUTensor block_input;
    GPUTensor normed1, normed2;
    GPUTensor ffn_H, ffn_A;
    GPUTensor attn_out, concat;
    GPUTensor post_attn;  // X after attention residual — true input of LN2
    GPUTensor ffn_mask;   // Feynman dropout mask (invalid when training=false)
    GPUTensor ln1_w, ln2_w;  // Reynolds laminar weight per token
    std::vector<HeadCache> heads;
};

struct HyperConfig {
    float curvature = 1.0f;
    bool  enabled   = true;
};

struct GPUPhysicsConfig {
    bool  nikhilam_kv     = true;
    bool  shunyam         = true;
    int   window          = 32;
    int   stride          = 8;
    bool  navier_stokes   = true;
    float ns_eta          = 0.1f;
    float ns_nu           = 0.05f;
    bool  feynman_dropout = true;
    float drop_p          = 0.1f;
    float drop_hbar       = 1.0f;
    bool  reynolds        = true;
    float re_crit         = 2.0f;
    float re_k            = 5.0f;
    bool  training        = false;
};

__global__ void ce_loss_kernel(const float* logits, const int* targets, float* loss_out, float* grad_out, int seq_len, int vocab_size);
__global__ void langevin_step_kernel(float* weights, float* velocity, const float* gradients, float lr, float friction, float noise_scale, unsigned int seed, int size);
__global__ void expmap0_kernel(float* X, int seq, int d, float curvature);
__global__ void logmap0_kernel(float* X, int seq, int d, float curvature);
__global__ void expmap0_bwd_kernel(const float* X_hyp, const float* dOut, float* dX, int seq, int d, float curvature);
__global__ void nikhilam_quantize_kernel(const float* src, int8_t* dst, float scale, int size);
__global__ void nikhilam_dequantize_kernel(const int8_t* src, float* dst, float scale, int size);
__global__ void absmax_kernel(const float* data, float* out, int size);
GPUTensor nikhilam_decompress(const NikhilamTensor& src, int rows, int cols);
__global__ void shunyam_mask_kernel(float* scores, int seq, int window, int stride);
__global__ void ns_advect_kernel(const float* Q, float* Qa, int seq, int d, float eta);
__global__ void ns_advect_bwd_kernel(const float* dQa, float* dQ, int seq, int d, float eta);
__global__ void ns_diffuse_kernel(const float* V, float* Vs, int seq, int d, float nu);
__global__ void ns_diffuse_bwd_kernel(const float* dVs, float* dV, int seq, int d, float nu);
__global__ void reynolds_norm_kernel(const float* X, const float* gamma, const float* beta, const float* rmean, const float* rvar, float* Y, float* W, int seq, int d, float re_crit, float k, float eps);
__global__ void reynolds_bwd_kernel(const float* X, const float* gamma, const float* dY, const float* W, const float* rmean, const float* rvar, float* dX, float* dGamma, float* dBeta, int seq, int d, float eps);
__global__ void run_stats_update_kernel(const float* X, float* rmean, float* rvar, int seq, int d, float decay);
__global__ void feynman_dropout_kernel(float* A, float* mask, int n, float p, float hbar, unsigned seed);

class ModelGPU {
public:
    ModelConfig cfg;
    HyperConfig hyper_cfg;
    GPUPhysicsConfig phys;
    std::vector<GPUTensor> run_mean, run_var;  // Reynolds BN running stats, 2 per layer
    unsigned dropout_seed = 12345u;

    GPUTensor gpu_embedding, gpu_pos_embedding, gpu_lm_head;
    std::vector<GPUBlock> gpu_blocks;

    int* d_token_ids = nullptr;
    GPUTensor last_hidden;
    GPUTensor X_euclidean;  // embedding before hyperbolic map (for backward)

    std::vector<LayerCache> layer_cache;

    // FP16 shadow weight pointers, parallel to all_parameters() — set by train loop before forward
    bool amp_enabled = false;
    std::vector<const __half*> fp16_param_ptrs;

    ModelGPU(const ModelConfig& cfg, const HyperConfig& hcfg = {});
    ~ModelGPU();

    void load_from_cpu(const LOGOSModel& cpu_model);
    GPUTensor forward(const std::vector<int>& token_ids);
    std::vector<GPUTensor*> all_parameters();
    std::vector<GPUTensor*> alloc_grad_buffers() const;
    void sync_to_cpu(LOGOSModel& cpu_model) const;
    void free_layer_cache();
    void apply_norm(const GPUTensor& X, const GPUTensor& gamma, const GPUTensor& beta, GPUTensor& Y, GPUTensor& W, int slot, int seq);
    void update_norm_stats();
    void print_kvcache_stats(int seq, int num_steps) const;
};
