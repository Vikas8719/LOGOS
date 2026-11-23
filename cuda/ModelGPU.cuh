#pragma once
#include "CublasGemm.cuh"
#include "../include/Model.hpp"
#include <vector>

struct GPUBlock {
    std::vector<GPUTensor> W_Q, W_K, W_V, W_O;
    GPUTensor W_proj;
    GPUTensor W1, b1, W2, b2;
    GPUTensor ln1_gamma, ln1_beta;
    GPUTensor ln2_gamma, ln2_beta;
};

struct HeadCache {
    GPUTensor Q, K, V;
    GPUTensor attn_probs;
    GPUTensor head_out;
    GPUTensor Q_adv;  
    GPUTensor V_s;    
};

struct LayerCache {
    GPUTensor block_input;
    GPUTensor normed1, normed2;
    GPUTensor ffn_H, ffn_A;
    GPUTensor attn_out, concat;
    GPUTensor post_attn;  
    GPUTensor ffn_mask;   
    GPUTensor ln1_w, ln2_w;  
    std::vector<HeadCache> heads;
};

struct HyperConfig {
    float curvature = 1.0f;
    bool  enabled   = false;   // default OFF; LOGOS_HYPERBOLIC=1 se on (train + generate dono me same rakho)
};

struct GPUPhysicsConfig {
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
    std::vector<GPUTensor> run_mean, run_var;  
    unsigned dropout_seed = 12345u;

    GPUTensor gpu_embedding, gpu_pos_embedding, gpu_lm_head;
    GPUTensor ln_f_gamma, ln_f_beta;   // final LayerNorm (lm_head se pehle); all_parameters() ke END me
    std::vector<GPUBlock> gpu_blocks;

    int* d_token_ids = nullptr;
    GPUTensor last_hidden;   // final-LN output (lm_head ka input)
    GPUTensor final_in;      // final LN se pehle ka residual stream (backward ke liye)
    GPUTensor X_euclidean;  
    std::vector<LayerCache> layer_cache;

    bool amp_enabled = false;
    std::vector<const __half*> fp16_param_ptrs;
    bool flash_logged = false;

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
};
