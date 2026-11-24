#pragma once
// Backward pass of the LOGOS transformer (GPU), extracted from train_gpu.cu into logos_core so that
// tests (finite-difference gradient check, kernel-vs-CPU checks) can link against the exact code that
// training uses. Signatures of run_backward() and the buffer structs are unchanged.
#include "CublasGemm.cuh"
#include "ModelGPU.cuh"
#include "../include/Model.hpp"
#include <vector>

// ---- backward kernels (defined in BackwardGPU.cu) ----------------------------------------------------
__global__ void embedding_bwd_kernel(const int* __restrict__ token_ids, const float* __restrict__ dX,
                                     float* __restrict__ d_emb, float* __restrict__ d_pos,
                                     int seq, int d, int vocab_size);
__global__ void vec_add_kernel(float* dst, const float* src, int size);
__global__ void scale_grads_kernel(float* grad, float scale, int size);
__global__ void vec_mul_kernel(float* dst, const float* src, int size);
__global__ void gelu_bwd_kernel(const float* __restrict__ ffn_H, const float* __restrict__ d_ffn_A,
                                float* __restrict__ d_ffn_H, int total);
__global__ void layernorm_bwd_kernel(const float* __restrict__ X, const float* __restrict__ gamma,
                                     const float* __restrict__ dY, float* __restrict__ dX,
                                     float* __restrict__ d_gamma, float* __restrict__ d_beta,
                                     int seq, int d, float eps);
// dS = P * (dP - rowsum(P*dP)) * inv_sqrt_DH ; one block (<=1024 threads, multiple of 32) per row
__global__ void softmax_bwd_kernel(const float* __restrict__ attn_probs, const float* __restrict__ d_attn_probs,
                                   float* __restrict__ d_scores, int seq, float inv_sqrt_DH);

// ---- pre-allocated per-head / per-layer buffers ------------------------------------------------------
struct HeadBwdBuffers {
    GPUTensor d_head_out;       // (seq, DH)
    GPUTensor d_attn_probs_buf; // (seq, seq) recomputed attention probabilities
    GPUTensor dV_s;             // (seq, DH)
    GPUTensor dV;               // (seq, DH)
    GPUTensor d_attn_probs;     // (seq, seq)
    GPUTensor d_scores;         // (seq, seq)
    GPUTensor dQ_adv;           // (seq, DH)
    GPUTensor dQ;               // (seq, DH)
    GPUTensor dK;               // (seq, DH)

    bool allocated = false;
    int  alloc_seq = 0, alloc_DH = 0;

    void ensure(int seq_, int DH_);
};

struct LayerBwdBuffers {
    GPUTensor d_concat;   // (seq, D)
    GPUTensor d_dX_ln1;   // (seq, D)
    bool allocated = false;
    int  alloc_seq = 0, alloc_D = 0;

    void ensure(int seq_, int D_);
};

// GPU0 main-thread buffers and dedicated GPU1-thread buffers (same names as before the extraction)
extern HeadBwdBuffers  s_head_bwd;
extern LayerBwdBuffers s_layer_bwd;
extern HeadBwdBuffers  s_head_bwd_gpu1;
extern LayerBwdBuffers s_layer_bwd_gpu1;

// Scratch tensors run_backward() needs (all sized (seq, D) or (seq, 4D)).
struct BackwardScratch {
    GPUTensor d_dX_out, d_d_ffn_out, d_d_ffn_A, d_d_ffn_H, d_d_normed2, d_dX_ln, d_dX_attn_in;
    int alloc_seq = 0, alloc_D = 0;

    void ensure(int seq, int D) {
        if (alloc_seq == seq && alloc_D == D) return;
        d_dX_out     = gpu_alloc(seq, D);
        d_d_ffn_out  = gpu_alloc(seq, D);
        d_d_ffn_A    = gpu_alloc(seq, 4 * D);
        d_d_ffn_H    = gpu_alloc(seq, 4 * D);
        d_d_normed2  = gpu_alloc(seq, D);
        d_dX_ln      = gpu_alloc(seq, D);
        d_dX_attn_in = gpu_alloc(seq, D);
        alloc_seq = seq; alloc_D = D;
    }
};

// Full model backward. `model.forward()` must have been called for the same tokens first (it fills
// layer_cache / final_in / last_hidden). Gradients are ACCUMULATED into `gpu_grads`
// (same order as model.all_parameters()); d_logits_grad = dLoss/dlogits, shape (>=seq, vocab).
void run_backward(
    ModelGPU& gpu_model, const ModelConfig& cfg,
    const std::vector<GPUTensor*>& gpu_grads,
    GPUTensor& d_logits_grad, GPUTensor& d_dX_out,
    GPUTensor& d_d_ffn_out, GPUTensor& d_d_ffn_A,
    GPUTensor& d_d_ffn_H,   GPUTensor& d_d_normed2,
    GPUTensor& d_dX_ln,     GPUTensor& d_dX_attn_in,
    int seq,
    HeadBwdBuffers* head_bwd_override = nullptr,
    LayerBwdBuffers* layer_bwd_override = nullptr);

// Convenience wrapper (tests / tools): allocates scratch on demand.
void run_backward_simple(ModelGPU& gpu_model, const std::vector<GPUTensor*>& gpu_grads,
                         GPUTensor& d_logits_grad, int seq, BackwardScratch& scratch,
                         HeadBwdBuffers* head_bwd_override = nullptr,
                         LayerBwdBuffers* layer_bwd_override = nullptr);
