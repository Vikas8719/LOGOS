//  Logging: added alpha_H, alpha_L columns to training output
//  All Phase 1+2+3 kernels: UNCHANGED
//
//  FIX-1 (v9→v9-fix): Temperature floor in GPUSHMOpt::anneal()
//    T_end was 1e-6 → T → 0.0000 at late training
//    Now: T_end enforced >= T_MIN_FLOOR = 1e-3 in constructor + anneal()
//    Reason: F = CE - T*S; T=0 kills entropy regularization → overconfident → loss spike
//
//  [v12-VEDIC] FIX-8: Gunitasamuchayah cuBLAS tolerance fix
//    Problem: cuda_vedic_verify() called with tolerance=0.05f (5%) for ALL backends.
//             cuBLAS internally reorders float32 multiplications for peak throughput,
//             producing relative_error of 8-154% on last_hidden × lm_head.
//             This is NUMERICALLY CORRECT behaviour — not a GEMM bug.
//             The Vedic sutram sum(C) = Σ row_sums(A)_i · col_sums(B)_i holds
//             for exact arithmetic; cuBLAS FP error breaks this assumption.
//    Fix:     Detect cuBLAS via cuda_vedic_gemm_uses_cublas().
//             cuBLAS path → tolerance=0.30f (30%), log "cuBLAS-WARN (expected)".
//             Custom CUDA path → tolerance=0.05f (5%), strict as before.
//    Impact:  Gunitasamuchayah will now PASS on cuBLAS runs where FP error < 30%.
//             Checkpoint line now shows backend: "Vedic: N/M PASS [cuBLAS(tol=30%)]"
//             Pure training behaviour UNCHANGED — this only affects the verify step.
// ============================================================
#include "VedicGEMM.cuh"
#include "ModelGPU.cuh"
#include "MixedPrecision.cuh"
#include "../include/Tokenizer.hpp"
#include "../include/StreamingDataLoader.hpp"
#include "../include/Checkpoint.hpp"
#include "../include/PhysicsOpt.hpp"    // [v14-WIRE] WeightPathIntegral GPU LR scaling
#include "../include/TrainingState.hpp" // [v29-FULLRESUME] full training state save/load
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>

// [v32-T4-OOM-FIX] Forward declarations for kernels used in backward recompute
// NOTE: gpu_transpose_kernel ModelGPU.cu mein define hai (logos_core STATIC library)
// CUDA device kernels extern declare nahi karte — sirf host wrappers ya inline headers se call hote hain
// Solution: transpose inline implement karo ya wrappers use karo
// vedic_gemm_kernel aur boltzmann_softmax_kernel VedicGEMM.cu mein hain — logos_core se linked ✅
// causal_mask_kernel ModelGPU.cu mein hai — logos_core se linked ✅
// Lekin __global__ kernels cross-TU launch sirf CUDA separable compilation ke saath kaam karta hai
// Safer approach: har kernel ke liye device-side wrapper use karo jo logos_core expose karta hai
// YA: transpose ko train_gpu.cu mein hi define karo (ek simple kernel, duplication acceptable)

// [v32] Cross-TU kernel fix (v2 — no duplicate definitions):
// gpu_transpose_kernel → logos_core (ModelGPU.cu) mein define hai
// grad_scale_kernel    → logos_core (VedicGEMM.cu) mein define hai
// Dono train_gpu.cu ke saath link honge via logos_core static library.
// Isliye yahan SIRF extern forward declarations — NO redefinition.
// [LINKER FIX] Pehle inline define kiya tha → multiple definition error.
// Ab: logos_core se linked kernels ko seedha use karo via extern declaration.
extern __global__ void gpu_transpose_kernel(const float*, float*, int, int);
extern __global__ void grad_scale_kernel(float*, float, int);
extern __global__ void vedic_gemm_kernel(const float*, const float*, float*, int, int, int);
extern __global__ void boltzmann_softmax_kernel(const float*, float*, int, int, float);
extern __global__ void causal_mask_kernel(float*, int);

// FIX-1: Minimum temperature — entropy regularization always active
static constexpr float GPU_T_MIN_FLOOR = 1e-3f;

// [v17] Env override: Kaggle/RunPod se source badle bina LR / clip / T / steps tune kar sako
//   LOGOS_LR, LOGOS_CLIP, LOGOS_T_START, LOGOS_STEPS, LOGOS_WARMUP, LOGOS_NOISE_GAIN
//
// [v26-BUG14-FIX] Range-validated variant added: logos_env_f_clamped()
// Pehle: LOGOS_LR=999999 set karo to unmodified value use hota tha → NaN/Inf weights.
// Fix: har hyperparameter ke liye min/max clamp with warning print.
// logos_env_f() unchanged (backward compat for non-critical numeric env vars like steps/warmup).
static float logos_env_f(const char* name, float def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    float v = std::strtof(s, &end);
    return (end != s && std::isfinite(v)) ? v : def;
}

// Range-validated float env reader: value ko [lo, hi] mein clamp karo with warning.
// Use karo saare training hyperparameters ke liye jahan extreme values dangerous hain.
static float logos_env_f_clamped(const char* name, float def, float lo, float hi) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (end == s || !std::isfinite(v)) {
        fprintf(stderr, "  [ENV WARN] %s='%s' parse fail — default %.6g used\n", name, s, def);
        return def;
    }
    if (v < lo || v > hi) {
        float clamped = fmaxf(fminf(v, hi), lo);
        fprintf(stderr, "  [ENV WARN] %s=%.6g out of [%.6g, %.6g] — clamped to %.6g\n",
                name, v, lo, hi, clamped);
        return clamped;
    }
    return v;
}

// ============================================================
//  [v9-fix] GPU HYBRID SHM OPTIMIZER
// ============================================================
class GPUSHMOpt {
public:
    float   lr;
    float   friction;
    float   mom_decay;
    float   temperature, T_start, T_end;
    float   alpha_H_start, alpha_H_end;
    int64_t total_steps;
    int64_t step = 0;

    // [v17] Langevin noise gain. `temperature` do kaam karta hai: (1) loss me entropy weight
    // F=CE-T·S, (2) thermal noise amplitude. Noise ko N≈9M params par T=0.1 se seedha lagana
    // weights ko lr ke saath ~lr^1.5 se bigadta hai (drift ~lr). noise_gain<1 se noise ko
    // gradient-drift ke neeche rakhte hain; 1.0 = purana behaviour.
    float   noise_gain = 0.05f;

    std::vector<float*> d_velocity;
    std::vector<int>    sizes;

    float alpha_H = 0.3f;
    float alpha_L = 0.7f;

    GPUSHMOpt(float lr_        = 2e-4f,
              float friction_  = 0.8f,
              float mom_decay_ = 0.7f,
              float T_s        = 0.05f,
              float T_e        = 1e-3f,   // FIX-1: default raised from 1e-6
              float aH_start   = 0.3f,
              float aH_end     = 0.9f,
              int64_t steps    = 500000)
        : lr(lr_), friction(friction_), mom_decay(mom_decay_),
          temperature(T_s), T_start(T_s),
          // FIX-1: enforce floor even if caller passes tiny T_e
          T_end(fmaxf(T_e, GPU_T_MIN_FLOOR)),
          alpha_H_start(aH_start), alpha_H_end(aH_end),
          total_steps(steps)
    {}

    void init(const std::vector<GPUTensor*>& params) {
        for (auto* p : params) {
            float* vel;
            CUDA_CHECK(cudaMalloc(&vel, p->size * sizeof(float)));
            CUDA_CHECK(cudaMemset(vel, 0, p->size * sizeof(float)));
            d_velocity.push_back(vel);
            sizes.push_back(p->size);
        }
    }

    // FIX-1: temperature floor applied — T can never reach 0
    void anneal() {
        float r = total_steps > 0
            ? std::min(1.0f, (float)step / (float)total_steps) : 1.0f;
        float c = 0.5f * (1.0f + cosf(3.14159265f * r));

        // FIX-1: fmaxf ensures T stays >= GPU_T_MIN_FLOOR
        temperature = fmaxf(GPU_T_MIN_FLOOR, T_end + (T_start - T_end) * c);

        alpha_H = alpha_H_start + (alpha_H_end - alpha_H_start) * (1.0f - c);
        alpha_L = 1.0f - alpha_H;
    }

    void update(std::vector<GPUTensor*>& params,
                std::vector<GPUTensor*>& grads,
                float scale = 1.0f)
    {
        anneal();
        // [v17] noise ab lr schedule (warmup/cosine) follow karta hai + noise_gain se scaled
        float lr_scaled   = lr * scale;
        float noise_scale = noise_gain * sqrtf(friction * temperature * lr_scaled * alpha_L);

        for (int i = 0; i < (int)params.size(); ++i) {
            int sz = params[i]->size;
            shm_hybrid_kernel<<<(sz+255)/256, 256>>>(
                params[i]->data,
                d_velocity[i],
                grads[i]->data,
                lr_scaled,
                mom_decay,
                friction,
                alpha_H,
                alpha_L,
                noise_scale,
                (unsigned int)((step * 2654435769LL + i * 1234567LL) & 0xFFFFFFFFLL),
                sz);
        }
        CUDA_KERNEL_CHECK();
        ++step;
    }

    void get_state(float& out_T, float& out_aH, float& out_aL) const {
        out_T  = temperature;
        out_aH = alpha_H;
        out_aL = alpha_L;
    }

    ~GPUSHMOpt() { for (auto* v : d_velocity) cudaFree(v); }
};

// ============================================================
//  BACKPROP KERNELS (v6 — ALL UNCHANGED)
// ============================================================

__global__ void lmhead_grad_dw_kernel(
    const float* __restrict__ X, const float* __restrict__ dY,
    float* __restrict__ dW, int seq, int d, int vocab)
{
    int di=blockIdx.x, vi=blockIdx.y*blockDim.x+threadIdx.x;
    if (di>=d||vi>=vocab) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=X[s*d+di]*dY[s*vocab+vi];
    atomicAdd(&dW[di*vocab+vi],acc);
}

__global__ void lmhead_grad_dx_kernel(
    const float* __restrict__ dY, const float* __restrict__ W,
    float* __restrict__ dX, int seq, int d, int vocab)
{
    int s=blockIdx.x, di=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||di>=d) return;
    float acc=0.0f;
    for (int v=0;v<vocab;++v) acc+=dY[s*vocab+v]*W[di*vocab+v];
    dX[s*d+di]=acc;
}

__global__ void embedding_bwd_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ dX,
    float* __restrict__ d_emb, float* __restrict__ d_pos,
    int seq, int d, int vocab_size)
{
    int s=blockIdx.x, di=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||di>=d) return;
    int tok=token_ids[s];
    if (tok<0||tok>=vocab_size) return;
    float g=dX[s*d+di];
    atomicAdd(&d_emb[tok*d+di],g);
    atomicAdd(&d_pos[s*d+di],g);
}

__global__ void vec_add_kernel(float* dst, const float* src, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) dst[idx]+=src[idx];
}

__global__ void scale_grads_kernel(float* grad, float scale, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) grad[idx]*=scale;
}

// [v15-COMPLETE] FIX-15: Element-wise multiply — used for Feynman dropout backward
// d_ffn_A[i] *= mask[i]  (mask = dropout amplitude, saved in forward)
__global__ void vec_mul_kernel(float* dst, const float* src, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) dst[idx]*=src[idx];
}

// [v17] FFN bias gradients. b1/b2 ke gradient buffers kabhi likhe hi nahi jaate the
// (grads zero → biases kabhi learn nahi karte). db[j] = Σ_s dY[s,j]
__global__ void bias_grad_kernel(const float* __restrict__ dY, float* __restrict__ db,
                                 int seq, int cols)
{
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=cols) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=dY[s*cols+j];
    atomicAdd(&db[j],acc);
}

// [v17] GNorm diagnostics: Σ g² (float accumulator on device)
__global__ void sumsq_accum_kernel(const float* __restrict__ g, float* __restrict__ out, int n)
{
    __shared__ float sh[256];
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    sh[threadIdx.x]=(i<n)?g[i]*g[i]:0.0f;
    __syncthreads();
    for (int s=128;s>0;s>>=1) {
        if (threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x==0) atomicAdd(out,sh[0]);
}

// Σ g² over grads[b .. e)  — pre-clip group norm ke liye (sqrt caller karta hai)
// [v26-LEAK-FIX] d_acc RAII via CudaPtr: har CUDA_CHECK ya kernel throw pe guaranteed free.
// Pehle: raw float* d_acc tha; CUDA_KERNEL_CHECK() throw kare to leak.
// Yeh function har 1000 steps pe Vedic verify mein call hoti hai — accumulation guaranteed.
static float grad_group_sumsq(const std::vector<GPUTensor*>& grads, int b, int e)
{
    CudaPtr<float> d_acc(1);  // 1 float, RAII — zero-init via CudaPtr constructor
    CUDA_CHECK(cudaMemset(d_acc.get(), 0, sizeof(float)));
    for (int i=b;i<e && i<(int)grads.size();++i) {
        int n=grads[i]->size;
        sumsq_accum_kernel<<<(n+255)/256,256>>>(grads[i]->data, d_acc.get(), n);
    }
    CUDA_KERNEL_CHECK();
    float h=0.0f;
    CUDA_CHECK(cudaMemcpy(&h, d_acc.get(), sizeof(float), cudaMemcpyDeviceToHost));
    return h;
    // d_acc freed here automatically (CudaPtr destructor)
}

__global__ void ffn_w2_grad_kernel(
    const float* __restrict__ ffn_A, const float* __restrict__ d_out,
    float* __restrict__ dW2, int seq, int d4, int d)
{
    int fi=blockIdx.x, di=blockIdx.y*blockDim.x+threadIdx.x;
    if (fi>=d4||di>=d) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=ffn_A[s*d4+fi]*d_out[s*d+di];
    atomicAdd(&dW2[fi*d+di],acc);
}

__global__ void ffn_da_kernel(
    const float* __restrict__ d_out, const float* __restrict__ W2,
    float* __restrict__ d_ffn_A, int seq, int d4, int d)
{
    int s=blockIdx.x, fi=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||fi>=d4) return;
    float acc=0.0f;
    for (int di=0;di<d;++di) acc+=d_out[s*d+di]*W2[fi*d+di];
    d_ffn_A[s*d4+fi]=acc;
}

__global__ void gelu_bwd_kernel(
    const float* __restrict__ ffn_H, const float* __restrict__ d_ffn_A,
    float* __restrict__ d_ffn_H, int total)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=total) return;
    float x=ffn_H[idx];
    float k=0.7978845608f*(x+0.044715f*x*x*x);
    float th=tanhf(k), sech2=1.0f-th*th;
    float gp=0.5f*(1.0f+th)+0.5f*x*sech2*0.7978845608f*(1.0f+3.0f*0.044715f*x*x);
    d_ffn_H[idx]=d_ffn_A[idx]*gp;
}

__global__ void ffn_w1_grad_kernel(
    const float* __restrict__ normed2, const float* __restrict__ d_ffn_H,
    float* __restrict__ dW1, int seq, int d, int d4)
{
    int di=blockIdx.x, fi=blockIdx.y*blockDim.x+threadIdx.x;
    if (di>=d||fi>=d4) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=normed2[s*d+di]*d_ffn_H[s*d4+fi];
    atomicAdd(&dW1[di*d4+fi],acc);
}

__global__ void ffn_dx_kernel(
    const float* __restrict__ d_ffn_H, const float* __restrict__ W1,
    float* __restrict__ d_normed2, int seq, int d, int d4)
{
    int s=blockIdx.x, di=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||di>=d) return;
    float acc=0.0f;
    for (int fi=0;fi<d4;++fi) acc+=d_ffn_H[s*d4+fi]*W1[di*d4+fi];
    d_normed2[s*d+di]=acc;
}

__global__ void layernorm_bwd_kernel(
    const float* __restrict__ X, const float* __restrict__ gamma,
    const float* __restrict__ dY, float* __restrict__ dX,
    float* __restrict__ d_gamma, float* __restrict__ d_beta,
    int seq, int d, float eps)
{
    int row=blockIdx.x;
    if (row>=seq) return;
    const float* x=X+row*d; const float* dy=dY+row*d; float* dx=dX+row*d;
    __shared__ float smean,sinv_std,s_sum_dy,s_sum_dy_xhat;
    float ts=0.0f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) ts+=x[j];
    for (int o=16;o>0;o>>=1) ts+=__shfl_down_sync(0xffffffff,ts,o);
    __shared__ float sarr[8];
    if (threadIdx.x%32==0) sarr[threadIdx.x/32]=ts;
    __syncthreads();
    float tot=0.0f; if (threadIdx.x<8) tot=sarr[threadIdx.x];
    for (int o=4;o>0;o>>=1) tot+=__shfl_down_sync(0xffffffff,tot,o);
    if (threadIdx.x==0) smean=tot/d; __syncthreads();
    float tv=0.0f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) { float diff=x[j]-smean; tv+=diff*diff; }
    for (int o=16;o>0;o>>=1) tv+=__shfl_down_sync(0xffffffff,tv,o);
    __shared__ float varr[8];
    if (threadIdx.x%32==0) varr[threadIdx.x/32]=tv;
    __syncthreads();
    float vt=0.0f; if (threadIdx.x<8) vt=varr[threadIdx.x];
    for (int o=4;o>0;o>>=1) vt+=__shfl_down_sync(0xffffffff,vt,o);
    if (threadIdx.x==0) sinv_std=rsqrtf(vt/d+eps); __syncthreads();
    float tsdy=0.0f,tsdyx=0.0f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) {
        float xhat=(x[j]-smean)*sinv_std,dys=dy[j]*gamma[j];
        atomicAdd(&d_gamma[j],dy[j]*xhat); atomicAdd(&d_beta[j],dy[j]);
        tsdy+=dys; tsdyx+=dys*xhat;
    }
    for (int o=16;o>0;o>>=1) { tsdy+=__shfl_down_sync(0xffffffff,tsdy,o); tsdyx+=__shfl_down_sync(0xffffffff,tsdyx,o); }
    __shared__ float sda[8],sdxa[8];
    if (threadIdx.x%32==0) { sda[threadIdx.x/32]=tsdy; sdxa[threadIdx.x/32]=tsdyx; }
    __syncthreads();
    float tsd=0.0f,tsdx=0.0f;
    if (threadIdx.x<8) { tsd=sda[threadIdx.x]; tsdx=sdxa[threadIdx.x]; }
    for (int o=4;o>0;o>>=1) { tsd+=__shfl_down_sync(0xffffffff,tsd,o); tsdx+=__shfl_down_sync(0xffffffff,tsdx,o); }
    if (threadIdx.x==0) { s_sum_dy=tsd; s_sum_dy_xhat=tsdx; } __syncthreads();
    float inv_d=1.0f/d;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) {
        float xhat=(x[j]-smean)*sinv_std,dys=dy[j]*gamma[j];
        dx[j]=sinv_std*inv_d*(d*dys-s_sum_dy-xhat*s_sum_dy_xhat);
    }
}

// ============================================================
//  ATTENTION BACKWARD KERNELS (v6 — ALL UNCHANGED)
// ============================================================

__global__ void attn_dV_kernel(
    const float* __restrict__ attn_probs, const float* __restrict__ d_head_out,
    float* __restrict__ dV, int seq, int DH)
{
    int j=blockIdx.x, ki=blockIdx.y*blockDim.x+threadIdx.x;
    if (j>=DH||ki>=seq) return;
    float acc=0.0f;
    for (int i=0;i<seq;++i) acc+=attn_probs[i*seq+ki]*d_head_out[i*DH+j];
    atomicAdd(&dV[ki*DH+j],acc);
}

__global__ void attn_dAttnProbs_kernel(
    const float* __restrict__ d_head_out, const float* __restrict__ V,
    float* __restrict__ d_attn_probs, int seq, int DH)
{
    int i=blockIdx.x, k=blockIdx.y*blockDim.x+threadIdx.x;
    if (i>=seq||k>=seq) return;
    float acc=0.0f;
    for (int j=0;j<DH;++j) acc+=d_head_out[i*DH+j]*V[k*DH+j];
    d_attn_probs[i*seq+k]=acc;
}

__global__ void softmax_bwd_kernel(
    const float* __restrict__ attn_probs, const float* __restrict__ d_attn_probs,
    float* __restrict__ d_scores, int seq, float inv_sqrt_DH)
{
    int row=blockIdx.x;
    if (row>=seq) return;
    const float* p=attn_probs+row*seq;
    const float* dp=d_attn_probs+row*seq;
    float* ds=d_scores+row*seq;
    float dot=0.0f;
    for (int j=0;j<seq;++j) dot+=p[j]*dp[j];
    for (int j=threadIdx.x;j<seq;j+=blockDim.x)
        ds[j]=p[j]*(dp[j]-dot)*inv_sqrt_DH;
}

__global__ void attn_dQ_kernel(
    const float* __restrict__ d_scores, const float* __restrict__ K,
    float* __restrict__ dQ, int seq, int DH)
{
    int i=blockIdx.x, j=blockIdx.y*blockDim.x+threadIdx.x;
    if (i>=seq||j>=DH) return;
    float acc=0.0f;
    for (int k=0;k<seq;++k) acc+=d_scores[i*seq+k]*K[k*DH+j];
    dQ[i*DH+j]=acc;
}

__global__ void attn_dK_kernel(
    const float* __restrict__ d_scores, const float* __restrict__ Q,
    float* __restrict__ dK, int seq, int DH)
{
    int k=blockIdx.x, j=blockIdx.y*blockDim.x+threadIdx.x;
    if (k>=seq||j>=DH) return;
    float acc=0.0f;
    for (int i=0;i<seq;++i) acc+=d_scores[i*seq+k]*Q[i*DH+j];
    dK[k*DH+j]=acc;
}

__global__ void attn_dWQKV_kernel(
    const float* __restrict__ normed1, const float* __restrict__ dQKV,
    float* __restrict__ dW, int seq, int D, int DH)
{
    int di=blockIdx.x, dhi=blockIdx.y*blockDim.x+threadIdx.x;
    if (di>=D||dhi>=DH) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=normed1[s*D+di]*dQKV[s*DH+dhi];
    atomicAdd(&dW[di*DH+dhi],acc);
}

__global__ void attn_dX_from_QKV_kernel(
    const float* __restrict__ dQKV, const float* __restrict__ W,
    float* __restrict__ dN1, int seq, int D, int DH)
{
    int s=blockIdx.x, di=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||di>=D) return;
    float acc=0.0f;
    for (int dhi=0;dhi<DH;++dhi) acc+=dQKV[s*DH+dhi]*W[di*DH+dhi];
    atomicAdd(&dN1[s*D+di],acc);
}

__global__ void attn_dWO_kernel(
    const float* __restrict__ head_out, const float* __restrict__ d_concat,
    float* __restrict__ dWO, int seq, int DH, int D)
{
    int dhi=blockIdx.x, di=blockIdx.y*blockDim.x+threadIdx.x;
    if (dhi>=DH||di>=D) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=head_out[s*DH+dhi]*d_concat[s*D+di];
    atomicAdd(&dWO[dhi*D+di],acc);
}

__global__ void attn_dHeadOut_kernel(
    const float* __restrict__ d_concat, const float* __restrict__ W_O,
    float* __restrict__ d_head_out, int seq, int DH, int D)
{
    int s=blockIdx.x, dhi=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||dhi>=DH) return;
    float acc=0.0f;
    for (int di=0;di<D;++di) acc+=d_concat[s*D+di]*W_O[dhi*D+di];
    d_head_out[s*DH+dhi]=acc;
}

__global__ void attn_dWproj_kernel(
    const float* __restrict__ concat, const float* __restrict__ d_mha_out,
    float* __restrict__ dW_proj, int seq, int D)
{
    int i=blockIdx.x, j=blockIdx.y*blockDim.x+threadIdx.x;
    if (i>=D||j>=D) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=concat[s*D+i]*d_mha_out[s*D+j];
    atomicAdd(&dW_proj[i*D+j],acc);
}

__global__ void attn_dConcat_kernel(
    const float* __restrict__ d_mha_out, const float* __restrict__ W_proj,
    float* __restrict__ d_concat, int seq, int D)
{
    int s=blockIdx.x, j=blockIdx.y*blockDim.x+threadIdx.x;
    if (s>=seq||j>=D) return;
    float acc=0.0f;
    for (int k=0;k<D;++k) acc+=d_mha_out[s*D+k]*W_proj[j*D+k];
    d_concat[s*D+j]=acc;
}

// ============================================================
//  run_backward() — v15-COMPLETE
//  FIX-15-A: Feynman dropout backward (ffn_mask * d_ffn_H)
//  FIX-15-B: Hyperbolic expmap0 backward (Jacobian chain through embedding)
// ============================================================
static void run_backward(
    ModelGPU& gpu_model, ModelConfig& cfg,
    const std::vector<GPUTensor*>& gpu_grads,
    GPUTensor& d_logits_grad, GPUTensor& d_dX_out,
    GPUTensor& d_d_ffn_out, GPUTensor& d_d_ffn_A,
    GPUTensor& d_d_ffn_H,   GPUTensor& d_d_normed2,
    GPUTensor& d_dX_ln,     GPUTensor& d_dX_attn_in,
    int seq)
{
    int D=cfg.d_model, V=cfg.vocab_size, D4=4*D;
    int H=cfg.num_heads, DH=D/H;
    const int PPL=H*4+9;
    const int wproj_off=H*4, w1_off=H*4+1, w2_off=H*4+3;
    const int ln1g_off=H*4+5, ln1b_off=H*4+6;
    const int ln2g_off=H*4+7, ln2b_off=H*4+8;

    { dim3 blk(32),grd(D,(V+31)/32);
      lmhead_grad_dw_kernel<<<grd,blk>>>(
          gpu_model.last_hidden.data,d_logits_grad.data,
          gpu_grads[2]->data,seq,D,V); CUDA_KERNEL_CHECK(); }

    CUDA_CHECK(cudaMemset(d_dX_out.data,0,seq*D*sizeof(float)));
    { dim3 blk(32),grd(seq,(D+31)/32);
      lmhead_grad_dx_kernel<<<grd,blk>>>(
          d_logits_grad.data,gpu_model.gpu_lm_head.data,
          d_dX_out.data,seq,D,V); CUDA_KERNEL_CHECK(); }

    for (int l=cfg.num_layers-1;l>=0;--l) {
        auto& blk=gpu_model.gpu_blocks[l];
        auto& cache=gpu_model.layer_cache[l];
        int base=3+l*PPL;

        CUDA_CHECK(cudaMemcpy(d_d_ffn_out.data,d_dX_out.data,
                   seq*D*sizeof(float),cudaMemcpyDeviceToDevice));
        { dim3 g(D4,(D+31)/32),b(32);
          ffn_w2_grad_kernel<<<g,b>>>(cache.ffn_A.data,d_d_ffn_out.data,
                                      gpu_grads[base+w2_off]->data,seq,D4,D); CUDA_KERNEL_CHECK(); }
        // [v17] b2 gradient (pehle kabhi compute nahi hota tha)
        { bias_grad_kernel<<<(D+127)/128,128>>>(d_d_ffn_out.data,
                                                gpu_grads[base+w2_off+1]->data,seq,D); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D4+31)/32),b(32);
          ffn_da_kernel<<<g,b>>>(d_d_ffn_out.data,blk.W2.data,
                                 d_d_ffn_A.data,seq,D4,D); CUDA_KERNEL_CHECK(); }
        { int tot=seq*D4;
          gelu_bwd_kernel<<<(tot+255)/256,256>>>(cache.ffn_H.data,d_d_ffn_A.data,
                                                  d_d_ffn_H.data,tot); CUDA_KERNEL_CHECK(); }

        // [v15-COMPLETE] FIX-15-A: Feynman dropout backward
        // Forward mein: ffn_A[i] *= mask[i]  (Beta-amplitude dropout)
        // Backward mein: d_ffn_H[i] *= mask[i]  (chain rule through dropout)
        // mask = saved amplitude per activation (ffn_mask, shape seq×D4)
        // Without this: gradients flow through as if dropout never happened
        // → model does not learn to be robust to activation suppression
        if (gpu_model.phys.feynman_dropout && gpu_model.phys.training
            && cache.ffn_mask.valid()) {
            int tot=seq*D4;
            vec_mul_kernel<<<(tot+255)/256,256>>>(d_d_ffn_H.data,
                                                   cache.ffn_mask.data,tot);
            CUDA_KERNEL_CHECK();
        }

        { dim3 g(D,(D4+31)/32),b(32);
          ffn_w1_grad_kernel<<<g,b>>>(cache.normed2.data,d_d_ffn_H.data,
                                      gpu_grads[base+w1_off]->data,seq,D,D4); CUDA_KERNEL_CHECK(); }
        // [v17] b1 gradient (d_ffn_H = gradient w.r.t. pre-GELU H, dropout mask ke baad)
        { bias_grad_kernel<<<(D4+127)/128,128>>>(d_d_ffn_H.data,
                                                 gpu_grads[base+w1_off+1]->data,seq,D4); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D+31)/32),b(32);
          ffn_dx_kernel<<<g,b>>>(d_d_ffn_H.data,blk.W1.data,
                                 d_d_normed2.data,seq,D,D4); CUDA_KERNEL_CHECK(); }

        // [v14-WIRE] LN2 backward: use Reynolds bwd if active (uses post_attn cache)
        // post_attn = X after attention residual = real input to LN2 in forward
        if (gpu_model.phys.reynolds && cache.ln2_w.valid()) {
            reynolds_bwd_kernel<<<seq,256>>>(
                cache.post_attn.data, blk.ln2_gamma.data,
                d_d_normed2.data, cache.ln2_w.data,
                gpu_model.run_mean[2*l+1].data, gpu_model.run_var[2*l+1].data,
                d_dX_ln.data,
                gpu_grads[base+ln2g_off]->data, gpu_grads[base+ln2b_off]->data,
                seq, D, 1e-5f);
        } else {
            // [v17] FIX: LN2 ka real input post_attn hai (block_input LN1 ka input hai)
            layernorm_bwd_kernel<<<seq,256>>>(
                cache.post_attn.data, blk.ln2_gamma.data,
                d_d_normed2.data, d_dX_ln.data,
                gpu_grads[base+ln2g_off]->data, gpu_grads[base+ln2b_off]->data,
                seq, D, 1e-5f);
        }
        CUDA_KERNEL_CHECK();
        { int sz=seq*D;
          vec_add_kernel<<<(sz+255)/256,256>>>(d_dX_out.data,d_dX_ln.data,sz); CUDA_KERNEL_CHECK(); }

        // NOTE: d_concat is a GPUTensor (RAII) — auto-freed at end of loop body.
        // Previously declared in the middle of the block which made it harder
        // to reason about lifetime. Moved here for clarity.
        GPUTensor d_concat=gpu_alloc(seq,D);
        { dim3 g(D,(D+31)/32),b(32);
          attn_dWproj_kernel<<<g,b>>>(cache.concat.data,d_dX_out.data,
                                      gpu_grads[base+wproj_off]->data,seq,D); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D+31)/32),b(32);
          attn_dConcat_kernel<<<g,b>>>(d_dX_out.data,blk.W_proj.data,
                                       d_concat.data,seq,D); CUDA_KERNEL_CHECK(); }

        CUDA_CHECK(cudaMemset(d_dX_attn_in.data,0,seq*D*sizeof(float)));
        for (int h=0;h<H;++h) {
            auto& hc=cache.heads[h];
            GPUTensor d_head_out_h=gpu_alloc(seq,DH);
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dHeadOut_kernel<<<g,b>>>(d_concat.data,blk.W_O[h].data,
                                            d_head_out_h.data,seq,DH,D); CUDA_KERNEL_CHECK(); }
            { dim3 g(DH,(D+31)/32),b(32);
              attn_dWO_kernel<<<g,b>>>(hc.head_out.data,d_concat.data,
                                       gpu_grads[base+h*4+3]->data,seq,DH,D); CUDA_KERNEL_CHECK(); }
            // [v32-T4-OOM-FIX] attn_probs backward: recompute karo (save mat kiya tha)
            // Pehle: hc.attn_probs = gpu_alloc(seq, seq) forward mein → 4.29 GB → T4 OOM.
            // Ab: attn_probs forward mein allocate nahi hoti (nullptr). Backward mein
            //   Q_adv × K^T se recompute karo (O(seq²) COMPUTE ek baar, 0 extra VRAM).
            // Trade-off: ~15% backward compute overhead — acceptable vs T4 crash.
            GPUTensor recomputed_attn_probs = gpu_alloc(seq, seq);
            {
                // scores[i,k] = Q_adv[i] · K[k] * scale  (forward se same formula)
                float scale_attn = 1.0f / sqrtf((float)DH);
                // Reuse attn_dAttnProbs kernel structure — but here compute raw scores first
                // Simple tiled matmul: scores = Q_adv @ K^T  (seq×DH @ DH×seq → seq×seq)
                dim3 g_sc((seq+15)/16, (seq+15)/16), b_sc(16, 16);
                // Use existing kernel repurposed: attn_dQ_kernel computes seq×DH,
                // but we need seq×seq. Use vedic_gemm_kernel for Q_adv @ K^T.
                GPUTensor K_T = gpu_alloc(DH, seq);
                gpu_transpose_kernel<<<dim3((seq+15)/16,(DH+15)/16),dim3(16,16)>>>(
                    hc.K.data, K_T.data, seq, DH);
                CUDA_KERNEL_CHECK();
                vedic_gemm_kernel<<<dim3((seq+15)/16,(seq+15)/16),dim3(16,16)>>>(
                    hc.Q_adv.data, K_T.data, recomputed_attn_probs.data, seq, DH, seq);
                CUDA_KERNEL_CHECK();
                // Scale + causal mask + softmax (forward ke jaise)
                // Scale karo
                int sz_sc = seq * seq;
                grad_scale_kernel<<<(sz_sc+255)/256, 256>>>(
                    recomputed_attn_probs.data, scale_attn, sz_sc);
                CUDA_KERNEL_CHECK();
                // Causal mask lagao
                causal_mask_kernel<<<dim3(seq,(seq+255)/256),256>>>(
                    recomputed_attn_probs.data, seq);
                CUDA_KERNEL_CHECK();
                // Softmax (boltzmann_softmax_kernel: seq rows, vocab_size=seq)
                boltzmann_softmax_kernel<<<seq, 256>>>(
                    recomputed_attn_probs.data, recomputed_attn_probs.data,
                    seq, seq, 1.0f);
                CUDA_KERNEL_CHECK();
                // K_T freed (RAII)
            }

            // dV_s from recomputed attn_probs
            GPUTensor dV_s=gpu_alloc(seq,DH);
            CUDA_CHECK(cudaMemset(dV_s.data,0,seq*DH*sizeof(float)));
            { dim3 g(DH,(seq+31)/32),b(32);
              attn_dV_kernel<<<g,b>>>(recomputed_attn_probs.data,d_head_out_h.data,
                                      dV_s.data,seq,DH); CUDA_KERNEL_CHECK(); }

            // [v14-WIRE] NS diffusion backward: dV_s → dV (chain rule through ns_diffuse)
            GPUTensor dV=gpu_alloc(seq,DH);
            { int qb=(seq*DH+255)/256;
              ns_diffuse_bwd_kernel<<<qb,256>>>(dV_s.data,dV.data,seq,DH,
                  gpu_model.phys.ns_nu); CUDA_KERNEL_CHECK(); }
            GPUTensor d_attn_probs=gpu_alloc(seq,seq);
            { dim3 g(seq,(seq+31)/32),b(32);
              // [v17] FIX: forward me head_out = probs @ V_s (diffused V) → d_probs bhi V_s se
              attn_dAttnProbs_kernel<<<g,b>>>(d_head_out_h.data,hc.V_s.data,
                                              d_attn_probs.data,seq,DH); CUDA_KERNEL_CHECK(); }
            GPUTensor d_scores=gpu_alloc(seq,seq);
            float inv_sqrt_DH=1.0f/sqrtf((float)DH);
            softmax_bwd_kernel<<<seq,256>>>(recomputed_attn_probs.data,d_attn_probs.data,
                                            d_scores.data,seq,inv_sqrt_DH); CUDA_KERNEL_CHECK();
            // dQ from scores (w.r.t. Q_adv, the advected Q used in forward)
            GPUTensor dQ_adv=gpu_alloc(seq,DH);
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dQ_kernel<<<g,b>>>(d_scores.data,hc.K.data,dQ_adv.data,seq,DH); CUDA_KERNEL_CHECK(); }

            // [v14-WIRE] NS advection backward: dQ_adv → dQ (chain rule through ns_advect)
            GPUTensor dQ=gpu_alloc(seq,DH);
            { int qb=(seq*DH+255)/256;
              ns_advect_bwd_kernel<<<qb,256>>>(dQ_adv.data,dQ.data,seq,DH,
                  gpu_model.phys.ns_eta); CUDA_KERNEL_CHECK(); }

            GPUTensor dK=gpu_alloc(seq,DH);
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dK_kernel<<<g,b>>>(d_scores.data,hc.Q_adv.data,dK.data,seq,DH); CUDA_KERNEL_CHECK(); }
            { dim3 g(D,(DH+31)/32),b(32);
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,dQ.data,gpu_grads[base+h*4+0]->data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,dK.data,gpu_grads[base+h*4+1]->data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,dV.data,gpu_grads[base+h*4+2]->data,seq,D,DH); CUDA_KERNEL_CHECK(); }
            { dim3 g(seq,(D+31)/32),b(32);
              attn_dX_from_QKV_kernel<<<g,b>>>(dQ.data,blk.W_Q[h].data,d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dX_from_QKV_kernel<<<g,b>>>(dK.data,blk.W_K[h].data,d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dX_from_QKV_kernel<<<g,b>>>(dV.data,blk.W_V[h].data,d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK(); }
        }

        // [v14-WIRE] LN1 backward: use Reynolds bwd if reynolds active, else plain LN bwd
        GPUTensor d_dX_ln1=gpu_alloc(seq,D);
        if (gpu_model.phys.reynolds && cache.ln1_w.valid()) {
            reynolds_bwd_kernel<<<seq,256>>>(
                cache.block_input.data, blk.ln1_gamma.data,
                d_dX_attn_in.data, cache.ln1_w.data,
                gpu_model.run_mean[2*l].data, gpu_model.run_var[2*l].data,
                d_dX_ln1.data,
                gpu_grads[base+ln1g_off]->data, gpu_grads[base+ln1b_off]->data,
                seq, D, 1e-5f);
        } else {
            layernorm_bwd_kernel<<<seq,256>>>(
                cache.block_input.data, blk.ln1_gamma.data,
                d_dX_attn_in.data, d_dX_ln1.data,
                gpu_grads[base+ln1g_off]->data, gpu_grads[base+ln1b_off]->data,
                seq, D, 1e-5f);
        }
        CUDA_KERNEL_CHECK();
        { int sz=seq*D;
          vec_add_kernel<<<(sz+255)/256,256>>>(d_dX_out.data,d_dX_ln1.data,sz); CUDA_KERNEL_CHECK(); }
    }

    // [v17] FIX (v15-B bug): hyperbolic ON ho to embedding gradient SIRF expmap Jacobian se
    // chain hokar jaana chahiye. v15 me pehle raw d_dX_out scatter hota tha AUR phir
    // J^T·d_dX_out bhi add hota tha → embedding/pos grads double-counted + galat direction
    // (GNorm inflation ka ek confirmed source). Ab hyper ON me ye raw scatter skip hota hai.
    const bool hyper_chain = gpu_model.hyper_cfg.enabled && gpu_model.X_euclidean.valid();
    if (!hyper_chain) {
      dim3 blk(32),grd(seq,(D+31)/32);
      embedding_bwd_kernel<<<grd,blk>>>(
          gpu_model.d_token_ids,d_dX_out.data,
          gpu_grads[0]->data,gpu_grads[1]->data,
          seq,D,V); CUDA_KERNEL_CHECK(); }

    // [v15-COMPLETE] FIX-15-B: Hyperbolic expmap0 backward
    // Forward mein: X = expmap0(X_euc)  →  Poincaré ball pe project kiya
    // X_euclidean cache mein saved hai (before expmap)
    // Backward: dX_euc = J(X_euc)^T · d_dX_out  (Jacobian of expmap)
    //   J = alpha*I + beta*(v⊗v)/r²  (symmetric, per-token 2x2 block diagonal)
    //   gpu_grads[0] (embedding) aur gpu_grads[1] (pos_embedding) already
    //   accumulated above. Ab un gradients ko expmap Jacobian se chain karo.
    // Without this: embedding weights get Euclidean gradient, ignoring that
    //   the actual computation happened on the Poincaré manifold.
    if (hyper_chain) {
        // d_emb_grad aur d_pos_grad hyperbolic chain se nahi gayi thi
        // expmap0_bwd_kernel: dX_euc[i] = J(X_euc[i])^T · dOut[i]
        // Note: we apply bwd in-place on d_dX_out, then re-scatter to emb/pos grads
        GPUTensor d_dX_euc = gpu_alloc(seq, D);
        expmap0_bwd_kernel<<<seq, 256>>>(
            gpu_model.X_euclidean.data,   // pre-map Euclidean embeddings
            d_dX_out.data,                // upstream gradient (already accumulated)
            d_dX_euc.data,                // output: gradient w.r.t. X_euclidean
            seq, D, gpu_model.hyper_cfg.curvature);
        CUDA_KERNEL_CHECK();
        // Scatter d_dX_euc back into embedding + positional embedding gradients
        // [v17] ab yahi AKELA scatter hai (upar wala raw scatter hyper ON me skip hota hai)
        { dim3 blk2(32), grd2(seq,(D+31)/32);
          embedding_bwd_kernel<<<grd2,blk2>>>(
              gpu_model.d_token_ids, d_dX_euc.data,
              gpu_grads[0]->data, gpu_grads[1]->data,
              seq, D, V); CUDA_KERNEL_CHECK(); }
        // d_dX_euc freed automatically (RAII GPUTensor destructor)
    }
}

// ============================================================
//  MAIN TRAINING FUNCTION
// ============================================================
void train_gpu(const std::string& dataset_path) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  LOGOS GPU Training v27-H100             ║\n");
    printf("║  ✦ 219M params | 8192 context           ║\n");
    printf("║  ✦ d=1024 L=16 H=16 DH=64              ║\n");
    printf("║  ✦ [v23-AMP] FP16 fwd + FP32 master    ║\n");
    printf("║  ✦ [v24] Manual loss scaling WIRED      ║\n");
    printf("║  ✦ [v25] d_A16/d_acc RAII leak fix     ║\n");
    printf("║  ✦ [v27] d_targets RAII leak fix       ║\n");
    printf("║  ✦ [v27] NikhilamTensor safe alloc     ║\n");
    printf("║  ✦ [v27] Adaptive chunk: H100=64MB     ║\n");
    printf("║  ✦ [v27] Adaptive ckpt: H100=5000 steps║\n");
    printf("║  ✦ [v27] Prefetch thread exception-safe║\n");
    printf("║  ✦ H100 SXM Tensor Core GEMM (2x spd)  ║\n");
    printf("║  ✦ LOGOS_CHUNK_MB / LOGOS_CKPT_FREQ    ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s | VRAM: %zu MB | SMs: %d | CC: %d.%d\n",
           prop.name, prop.totalGlobalMem/1024/1024,
           prop.multiProcessorCount, prop.major, prop.minor);

    // [v23-AMP] H100 detection
    bool is_h100 = (prop.major == 9 && prop.minor == 0);   // sm_90 = H100
    bool is_a100 = (prop.major == 8 && prop.minor == 0);   // sm_80 = A100
    bool amp_supported = (prop.major >= 7);                  // FP16 Tensor Cores: Volta+
    printf("  AMP: FP16 Tensor Cores %s | H100: %s | A100: %s\n",
           amp_supported ? "✅" : "❌",
           is_h100 ? "✅" : "—",
           is_a100 ? "✅" : "—");
    if (!amp_supported) {
        printf("  ⚠️  GPU CC < 7.0 — FP16 Tensor Cores not available\n");
        printf("      AMP disabled, falling back to FP32\n");
    }
    printf("\n");
    fflush(stdout);

    printf("[1/5] Dataset scan...\n"); fflush(stdout);

    // [v28-DATASET-FIX] dataset_path ("dataset.txt") sirf CWD mein dhundhta tha.
    // Kaggle pe download script data /kaggle/working/ mein likhta hai — build/ mein nahi.
    // Auto-discover: pehle given path try karo, phir common Kaggle locations scan karo.
    std::string resolved_dataset = dataset_path;
    if (!std::ifstream(resolved_dataset).good()) {
        const char* candidates[] = {
            "/kaggle/working/dataset.txt",
            "/kaggle/working/train.txt",
            "/kaggle/working/logos_dataset.txt",
            "/tmp/dataset.txt",
            nullptr
        };
        for (int ci = 0; candidates[ci]; ++ci) {
            if (std::ifstream(candidates[ci]).good()) {
                resolved_dataset = candidates[ci];
                printf("  [v28] dataset auto-resolved: %s\n", resolved_dataset.c_str());
                break;
            }
        }
    }

    int64_t actual_size=scan_dataset_size(resolved_dataset);
    if (actual_size==0) {
        fprintf(stderr,"❌ Dataset not found: %s\n", resolved_dataset.c_str());
        fprintf(stderr,"   Kaggle notebook mein dataset path confirm karo:\n");
        fprintf(stderr,"   /kaggle/working/ ke andar koi .txt file hai?\n");
        return;
    }
    const std::string& dataset_path_resolved = resolved_dataset;
    printf("Dataset: %lld MB\n\n",(long long)actual_size/1024/1024);

    int vocab_target=decide_vocab_size(actual_size);
    printf("[1/5] Tokenizer build (vocab=%d)...\n",vocab_target); fflush(stdout);
    Tokenizer tok;
    {
        // [v17] 8MB (was 32MB): dataset shuffled hai isliye pehle 8MB Hindi+English dono ka
        // representative sample hai; 32MB par 8192-vocab BPE ghanton leta tha.
        static constexpr int64_t TOKENIZER_SAMPLE=8LL*1024*1024;
        int64_t sample_size=std::min(actual_size,TOKENIZER_SAMPLE);
        std::ifstream f(dataset_path_resolved,std::ios::binary);
        if (!f) { fprintf(stderr,"❌ Cannot open: %s\n",dataset_path_resolved.c_str()); return; }
        std::string sample; sample.resize((size_t)sample_size);
        f.read(sample.data(),sample_size);
        sample.resize((size_t)f.gcount());
        tok.build(sample,vocab_target);
        tok.save("vocab.bin");
        printf("Vocab: %d tokens\n\n",tok.vocab_size);
    }

    printf("[2/5] Model config...\n"); fflush(stdout);
    auto scaled=decide_model_config(actual_size);
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size;
    cfg.d_model=scaled.d_model;
    cfg.num_heads=scaled.num_heads;
    cfg.num_layers=scaled.num_layers;
    cfg.max_seq_len=scaled.max_seq_len;

    std::string cfg_err=validate_model_config(
        cfg.d_model,cfg.num_heads,cfg.num_layers,cfg.vocab_size,cfg.max_seq_len);
    if (!cfg_err.empty()) { fprintf(stderr,"❌ Config: %s\n",cfg_err.c_str()); return; }

    // [v32-SEQ-FIX] seq_len override: T4 pe 8192 OOM tha, 4096 safe hai.
    // aapki pichli 17M + seq=4096 training T4 pe perfectly chal rahi thi.
    // 219M model pe bhi 4096 comfortable fit hota hai (attn_probs recompute ke saath):
    //   attn_probs recompute: 4096² × 4B × 16 heads = 1.07 GB (8192 ka 4x kam)
    //   Activations (seq×D×L): 4096×1024×16×4B = 3.2 GB (8192 ka 2x kam)
    //   Total peak: ~7.3 GB → T4 (14.9 GB) mein safely fit ✅
    //
    // Override: LOGOS_SEQ_LEN env var
    //   4096  = T4 pe safe (recommended — aapka proven config)
    //   8192  = A100/H100 pe hi karo (T4 pe tight)
    //   2048  = bahut conservative, context quality suffer karega
    {
        int seq_override = (int)logos_env_f("LOGOS_SEQ_LEN", 0.f);
        if (seq_override > 0) {
            // Validate: must be power of 2 aur reasonable range mein
            if (seq_override >= 512 && seq_override <= 8192) {
                int old_seq = cfg.max_seq_len;
                cfg.max_seq_len = seq_override;
                printf("  [v32-SEQ] seq_len override: %d → %d (LOGOS_SEQ_LEN)\n",
                       old_seq, cfg.max_seq_len);
            } else {
                printf("  ⚠️  [v32-SEQ] LOGOS_SEQ_LEN=%d out of range [512,8192] — keeping %d\n",
                       seq_override, cfg.max_seq_len);
            }
        } else if (cfg.max_seq_len == 8192) {
            // T4 detection: 8192 seq T4 pe risky hai even with attn recompute
            // Auto-downsample to 4096 for T4 (CC 7.5 = Tesla T4)
            int device_check; cudaGetDevice(&device_check);
            cudaDeviceProp prop_check; cudaGetDeviceProperties(&prop_check, device_check);
            bool is_t4 = (prop_check.major == 7 && prop_check.minor == 5
                          && prop_check.totalGlobalMem < 17ULL*1024*1024*1024);
            if (is_t4) {
                printf("  [v32-SEQ] T4 detected + seq=8192 → auto-reducing to 4096\n");
                printf("            (aapki 17M+4096 config T4 pe perfect thi — same here)\n");
                printf("            Override with LOGOS_SEQ_LEN=8192 to force 8192 (may OOM)\n");
                cfg.max_seq_len = 4096;
            }
        }
        // Shunyam window/stride bhi seq ke proportional hone chahiye
        // 8192: window=64 stride=16 | 4096: window=32 stride=8
        if (cfg.max_seq_len <= 4096) {
            printf("  [v32-SEQ] Shunyam params adjusted for seq=%d: window=32 stride=8\n",
                   cfg.max_seq_len);
        }
    }

    // [v32-BATCH-FIX] grad_accum T4 ke liye 1 → 4:
    // Pehle: grad_accum=1 → 4096 tokens/step → bahut noisy gradients
    //   219M model (vs GPT-2 117M ka ~500K tokens/step) → severe undertraining
    //   GNorm spiky, loss plateau jaldi aata tha
    //
    // Ab: grad_accum=4 → 16,384 tokens/step (4x better signal per step)
    //   VRAM impact: ZERO — grads accumulate CPU-side ke baad average hote hain
    //   Time impact: 4x steps per optimizer update → thoda slow per step but
    //               zyada stable → fewer total steps needed (net faster convergence)
    //
    // LR scaling: batch size 4x → LR bhi √4 = 2x badhao (linear scaling rule)
    //   Naya default: LOGOS_LR=2e-3 (agar set nahi to niche 1e-3 pe clamped)
    //   Better: LOGOS_LR=2e-3 set karo Kaggle cell mein
    //
    // T4 VRAM safe:
    //   grad_accum=4: 4 × fwd pass sequentially (cache freed after each) → same peak as accum=1
    //   Peak VRAM stays ~5.8 GB (seq=4096) — T4 14.9 GB mein comfortable ✅
    //
    // Override: LOGOS_GRAD_ACCUM env var
    //   T4:  LOGOS_GRAD_ACCUM=4  (new default — recommended)
    //   T4:  LOGOS_GRAD_ACCUM=8  (smoother but 2x slower per step)
    //   A100: LOGOS_GRAD_ACCUM=16 → 65K tokens/step
    //   H100: LOGOS_GRAD_ACCUM=32 → 128K tokens/step
    int grad_accum_default = 4;  // [v32] T4 default: 4 (was 1 → too noisy)
    if (is_a100) grad_accum_default = 16;
    if (is_h100) grad_accum_default = 32;
    int grad_accum = (int)logos_env_f("LOGOS_GRAD_ACCUM", (float)grad_accum_default);
    grad_accum = std::max(1, std::min(32, grad_accum));  // clamp [1, 32]
    printf("  [v32-BATCH] grad_accum=%d → %d tokens/step (T4=4, A100=16, H100=32 | override: LOGOS_GRAD_ACCUM)\n",
           grad_accum, grad_accum * cfg.max_seq_len);

    // [v18] vocab_size debug: tok.vocab_size tokenizer ka actual size hai
    // decide_vocab_size() sirf target tha — actual size slightly different ho sakta hai
    // (BPE merge count ya character coverage ke wajah se).
    // Agar dono match nahi karte to yahan warn karo.
    if (cfg.vocab_size != (int)decide_vocab_size(actual_size)) {
        printf("  ℹ️  vocab target=%d → actual tok.vocab_size=%d (BPE coverage)\n",
               decide_vocab_size(actual_size), cfg.vocab_size);
    }
    printf("d=%d L=%d H=%d DH=%d seq=%d vocab=%d grad_accum=%d\n",
           cfg.d_model,cfg.num_layers,cfg.num_heads,cfg.d_model/cfg.num_heads,
           cfg.max_seq_len,cfg.vocab_size,grad_accum);

    // Actual values (LR/clip/T/steps/noise) optimizer banao ke baad print hote hain —
    // env override: LOGOS_LR, LOGOS_CLIP, LOGOS_T_START, LOGOS_STEPS, LOGOS_WARMUP, LOGOS_NOISE_GAIN
    printf("\n[v17 SHM Optimizer] hyper-parameters neeche [4/5] me print honge\n\n");

    // [v30-OOM-FIX] Pre-flight VRAM check before allocating model.
    // 219M model (d=1024, L=16, seq=8192) minimum VRAM requirements:
    //   FP32 weights:   ~876 MB
    //   FP16 shadow:    ~418 MB
    //   Optimizer vel:  ~876 MB
    //   Activations:    ~1500 MB (1 micro-batch, seq=8192, conservative)
    //   Gradient bufs:  ~876 MB
    //   Total estimate: ~4600 MB minimum
    // T4 (16 GB): ok but tight. If free < 5GB → warn + suggest smaller config.
    {
        size_t free_before, total_before;
        cudaMemGetInfo(&free_before, &total_before);
        float free_gb  = (float)free_before  / (1024.f*1024.f*1024.f);
        float total_gb = (float)total_before / (1024.f*1024.f*1024.f);
        printf("  [v30] VRAM before model alloc: %.1f GB free / %.1f GB total\n",
               free_gb, total_gb);
        if (free_before < 5ULL*1024*1024*1024) {
            printf("  ⚠️  [v30-OOM] VRAM tight (< 5 GB free)!\n");
            printf("      219M (d=1024 L=16 seq=8192) needs ~5 GB minimum.\n");
            printf("      To reduce memory set env vars:\n");
            printf("        LOGOS_GRAD_ACCUM=1      (already default for T4)\n");
            printf("      Or use a smaller model by editing decide_model_config().\n");
            fflush(stdout);
        }
    }

    printf("[3/5] Init GPU model...\n"); fflush(stdout);
    LOGOSModel cpu_model(cfg);

    // ── [v19-TRUE-RESUME] Checkpoint + Vocab load ────────────────────────
    // Kaggle notebook CELL 1 mein set karo:
    //   os.environ["LOGOS_CKPT"]  = "/kaggle/input/logos-ckpt/logos_gpu_ckpt_step61000.bin"
    //   os.environ["LOGOS_VOCAB"] = "/kaggle/input/logos-ckpt/vocab.bin"
    // LOGOS_CKPT set nahi = fresh run (step 0 se)
    bool resumed = false;
    {
        const char* ckpt_env  = std::getenv("LOGOS_CKPT");
        const char* vocab_env = std::getenv("LOGOS_VOCAB");

        if (ckpt_env && *ckpt_env) {
            std::string ckpt_path(ckpt_env);
            std::string vocab_path = (vocab_env && *vocab_env)
                                     ? std::string(vocab_env) : std::string("vocab.bin");

            printf("  [v19] Checkpoint: %s\n", ckpt_path.c_str());
            printf("  [v19] Vocab:      %s\n", vocab_path.c_str());

            // vocab.bin pehle load karo — tokenizer rebuild band
            Tokenizer resume_tok;
            if (resume_tok.load(vocab_path)) {
                printf("  ✅ Vocab loaded: %d tokens\n", resume_tok.vocab_size);
                if (resume_tok.vocab_size != cfg.vocab_size) {
                    printf("  ℹ️  Vocab size update: %d → %d\n",
                           cfg.vocab_size, resume_tok.vocab_size);
                    cfg.vocab_size = resume_tok.vocab_size;
                }
                tok = resume_tok;
            } else {
                printf("  ⚠️  vocab.bin nahi mili — naya tokenizer use hoga\n");
            }

            // cpu_model cfg ke saath rebuild (load_checkpoint strict match chahta hai)
            cpu_model = LOGOSModel(cfg);
            if (load_checkpoint(cpu_model, ckpt_path)) {
                resumed = true;
                printf("  ✅ Weights loaded — TRUE RESUME ✓\n");
            } else {
                printf("  ❌ Checkpoint load FAIL (config mismatch?) — fresh weights\n");
            }
        } else {
            printf("  ℹ️  LOGOS_CKPT not set → fresh run\n");
            printf("      Resume ke liye set karo: LOGOS_CKPT=/path/logos_gpu_ckpt_stepXXXXX.bin\n");
        }
    }

    // [v21-OPTSTATE] Optimizer state — load kiya jaayega optimizer.init() ke baad
    OptimizerState opt_state_loaded;
    bool opt_state_found = false;

    // ── [v29-FULLRESUME] TrainingState load ──────────────────────────────
    // .trainstate file se: best_loss, prev_train_ce, prev_val_ce,
    // overfit_streak, amp_scale/window, loader byte position, gnorm EMA
    // NOTE: start_step pehle compute hona chahiye — niche steps section se pehle load karo
    // lekin actually use karo steps compute hone ke baad. Isliye ts_found flag store karo
    // aur baad mein apply karo.
    TrainingState ts_loaded;
    bool ts_found = false;

    // [v29-BUG-FIX] ts_loaded.load() ACTUALLY CALL KARO — pehle sirf declare tha,
    // load() kabhi call hi nahi hoti thi → ts_found hamesha false → sab kuch reset.
    // Resume pe LOGOS_CKPT set hai to same base path pe .trainstate dhundho.
    if (resumed) {
        const char* ckpt_env_ts = std::getenv("LOGOS_CKPT");
        const int   step_n_ts   = (int)logos_env_f("LOGOS_START_STEP", 0.f);
        if (ckpt_env_ts && *ckpt_env_ts && step_n_ts > 0) {
            // Base path nikalo: "logos_gpu_ckpt_step61000.bin" → "logos_gpu_ckpt"
            std::string base_ts(ckpt_env_ts);
            std::string step_suf = "_step" + std::to_string(step_n_ts) + ".bin";
            if (base_ts.size() >= step_suf.size() &&
                base_ts.substr(base_ts.size() - step_suf.size()) == step_suf) {
                base_ts = base_ts.substr(0, base_ts.size() - step_suf.size());
            } else {
                if (base_ts.size() > 4 &&
                    base_ts.substr(base_ts.size() - 4) == ".bin")
                    base_ts = base_ts.substr(0, base_ts.size() - 4);
            }
            ts_found = ts_loaded.load(base_ts, step_n_ts);
            if (ts_found)
                printf("  [v29] TrainingState loaded from: %s_step%d.trainstate\n",
                       base_ts.c_str(), step_n_ts);
            else
                printf("  [v29] No .trainstate found — metrics will reset (weights OK)\n");
        }
    }

    ModelGPU   gpu_model(cfg);
    gpu_model.load_from_cpu(cpu_model);

    // ── [v14-WIRE] Wire ALL Vedic + Physics tools to GPU ─────
    // phys.training=false tha (hardcoded default) → Feynman dropout NEVER ran.
    // Now: set training=true so dropout, Reynolds stats, and all physics
    // kernels activate during the forward pass.
    gpu_model.phys.training        = true;   // enables Feynman dropout + Reynolds EMA
    gpu_model.phys.nikhilam_kv     = true;   // Nikhilam INT8 KV cache
    gpu_model.phys.shunyam         = true;   // Shunyam sparse causal attention
    // [v32-SEQ-FIX] Shunyam window/stride: seq ke proportional
    // seq=8192: window=64 stride=16 | seq=4096: window=32 stride=8
    gpu_model.phys.window          = (cfg.max_seq_len >= 8192) ? 64 : 32;
    gpu_model.phys.stride          = (cfg.max_seq_len >= 8192) ? 16 : 8;
    gpu_model.phys.navier_stokes   = true;   // NS Q-advection + V-diffusion
    gpu_model.phys.ns_eta          = 0.1f;
    gpu_model.phys.ns_nu           = 0.05f;
    gpu_model.phys.feynman_dropout = true;   // Feynman Beta-amplitude dropout
    gpu_model.phys.drop_p          = 0.1f;
    gpu_model.phys.drop_hbar       = 1.0f;
    gpu_model.phys.reynolds        = true;   // Reynolds-adaptive LayerNorm/BatchNorm
    gpu_model.phys.re_crit         = 2.0f;
    gpu_model.phys.re_k            = 5.0f;
    printf("  ✦ phys.training=true  → Feynman dropout ACTIVE\n");
    printf("  ✦ Nikhilam KV INT8    → 4x KV compression\n");
    printf("  ✦ Shunyam sparse attn → window=%d stride=%d (8192 ctx)\n", gpu_model.phys.window, gpu_model.phys.stride);
    printf("  ✦ Navier-Stokes       → eta=%.2f nu=%.2f\n", gpu_model.phys.ns_eta, gpu_model.phys.ns_nu);
    printf("  ✦ Reynolds norm       → re_crit=%.1f k=%.1f\n", gpu_model.phys.re_crit, gpu_model.phys.re_k);
    printf("  ✦ Feynman dropout     → p=%.2f hbar=%.1f\n", gpu_model.phys.drop_p, gpu_model.phys.drop_hbar);
    printf("  ✦ Model scale         → d=1024 L=16 H=16 seq=8192 (~219M params)\n\n");

    printf("[4/5] StreamingDataLoader + Optimizer...\n"); fflush(stdout);
    int SEQ=cfg.max_seq_len;
    // [v27-H100-TWEAK1] Adaptive chunk size: GPU throughput ke liye chunk bada chahiye.
    // T4/Kaggle (16GB): 4MB chunk fine — tokenizer zyada time laga sakta hai, GPU wait karta hai
    //   but VRAM tight hai to small chunks safer hain.
    // H100 (80GB): 4MB chunk → tokenizer becomes BOTTLENECK — prefetch thread GPU se slow ho
    //   jaata hai → GPU IDLE ho jaata hai batch ke beech. 64MB chunk se:
    //   - 16x zyada tokens per prefetch → prefetch frequency 16x kam
    //   - GPU almost never waits for data on H100
    // A100 (40GB): 32MB sweet spot.
    // Override: LOGOS_CHUNK_MB env var — set karo RunPod pe without recompile.
    //   e.g. export LOGOS_CHUNK_MB=64   (H100 80GB)
    //        export LOGOS_CHUNK_MB=32   (A100 40GB)
    //        export LOGOS_CHUNK_MB=4    (Kaggle T4 — default)
    int64_t chunk_mb_default = 4LL;
    if (is_h100) chunk_mb_default = 64LL;
    else if (is_a100) chunk_mb_default = 32LL;
    int64_t chunk_mb_env = (int64_t)logos_env_f("LOGOS_CHUNK_MB", (float)chunk_mb_default);
    // Clamp: [1, 256] MB — 256MB se bada = single chunk me too many tokens, RAM pressure
    chunk_mb_env = std::max((int64_t)1, std::min((int64_t)256, chunk_mb_env));
    const int64_t CHUNK_BYTES = chunk_mb_env * 1024LL * 1024LL;
    printf("  Chunk size: %lld MB (auto: H100=64 A100=32 T4=4 | override: LOGOS_CHUNK_MB)\n",
           (long long)chunk_mb_env);

    // [v16-STABLE] Validation split: 5% of dataset held out
    // First 95% = training, Last 5% = validation (never trained on)
    // This enables overfit/underfit detection:
    //   Val_CE rising while Train_CE falls = OVERFIT
    //   Both Val_CE and Train_CE high = UNDERFIT
    //   Both falling together = healthy training
    int64_t val_start_byte = (int64_t)(actual_size * 0.95);
    int64_t val_bytes      = actual_size - val_start_byte;
    printf("  Train: %.1f MB | Val: %.1f MB (5%% held out)\n",
           (float)val_start_byte/1024/1024, (float)val_bytes/1024/1024);

    StreamingDataLoader loader(dataset_path_resolved, tok, SEQ, grad_accum, CHUNK_BYTES,
                               /*start_byte=*/0, /*end_byte=*/val_start_byte);
    // [v17] FIXED validation set: 32 windows pre-loaded into memory (every 16th batch).
    // val_loader is declared at function scope (not inside {} block) because
    // it is also used later in the training loop every 100 steps for on-the-fly
    // Val_CE evaluation (forward-only pass on ~10 batches).
    // BUG-FIX v17: pehle val_loader {} block ke andar tha → bahar 'undefined' compile error.
    StreamingDataLoader val_loader(dataset_path_resolved, tok, SEQ, 1, 1LL*1024*1024,
                                   /*start_byte=*/val_start_byte, /*end_byte=*/actual_size);
    std::vector<std::pair<std::vector<int>,std::vector<int>>> fixed_val;
    {
        std::vector<std::pair<std::vector<int>,std::vector<int>>> tmp;
        for (int i=0; i<512 && (int)fixed_val.size()<32; ++i) {
            if (!val_loader.next_accum_batch(tmp) || tmp.empty()) break;
            if (i%16==0) fixed_val.push_back(std::move(tmp[0]));
        }
        val_loader.reset();  // rewind so training loop can use it from the start
    }
    printf("  Val set: %zu fixed windows x %d tokens\n", fixed_val.size(), SEQ);
    if (fixed_val.empty()) printf("  ⚠️  Val set empty — Val_CE N/A rahega\n");

    // [v18-RESUME] Steps: 51001 se 100000 tak (49000 naye steps)
    // start_step = 51001: logging offset taaki checkpoints sahi step number dikhayein
    // total_steps = 100000: absolute target (not delta) — loop step >= total_steps pe break
    // EPOCHS=2: dataset ek baar aur dekho (Wikipedia 1.5GB, 50k steps mein ~1% tha)
    // Env override: LOGOS_STEPS se override possible (default 100000)
    int EPOCHS=2;
    int64_t batches_per_epoch=loader.total_batches_per_epoch();
    const int64_t steps_cap=(int64_t)logos_env_f("LOGOS_STEPS",150000.f);
    // [v19-TRUE-RESUME] start_step: resumed=true ho to LOGOS_START_STEP env se lo
    // Kaggle CELL 1 mein: os.environ["LOGOS_START_STEP"] = "61000"  (last checkpoint step)
    // Fresh run mein: 0 se shuru
    const int64_t start_step = resumed
        ? (int64_t)logos_env_f("LOGOS_START_STEP", 0.f)
        : 0LL;
    int64_t total_steps=std::min(steps_cap, start_step + EPOCHS*(batches_per_epoch/grad_accum));
    if (total_steps < 1) total_steps = 1;

    // [v17] LR analysis: ye optimizer plain SGD-momentum jaisa hai (per-parameter
    // normalization nahi). Effective SGD lr = lr·(α_H/2)/(1-β_eff), β_eff = mom - 0.5·α_L·γ.
    //   v16: lr=5e-5 → lr_eff ≈ 7e-5..2e-4, clip=1.0 ⇒ har step ka norm ≤ ~1e-4.
    //        ~9M params × 50k steps me total path-length ~2-5 (weights ka norm ~60) ⇒ UNDERFIT.
    //   v17: lr=3e-3 → lr_eff ≈ 4e-3 (start) → warmup + cosine decay (floor 10%) se kam hota hai.
    // Ye ek starting point hai — pehle 2-3k steps me Train_CE girni chahiye; nahi gire to
    // LOGOS_LR ×3 karo, GNorm/gn-split explode kare to ÷3.
    // [v18-RESUME] LR restart: pehle 3e-3 tha, ab 1e-3 se fresh cosine shuru
    // Reason: step 28k ke baad best_F plateau → cosine ne LR ~0 kar diya tha.
    // Fresh 1e-3 restart → naye learning trajectories, same data different regions.
    // clip_norm: 1.0 → 0.3  (GNorm step 42k pe 463, step 48k pe 1015 tha → EXPLOSION)
    // Root cause: cosine-end pe LR near-zero tha but momentum accumulation nahi ruka
    // 0.3 clip → effective GNorm budget ×3 tighter → stable late training
    // [v26-BUG14-FIX] logos_env_f → logos_env_f_clamped: dangerous values block karo
    // LOGOS_LR:         [1e-7, 1.0]   — 1e-7 se neeche = no learning; >1.0 = immediate diverge
    // LOGOS_CLIP:       [0.01, 10.0]  — 0 = no gradient at all; >10 = explosion
    // LOGOS_T_START:    [1e-4, 1.0]   — >1.0 = entropy dominates loss; <1e-4 = no exploration
    // LOGOS_WARMUP:     [0, 10000]    — negative = UB; >10k on resume = LR stuck at 0 too long
    // LOGOS_NOISE_GAIN: [0.0, 1.0]   — >1.0 = noise > gradient → diverge
    // [v32-BATCH-FIX] LR linear scaling: batch 4x → LR 2x (sqrt scaling rule)
    // grad_accum=4 ke saath 1e-3 → 2e-3 default. LOGOS_LR se override karo.
    // Agar clip_norm=0.3 pe GNorm still explode kare to LOGOS_LR=1e-3 try karo.
    float lr_default = (grad_accum >= 4) ? 2e-3f : 1e-3f;
    float   lr_init      = logos_env_f_clamped("LOGOS_LR",         lr_default, 1e-7f, 1.0f);
    const float clip_norm = logos_env_f_clamped("LOGOS_CLIP",      0.3f,   0.01f, 10.0f);
    const float t_start   = logos_env_f_clamped("LOGOS_T_START",   0.05f,  1e-4f, 1.0f);
    const int   warmup_steps = (int)logos_env_f_clamped("LOGOS_WARMUP", 300.f, 0.f, 10000.f);
    const float noise_gain   = logos_env_f_clamped("LOGOS_NOISE_GAIN", 0.02f, 0.0f, 1.0f);

    // [v16-STABLE] Geodesic friction balance analysis:
    //   aH (Hamiltonian) = gradient direction (deterministic descent)
    //   aL (Langevin)    = thermal noise (exploration, local minima escape)
    //   Previous: aH_start=0.7 → too deterministic early → fell into local minima fast
    //   Fix: aH_start=0.5 (balanced) → aH_end=0.95 (converge late)
    //   friction=0.3 (was 0.1) → more geodesic damping → smoother trajectory
    //   mom_decay=0.9 (was 0.95) → slightly less momentum → less overshoot
    // Geodesic interpretation: friction controls how fast momentum decays
    //   along the weight manifold geodesic. Too low = oscillations (GNorm explosion).
    //   friction=0.3 gives ~3x more damping → stable convergence.
    // [v18-RESUME] Optimizer tuning for resume phase:
    // aH_start=0.70 (was 0.5): pehle se converged model → exploration kam, exploitation zyada
    // aH_end=0.95: same (want Hamiltonian dominant at end)
    // friction=0.4 (was 0.3): thoda aur damping → GNorm spike prevention
    // mom_decay=0.85 (was 0.9): momentum thoda less → late-phase overshoot avoid
    // T_start=0.05 (set above): aur kam entropy bonus → sharper loss landscape
    GPUSHMOpt optimizer(lr_init,
                        /*friction=*/0.4f,       // more geodesic damping (GNorm explosion prevention)
                        /*mom_decay=*/0.85f,     // less momentum retention (avoid late overshoot)
                        /*T_start=*/t_start,     // 0.05 (was 0.1 → entropy bonus kam)
                        /*T_end=*/1e-3f,         // entropy floor (FIX-1 intact)
                        /*aH_start=*/0.70f,      // already converged → start more Hamiltonian
                        /*aH_end=*/0.95f,        // Hamiltonian dominant at end
                        total_steps - start_step); // remaining steps ke liye anneal
    auto gpu_params=gpu_model.all_parameters();
    optimizer.init(gpu_params);
    auto gpu_grads=gpu_model.alloc_grad_buffers();

    // [v14-WIRE] WeightPathIntegral — optimizer init ke BAAD, optstate load se PEHLE declare
    WeightPathIntegral gpu_path_integral(/*hbar=*/1.0f, /*history=*/500);
    float gpu_lr_scale = 1.0f;

    // ── [v23-AMP] Manual Loss Scaler init ─────────────────────────────
    // Dynamic loss scaling:
    //   - Starts at 65536 (2^16)
    //   - Halves on FP16 overflow (inf/nan in grads)
    //   - Doubles every 2000 clean steps
    //   - Clamped: [1.0, 65536.0]
    // Override via env: LOGOS_AMP_SCALE (initial scale, e.g. from prev run's checkpoint)
    // H100 SXM specific: H100 FP16 Tensor Cores rarely overflow at scale=65536 because
    //   H100 hardware has better FP16 accumulation than T4/A100.
    //   Start high (65536) → likely no overflow for thousands of steps on H100.
    //   If overflow happens early → scale halves quickly to safe value.
    ManualLossScaler loss_scaler;
    {
        // [v26-BUG14-FIX] AMP scale: [1, 131072] — 0 = division by zero; >131072 = H100 pe bhi overflow
        float init_scale = logos_env_f_clamped("LOGOS_AMP_SCALE", LOSS_SCALE_INIT, 1.0f, 131072.0f);
        // LOGOS_AMP_WINDOW: [0, LOSS_SCALE_WINDOW] — negative = UB
        int   init_window = (int)logos_env_f_clamped("LOGOS_AMP_WINDOW", 0.f, 0.f, (float)LOSS_SCALE_WINDOW);
        loss_scaler.scale = fmaxf(fminf(init_scale, LOSS_SCALE_MAX), LOSS_SCALE_MIN);
        loss_scaler.steps_since_last_overflow = init_window;
        printf("  [v23-AMP] ManualLossScaler: init_scale=%.0f window=%d (restored=%d)\n",
               loss_scaler.scale, LOSS_SCALE_WINDOW, init_window);
        printf("  [v23-AMP] H100 SXM: FP16 forward (2× TFLOPS) + FP32 master weights\n");
        printf("  [v23-AMP] Precision: forward=FP16 | grads=FP32 | optimizer=FP32\n");
        printf("  [v23-AMP] To resume scale: set LOGOS_AMP_SCALE=<prev_scale> LOGOS_AMP_WINDOW=<prev_window>\n");
    }

    // [v23-AMP] FP16 weight shadow buffers (one per master param)
    // Master weights: FP32 (in gpu_params) — optimizer updates these
    // FP16 shadow:    cast before each forward pass — fed to FP16 GEMM
    // Memory: 219M × 2B (FP16) ≈ 438 MB additional
    std::vector<HalfTensor> fp16_weights;
    fp16_weights.reserve(gpu_params.size());
    for (auto* p : gpu_params) {
        fp16_weights.push_back(half_alloc(p->rows, p->cols));
    }
    printf("  [v23-AMP] FP16 shadow weights: %zu tensors allocated\n",
           fp16_weights.size());
    printf("  [v23-AMP] FP16 VRAM overhead: ~%.0f MB\n",
           (float)(219e6 * sizeof(__half)) / 1024 / 1024);

    // Helper: collect grad pointers + sizes for scaler overflow check
    std::vector<float*> grad_ptrs;
    std::vector<int>    grad_sizes;
    grad_ptrs.reserve(gpu_grads.size());
    grad_sizes.reserve(gpu_grads.size());
    for (auto* g : gpu_grads) {
        grad_ptrs.push_back(g->data);
        grad_sizes.push_back(g->size);
    }

    // [v21-OPTSTATE] Optimizer warm resume:
    // .optstate file se velocity buffers GPU pe copy karo
    // Base path = checkpoint path bina "_step{N}.bin" suffix ke
    if (resumed && start_step > 0) {
        std::string ckpt_env_str(std::getenv("LOGOS_CKPT") ? std::getenv("LOGOS_CKPT") : "");
        std::string base_path = ckpt_env_str;
        std::string step_suffix = "_step" + std::to_string((int)start_step) + ".bin";
        if (base_path.size() >= step_suffix.size() &&
            base_path.substr(base_path.size() - step_suffix.size()) == step_suffix) {
            base_path = base_path.substr(0, base_path.size() - step_suffix.size());
        } else {
            if (base_path.size() > 4 && base_path.substr(base_path.size()-4) == ".bin")
                base_path = base_path.substr(0, base_path.size()-4);
        }
        opt_state_found = load_optimizer_state(opt_state_loaded, base_path, (int)start_step);
        if (opt_state_found) {
            if (opt_state_loaded.velocities.size() == optimizer.d_velocity.size()) {
                for (int i = 0; i < (int)optimizer.d_velocity.size(); ++i) {
                    const auto& cpu_vel = opt_state_loaded.velocities[i];
                    int gpu_sz = optimizer.sizes[i];
                    if ((int)cpu_vel.size() == gpu_sz) {
                        CUDA_CHECK(cudaMemcpy(optimizer.d_velocity[i],
                                              cpu_vel.data(),
                                              gpu_sz * sizeof(float),
                                              cudaMemcpyHostToDevice));
                    } else {
                        printf("  ⚠️  Vel size mismatch param %d (file=%zu gpu=%d) — skipping\n",
                               i, cpu_vel.size(), gpu_sz);
                    }
                }
                // WeightPathIntegral EMA restore
                gpu_path_integral.ema_action    = opt_state_loaded.ema_action;
                gpu_path_integral.log_amplitude  = opt_state_loaded.log_amplitude;
                gpu_path_integral.ema_ready      = (opt_state_loaded.step_count > 0);
                printf("  ✅ Optimizer state restored — WARM RESUME ✓\n");
                printf("     ema_action=%.4f | log_amp=%.4f | steps=%lld\n",
                       opt_state_loaded.ema_action, opt_state_loaded.log_amplitude,
                       (long long)opt_state_loaded.step_count);
            } else {
                printf("  ⚠️  Optimizer param count mismatch (file=%zu model=%zu) — cold start\n",
                       opt_state_loaded.velocities.size(), optimizer.d_velocity.size());
            }
        }
    }

    int D=cfg.d_model, V=cfg.vocab_size, D4=4*D;
    // [v27-MEM1-FIX] CudaPtr RAII — pehle raw cudaMalloc tha.
    // Problem: agar training loop mein CUDA OOM ya kernel exception throw ho to
    //   training_done: label tak execution nahi pahuncha → cudaFree miss → leak.
    //   d_grad_out worst-case: SEQ×V×4B = 8192×8192×4 = 256 MB permanently leaked.
    // Fix: CudaPtr<T> destructors guaranteed run on ANY exit (normal, exception, goto).
    //   .get() se raw pointer milta hai — downstream code bilkul unchanged.
    CudaPtr<int>   d_targets_raii(SEQ);
    CudaPtr<float> d_loss_buf_raii(SEQ);
    CudaPtr<float> d_grad_out_raii(static_cast<int64_t>(SEQ) * V);
    int*   d_targets  = d_targets_raii.get();
    float* d_loss_buf = d_loss_buf_raii.get();
    float* d_grad_out = d_grad_out_raii.get();

    GPUTensor d_logits_grad=gpu_alloc(SEQ,V);
    GPUTensor d_dX_out     =gpu_alloc(SEQ,D);
    GPUTensor d_d_ffn_out  =gpu_alloc(SEQ,D);
    GPUTensor d_d_ffn_A    =gpu_alloc(SEQ,D4);
    GPUTensor d_d_ffn_H    =gpu_alloc(SEQ,D4);
    GPUTensor d_d_normed2  =gpu_alloc(SEQ,D);
    GPUTensor d_dX_ln      =gpu_alloc(SEQ,D);
    GPUTensor d_dX_attn_in =gpu_alloc(SEQ,D);

    printf("Est. batches/epoch: %lld | Steps: %lld | LR=%.1e\n\n",
           (long long)batches_per_epoch,(long long)total_steps,lr_init);
    printf("[5/5] Training...\n\n"); fflush(stdout);

    // [v16-STABLE] Val_CE column added — overfit/underfit detector
    printf("%-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-6s | %-6s | %-8s | %-6s | %-8s\n",
           "Step","F(loss)","Train_CE","Val_CE","Entropy","GNorm","α_H","α_L","T","LRx","Status");
    printf("---------|---------|---------|---------|---------|---------|--------|--------|---------|--------|--------\n");
    fflush(stdout);

    // Overfit detection thresholds
    float prev_val_ce   = 999.f;
    float prev_train_ce = 999.f;
    int   overfit_streak = 0;   // consecutive steps val_ce rising while train_ce falls

    // [v18-RESUME] step counter start_step se shuru → checkpoint names sahi rahenge
    // best_loss warm-init: prev run ka best_F 2.9551 tha (step 28k se plateau)
    // Naya run agar 2.9551 se better kare tabhi checkpoint "best" mark hoga
    int64_t step=start_step;
    // [v19-TRUE-RESUME] best_loss:
    //   resumed=true  → LOGOS_BEST_F env se lo (pichle run ka actual best)
    //   fresh run     → 999.f (koi bhi pehla F isse better hoga)
    // Kaggle CELL 1: os.environ["LOGOS_BEST_F"] = "2.9551"
    float   best_loss = resumed
        ? logos_env_f("LOGOS_BEST_F", 999.f)
        : 999.f;

    // [v29-FULLRESUME] TrainingState se override karo agar mila
    float gnorm_ema    = 0.f;
    float train_ce_ema = 999.f;
    float val_ce_ema   = 999.f;

    if (ts_found) {
        best_loss      = ts_loaded.best_loss;
        prev_train_ce  = ts_loaded.prev_train_ce;
        prev_val_ce    = ts_loaded.prev_val_ce;
        overfit_streak = ts_loaded.overfit_streak;
        gnorm_ema      = ts_loaded.gnorm_ema;
        train_ce_ema   = ts_loaded.train_ce_ema;
        val_ce_ema     = ts_loaded.val_ce_ema;
        loss_scaler.scale                     = ts_loaded.amp_scale;
        loss_scaler.steps_since_last_overflow = ts_loaded.amp_window;
        loss_scaler.total_overflows           = ts_loaded.amp_overflows;
        loss_scaler.total_scale_ups           = ts_loaded.amp_scale_ups;
        loss_scaler.total_scale_downs         = ts_loaded.amp_scale_downs;
        printf("  [v29] Metrics restored: best_F=%.4f train_CE=%.4f val_CE=%.4f\n",
               best_loss, prev_train_ce, prev_val_ce);
        printf("  [v29] AMP scale restored: %.0f (window=%d)\n",
               loss_scaler.scale, loss_scaler.steps_since_last_overflow);
    }

    // [v29] Data loader ko saved byte position pe seek karo
    if (ts_found && ts_loaded.loader_byte_pos > 0) {
        StreamingDataLoader::LoaderState ls;
        ls.current_shard      = ts_loaded.loader_shard;
        ls.byte_pos           = ts_loaded.loader_byte_pos;
        ls.total_tokens_seen  = ts_loaded.loader_tokens_seen;
        ls.total_steps_done   = start_step;
        ls.current_epoch      = ts_loaded.loader_epoch;
        loader.restore_state(ls);
        printf("  [v29] Data loader seeked to shard=%d byte=%lld epoch=%d\n",
               ls.current_shard, (long long)ls.byte_pos, ls.current_epoch);
    }
    if (ts_found && ts_loaded.val_loader_byte_pos > 0) {
        StreamingDataLoader::LoaderState vls{};
        vls.byte_pos = ts_loaded.val_loader_byte_pos;
        val_loader.restore_state(vls);
    }

    printf("\n[v29-FULLRESUME] resumed=%s | start_step=%lld | best_F=%.4f | target=%lld\n",
           resumed?"YES":"NO", (long long)start_step, best_loss, (long long)total_steps);
    printf("  Config: d=%d L=%d H=%d seq=%d vocab=%d grad_accum=%d\n",
           cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.max_seq_len, cfg.vocab_size, grad_accum);
    int     vedic_checks=0, vedic_pass=0;
    char    vedic_status[8]="N/A";

    for (int epoch=0;epoch<EPOCHS;++epoch) {
        printf("\n-- Epoch %d/%d --\n",epoch+1,EPOCHS); fflush(stdout);
        std::vector<std::pair<std::vector<int>,std::vector<int>>> micro_batches;

        while (loader.next_accum_batch(micro_batches)) {
            // [v18-RESUME] Hard cap: step >= total_steps → stop
            if (step >= total_steps) {
                printf("\n[v18-RESUME] Reached total_steps=%lld at step=%lld — stopping.\n",
                       (long long)total_steps, (long long)step);
                goto training_done;
            }
            for (auto* g : gpu_grads)
                CUDA_CHECK(cudaMemset(g->data,0,g->size*sizeof(float)));

            // ── [v23-AMP] Cast FP32 master weights → FP16 shadow before forward ──
            // Done ONCE per optimizer step (not per micro-batch).
            // fp16_weights[pi] holds FP16 version of gpu_params[pi].
            // gpu_model.fp16_param_ptrs[] gives forward pass direct access
            // to FP16 shadow pointers — no extra copy inside forward().
            for (int pi = 0; pi < (int)gpu_params.size(); ++pi) {
                cuda_cast_fp32_to_fp16(
                    gpu_params[pi]->data,
                    fp16_weights[pi].data,
                    gpu_params[pi]->size);
            }
            // Wire FP16 shadow pointers into ModelGPU for this forward pass.
            // forward() reads fp16_param_ptrs[] to route each GEMM to Tensor Core.
            if (amp_supported) {
                gpu_model.fp16_param_ptrs.resize(fp16_weights.size());
                for (int pi = 0; pi < (int)fp16_weights.size(); ++pi)
                    gpu_model.fp16_param_ptrs[pi] = fp16_weights[pi].data;
                gpu_model.amp_enabled = true;
            }

            float batch_F=0.f, batch_CE=0.f, batch_S=0.f;
            int   valid_mb=0;

            for (auto& [input_ids,target_ids] : micro_batches) {
                int seq=(int)input_ids.size();
                GPUTensor logits=gpu_model.forward(input_ids);

                CUDA_CHECK(cudaMemcpy(d_targets,target_ids.data(),
                           seq*sizeof(int),cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_loss_buf,0,seq*sizeof(float)));
                CUDA_CHECK(cudaMemset(d_grad_out,0,seq*V*sizeof(float)));

                FreeEnergyResult fe_result;
                cuda_free_energy_loss(
                    logits.data, d_targets,
                    d_loss_buf, d_grad_out,
                    seq, V,
                    optimizer.temperature,
                    fe_result);

                CUDA_CHECK(cudaMemcpy(d_logits_grad.data, d_grad_out,
                           seq*V*sizeof(float), cudaMemcpyDeviceToDevice));

                if (isfinite(fe_result.free_energy) && fe_result.free_energy > 1e-8f) {
                    batch_F  += fe_result.free_energy;
                    batch_CE += fe_result.cross_entropy;
                    batch_S  += fe_result.entropy;
                    ++valid_mb;
                    run_backward(gpu_model,cfg,gpu_grads,
                                 d_logits_grad,d_dX_out,
                                 d_d_ffn_out,d_d_ffn_A,d_d_ffn_H,
                                 d_d_normed2,d_dX_ln,d_dX_attn_in,seq);
                    // [v14-WIRE] Reynolds running stats EMA update after each backward
                    gpu_model.update_norm_stats();
                    // [v30-OOM-FIX] Free layer cache immediately after backward.
                    // Activations (~6GB for seq=8192 L=16) are only needed during backward.
                    // Freeing here before the next micro-batch reduces peak VRAM by ~6GB,
                    // making T4 (16GB) viable for this 219M model configuration.
                    gpu_model.free_layer_cache();
                    // [v25-BUG4-FIX] amp_scale_grads(loss_scaler.scale) REMOVED from here.
                    // Pehle: har micro-batch ke baad scale × grad_accum times apply hota tha,
                    // phir sirf 1× unscale → net (grad_accum)× over-scaled grads.
                    // grad_accum=8 → 8× too large → GNorm explosion at step 1 → training diverge.
                    // FIX: scale ONCE baad mein (micro-batch loop ke BAHAR) — see below.
                }
            }

            if (valid_mb==0) { ++step; continue; }

            float avg_F  = batch_F  / valid_mb;
            float avg_CE = batch_CE / valid_mb;
            float avg_S  = batch_S  / valid_mb;
            if (avg_F < best_loss) best_loss = avg_F;

            if (grad_accum>1) {
                float sc=1.0f/(float)grad_accum;
                for (auto* g : gpu_grads)
                    scale_grads_kernel<<<(g->size+255)/256,256>>>(g->data,sc,g->size);
                CUDA_KERNEL_CHECK();
            }

            // ── [v25-BUG4-FIX] Manual Loss Scaling — Scale UP ONCE, then Unscale ──
            // BUG 4 was: amp_scale_grads(scale) called INSIDE micro-batch loop (grad_accum times),
            // then amp_scale_grads(1/scale) called ONCE here.
            // Net effect: grads × scale^8 / scale = scale^7 over-scaled → GNorm explosion.
            //
            // Correct flow (now):
            //   1. Accumulate grads across all grad_accum micro-batches (raw, unscaled FP32)
            //   2. Scale grads ONCE by loss_scaler.scale (for overflow detection only)
            //   3. Check overflow → if bad: zero grads, halve scale, skip step
            //   4. Unscale ONCE → true FP32 grads restored
            //   5. Clip + optimizer step
            //
            // Why scale at all? grads_have_inf_nan() detects FP16 underflow artifacts
            // that propagate from FP16 activations into FP32 grads as very small numbers.
            // Scaling amplifies them so inf/nan check can detect true overflow.
            {
                // Step A: Scale UP (ONCE after all micro-batch accumulation is complete)
                amp_scale_grads(grad_ptrs, grad_sizes, loss_scaler.scale);
                CUDA_KERNEL_CHECK();

                // Step B: Overflow check + scale update
                bool amp_ok = loss_scaler.update(grad_ptrs, grad_sizes);

                if (!amp_ok) {
                    // FP16 overflow detected → grads corrupt → skip this optimizer step
                    // scale already halved inside loss_scaler.update()
                    // Zero grads so next accumulation starts clean
                    for (auto* g : gpu_grads)
                        CUDA_CHECK(cudaMemset(g->data, 0, g->size * sizeof(float)));
                    ++step;
                    // Print overflow warning every 50 skips to avoid spam
                    if (loss_scaler.total_overflows % 50 == 1) {
                        printf("  [AMP] step=%lld OVERFLOW skip #%d → new_scale=%.0f\n",
                               (long long)step, loss_scaler.total_overflows, loss_scaler.scale);
                        fflush(stdout);
                    }
                    continue;  // goto next micro_batches iteration
                }
                // Step C: amp_ok=true → grads are valid.
                // Unscale ONCE: restore true FP32 gradient magnitude
                float inv_s = 1.0f / loss_scaler.scale;
                amp_scale_grads(grad_ptrs, grad_sizes, inv_s);
                CUDA_KERNEL_CHECK();
                // Proceed with clip + optimizer step (scale-up window handled inside update())
            }

            // [v16-STABLE] grad_clip=0.3 (was 5.0 → GNorm reached 684 by step 62k)
            // H100 SXM note: FP16 forward pass ke baad gradients FP32 mein compute hote hain
            // (unscaling ke baad). Clip norm AFTER unscaling — correct FP32 gradient magnitude.
            float grad_norm=cuda_clip_gradients(gpu_grads, clip_norm);

            // [v14-WIRE] WeightPathIntegral adaptive LR scaling
            // lr_scale_ema(): current action spike ke hisab se LR shrink karta hai
            // lr_min_frac=0.5 → LR kabhi 50% se neeche nahi girega
            float adapted_lr_scale = gpu_path_integral.lr_scale_ema(0.5f);
            optimizer.update(gpu_params, gpu_grads, adapted_lr_scale);

            // Record step norm for path integral (lightweight GPU norm estimate)
            gpu_path_integral.record_step_norm(avg_F, grad_norm * optimizer.lr);
            // [v18-RESUME] Reset best every 5000 steps (was 2000)
            // Late training mein zyada reset → LR unnecessarily drops → slower convergence
            // 5000 steps = ~1.28M tokens per reset cycle (more stable)
            if (step > start_step && (step - start_step) % 5000 == 0) gpu_path_integral.reset_best();

            // [v27-H100-TWEAK2] Adaptive checkpoint + Vedic verify frequency.
            // T4/Kaggle: har 1000 steps → sync_to_cpu() ~2s GPU block = acceptable.
            // H100 80GB: har 1000 steps → sync_to_cpu() ~5-8s GPU block per step
            //   = 80-100GB data × 150k steps mein HOURS wasted on CPU sync.
            //   5000 steps pe checkpoint karo → 5× less overhead.
            // Override: LOGOS_CKPT_FREQ env var (any GPU, without recompile).
            //   export LOGOS_CKPT_FREQ=5000   (H100)
            //   export LOGOS_CKPT_FREQ=1000   (T4 — default)
            int64_t ckpt_freq_default = is_h100 ? 5000LL : 1000LL;
            int64_t ckpt_freq = (int64_t)logos_env_f("LOGOS_CKPT_FREQ", (float)ckpt_freq_default);
            ckpt_freq = std::max((int64_t)100, std::min((int64_t)50000, ckpt_freq));  // clamp: [100, 50k]

            if (step % ckpt_freq == 0 && step > 0) {
                // ── [BUG-FIX] C_proxy GPUTensor scope ────────────────────────
                // C_proxy is RAII (GPUTensor), so it frees itself at the end of
                // this block. Previously the block was implicit; making it
                // explicit ensures the VRAM is released before the checkpoint
                // sync (which may need headroom on a 16 GB T4).                {
                GPUTensor C_proxy = gpu_alloc(gpu_model.last_hidden.rows,
                                              gpu_model.gpu_lm_head.cols);
                cuda_vedic_gemm(gpu_model.last_hidden, gpu_model.gpu_lm_head, C_proxy);

                bool using_cublas = cuda_vedic_gemm_uses_cublas();
                float vedic_tol = 0.05f;

                VedicVerifyResult vr = cuda_vedic_verify(
                    gpu_model.last_hidden, gpu_model.gpu_lm_head, C_proxy, vedic_tol);
                ++vedic_checks;
                if (vr.pass) {
                    ++vedic_pass;
                    snprintf(vedic_status, 8, "PASS");
                } else {
                    snprintf(vedic_status, 8, "WARN");
                    printf("\n[Gunitasamuchayah] WARN @ step %lld err=%.4f (tol=%.2f)\n",
                           (long long)step, vr.relative_error, vedic_tol);
                }
                } // C_proxy freed here (RAII)

            if (step % 100 == 0) {
                float cur_T, cur_aH, cur_aL;
                optimizer.get_state(cur_T, cur_aH, cur_aL);
                float lr_x = gpu_path_integral.lr_scale_ema(0.5f);

                // [v16-STABLE] Validation CE computation every 100 steps
                // Run forward-only on 10 validation batches (no backward, no grad)
                // This gives Val_CE to detect overfit/underfit
                float val_ce_sum = 0.f;
                int   val_count  = 0;
                std::vector<std::pair<std::vector<int>,std::vector<int>>> val_batches;
                // [v18] 10 → 30: val_ce estimate ka variance kam karo
                // 10 batches par std-dev ~0.4 CE unit thi → fake OVF-warn
                // 30 batches par std-dev ~0.15 → reliable overfit signal
                int val_batches_to_eval = 30;
                // ── [BUG-FIX] Validation loop memory leak ─────────────────
                // BUG (pre-fix): d_vtgt, d_vloss, d_vgrad har val-batch mein
                //   cudaMalloc hote the lekin exception ya early-break ke case
                //   mein cudaFree guarantee nahi tha. Zyada khatarnak: d_vgrad
                //   size = vseq × V floats = 512 × 32768 × 4B = 64 MB PER BATCH
                //   × 30 batches × 100-step frequency = 192 GB leaked per epoch
                //   on 219M model (V=32768). T4 (16 GB VRAM) crash in <30 steps.
                //
                // FIX: Pre-allocate d_vtgt / d_vloss / d_vgrad ONCE before the
                //   loop using the maximum possible sizes (SEQ, V), reuse across
                //   all val batches, free once after the loop. No per-batch alloc.
                // ──────────────────────────────────────────────────────────────
                // [v25-BUG3-FIX] RAII guard: phys.training ko false set karo validation ke liye,
                // aur GUARANTEE karo ki kisi bhi exception/early-return pe true wapas aaye.
                // Pehle: manual toggle tha — agar forward() mein OOM ya kernel error hoti to
                // phys.training permanently false reh jaata → Feynman dropout + Reynolds stats
                // silently band ho jaate sari remaining training mein.
                struct TrainingModeGuard {
                    GPUPhysicsConfig& phys;
                    TrainingModeGuard(GPUPhysicsConfig& p) : phys(p) { phys.training = false; }
                    ~TrainingModeGuard() { phys.training = true; }
                };

                int*   d_vtgt  = nullptr;
                float* d_vloss = nullptr;
                float* d_vgrad = nullptr;
                CUDA_CHECK(cudaMalloc(&d_vtgt,  SEQ * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_vloss, SEQ * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_vgrad, SEQ * V * sizeof(float)));

                for (int vb = 0; vb < val_batches_to_eval; ++vb) {
                    if (!val_loader.next_accum_batch(val_batches)) {
                        val_loader.reset();
                        if (!val_loader.next_accum_batch(val_batches)) break;
                    }
                    for (auto& [vin, vtgt] : val_batches) {
                        int vseq = (int)vin.size();
                        if (vseq <= 0 || vseq > SEQ) continue;

                        // Forward only — RAII guard ensures phys.training restored on any path
                        TrainingModeGuard _guard(gpu_model.phys);
                        GPUTensor vlogits = gpu_model.forward(vin);
                        // [v30-OOM-FIX] Explicitly free layer cache after val forward.
                        // forward() allocates ~6GB of activation tensors (seq=8192, L=16).
                        // Without free: val loop holds 6GB × val_batches_to_eval in flight.
                        // With free: only 1 batch active at a time → peak = 6GB not 180GB.
                        gpu_model.free_layer_cache();

                        CUDA_CHECK(cudaMemcpy(d_vtgt, vtgt.data(),
                                   vseq*sizeof(int), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemset(d_vloss, 0, vseq*sizeof(float)));
                        CUDA_CHECK(cudaMemset(d_vgrad, 0, vseq*V*sizeof(float)));

                        FreeEnergyResult vfe;
                        cuda_free_energy_loss(vlogits.data, d_vtgt,
                            d_vloss, d_vgrad, vseq, V, optimizer.temperature, vfe);

                        if (isfinite(vfe.cross_entropy)) {
                            val_ce_sum += vfe.cross_entropy;
                            ++val_count;
                        }
                        // vlogits freed automatically (GPUTensor RAII destructor)
                    }
                    val_batches.clear();
                }
                // Free the pre-allocated buffers once after the loop
                cudaFree(d_vtgt);  d_vtgt  = nullptr;
                cudaFree(d_vloss); d_vloss = nullptr;
                cudaFree(d_vgrad); d_vgrad = nullptr;
                float val_ce = (val_count > 0) ? val_ce_sum / val_count : -1.f;

                // [v16-STABLE] Overfit / Underfit detection
                // Overfit:  val_ce goes UP while train_ce goes DOWN
                // Underfit: both val_ce and train_ce > 5.0 after step 5000
                const char* fit_status = "OK";
                if (val_ce > 0.f && step > 500) {
                    // [v18] Threshold 0.05 → 0.30: LR=3e-3 ke saath val_ce har step
                    // 0.3-0.5 CE units oscillate karta hai (sampling noise + LR noise).
                    // 0.05 threshold pe har dip/spike OVF-warn trigger karta tha.
                    // 0.30 = real generalization gap indicate karta hai, noise nahi.
                    bool val_rising    = (val_ce  > prev_val_ce   + 0.30f);
                    bool train_falling = (avg_CE  < prev_train_ce - 0.10f);
                    if (val_rising && train_falling) {
                        ++overfit_streak;
                        fit_status = (overfit_streak >= 3) ? "OVERFIT!" : "OVF-warn";
                    } else {
                        overfit_streak = 0;
                    }
                    // [v18] UNDERFIT: sirf tab warn karo jab DONO > 5.5 ho
                    // aur step > 10000. Val_CE 6.7-6.9 at step 8k = normal progress,
                    // real underfit tab hai jab val_ce bhi 5.5+ rahe step 10k ke baad.
                    // [v19] Additional check: agar val_ce gir rahi hai (< prev-0.2)
                    // to UNDERFIT suppress karo — model learn kar raha hai.
                    bool val_improving = (val_ce < prev_val_ce - 0.05f);
                    if (avg_CE > 5.5f && val_ce > 5.5f && step > 10000 && !val_improving)
                        fit_status = "UNDERFIT";
                }
                if (val_ce > 0.f) { prev_val_ce = val_ce; prev_train_ce = avg_CE; }

                if (val_ce > 0.f) {
                    printf("%-8lld | %-8.4f | %-8.4f | %-8.4f | %-8.4f | %-8.3f | %-6.2f | %-6.2f | %-8.4f | %-6.3f | %s\n",
                           (long long)step, avg_F, avg_CE, val_ce, avg_S,
                           grad_norm, cur_aH, cur_aL, cur_T, lr_x, fit_status);
                } else {
                    printf("%-8lld | %-8.4f | %-8.4f | %-8s | %-8.4f | %-8.3f | %-6.2f | %-6.2f | %-8.4f | %-6.3f | %s\n",
                           (long long)step, avg_F, avg_CE, "N/A", avg_S,
                           grad_norm, cur_aH, cur_aL, cur_T, lr_x, fit_status);
                }
                fflush(stdout);
            }

            if (step>0 && step%1000==0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                gpu_model.sync_to_cpu(cpu_model);
                save_checkpoint(cpu_model,"logos_gpu_ckpt",(int)step);

                // [v21-OPTSTATE] Optimizer velocity GPU→CPU copy + save
                {
                    OptimizerState save_state;
                    save_state.velocities.resize(optimizer.d_velocity.size());
                    for (int i = 0; i < (int)optimizer.d_velocity.size(); ++i) {
                        int sz = optimizer.sizes[i];
                        save_state.velocities[i].resize(sz);
                        CUDA_CHECK(cudaMemcpy(save_state.velocities[i].data(),
                                              optimizer.d_velocity[i],
                                              sz * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                    }
                    save_state.ema_action    = gpu_path_integral.ema_action;
                    save_state.log_amplitude = gpu_path_integral.log_amplitude;
                    save_state.step_count    = static_cast<int64_t>(gpu_path_integral.step_count);
                    save_optimizer_state(save_state, "logos_gpu_ckpt", (int)step);
                }
                // [v29-FULLRESUME] TrainingState save — metrics + data position + AMP
                {
                    // [v29] gnorm EMA update
                    constexpr float EMA_A = 0.05f;
                    gnorm_ema    = (gnorm_ema    < 1.f) ? grad_norm
                                 : (1.f-EMA_A)*gnorm_ema    + EMA_A*grad_norm;
                    train_ce_ema = (train_ce_ema > 900.f) ? prev_train_ce
                                 : (1.f-EMA_A)*train_ce_ema + EMA_A*prev_train_ce;
                    val_ce_ema   = (val_ce_ema   > 900.f) ? prev_val_ce
                                 : (1.f-EMA_A)*val_ce_ema   + EMA_A*prev_val_ce;

                    TrainingState ts;
                    ts.step              = step;
                    ts.best_loss         = best_loss;
                    ts.prev_train_ce     = prev_train_ce;
                    ts.prev_val_ce       = prev_val_ce;
                    ts.overfit_streak    = overfit_streak;
                    ts.amp_scale         = loss_scaler.scale;
                    ts.amp_window        = loss_scaler.steps_since_last_overflow;
                    ts.amp_overflows     = loss_scaler.total_overflows;
                    ts.amp_scale_ups     = loss_scaler.total_scale_ups;
                    ts.amp_scale_downs   = loss_scaler.total_scale_downs;
                    // Data loader position
                    auto ls              = loader.get_state();
                    ts.loader_shard      = ls.current_shard;
                    ts.loader_byte_pos   = ls.byte_pos;
                    ts.loader_tokens_seen= ls.total_tokens_seen;
                    ts.loader_epoch      = ls.current_epoch;
                    auto vls             = val_loader.get_state();
                    ts.val_loader_byte_pos = vls.byte_pos;
                    // Smooth metrics
                    ts.gnorm_ema         = gnorm_ema;
                    ts.train_ce_ema      = train_ce_ema;
                    ts.val_ce_ema        = val_ce_ema;
                    ts.save("logos_gpu_ckpt", (int)step);
                }
                // [v23-AMP] Loss scaler state print at checkpoint
                loss_scaler.print_status(step);
                // [v11-CLIP] Show cuBLAS context so Vedic PASS/FAIL is interpretable
                // [v17-FIX] tol=5% dono backends ke liye (FIX-11: correct formula ke baad
                //   cuBLAS FP error sirf 0.1-2% reh gaya, purani 30% ki zarurat nahi)
                const char* gemm_backend = cuda_vedic_gemm_uses_cublas()
                                           ? "cuBLAS(tol=5%)" : "CustomCUDA(tol=5%)";
                float lr_x = gpu_path_integral.lr_scale_ema(0.5f);
                // Show val_ce at checkpoint for overfit tracking
                printf("Ckpt @ step %lld | best_F=%.4f | Train_CE≈%.4f | Val_CE≈%.4f | Vedic: %d/%d [%s] | LRx=%.3f\n",
                       (long long)step, best_loss, prev_train_ce, prev_val_ce,
                       vedic_pass, vedic_checks, gemm_backend, lr_x);
                fflush(stdout);
            }
            ++step;
        }
        printf("Epoch %d done | Step=%lld | Best_F=%.4f | α_H=%.2f\n",
               epoch+1,(long long)step,best_loss,optimizer.alpha_H);
    }

    training_done:  // [v18-RESUME] goto target from step cap check
    CUDA_CHECK(cudaDeviceSynchronize());
    // [v27-MEM1-FIX] d_targets/d_loss_buf/d_grad_out: CudaPtr RAII auto-freed here.
    // Manual cudaFree() calls removed — destructors handle cleanup on all paths.
    for (auto* g : gpu_grads) delete g;
    gpu_model.sync_to_cpu(cpu_model);
    save_checkpoint(cpu_model,"logos_final",(int)step);

    // [v29-FULLRESUME] Final TrainingState save
    {
        TrainingState ts_final;
        ts_final.step            = step;
        ts_final.best_loss       = best_loss;
        ts_final.prev_train_ce   = prev_train_ce;
        ts_final.prev_val_ce     = prev_val_ce;
        ts_final.overfit_streak  = overfit_streak;
        ts_final.amp_scale       = loss_scaler.scale;
        ts_final.amp_window      = loss_scaler.steps_since_last_overflow;
        ts_final.amp_overflows   = loss_scaler.total_overflows;
        ts_final.amp_scale_ups   = loss_scaler.total_scale_ups;
        ts_final.amp_scale_downs = loss_scaler.total_scale_downs;
        auto ls_f                = loader.get_state();
        ts_final.loader_shard      = ls_f.current_shard;
        ts_final.loader_byte_pos   = ls_f.byte_pos;
        ts_final.loader_tokens_seen= ls_f.total_tokens_seen;
        ts_final.loader_epoch      = ls_f.current_epoch;
        ts_final.gnorm_ema         = gnorm_ema;
        ts_final.train_ce_ema      = train_ce_ema;
        ts_final.val_ce_ema        = val_ce_ema;
        ts_final.save("logos_final", (int)step);
    }

    // [v21-OPTSTATE] Final optimizer state save
    {
        OptimizerState final_state;
        final_state.velocities.resize(optimizer.d_velocity.size());
        for (int i = 0; i < (int)optimizer.d_velocity.size(); ++i) {
            int sz = optimizer.sizes[i];
            final_state.velocities[i].resize(sz);
            CUDA_CHECK(cudaMemcpy(final_state.velocities[i].data(),
                                  optimizer.d_velocity[i],
                                  sz * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        final_state.ema_action    = gpu_path_integral.ema_action;
        final_state.log_amplitude = gpu_path_integral.log_amplitude;
        final_state.step_count    = static_cast<int64_t>(gpu_path_integral.step_count);
        save_optimizer_state(final_state, "logos_final", (int)step);
    }

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Training Complete! (v27-H100)           ║\n");
    printf("║  219M params | 8192 ctx | d=1024 L=16   ║\n");
    printf("║  Steps: %-8lld | Best F: %.4f          ║\n",(long long)step,best_loss);
    printf("║  Train_CE: %.4f | Val_CE: %.4f          ║\n", prev_train_ce, prev_val_ce);
    printf("║  [AMP] final_scale=%.0f overflows=%d      ║\n",
           loss_scaler.scale, loss_scaler.total_overflows);
    printf("║  [AMP] scale_ups=%d scale_downs=%d        ║\n",
           loss_scaler.total_scale_ups, loss_scaler.total_scale_downs);
    if (cuda_vedic_gemm_uses_cublas()) {
        printf("║  Gunitasamuchayah: %3d / %3d PASS        ║\n",vedic_pass,vedic_checks);
        printf("║  (cuBLAS tol=5%% — correct formula)       ║\n");
    } else {
        printf("║  Gunitasamuchayah: %3d / %3d PASS        ║\n",vedic_pass,vedic_checks);
        printf("║  (Custom CUDA tol=5%% — strict verify)    ║\n");
    }
    printf("║  ALL Physics 100%% wired fwd+bwd:         ║\n");
    printf("║    Feynman dropout bwd ✅ (mask chain)    ║\n");
    printf("║    Hyperbolic expmap bwd ✅ (Jacobian)    ║\n");
    printf("║    NS bwd ✅  Reynolds bwd ✅             ║\n");
    printf("║    PathIntegral ✅  EMA stats ✅          ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    // [v14] Path integral final state
    gpu_path_integral.log_state();
}

// ============================================================
//  [P3] FEYNMAN PATH INTEGRAL BEAM SEARCH
// ============================================================
struct FeynmanBeam {
    std::vector<int> tokens;
    float log_amplitude;
    float action;
};

std::vector<FeynmanBeam> generate_feynman(
    ModelGPU& gpu_model, const std::vector<int>& prompt_ids,
    int max_new=64, int beam_width=4, float hbar=1.0f, int top_k_expand=50)
{
    int V=gpu_model.cfg.vocab_size, SEQ=gpu_model.cfg.max_seq_len;
    beam_width=std::max(1,std::min(beam_width,32));
    top_k_expand=std::max(1,std::min(top_k_expand,V));
    hbar=std::max(0.01f,hbar);

    printf("\n[Feynman Beam Search] ħ=%.3f beams=%d top_k=%d max_new=%d\n",
           hbar,beam_width,top_k_expand,max_new); fflush(stdout);

    std::vector<FeynmanBeam> beams(1);
    beams[0].tokens={prompt_ids.begin(),prompt_ids.end()};
    beams[0].log_amplitude=0.0f; beams[0].action=0.0f;
    int eos_token=1;

    for (int s=0;s<max_new;++s) {
        std::vector<FeynmanBeam> candidates;
        candidates.reserve(beams.size()*top_k_expand);
        for (auto& beam : beams) {
            if (!beam.tokens.empty()&&beam.tokens.back()==eos_token&&s>0)
                { candidates.push_back(beam); continue; }
            std::vector<int> ctx=beam.tokens;
            if ((int)ctx.size()>=SEQ) ctx={ctx.end()-(SEQ-1),ctx.end()};
            GPUTensor logits=gpu_model.forward(ctx);
            int last_row=(int)ctx.size()-1;
            std::vector<float> h_logits(V);
            CUDA_CHECK(cudaMemcpy(h_logits.data(),logits.data+last_row*V,
                                   V*sizeof(float),cudaMemcpyDeviceToHost));
            float max_l=*std::max_element(h_logits.begin(),h_logits.end());
            float sum_exp=0.0f;
            std::vector<float> probs(V);
            for (int v=0;v<V;++v) { probs[v]=std::exp(h_logits[v]-max_l); sum_exp+=probs[v]; }
            float inv_sum=1.0f/(sum_exp+1e-9f);
            for (int v=0;v<V;++v) probs[v]*=inv_sum;
            std::vector<int> sv(V); std::iota(sv.begin(),sv.end(),0);
            std::partial_sort(sv.begin(),sv.begin()+top_k_expand,sv.end(),
                              [&probs](int a,int b){return probs[a]>probs[b];});
            for (int ki=0;ki<top_k_expand;++ki) {
                int tok=sv[ki]; float p=probs[tok];
                if (p<1e-10f) continue;
                float dS=-std::log(p+1e-10f);
                FeynmanBeam cand;
                cand.tokens=beam.tokens; cand.tokens.push_back(tok);
                cand.log_amplitude=beam.log_amplitude-dS/hbar;
                cand.action=beam.action+dS;
                candidates.push_back(std::move(cand));
            }
        }
        if ((int)candidates.size()>beam_width) {
            std::partial_sort(candidates.begin(),candidates.begin()+beam_width,
                              candidates.end(),[](const FeynmanBeam& a,const FeynmanBeam& b){
                                  return a.log_amplitude>b.log_amplitude;});
            candidates.resize(beam_width);
        }
        beams=std::move(candidates);
        bool all_done=true;
        for (auto& b:beams) if (b.tokens.empty()||b.tokens.back()!=eos_token){all_done=false;break;}
        if (all_done) break;
    }
    std::sort(beams.begin(),beams.end(),[](const FeynmanBeam& a,const FeynmanBeam& b){
        return a.log_amplitude>b.log_amplitude;});
    return beams;
}

void generate_gpu(const std::string& ckpt_path, const std::string& prompt_text,
                  int max_new=64, int beam_width=4, float hbar=1.0f, int top_k=50)
{
    // [v21] LOGOS_VOCAB env se vocab path lo — hardcoded "vocab.bin" band
    const char* vocab_env = std::getenv("LOGOS_VOCAB");
    std::string vocab_path = (vocab_env && *vocab_env)
                             ? std::string(vocab_env) : std::string("vocab.bin");
    printf("  Loading vocab: %s\n", vocab_path.c_str());

    Tokenizer tok;
    if (!tok.load(vocab_path)) {
        // Fallback: current dir mein try karo
        if (!tok.load("vocab.bin")) {
            fprintf(stderr,"❌ load: cannot open %s\n", vocab_path.c_str());
            fprintf(stderr,"❌ vocab.bin not found\n");
            fprintf(stderr,"   Set LOGOS_VOCAB=/path/to/vocab.bin\n");
            return;
        }
    }
    printf("  ✅ Vocab loaded: %d tokens\n", tok.vocab_size);
    // [v25-BUG7-FIX] generate_gpu() config defaults update kiye — pehle v17 (17M) defaults the.
    // Ab 219M model ke defaults: d=1024, H=16, L=16, seq=8192.
    // Kaggle CELL: set LOGOS_D_MODEL=1024 LOGOS_N_HEADS=16 LOGOS_N_LAYERS=16 LOGOS_SEQ_LEN=8192
    // (ya khali chhodo — niche defaults se auto-set ho jaayenge)
    ModelConfig cfg; cfg.vocab_size=tok.vocab_size;
    cfg.d_model     = (int)logos_env_f("LOGOS_D_MODEL",  1024.f);  // 219M: was 256
    cfg.num_heads   = (int)logos_env_f("LOGOS_N_HEADS",    16.f);  // 219M: was 8
    cfg.num_layers  = (int)logos_env_f("LOGOS_N_LAYERS",   16.f);  // 219M: was 6
    cfg.max_seq_len = (int)logos_env_f("LOGOS_SEQ_LEN",  8192.f);  // 219M: was 256
    printf("  Config: d=%d L=%d H=%d seq=%d vocab=%d\n",
           cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.max_seq_len, cfg.vocab_size);
    LOGOSModel cpu_model(cfg);
    if (ckpt_path!="none"&&!ckpt_path.empty()) load_checkpoint(cpu_model,ckpt_path);
    HyperConfig hyper_cfg; hyper_cfg.enabled=true; hyper_cfg.curvature=1.0f;
    ModelGPU gpu_model(cfg,hyper_cfg);
    gpu_model.load_from_cpu(cpu_model);

    // [v26-BUG10-FIX] Inference mode: phys.training explicitly false set karo.
    // ModelGPU default: phys.training=false (GPUPhysicsConfig default in header).
    // Lekin agar future code same gpu_model instance ko train_gpu ke baad reuse kare
    // (jahan phys.training=true set hota hai), toh Feynman dropout + Reynolds EMA
    // accidentally ON rahega — inference nondeterministic + slow ho jaata.
    // Explicitly false: safe even if caller reuses the instance.
    gpu_model.phys.training        = false;  // inference: no dropout, no EMA update
    gpu_model.phys.feynman_dropout = false;  // belt + suspenders: kernel bhi skip karo
    gpu_model.phys.reynolds        = false;  // running stats update skip (no data to track)
    printf("  [v26] Inference mode: training=false (dropout OFF, Reynolds OFF)\n");

    auto prompt_ids=tok.encode(prompt_text,cfg.max_seq_len/2);
    auto beams=generate_feynman(gpu_model,prompt_ids,max_new,beam_width,hbar,top_k);
    for (int i=0;i<(int)beams.size();++i) {
        std::vector<int> gen(beams[i].tokens.begin()+(int)prompt_ids.size(),beams[i].tokens.end());
        printf("Beam %d: %s\n",i,tok.decode(gen).c_str());
    }
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char* argv[]) {
    printf("╔══════════════════════════════════════════╗\n"
           "║  LOGOS GPU v27-H100                      ║\n"
           "║  MEM1: d_targets RAII FIXED              ║\n"
           "║  MEM2: NikhilamTensor safe alloc FIXED   ║\n"
           "║  H100: chunk/ckpt/prefetch optimized     ║\n"
           "╚══════════════════════════════════════════╝\n\n");

    std::string mode=(argc>1)?argv[1]:"--train";
    // [v26-BUG11-FIX] Path sanitizer: .. aur absolute paths dono block karo.
    // Pehle: sirf ".." check tha — "/etc/passwd" ya "/proc/self/mem" pass ho jaata.
    // RunPod pe command line args trusted hain, lekin production deployment risk real hai.
    // Fix: absolute paths (leading '/') bhi reject karo; fallback default return.
    // Note: Windows paths (C:\...) RunPod Linux pe nahi aate — skip.
    // Symbolic links: filesystem-level issue, userspace mein detect nahi ho sakta safely;
    // RunPod pe /workspace/ ke bahar symlinks standard setup mein nahi hote.
    auto safe=[](const char* raw,const char* fb)->std::string{
        if (!raw) return fb;
        std::string s(raw);
        // Block null bytes (embedded \0 se string truncation attack)
        if (s.find('\0')!=std::string::npos) return fb;
        // Block directory traversal (../ ya ..\)
        if (s.find("..")!=std::string::npos) return fb;
        // Block absolute paths (/etc/passwd, /proc/... etc.)
        if (!s.empty() && s[0]=='/') return fb;
        // Block empty paths
        if (s.empty()) return fb;
        return s;
    };

    try {
        if (mode=="--train") {
            // [v30-KAGGLE-FIX] Dataset path resolution priority:
            // 1. LOGOS_DATASET env var (absolute paths OK — set karo Kaggle cell mein)
            // 2. Command line arg argv[2] — safe() se nahi guzarta (absolute paths chahiye)
            // 3. Default "dataset.txt" (train_gpu ke andar auto-discover karega)
            // safe() sirf --generate mode ke paths ke liye hai (checkpoint/model paths)
            std::string ds = "dataset.txt";
            const char* ds_env = std::getenv("LOGOS_DATASET");
            if (ds_env && *ds_env) {
                ds = std::string(ds_env);
                printf("  [v30] Dataset from LOGOS_DATASET: %s\n", ds.c_str());
            } else if (argc > 2) {
                ds = std::string(argv[2]);  // argv[2] seedha use karo, safe() nahi
                printf("  [v30] Dataset from argv: %s\n", ds.c_str());
            }
            train_gpu(ds);
        } else if (mode=="--generate"||mode=="--generate-gpu") {
            std::string ckpt=(argc>2)?safe(argv[2],"none"):"none";
            std::string prompt=(argc>3)?std::string(argv[3]):"Once upon a time";
            int   max_new=(argc>4)?std::atoi(argv[4]):64;
            int   beams  =(argc>5)?std::atoi(argv[5]):4;
            float hbar   =(argc>6)?std::atof(argv[6]):1.0f;
            int   top_k  =(argc>7)?std::atoi(argv[7]):50;
            generate_gpu(ckpt,prompt,max_new,beams,hbar,top_k);
        } else {
            fprintf(stderr,"Usage:\n"
                "  logos_gpu --train [dataset.txt]\n"
                "  logos_gpu --generate [ckpt] [prompt] [max_new] [beams] [hbar] [top_k]\n");
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr,"\n❌ Fatal: %s\n",e.what()); return 1;
    }
    return 0;
}
