
#include "VedicGEMM.cuh"
#include "ModelGPU.cuh"
#include "MixedPrecision.cuh"
#include "../include/Tokenizer.hpp"
#include "../include/StreamingDataLoader.hpp"
#include "../include/Checkpoint.hpp"
#include "../include/PhysicsOpt.hpp"    
#include "../include/TrainingState.hpp" 
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
#include <thread>       // [v38-PARALLEL] std::thread
#include <atomic>       // [v38-PARALLEL] atomic flags
#include <mutex>        // [v38-PARALLEL] grad reduce lock




extern __global__ void gpu_transpose_kernel(const float*, float*, int, int);
extern __global__ void grad_scale_kernel(float*, float, int);
extern __global__ void vedic_gemm_kernel(const float*, const float*, float*, int, int, int);
extern __global__ void boltzmann_softmax_kernel(const float*, float*, int, int, float);
extern __global__ void causal_mask_kernel(float*, int);
extern __global__ void shm_hybrid_kernel(float*, float*, const float*, float, float, float, float, float, float, unsigned int, int);
extern __global__ void reynolds_bwd_kernel(const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, int, int, float);
extern __global__ void ns_diffuse_bwd_kernel(const float*, float*, int, int, float);
extern __global__ void ns_advect_bwd_kernel(const float*, float*, int, int, float);
extern __global__ void expmap0_bwd_kernel(const float*, const float*, float*, int, int, float);


static constexpr float GPU_T_MIN_FLOOR = 1e-3f;


static float logos_env_f(const char* name, float def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    float v = std::strtof(s, &end);
    return (end != s && std::isfinite(v)) ? v : def;
}


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


class GPUSHMOpt {
public:
    float   lr;
    float   friction;
    float   mom_decay;
    float   temperature, T_start, T_end;
    float   alpha_H_start, alpha_H_end;
    int64_t total_steps;
    int64_t step = 0;

    float   noise_gain = 0.05f;
    int     warmup_steps = 100;  // [v40-BUG4] warmup field add kiya

    std::vector<float*> d_velocity;
    std::vector<int>    sizes;

    float alpha_H = 0.3f;
    float alpha_L = 0.7f;

    GPUSHMOpt(float lr_        = 2e-4f,
              float friction_  = 0.8f,
              float mom_decay_ = 0.7f,
              float T_s        = 0.05f,
              float T_e        = 1e-3f,   
              float aH_start   = 0.3f,
              float aH_end     = 0.9f,
              int64_t steps    = 500000)
        : lr(lr_), friction(friction_), mom_decay(mom_decay_),
          temperature(T_s), T_start(T_s),
          
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

    
    void anneal() {
        float r = total_steps > 0
            ? std::min(1.0f, (float)step / (float)total_steps) : 1.0f;
        float c = 0.5f * (1.0f + cosf(3.14159265f * r));

  
        temperature = fmaxf(GPU_T_MIN_FLOOR, T_end + (T_start - T_end) * c);

        alpha_H = alpha_H_start + (alpha_H_end - alpha_H_start) * (1.0f - c);
        alpha_L = 1.0f - alpha_H;
    }

    void update(std::vector<GPUTensor*>& params,
                std::vector<GPUTensor*>& grads,
                float scale = 1.0f)
    {
        anneal();

        // [v40-BUG4] warmup scaling — pehle warmup_steps tak LR linearly badhta hai
        float warmup_scale = (warmup_steps > 0 && step < (int64_t)warmup_steps)
            ? (float)(step + 1) / (float)warmup_steps
            : 1.0f;
        float lr_scaled   = lr * scale * warmup_scale;
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


__global__ void vec_mul_kernel(float* dst, const float* src, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) dst[idx]*=src[idx];
}


__global__ void bias_grad_kernel(const float* __restrict__ dY, float* __restrict__ db,
                                 int seq, int cols)
{
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=cols) return;
    float acc=0.0f;
    for (int s=0;s<seq;++s) acc+=dY[s*cols+j];
    atomicAdd(&db[j],acc);
}


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


[[maybe_unused]] static float grad_group_sumsq(const std::vector<GPUTensor*>& grads, int b, int e)
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


// [v40-BUG1-FIX] Per-head backward buffers — ek baar allocate, har call reuse karo
struct HeadBwdBuffers {
    GPUTensor d_head_out;      // (seq, DH)
    GPUTensor d_attn_probs_buf;// (seq, seq) — recomputed attn probs
    GPUTensor dV_s;            // (seq, DH)
    GPUTensor dV;              // (seq, DH)
    GPUTensor d_attn_probs;    // (seq, seq)
    GPUTensor d_scores;        // (seq, seq)
    GPUTensor dQ_adv;          // (seq, DH)
    GPUTensor dQ;              // (seq, DH)
    GPUTensor dK;              // (seq, DH)
    GPUTensor K_T;             // (DH, seq)

    bool allocated = false;
    int  alloc_seq = 0, alloc_DH = 0;

    void ensure(int seq_, int DH_) {
        if (allocated && alloc_seq == seq_ && alloc_DH == DH_) return;
        // Free purane buffers agar size change hua
        if (allocated) {
            cudaFree(d_head_out.data);      cudaFree(d_attn_probs_buf.data);
            cudaFree(dV_s.data);            cudaFree(dV.data);
            cudaFree(d_attn_probs.data);    cudaFree(d_scores.data);
            cudaFree(dQ_adv.data);          cudaFree(dQ.data);
            cudaFree(dK.data);              cudaFree(K_T.data);
        }
        d_head_out       = gpu_alloc(seq_, DH_);
        d_attn_probs_buf = gpu_alloc(seq_, seq_);
        dV_s             = gpu_alloc(seq_, DH_);
        dV               = gpu_alloc(seq_, DH_);
        d_attn_probs     = gpu_alloc(seq_, seq_);
        d_scores         = gpu_alloc(seq_, seq_);
        dQ_adv           = gpu_alloc(seq_, DH_);
        dQ               = gpu_alloc(seq_, DH_);
        dK               = gpu_alloc(seq_, DH_);
        K_T              = gpu_alloc(DH_, seq_);
        allocated = true; alloc_seq = seq_; alloc_DH = DH_;
    }

    ~HeadBwdBuffers() {
        if (!allocated) return;
        cudaFree(d_head_out.data);      cudaFree(d_attn_probs_buf.data);
        cudaFree(dV_s.data);            cudaFree(dV.data);
        cudaFree(d_attn_probs.data);    cudaFree(d_scores.data);
        cudaFree(dQ_adv.data);          cudaFree(dQ.data);
        cudaFree(dK.data);              cudaFree(K_T.data);
    }
};

// [v40-BUG1] Per-layer d_concat aur d_dX_ln1 bhi pre-alloc
struct LayerBwdBuffers {
    GPUTensor d_concat;    // (seq, D)
    GPUTensor d_dX_ln1;   // (seq, D)
    bool allocated = false;
    int  alloc_seq = 0, alloc_D = 0;

    void ensure(int seq_, int D_) {
        if (allocated && alloc_seq == seq_ && alloc_D == D_) return;
        if (allocated) { cudaFree(d_concat.data); cudaFree(d_dX_ln1.data); }
        d_concat  = gpu_alloc(seq_, D_);
        d_dX_ln1  = gpu_alloc(seq_, D_);
        allocated = true; alloc_seq = seq_; alloc_D = D_;
    }

    ~LayerBwdBuffers() {
        if (!allocated) return;
        cudaFree(d_concat.data); cudaFree(d_dX_ln1.data);
    }
};

// [v40-BUG1] Static buffers — GPU0 main thread ke liye only
static HeadBwdBuffers s_head_bwd;
static LayerBwdBuffers s_layer_bwd;

// [v41-MULTIGPU-FIX] GPU1 thread ke liye alag thread-local buffers
static HeadBwdBuffers s_head_bwd_gpu1;
static LayerBwdBuffers s_layer_bwd_gpu1;

static void run_backward(
    ModelGPU& gpu_model, ModelConfig& cfg,
    const std::vector<GPUTensor*>& gpu_grads,
    GPUTensor& d_logits_grad, GPUTensor& d_dX_out,
    GPUTensor& d_d_ffn_out, GPUTensor& d_d_ffn_A,
    GPUTensor& d_d_ffn_H,   GPUTensor& d_d_normed2,
    GPUTensor& d_dX_ln,     GPUTensor& d_dX_attn_in,
    int seq,
    HeadBwdBuffers* head_bwd_override = nullptr,
    LayerBwdBuffers* layer_bwd_override = nullptr)
{
    int D=cfg.d_model, V=cfg.vocab_size, D4=4*D;
    int H=cfg.num_heads, DH=D/H;
    const int PPL=H*4+9;
    const int wproj_off=H*4, w1_off=H*4+1, w2_off=H*4+3;
    const int ln1g_off=H*4+5, ln1b_off=H*4+6;
    const int ln2g_off=H*4+7, ln2b_off=H*4+8;

    // [v41-MULTIGPU-FIX] GPU1 thread ke liye override buffers use karo, warna static GPU0 buffers
    HeadBwdBuffers&  head_bwd  = head_bwd_override  ? *head_bwd_override  : s_head_bwd;
    LayerBwdBuffers& layer_bwd = layer_bwd_override ? *layer_bwd_override : s_layer_bwd;

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
        { bias_grad_kernel<<<(D+127)/128,128>>>(d_d_ffn_out.data,
                                                gpu_grads[base+w2_off+1]->data,seq,D); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D4+31)/32),b(32);
          ffn_da_kernel<<<g,b>>>(d_d_ffn_out.data,blk.W2.data,
                                 d_d_ffn_A.data,seq,D4,D); CUDA_KERNEL_CHECK(); }
        { int tot=seq*D4;
          gelu_bwd_kernel<<<(tot+255)/256,256>>>(cache.ffn_H.data,d_d_ffn_A.data,
                                                  d_d_ffn_H.data,tot); CUDA_KERNEL_CHECK(); }

       
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
        
        { bias_grad_kernel<<<(D4+127)/128,128>>>(d_d_ffn_H.data,
                                                 gpu_grads[base+w1_off+1]->data,seq,D4); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D+31)/32),b(32);
          ffn_dx_kernel<<<g,b>>>(d_d_ffn_H.data,blk.W1.data,
                                 d_d_normed2.data,seq,D,D4); CUDA_KERNEL_CHECK(); }

        
        if (gpu_model.phys.reynolds && cache.ln2_w.valid()) {
            reynolds_bwd_kernel<<<seq,256>>>(
                cache.post_attn.data, blk.ln2_gamma.data,
                d_d_normed2.data, cache.ln2_w.data,
                gpu_model.run_mean[2*l+1].data, gpu_model.run_var[2*l+1].data,
                d_dX_ln.data,
                gpu_grads[base+ln2g_off]->data, gpu_grads[base+ln2b_off]->data,
                seq, D, 1e-5f);
        } else {
            
            layernorm_bwd_kernel<<<seq,256>>>(
                cache.post_attn.data, blk.ln2_gamma.data,
                d_d_normed2.data, d_dX_ln.data,
                gpu_grads[base+ln2g_off]->data, gpu_grads[base+ln2b_off]->data,
                seq, D, 1e-5f);
        }
        CUDA_KERNEL_CHECK();
        { int sz=seq*D;
          vec_add_kernel<<<(sz+255)/256,256>>>(d_dX_out.data,d_dX_ln.data,sz); CUDA_KERNEL_CHECK(); }

        // [v40-BUG1-FIX] d_concat — pre-alloc reuse (layer_bwd.d_concat)
        GPUTensor& d_concat = layer_bwd.d_concat;
        { dim3 g(D,(D+31)/32),b(32);
          attn_dWproj_kernel<<<g,b>>>(cache.concat.data,d_dX_out.data,
                                      gpu_grads[base+wproj_off]->data,seq,D); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D+31)/32),b(32);
          attn_dConcat_kernel<<<g,b>>>(d_dX_out.data,blk.W_proj.data,
                                       d_concat.data,seq,D); CUDA_KERNEL_CHECK(); }

        // [v40-BUG1-FIX] Pre-allocated buffers use karo — har head loop mein alloc nahi
        head_bwd.ensure(seq, DH);
        layer_bwd.ensure(seq, D);

        CUDA_CHECK(cudaMemset(d_dX_attn_in.data,0,seq*D*sizeof(float)));
        for (int h=0;h<H;++h) {
            auto& hc=cache.heads[h];
            // d_head_out_h — reuse head_bwd.d_head_out
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dHeadOut_kernel<<<g,b>>>(d_concat.data,blk.W_O[h].data,
                                            head_bwd.d_head_out.data,seq,DH,D); CUDA_KERNEL_CHECK(); }
            { dim3 g(DH,(D+31)/32),b(32);
              attn_dWO_kernel<<<g,b>>>(hc.head_out.data,d_concat.data,
                                       gpu_grads[base+h*4+3]->data,seq,DH,D); CUDA_KERNEL_CHECK(); }

            // [v40-BUG1-FIX] recomputed_attn_probs — pre-alloc buf reuse
            {
                float scale_attn = 1.0f / sqrtf((float)DH);
                // K_T — pre-alloc reuse
                gpu_transpose_kernel<<<dim3((seq+15)/16,(DH+15)/16),dim3(16,16)>>>(
                    hc.K.data, head_bwd.K_T.data, seq, DH);
                CUDA_KERNEL_CHECK();
                vedic_gemm_kernel<<<dim3((seq+15)/16,(seq+15)/16),dim3(16,16)>>>(
                    hc.Q_adv.data, head_bwd.K_T.data,
                    head_bwd.d_attn_probs_buf.data, seq, DH, seq);
                CUDA_KERNEL_CHECK();
                int sz_sc = seq * seq;
                grad_scale_kernel<<<(sz_sc+255)/256, 256>>>(
                    head_bwd.d_attn_probs_buf.data, scale_attn, sz_sc);
                CUDA_KERNEL_CHECK();
                causal_mask_kernel<<<dim3(seq,(seq+255)/256),256>>>(
                    head_bwd.d_attn_probs_buf.data, seq);
                CUDA_KERNEL_CHECK();
                boltzmann_softmax_kernel<<<seq, 256>>>(
                    head_bwd.d_attn_probs_buf.data,
                    head_bwd.d_attn_probs_buf.data,
                    seq, seq, 1.0f);
                CUDA_KERNEL_CHECK();
            }

            // dV_s — pre-alloc reuse
            CUDA_CHECK(cudaMemset(head_bwd.dV_s.data,0,seq*DH*sizeof(float)));
            { dim3 g(DH,(seq+31)/32),b(32);
              attn_dV_kernel<<<g,b>>>(head_bwd.d_attn_probs_buf.data,
                                      head_bwd.d_head_out.data,
                                      head_bwd.dV_s.data,seq,DH); CUDA_KERNEL_CHECK(); }

            // dV — pre-alloc reuse
            { int qb=(seq*DH+255)/256;
              ns_diffuse_bwd_kernel<<<qb,256>>>(head_bwd.dV_s.data,
                                                head_bwd.dV.data,seq,DH,
                  gpu_model.phys.ns_nu); CUDA_KERNEL_CHECK(); }

            // d_attn_probs — pre-alloc reuse
            { dim3 g(seq,(seq+31)/32),b(32);
              attn_dAttnProbs_kernel<<<g,b>>>(head_bwd.d_head_out.data,hc.V_s.data,
                                              head_bwd.d_attn_probs.data,seq,DH); CUDA_KERNEL_CHECK(); }

            // d_scores — pre-alloc reuse
            float inv_sqrt_DH=1.0f/sqrtf((float)DH);
            softmax_bwd_kernel<<<seq,256>>>(head_bwd.d_attn_probs_buf.data,
                                            head_bwd.d_attn_probs.data,
                                            head_bwd.d_scores.data,
                                            seq,inv_sqrt_DH); CUDA_KERNEL_CHECK();

            // dQ_adv — pre-alloc reuse
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dQ_kernel<<<g,b>>>(head_bwd.d_scores.data,hc.K.data,
                                      head_bwd.dQ_adv.data,seq,DH); CUDA_KERNEL_CHECK(); }

            // dQ — pre-alloc reuse
            { int qb=(seq*DH+255)/256;
              ns_advect_bwd_kernel<<<qb,256>>>(head_bwd.dQ_adv.data,
                                               head_bwd.dQ.data,seq,DH,
                  gpu_model.phys.ns_eta); CUDA_KERNEL_CHECK(); }

            // dK — pre-alloc reuse
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dK_kernel<<<g,b>>>(head_bwd.d_scores.data,hc.Q_adv.data,
                                      head_bwd.dK.data,seq,DH); CUDA_KERNEL_CHECK(); }

            { dim3 g(D,(DH+31)/32),b(32);
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,head_bwd.dQ.data,
                                         gpu_grads[base+h*4+0]->data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,head_bwd.dK.data,
                                         gpu_grads[base+h*4+1]->data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,head_bwd.dV.data,
                                         gpu_grads[base+h*4+2]->data,seq,D,DH); CUDA_KERNEL_CHECK(); }
            { dim3 g(seq,(D+31)/32),b(32);
              attn_dX_from_QKV_kernel<<<g,b>>>(head_bwd.dQ.data,blk.W_Q[h].data,
                                               d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dX_from_QKV_kernel<<<g,b>>>(head_bwd.dK.data,blk.W_K[h].data,
                                               d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dX_from_QKV_kernel<<<g,b>>>(head_bwd.dV.data,blk.W_V[h].data,
                                               d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK(); }
        }

        // [v40-BUG1-FIX] d_dX_ln1 — pre-alloc reuse
        GPUTensor& d_dX_ln1 = layer_bwd.d_dX_ln1;
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

    
    const bool hyper_chain = gpu_model.hyper_cfg.enabled && gpu_model.X_euclidean.valid();
    if (!hyper_chain) {
      dim3 blk(32),grd(seq,(D+31)/32);
      embedding_bwd_kernel<<<grd,blk>>>(
          gpu_model.d_token_ids,d_dX_out.data,
          gpu_grads[0]->data,gpu_grads[1]->data,
          seq,D,V); CUDA_KERNEL_CHECK(); }

    
    if (hyper_chain) {
        
        GPUTensor d_dX_euc = gpu_alloc(seq, D);
        expmap0_bwd_kernel<<<seq, 256>>>(
            gpu_model.X_euclidean.data,   
            d_dX_out.data,                
            d_dX_euc.data,                
            seq, D, gpu_model.hyper_cfg.curvature);
        CUDA_KERNEL_CHECK();
        
        { dim3 blk2(32), grd2(seq,(D+31)/32);
          embedding_bwd_kernel<<<grd2,blk2>>>(
              gpu_model.d_token_ids, d_dX_euc.data,
              gpu_grads[0]->data, gpu_grads[1]->data,
              seq, D, V); CUDA_KERNEL_CHECK(); }
        
    }
}

void train_gpu(const std::string& dataset_path) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  LOGOS GPU Training v30                  ║\n");
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
    printf("║  ✦ [v29] Full resume: DataState+Stats  ║\n");
    printf("║  ✦ [v30] Dataset auto-resolve + vocab  ║\n");
    printf("║  ✦ [v33] vocab.bin skip-rebuild ~920s  ║\n");
    printf("║  ✦ H100 SXM Tensor Core GEMM (2x spd)  ║\n");
    printf("║  ✦ LOGOS_CHUNK_MB / LOGOS_CKPT_FREQ    ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s | VRAM: %zu MB | SMs: %d | CC: %d.%d\n",
           prop.name, prop.totalGlobalMem/1024/1024,
           prop.multiProcessorCount, prop.major, prop.minor);

    // [v37-MULTIGPU] Kitne GPU available hain check karo
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    bool use_multi_gpu = (num_gpus >= 2);
    if (use_multi_gpu) {
        // GPU 1 bhi same family hai check karo
        cudaDeviceProp prop1;
        cudaGetDeviceProperties(&prop1, 1);
        // CC match hona chahiye warna disable karo
        if (prop1.major != prop.major || prop1.minor != prop.minor) {
            printf("  [v37-MULTIGPU] GPU0 CC=%d.%d != GPU1 CC=%d.%d — single GPU mode\n",
                   prop.major, prop.minor, prop1.major, prop1.minor);
            use_multi_gpu = false;
        }
    }
    if (use_multi_gpu) {
        cudaDeviceProp prop1;
        cudaGetDeviceProperties(&prop1, 1);
        printf("  [v37-MULTIGPU] %d GPUs detected!\n", num_gpus);
        printf("    GPU0: %s | %.1f GB\n", prop.name,  (float)prop.totalGlobalMem/1024/1024/1024);
        printf("    GPU1: %s | %.1f GB\n", prop1.name, (float)prop1.totalGlobalMem/1024/1024/1024);
        printf("  [v37-MULTIGPU] Data Parallel ON — micro-batches split across GPUs\n");
    } else {
        printf("  [v37-MULTIGPU] Single GPU mode (%d GPU detected)\n", num_gpus);
    }

    bool is_h100 = (prop.major == 9 && prop.minor == 0);   
    bool is_a100 = (prop.major == 8 && prop.minor == 0);   
    bool is_t4   = (prop.major == 7 && prop.minor == 5
                    && prop.totalGlobalMem < 17ULL*1024*1024*1024); 
    bool amp_supported = (prop.major >= 7);                  
    printf("  GPU detected: H100=%s | A100=%s | T4=%s | CC=%d.%d | VRAM=%.1fGB\n",
           is_h100?"✅":"—", is_a100?"✅":"—", is_t4?"✅":"—",
           prop.major, prop.minor,
           (float)prop.totalGlobalMem/1024/1024/1024);
    printf("  AMP: FP16 Tensor Cores %s\n", amp_supported ? "✅" : "❌");
    if (!amp_supported) {
        printf("  ⚠️  GPU CC < 7.0 — FP16 Tensor Cores not available\n");
        printf("      AMP disabled, falling back to FP32\n");
    }
    printf("\n");
    fflush(stdout);

    printf("[1/5] Dataset scan...\n"); fflush(stdout);

    
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
        
        bool skip_rebuild = false;
        const char* rebuild_env = std::getenv("LOGOS_REBUILD_VOCAB");
        bool force_rebuild = (rebuild_env && rebuild_env[0]=='1');

        if (!force_rebuild && std::ifstream("vocab.bin").good()) {
            printf("  [v33-FAST] vocab.bin found — loading (skip rebuild, ~920s saved!)\n");
            printf("             Force rebuild: set LOGOS_REBUILD_VOCAB=1\n");
            if (tok.load("vocab.bin")) {
                printf("  ✅ Vocab loaded: %d tokens\n\n", tok.vocab_size);
                skip_rebuild = true;
            } else {
                printf("  ⚠️  vocab.bin load failed — rebuilding\n");
            }
        }

        if (!skip_rebuild) {
            
            int64_t tok_mb = (int64_t)logos_env_f("LOGOS_TOKENIZER_MB", 4.f);
            tok_mb = std::max((int64_t)1, std::min((int64_t)32, tok_mb));
            int64_t TOKENIZER_SAMPLE = tok_mb * 1024LL * 1024LL;
            printf("  [v33-FAST] BPE sample: %lld MB (override: LOGOS_TOKENIZER_MB)\n",
                   (long long)tok_mb);
            int64_t sample_size=std::min(actual_size, TOKENIZER_SAMPLE);
            std::ifstream f(dataset_path_resolved,std::ios::binary);
            if (!f) { fprintf(stderr,"❌ Cannot open: %s\n",dataset_path_resolved.c_str()); return; }
            std::string sample; sample.resize((size_t)sample_size);
            f.read(sample.data(),sample_size);
            sample.resize((size_t)f.gcount());
            tok.build(sample,vocab_target);
            tok.save("vocab.bin");
            printf("Vocab: %d tokens\n\n",tok.vocab_size);
        }
    }

    printf("[2/5] Model config...\n"); fflush(stdout);
    auto scaled=decide_model_config(actual_size);
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size;
    cfg.d_model=scaled.d_model;
    cfg.num_heads=scaled.num_heads;
    cfg.num_layers=scaled.num_layers;
    cfg.max_seq_len=scaled.max_seq_len;

    {
        int d_override = (int)logos_env_f("LOGOS_D_MODEL", 0.f);
        int h_override = (int)logos_env_f("LOGOS_N_HEADS", 0.f);
        int l_override = (int)logos_env_f("LOGOS_N_LAYERS", 0.f);
        if (d_override > 0) cfg.d_model    = d_override;
        if (h_override > 0) cfg.num_heads  = h_override;
        if (l_override > 0) cfg.num_layers = l_override;
        
        if (is_t4 && d_override == 0) {
            cfg.d_model   = 256;
            cfg.num_heads = 8;
            cfg.num_layers = 6;
            printf("  [v34-T4] T4 detected → 17M config (d=256 L=6 H=8)\n");
            printf("           219M ke liye LOGOS_D_MODEL=1024 LOGOS_N_HEADS=16 LOGOS_N_LAYERS=16 set karo\n");
            printf("           (CELL 1C mein ye env vars already set hain)\n");
        }
    }
    std::string cfg_err = validate_model_config(
        cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.vocab_size, cfg.max_seq_len);
    if (!cfg_err.empty()) { fprintf(stderr,"❌ Config: %s\n",cfg_err.c_str()); return; }
    {
        int seq_override = (int)logos_env_f("LOGOS_SEQ_LEN", 0.f);
        if (seq_override > 0) {
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
        
        if (cfg.max_seq_len <= 4096) {
            printf("  [v32-SEQ] Shunyam params adjusted for seq=%d: window=32 stride=8\n",
                   cfg.max_seq_len);
        }
    }

    
    int grad_accum_default = 4;  
    if (is_t4)   grad_accum_default = 4;   
    if (is_a100)  grad_accum_default = 16;  
    if (is_h100)  grad_accum_default = 32;  
    int grad_accum = (int)logos_env_f("LOGOS_GRAD_ACCUM", (float)grad_accum_default);
    grad_accum = std::max(1, std::min(32, grad_accum));  // clamp [1, 32]
    printf("  [v32-BATCH] grad_accum=%d → %d tokens/step (T4=4, A100=16, H100=32 | override: LOGOS_GRAD_ACCUM)\n",
           grad_accum, grad_accum * cfg.max_seq_len);

    
    if (cfg.vocab_size != (int)decide_vocab_size(actual_size)) {
        printf("  ℹ️  vocab target=%d → actual tok.vocab_size=%d (BPE coverage)\n",
               decide_vocab_size(actual_size), cfg.vocab_size);
    }
    printf("d=%d L=%d H=%d DH=%d seq=%d vocab=%d grad_accum=%d\n",
           cfg.d_model,cfg.num_layers,cfg.num_heads,cfg.d_model/cfg.num_heads,
           cfg.max_seq_len,cfg.vocab_size,grad_accum);

    
    printf("\n[v17 SHM Optimizer] hyper-parameters neeche [4/5] me print honge\n\n");

   
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

    
    OptimizerState opt_state_loaded;
    bool opt_state_found = false;

    
    TrainingState ts_loaded;
    bool ts_found = false;

    
    if (resumed) {
        const char* ckpt_env_ts = std::getenv("LOGOS_CKPT");
        const int   step_n_ts   = (int)logos_env_f("LOGOS_START_STEP", 0.f);
        if (ckpt_env_ts && *ckpt_env_ts && step_n_ts > 0) {
            
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

    // [v37-MULTIGPU] GPU1 pe replica banao
    ModelGPU* gpu_model1 = nullptr;
    if (use_multi_gpu) {
        cudaSetDevice(1);
        gpu_model1 = new ModelGPU(cfg);
        gpu_model1->load_from_cpu(cpu_model);
        cudaSetDevice(0);  // wapas GPU0 pe
        printf("  [v37-MULTIGPU] GPU1 model replica ready\n");
    }

    
    gpu_model.phys.training        = true;
    gpu_model.phys.nikhilam_kv     = true;
    gpu_model.phys.shunyam         = true;
    gpu_model.phys.window          = (cfg.max_seq_len >= 8192) ? 64 : 32;
    gpu_model.phys.stride          = (cfg.max_seq_len >= 8192) ? 16 : 8;
    gpu_model.phys.navier_stokes   = true;
    gpu_model.phys.ns_eta          = 0.1f;
    gpu_model.phys.ns_nu           = 0.05f;
    gpu_model.phys.feynman_dropout = true;
    gpu_model.phys.drop_p          = 0.1f;
    gpu_model.phys.drop_hbar       = 1.0f;
    gpu_model.phys.reynolds        = true;
    gpu_model.phys.re_crit         = 2.0f;
    gpu_model.phys.re_k            = 5.0f;

    // [v37-MULTIGPU] GPU1 pe same physics config
    if (use_multi_gpu && gpu_model1) {
        gpu_model1->phys = gpu_model.phys;
        gpu_model1->phys.drop_p        = 0.1f;
        gpu_model1->phys.drop_hbar     = 1.0f;
    }
    printf("  ✦ phys.training=true  → Feynman dropout ACTIVE\n");
    printf("  ✦ Nikhilam KV INT8    → 4x KV compression\n");
    printf("  ✦ Shunyam sparse attn → window=%d stride=%d (8192 ctx)\n", gpu_model.phys.window, gpu_model.phys.stride);
    printf("  ✦ Navier-Stokes       → eta=%.2f nu=%.2f\n", gpu_model.phys.ns_eta, gpu_model.phys.ns_nu);
    printf("  ✦ Reynolds norm       → re_crit=%.1f k=%.1f\n", gpu_model.phys.re_crit, gpu_model.phys.re_k);
    printf("  ✦ Feynman dropout     → p=%.2f hbar=%.1f\n", gpu_model.phys.drop_p, gpu_model.phys.drop_hbar);
    printf("  ✦ Model scale         → d=1024 L=16 H=16 seq=8192 (~219M params)\n\n");

    printf("[4/5] StreamingDataLoader + Optimizer...\n"); fflush(stdout);
    int SEQ=cfg.max_seq_len;
    
    int64_t chunk_mb_default = 4LL;
    if (is_h100) chunk_mb_default = 64LL;
    else if (is_a100) chunk_mb_default = 32LL;
    int64_t chunk_mb_env = (int64_t)logos_env_f("LOGOS_CHUNK_MB", (float)chunk_mb_default);
    chunk_mb_env = std::max((int64_t)1, std::min((int64_t)256, chunk_mb_env));
    const int64_t CHUNK_BYTES = chunk_mb_env * 1024LL * 1024LL;
    printf("  Chunk size: %lld MB (auto: H100=64 A100=32 T4=4 | override: LOGOS_CHUNK_MB)\n",
           (long long)chunk_mb_env);

    
    int64_t val_start_byte = (int64_t)(actual_size * 0.95);
    int64_t val_bytes      = actual_size - val_start_byte;
    printf("  Train: %.1f MB | Val: %.1f MB (5%% held out)\n",
           (float)val_start_byte/1024/1024, (float)val_bytes/1024/1024);

    StreamingDataLoader loader(dataset_path_resolved, tok, SEQ, grad_accum, CHUNK_BYTES,
                               /*start_byte=*/0, /*end_byte=*/val_start_byte);
    
    StreamingDataLoader val_loader(dataset_path_resolved, tok, SEQ, 1, 1LL*1024*1024,
                                   /*start_byte=*/val_start_byte, /*end_byte=*/actual_size);
    std::vector<std::pair<std::vector<int>,std::vector<int>>> fixed_val;
    {
        std::vector<std::pair<std::vector<int>,std::vector<int>>> tmp;
        for (int i=0; i<512 && (int)fixed_val.size()<32; ++i) {
            if (!val_loader.next_accum_batch(tmp) || tmp.empty()) break;
            if (i%16==0) fixed_val.push_back(std::move(tmp[0]));
        }
        val_loader.reset();  
    }
    printf("  Val set: %zu fixed windows x %d tokens\n", fixed_val.size(), SEQ);
    if (fixed_val.empty()) printf("  ⚠️  Val set empty — Val_CE N/A rahega\n");

    
    int EPOCHS=2;
    int64_t batches_per_epoch=loader.total_batches_per_epoch();
    const int64_t steps_cap=(int64_t)logos_env_f("LOGOS_STEPS", 500000.f);  // production default; testing ke liye set karo LOGOS_STEPS=20
    
    const int64_t start_step = resumed
        ? (int64_t)logos_env_f("LOGOS_START_STEP", 0.f)
        : 0LL;
    int64_t total_steps=std::min(steps_cap, start_step + EPOCHS*(batches_per_epoch/grad_accum));
    if (total_steps < 1) total_steps = 1;

    
    float lr_default = (grad_accum >= 4) ? 2e-3f : 1e-3f;
    float   lr_init      = logos_env_f_clamped("LOGOS_LR",         lr_default, 1e-7f, 1.0f);
    const float clip_norm = logos_env_f_clamped("LOGOS_CLIP",      0.3f,   0.01f, 10.0f);
    const float t_start   = logos_env_f_clamped("LOGOS_T_START",   0.05f,  1e-4f, 1.0f);
    
    const int   warmup_steps = (int)logos_env_f_clamped("LOGOS_WARMUP", 100.f, 0.f, 10000.f);
    const float noise_gain   = logos_env_f_clamped("LOGOS_NOISE_GAIN", 0.02f, 0.0f, 1.0f);

    
    GPUSHMOpt optimizer(lr_init,
                        /*friction=*/0.4f,
                        /*mom_decay=*/0.85f,
                        /*T_start=*/t_start,
                        /*T_end=*/1e-3f,
                        /*aH_start=*/0.70f,
                        /*aH_end=*/0.95f,
                        total_steps - start_step);
    optimizer.noise_gain  = noise_gain;   // [v40-BUG4] dead code fix — ab use hota hai
    optimizer.warmup_steps = warmup_steps; // [v40-BUG4] warmup bhi wired
    auto gpu_params=gpu_model.all_parameters();
    optimizer.init(gpu_params);
    auto gpu_grads=gpu_model.alloc_grad_buffers();
    struct GradGuard {
        std::vector<GPUTensor*>& grads;
        GradGuard(std::vector<GPUTensor*>& g) : grads(g) {}
        ~GradGuard() {
            for (auto* g : grads) {
                if (g->data) cudaFree(g->data);  // [v40-BUG5] cudaFree missing tha — leak fix
                delete g;
            }
        }
    } grad_guard(gpu_grads);

    // [v37-MULTIGPU] GPU1 grad buffers
    std::vector<GPUTensor*> gpu_grads1;
    struct GradGuard1 {
        std::vector<GPUTensor*>& grads;
        bool active;
        GradGuard1(std::vector<GPUTensor*>& g, bool a) : grads(g), active(a) {}
        ~GradGuard1() { if (active) for (auto* g : grads) { cudaFree(g->data); delete g; } }
    };
    if (use_multi_gpu && gpu_model1) {
        cudaSetDevice(1);
        gpu_grads1 = gpu_model1->alloc_grad_buffers();
        cudaSetDevice(0);
    }
    GradGuard1 grad_guard1(gpu_grads1, use_multi_gpu);

    // [v40-BUG3] grad reduce ke liye pre-allocated CPU buffer + GPU tmp
    // Pehle max param size dhundho
    int max_param_sz = 0;
    for (auto* p : gpu_params) max_param_sz = std::max(max_param_sz, p->size);
    std::vector<float> h_grad_reduce_buf(max_param_sz);  // ek baar alloc, reuse
    float* d_grad_reduce_tmp = nullptr;
    if (use_multi_gpu && max_param_sz > 0) {
        CUDA_CHECK(cudaMalloc(&d_grad_reduce_tmp, max_param_sz * sizeof(float)));
        printf("  [v40-BUG3] Grad reduce buffer pre-alloc: %.1f MB\n",
               (float)max_param_sz * sizeof(float) / 1024 / 1024);
    }
    struct GradReduceGuard {
        float* ptr;
        GradReduceGuard(float* p) : ptr(p) {}
        ~GradReduceGuard() { if (ptr) cudaFree(ptr); }
    } grad_reduce_guard(d_grad_reduce_tmp);

    
    WeightPathIntegral gpu_path_integral(/*hbar=*/1.0f, /*history=*/500);
    
    ManualLossScaler loss_scaler;
    {
        
        float init_scale = logos_env_f_clamped("LOGOS_AMP_SCALE", LOSS_SCALE_INIT, 1.0f, 131072.0f);
        
        int   init_window = (int)logos_env_f_clamped("LOGOS_AMP_WINDOW", 0.f, 0.f, (float)LOSS_SCALE_WINDOW);
        loss_scaler.scale = fmaxf(fminf(init_scale, LOSS_SCALE_MAX), LOSS_SCALE_MIN);
        loss_scaler.steps_since_last_overflow = init_window;
        printf("  [v23-AMP] ManualLossScaler: init_scale=%.0f window=%d (restored=%d)\n",
               loss_scaler.scale, LOSS_SCALE_WINDOW, init_window);
        const char* gpu_name = is_h100 ? "H100 SXM" : is_a100 ? "A100" : is_t4 ? "T4" : "GPU";
        printf("  [v23-AMP] %s: FP16 forward (2× TFLOPS) + FP32 master weights\n", gpu_name);
        printf("  [v23-AMP] Precision: forward=FP16 | grads=FP32 | optimizer=FP32\n");
        printf("  [v23-AMP] To resume scale: set LOGOS_AMP_SCALE=<prev_scale> LOGOS_AMP_WINDOW=<prev_window>\n");
    }

    
    std::vector<HalfTensor> fp16_weights;
    fp16_weights.reserve(gpu_params.size());
    for (auto* p : gpu_params) {
        fp16_weights.push_back(half_alloc(p->rows, p->cols));
    }
    printf("  [v23-AMP] FP16 shadow weights: %zu tensors allocated\n",
           fp16_weights.size());
    printf("  [v23-AMP] FP16 VRAM overhead: ~%.0f MB\n",
           (float)(219e6 * sizeof(__half)) / 1024 / 1024);

    
    std::vector<float*> grad_ptrs;
    std::vector<int>    grad_sizes;
    grad_ptrs.reserve(gpu_grads.size());
    grad_sizes.reserve(gpu_grads.size());
    for (auto* g : gpu_grads) {
        grad_ptrs.push_back(g->data);
        grad_sizes.push_back(g->size);
    }

    
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

    printf("%-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-6s | %-6s | %-8s | %-6s | %-8s\n",
           "Step","F(loss)","Train_CE","Val_CE","Entropy","GNorm","α_H","α_L","T","LRx","Status");
    printf("---------|---------|---------|---------|---------|---------|--------|--------|---------|--------|--------\n");
    fflush(stdout);

    
    float prev_val_ce   = 999.f;
    float prev_train_ce = 999.f;
    int   overfit_streak = 0;  

   
    int64_t step=start_step;
    
    float   best_loss = resumed
        ? logos_env_f("LOGOS_BEST_F", 999.f)
        : 999.f;

    
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
    // [v35] Step-0 initial checkpoint — session crash ke baad bhi resume ho sake
    if (!resumed && start_step == 0) {
        printf("  [v35] Step-0 initial checkpoint save kar raha hoon (crash recovery ke liye)...\n");
        CUDA_CHECK(cudaDeviceSynchronize());
        gpu_model.sync_to_cpu(cpu_model);
        save_checkpoint(cpu_model, "logos_gpu_ckpt", 0);
        printf("  ✅ logos_gpu_ckpt_step0.bin saved\n");
        fflush(stdout);
    }
    // [v35] Early-stop guard: agar step 0 pe hi total_steps==0 ho toh crash se bachao
    if (total_steps <= start_step) {
        printf("\n⚠️  [v35] total_steps=%lld <= start_step=%lld — kuch nahi karna!\n",
               (long long)total_steps, (long long)start_step);
        printf("     LOGOS_STEPS env var set karo: export LOGOS_STEPS=500000\n");
        printf("     Ya LOGOS_START_STEP check karo agar resume kar rahe ho.\n");
        return;
    }

    int     vedic_checks=0, vedic_pass=0;
    char    vedic_status[8]="N/A";

    // [v40-BUG7] loop ke bahar ek baar calculate karo — env read har step nahi
    int64_t ckpt_freq_default = is_h100 ? 5000LL : is_a100 ? 3000LL : 50LL;
    int64_t ckpt_freq = (int64_t)logos_env_f("LOGOS_CKPT_FREQ", (float)ckpt_freq_default);
    ckpt_freq = std::max((int64_t)1, std::min((int64_t)50000, ckpt_freq));

    int64_t log_freq_default = 1LL;
    int64_t log_freq = (int64_t)logos_env_f("LOGOS_LOG_FREQ", (float)log_freq_default);
    log_freq = std::max((int64_t)1, std::min((int64_t)10000, log_freq));

    int val_batches_to_eval = (int)logos_env_f("LOGOS_VAL_BATCHES", 10.f);
    val_batches_to_eval = std::max(1, std::min(100, val_batches_to_eval));

    printf("  [v40] ckpt_freq=%lld | log_freq=%lld | val_batches=%d\n",
           (long long)ckpt_freq, (long long)log_freq, val_batches_to_eval);

    for (int epoch=0;epoch<EPOCHS;++epoch) {
        printf("\n-- Epoch %d/%d --\n",epoch+1,EPOCHS); fflush(stdout);
        std::vector<std::pair<std::vector<int>,std::vector<int>>> micro_batches;

        while (loader.next_accum_batch(micro_batches)) {
            if (step >= total_steps) {
                printf("\n[v18-RESUME] Reached total_steps=%lld at step=%lld — stopping.\n",
                       (long long)total_steps, (long long)step);
                goto training_done;
            }
            for (auto* g : gpu_grads)
                CUDA_CHECK(cudaMemset(g->data,0,g->size*sizeof(float)));

            
            for (int pi = 0; pi < (int)gpu_params.size(); ++pi) {
                cuda_cast_fp32_to_fp16(
                    gpu_params[pi]->data,
                    fp16_weights[pi].data,
                    gpu_params[pi]->size);
            }
            
            if (amp_supported) {
                gpu_model.fp16_param_ptrs.resize(fp16_weights.size());
                for (int pi = 0; pi < (int)fp16_weights.size(); ++pi)
                    gpu_model.fp16_param_ptrs[pi] = fp16_weights[pi].data;
                gpu_model.amp_enabled = true;
            }

            float batch_F=0.f, batch_CE=0.f, batch_S=0.f;
            int   valid_mb=0;

            // [v38-PARALLEL] GPU0 aur GPU1 ek saath chalein — true parallel
            // micro_batches[0,2] → GPU0 thread | micro_batches[1,3] → GPU1 thread
            struct MBResult {
                float F=0.f, CE=0.f, S=0.f;
                int valid=0;
            };
            MBResult res0, res1;
            std::exception_ptr thread_ex; // [v43] GPU1 thread exception forward karne ke liye

            // GPU0 ka kaam — main thread karta hai
            auto run_on_gpu = [&](int gpu_id, ModelGPU& model,
                                  std::vector<GPUTensor*>& grads,
                                  MBResult& result)
            {
              try {
                cudaSetDevice(gpu_id);
                // grad zero
                for (auto* g : grads)
                    cudaMemset(g->data, 0, g->size*sizeof(float));

                for (int mbi = gpu_id; mbi < (int)micro_batches.size(); mbi += 2) {
                    auto& [input_ids, target_ids] = micro_batches[mbi];
                    int seq = (int)input_ids.size();

                    // GPU1 ke liye local buffers
                    CudaPtr<int>   loc_tgt(SEQ);
                    CudaPtr<float> loc_loss(SEQ);
                    CudaPtr<float> loc_grad((int64_t)SEQ * V);

                    GPUTensor logits = model.forward(input_ids);

                    cudaMemcpy(loc_tgt.get(), target_ids.data(),
                               seq*sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemset(loc_loss.get(), 0, seq*sizeof(float));
                    cudaMemset(loc_grad.get(), 0, seq*V*sizeof(float));

                    FreeEnergyResult fe;
                    cuda_free_energy_loss(logits.data, loc_tgt.get(),
                                          loc_loss.get(), loc_grad.get(),
                                          seq, V, optimizer.temperature, fe);

                    GPUTensor logits_grad = gpu_alloc(SEQ, V);
                    cudaMemcpy(logits_grad.data, loc_grad.get(),
                               seq*V*sizeof(float), cudaMemcpyDeviceToDevice);

                    if (isfinite(fe.free_energy) && fe.free_energy > 1e-8f) {
                        result.F  += fe.free_energy;
                        result.CE += fe.cross_entropy;
                        result.S  += fe.entropy;
                        ++result.valid;

                        GPUTensor t_dX     = gpu_alloc(SEQ, D);
                        GPUTensor t_ffnout = gpu_alloc(SEQ, D);
                        GPUTensor t_ffnA   = gpu_alloc(SEQ, D4);
                        GPUTensor t_ffnH   = gpu_alloc(SEQ, D4);
                        GPUTensor t_n2     = gpu_alloc(SEQ, D);
                        GPUTensor t_ln     = gpu_alloc(SEQ, D);
                        GPUTensor t_attn   = gpu_alloc(SEQ, D);

                        // [v41-MULTIGPU-FIX] GPU1 thread ke liye dedicated buffers pass karo
                        HeadBwdBuffers*  hbuf = (gpu_id == 1) ? &s_head_bwd_gpu1  : nullptr;
                        LayerBwdBuffers* lbuf = (gpu_id == 1) ? &s_layer_bwd_gpu1 : nullptr;

                        run_backward(model, cfg, grads,
                                     logits_grad, t_dX,
                                     t_ffnout, t_ffnA, t_ffnH,
                                     t_n2, t_ln, t_attn, seq, hbuf, lbuf);
                        model.update_norm_stats();
                        model.free_layer_cache();
                    }
                }
                cudaDeviceSynchronize();  // is GPU ka kaam khatam
              } catch (...) {
                // [v43] GPU1 thread exception — main thread pe forward karo
                if (gpu_id == 1) thread_ex = std::current_exception();
                else             throw;  // GPU0 = main thread, seedha rethrow
              }
            };

            if (use_multi_gpu && gpu_model1) {
                // [v38-PARALLEL] GPU1 alag thread mein — GPU0 ke saath ek saath!
                std::thread t1(run_on_gpu, 1, std::ref(*gpu_model1),
                               std::ref(gpu_grads1), std::ref(res1));

                // GPU0 main thread mein
                run_on_gpu(0, gpu_model, gpu_grads, res0);

                t1.join();  // dono khatam hone ka wait

                // [v44-FIX] GPU1 thread exception rethrow karo (terminate prevent)
                if (thread_ex) std::rethrow_exception(thread_ex);

                // results merge karo
                batch_F  = res0.F  + res1.F;
                batch_CE = res0.CE + res1.CE;
                batch_S  = res0.S  + res1.S;
                valid_mb = res0.valid + res1.valid;

                // [v40-BUG3] pre-allocated buffer use karo — har step cudaMalloc nahi
                cudaSetDevice(0);
                for (int gi = 0; gi < (int)gpu_grads.size() && gi < (int)gpu_grads1.size(); ++gi) {
                    int sz = gpu_grads[gi]->size;
                    // GPU1 → CPU (pre-alloc buf)
                    cudaSetDevice(1);
                    cudaMemcpy(h_grad_reduce_buf.data(), gpu_grads1[gi]->data,
                               sz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemset(gpu_grads1[gi]->data, 0, sz*sizeof(float));
                    // CPU → GPU0 tmp (pre-alloc)
                    cudaSetDevice(0);
                    cudaMemcpy(d_grad_reduce_tmp, h_grad_reduce_buf.data(),
                               sz*sizeof(float), cudaMemcpyHostToDevice);
                    vec_add_kernel<<<(sz+255)/256,256>>>(gpu_grads[gi]->data, d_grad_reduce_tmp, sz);
                }
                CUDA_KERNEL_CHECK();
            } else {
                // single GPU — purana loop
                for (auto& [input_ids, target_ids] : micro_batches) {
                    int seq = (int)input_ids.size();
                    GPUTensor logits = gpu_model.forward(input_ids);
                    CUDA_CHECK(cudaMemcpy(d_targets, target_ids.data(),
                               seq*sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemset(d_loss_buf, 0, seq*sizeof(float)));
                    CUDA_CHECK(cudaMemset(d_grad_out, 0, seq*V*sizeof(float)));
                    FreeEnergyResult fe_result;
                    cuda_free_energy_loss(logits.data, d_targets, d_loss_buf, d_grad_out,
                                          seq, V, optimizer.temperature, fe_result);
                    CUDA_CHECK(cudaMemcpy(d_logits_grad.data, d_grad_out,
                               seq*V*sizeof(float), cudaMemcpyDeviceToDevice));
                    if (isfinite(fe_result.free_energy) && fe_result.free_energy > 1e-8f) {
                        batch_F  += fe_result.free_energy;
                        batch_CE += fe_result.cross_entropy;
                        batch_S  += fe_result.entropy;
                        ++valid_mb;
                        run_backward(gpu_model, cfg, gpu_grads,
                                     d_logits_grad, d_dX_out,
                                     d_d_ffn_out, d_d_ffn_A, d_d_ffn_H,
                                     d_d_normed2, d_dX_ln, d_dX_attn_in, seq);
                        gpu_model.update_norm_stats();
                        gpu_model.free_layer_cache();
                    }
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

            
            {
                
                amp_scale_grads(grad_ptrs, grad_sizes, loss_scaler.scale);
                CUDA_KERNEL_CHECK();

                
                bool amp_ok = loss_scaler.update(grad_ptrs, grad_sizes);

                if (!amp_ok) {
                    
                    for (auto* g : gpu_grads)
                        CUDA_CHECK(cudaMemset(g->data, 0, g->size * sizeof(float)));
                    ++step;
                    
                    if (loss_scaler.total_overflows % 50 == 1) {
                        printf("  [AMP] step=%lld OVERFLOW skip #%d → new_scale=%.0f\n",
                               (long long)step, loss_scaler.total_overflows, loss_scaler.scale);
                        fflush(stdout);
                    }
                    continue;  
                }
                
                float inv_s = 1.0f / loss_scaler.scale;
                amp_scale_grads(grad_ptrs, grad_sizes, inv_s);
                CUDA_KERNEL_CHECK();
                
            }

            
            float grad_norm=cuda_clip_gradients(gpu_grads, clip_norm);

            
            float adapted_lr_scale = gpu_path_integral.lr_scale_ema(0.5f);
            optimizer.update(gpu_params, gpu_grads, adapted_lr_scale);

            // [v37-MULTIGPU] GPU0 updated weights → GPU1 sync (har step)
            if (use_multi_gpu && gpu_model1) {
                cudaSetDevice(0);
                auto params0 = gpu_model.all_parameters();
                cudaSetDevice(1);
                auto params1 = gpu_model1->all_parameters();
                for (int pi2 = 0; pi2 < (int)params0.size() && pi2 < (int)params1.size(); ++pi2) {
                    int sz = params0[pi2]->size;
                    std::vector<float> h_w(sz);
                    cudaSetDevice(0);
                    CUDA_CHECK(cudaMemcpy(h_w.data(), params0[pi2]->data,
                               sz*sizeof(float), cudaMemcpyDeviceToHost));
                    cudaSetDevice(1);
                    CUDA_CHECK(cudaMemcpy(params1[pi2]->data, h_w.data(),
                               sz*sizeof(float), cudaMemcpyHostToDevice));
                }
                cudaSetDevice(0);
            }

            
            gpu_path_integral.record_step_norm(avg_F, grad_norm * optimizer.lr);
            
            if (step > start_step && (step - start_step) % 5000 == 0) gpu_path_integral.reset_best();

            // [v40-BUG7] ckpt_freq aur log_freq ab loop ke bahar define hain

            if (step % ckpt_freq == 0 && step > 0) {
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
            }  // end: if (step % ckpt_freq == 0 && step > 0)

            // [v40-BUG7] log_freq aur val_batches ab loop ke bahar define hain
            if (step % log_freq == 0) {
                float cur_T, cur_aH, cur_aL;
                optimizer.get_state(cur_T, cur_aH, cur_aL);
                float lr_x = gpu_path_integral.lr_scale_ema(0.5f);

                
                float val_ce_sum = 0.f;
                int   val_count  = 0;
                std::vector<std::pair<std::vector<int>,std::vector<int>>> val_batches;
                // [v40-BUG7] val_batches_to_eval ab loop ke bahar define hai
                
                struct TrainingModeGuard {
                    GPUPhysicsConfig& phys;
                    TrainingModeGuard(GPUPhysicsConfig& p) : phys(p) { phys.training = false; }
                    ~TrainingModeGuard() { phys.training = true; }
                };

                int*   d_vtgt  = nullptr;
                float* d_vloss = nullptr;
                float* d_vgrad = nullptr;
                CudaPtr<int>   d_vtgt_raii(SEQ);
                CudaPtr<float> d_vloss_raii(SEQ);
                CudaPtr<float> d_vgrad_raii(SEQ * V);
                d_vtgt  = d_vtgt_raii.get();
                d_vloss = d_vloss_raii.get();
                d_vgrad = d_vgrad_raii.get();

                for (int vb = 0; vb < val_batches_to_eval; ++vb) {
                    if (!val_loader.next_accum_batch(val_batches)) {
                        val_loader.reset();
                        if (!val_loader.next_accum_batch(val_batches)) break;
                    }
                    for (auto& [vin, vtgt] : val_batches) {
                        int vseq = (int)vin.size();
                        if (vseq <= 0 || vseq > SEQ) continue;

                        TrainingModeGuard _guard(gpu_model.phys);
                        GPUTensor vlogits = gpu_model.forward(vin);
                        
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
                    }
                    val_batches.clear();
                }
                float val_ce = (val_count > 0) ? val_ce_sum / val_count : -1.f;

                
                const char* fit_status = "OK";
                if (val_ce > 0.f && step > 500) {
                    
                    bool val_rising    = (val_ce  > prev_val_ce   + 0.30f);
                    bool train_falling = (avg_CE  < prev_train_ce - 0.10f);
                    if (val_rising && train_falling) {
                        ++overfit_streak;
                        fit_status = (overfit_streak >= 3) ? "OVERFIT!" : "OVF-warn";
                    } else {
                        overfit_streak = 0;
                    }
                    
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

            if (step>0 && step%ckpt_freq==0) {
                
                CUDA_CHECK(cudaDeviceSynchronize());
                gpu_model.sync_to_cpu(cpu_model);
                save_checkpoint(cpu_model,"logos_gpu_ckpt",(int)step);

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
                
                {
                    
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
                    
                    auto ls              = loader.get_state();
                    ts.loader_shard      = ls.current_shard;
                    ts.loader_byte_pos   = ls.byte_pos;
                    ts.loader_tokens_seen= ls.total_tokens_seen;
                    ts.loader_epoch      = ls.current_epoch;
                    auto vls             = val_loader.get_state();
                    ts.val_loader_byte_pos = vls.byte_pos;
                    ts.gnorm_ema         = gnorm_ema;
                    ts.train_ce_ema      = train_ce_ema;
                    ts.val_ce_ema        = val_ce_ema;
                    ts.save("logos_gpu_ckpt", (int)step);
                }
                
                loss_scaler.print_status(step);
                
                const char* gemm_backend = cuda_vedic_gemm_uses_cublas()
                                           ? "cuBLAS(tol=5%)" : "CustomCUDA(tol=5%)";
                float lr_x = gpu_path_integral.lr_scale_ema(0.5f);
                
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

    training_done:
    CUDA_CHECK(cudaDeviceSynchronize());
    if (use_multi_gpu && gpu_model1) {
        cudaSetDevice(1);
        cudaDeviceSynchronize();
        delete gpu_model1;
        gpu_model1 = nullptr;
        cudaSetDevice(0);
        printf("  [v37-MULTIGPU] GPU1 cleanup done\n");
    }

    gpu_model.sync_to_cpu(cpu_model);
    save_checkpoint(cpu_model,"logos_final",(int)step);

    
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
    printf("║  Training Complete! (v30)                ║\n");
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
    gpu_path_integral.log_state();
}


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
    
    const char* vocab_env = std::getenv("LOGOS_VOCAB");
    std::string vocab_path = (vocab_env && *vocab_env)
                             ? std::string(vocab_env) : std::string("vocab.bin");
    printf("  Loading vocab: %s\n", vocab_path.c_str());

    Tokenizer tok;
    if (!tok.load(vocab_path)) {
        
        if (!tok.load("vocab.bin")) {
            fprintf(stderr,"❌ load: cannot open %s\n", vocab_path.c_str());
            fprintf(stderr,"❌ vocab.bin not found\n");
            fprintf(stderr,"   Set LOGOS_VOCAB=/path/to/vocab.bin\n");
            return;
        }
    }
    printf("  ✅ Vocab loaded: %d tokens\n", tok.vocab_size);
    
    ModelConfig cfg; cfg.vocab_size=tok.vocab_size;
    cfg.d_model     = (int)logos_env_f("LOGOS_D_MODEL",  1024.f);  
    cfg.num_heads   = (int)logos_env_f("LOGOS_N_HEADS",    16.f);  
    cfg.num_layers  = (int)logos_env_f("LOGOS_N_LAYERS",   16.f);  
    cfg.max_seq_len = (int)logos_env_f("LOGOS_SEQ_LEN",  8192.f);  
    printf("  Config: d=%d L=%d H=%d seq=%d vocab=%d\n",
           cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.max_seq_len, cfg.vocab_size);
    LOGOSModel cpu_model(cfg);
    if (ckpt_path!="none"&&!ckpt_path.empty()) load_checkpoint(cpu_model,ckpt_path);
    HyperConfig hyper_cfg; hyper_cfg.enabled=true; hyper_cfg.curvature=1.0f;
    ModelGPU gpu_model(cfg,hyper_cfg);
    gpu_model.load_from_cpu(cpu_model);

    
    gpu_model.phys.training        = false;  
    gpu_model.phys.feynman_dropout = false;  
    gpu_model.phys.reynolds        = false;  
    printf("  [v26] Inference mode: training=false (dropout OFF, Reynolds OFF)\n");

    auto prompt_ids=tok.encode(prompt_text,cfg.max_seq_len/2);
    auto beams=generate_feynman(gpu_model,prompt_ids,max_new,beam_width,hbar,top_k);
    for (int i=0;i<(int)beams.size();++i) {
        std::vector<int> gen(beams[i].tokens.begin()+(int)prompt_ids.size(),beams[i].tokens.end());
        printf("Beam %d: %s\n",i,tok.decode(gen).c_str());
    }
}


int main(int argc, char* argv[]) {
    printf("╔══════════════════════════════════════════╗\n"
           "║  LOGOS GPU v30                           ║\n"
           "║  MEM1: d_targets RAII FIXED              ║\n"
           "║  MEM2: NikhilamTensor safe alloc FIXED   ║\n"
           "║  H100: chunk/ckpt/prefetch optimized     ║\n"
           "║  v29: Full resume (DataState+TrainStats) ║\n"
           "║  v30: Dataset auto-resolve + vocab cache ║\n"
           "╚══════════════════════════════════════════╝\n\n");

    std::string mode=(argc>1)?argv[1]:"--train";
    
    auto safe=[](const char* raw,const char* fb)->std::string{
        if (!raw) return fb;
        std::string s(raw);
        if (s.find('\0')!=std::string::npos) return fb;
        if (s.find("..")!=std::string::npos) return fb;
        // Note: absolute paths (starting with '/') are allowed for checkpoint paths
        if (s.empty()) return fb;
        return s;
    };

    auto safe_prompt=[](const char* raw,const char* fb)->std::string{
        if (!raw) return fb;
        std::string s(raw);
        if (s.empty()) return fb;
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) {
            if (c >= 0x20 || c == '\t') out += (char)c;
        }
        return out.empty() ? fb : out;
    };

    try {
        if (mode=="--train") {
            
            std::string ds = "dataset.txt";
            const char* ds_env = std::getenv("LOGOS_DATASET");
            if (ds_env && *ds_env) {
                ds = std::string(ds_env);
                printf("  [v30] Dataset from LOGOS_DATASET: %s\n", ds.c_str());
            } else if (argc > 2) {
                ds = std::string(argv[2]);  
                printf("  [v30] Dataset from argv: %s\n", ds.c_str());
            }
            train_gpu(ds);
        } else if (mode=="--generate"||mode=="--generate-gpu") {
            std::string ckpt=(argc>2)?safe(argv[2],"none"):"none";
            std::string prompt=(argc>3)?safe_prompt(argv[3],"Once upon a time"):"Once upon a time";
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
