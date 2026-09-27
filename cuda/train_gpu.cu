//  Logging: added alpha_H, alpha_L columns to training output
//  All Phase 1+2+3 kernels: UNCHANGED
//
//  FIX-1 (v9→v9-fix): Temperature floor in GPUSHMOpt::anneal()
//    T_end was 1e-6 → T → 0.0000 at late training
//    Now: T_end enforced >= T_MIN_FLOOR = 1e-3 in constructor + anneal()
//    Reason: F = CE - T*S; T=0 kills entropy regularization → overconfident → loss spike
// ============================================================
#include "VedicGEMM.cuh"
#include "ModelGPU.cuh"
#include "../include/Tokenizer.hpp"
#include "../include/StreamingDataLoader.hpp"
#include "../include/Checkpoint.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdint>

// FIX-1: Minimum temperature — entropy regularization always active
static constexpr float GPU_T_MIN_FLOOR = 1e-3f;

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

    std::vector<float*> d_velocity;
    std::vector<int>    sizes;

    float alpha_H = 0.3f;
    float alpha_L = 0.7f;

    GPUSHMOpt(float lr_        = 2e-4f,
              float friction_  = 0.1f,
              float mom_decay_ = 0.9f,
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
        float noise_scale = sqrtf(friction * temperature * lr * alpha_L);
        float lr_scaled   = lr * scale;

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
//  run_backward() — v6 unchanged
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
        { dim3 g(seq,(D4+31)/32),b(32);
          ffn_da_kernel<<<g,b>>>(d_d_ffn_out.data,blk.W2.data,
                                 d_d_ffn_A.data,seq,D4,D); CUDA_KERNEL_CHECK(); }
        { int tot=seq*D4;
          gelu_bwd_kernel<<<(tot+255)/256,256>>>(cache.ffn_H.data,d_d_ffn_A.data,
                                                  d_d_ffn_H.data,tot); CUDA_KERNEL_CHECK(); }
        { dim3 g(D,(D4+31)/32),b(32);
          ffn_w1_grad_kernel<<<g,b>>>(cache.normed2.data,d_d_ffn_H.data,
                                      gpu_grads[base+w1_off]->data,seq,D,D4); CUDA_KERNEL_CHECK(); }
        { dim3 g(seq,(D+31)/32),b(32);
          ffn_dx_kernel<<<g,b>>>(d_d_ffn_H.data,blk.W1.data,
                                 d_d_normed2.data,seq,D,D4); CUDA_KERNEL_CHECK(); }

        layernorm_bwd_kernel<<<seq,256>>>(
            cache.block_input.data,blk.ln2_gamma.data,
            d_d_normed2.data,d_dX_ln.data,
            gpu_grads[base+ln2g_off]->data,gpu_grads[base+ln2b_off]->data,
            seq,D,1e-5f); CUDA_KERNEL_CHECK();
        { int sz=seq*D;
          vec_add_kernel<<<(sz+255)/256,256>>>(d_dX_out.data,d_dX_ln.data,sz); CUDA_KERNEL_CHECK(); }

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
            GPUTensor dV=gpu_alloc(seq,DH);
            CUDA_CHECK(cudaMemset(dV.data,0,seq*DH*sizeof(float)));
            { dim3 g(DH,(seq+31)/32),b(32);
              attn_dV_kernel<<<g,b>>>(hc.attn_probs.data,d_head_out_h.data,
                                      dV.data,seq,DH); CUDA_KERNEL_CHECK(); }
            GPUTensor d_attn_probs=gpu_alloc(seq,seq);
            { dim3 g(seq,(seq+31)/32),b(32);
              attn_dAttnProbs_kernel<<<g,b>>>(d_head_out_h.data,hc.V.data,
                                              d_attn_probs.data,seq,DH); CUDA_KERNEL_CHECK(); }
            GPUTensor d_scores=gpu_alloc(seq,seq);
            float inv_sqrt_DH=1.0f/sqrtf((float)DH);
            softmax_bwd_kernel<<<seq,256>>>(hc.attn_probs.data,d_attn_probs.data,
                                            d_scores.data,seq,inv_sqrt_DH); CUDA_KERNEL_CHECK();
            GPUTensor dQ=gpu_alloc(seq,DH);
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dQ_kernel<<<g,b>>>(d_scores.data,hc.K.data,dQ.data,seq,DH); CUDA_KERNEL_CHECK(); }
            GPUTensor dK=gpu_alloc(seq,DH);
            { dim3 g(seq,(DH+31)/32),b(32);
              attn_dK_kernel<<<g,b>>>(d_scores.data,hc.Q.data,dK.data,seq,DH); CUDA_KERNEL_CHECK(); }
            { dim3 g(D,(DH+31)/32),b(32);
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,dQ.data,gpu_grads[base+h*4+0]->data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,dK.data,gpu_grads[base+h*4+1]->data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dWQKV_kernel<<<g,b>>>(cache.normed1.data,dV.data,gpu_grads[base+h*4+2]->data,seq,D,DH); CUDA_KERNEL_CHECK(); }
            { dim3 g(seq,(D+31)/32),b(32);
              attn_dX_from_QKV_kernel<<<g,b>>>(dQ.data,blk.W_Q[h].data,d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dX_from_QKV_kernel<<<g,b>>>(dK.data,blk.W_K[h].data,d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK();
              attn_dX_from_QKV_kernel<<<g,b>>>(dV.data,blk.W_V[h].data,d_dX_attn_in.data,seq,D,DH); CUDA_KERNEL_CHECK(); }
        }

        GPUTensor d_dX_ln1=gpu_alloc(seq,D);
        layernorm_bwd_kernel<<<seq,256>>>(
            cache.block_input.data,blk.ln1_gamma.data,
            d_dX_attn_in.data,d_dX_ln1.data,
            gpu_grads[base+ln1g_off]->data,gpu_grads[base+ln1b_off]->data,
            seq,D,1e-5f); CUDA_KERNEL_CHECK();
        { int sz=seq*D;
          vec_add_kernel<<<(sz+255)/256,256>>>(d_dX_out.data,d_dX_ln1.data,sz); CUDA_KERNEL_CHECK(); }
    }

    { dim3 blk(32),grd(seq,(D+31)/32);
      embedding_bwd_kernel<<<grd,blk>>>(
          gpu_model.d_token_ids,d_dX_out.data,
          gpu_grads[0]->data,gpu_grads[1]->data,
          seq,D,V); CUDA_KERNEL_CHECK(); }
}

// ============================================================
//  MAIN TRAINING FUNCTION
// ============================================================
void train_gpu(const std::string& dataset_path) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  LOGOS GPU Training v10-HAM              ║\n");
    printf("║  FIX-4: Hamiltonian-dominant optimizer   ║\n");
    printf("║  FIX-5: T_start=0.5 (real annealing)    ║\n");
    printf("║  FIX-6: α_H: 0.7→0.99 (strong gradient) ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,device);
    printf("GPU: %s | VRAM: %zu MB | SMs: %d | CC: %d.%d\n\n",
           prop.name,prop.totalGlobalMem/1024/1024,
           prop.multiProcessorCount,prop.major,prop.minor);
    fflush(stdout);

    printf("[1/5] Dataset scan...\n"); fflush(stdout);
    int64_t actual_size=scan_dataset_size(dataset_path);
    if (actual_size==0) {
        fprintf(stderr,"❌ Dataset not found: %s\n",dataset_path.c_str()); return;
    }
    printf("Dataset: %lld MB\n\n",(long long)actual_size/1024/1024);

    int vocab_target=decide_vocab_size(actual_size);
    printf("[1/5] Tokenizer build (vocab=%d)...\n",vocab_target); fflush(stdout);
    Tokenizer tok;
    {
        static constexpr int64_t TOKENIZER_SAMPLE=32LL*1024*1024;
        int64_t sample_size=std::min(actual_size,TOKENIZER_SAMPLE);
        std::ifstream f(dataset_path,std::ios::binary);
        if (!f) { fprintf(stderr,"❌ Cannot open: %s\n",dataset_path.c_str()); return; }
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

    int grad_accum=1;
    if (actual_size>200LL*1024*1024) grad_accum=4;
    else if (actual_size>50LL*1024*1024) grad_accum=2;

    printf("d=%d L=%d H=%d DH=%d seq=%d vocab=%d grad_accum=%d\n",
           cfg.d_model,cfg.num_layers,cfg.num_heads,cfg.d_model/cfg.num_heads,
           cfg.max_seq_len,cfg.vocab_size,grad_accum);

    printf("\n[v10-HAM Hamiltonian-Dominant SHM Optimizer]\n");
    printf("  ✦ FIX-4: α_H: 0.7→0.99 (Hamiltonian dominant throughout)\n");
    printf("  ✦ FIX-5: T_start=0.5 → T_end=1e-3 (real cosine annealing)\n");
    printf("  ✦ FIX-6: friction=0.1 mom=0.95 (strong momentum build-up)\n");
    printf("  ✦ Loss: Free Energy F = CE - T·S (thermodynamic)\n\n");

    printf("[3/5] Init GPU model...\n"); fflush(stdout);
    LOGOSModel cpu_model(cfg);
    ModelGPU   gpu_model(cfg);
    gpu_model.load_from_cpu(cpu_model);

    printf("[4/5] StreamingDataLoader + Optimizer...\n"); fflush(stdout);
    int SEQ=cfg.max_seq_len;
    static constexpr int64_t CHUNK_BYTES=4LL*1024*1024;
    StreamingDataLoader loader(dataset_path,tok,SEQ,grad_accum,CHUNK_BYTES);

    int EPOCHS=3;
    int64_t batches_per_epoch=loader.total_batches_per_epoch();
    int64_t total_steps=EPOCHS*(batches_per_epoch/grad_accum);
    float   lr_init=(cfg.d_model>=256)?1e-4f:2e-4f;

    // FIX-4/5/6: Hamiltonian-dominant mode
    // T_start=0.5  → real annealing hoga (1e-5 se floor pe hi stuck tha)
    // T_end=1e-3   → entropy floor preserved (FIX-1 bhi intact)
    // aH_start=0.7 → Hamiltonian dominant from step 0 (gradient direction strong)
    // aH_end=0.99  → near-pure Hamiltonian at end (deterministic convergence)
    // friction=0.1 → kam friction = momentum build up kare
    // mom_decay=0.95 → stronger momentum retention
    GPUSHMOpt optimizer(lr_init,
                        /*friction=*/0.1f,
                        /*mom_decay=*/0.95f,
                        /*T_start=*/0.5f,
                        /*T_end=*/1e-3f,
                        /*aH_start=*/0.7f,
                        /*aH_end=*/0.99f,
                        total_steps);
    auto gpu_params=gpu_model.all_parameters();
    optimizer.init(gpu_params);
    auto gpu_grads=gpu_model.alloc_grad_buffers();

    int D=cfg.d_model, V=cfg.vocab_size, D4=4*D;
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets,SEQ*sizeof(int)));
    float* d_loss_buf;
    CUDA_CHECK(cudaMalloc(&d_loss_buf,SEQ*sizeof(float)));
    float* d_grad_out;
    CUDA_CHECK(cudaMalloc(&d_grad_out,SEQ*V*sizeof(float)));

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

    printf("%-8s | %-8s | %-8s | %-8s | %-8s | %-6s | %-6s | %-8s | %-8s\n",
           "Step","F(loss)","CE","Entropy","GNorm","α_H","α_L","T","Vedic");
    printf("---------|---------|---------|---------|---------|--------|--------|---------|--------\n");
    fflush(stdout);

    int64_t step=0;
    float   best_loss=999.f;
    int     vedic_checks=0, vedic_pass=0;
    char    vedic_status[8]="N/A";

    for (int epoch=0;epoch<EPOCHS;++epoch) {
        printf("\n-- Epoch %d/%d --\n",epoch+1,EPOCHS); fflush(stdout);
        std::vector<std::pair<std::vector<int>,std::vector<int>>> micro_batches;

        while (loader.next_accum_batch(micro_batches)) {
            for (auto* g : gpu_grads)
                CUDA_CHECK(cudaMemset(g->data,0,g->size*sizeof(float)));

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

            float grad_norm=cuda_clip_gradients(gpu_grads,1.0f);
            optimizer.update(gpu_params,gpu_grads);

            if (step % 1000 == 0 && step > 0) {
                GPUTensor C_proxy = gpu_alloc(gpu_model.last_hidden.rows,
                                              gpu_model.gpu_lm_head.cols);
                cuda_vedic_gemm(gpu_model.last_hidden, gpu_model.gpu_lm_head, C_proxy);
                VedicVerifyResult vr = cuda_vedic_verify(
                    gpu_model.last_hidden, gpu_model.gpu_lm_head, C_proxy, 0.05f);
                ++vedic_checks;
                if (vr.pass) { ++vedic_pass; snprintf(vedic_status,8,"PASS"); }
                else {
                    snprintf(vedic_status,8,"WARN");
                    printf("\n[Gunitasamuchayah] WARN @ step %lld err=%.4f\n",
                           (long long)step, vr.relative_error);
                }
            }

            if (step % 100 == 0) {
                float cur_T, cur_aH, cur_aL;
                optimizer.get_state(cur_T, cur_aH, cur_aL);
                printf("%-8lld | %-8.4f | %-8.4f | %-8.4f | %-8.3f | %-6.2f | %-6.2f | %-8.4f | %s\n",
                       (long long)step, avg_F, avg_CE, avg_S,
                       grad_norm, cur_aH, cur_aL, cur_T, vedic_status);
                fflush(stdout);
            }

            if (step>0 && step%1000==0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                gpu_model.sync_to_cpu(cpu_model);
                save_checkpoint(cpu_model,"logos_gpu_ckpt",(int)step);
                printf("Ckpt @ step %lld | best_F=%.4f | Vedic: %d/%d PASS\n",
                       (long long)step, best_loss, vedic_pass, vedic_checks);
                fflush(stdout);
            }
            ++step;
        }
        printf("Epoch %d done | Step=%lld | Best_F=%.4f | α_H=%.2f\n",
               epoch+1,(long long)step,best_loss,optimizer.alpha_H);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_targets); cudaFree(d_loss_buf); cudaFree(d_grad_out);
    for (auto* g : gpu_grads) delete g;
    gpu_model.sync_to_cpu(cpu_model);
    save_checkpoint(cpu_model,"logos_final",(int)step);

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Training Complete! (v10-HAM)             ║\n");
    printf("║  Steps: %-8lld | Best F: %.4f          ║\n",(long long)step,best_loss);
    printf("║  Gunitasamuchayah: %3d / %3d PASS        ║\n",vedic_pass,vedic_checks);
    printf("╚══════════════════════════════════════════╝\n");
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
    Tokenizer tok;
    if (!tok.load("vocab.bin")) { fprintf(stderr,"❌ vocab.bin not found\n"); return; }
    ModelConfig cfg; cfg.vocab_size=tok.vocab_size;
    cfg.d_model=128; cfg.num_heads=4; cfg.num_layers=4; cfg.max_seq_len=128;
    LOGOSModel cpu_model(cfg);
    if (ckpt_path!="none"&&!ckpt_path.empty()) load_checkpoint(cpu_model,ckpt_path);
    HyperConfig hyper_cfg; hyper_cfg.enabled=true; hyper_cfg.curvature=1.0f;
    ModelGPU gpu_model(cfg,hyper_cfg);
    gpu_model.load_from_cpu(cpu_model);
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
           "║  LOGOS GPU v9-fix                        ║\n"
           "║  T_floor | Anurupyena fix | PathIntegral ║\n"
           "╚══════════════════════════════════════════╝\n\n");

    std::string mode=(argc>1)?argv[1]:"--train";
    auto safe=[](const char* raw,const char* fb)->std::string{
        if (!raw) return fb;
        std::string s(raw);
        if (s.find("..")!=std::string::npos||s.find('\0')!=std::string::npos) return fb;
        return s;
    };

    try {
        if (mode=="--train") {
            std::string ds=(argc>2)?safe(argv[2],"dataset.txt"):"dataset.txt";
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
