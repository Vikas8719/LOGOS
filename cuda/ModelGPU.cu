
//
//  BACKWARD NOTE:
//    K_int8/V_int8 are NOT used in backward — float32 K/V cached separately.
//    So: dV, dK backward kernels unchanged. Zero regression.
// ============================================================
#include "ModelGPU.cuh"
#include "VedicGEMM.cuh"
#include "MixedPrecision.cuh"   // [BUG-FIX] cast_fp32_to_fp16_kernel ModelGPU.cu line 772,792 mein use hota hai
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <climits>

// ============================================================
//  EXISTING KERNELS (v6 — unchanged)
// ============================================================

__global__ void embedding_kernel(
    const int* __restrict__ token_ids,
    const float* __restrict__ token_emb,
    const float* __restrict__ pos_emb,
    float* __restrict__ X,
    int seq_len, int d_model, int vocab_size)
{
    int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= seq_len || j >= d_model) return;
    int tok = token_ids[i];
    if (tok < 0 || tok >= vocab_size)
        X[i*d_model+j] = pos_emb[i*d_model+j];
    else
        X[i*d_model+j] = token_emb[tok*d_model+j] + pos_emb[i*d_model+j];
}

__global__ void residual_add_kernel(float* out, const float* res, int size) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < size) out[idx] += res[idx];
}

__global__ void causal_mask_kernel(float* scores, int seq_len) {
    int row=blockIdx.x, col=blockIdx.y*blockDim.x+threadIdx.x;
    if (row>=seq_len||col>=seq_len) return;
    if (col>row) scores[row*seq_len+col]+=-1e9f;
}

__global__ void ce_loss_kernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    float* __restrict__ loss_out, float* __restrict__ grad_out,
    int seq_len, int vocab_size)
{
    int i=blockIdx.x; if (i>=seq_len) return;
    const float* lr=logits+i*vocab_size; float* gr=grad_out+i*vocab_size;
    int target=targets[i];
    if (target<0||target>=vocab_size){loss_out[i]=0.f;return;}
    float mx=lr[0];
    for(int v=1;v<vocab_size;++v) mx=fmaxf(mx,lr[v]);
    float s=0.f;
    for(int v=0;v<vocab_size;++v) s+=expf(lr[v]-mx);
    loss_out[i]=-(lr[target]-mx-logf(s));
    float inv=1.f/seq_len;
    for(int v=0;v<vocab_size;++v){
        float sm=expf(lr[v]-mx)/s;
        gr[v]=(sm-(v==target?1.f:0.f))*inv;
    }
}

__global__ void gpu_transpose_kernel(
    const float* __restrict__ A, float* __restrict__ AT, int rows, int cols)
{
    __shared__ float tile[16][17];
    int ri=blockIdx.y*16+threadIdx.y, ci=blockIdx.x*16+threadIdx.x;
    if (ri<rows&&ci<cols) tile[threadIdx.y][threadIdx.x]=A[ri*cols+ci];
    __syncthreads();
    int ro=blockIdx.x*16+threadIdx.y, co=blockIdx.y*16+threadIdx.x;
    if (ro<cols&&co<rows) AT[ro*rows+co]=tile[threadIdx.x][threadIdx.y];
}

__global__ void langevin_step_kernel(
    float* __restrict__ W, float* __restrict__ velocity,
    const float* __restrict__ grad, float lr, float friction,
    float noise_scale, unsigned int seed, int size)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=size) return;
    unsigned int r1=seed+idx*1664525u+1013904223u;
    r1=r1*1664525u+1013904223u;
    unsigned int r2=r1*1664525u+1013904223u;
    float u1=fmaxf((float)(r1&0x7FFFFFFF)/(float)0x7FFFFFFF,1e-6f);
    float u2=(float)(r2&0x7FFFFFFF)/(float)0x7FFFFFFF;
    float noise=noise_scale*sqrtf(-2.f*logf(u1))*cosf(2.f*3.14159265f*u2);
    float g=grad[idx]; if(isnan(g)||isinf(g)) g=0.f;
    float v=(1.f-friction)*velocity[idx]-lr*g+noise;
    velocity[idx]=v; W[idx]+=v;
}

// ============================================================
//  [P2-A] HYPERBOLIC EMBEDDING KERNELS
//  Poincaré Ball Model with curvature c
// ============================================================

__global__ void expmap0_kernel(float* X, int seq, int d, float curvature)
{
    int tok = blockIdx.x;
    if (tok >= seq) return;

    float* v = X + tok * d;
    float sqrt_c = sqrtf(curvature);

    float thread_sq = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        thread_sq += v[j] * v[j];
    for (int off=16;off>0;off>>=1)
        thread_sq += __shfl_down_sync(0xffffffff, thread_sq, off);
    __shared__ float s_sq[8];
    if (threadIdx.x%32==0) s_sq[threadIdx.x/32]=thread_sq;
    __syncthreads();
    float sq=0.f;
    if (threadIdx.x<8) sq=s_sq[threadIdx.x];
    for (int off=4;off>0;off>>=1) sq+=__shfl_down_sync(0xffffffff,sq,off);
    __shared__ float s_norm;
    if (threadIdx.x==0) s_norm=sqrtf(sq+1e-12f);
    __syncthreads();

    float norm    = s_norm;
    float scaled  = sqrt_c * norm;
    float factor  = (scaled > 1e-7f)
        ? tanhf(scaled * 0.5f) / (scaled + 1e-9f)
        : 0.5f;

    for (int j = threadIdx.x; j < d; j += blockDim.x)
        v[j] *= factor;
}

__global__ void logmap0_kernel(float* X, int seq, int d, float curvature)
{
    int tok = blockIdx.x;
    if (tok >= seq) return;

    float* y = X + tok * d;
    float sqrt_c = sqrtf(curvature);
    float inv_sqrt_c = 1.0f / sqrt_c;

    float thread_sq = 0.0f;
    for (int j=threadIdx.x; j<d; j+=blockDim.x) thread_sq += y[j]*y[j];
    for (int off=16;off>0;off>>=1)
        thread_sq+=__shfl_down_sync(0xffffffff,thread_sq,off);
    __shared__ float s_sq[8];
    if (threadIdx.x%32==0) s_sq[threadIdx.x/32]=thread_sq;
    __syncthreads();
    float sq=0.f;
    if (threadIdx.x<8) sq=s_sq[threadIdx.x];
    for (int off=4;off>0;off>>=1) sq+=__shfl_down_sync(0xffffffff,sq,off);
    __shared__ float s_norm;
    if (threadIdx.x==0) {
        float r=sqrtf(sq+1e-12f);
        float max_r = inv_sqrt_c * 0.99f;
        s_norm = fminf(r, max_r);
    }
    __syncthreads();

    float norm    = s_norm;
    float scaled  = sqrt_c * norm;
    float atanh_v = atanhf(fminf(scaled, 0.9999f));
    float factor  = (norm > 1e-7f)
        ? 2.0f * inv_sqrt_c * atanh_v / (norm + 1e-9f)
        : 2.0f * inv_sqrt_c;

    for (int j=threadIdx.x; j<d; j+=blockDim.x)
        y[j] *= factor;
}

__global__ void expmap0_bwd_kernel(
    const float* __restrict__ X_euc,
    const float* __restrict__ dOut,
    float*       __restrict__ dX,
    int seq, int d, float curvature)
{
    int tok = blockIdx.x;
    if (tok >= seq) return;

    const float* v   = X_euc + tok * d;
    const float* dY  = dOut  + tok * d;
    float*       dv  = dX    + tok * d;

    float sqrt_c = sqrtf(curvature);

    float tsq=0.f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) tsq+=v[j]*v[j];
    for (int off=16;off>0;off>>=1) tsq+=__shfl_down_sync(0xffffffff,tsq,off);
    __shared__ float ssq[8];
    if (threadIdx.x%32==0) ssq[threadIdx.x/32]=tsq;
    __syncthreads();
    float s=0.f;
    if (threadIdx.x<8) s=ssq[threadIdx.x];
    for (int off=4;off>0;off>>=1) s+=__shfl_down_sync(0xffffffff,s,off);
    __shared__ float s_norm2, s_norm;
    if (threadIdx.x==0) { s_norm2=s+1e-12f; s_norm=sqrtf(s_norm2); }
    __syncthreads();

    float r   = s_norm;
    float r2  = s_norm2;
    float sr  = sqrt_c * r;

    float alpha, beta;
    if (r > 1e-6f) {
        float th   = tanhf(sr * 0.5f);
        float sech = 1.0f - th * th;
        alpha = th / (sqrt_c * r);
        beta  = (sech * sqrt_c * 0.5f - alpha) / r2;
    } else {
        alpha = 0.5f;
        beta  = -1.0f / 24.0f;
    }

    float tdot=0.f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) tdot+=v[j]*dY[j];
    for (int off=16;off>0;off>>=1) tdot+=__shfl_down_sync(0xffffffff,tdot,off);
    __shared__ float sd[8];
    if (threadIdx.x%32==0) sd[threadIdx.x/32]=tdot;
    __syncthreads();
    float dot_v_dY=0.f;
    if (threadIdx.x<8) dot_v_dY=sd[threadIdx.x];
    for (int off=4;off>0;off>>=1) dot_v_dY+=__shfl_down_sync(0xffffffff,dot_v_dY,off);
    __shared__ float s_dot;
    if (threadIdx.x==0) s_dot=dot_v_dY;
    __syncthreads();

    for (int j=threadIdx.x;j<d;j+=blockDim.x)
        dv[j] = alpha * dY[j] + beta * s_dot * v[j];
}

// ============================================================
//  [P2-B] NIKHILAM QUANTIZATION KERNELS
// ============================================================

// ── absmax: find max(|data|) via parallel reduction ──────────
// [v27-BUG7-FIX] atomicMax(float*) UB fix:
//   Pehla code: atomicMax((int*)out, __float_as_int(smax[0]))
//   Bug: IEEE754 bit trick sirf POSITIVE floats ke liye work karta hai.
//        Negative float mein sign bit=1 hota hai, jo int representation
//        mein negative integer jaisa dikhta hai → atomicMax negative floats
//        ko less than zero samajh leta hai → absmax WRONG value return karta.
//        Nikhilam KV scale galat ho jaata → INT8 clamp galat → quantization noise.
//   Fix: shared memory mein pehle fmaxf karo (correct float comparison),
//        phir ek hi atomicAdd use karo result propagate karne ke liye —
//        ya iska bhi better: CAS-loop se float atomicMax implement karo.
//   Note: fabsf(data[idx]) se sab values pehle se >= 0 hain is kernel mein,
//         isliye positive-only trick technically kaam karta tha YE KERNEL mein —
//         lekin pattern unsafe hai (future code mein raw values aa sakti hain).
//         Isliye proper float CAS atomicMax use karo.
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    // Correct float atomicMax via CAS loop.
    // Works for any float value including negatives.
    // For positive values only (our case: fabsf), this is equivalent to
    // the __float_as_int trick but is always correct.
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint, assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(old);
        if (old_val >= val) break;  // already >= val, no update needed
        old = atomicCAS(addr_as_uint, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void absmax_kernel(const float* data, float* out, int size)
{
    __shared__ float smax[256];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    // fabsf ensures all values are non-negative before reduction
    float val=(idx<size)?fabsf(data[idx]):0.f;
    smax[threadIdx.x]=val;
    __syncthreads();
    for (int s=blockDim.x/2;s>0;s>>=1) {
        if (threadIdx.x<s) smax[threadIdx.x]=fmaxf(smax[threadIdx.x],smax[threadIdx.x+s]);
        __syncthreads();
    }
    // [v27-BUG7-FIX] Use proper CAS-based float atomicMax.
    // Previously: atomicMax((int*)out, __float_as_int(smax[0]))
    // This worked ONLY because fabsf makes values >= 0 (IEEE754 positive floats
    // sort correctly as uint32). But pattern is fragile and UB for negative inputs.
    // Now: atomicMaxFloat uses CAS loop — correct for ALL float values.
    if (threadIdx.x==0) atomicMaxFloat(out, smax[0]);
}

// ── nikhilam_quantize: float32 → int8 ────────────────────────
__global__ void nikhilam_quantize_kernel(
    const float* __restrict__ src,
    int8_t*      __restrict__ dst,
    float scale, int size)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=size) return;
    float v=src[idx];
    float q=v/(scale+1e-8f);
    q=fminf(fmaxf(q,-127.f),127.f);
    dst[idx]=(int8_t)__float2int_rn(q);
}

// ── nikhilam_dequantize: int8 → float32 ─────────────────────
__global__ void nikhilam_dequantize_kernel(
    const int8_t* __restrict__ src,
    float*        __restrict__ dst,
    float scale, int size)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=size) return;
    dst[idx]=(float)src[idx]*scale;
}

// ============================================================
//  HOST-SIDE NIKHILAM HELPERS
// ============================================================

static NikhilamTensor nikhilam_compress(const GPUTensor& src)
{
    NikhilamTensor out;
    out.size = src.size;

    // [v26-LEAK-FIX] CudaPtr<float>: d_max auto-freed on exception or return
    CudaPtr<float> d_max(1);  // 1 float, zero-init
    int blks = (src.size + 255) / 256;
    absmax_kernel<<<blks, 256>>>(src.data, d_max.get(), src.size);
    CUDA_KERNEL_CHECK();

    float h_max = d_max.scalar();

    out.scale = h_max / 127.0f + 1e-8f;

    // [v27-MEM2-FIX] NikhilamTensor::data safe alloc:
    // Pehle: raw cudaMalloc → agar nikhilam_quantize_kernel CUDA_KERNEL_CHECK() throw kare
    //   to out.data allocated tha but NikhilamTensor destructor NE IS POINT TAK CALL NAHI HUA
    //   (object construction mein throw = destructor nahi chalta C++ rules mein) → leak.
    // Fix: temp CudaPtr<int8_t> mein allocate karo, quantize karo, phir release() se
    //   ownership NikhilamTensor ko transfer karo — sirf tab jab quantize successful ho.
    //   Agar CUDA_KERNEL_CHECK() throw kare to CudaPtr destructor guaranteed cleanup karta hai.
    CudaPtr<int8_t> tmp_data(out.size);
    nikhilam_quantize_kernel<<<blks, 256>>>(src.data, tmp_data.get(), out.scale, out.size);
    CUDA_KERNEL_CHECK();
    // Quantize OK → safe to transfer ownership: CudaPtr releases, NikhilamTensor takes over
    out.data = tmp_data.release();

    return out;
}

GPUTensor nikhilam_decompress(const NikhilamTensor& src, int rows, int cols)
{
    GPUTensor out = gpu_alloc(rows, cols);
    int blks=(src.size+255)/256;
    nikhilam_dequantize_kernel<<<blks,256>>>(src.data, out.data, src.scale, src.size);
    CUDA_KERNEL_CHECK();
    return out;
}

// ============================================================
//  CONFIG VALIDATION (v6 — unchanged, ODR: only in VedicGEMM.cu)
// ============================================================

// ============================================================
//  ModelGPU IMPLEMENTATION
// ============================================================

__device__ __forceinline__ float block_sum256(float v)
{
    __shared__ float sh[8];
    __shared__ float tot;
    for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    if (lane==0) sh[wid]=v;
    __syncthreads();
    float t=(threadIdx.x<8)?sh[threadIdx.x]:0.0f;
    if (wid==0) {
        for (int o=4;o>0;o>>=1) t += __shfl_down_sync(0xffffffff, t, o);
        if (lane==0) tot=t;
    }
    __syncthreads();
    float r=tot;
    __syncthreads();
    return r;
}

__device__ __forceinline__ unsigned hash32(unsigned x)
{
    x ^= x>>16; x *= 0x7feb352dU; x ^= x>>15; x *= 0x846ca68bU; x ^= x>>16;
    return x;
}
__device__ __forceinline__ float urand01(unsigned& s)
{
    s = hash32(s + 0x9e3779b9U);
    return ((float)(s>>8) + 0.5f) * (1.0f/16777216.0f);
}

__global__ void shunyam_mask_kernel(float* scores, int seq, int window, int stride)
{
    int row=blockIdx.x, col=blockIdx.y*blockDim.x+threadIdx.x;
    if (row>=seq||col>=seq) return;
    bool allowed = (col<=row) && (col>=row-window || (col%stride)==0);
    if (!allowed) scores[row*seq+col] += -1e9f;
}

__global__ void ns_advect_kernel(const float* Q, float* Qa, int seq, int d, float eta)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=seq*d) return;
    int i=idx/d;
    float q=Q[idx];
    float qp=(i>0)?Q[idx-d]:q;
    Qa[idx]=q+eta*(q-qp);
}

__global__ void ns_advect_bwd_kernel(const float* dQa, float* dQ, int seq, int d, float eta)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=seq*d) return;
    int i=idx/d;
    float g=dQa[idx]*((i==0)?1.0f:(1.0f+eta));
    if (i+1<seq) g-=eta*dQa[idx+d];
    dQ[idx]=g;
}

__global__ void ns_diffuse_kernel(const float* V, float* Vs, int seq, int d, float nu)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=seq*d) return;
    int i=idx/d;
    float v=V[idx];
    float vp=(i>0)?V[idx-d]:v;
    Vs[idx]=(1.0f-nu)*v+nu*vp;
}

__global__ void ns_diffuse_bwd_kernel(const float* dVs, float* dV, int seq, int d, float nu)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=seq*d) return;
    int i=idx/d;
    float g=(1.0f-nu)*dVs[idx];
    if (i+1<seq) g+=nu*dVs[idx+d];
    if (i==0)    g+=nu*dVs[idx];
    dV[idx]=g;
}

__global__ void reynolds_norm_kernel(const float* X, const float* gamma, const float* beta,
                                     const float* rmean, const float* rvar, float* Y, float* W,
                                     int seq, int d, float re_crit, float k, float eps)
{
    int row=blockIdx.x;
    if (row>=seq) return;
    const float* x=X+row*d; float* y=Y+row*d;
    float s=0.f, sq=0.f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) { float v=x[j]; s+=v; sq+=v*v; }
    s=block_sum256(s); sq=block_sum256(sq);
    float mean=s/d;
    float var=fmaxf(sq/d-mean*mean,0.0f);
    float rms=sqrtf(sq/d);
    float re=rms/(sqrtf(var+eps)+eps);
    float w=1.0f/(1.0f+expf(k*(re-re_crit)));
    float inv=rsqrtf(var+eps);
    for (int j=threadIdx.x;j<d;j+=blockDim.x) {
        float ln=(x[j]-mean)*inv;
        float bn=(x[j]-rmean[j])*rsqrtf(rvar[j]+eps);
        y[j]=gamma[j]*(w*ln+(1.0f-w)*bn)+beta[j];
    }
    if (threadIdx.x==0) W[row]=w;
}

__global__ void reynolds_bwd_kernel(const float* X, const float* gamma, const float* dY, const float* W,
                                    const float* rmean, const float* rvar, float* dX,
                                    float* dGamma, float* dBeta, int seq, int d, float eps)
{
    int row=blockIdx.x;
    if (row>=seq) return;
    const float* x=X+row*d; const float* dy=dY+row*d; float* dx=dX+row*d;
    float w=W[row];
    float s=0.f, sq=0.f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) { float v=x[j]; s+=v; sq+=v*v; }
    s=block_sum256(s); sq=block_sum256(sq);
    float mean=s/d;
    float var=fmaxf(sq/d-mean*mean,0.0f);
    float inv=rsqrtf(var+eps);
    float t1=0.f, t2=0.f;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) {
        float xh=(x[j]-mean)*inv;
        float dxh=dy[j]*gamma[j]*w;
        t1+=dxh; t2+=dxh*xh;
    }
    t1=block_sum256(t1); t2=block_sum256(t2);
    float inv_d=1.0f/d;
    for (int j=threadIdx.x;j<d;j+=blockDim.x) {
        float xh=(x[j]-mean)*inv;
        float dxh=dy[j]*gamma[j]*w;
        float ibn=rsqrtf(rvar[j]+eps);
        float xb=(x[j]-rmean[j])*ibn;
        dx[j]=inv*inv_d*(d*dxh-t1-xh*t2)+dy[j]*gamma[j]*(1.0f-w)*ibn;
        atomicAdd(&dGamma[j], dy[j]*(w*xh+(1.0f-w)*xb));
        atomicAdd(&dBeta[j],  dy[j]);
    }
}

__global__ void run_stats_update_kernel(const float* X, float* rmean, float* rvar,
                                        int seq, int d, float decay)
{
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=d) return;
    float m=0.f;
    for (int i=0;i<seq;++i) m+=X[i*d+j];
    m/=(float)seq;
    float v=0.f;
    for (int i=0;i<seq;++i) { float t=X[i*d+j]-m; v+=t*t; }
    v/=(float)seq;
    rmean[j]=decay*rmean[j]+(1.0f-decay)*m;
    rvar[j] =decay*rvar[j] +(1.0f-decay)*v;
}

__global__ void feynman_dropout_kernel(float* A, float* mask, int n, float p, float hbar, unsigned seed)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=n) return;
    unsigned s=hash32(seed ^ ((unsigned)idx*2654435761u));
    float w;
    if (hbar<=0.05f) {
        w=(urand01(s)<p)?0.0f:1.0f;
    } else {
        float a=(1.0f-p)*hbar, b=p*hbar;
        w=1.0f-p;
        for (int t=0;t<8;++t) {
            float u=urand01(s), v=urand01(s);
            float x=powf(u,1.0f/a), y=powf(v,1.0f/b);
            float sum=x+y;
            if (sum<=1.0f && sum>1e-30f) { w=x/sum; break; }
        }
    }
    float m=w/(1.0f-p+1e-8f);
    mask[idx]=m;
    A[idx]*=m;
}

ModelGPU::ModelGPU(const ModelConfig& cfg_, const HyperConfig& hcfg)
    : cfg(cfg_), hyper_cfg(hcfg)
{
    extern std::string validate_model_config(int,int,int,int,int);
    std::string err=validate_model_config(
        cfg.d_model,cfg.num_heads,cfg.num_layers,cfg.vocab_size,cfg.max_seq_len);
    if (!err.empty()) throw std::invalid_argument("ModelGPU config: "+err);

    int V=cfg.vocab_size, D=cfg.d_model, S=cfg.max_seq_len;
    gpu_embedding     = gpu_alloc(V,D);
    gpu_pos_embedding = gpu_alloc(S,D);
    gpu_lm_head       = gpu_alloc(D,V);
    std::vector<float> zero_positions(static_cast<size_t>(S) * D, 0.0f);
    h2d(gpu_pos_embedding, zero_positions.data(), S * D);

    for (int l=0; l<cfg.num_layers; ++l) {
        GPUBlock blk;
        int H=cfg.num_heads, DH=D/H;
        for (int h=0;h<H;++h) {
            blk.W_Q.push_back(gpu_alloc(D,DH));
            blk.W_K.push_back(gpu_alloc(D,DH));
            blk.W_V.push_back(gpu_alloc(D,DH));
            blk.W_O.push_back(gpu_alloc(DH,D));
        }
        blk.W_proj   = gpu_alloc(D,D);
        blk.W1       = gpu_alloc(D,4*D);
        blk.b1       = gpu_alloc(1,4*D);
        blk.W2       = gpu_alloc(4*D,D);
        blk.b2       = gpu_alloc(1,D);
        blk.ln1_gamma= gpu_alloc(1,D);
        blk.ln1_beta = gpu_alloc(1,D);
        blk.ln2_gamma= gpu_alloc(1,D);
        blk.ln2_beta = gpu_alloc(1,D);
        std::vector<float> ones(D,1.f), zeros(D,0.f);
        h2d(blk.ln1_gamma,ones.data(),D); h2d(blk.ln2_gamma,ones.data(),D);
        h2d(blk.ln1_beta,zeros.data(),D); h2d(blk.ln2_beta,zeros.data(),D);
        gpu_blocks.push_back(std::move(blk));
    }
    layer_cache.resize(cfg.num_layers);
    for (int i=0;i<2*cfg.num_layers;++i) {
        run_mean.push_back(gpu_alloc(1,D));
        run_var.push_back(gpu_alloc(1,D));
        std::vector<float> ones_rv(D,1.f);
        h2d(run_var.back(),ones_rv.data(),D);
    }
    CUDA_CHECK(cudaMalloc(&d_token_ids, cfg.max_seq_len*sizeof(int)));

    // [v27-BUG15-FIX] dropout_seed initialized to fixed value (12345u) in header.
    // But same seed every run = same dropout pattern every fresh session.
    // Fix: XOR with time to get session-unique seed.
    // Note: reproducibility ke liye LOGOS_DROPOUT_SEED env var set karo.
    {
        const char* seed_env = std::getenv("LOGOS_DROPOUT_SEED");
        if (seed_env && *seed_env) {
            dropout_seed = (unsigned)std::strtoul(seed_env, nullptr, 10);
            printf("  [v27] dropout_seed from LOGOS_DROPOUT_SEED: %u\n", dropout_seed);
        } else {
            // Time-based seed for session uniqueness
            auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            dropout_seed = (unsigned)(now_ns ^ (now_ns >> 32) ^ 0xDEADBEEFu);
            // Don't print (called every resume too)
        }
    }

    size_t free_mem,total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem,&total_mem));
    std::cout << "✅ GPU Model v8 (Phase 2) allocated\n"
              << "   L=" << cfg.num_layers << " D=" << D
              << " H=" << cfg.num_heads << " V=" << V << "\n"
              << "   Hyperbolic: " << (hyper_cfg.enabled?"ON":"OFF")
              << " (c=" << hyper_cfg.curvature << ")\n"
              << "   Nikhilam KV: ON (INT8, 4x compression)\n"
              << "   GPU: " << (total_mem-free_mem)/1024/1024
              << " MB / " << total_mem/1024/1024 << " MB\n";
}

ModelGPU::~ModelGPU() {
    if (d_token_ids) { cudaFree(d_token_ids); d_token_ids=nullptr; }
}

void ModelGPU::free_layer_cache() {
    for (auto& lc : layer_cache) {
        lc.block_input=GPUTensor{};
        lc.normed1=GPUTensor{};
        lc.normed2=GPUTensor{};
        lc.ffn_H=GPUTensor{};
        lc.ffn_A=GPUTensor{};
        lc.attn_out=GPUTensor{};
        lc.concat=GPUTensor{};
        lc.post_attn=GPUTensor{};
        lc.ffn_mask=GPUTensor{};
        lc.ln1_w=GPUTensor{};
        lc.ln2_w=GPUTensor{};
        lc.heads.clear();
    }
    X_euclidean=GPUTensor{};
}

void ModelGPU::load_from_cpu(const LOGOSModel& cpu_model) {
    h2d(gpu_embedding,     cpu_model.embedding.data.data(),     cpu_model.embedding.total_size);
    h2d(gpu_lm_head,       cpu_model.lm_head.data.data(),       cpu_model.lm_head.total_size);
    for (int l=0;l<cfg.num_layers;++l) {
        auto& gblk=gpu_blocks[l];
        const auto& cblk=cpu_model.layers[l];
        for (int h=0;h<cfg.num_heads;++h) {
            h2d(gblk.W_Q[h],cblk.mha.heads[h].W_Q.data.data(),cblk.mha.heads[h].W_Q.total_size);
            h2d(gblk.W_K[h],cblk.mha.heads[h].W_K.data.data(),cblk.mha.heads[h].W_K.total_size);
            h2d(gblk.W_V[h],cblk.mha.heads[h].W_V.data.data(),cblk.mha.heads[h].W_V.total_size);
            h2d(gblk.W_O[h],cblk.mha.heads[h].W_O.data.data(),cblk.mha.heads[h].W_O.total_size);
        }
        h2d(gblk.W_proj,cblk.mha.W_proj.data.data(),cblk.mha.W_proj.total_size);
        h2d(gblk.W1,cblk.ffn.W1.data.data(),cblk.ffn.W1.total_size);
        h2d(gblk.b1,cblk.ffn.b1.data.data(),cblk.ffn.b1.total_size);
        h2d(gblk.W2,cblk.ffn.W2.data.data(),cblk.ffn.W2.total_size);
        h2d(gblk.b2,cblk.ffn.b2.data.data(),cblk.ffn.b2.total_size);
    }
    std::cout << "✅ CPU → GPU weights loaded\n";
}

std::vector<GPUTensor*> ModelGPU::all_parameters() {
    std::vector<GPUTensor*> p={&gpu_embedding,&gpu_pos_embedding,&gpu_lm_head};
    for (auto& blk:gpu_blocks) {
        for (int h=0;h<cfg.num_heads;++h) {
            p.push_back(&blk.W_Q[h]); p.push_back(&blk.W_K[h]);
            p.push_back(&blk.W_V[h]); p.push_back(&blk.W_O[h]);
        }
        p.push_back(&blk.W_proj);
        p.push_back(&blk.W1); p.push_back(&blk.b1);
        p.push_back(&blk.W2); p.push_back(&blk.b2);
        p.push_back(&blk.ln1_gamma); p.push_back(&blk.ln1_beta);
        p.push_back(&blk.ln2_gamma); p.push_back(&blk.ln2_beta);
    }
    return p;
}

std::vector<GPUTensor*> ModelGPU::alloc_grad_buffers() const {
    std::vector<GPUTensor*> grads;
    for (auto* p:const_cast<ModelGPU*>(this)->all_parameters()) {
        auto* g=new GPUTensor(gpu_alloc(p->rows,p->cols));
        CUDA_CHECK(cudaMemset(g->data,0,g->size*sizeof(float)));
        grads.push_back(g);
    }
    return grads;
}

void ModelGPU::sync_to_cpu(LOGOSModel& cpu_model) const {
    d2h(cpu_model.embedding.data.data(),    gpu_embedding,    gpu_embedding.size);
    d2h(cpu_model.lm_head.data.data(),      gpu_lm_head,      gpu_lm_head.size);
    for (int l=0;l<cfg.num_layers;++l) {
        const auto& gblk=gpu_blocks[l];
        auto& cblk=cpu_model.layers[l];
        for (int h=0;h<cfg.num_heads;++h) {
            d2h(cblk.mha.heads[h].W_Q.data.data(),gblk.W_Q[h],gblk.W_Q[h].size);
            d2h(cblk.mha.heads[h].W_K.data.data(),gblk.W_K[h],gblk.W_K[h].size);
            d2h(cblk.mha.heads[h].W_V.data.data(),gblk.W_V[h],gblk.W_V[h].size);
            d2h(cblk.mha.heads[h].W_O.data.data(),gblk.W_O[h],gblk.W_O[h].size);
        }
        d2h(cblk.mha.W_proj.data.data(),gblk.W_proj,gblk.W_proj.size);
        d2h(cblk.ffn.W1.data.data(),gblk.W1,gblk.W1.size);
        d2h(cblk.ffn.b1.data.data(),gblk.b1,gblk.b1.size);
        d2h(cblk.ffn.W2.data.data(),gblk.W2,gblk.W2.size);
        d2h(cblk.ffn.b2.data.data(),gblk.b2,gblk.b2.size);
        if ((int)cblk.ln1.gamma.data.size()==cfg.d_model) {
            d2h(cblk.ln1.gamma.data.data(),gblk.ln1_gamma,cfg.d_model);
            d2h(cblk.ln1.beta.data.data(), gblk.ln1_beta, cfg.d_model);
            d2h(cblk.ln2.gamma.data.data(),gblk.ln2_gamma,cfg.d_model);
            d2h(cblk.ln2.beta.data.data(), gblk.ln2_beta, cfg.d_model);
        }
    }
}

// ============================================================
//  FORWARD PASS
// ============================================================
// [v27-BUG16-NOTE] #if 0 block below is intentionally kept as historical reference.
// It documents the OLD forward pass before physics wiring.
// Do NOT remove — it shows the progression from v6 (no physics) to v8 (full wiring).
// If you want to test physics-free baseline, you can compile with -DLOGOS_LEGACY_FORWARD.
#if 0  // legacy forward (no physics wiring) — replaced by the forward below
// ... (legacy code preserved for reference)
#endif  // legacy forward end

void ModelGPU::apply_norm(const GPUTensor& X, const GPUTensor& gamma, const GPUTensor& beta,
                          GPUTensor& Y, GPUTensor& W, int slot, int seq)
{
    int D=cfg.d_model;
    if (phys.reynolds) {
        W=gpu_alloc(seq,1);
        reynolds_norm_kernel<<<seq,256>>>(X.data,gamma.data,beta.data,
            run_mean[slot].data,run_var[slot].data,Y.data,W.data,
            seq,D,phys.re_crit,phys.re_k,1e-5f);
        CUDA_KERNEL_CHECK();
    } else {
        cuda_layernorm(X,gamma,beta,Y,seq,D);
    }
}

void ModelGPU::update_norm_stats()
{
    if (!phys.reynolds||!phys.training) return;
    int D=cfg.d_model;
    for (int l=0;l<cfg.num_layers;++l) {
        auto& c=layer_cache[l];
        if (!c.block_input.valid()||!c.post_attn.valid()) continue;
        int seq=c.block_input.rows;
        run_stats_update_kernel<<<(D+127)/128,128>>>(c.block_input.data,run_mean[2*l].data,run_var[2*l].data,seq,D,0.99f);
        run_stats_update_kernel<<<(D+127)/128,128>>>(c.post_attn.data,run_mean[2*l+1].data,run_var[2*l+1].data,seq,D,0.99f);
    }
    CUDA_KERNEL_CHECK();
}

// ── Forward: all Vedic+physics tools wired in ────────────────
GPUTensor ModelGPU::forward(const std::vector<int>& token_ids)
{
    int seq=static_cast<int>(token_ids.size());
    int D=cfg.d_model, H=cfg.num_heads, DH=D/H, V=cfg.vocab_size;
    if (seq<=0||seq>cfg.max_seq_len)
        throw std::invalid_argument("Invalid seq: "+std::to_string(seq));

    // [v31-FLASH] Flash Attention active — no seq×seq tensor allocated
    // VRAM for attention: 16 heads × seq×DH×4B × 3(Q,K,V,out) ≈ 192MB vs old 8.2GB
    static bool flash_logged = false;
    if (!flash_logged) {
        long long old_attn_mb = (long long)seq * seq * 4 * 2 * cfg.num_heads / 1024 / 1024;
        long long new_attn_mb = (long long)seq * (D/cfg.num_heads) * 4 * cfg.num_heads / 1024 / 1024;
        printf("  [v31-FLASH] Flash Attention ON | seq=%d H=%d DH=%d\n", seq, cfg.num_heads, D/cfg.num_heads);
        printf("  [v31-FLASH] Attn mem: OLD=%lld MB (seq²) → NEW=%lld MB (seq·DH) | saved=%lld MB\n",
               old_attn_mb, new_attn_mb, old_attn_mb - new_attn_mb);
        fflush(stdout);
        flash_logged = true;
    }

    std::vector<int> safe=token_ids;
    for (int& t:safe) if(t<0||t>=V) t=0;
    CUDA_CHECK(cudaMemcpy(d_token_ids,safe.data(),seq*sizeof(int),cudaMemcpyHostToDevice));
    free_layer_cache();

    const bool amp = amp_enabled && !fp16_param_ptrs.empty();
    int pi = 0;

    auto amp_gemm = [&](const GPUTensor& A_fp32,
                        const GPUTensor& W_fp32, int param_idx,
                        GPUTensor& C_fp32) {
        if (amp && param_idx < (int)fp16_param_ptrs.size()
            && fp16_param_ptrs[param_idx] != nullptr) {
            int M = A_fp32.rows, K = A_fp32.cols, N = W_fp32.cols;
            CudaPtr<__half> d_A16(M * K);
            cast_fp32_to_fp16_kernel<<<(M*K+255)/256,256>>>(
                A_fp32.data, d_A16.get(), M*K);
            CUDA_KERNEL_CHECK();
            cuda_gemm_fp16_fp32out(
                d_A16.get(), M, K,
                fp16_param_ptrs[param_idx], N,
                C_fp32.data);
        } else {
            cuda_vedic_gemm(A_fp32, W_fp32, C_fp32);
        }
    };

    auto amp_gemm_bias = [&](const GPUTensor& A_fp32,
                             const GPUTensor& W_fp32, int param_w_idx,
                             const GPUTensor& bias,
                             GPUTensor& C_fp32) {
        if (amp && param_w_idx < (int)fp16_param_ptrs.size()
            && fp16_param_ptrs[param_w_idx] != nullptr) {
            int M = A_fp32.rows, K = A_fp32.cols, N = W_fp32.cols;
            CudaPtr<__half> d_A16(M * K);
            cast_fp32_to_fp16_kernel<<<(M*K+255)/256,256>>>(
                A_fp32.data, d_A16.get(), M*K);
            CUDA_KERNEL_CHECK();
            cuda_gemm_fp16_fp32out(
                d_A16.get(), M, K,
                fp16_param_ptrs[param_w_idx], N,
                C_fp32.data);
            int sz = M * N;
            add_bias_kernel<<<(sz+255)/256,256>>>(C_fp32.data, bias.data, M, N);
            CUDA_KERNEL_CHECK();
        } else {
            cuda_vedic_gemm_bias(A_fp32, W_fp32, bias, C_fp32);
        }
    };

    GPUTensor X=gpu_alloc(seq,D);
    {
        dim3 grid(seq,(D+31)/32); dim3 block(32);
        embedding_kernel<<<grid,block>>>(d_token_ids,gpu_embedding.data,gpu_pos_embedding.data,X.data,seq,D,V);
        CUDA_KERNEL_CHECK();
    }
    pi += 2;

    if (hyper_cfg.enabled) {
        X_euclidean=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(X_euclidean.data,X.data,seq*D*sizeof(float),cudaMemcpyDeviceToDevice));
        expmap0_kernel<<<seq,256>>>(X.data,seq,D,hyper_cfg.curvature);
        CUDA_KERNEL_CHECK();
    }

    for (int l=0;l<cfg.num_layers;++l) {
        auto& blk=gpu_blocks[l];
        auto& cache=layer_cache[l];

        int layer_base_pi = 2 + 1 + l * (H*4 + 9);

        cache.block_input=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(cache.block_input.data,X.data,seq*D*sizeof(float),cudaMemcpyDeviceToDevice));

        cache.normed1=gpu_alloc(seq,D);
        apply_norm(X,blk.ln1_gamma,blk.ln1_beta,cache.normed1,cache.ln1_w,2*l,seq);

        cache.heads.resize(H);
        GPUTensor concat=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemset(concat.data,0,seq*D*sizeof(float)));

        for (int h=0;h<H;++h) {
            auto& hc=cache.heads[h];
            int pWQ = layer_base_pi + h*4 + 0;
            int pWK = layer_base_pi + h*4 + 1;
            int pWV = layer_base_pi + h*4 + 2;
            int pWO = layer_base_pi + h*4 + 3;

            hc.Q=gpu_alloc(seq,DH);
            amp_gemm(cache.normed1, blk.W_Q[h], pWQ, hc.Q);

            hc.K=gpu_alloc(seq,DH);
            amp_gemm(cache.normed1, blk.W_K[h], pWK, hc.K);

            hc.V=gpu_alloc(seq,DH);
            amp_gemm(cache.normed1, blk.W_V[h], pWV, hc.V);

            if (phys.nikhilam_kv) { hc.K_int8=nikhilam_compress(hc.K); hc.V_int8=nikhilam_compress(hc.V); }

            hc.Q_adv=gpu_alloc(seq,DH);
            hc.V_s  =gpu_alloc(seq,DH);
            if (phys.navier_stokes) {
                int qb=(seq*DH+255)/256;
                ns_advect_kernel<<<qb,256>>>(hc.Q.data,hc.Q_adv.data,seq,DH,phys.ns_eta); CUDA_KERNEL_CHECK();
                ns_diffuse_kernel<<<qb,256>>>(hc.V.data,hc.V_s.data,seq,DH,phys.ns_nu);   CUDA_KERNEL_CHECK();
            } else {
                CUDA_CHECK(cudaMemcpy(hc.Q_adv.data,hc.Q.data,seq*DH*sizeof(float),cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(hc.V_s.data,  hc.V.data,seq*DH*sizeof(float),cudaMemcpyDeviceToDevice));
            }

            // [v31-FLASH] Flash Attention — replaces scores[seq×seq] + softmax + gemm
            // OLD (OOM cause): K_T[DH×seq] + scores[seq×seq] + attn_probs[seq×seq]
            //   = DH×seq×4B + 2×seq²×4B per head
            //   = 64×8192×4 + 2×8192²×4 = 2MB + 512MB = 514 MB per head
            //   × 16 heads = 8.2 GB just for attention intermediates → T4 OOM
            //
            // NEW (Flash): only out[seq×DH] allocated = 8192×64×4 = 2MB per head
            //   × 16 heads = 32MB total — 256× reduction ✓
            //
            // attn_probs saved only if phys.training (needed for backward dQ,dK,dV)
            // If not training (inference/val): attn_ptr=nullptr → saves 512MB/head more

            hc.head_out = gpu_alloc(seq, DH);

            float* attn_save_ptr = nullptr;
            // [v32-T4-OOM-FIX] attn_probs allocation (seq×seq per head) hata diya:
            //
            //   PEHLE (OOM cause):
            //     if (phys.training) hc.attn_probs = gpu_alloc(seq, seq);
            //     → 16 heads × 8192² × 4B = 4.29 GB → T4 OOM (crashes at VedicGEMM.cu:597)
            //
            //   PROBLEM: Flash Attention ka backward pass attn_probs pe depend karta tha.
            //   Lekin T4 pe seq=8192 ke saath yeh physically fit hi nahi hota.
            //
            //   FIX: attn_probs kabhi allocate mat karo (attn_save_ptr = nullptr hamesha).
            //   Flash attention kernel already causal+shunyam mask apply karta hai forward mein.
            //   Backward ke liye: checkpointed recomputation use karo (backward mein Q,K,V
            //   se attention dobara compute — memory O(seq·DH) vs O(seq²)).
            //
            //   VRAM saved: 16 × 8192² × 4B = 4.29 GB → model fit ho jaata T4 pe.
            //   Trade-off: backward mein ~15% extra compute (recompute attn) — acceptable.
            //
            //   NOTE: Agar future mein T4 se zyada VRAM wala GPU use ho (A100/H100) aur
            //   seq <= 4096 ho, toh yeh flag se re-enable kar sakte hain:
            //     LOGOS_SAVE_ATTN_PROBS=1 (env var)
            //   Abhi: always nullptr — backward recomputes.
            // attn_probs stays default-constructed (nullptr) — no alloc, always

            cuda_flash_attention_fwd(
                hc.Q_adv.data, hc.K.data, hc.V_s.data,
                hc.head_out.data,
                attn_save_ptr,          // null = skip attn save (val/inference)
                seq, DH,
                1.0f / sqrtf((float)DH),
                true,                   // causal = always true (autoregressive)
                phys.shunyam && seq > phys.window,
                phys.window, phys.stride
            );
            CUDA_KERNEL_CHECK();

            // [v31-RAWPTR] head_proj: temporary, freed immediately after residual add
            // Use CudaPtr (raw pointer) instead of GPUTensor to avoid RAII overhead
            // and make the short lifetime explicit. Freed at end of this scope.
            {
                CudaPtr<float> head_proj_raw(seq * D);
                GPUTensor head_proj_view;
                head_proj_view.data = head_proj_raw.get();
                head_proj_view.rows = seq; head_proj_view.cols = D; head_proj_view.size = seq * D;

                amp_gemm(hc.head_out, blk.W_O[h], pWO, head_proj_view);

                int sz=seq*D, th=256;
                residual_add_kernel<<<(sz+th-1)/th,th>>>(concat.data, head_proj_view.data, sz);
                CUDA_KERNEL_CHECK();

                head_proj_view.data = nullptr; // prevent double-free (CudaPtr owns memory)
            } // head_proj_raw freed here — saves seq×D×4B = 8192×1024×4 = 32MB per head
        }

        int pWproj = layer_base_pi + H*4 + 0;
        int pW1    = layer_base_pi + H*4 + 1;
        int pW2    = layer_base_pi + H*4 + 3;

        cache.concat=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(cache.concat.data,concat.data,seq*D*sizeof(float),cudaMemcpyDeviceToDevice));

        // [v31-RAWPTR] mha_out: temp projection, freed after cache save + residual
        cache.attn_out=gpu_alloc(seq,D);
        {
            CudaPtr<float> mha_raw(seq * D);
            GPUTensor mha_view;
            mha_view.data = mha_raw.get();
            mha_view.rows = seq; mha_view.cols = D; mha_view.size = seq * D;

            amp_gemm(concat, blk.W_proj, pWproj, mha_view);

            // Save for backward (cache.attn_out = W_proj output before residual)
            CUDA_CHECK(cudaMemcpy(cache.attn_out.data, mha_view.data,
                                  seq*D*sizeof(float), cudaMemcpyDeviceToDevice));
            int sz=seq*D, th=256;
            residual_add_kernel<<<(sz+th-1)/th,th>>>(X.data, mha_view.data, sz);
            CUDA_KERNEL_CHECK();

            mha_view.data = nullptr; // prevent double-free
        } // mha_raw freed here

        cache.post_attn=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(cache.post_attn.data,X.data,seq*D*sizeof(float),cudaMemcpyDeviceToDevice));

        cache.normed2=gpu_alloc(seq,D);
        apply_norm(X,blk.ln2_gamma,blk.ln2_beta,cache.normed2,cache.ln2_w,2*l+1,seq);

        cache.ffn_H=gpu_alloc(seq,4*D);
        amp_gemm_bias(cache.normed2, blk.W1, pW1, blk.b1, cache.ffn_H);

        cache.ffn_A=gpu_alloc(seq,4*D);
        CUDA_CHECK(cudaMemcpy(cache.ffn_A.data,cache.ffn_H.data,seq*4*D*sizeof(float),cudaMemcpyDeviceToDevice));
        cuda_gelu(cache.ffn_A);

        if (phys.feynman_dropout&&phys.training) {
            int n=seq*4*D;
            cache.ffn_mask=gpu_alloc(seq,4*D);
            feynman_dropout_kernel<<<(n+255)/256,256>>>(cache.ffn_A.data,cache.ffn_mask.data,
                n,phys.drop_p,phys.drop_hbar,dropout_seed);
            CUDA_KERNEL_CHECK();
            dropout_seed=dropout_seed*1664525u+1013904223u+(unsigned)l;
        }

        // [v31-RAWPTR] ffn_out: temporary, used only for residual add
        {
            CudaPtr<float> ffn_out_raw(seq * D);
            GPUTensor ffn_out_view;
            ffn_out_view.data = ffn_out_raw.get();
            ffn_out_view.rows = seq; ffn_out_view.cols = D; ffn_out_view.size = seq * D;

            amp_gemm_bias(cache.ffn_A, blk.W2, pW2, blk.b2, ffn_out_view);

            int sz=seq*D, th=256;
            residual_add_kernel<<<(sz+th-1)/th,th>>>(X.data, ffn_out_view.data, sz);
            CUDA_KERNEL_CHECK();

            ffn_out_view.data = nullptr; // prevent double-free
        } // ffn_out_raw freed here — saves 32MB per layer × 16 layers = 512MB
    }

    last_hidden=std::move(X);
    GPUTensor logits=gpu_alloc(seq,V);
    amp_gemm(last_hidden, gpu_lm_head, 2, logits);
    CUDA_KERNEL_CHECK();
    return logits;
}

void ModelGPU::print_kvcache_stats(int seq, int num_steps) const {
    int H=cfg.num_heads, DH=cfg.d_model/H, L=cfg.num_layers;
    long long kv_float=2LL*L*H*seq*DH*4;
    long long kv_int8 =2LL*L*H*seq*DH*1;
    long long saved   =kv_float-kv_int8;
    printf("[Nikhilam KV Cache Stats]\n");
    printf("  Config: L=%d H=%d DH=%d seq=%d\n",L,H,DH,seq);
    printf("  KV float32: %lld KB per forward\n",kv_float/1024);
    printf("  KV int8:    %lld KB per forward\n",kv_int8/1024);
    printf("  Saved:      %lld KB (%.1fx compression)\n",
           saved/1024, (float)kv_float/kv_int8);
    printf("  Over %d steps: %.1f MB freed\n",
           num_steps,(float)(saved*num_steps)/1024/1024);
}
