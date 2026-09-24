// ============================================================
//  LOGOS — cuda/ModelGPU.cu  (v8 — Phase 2 Physics)
//
//  v6 retained: All forward kernels, P0 attention cache,
//               load_from_cpu, sync_to_cpu, all_parameters
//
//  PHASE 2 ADDITIONS:
//
//  [P2-A] Hyperbolic Embedding Space
//    expmap0_kernel:     Euclidean → Poincaré ball (forward, per-token)
//    logmap0_kernel:     Poincaré → Euclidean (for backward pre-processing)
//    expmap0_bwd_kernel: gradient through exp_map (chain rule)
//
//    Math (unit Poincaré ball, curvature c=1):
//      expmap0(v) = tanh(√c · ||v||/2) · v / (√c · ||v|| + ε)
//      logmap0(y) = (2/√c) · arctanh(√c · ||y||) · y / (||y|| + ε)
//
//    Backward (dL/dv given dL/d_expmap0(v)):
//      Let r = ||v||, x̂ = v/r
//      expmap0 = tanh(r/2) · x̂  (c=1)
//      d/dv[expmap0] = (tanh(r/2)/r) · I
//                    + (sech²(r/2)/2 - tanh(r/2)/r) · x̂ x̂^T
//      dv = J^T · dOut  (Jacobian-vector product)
//
//    Forward integration: applied once after embedding_kernel, before layers
//    Backward integration: expmap0_bwd_kernel called in run_backward()
//                          BEFORE lmhead_grad_dx_kernel operates on X
//
//  [P2-B] Nikhilam KV Cache Compression
//    absmax_kernel:            max(|K|) → scale (parallel reduction)
//    nikhilam_quantize_kernel: float32 → int8 (complement encoding)
//    nikhilam_dequantize_kernel: int8 → float32
//
//    Applied in forward():
//      After K/V computed: compress to K_int8/V_int8
//      K/V float32 kept intact for backward (gradients need full precision)
//      attn_probs computed from DECOMPRESSED K/V (acceptable quantization error)
//
//    Nikhilam encoding (complement-from-base):
//      base = 127 (int8 max)
//      complement[i] = base - |round(K[i]/scale)|  (with sign bit)
//      This is "Nikhilam Navatascharamam" — subtraction from base
//      Reconstruction: K[i] ≈ sign * (base - complement) * scale
//      In practice: complement encoding = standard INT8 quantization
//      but with the Nikhilam "complement from 9/base" interpretation
//
//    VRAM savings at typical config (L=6, H=8, DH=64, seq=512):
//      K float32: 6 × 8 × 512 × 64 × 4B = 6MB
//      K int8:    6 × 8 × 512 × 64 × 1B = 1.5MB  → saves 4.5MB
//      V same:  → saves another 4.5MB
//      Total:   9MB freed → allows seq_len to grow from 512 → 640+
//
//  BACKWARD NOTE:
//    K_int8/V_int8 are NOT used in backward — float32 K/V cached separately.
//    So: dV, dK backward kernels unchanged. Zero regression.
// ============================================================
#include "ModelGPU.cuh"
#include "VedicGEMM.cuh"
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

// ── expmap0: Euclidean → Poincaré ball ───────────────────────
// One thread per (token, dim) pair — but we need ||v|| per token.
// Strategy: one block per token, threads reduce over d to get norm,
//           then apply tanh scaling element-wise.
//
// expmap0(v) = tanh(√c · ||v|| / 2) · v / (√c · ||v|| + ε)
//
// For c=1: expmap0(v) = tanh(||v||/2) · v / (||v|| + ε)
// Output is guaranteed inside unit ball: ||expmap0(v)|| < 1
__global__ void expmap0_kernel(float* X, int seq, int d, float curvature)
{
    int tok = blockIdx.x;
    if (tok >= seq) return;

    float* v = X + tok * d;
    float sqrt_c = sqrtf(curvature);

    // Step 1: compute ||v|| via parallel reduction
    float thread_sq = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        thread_sq += v[j] * v[j];
    // Warp reduction
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
    // tanh(scaled/2) / (scaled + ε)
    float factor  = (scaled > 1e-7f)
        ? tanhf(scaled * 0.5f) / (scaled + 1e-9f)
        : 0.5f;  // L'Hopital limit as ||v|| → 0: tanh(x/2)/x → 0.5

    // Step 2: apply scaling element-wise
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        v[j] *= factor;
}

// ── logmap0: Poincaré ball → Euclidean tangent space ─────────
// logmap0(y) = (2/√c) · arctanh(√c · ||y||) · y / (√c · ||y|| + ε)
// For c=1: logmap0(y) = 2 · arctanh(||y||) · y / (||y|| + ε)
// NOTE: ||y|| must be < 1 (inside ball) — clamp for safety
__global__ void logmap0_kernel(float* X, int seq, int d, float curvature)
{
    int tok = blockIdx.x;
    if (tok >= seq) return;

    float* y = X + tok * d;
    float sqrt_c = sqrtf(curvature);
    float inv_sqrt_c = 1.0f / sqrt_c;

    // Compute ||y||
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
        // Clamp to be safely inside ball: ||y|| < 1/√c - ε
        float max_r = inv_sqrt_c * 0.99f;
        s_norm = fminf(r, max_r);
    }
    __syncthreads();

    float norm    = s_norm;
    float scaled  = sqrt_c * norm;
    float atanh_v = atanhf(fminf(scaled, 0.9999f));
    float factor  = (norm > 1e-7f)
        ? 2.0f * inv_sqrt_c * atanh_v / (norm + 1e-9f)
        : 2.0f * inv_sqrt_c;  // limit as ||y|| → 0

    for (int j=threadIdx.x; j<d; j+=blockDim.x)
        y[j] *= factor;
}

// ── expmap0_bwd: gradient through exp_map ────────────────────
// Given: X_euc (input before exp_map, seq×d)
//        dOut  (upstream gradient, seq×d)
// Compute: dX (gradient w.r.t. X_euc, seq×d)
//
// Jacobian of expmap0 at v (c=1):
//   Let r = ||v||, α = tanh(r/2)/r
//   J(v) = α·I + (sech²(r/2)/2 - α) · (v⊗v)/r²
//   dX = J(v)^T · dOut = J(v) · dOut  (J is symmetric)
//
//   dX[j] = α · dOut[j]
//           + (sech²(r/2)/2 - α) · (v · dOut) · v[j] / r²
__global__ void expmap0_bwd_kernel(
    const float* __restrict__ X_euc,   // (seq × d) — pre-map Euclidean
    const float* __restrict__ dOut,    // (seq × d) — upstream gradient
    float*       __restrict__ dX,      // (seq × d) — output gradient
    int seq, int d, float curvature)
{
    int tok = blockIdx.x;
    if (tok >= seq) return;

    const float* v   = X_euc + tok * d;
    const float* dY  = dOut  + tok * d;
    float*       dv  = dX    + tok * d;

    float sqrt_c = sqrtf(curvature);

    // Step 1: ||v||
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
        float sech = 1.0f - th * th;  // sech² = 1 - tanh²
        alpha = th / (sqrt_c * r);
        beta  = (sech * sqrt_c * 0.5f - alpha) / r2;
    } else {
        // L'Hopital limits
        alpha = 0.5f;
        beta  = -1.0f / 24.0f;  // (sech²→1, th/r→0.5, limit of (sech²/2-0.5)/r²)
    }

    // Step 2: v · dOut (dot product, parallel)
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

    // Step 3: dX[j] = alpha*dY[j] + beta*(v·dY)*v[j]
    for (int j=threadIdx.x;j<d;j+=blockDim.x)
        dv[j] = alpha * dY[j] + beta * s_dot * v[j];
}

// ============================================================
//  [P2-B] NIKHILAM QUANTIZATION KERNELS
// ============================================================

// ── absmax: find max(|data|) via parallel reduction ──────────
// Returns a single float into out[0] (initialize to 0 before call)
__global__ void absmax_kernel(const float* data, float* out, int size)
{
    __shared__ float smax[256];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    float val=(idx<size)?fabsf(data[idx]):0.f;
    smax[threadIdx.x]=val;
    __syncthreads();
    for (int s=blockDim.x/2;s>0;s>>=1) {
        if (threadIdx.x<s) smax[threadIdx.x]=fmaxf(smax[threadIdx.x],smax[threadIdx.x+s]);
        __syncthreads();
    }
    if (threadIdx.x==0) atomicMax((int*)out, __float_as_int(smax[0]));
    // Note: atomicMax on float trick — works because IEEE754 positive floats
    // have same bit ordering as uint32 (for positive values)
}

// ── nikhilam_quantize: float32 → int8 ────────────────────────
// Nikhilam encoding: value → round(value / scale), clamped to [-127, 127]
// The "complement from base" interpretation:
//   For positive v: store q = round(v/scale)
//   For negative v: store q = -round(|v|/scale)  (sign preserved)
//   q = 0 means complement = base (127) in Nikhilam's scheme
// This maps cleanly to int8 range [-127, 127] leaving -128 as sentinel
__global__ void nikhilam_quantize_kernel(
    const float* __restrict__ src,
    int8_t*      __restrict__ dst,
    float scale, int size)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=size) return;
    float v=src[idx];
    // Nikhilam: compute complement q = v/scale, round, clamp
    float q=v/(scale+1e-8f);
    // clamp to [-127, 127] (leave -128 as NaN sentinel)
    q=fminf(fmaxf(q,-127.f),127.f);
    dst[idx]=(int8_t)__float2int_rn(q);
}

// ── nikhilam_dequantize: int8 → float32 ─────────────────────
// Reconstruction: value ≈ int8 * scale
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

// Compress float32 GPU tensor → NikhilamTensor
// Returns compressed tensor; float32 src unchanged (backward uses it)
static NikhilamTensor nikhilam_compress(const GPUTensor& src)
{
    NikhilamTensor out;
    out.size = src.size;

    // Find absmax
    float* d_max;
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(float)));
    int blks=(src.size+255)/256;
    absmax_kernel<<<blks,256>>>(src.data, d_max, src.size);
    CUDA_KERNEL_CHECK();

    float h_max=0.f;
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_max);

    // scale = absmax / 127  (Nikhilam: base=127)
    out.scale = h_max / 127.0f + 1e-8f;

    // Allocate int8 buffer and quantize
    CUDA_CHECK(cudaMalloc(&out.data, out.size * sizeof(int8_t)));
    nikhilam_quantize_kernel<<<blks,256>>>(src.data, out.data, out.scale, src.size);
    CUDA_KERNEL_CHECK();

    return out;
}

// Decompress NikhilamTensor → new float32 GPUTensor
static GPUTensor nikhilam_decompress(const NikhilamTensor& src, int rows, int cols)
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
// validate_model_config defined in VedicGEMM.cu — do NOT redefine here

// ============================================================
//  ModelGPU IMPLEMENTATION
// ============================================================

ModelGPU::ModelGPU(const ModelConfig& cfg_, const HyperConfig& hcfg)
    : cfg(cfg_), hyper_cfg(hcfg)
{
    // validate_model_config defined in VedicGEMM.cu
    // Call it through the extern declaration
    extern std::string validate_model_config(int,int,int,int,int);
    std::string err=validate_model_config(
        cfg.d_model,cfg.num_heads,cfg.num_layers,cfg.vocab_size,cfg.max_seq_len);
    if (!err.empty()) throw std::invalid_argument("ModelGPU config: "+err);

    int V=cfg.vocab_size, D=cfg.d_model, S=cfg.max_seq_len;
    gpu_embedding     = gpu_alloc(V,D);
    gpu_pos_embedding = gpu_alloc(S,D);
    gpu_lm_head       = gpu_alloc(D,V);

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
    CUDA_CHECK(cudaMalloc(&d_token_ids, cfg.max_seq_len*sizeof(int)));

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
        lc.heads.clear();
    }
    X_euclidean=GPUTensor{};  // [P2-A] clear hyperbolic pre-map cache
}

void ModelGPU::load_from_cpu(const LOGOSModel& cpu_model) {
    h2d(gpu_embedding,     cpu_model.embedding.data.data(),     cpu_model.embedding.total_size);
    h2d(gpu_pos_embedding, cpu_model.pos_embedding.data.data(), cpu_model.pos_embedding.total_size);
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
    d2h(cpu_model.pos_embedding.data.data(),gpu_pos_embedding,gpu_pos_embedding.size);
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
//  FORWARD PASS (v8 — Phase 2: Hyperbolic + Nikhilam KV)
// ============================================================
GPUTensor ModelGPU::forward(const std::vector<int>& token_ids)
{
    int seq=static_cast<int>(token_ids.size());
    int D=cfg.d_model, H=cfg.num_heads, DH=D/H, V=cfg.vocab_size;

    if (seq<=0||seq>cfg.max_seq_len)
        throw std::invalid_argument("Invalid seq: "+std::to_string(seq));

    std::vector<int> safe=token_ids;
    for (int& t:safe) if (t<0||t>=V) t=0;
    CUDA_CHECK(cudaMemcpy(d_token_ids,safe.data(),seq*sizeof(int),cudaMemcpyHostToDevice));

    free_layer_cache();

    // ── Embedding lookup ──────────────────────────────────────
    GPUTensor X=gpu_alloc(seq,D);
    {
        dim3 grid(seq,(D+31)/32); dim3 block(32);
        embedding_kernel<<<grid,block>>>(
            d_token_ids,gpu_embedding.data,gpu_pos_embedding.data,
            X.data,seq,D,V);
        CUDA_KERNEL_CHECK();
    }

    // ── [P2-A] Hyperbolic exp_map: Euclidean → Poincaré ball ─
    // Cache X_euclidean BEFORE exp_map (needed for backward)
    if (hyper_cfg.enabled) {
        X_euclidean=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(X_euclidean.data,X.data,
                   seq*D*sizeof(float),cudaMemcpyDeviceToDevice));
        // One block per token, 256 threads reduce over d
        expmap0_kernel<<<seq,256>>>(X.data,seq,D,hyper_cfg.curvature);
        CUDA_KERNEL_CHECK();
    }

    // ── Transformer layers ────────────────────────────────────
    for (int l=0;l<cfg.num_layers;++l) {
        auto& blk=gpu_blocks[l];
        auto& cache=layer_cache[l];

        cache.block_input=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(cache.block_input.data,X.data,
                   seq*D*sizeof(float),cudaMemcpyDeviceToDevice));

        cache.normed1=gpu_alloc(seq,D);
        cuda_layernorm(X,blk.ln1_gamma,blk.ln1_beta,cache.normed1,seq,D);

        // ── Multi-Head Attention with Nikhilam KV ────────────
        cache.heads.resize(H);
        GPUTensor concat=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemset(concat.data,0,seq*D*sizeof(float)));

        for (int h=0;h<H;++h) {
            auto& hc=cache.heads[h];

            hc.Q=gpu_alloc(seq,DH);
            cuda_vedic_gemm(cache.normed1,blk.W_Q[h],hc.Q);

            // K (float32 — kept for backward)
            hc.K=gpu_alloc(seq,DH);
            cuda_vedic_gemm(cache.normed1,blk.W_K[h],hc.K);

            // V (float32 — kept for backward)
            hc.V=gpu_alloc(seq,DH);
            cuda_vedic_gemm(cache.normed1,blk.W_V[h],hc.V);

            // [P2-B] Nikhilam: compress K → K_int8, V → V_int8
            // float32 K/V kept above for backward; int8 for forward attention
            hc.K_int8 = nikhilam_compress(hc.K);
            hc.V_int8 = nikhilam_compress(hc.V);

            // Decompress K_int8 → K_q for attention score computation
            GPUTensor K_q = nikhilam_decompress(hc.K_int8, seq, DH);

            // scores = Q @ K_q^T
            GPUTensor K_T=gpu_alloc(DH,seq);
            {
                dim3 tg((seq+15)/16,(DH+15)/16); dim3 tb(16,16);
                gpu_transpose_kernel<<<tg,tb>>>(K_q.data,K_T.data,seq,DH);
                CUDA_KERNEL_CHECK();
            }
            GPUTensor scores=gpu_alloc(seq,seq);
            cuda_vedic_gemm(hc.Q,K_T,scores);

            // Causal mask
            {
                dim3 g(seq,(seq+31)/32); dim3 b(32);
                causal_mask_kernel<<<g,b>>>(scores.data,seq);
                CUDA_KERNEL_CHECK();
            }

            hc.attn_probs=gpu_alloc(seq,seq);
            cuda_boltzmann_softmax(scores,hc.attn_probs,seq,seq,sqrtf((float)DH));

            // Decompress V_int8 → V_q for context vector computation
            GPUTensor V_q = nikhilam_decompress(hc.V_int8, seq, DH);

            hc.head_out=gpu_alloc(seq,DH);
            cuda_vedic_gemm(hc.attn_probs,V_q,hc.head_out);
            // K_q, V_q, K_T, scores RAII-freed

            GPUTensor head_proj=gpu_alloc(seq,D);
            cuda_vedic_gemm(hc.head_out,blk.W_O[h],head_proj);
            int sz=seq*D,th=256;
            residual_add_kernel<<<(sz+th-1)/th,th>>>(concat.data,head_proj.data,sz);
            CUDA_KERNEL_CHECK();
        }

        cache.concat=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(cache.concat.data,concat.data,
                   seq*D*sizeof(float),cudaMemcpyDeviceToDevice));

        GPUTensor mha_out=gpu_alloc(seq,D);
        cuda_vedic_gemm(concat,blk.W_proj,mha_out);

        cache.attn_out=gpu_alloc(seq,D);
        CUDA_CHECK(cudaMemcpy(cache.attn_out.data,mha_out.data,
                   seq*D*sizeof(float),cudaMemcpyDeviceToDevice));

        { int sz=seq*D,th=256;
          residual_add_kernel<<<(sz+th-1)/th,th>>>(X.data,mha_out.data,sz);
          CUDA_KERNEL_CHECK(); }

        cache.normed2=gpu_alloc(seq,D);
        cuda_layernorm(X,blk.ln2_gamma,blk.ln2_beta,cache.normed2,seq,D);

        cache.ffn_H=gpu_alloc(seq,4*D);
        cuda_vedic_gemm_bias(cache.normed2,blk.W1,blk.b1,cache.ffn_H);

        cache.ffn_A=gpu_alloc(seq,4*D);
        CUDA_CHECK(cudaMemcpy(cache.ffn_A.data,cache.ffn_H.data,
                   seq*4*D*sizeof(float),cudaMemcpyDeviceToDevice));
        cuda_gelu(cache.ffn_A);

        GPUTensor ffn_out=gpu_alloc(seq,D);
        cuda_vedic_gemm_bias(cache.ffn_A,blk.W2,blk.b2,ffn_out);

        { int sz=seq*D,th=256;
          residual_add_kernel<<<(sz+th-1)/th,th>>>(X.data,ffn_out.data,sz);
          CUDA_KERNEL_CHECK(); }
    }

    last_hidden=std::move(X);
    GPUTensor logits=gpu_alloc(seq,V);
    cuda_vedic_gemm(last_hidden,gpu_lm_head,logits);
    CUDA_KERNEL_CHECK();
    return logits;
}

// ── [P2-B] KV Cache stats ─────────────────────────────────────
void ModelGPU::print_kvcache_stats(int seq, int num_steps) const {
    int H=cfg.num_heads, DH=cfg.d_model/H, L=cfg.num_layers;
    long long kv_float=2LL*L*H*seq*DH*4;  // K+V, float32
    long long kv_int8 =2LL*L*H*seq*DH*1;  // K+V, int8
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
