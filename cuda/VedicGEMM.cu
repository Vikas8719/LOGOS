

#include "VedicGEMM.cuh"
#include "MixedPrecision.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <climits>

#ifndef LOGOS_USE_CUBLAS
#define LOGOS_USE_CUBLAS 0
#endif

#ifndef LOGOS_CUBLAS_TF32
#define LOGOS_CUBLAS_TF32 0
#endif

#if LOGOS_USE_CUBLAS
#include <cublas_v2.h>
#endif

#define TILE_SIZE 16

// hash32: GPU-side fast integer hash (needed by SHM/Langevin kernels)
__device__ __forceinline__ unsigned int hash32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}


__global__ void vedic_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

__global__ void vedic_gemm_bias_kernel(
    const float* __restrict__ A, const float* __restrict__ W,
    const float* __restrict__ bias, float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileW[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int w_row = t * TILE_SIZE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileW[threadIdx.y][threadIdx.x] =
            (w_row < K && col < N) ? W[w_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileW[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc + bias[col];
}

__global__ void add_bias_kernel(float* __restrict__ C,
                                const float* __restrict__ bias,
                                int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = M * N;
    if (idx < size) C[idx] += bias[idx % N];
}


__global__ void boltzmann_softmax_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int seq_len, int vocab_size, float temperature)
{
    int row = blockIdx.x;
    if (row >= seq_len) return;
    const float* in_row  = input  + row * vocab_size;
    float*       out_row = output + row * vocab_size;

    float thread_max = -1e38f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x)
        thread_max = fmaxf(thread_max, in_row[i]);
    for (int off=16;off>0;off>>=1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff,thread_max,off));
    __shared__ float smax_arr[8];
    if (threadIdx.x%32==0) smax_arr[threadIdx.x/32]=thread_max;
    __syncthreads();
    float row_max=-1e38f;
    if (threadIdx.x<8) row_max=smax_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1)
        row_max=fmaxf(row_max,__shfl_down_sync(0xffffffff,row_max,off));
    __shared__ float smax;
    if (threadIdx.x==0) smax=row_max;
    __syncthreads();

    float thread_sum=0.0f;
    for (int i=threadIdx.x;i<vocab_size;i+=blockDim.x) {
        float e=expf((in_row[i]-smax)/temperature);
        out_row[i]=e; thread_sum+=e;
    }
    for (int off=16;off>0;off>>=1)
        thread_sum+=__shfl_down_sync(0xffffffff,thread_sum,off);
    __shared__ float ssum_arr[8];
    if (threadIdx.x%32==0) ssum_arr[threadIdx.x/32]=thread_sum;
    __syncthreads();
    float row_sum=0.0f;
    if (threadIdx.x<8) row_sum=ssum_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1)
        row_sum+=__shfl_down_sync(0xffffffff,row_sum,off);
    __shared__ float ssum;
    if (threadIdx.x==0) ssum=row_sum+1e-9f;
    __syncthreads();
    for (int i=threadIdx.x;i<vocab_size;i+=blockDim.x) out_row[i]/=ssum;
}


__global__ void layernorm_kernel(
    const float* __restrict__ X, const float* __restrict__ gamma,
    const float* __restrict__ beta, float* __restrict__ Y,
    int seq_len, int d_model, float eps)
{
    int row = blockIdx.x;
    if (row >= seq_len) return;
    const float* x = X + row*d_model;
    float*       y = Y + row*d_model;

    float ts=0.0f;
    for (int j=threadIdx.x;j<d_model;j+=blockDim.x) ts+=x[j];
    for (int off=16;off>0;off>>=1) ts+=__shfl_down_sync(0xffffffff,ts,off);
    __shared__ float smean_arr[8];
    if (threadIdx.x%32==0) smean_arr[threadIdx.x/32]=ts;
    __syncthreads();
    float total=0.0f;
    if (threadIdx.x<8) total=smean_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1) total+=__shfl_down_sync(0xffffffff,total,off);
    __shared__ float smean;
    if (threadIdx.x==0) smean=total/d_model;
    __syncthreads();

    float tv=0.0f;
    for (int j=threadIdx.x;j<d_model;j+=blockDim.x) {
        float diff=x[j]-smean; tv+=diff*diff;
    }
    for (int off=16;off>0;off>>=1) tv+=__shfl_down_sync(0xffffffff,tv,off);
    __shared__ float svar_arr[8];
    if (threadIdx.x%32==0) svar_arr[threadIdx.x/32]=tv;
    __syncthreads();
    float vt=0.0f;
    if (threadIdx.x<8) vt=svar_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1) vt+=__shfl_down_sync(0xffffffff,vt,off);
    __shared__ float sinv_std;
    if (threadIdx.x==0) sinv_std=rsqrtf(vt/d_model+eps);
    __syncthreads();
    for (int j=threadIdx.x;j<d_model;j+=blockDim.x)
        y[j]=gamma[j]*(x[j]-smean)*sinv_std+beta[j];
}


__global__ void gelu_kernel(float* data, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) {
        float x=data[idx];
        data[idx]=0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));
    }
}


__global__ void grad_norm_kernel(const float* __restrict__ grad,
                                  float* __restrict__ accum, int size) {
    __shared__ float sdata[256];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    float val=(idx<size)?grad[idx]*grad[idx]:0.0f;
    sdata[threadIdx.x]=val;
    __syncthreads();
    for (int s=blockDim.x/2;s>0;s>>=1) {
        if (threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x==0) atomicAdd(accum, sdata[0]);
}

__global__ void grad_scale_kernel(float* grad, float scale, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) grad[idx]*=scale;
}

__global__ void row_sum_kernel(
    const float* __restrict__ A, float* __restrict__ row_sums,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += A[row * K + k];
    
    for (int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    __shared__ float s[8];
    if (threadIdx.x%32==0) s[threadIdx.x/32]=acc;
    __syncthreads();
    float t=0.0f;
    if (threadIdx.x<8) t=s[threadIdx.x];
    for (int off=4;off>0;off>>=1) t+=__shfl_down_sync(0xffffffff,t,off);
    if (threadIdx.x==0) row_sums[row]=t;
}

__global__ void col_sum_kernel(
    const float* __restrict__ B, float* __restrict__ col_sums,
    int K, int N)
{
    int col = blockIdx.x;
    if (col >= N) return;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += B[k * N + col];
    for (int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    __shared__ float s[8];
    if (threadIdx.x%32==0) s[threadIdx.x/32]=acc;
    __syncthreads();
    float t=0.0f;
    if (threadIdx.x<8) t=s[threadIdx.x];
    for (int off=4;off>0;off>>=1) t+=__shfl_down_sync(0xffffffff,t,off);
    if (threadIdx.x==0) col_sums[col]=t;
}

__global__ void col_sum_A_kernel(
    const float* __restrict__ A, float* __restrict__ col_sums,
    int M, int K)
{
    int col = blockIdx.x;
    if (col >= K) return;
    float acc = 0.0f;
    for (int m = threadIdx.x; m < M; m += blockDim.x)
        acc += A[m * K + col];
    for (int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    __shared__ float s[8];
    if (threadIdx.x%32==0) s[threadIdx.x/32]=acc;
    __syncthreads();
    float t=0.0f;
    if (threadIdx.x<8) t=s[threadIdx.x];
    for (int off=4;off>0;off>>=1) t+=__shfl_down_sync(0xffffffff,t,off);
    if (threadIdx.x==0) col_sums[col]=t;
}


__global__ void array_sum_kernel(
    const float* __restrict__ arr, float* __restrict__ out, int n)
{
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (idx < n) ? arr[idx] : 0.0f;
    __syncthreads();
    for (int s=blockDim.x/2;s>0;s>>=1) {
        if (threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x==0) atomicAdd(out, sdata[0]);
}


__global__ void free_energy_loss_kernel(
    const float* __restrict__ logits,  // (seq × vocab)
    const int*   __restrict__ targets, // (seq,)
    float*       __restrict__ loss_out,// (seq,) — per-token F
    float*       __restrict__ grad_out,// (seq × vocab) — dF/dlogit
    float*       __restrict__ ce_buf,  // (seq,) — per-token CE (for monitoring)
    float*       __restrict__ S_buf,   // (seq,) — per-token entropy S
    int seq_len, int vocab_size,
    float temperature)
{
    int i = blockIdx.x;
    if (i >= seq_len) return;

    const float* logit_row = logits   + i * vocab_size;
    float*       grad_row  = grad_out + i * vocab_size;
    int target = targets[i];

    float thread_max = -1e38f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x)
        thread_max = fmaxf(thread_max, logit_row[v]);
    for (int off=16;off>0;off>>=1)
        thread_max=fmaxf(thread_max,__shfl_down_sync(0xffffffff,thread_max,off));
    __shared__ float smax_arr[8];
    if (threadIdx.x%32==0) smax_arr[threadIdx.x/32]=thread_max;
    __syncthreads();
    float row_max=-1e38f;
    if (threadIdx.x<8) row_max=smax_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1)
        row_max=fmaxf(row_max,__shfl_down_sync(0xffffffff,row_max,off));
    __shared__ float smax;
    if (threadIdx.x==0) smax=row_max;
    __syncthreads();

    float thread_sum=0.0f;
    for (int v=threadIdx.x;v<vocab_size;v+=blockDim.x) {
        float e=expf(logit_row[v]-smax);
        grad_row[v]=e;  // reuse grad buffer to store exp(logit)
        thread_sum+=e;
    }
    for (int off=16;off>0;off>>=1)
        thread_sum+=__shfl_down_sync(0xffffffff,thread_sum,off);
    __shared__ float ssum_arr[8];
    if (threadIdx.x%32==0) ssum_arr[threadIdx.x/32]=thread_sum;
    __syncthreads();
    float row_sum=0.0f;
    if (threadIdx.x<8) row_sum=ssum_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1)
        row_sum+=__shfl_down_sync(0xffffffff,row_sum,off);
    __shared__ float ssum;
    if (threadIdx.x==0) ssum=row_sum+1e-9f;
    __syncthreads();

    for (int v=threadIdx.x;v<vocab_size;v+=blockDim.x)
        grad_row[v] /= ssum;  // grad_row now = softmax probs p[v]

    __shared__ float sce;
    if (threadIdx.x == 0) {
        float log_p_target = (target >= 0 && target < vocab_size)
            ? logf(fmaxf(grad_row[target], 1e-9f))
            : 0.0f;
        sce = -log_p_target;
    }
    __syncthreads();
    float thread_S = 0.0f;
    for (int v=threadIdx.x;v<vocab_size;v+=blockDim.x) {
        float pv = grad_row[v];
        if (pv > 1e-12f) thread_S -= pv * logf(pv);
    }
    for (int off=16;off>0;off>>=1)
        thread_S+=__shfl_down_sync(0xffffffff,thread_S,off);
    __shared__ float sS_arr[8];
    if (threadIdx.x%32==0) sS_arr[threadIdx.x/32]=thread_S;
    __syncthreads();
    float row_S=0.0f;
    if (threadIdx.x<8) row_S=sS_arr[threadIdx.x];
    for (int off=4;off>0;off>>=1)
        row_S+=__shfl_down_sync(0xffffffff,row_S,off);
    __shared__ float sS;
    if (threadIdx.x==0) sS=row_S;
    __syncthreads();

    if (threadIdx.x == 0) {
        float F = sce - temperature * sS;
        if (!isfinite(F)) F = sce;  // fallback: NaN guard
        loss_out[i] = F;
        ce_buf[i]   = sce;
        S_buf[i]    = sS;
    }
    __syncthreads();

    
    float inv_seq = 1.0f / seq_len;
    for (int v=threadIdx.x;v<vocab_size;v+=blockDim.x) {
        float pv   = grad_row[v];
        float y_v  = (v == target) ? 1.0f : 0.0f;
        float ce_g = (pv - y_v) * inv_seq;
        float H_g  = temperature * pv * (logf(pv + 1e-9f) + sS) * inv_seq;
        grad_row[v] = (target >= 0 && target < vocab_size)
            ? (ce_g + H_g)
            : 0.0f;
    }
}


__global__ void shm_hybrid_kernel(
    float* __restrict__       weights,
    float* __restrict__       velocity,
    const float* __restrict__ gradients,
    float lr,
    float mom_decay,      
    float friction,       
    float alpha_H,        
    float alpha_L,        
    float noise_scale,    
    unsigned int seed,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned int rng = hash32((unsigned int)(idx * 2654435769u) ^ seed);
    rng ^= rng >> 16; rng *= 0x7feb352dU; rng ^= rng >> 15; rng *= 0x846ca68bU; rng ^= rng >> 16;

    float u       = (float)rng / 4294967296.0f;
    float thermal = noise_scale * (u * 2.0f - 1.0f) * 1.7320508f;

    float g = gradients[idx];
    float v = velocity[idx];
    float w = weights[idx];

    
    if (!isfinite(g)) g = 0.0f;

    
    float ham_kick      = -(alpha_H * 0.5f) * g;
    
    float lang_friction = (alpha_L > 1e-7f) ? (-(alpha_L * 0.5f) * friction * v) : 0.0f;
    float thermal_noise = (alpha_L > 1e-7f) ? thermal : 0.0f;

    float v_half = mom_decay * v + ham_kick + lang_friction + thermal_noise;

   
    float w_new = w + lr * v_half;

 
    if (!isfinite(w_new)) w_new = w;

    velocity[idx] = v_half;
    weights[idx]  = w_new;
}


__global__ void leapfrog_langevin_kernel(
    float* __restrict__       weights,
    float* __restrict__       velocity,
    const float* __restrict__ gradients,
    float lr,
    float friction,
    float noise_scale,
    unsigned int seed,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned int rng = hash32((unsigned int)(idx * 2654435769u) ^ seed);
    rng ^= rng >> 16; rng *= 0x7feb352dU; rng ^= rng >> 15; rng *= 0x846ca68bU; rng ^= rng >> 16;

    float u = (float)rng / 4294967296.0f;
    float thermal = noise_scale * (u * 2.0f - 1.0f) * 1.7320508f;

    float g  = gradients[idx];
    float v  = velocity[idx];
    float w  = weights[idx];

    
    if (!isfinite(g)) g = 0.0f;

    
    float v_half = friction * v - (lr * 0.5f) * g + thermal;

    
    float w_new = w + lr * v_half;

    
    if (!isfinite(w_new)) w_new = w;

    velocity[idx] = v_half;  // store v_{t+½} → becomes v_t for next step
    weights[idx]  = w_new;   // W_{t+1}
}



GPUTensor gpu_alloc(int rows, int cols) {
    GPUTensor t;
    t.rows=rows; t.cols=cols; t.size=rows*cols;
    CUDA_CHECK(cudaMalloc(&t.data, t.size*sizeof(float)));
    CUDA_CHECK(cudaMemset(t.data, 0, t.size*sizeof(float)));
    return t;
}

void gpu_free(GPUTensor& t) {
    if (t.data) { cudaFree(t.data); t.data=nullptr; t.size=0; }
}

void h2d(GPUTensor& dst, const float* src, int size) {
    CUDA_CHECK(cudaMemcpy(dst.data, src, size*sizeof(float), cudaMemcpyHostToDevice));
}

void d2h(float* dst, const GPUTensor& src, int size) {
    CUDA_CHECK(cudaMemcpy(dst, src.data, size*sizeof(float), cudaMemcpyDeviceToHost));
}

#if LOGOS_USE_CUBLAS
static void cublas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + operation +
                                 ": status=" + std::to_string(static_cast<int>(status)));
    }
}

static cublasHandle_t logos_cublas_handle() {
    struct HandleOwner {
        cublasHandle_t h = nullptr;
        HandleOwner() {
            cublas_check(cublasCreate(&h), "cublasCreate");
            cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        }
        ~HandleOwner() { if (h) { cublasDestroy(h); h = nullptr; } }
    };
    static HandleOwner owner;
    return owner.h;
}
#endif

bool cuda_vedic_gemm_uses_cublas() {
#if LOGOS_USE_CUBLAS
    return true;
#else
    return false;
#endif
}

void cuda_vedic_gemm(const GPUTensor& A, const GPUTensor& B, GPUTensor& C) {
    int M=A.rows, K=A.cols, N=B.cols;
#if LOGOS_USE_CUBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = logos_cublas_handle();
#if LOGOS_CUBLAS_TF32
    
    cublas_check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B.data, CUDA_R_32F, N,
                               A.data, CUDA_R_32F, K,
                               &beta, C.data, CUDA_R_32F, N,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx");
#else
    cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K, &alpha, B.data, N,
                              A.data, K, &beta, C.data, N),
                "cublasSgemm");
#endif
#else
    dim3 block(TILE_SIZE,TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);
    vedic_gemm_kernel<<<grid,block>>>(A.data,B.data,C.data,M,K,N);
    CUDA_KERNEL_CHECK();
#endif
}

void cuda_vedic_gemm_bias(const GPUTensor& A, const GPUTensor& W,
                           const GPUTensor& bias, GPUTensor& C) {
    int M=A.rows, N=W.cols;
#if LOGOS_USE_CUBLAS
    cuda_vedic_gemm(A, W, C);
    int size = M * N;
    add_bias_kernel<<<(size + 255) / 256, 256>>>(C.data, bias.data, M, N);
    CUDA_KERNEL_CHECK();
#else
    int K=A.cols;
    dim3 block(TILE_SIZE,TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);
    vedic_gemm_bias_kernel<<<grid,block>>>(A.data,W.data,bias.data,C.data,M,K,N);
    CUDA_KERNEL_CHECK();
#endif
}

void cuda_boltzmann_softmax(const GPUTensor& scores, GPUTensor& probs,
                             int seq_len, int vocab_size, float temperature) {
    boltzmann_softmax_kernel<<<seq_len,256>>>(
        scores.data,probs.data,seq_len,vocab_size,temperature);
    CUDA_KERNEL_CHECK();
}

void cuda_layernorm(const GPUTensor& X, const GPUTensor& gamma,
                    const GPUTensor& beta, GPUTensor& Y,
                    int seq_len, int d_model, float eps) {
    layernorm_kernel<<<seq_len,256>>>(
        X.data,gamma.data,beta.data,Y.data,seq_len,d_model,eps);
    CUDA_KERNEL_CHECK();
}

void cuda_gelu(GPUTensor& data) {
    int threads=256, blocks=(data.size+255)/256;
    gelu_kernel<<<blocks,threads>>>(data.data, data.size);
    CUDA_KERNEL_CHECK();
}

float cuda_clip_gradients(std::vector<GPUTensor*>& grads, float max_norm) {
    int threads = 256;
    CudaPtr<float> d_total_sq(1);

    for (auto* g : grads) {
        int sz = g->size, blocks = (sz + threads - 1) / threads;
        grad_norm_kernel<<<blocks, threads>>>(g->data, d_total_sq.get(), sz);
        CUDA_KERNEL_CHECK();
    }
    CUDA_CHECK(cudaStreamSynchronize(0));

    float total_norm_sq = d_total_sq.scalar();
    float total_norm    = sqrtf(total_norm_sq + 1e-12f);

    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (auto* g : grads) {
            int sz = g->size, blocks = (sz + threads - 1) / threads;
            grad_scale_kernel<<<blocks, threads>>>(g->data, scale, sz);
        }
        CUDA_KERNEL_CHECK();
    }
    return total_norm;
}


VedicVerifyResult cuda_vedic_verify(const GPUTensor& A,
                                     const GPUTensor& B,
                                     const GPUTensor& C,
                                     float tolerance)
{
    int M=A.rows, K=A.cols, N=B.cols;

    CudaPtr<float> d_col_sums_A(K);
    CudaPtr<float> d_row_sums_B(K);
    CudaPtr<float> d_sum_C(1);

    col_sum_A_kernel<<<K, 256>>>(A.data, d_col_sums_A.get(), M, K);
    CUDA_KERNEL_CHECK();

    row_sum_kernel<<<K, 256>>>(B.data, d_row_sums_B.get(), K, N);
    CUDA_KERNEL_CHECK();



   
    {
        int sz = C.size, blk = (sz + 255) / 256;
        array_sum_kernel<<<blk, 256>>>(C.data, d_sum_C.get(), sz);
        CUDA_KERNEL_CHECK();
    }

    CUDA_CHECK(cudaStreamSynchronize(0));

    std::vector<float> h_col_A(K), h_row_B(K);
    d_col_sums_A.to_host(h_col_A.data(), K);
    d_row_sums_B.to_host(h_row_B.data(), K);

    
    double checksum_vedic = 0.0;
    for (int k = 0; k < K; ++k)
        checksum_vedic += (double)h_col_A[k] * (double)h_row_B[k];

    float h_sum_C = d_sum_C.scalar();
    

    VedicVerifyResult res;
    res.checksum_C     = h_sum_C;
    res.checksum_vedic = (float)checksum_vedic;
    float denom        = fabsf(res.checksum_vedic) + 1e-6f;
    res.relative_error = fabsf(h_sum_C - res.checksum_vedic) / denom;
    res.pass           = res.relative_error < tolerance;
    return res;
}


void cuda_free_energy_loss(
    const float* d_logits,
    const int*   d_targets,
    float*       d_loss_buf,
    float*       d_grad_out,
    int seq_len, int vocab_size,
    float temperature,
    FreeEnergyResult& out_result)
{
    
    float *d_ce_buf = nullptr, *d_S_buf = nullptr;
    CudaPtr<float> d_ce_raii(seq_len);
    CudaPtr<float> d_S_raii(seq_len);
    d_ce_buf = d_ce_raii.get();
    d_S_buf  = d_S_raii.get();
    CUDA_CHECK(cudaMemset(d_ce_buf, 0, seq_len*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_S_buf,  0, seq_len*sizeof(float)));

    
    free_energy_loss_kernel<<<seq_len, 256>>>(
        d_logits, d_targets,
        d_loss_buf, d_grad_out,
        d_ce_buf, d_S_buf,
        seq_len, vocab_size, temperature);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(0));

    std::vector<float> h_F(seq_len), h_CE(seq_len), h_S(seq_len);
    CUDA_CHECK(cudaMemcpy(h_F.data(),  d_loss_buf, seq_len*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_CE.data(), d_ce_buf,   seq_len*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_S.data(),  d_S_buf,    seq_len*sizeof(float), cudaMemcpyDeviceToHost));

    float sum_F=0, sum_CE=0, sum_S=0; int cnt=0;
    for (int i=0; i<seq_len; ++i) {
        if (isfinite(h_F[i])) { sum_F+=h_F[i]; sum_CE+=h_CE[i]; sum_S+=h_S[i]; ++cnt; }
    }
    if (cnt > 0) {
        out_result.cross_entropy = sum_CE / cnt;
        out_result.entropy       = sum_S  / cnt;
        out_result.free_energy   = sum_F  / cnt;
        out_result.temperature   = temperature;
    } else {
        out_result = {0.f, 0.f, 0.f, temperature};
    }
}


#define FA_TILE     16     
#define FA_DH_MAX   64     // [v44-FIX] T4 pe DH=64 (d=1024/H=16) — 128 se ghata ke smem half kiya

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,     
    const float* __restrict__ K,     
    const float* __restrict__ V,     
    float*       __restrict__ Out,   
    float*       __restrict__ Aptr,  
    int seq, int DH,
    float scale,
    bool causal,
    bool shunyam, int window, int stride)
{
    
    int q_start = blockIdx.x * FA_TILE;
    int tid      = threadIdx.x;   

    
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + FA_TILE * FA_DH_MAX;
    float* sV = sK + FA_TILE * FA_DH_MAX;
    float* sO = sV + FA_TILE * FA_DH_MAX;
    float* sm = sO + FA_TILE * FA_DH_MAX;  
    float* sl = sm + FA_TILE;              

    int q_row   = q_start + tid;
    bool q_valid = (q_row < seq);

    
    if (q_valid) {
        const float* qptr = Q + q_row * DH;
        for (int d = 0; d < DH; ++d)
            sQ[tid * FA_DH_MAX + d] = qptr[d];
    } else {
        for (int d = 0; d < DH; ++d)
            sQ[tid * FA_DH_MAX + d] = 0.0f;
    }

    
    sm[tid] = -1e30f;
    sl[tid] = 0.0f;
    for (int d = 0; d < DH; ++d)
        sO[tid * FA_DH_MAX + d] = 0.0f;

    __syncthreads();

    int kv_tiles = (seq + FA_TILE - 1) / FA_TILE;

    for (int kv_t = 0; kv_t < kv_tiles; ++kv_t) {
        int kv_start = kv_t * FA_TILE;

        int kv_row = kv_start + tid;
        if (kv_row < seq) {
            const float* kptr = K + kv_row * DH;
            for (int d = 0; d < DH; ++d)
                sK[tid * FA_DH_MAX + d] = kptr[d];
        } else {
            for (int d = 0; d < DH; ++d)
                sK[tid * FA_DH_MAX + d] = 0.0f;
        }

        if (kv_row < seq) {
            const float* vptr = V + kv_row * DH;
            for (int d = 0; d < DH; ++d)
                sV[tid * FA_DH_MAX + d] = vptr[d];
        } else {
            for (int d = 0; d < DH; ++d)
                sV[tid * FA_DH_MAX + d] = 0.0f;
        }

        __syncthreads();

        if (q_valid) {
            
            float scores_local[FA_TILE];
            float tile_max = -1e30f;

            for (int kj = 0; kj < FA_TILE; ++kj) {
                int k_abs = kv_start + kj;
                if (k_abs >= seq) { scores_local[kj] = -1e30f; continue; }

                
                if (causal && k_abs > q_row) { scores_local[kj] = -1e30f; continue; }

                
                if (shunyam && seq > window) {
                    bool in_window  = (k_abs >= q_row - window) && (k_abs <= q_row);
                    bool is_global  = (k_abs % stride == 0);
                    if (!in_window && !is_global) { scores_local[kj] = -1e30f; continue; }
                }

                
                float dot = 0.0f;
                const float* qi = sQ + tid * FA_DH_MAX;
                const float* ki = sK + kj  * FA_DH_MAX;
                #pragma unroll 8
                for (int d = 0; d < DH; ++d) dot += qi[d] * ki[d];
                scores_local[kj] = dot * scale;
                tile_max = fmaxf(tile_max, scores_local[kj]);
            }

            
            float m_new     = fmaxf(sm[tid], tile_max);
            float exp_shift = expf(sm[tid] - m_new);  

            
            for (int d = 0; d < DH; ++d)
                sO[tid * FA_DH_MAX + d] *= exp_shift;
            float l_new = sl[tid] * exp_shift;

            
            for (int kj = 0; kj < FA_TILE; ++kj) {
                float s = scores_local[kj];
                if (s <= -1e29f) continue;  

                float e = expf(s - m_new);
                l_new += e;

                
                const float* vi = sV + kj * FA_DH_MAX;
                float*       oi = sO + tid * FA_DH_MAX;
                #pragma unroll 8
                for (int d = 0; d < DH; ++d)
                    oi[d] += e * vi[d];

                
                if (Aptr != nullptr) {
                    int k_abs = kv_start + kj;
                    if (k_abs < seq)
                        Aptr[q_row * seq + k_abs] = e;  
                }
            }

            sm[tid] = m_new;
            sl[tid] = l_new;
        }

        __syncthreads();
    }

    if (q_valid) {
        float inv_l = (sl[tid] > 1e-9f) ? (1.0f / sl[tid]) : 0.0f;
        float*       out_row = Out + q_row * DH;
        const float* oi      = sO  + tid * FA_DH_MAX;
        for (int d = 0; d < DH; ++d)
            out_row[d] = oi[d] * inv_l;

        
        if (Aptr != nullptr) {
            float* arow = Aptr + q_row * seq;
            for (int k_abs = 0; k_abs < seq; ++k_abs)
                arow[k_abs] *= inv_l;
        }
    }
}

void cuda_flash_attention_fwd(
    const float* Q, const float* K, const float* V,
    float* out,
    float* attn_ptr,      // nullable
    int seq, int DH,
    float scale,
    bool causal,
    bool shunyam, int window, int stride)
{
    if (DH > FA_DH_MAX) {
        throw std::runtime_error(
            "[v31-FLASH] DH=" + std::to_string(DH) +
            " exceeds FA_DH_MAX=" + std::to_string(FA_DH_MAX) +
            ". Increase FA_DH_MAX or reduce num_heads.");
    }
    // [v44-FIX] FA_DH_MAX=64 (T4 DH=64), smem exact calculate karo
    size_t smem_bytes = sizeof(float) * (4 * FA_TILE * FA_DH_MAX + 2 * FA_TILE);

    // T4 shared memory limit: 48KB. Check karo
    {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        size_t max_smem = prop.sharedMemPerBlockOptin;
        if (smem_bytes > max_smem) {
            // fallback: FA_DH_MAX ke bajaay exact DH use karo
            smem_bytes = sizeof(float) * (4 * FA_TILE * DH + 2 * FA_TILE);
            if (smem_bytes > max_smem) {
                throw std::runtime_error(
                    "[v31-FLASH] smem " + std::to_string(smem_bytes) +
                    " > GPU max " + std::to_string(max_smem) +
                    " bytes. DH=" + std::to_string(DH));
            }
        }
    }

    cudaError_t attr_err = cudaFuncSetAttribute(flash_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    if (attr_err != cudaSuccess) {
        // attribute set fail hone par bhi proceed karo — kernel already launched nahi
        fprintf(stderr, "[v31-FLASH] cudaFuncSetAttribute warning: %s\n",
                cudaGetErrorString(attr_err));
        cudaGetLastError(); // clear error
    }

    int q_tiles = (seq + FA_TILE - 1) / FA_TILE;
    flash_attention_kernel<<<q_tiles, FA_TILE, smem_bytes>>>(
        Q, K, V, out, attn_ptr,
        seq, DH, scale, causal,
        shunyam, window, stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "CUDA Error at %s:%d — %s", __FILE__, __LINE__,
            cudaGetErrorString(err));
        fprintf(stderr, "%s\n", msg);
        throw std::runtime_error(msg);
    }
}

std::string validate_model_config(int d_model, int num_heads, int num_layers,
                                   int vocab_size, int max_seq_len) {
    if (d_model <= 0)
        return "d_model must be > 0, got " + std::to_string(d_model);
    if (num_heads <= 0)
        return "num_heads must be > 0, got " + std::to_string(num_heads);
    if (d_model % num_heads != 0)
        return "d_model (" + std::to_string(d_model) +
               ") must be divisible by num_heads (" + std::to_string(num_heads) + ")";
    if (num_layers <= 0)
        return "num_layers must be > 0, got " + std::to_string(num_layers);
    if (vocab_size <= 0)
        return "vocab_size must be > 0, got " + std::to_string(vocab_size);
    if (max_seq_len <= 0)
        return "max_seq_len must be > 0, got " + std::to_string(max_seq_len);
    int head_dim = d_model / num_heads;
    if (head_dim < 4)
        return "head_dim (d_model/num_heads=" + std::to_string(head_dim) +
               ") too small, minimum 4";
    long long max_elem = (long long)max_seq_len * vocab_size;
    if (max_elem > (long long)INT_MAX)
        return "seq_len * vocab_size = " + std::to_string(max_elem) +
               " overflows int32";
    return "";
}


void cuda_gemm_fp16_fp32out(
    const __half* A, int M, int K,
    const __half* B, int N,
    float* C,
    float alpha, float beta)
{
#if LOGOS_USE_CUBLAS
    h100_hgemm_fp32_acc(logos_cublas_handle(), A, M, K, B, N, C, alpha, beta);
    CUDA_KERNEL_CHECK();
#else
    
    fprintf(stderr, "[v23-AMP] WARNING: FP16 GEMM called without cuBLAS — "
                    "falling back to FP32 conversion (slow)\n");
    
    float *d_A32, *d_B32;
    cudaMalloc(&d_A32, M * K * sizeof(float));
    cudaMalloc(&d_B32, K * N * sizeof(float));
    cast_fp16_to_fp32_kernel<<<(M*K+255)/256, 256>>>(A, d_A32, M*K);
    cast_fp16_to_fp32_kernel<<<(K*N+255)/256, 256>>>(B, d_B32, K*N);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    vedic_gemm_kernel<<<grid, block>>>(d_A32, d_B32, C, M, K, N);
    CUDA_KERNEL_CHECK();
    cudaFree(d_A32);
    cudaFree(d_B32);
#endif
}


void cuda_gemm_fp16_fp16out(
    const __half* A, int M, int K,
    const __half* B, int N,
    __half* C,
    float alpha, float beta)
{
#if LOGOS_USE_CUBLAS
    h100_hgemm_half_out(logos_cublas_handle(), A, M, K, B, N, C, alpha, beta);
    CUDA_KERNEL_CHECK();
#else
    
    (void)A; (void)M; (void)K; (void)B; (void)N; (void)C;
    (void)alpha; (void)beta;
    throw std::runtime_error("[v23-AMP] FP16→FP16 GEMM requires cuBLAS");
#endif
}


void cuda_layernorm_fp16(
    const __half* X, const float* gamma, const float* beta,
    __half* Y, int seq, int d, float eps)
{
    layernorm_fp16_kernel<<<seq, 256>>>(X, gamma, beta, Y, seq, d, eps);
    CUDA_KERNEL_CHECK();
}


void cuda_embedding_lookup_fp16(
    const int* token_ids, const float* embedding_fp32,
    __half* out_fp16, int seq, int d, int vocab_size)
{
    dim3 block(32);
    dim3 grid(seq, (d + 31) / 32);
    embedding_lookup_fp16_kernel<<<grid, block>>>(
        token_ids, embedding_fp32, out_fp16, seq, d, vocab_size);
    CUDA_KERNEL_CHECK();
}


void cuda_logits_fp16_to_fp32(const __half* logits_fp16, float* logits_fp32, int n) {
    int blocks = (n + 255) / 256;
    logits_fp16_to_fp32_kernel<<<blocks, 256>>>(logits_fp16, logits_fp32, n);
    CUDA_KERNEL_CHECK();
}


void cuda_scale_tensor(float* data, float scale, int n) {
    scale_tensor(data, scale, n);
    CUDA_KERNEL_CHECK();
}
