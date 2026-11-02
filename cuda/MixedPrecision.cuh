#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "CUDA Error at %s:%d — %s", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
            fprintf(stderr, "%s\n", _msg); \
            throw std::runtime_error(_msg); \
        } \
    } while(0)
#endif

#ifndef CUDA_KERNEL_CHECK
#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t _e = cudaGetLastError(); \
        if (_e != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "Kernel error at %s:%d — %s", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
            fprintf(stderr, "%s\n", _msg); \
            throw std::runtime_error(_msg); \
        } \
    } while(0)
#endif

static constexpr float LOSS_SCALE_INIT    = 65536.0f;   
static constexpr float LOSS_SCALE_MAX     = 65536.0f;   
static constexpr float LOSS_SCALE_MIN     = 1.0f;       
static constexpr int   LOSS_SCALE_WINDOW  = 2000;       
static constexpr float LOSS_SCALE_UP      = 2.0f;       
static constexpr float LOSS_SCALE_DOWN    = 0.5f;       

struct HalfTensor {
    __half* data = nullptr;
    int rows = 0, cols = 0, size = 0;

    HalfTensor() = default;
    HalfTensor(const HalfTensor&) = delete;
    HalfTensor& operator=(const HalfTensor&) = delete;

    HalfTensor(HalfTensor&& o) noexcept
        : data(o.data), rows(o.rows), cols(o.cols), size(o.size)
    { o.data = nullptr; o.rows = o.cols = o.size = 0; }

    HalfTensor& operator=(HalfTensor&& o) noexcept {
        if (this != &o) {
            if (data) cudaFree(data);
            data=o.data; rows=o.rows; cols=o.cols; size=o.size;
            o.data=nullptr; o.rows=o.cols=o.size=0;
        }
        return *this;
    }

    ~HalfTensor() { if (data) { cudaFree(data); data=nullptr; } }
    bool valid() const { return data != nullptr && size > 0; }
};

inline HalfTensor half_alloc(int rows, int cols) {
    HalfTensor t;
    t.rows=rows; t.cols=cols; t.size=rows*cols;
    CUDA_CHECK(cudaMalloc(&t.data, t.size * sizeof(__half)));
    CUDA_CHECK(cudaMemset(t.data, 0, t.size * sizeof(__half)));
    return t;
}


namespace {

__global__ void cast_fp32_to_fp16_kernel(
    const float* __restrict__ src,
    __half*      __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    v = fmaxf(fminf(v, 65504.0f), -65504.0f);
    dst[i] = __float2half(v);
}

__global__ void cast_fp16_to_fp32_kernel(
    const __half* __restrict__ src,
    float*        __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

} // anonymous namespace (cast kernels)

inline void cast_fp32_to_fp16(const float* src, __half* dst, int n,
                               cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    cast_fp32_to_fp16_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

inline void cast_fp16_to_fp32(const __half* src, float* dst, int n,
                               cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    cast_fp16_to_fp32_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

inline void cuda_cast_fp32_to_fp16(const float* src, __half* dst, int n,
                                    cudaStream_t stream = 0) {
    cast_fp32_to_fp16(src, dst, n, stream);
}

inline void cuda_cast_fp16_to_fp32(const __half* src, float* dst, int n,
                                    cudaStream_t stream = 0) {
    cast_fp16_to_fp32(src, dst, n, stream);
}


inline void scale_tensor(float* data, float scale, int n, cudaStream_t stream);


inline void amp_scale_grads(const std::vector<float*>& grad_ptrs,
                             const std::vector<int>& sizes,
                             float scale,
                             cudaStream_t stream = 0)
{
    for (int i = 0; i < (int)grad_ptrs.size(); ++i)
        scale_tensor(grad_ptrs[i], scale, sizes[i], stream);
}



namespace {

__global__ void scale_tensor_inplace_kernel(
    float* __restrict__ data,
    float scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] *= scale;
}


__global__ void check_inf_nan_kernel(
    const float* __restrict__ data,
    int*         __restrict__ flag,   // output: 1 if inf/nan found
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!isfinite(data[i])) atomicOr(flag, 1);
}

} 


inline void scale_tensor(float* data, float scale, int n,
                         cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    scale_tensor_inplace_kernel<<<blocks, 256, 0, stream>>>(data, scale, n);
}


inline bool grads_have_inf_nan(const std::vector<float*>& grad_ptrs,
                                const std::vector<int>& sizes)
{
    CudaPtr<int> d_flag(1);  

    for (int i = 0; i < (int)grad_ptrs.size(); ++i) {
        int n = sizes[i];
        int blocks = (n + 255) / 256;
        check_inf_nan_kernel<<<blocks, 256>>>(grad_ptrs[i], d_flag.get(), n);
       
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_flag.scalar() != 0;
}


#if LOGOS_USE_CUBLAS
#include <cublas_v2.h>

inline void h100_hgemm_fp32_acc(
    cublasHandle_t handle,
    const __half* A,   int M, int K,
    const __half* B,   int N,
    float*        C,
    float alpha = 1.0f, float beta = 0.0f)
{
   
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,      
        A, CUDA_R_16F, K,       
        &beta,
        C, CUDA_R_32F, N,      
        CUBLAS_COMPUTE_32F,     
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("h100_hgemm_fp32_acc failed: status=") +
            std::to_string((int)status));
    }
}

inline void h100_hgemm_half_out(
    cublasHandle_t handle,
    const __half* A,   int M, int K,
    const __half* B,   int N,
    __half*       C,
    float alpha = 1.0f, float beta = 0.0f)
{
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,       
        CUBLAS_COMPUTE_32F,     
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("h100_hgemm_half_out failed: status=") +
            std::to_string((int)status));
    }
}
#endif 


class ManualLossScaler {
public:
    float  scale;          
    int    steps_since_last_overflow;  
    int    total_overflows;            
    int    total_scale_ups;           
    int    total_scale_downs;          
    long long total_steps;             

    ManualLossScaler()
        : scale(LOSS_SCALE_INIT)
        , steps_since_last_overflow(0)
        , total_overflows(0)
        , total_scale_ups(0)
        , total_scale_downs(0)
        , total_steps(0)
    {}

    
    float get_scale() const { return scale; }

  
    void unscale(const std::vector<float*>& grad_ptrs,
                 const std::vector<int>&    sizes,
                 cudaStream_t stream = 0)
    {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < (int)grad_ptrs.size(); ++i)
            scale_tensor(grad_ptrs[i], inv_scale, sizes[i], stream);
    }

    
    bool update(const std::vector<float*>& grad_ptrs,
                const std::vector<int>&    sizes)
    {
        ++total_steps;
        bool overflow = grads_have_inf_nan(grad_ptrs, sizes);

        if (overflow) {
            
            scale = fmaxf(scale * LOSS_SCALE_DOWN, LOSS_SCALE_MIN);
            steps_since_last_overflow = 0;
            ++total_overflows;
            ++total_scale_downs;
            return false;  
        }

        
        ++steps_since_last_overflow;
        if (steps_since_last_overflow >= LOSS_SCALE_WINDOW) {
            
            scale = fminf(scale * LOSS_SCALE_UP, LOSS_SCALE_MAX);
            steps_since_last_overflow = 0;
            ++total_scale_ups;
        }
        return true; 
    }

    void print_status(long long step) const {
        printf("  [AMP] step=%lld scale=%.0f overflows=%d ups=%d downs=%d ok_window=%d/%d\n",
               (long long)step, scale,
               total_overflows, total_scale_ups, total_scale_downs,
               steps_since_last_overflow, LOSS_SCALE_WINDOW);
    }

    struct State {
        float scale;
        int   steps_since_last_overflow;
        int   total_overflows;
    };
    State get_state() const {
        return {scale, steps_since_last_overflow, total_overflows};
    }
    void set_state(const State& s) {
        scale                       = s.scale;
        steps_since_last_overflow   = s.steps_since_last_overflow;
        total_overflows             = s.total_overflows;
    }
};


namespace {
__global__ void layernorm_fp16_kernel(
    const __half* __restrict__ X,      // (seq × d) FP16 input
    const float*  __restrict__ gamma,  // (d,) FP32 scale — master param
    const float*  __restrict__ beta,   // (d,) FP32 bias  — master param
    __half*       __restrict__ Y,      // (seq × d) FP16 output
    int seq, int d, float eps)
{
    int row = blockIdx.x;
    if (row >= seq) return;
    const __half* x = X + row * d;
    __half*       y = Y + row * d;

    
    float sum = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        sum += __half2float(x[j]);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float smean_arr[8];
    if (threadIdx.x % 32 == 0) smean_arr[threadIdx.x / 32] = sum;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x < 8) tot = smean_arr[threadIdx.x];
    for (int off = 4; off > 0; off >>= 1) tot += __shfl_down_sync(0xffffffff, tot, off);
    __shared__ float smean;
    if (threadIdx.x == 0) smean = tot / d;
    __syncthreads();

    float var = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float diff = __half2float(x[j]) - smean;
        var += diff * diff;
    }
    for (int off = 16; off > 0; off >>= 1) var += __shfl_down_sync(0xffffffff, var, off);
    __shared__ float svar_arr[8];
    if (threadIdx.x % 32 == 0) svar_arr[threadIdx.x / 32] = var;
    __syncthreads();
    float vtot = 0.0f;
    if (threadIdx.x < 8) vtot = svar_arr[threadIdx.x];
    for (int off = 4; off > 0; off >>= 1) vtot += __shfl_down_sync(0xffffffff, vtot, off);
    __shared__ float sinv_std;
    if (threadIdx.x == 0) sinv_std = rsqrtf(vtot / d + eps);
    __syncthreads();

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float xf  = __half2float(x[j]);
        float yf  = gamma[j] * (xf - smean) * sinv_std + beta[j];
        yf = fmaxf(fminf(yf, 65504.0f), -65504.0f);
        y[j] = __float2half(yf);
    }
}
} 
namespace {
__global__ void embedding_lookup_fp16_kernel(
    const int*   __restrict__ token_ids,   // (seq,)
    const float* __restrict__ embedding,   // (vocab × d) FP32 master
    __half*      __restrict__ out,         // (seq × d) FP16 output
    int seq, int d, int vocab_size)
{
    int s  = blockIdx.x;
    int di = blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= seq || di >= d) return;

    int tok = token_ids[s];
    if (tok < 0 || tok >= vocab_size) tok = 0;  // OOB guard

    float v = embedding[tok * d + di];
    v = fmaxf(fminf(v, 65504.0f), -65504.0f);
    out[s * d + di] = __float2half(v);
}
} 
namespace {
__global__ void logits_fp16_to_fp32_kernel(
    const __half* __restrict__ logits_fp16,   // (seq × vocab) FP16
    float*        __restrict__ logits_fp32,   // (seq × vocab) FP32
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    logits_fp32[i] = __half2float(logits_fp16[i]);
}

__global__ void fp32_grad_to_fp16_scaled_kernel(
    const float* __restrict__ grad_fp32,     // (n,) FP32 gradient
    __half*      __restrict__ grad_fp16,     // (n,) FP16 scaled gradient
    float scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = grad_fp32[i] * scale;
    v = fmaxf(fminf(v, 65504.0f), -65504.0f);
    grad_fp16[i] = __float2half(v);
}
} 
struct AMPStatus {
    float  current_scale;
    int    overflow_this_step;
    long long total_steps;
    int    total_overflows;
    float  effective_lr;       
};


