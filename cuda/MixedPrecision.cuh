#pragma once
// ============================================================
//  LOGOS — MixedPrecision.cuh  (v23-AMP)
//  Mixed Precision Training: FP16 forward + FP32 master weights
//  Manual Loss Scaling for H100 SXM Tensor Core optimization
//
//  Architecture:
//  ┌─────────────────────────────────────────────────────┐
//  │  MASTER WEIGHTS (FP32) ← always stored in FP32     │
//  │        ↓  cast_fp32_to_fp16()                       │
//  │  FP16 WEIGHTS (shadow) ← used in forward pass       │
//  │        ↓  GEMM via cuBLAS HALF (Tensor Cores)       │
//  │  FP16 ACTIVATIONS      ← forward/backward in FP16   │
//  │        ↓  loss scaling (×scale_factor)               │
//  │  FP32 GRADIENTS        ← unscaled, accumulated      │
//  │        ↓  SHM optimizer (FP32)                       │
//  │  MASTER WEIGHTS (FP32) ← updated                    │
//  └─────────────────────────────────────────────────────┘
//
//  H100 SXM Tensor Core throughput:
//    FP32:  989 TFLOPS (dense)
//    FP16: 1979 TFLOPS (dense) — 2x faster
//    BF16: 1979 TFLOPS (dense) — 2x faster
//    FP8:  3958 TFLOPS (dense) — 4x faster (v24 future)
//
//  Why FP16 not BF16:
//    FP16: range=[6e-5, 65504], mantissa=10 bits → higher precision
//    BF16: range=[1e-38, 3.4e38], mantissa=7 bits → wider range but lossy
//    For LOGOS (Vedic GEMM + physics kernels): FP16 preferred
//    Loss scaling handles FP16 underflow → no need for BF16's wider range
//
//  Manual Loss Scaling Strategy:
//    Problem:  FP16 underflows at ~6e-5 (gradients become zero → no learning)
//    Solution: Scale loss UP by large factor → gradients amplified in FP16
//              Scale gradients DOWN before optimizer → correct FP32 gradients
//
//    Dynamic algorithm (same as PyTorch AMP):
//      1. Start: scale = LOSS_SCALE_INIT = 65536.0f
//      2. Every step: scaled_loss = loss × scale
//      3. Backward: scaled_grad = grad(scaled_loss)  [FP16, amplified]
//      4. Check: any scaled_grad is inf/nan?
//         YES → scale /= 2 (halve), skip optimizer step, try again
//         NO  → unscale: grad = scaled_grad / scale (FP32)
//              optimizer step with unscaled FP32 gradients
//              if no overflow for LOSS_SCALE_WINDOW steps: scale *= 2 (double)
//      5. Clamp: scale ∈ [LOSS_SCALE_MIN, LOSS_SCALE_MAX]
//
//  FP16 GPUTensor:
//    - Separate struct HalfTensor for __half* buffers
//    - RAII: cudaMalloc/cudaFree on __half
//    - Conversion: cast_fp32_to_fp16() / cast_fp16_to_fp32() kernels
//    - Used for: weights_fp16, activations (forward/backward in ModelGPU)
// ============================================================

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

// ── Loss scaling constants ────────────────────────────────────
static constexpr float LOSS_SCALE_INIT    = 65536.0f;   // 2^16 — start high
static constexpr float LOSS_SCALE_MAX     = 65536.0f;   // cap (beyond causes overflow)
static constexpr float LOSS_SCALE_MIN     = 1.0f;       // floor (below = no benefit)
static constexpr int   LOSS_SCALE_WINDOW  = 2000;       // steps before doubling scale
static constexpr float LOSS_SCALE_UP      = 2.0f;       // multiply when no overflow
static constexpr float LOSS_SCALE_DOWN    = 0.5f;       // halve on overflow detected

// ── FP16 Tensor (RAII, mirrors GPUTensor interface) ──────────
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

// ── Allocate FP16 tensor ──────────────────────────────────────
inline HalfTensor half_alloc(int rows, int cols) {
    HalfTensor t;
    t.rows=rows; t.cols=cols; t.size=rows*cols;
    CUDA_CHECK(cudaMalloc(&t.data, t.size * sizeof(__half)));
    CUDA_CHECK(cudaMemset(t.data, 0, t.size * sizeof(__half)));
    return t;
}

// ============================================================
//  CONVERSION KERNELS
// ============================================================

// FP32 → FP16 (cast, with clamp to FP16 range to prevent inf)
__global__ void cast_fp32_to_fp16_kernel(
    const float* __restrict__ src,
    __half*      __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    // Clamp to FP16 representable range: [-65504, 65504]
    // Values outside → ±inf in FP16 → corrupts GEMM
    v = fmaxf(fminf(v, 65504.0f), -65504.0f);
    dst[i] = __float2half(v);
}

// FP16 → FP32 (cast)
__global__ void cast_fp16_to_fp32_kernel(
    const __half* __restrict__ src,
    float*        __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

// ── Host wrappers for conversion ──────────────────────────────
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

// ── Aliases used in train_gpu.cu (cuda_ prefix convention) ───
// [v23-AMP] train_gpu.cu mein cuda_cast_fp32_to_fp16() call hota hai.
// Yeh wrapper ensure karta hai: FP32 master weights → FP16 shadow
// har optimizer step se pehle (forward pass ke liye).
inline void cuda_cast_fp32_to_fp16(const float* src, __half* dst, int n,
                                    cudaStream_t stream = 0) {
    cast_fp32_to_fp16(src, dst, n, stream);
}

inline void cuda_cast_fp16_to_fp32(const __half* src, float* dst, int n,
                                    cudaStream_t stream = 0) {
    cast_fp16_to_fp32(src, dst, n, stream);
}

// ── [BUG5-FIX] scale_tensor forward declaration ──────────────
// amp_scale_grads() (neeche) scale_tensor() use karta hai jo definition mein
// baad mein aata hai. Forward declare karo taaki compiler usse jaane.
// Note: default argument (stream=0) sirf DEFINITION mein hoga — yahan nahi.
// C++ rule: ek hi TU mein same default argument do baar = redefinition error.
inline void scale_tensor(float* data, float scale, int n, cudaStream_t stream);

// ── Bulk gradient scaling helper ──────────────────────────────
// [v23-AMP] Gradient amplify/unscale batch operation.
// Usage:
//   BEFORE optimizer: amp_scale_grads(ptrs, sizes, 1.0f/scale) — unscale
//   (grads were already scaled inside backward by loss_scaler.scale)
// Always call ManualLossScaler::update() BEFORE this to check overflow.
// If update() returns false (overflow), skip this call and the optimizer step.
inline void amp_scale_grads(const std::vector<float*>& grad_ptrs,
                             const std::vector<int>& sizes,
                             float scale,
                             cudaStream_t stream = 0)
{
    for (int i = 0; i < (int)grad_ptrs.size(); ++i)
        scale_tensor(grad_ptrs[i], scale, sizes[i], stream);
}

// ============================================================
//  LOSS SCALING KERNELS
// ============================================================

// Scale gradients in-place: grad *= scale_factor (FP32)
// Used BEFORE backward to amplify (scale up)
// Used AFTER backward to reduce (scale down = /scale)
__global__ void scale_tensor_inplace_kernel(
    float* __restrict__ data,
    float scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] *= scale;
}

// Check for inf/nan in gradient tensor — returns 1 if any found
// Used to detect FP16 overflow after scaled backward
__global__ void check_inf_nan_kernel(
    const float* __restrict__ data,
    int*         __restrict__ flag,   // output: 1 if inf/nan found
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!isfinite(data[i])) atomicOr(flag, 1);
}

// ── Host wrapper: scale FP32 tensor ─────────────────────────
inline void scale_tensor(float* data, float scale, int n,
                         cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    scale_tensor_inplace_kernel<<<blocks, 256, 0, stream>>>(data, scale, n);
}

// ── Host: check all grad tensors for inf/nan ─────────────────
// Returns true if any gradient has inf or nan
// [v25-BUG5-FIX] cudaDeviceSynchronize() sirf copy ke pehle (lazy sync) → ~0.5ms/step saved.
// [v26-LEAK-FIX] CudaPtr<int> use karo — d_flag kabhi leak nahi hoga exception par.
inline bool grads_have_inf_nan(const std::vector<float*>& grad_ptrs,
                                const std::vector<int>& sizes)
{
    // RAII: d_flag auto-freed on any exit path (normal, exception, early return)
    CudaPtr<int> d_flag(1);  // 1 int, zero-initialised

    for (int i = 0; i < (int)grad_ptrs.size(); ++i) {
        int n = sizes[i];
        int blocks = (n + 255) / 256;
        check_inf_nan_kernel<<<blocks, 256>>>(grad_ptrs[i], d_flag.get(), n);
        // No sync here — kernels queue asynchronously on default stream
    }
    // Single sync: wait for ALL check_inf_nan_kernel launches to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // scalar() does device->host copy of the single int flag
    return d_flag.scalar() != 0;
}

// ============================================================
//  H100 TENSOR CORE GEMM (FP16 → FP32 accumulate)
//  Uses cuBLAS cublasGemmEx with CUBLAS_COMPUTE_32F
//  Input:  A (__half, M×K),  B (__half, K×N)
//  Output: C (float,  M×N)   ← accumulated in FP32
//
//  H100 advantage: FP16 Tensor Core path = 2x throughput vs FP32
//  Accumulate in FP32: maintains numerical precision
// ============================================================
#if LOGOS_USE_CUBLAS
#include <cublas_v2.h>

inline void h100_hgemm_fp32_acc(
    cublasHandle_t handle,
    const __half* A,   int M, int K,
    const __half* B,   int N,
    float*        C,
    float alpha = 1.0f, float beta = 0.0f)
{
    // cublasGemmEx: FP16 in, FP32 accumulate, FP32 out
    // CUBLAS_COMPUTE_32F: accumulation in FP32 (prevents precision loss)
    // CUBLAS_GEMM_DEFAULT_TENSOR_OP: auto-select Tensor Core path
    //
    // Note: cuBLAS uses column-major, row-major C=A*B → col-major C^T=B^T*A^T
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,       // B^T in col-major = B in row-major
        A, CUDA_R_16F, K,       // A^T in col-major = A in row-major
        &beta,
        C, CUDA_R_32F, N,       // C output in FP32
        CUBLAS_COMPUTE_32F,     // accumulate in FP32 (not TF32, full precision)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("h100_hgemm_fp32_acc failed: status=") +
            std::to_string((int)status));
    }
}

// FP16→FP16 GEMM (for activations where full FP32 output not needed)
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
        C, CUDA_R_16F, N,       // output in FP16
        CUBLAS_COMPUTE_32F,     // still accumulate in FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("h100_hgemm_half_out failed: status=") +
            std::to_string((int)status));
    }
}
#endif // LOGOS_USE_CUBLAS

// ============================================================
//  MANUAL LOSS SCALER CLASS
//  Dynamic loss scaling with overflow detection
//  Thread-safe (single GPU, single stream assumed)
// ============================================================
class ManualLossScaler {
public:
    float  scale;          // current loss scale factor
    int    steps_since_last_overflow;  // counter for scale-up
    int    total_overflows;            // diagnostics
    int    total_scale_ups;            // diagnostics
    int    total_scale_downs;          // diagnostics
    long long total_steps;             // all optimizer steps attempted

    ManualLossScaler()
        : scale(LOSS_SCALE_INIT)
        , steps_since_last_overflow(0)
        , total_overflows(0)
        , total_scale_ups(0)
        , total_scale_downs(0)
        , total_steps(0)
    {}

    // ── Step 1: Get scale to multiply loss before backward ───
    float get_scale() const { return scale; }

    // ── Step 2: After backward, unscale gradients ────────────
    // Divides all gradients by scale (restores true gradient)
    void unscale(const std::vector<float*>& grad_ptrs,
                 const std::vector<int>&    sizes,
                 cudaStream_t stream = 0)
    {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < (int)grad_ptrs.size(); ++i)
            scale_tensor(grad_ptrs[i], inv_scale, sizes[i], stream);
    }

    // ── Step 3: Check overflow, update scale, return ok ──────
    // Returns true  → gradients are valid, proceed with optimizer step
    // Returns false → overflow detected, skip optimizer step this round
    bool update(const std::vector<float*>& grad_ptrs,
                const std::vector<int>&    sizes)
    {
        ++total_steps;
        bool overflow = grads_have_inf_nan(grad_ptrs, sizes);

        if (overflow) {
            // Scale too high → FP16 overflowed → halve scale
            scale = fmaxf(scale * LOSS_SCALE_DOWN, LOSS_SCALE_MIN);
            steps_since_last_overflow = 0;
            ++total_overflows;
            ++total_scale_downs;
            return false;  // skip this optimizer step
        }

        // No overflow → count toward scale-up window
        ++steps_since_last_overflow;
        if (steps_since_last_overflow >= LOSS_SCALE_WINDOW) {
            // Stable for WINDOW steps → try doubling scale
            scale = fminf(scale * LOSS_SCALE_UP, LOSS_SCALE_MAX);
            steps_since_last_overflow = 0;
            ++total_scale_ups;
        }
        return true;  // gradients valid, proceed
    }

    // ── Diagnostics ──────────────────────────────────────────
    void print_status(long long step) const {
        printf("  [AMP] step=%lld scale=%.0f overflows=%d ups=%d downs=%d ok_window=%d/%d\n",
               (long long)step, scale,
               total_overflows, total_scale_ups, total_scale_downs,
               steps_since_last_overflow, LOSS_SCALE_WINDOW);
    }

    // ── Serialize for checkpoint ──────────────────────────────
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

// ============================================================
//  FP16 LayerNorm Kernel
//  LOGOS uses Reynolds norm (blend of LN + BN).
//  FP16 inputs → FP32 intermediate → FP16 output
//  Critical: LN MUST accumulate mean/var in FP32 to avoid precision loss
//  (FP16 variance of d=1024 values → catastrophic cancellation)
// ============================================================
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

    // ── Mean (FP32 accumulation) ──────────────────────────────
    float sum = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        sum += __half2float(x[j]);
    // warp reduce
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

    // ── Variance (FP32 accumulation) ─────────────────────────
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

    // ── Normalize + scale (FP32 intermediate → FP16 output) ──
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float xf  = __half2float(x[j]);
        float yf  = gamma[j] * (xf - smean) * sinv_std + beta[j];
        // Clamp before cast to FP16 to avoid inf
        yf = fmaxf(fminf(yf, 65504.0f), -65504.0f);
        y[j] = __float2half(yf);
    }
}

// ============================================================
//  MIXED PRECISION EMBEDDING LOOKUP
//  Token IDs → FP32 embeddings → cast to FP16
//  Master embedding stays FP32; FP16 copy fed to transformer
// ============================================================
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

// ============================================================
//  SOFTMAX (FP16 input → FP32 output for loss computation)
//  Loss kernel needs FP32 logits for numerical stability
//  FP16 logits → FP32 softmax → FP32 CE/entropy → FP16 grad (scaled)
// ============================================================
__global__ void logits_fp16_to_fp32_kernel(
    const __half* __restrict__ logits_fp16,   // (seq × vocab) FP16
    float*        __restrict__ logits_fp32,   // (seq × vocab) FP32
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    logits_fp32[i] = __half2float(logits_fp16[i]);
}

// FP32 gradients → scale → FP16 (for backward through FP16 activations)
__global__ void fp32_grad_to_fp16_scaled_kernel(
    const float* __restrict__ grad_fp32,     // (n,) FP32 gradient
    __half*      __restrict__ grad_fp16,     // (n,) FP16 scaled gradient
    float scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = grad_fp32[i] * scale;
    // Clamp before FP16 cast
    v = fmaxf(fminf(v, 65504.0f), -65504.0f);
    grad_fp16[i] = __float2half(v);
}

// ============================================================
//  AMP STATUS STRUCT — for logging/monitoring
// ============================================================
struct AMPStatus {
    float  current_scale;
    int    overflow_this_step;
    long long total_steps;
    int    total_overflows;
    float  effective_lr;       // lr * scale (for monitoring)
};

// ============================================================
//  CMakeLists.txt pe add karne ka reminder:
//  - LOGOS_USE_CUBLAS=ON (required for h100_hgemm_fp32_acc)
//  - Add cuda_fp16.h path: automatically included via CUDA toolkit
//  - H100 arch: -arch=sm_90 (Hopper) OR cmake CUDA_ARCH=90
// ============================================================
