#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>   // [BUG1-FIX] __half type ke liye — VedicGEMM line 218-224 undefined
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cmath>

// ============================================================
//  CudaPtr<T> — RAII smart pointer for raw GPU memory
//  Replaces bare cudaMalloc/cudaFree patterns that leak on exceptions.
//
//  Usage:
//    CudaPtr<float>  d_buf(n);         // alloc n floats, zero-init
//    CudaPtr<int8_t> d_i8(n, false);   // alloc without zeroing
//    float* raw = d_buf.get();          // raw ptr for kernel calls
//    T val = d_scalar.scalar();         // single value device->host
//    // auto freed at scope exit — even on exception / early return
//
//  Rules:
//    - Non-copyable (like unique_ptr)
//    - Movable
//    - .get() returns raw device ptr
//    - Destructor calls cudaFree — never throws
// ============================================================
template<typename T>
struct CudaPtr {
    T*  ptr  = nullptr;
    int size = 0;

    CudaPtr() = default;

    explicit CudaPtr(int n, bool zero = true) : size(n) {
        if (n <= 0) { ptr = nullptr; return; }
        cudaError_t e = cudaMalloc(&ptr, (size_t)n * sizeof(T));
        if (e != cudaSuccess) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "CudaPtr cudaMalloc(%d x %zu B) failed: %s",
                n, sizeof(T), cudaGetErrorString(e));
            throw std::runtime_error(msg);
        }
        if (zero) {
            e = cudaMemset(ptr, 0, (size_t)n * sizeof(T));
            if (e != cudaSuccess) {
                cudaFree(ptr); ptr = nullptr;
                char msg2[256];
                snprintf(msg2, sizeof(msg2),
                    "CudaPtr cudaMemset failed: %s", cudaGetErrorString(e));
                throw std::runtime_error(msg2);
            }
        }
    }

    CudaPtr(const CudaPtr&)            = delete;
    CudaPtr& operator=(const CudaPtr&) = delete;

    CudaPtr(CudaPtr&& o) noexcept : ptr(o.ptr), size(o.size)
    { o.ptr = nullptr; o.size = 0; }

    CudaPtr& operator=(CudaPtr&& o) noexcept {
        if (this != &o) {
            release_internal();
            ptr = o.ptr; size = o.size;
            o.ptr = nullptr; o.size = 0;
        }
        return *this;
    }

    ~CudaPtr() { release_internal(); }

    void memset_zero() {
        if (ptr && size > 0) cudaMemset(ptr, 0, (size_t)size * sizeof(T));
    }

    T*       get()       { return ptr; }
    const T* get() const { return ptr; }

    void from_host(const T* src, int n) {
        cudaMemcpy(ptr, src, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
    }
    void to_host(T* dst, int n) const {
        cudaMemcpy(dst, ptr, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost);
    }
    // Single scalar device->host (for reduction outputs)
    T scalar() const {
        T val{};
        cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost);
        return val;
    }

    bool empty() const { return ptr == nullptr || size == 0; }

    // [v27-MEM2-FIX] release(): ownership transfer ke liye (unique_ptr::release() jaisa).
    // Caller raw ptr ka ownership leta hai — CudaPtr nullptr ho jaata hai, destructor free nahi karega.
    // Use case: CudaPtr<int8_t> mein alloc karo, quantize karo, phir NikhilamTensor ko do.
    // Agar quantize fail ho to CudaPtr destructor cleanup karta hai; success pe caller owns it.
    T* release() noexcept {
        T* p = ptr;
        ptr  = nullptr;
        size = 0;
        return p;
    }

private:
    void release_internal() noexcept {
        if (ptr) { cudaFree(ptr); ptr = nullptr; size = 0; }
    }
};

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

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t _e = cudaGetLastError(); \
        if (_e != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "Kernel launch error at %s:%d — %s", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
            fprintf(stderr, "%s\n", _msg); \
            throw std::runtime_error(_msg); \
        } \
    } while(0)

// RAII GPU float32 tensor
struct GPUTensor {
    float* data = nullptr;
    int rows = 0, cols = 0, size = 0;

    GPUTensor() = default;
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;

    GPUTensor(GPUTensor&& o) noexcept
        : data(o.data), rows(o.rows), cols(o.cols), size(o.size)
    { o.data = nullptr; o.rows = o.cols = o.size = 0; }

    GPUTensor& operator=(GPUTensor&& o) noexcept {
        if (this != &o) {
            if (data) cudaFree(data);
            data=o.data; rows=o.rows; cols=o.cols; size=o.size;
            o.data=nullptr; o.rows=o.cols=o.size=0;
        }
        return *this;
    }

    ~GPUTensor() { if (data) { cudaFree(data); data=nullptr; } }
    bool valid() const { return data != nullptr && size > 0; }
};

// Vedic sutram verify result: sum(C) vs dot(col_sums_A, row_sums_B)
struct VedicVerifyResult {
    float checksum_C;
    float checksum_vedic;
    float relative_error;
    bool  pass;
};

// Free energy loss result per batch step
struct FreeEnergyResult {
    float cross_entropy;
    float entropy;
    float free_energy;
    float temperature;
};

GPUTensor gpu_alloc(int rows, int cols);
void      gpu_free(GPUTensor& t);
void      h2d(GPUTensor& dst, const float* src, int size);
void      d2h(float* dst, const GPUTensor& src, int size);

void  cuda_vedic_gemm(const GPUTensor& A, const GPUTensor& B, GPUTensor& C);
bool  cuda_vedic_gemm_uses_cublas();
void  cuda_vedic_gemm_bias(const GPUTensor& A, const GPUTensor& W, const GPUTensor& bias, GPUTensor& C);
void  cuda_boltzmann_softmax(const GPUTensor& scores, GPUTensor& probs, int seq_len, int vocab_size, float temperature);
void  cuda_layernorm(const GPUTensor& X, const GPUTensor& gamma, const GPUTensor& beta, GPUTensor& Y, int seq_len, int d_model, float eps = 1e-5f);
void  cuda_gelu(GPUTensor& data);
float cuda_clip_gradients(std::vector<GPUTensor*>& grads, float max_norm);

// add_bias_kernel defined in VedicGEMM.cu — declared here for cross-file launch in ModelGPU.cu
extern __global__ void add_bias_kernel(float* __restrict__ C, const float* __restrict__ bias, int M, int N);

VedicVerifyResult cuda_vedic_verify(const GPUTensor& A, const GPUTensor& B, const GPUTensor& C, float tolerance = 0.01f);

void cuda_free_energy_loss(
    const float* d_logits,
    const int*   d_targets,
    float*       d_loss_buf,
    float*       d_grad_out,
    int seq_len, int vocab_size,
    float temperature,
    FreeEnergyResult& out_result);

__global__ void leapfrog_langevin_kernel(
    float* __restrict__ weights, float* __restrict__ velocity,
    const float* __restrict__ gradients,
    float lr, float friction, float noise_scale, unsigned int seed, int size);

__global__ void shm_hybrid_kernel(
    float* __restrict__ weights, float* __restrict__ velocity,
    const float* __restrict__ gradients,
    float lr, float mom_decay, float friction,
    float alpha_H, float alpha_L, float noise_scale,
    unsigned int seed, int size);

std::string validate_model_config(int d_model, int num_heads, int num_layers, int vocab_size, int max_seq_len);

// FP16/FP32 conversion and mixed precision GEMM (H100 Tensor Core)
// [BUG4-FIX] cuda_cast_fp32_to_fp16 / cuda_cast_fp16_to_fp32 declarations REMOVED —
// MixedPrecision.cuh mein inline definitions hain (cuda_ prefix wrappers).
// Duplicate declaration → ambiguous call error at link time.
// train_gpu.cu #includes MixedPrecision.cuh AFTER VedicGEMM.cuh, isliye definitions
// wahan se milti hain — yahan redeclaration ki zarurat nahi.
void cuda_gemm_fp16_fp32out(const __half* A, int M, int K, const __half* B, int N, float* C, float alpha=1.f, float beta=0.f);
void cuda_gemm_fp16_fp16out(const __half* A, int M, int K, const __half* B, int N, __half* C, float alpha=1.f, float beta=0.f);
void cuda_layernorm_fp16(const __half* X, const float* gamma, const float* beta, __half* Y, int seq, int d, float eps=1e-5f);
void cuda_embedding_lookup_fp16(const int* token_ids, const float* embedding_fp32, __half* out_fp16, int seq, int d, int vocab_size);
void cuda_logits_fp16_to_fp32(const __half* logits_fp16, float* logits_fp32, int n);
void cuda_scale_tensor(float* data, float scale, int n);
