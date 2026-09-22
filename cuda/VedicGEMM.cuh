#pragma once
// ============================================================
//  LOGOS — cuda/VedicGEMM.cuh
//  GPU Tensor struct + function declarations
// ============================================================
#include <cuda_runtime.h>

// GPU Tensor — data GPU memory mein rehta hai
struct GPUTensor {
    float* data = nullptr;   // cudaMalloc pointer
    int rows = 0;
    int cols = 0;
    int size = 0;            // rows * cols
};

// Memory management
GPUTensor gpu_alloc(int rows, int cols);
void      gpu_free(GPUTensor& t);

// Data transfer
void h2d(GPUTensor& dst, const float* src, int size);   // CPU → GPU
void d2h(float* dst, const GPUTensor& src, int size);   // GPU → CPU

// GPU Kernels (host-side wrappers)
void cuda_vedic_gemm(const GPUTensor& A, const GPUTensor& B, GPUTensor& C);

void cuda_vedic_gemm_bias(const GPUTensor& A, const GPUTensor& W,
                           const GPUTensor& bias, GPUTensor& C);

void cuda_boltzmann_softmax(const GPUTensor& scores, GPUTensor& probs,
                             int seq_len, int vocab_size, float temperature);

void cuda_layernorm(const GPUTensor& X, const GPUTensor& gamma,
                    const GPUTensor& beta, GPUTensor& Y,
                    int seq_len, int d_model, float eps = 1e-5f);

void cuda_gelu(GPUTensor& data);
