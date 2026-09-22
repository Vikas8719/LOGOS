#pragma once
// ============================================================
//  LOGOS — cuda/ModelGPU.cuh
//  GPU Model declarations
// ============================================================
#include "VedicGEMM.cuh"
#include "../include/Model.hpp"
#include <vector>

// GPU version of one Transformer block's weights
struct GPUBlock {
    std::vector<GPUTensor> W_Q, W_K, W_V, W_O;  // per head
    GPUTensor W_proj;
    GPUTensor W1, b1, W2, b2;
    GPUTensor ln1_gamma, ln1_beta;
    GPUTensor ln2_gamma, ln2_beta;
};

__global__ void ce_loss_kernel(const float* logits, const int* targets,
                               float* loss_out, float* grad_out,
                               int seq_len, int vocab_size);
__global__ void langevin_step_kernel(float* weights, float* velocity,
                                     const float* gradients, float lr,
                                     float friction, float noise_scale,
                                     unsigned int seed, int size);

class ModelGPU {
public:
    ModelConfig cfg;

    // Top-level weights on GPU
    GPUTensor gpu_embedding;
    GPUTensor gpu_pos_embedding;
    GPUTensor gpu_lm_head;
    std::vector<GPUBlock> gpu_blocks;

    int* d_token_ids = nullptr;   // GPU buffer for token IDs
    GPUTensor last_hidden;

    ModelGPU(const ModelConfig& cfg);
    ~ModelGPU();

    // Load weights from CPU model → GPU
    void load_from_cpu(const LOGOSModel& cpu_model);

    // Forward pass (returns logits on GPU)
    GPUTensor forward(const std::vector<int>& token_ids);

    std::vector<GPUTensor*> all_parameters();
    std::vector<GPUTensor*> alloc_grad_buffers() const;
    void sync_to_cpu(LOGOSModel& cpu_model) const;
};
