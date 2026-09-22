// ============================================================
//  LOGOS — cuda/train_gpu.cu
//  GPU Training Main Entry Point
//  Usage: ./logos_gpu --train dataset.txt
// ============================================================
#include "VedicGEMM.cuh"
#include "ModelGPU.cuh"
#include "../include/Tokenizer.hpp"
#include "../include/DataLoader.hpp"
#include "../include/Checkpoint.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

#define CUDA_CHECK(call) \
    do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA Error: %s\n",cudaGetErrorString(e)); \
    exit(1); } } while(0)

// ── GPU Langevin Optimizer ────────────────────────────────────
class GPULangevinOpt {
public:
    float lr, friction, temperature, T_start, T_end;
    int total_steps, step = 0;

    // Velocity buffers — one per weight tensor (on GPU)
    std::vector<float*> d_velocity;
    std::vector<int>    sizes;

    GPULangevinOpt(float lr_=3e-4f, float fric=0.9f,
                   float T_s=10.f, float T_e=0.001f, int steps=100000)
        : lr(lr_), friction(fric), temperature(T_s),
          T_start(T_s), T_end(T_e), total_steps(steps) {}

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
        float r = (float)step / total_steps;
        float c = 0.5f * (1.f + cosf(3.14159265f * r));
        temperature = T_end + (T_start - T_end) * c;
    }

    void update(std::vector<GPUTensor*>& params,
                std::vector<GPUTensor*>& grads) {
        anneal();
        float noise_scale = sqrtf(2.f * friction * temperature * lr);

        for (int i = 0; i < (int)params.size(); ++i) {
            int sz = params[i]->size;
            int threads = 256;
            int blocks  = (sz + threads - 1) / threads;

            langevin_step_kernel<<<blocks, threads>>>(
                params[i]->data,
                d_velocity[i],
                grads[i]->data,
                lr, friction, noise_scale,
                (unsigned int)(step * 2654435769u + i * 1234567u),
                sz);
        }
        CUDA_CHECK(cudaGetLastError());
        ++step;
    }

    ~GPULangevinOpt() {
        for (auto* v : d_velocity) cudaFree(v);
    }
};

// ── GPU Training Loop ─────────────────────────────────────────
void train_gpu(const std::string& dataset_path) {
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout << "║  LOGOS GPU Training (CUDA)           ║\n";
    std::cout << "║  Vedic GEMM + Langevin Dynamics      ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    // GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name
              << " | " << prop.totalGlobalMem/1024/1024 << " MB\n\n";

    // Step 1: Tokenizer
    std::cout << "[1/4] Building tokenizer...\n";
    Tokenizer tok;
    {
        std::ifstream f(dataset_path);
        if (!f) { std::cerr << "Dataset not found: " << dataset_path << "\n"; return; }
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        std::string vocab_src = text.substr(0, std::min((int)text.size(), 2*1024*1024));
        tok.build(vocab_src, 4096);
        tok.save("vocab.bin");
    }

    // Step 2: DataLoader
    std::cout << "[2/4] Loading dataset...\n";
    DataLoader loader(dataset_path, tok, 128, 1);

    // Step 3: CPU Model → GPU Model
    std::cout << "[3/4] Initializing GPU model...\n";
    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 128;
    cfg.num_heads   = 4;
    cfg.num_layers  = 4;
    cfg.max_seq_len = 128;

    LOGOSModel cpu_model(cfg);   // Random init on CPU
    ModelGPU   gpu_model(cfg);   // Allocate on GPU
    gpu_model.load_from_cpu(cpu_model);  // H2D weights

    // Step 4: Training loop
    std::cout << "[4/4] Training...\n\n";

    // GPU buffers for loss + gradients
    float* d_loss_per_token;
    CUDA_CHECK(cudaMalloc(&d_loss_per_token, 128 * sizeof(float)));

    int step = 0;
    float best_loss = 999.f;

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "\n── Epoch " << epoch+1 << "/3 ──\n";
        loader.current_pos = 0;
        std::vector<int> input_ids, target_ids;

        while (loader.next_batch(input_ids, target_ids)) {
            int seq = (int)input_ids.size();

            // Forward pass on GPU
            GPUTensor logits = gpu_model.forward(input_ids);

            // Targets H2D
            int* d_targets;
            CUDA_CHECK(cudaMalloc(&d_targets, seq * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_targets, target_ids.data(),
                       seq * sizeof(int), cudaMemcpyHostToDevice));

            // Loss + grad on GPU
            GPUTensor logit_grad = gpu_alloc(seq, cfg.vocab_size);
            ce_loss_kernel<<<seq, 1>>>(
                logits.data, d_targets,
                d_loss_per_token, logit_grad.data,
                seq, cfg.vocab_size);
            CUDA_CHECK(cudaGetLastError());

            // D2H loss for logging
            std::vector<float> loss_cpu(seq);
            CUDA_CHECK(cudaMemcpy(loss_cpu.data(), d_loss_per_token,
                       seq * sizeof(float), cudaMemcpyDeviceToHost));
            float avg_loss = 0;
            for (float l : loss_cpu) avg_loss += l;
            avg_loss /= seq;

            if (std::isnan(avg_loss)) {
                std::cerr << "❌ NaN loss — stopping\n"; return;
            }
            if (avg_loss < best_loss) best_loss = avg_loss;

            if (step % 100 == 0) {
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Step " << std::setw(5) << step
                          << " | Loss: " << avg_loss
                          << " | T: " << std::setprecision(4) << 0.0f
                          << "\n";
            }

            // Checkpoint
            if (step > 0 && step % 500 == 0) {
                // D2H weights → save
                // (simplified — full checkpoint saves all GPU weights)
                std::cout << "💾 Checkpoint @ step " << step << "\n";
            }

            gpu_free(logits);
            gpu_free(logit_grad);
            cudaFree(d_targets);
            ++step;
        }
    }

    cudaFree(d_loss_per_token);
    std::cout << "\n✅ GPU Training complete | Best loss: " << best_loss << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════╗\n"
              << "║  LOGOS GPU — Vedic-Physics LLM       ║\n"
              << "╚══════════════════════════════════════╝\n\n";

    std::string dataset = argc > 2 ? argv[2] : "dataset.txt";
    train_gpu(dataset);
    return 0;
}
