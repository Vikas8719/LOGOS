// ============================================================
//  LOGOS — cuda/train_gpu.cu
//  GPU Training — Langevin Dynamics Optimizer
//  Dataset: 500MB+ TinyStories
//  Model:   d=256, L=6, H=8 — proper size for real learning
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
#include <algorithm>

#define CUDA_CHECK(call) \
    do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA Error %s:%d: %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e)); exit(1); } } while(0)

// ── GPU Langevin Optimizer ────────────────────────────────────
// dW = -γ·v  - lr·∇L + √(2γkT)·η
// CRITICAL: T_start=0.05 (low — prevents noise explosion)
class GPULangevinOpt {
public:
    float lr, friction, temperature, T_start, T_end;
    int   total_steps, step = 0;

    std::vector<float*> d_velocity;
    std::vector<int>    sizes;

    GPULangevinOpt(float lr_   = 3e-4f,
                   float fric  = 0.9f,
                   float T_s   = 0.05f,   // LOW — no explosion
                   float T_e   = 1e-6f,
                   int   steps = 500000)
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
        float r = std::min(1.0f, (float)step / (float)total_steps);
        float c = 0.5f * (1.f + cosf(3.14159265f * r));
        temperature = T_end + (T_start - T_end) * c;
    }

    void update(std::vector<GPUTensor*>& params,
                std::vector<GPUTensor*>& grads)
    {
        anneal();
        // noise_scale * 0.01 extra dampening
        float noise_scale = sqrtf(2.f * friction * temperature * lr) * 0.01f;

        for (int i = 0; i < (int)params.size(); ++i) {
            int sz      = params[i]->size;
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

    void log() const {
        printf("LangevinOpt | step=%d T=%.2e lr=%.2e\n",
               step, temperature, lr);
    }

    ~GPULangevinOpt() {
        for (auto* v : d_velocity) cudaFree(v);
    }
};

// ── GPU backward helper: dW = X^T @ dY ───────────────────────
// Simple kernel for lm_head gradient
__global__ void gemm_backward_dw_kernel(
    const float* __restrict__ X,    // (seq, d_model)
    const float* __restrict__ dY,   // (seq, vocab)
    float*       __restrict__ dW,   // (d_model, vocab) — accumulated
    int seq, int d_model, int vocab)
{
    // Each block handles one (d, v) element of dW
    int d = blockIdx.x;
    int v = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= d_model || v >= vocab) return;

    float acc = 0.0f;
    for (int s = 0; s < seq; ++s)
        acc += X[s * d_model + d] * dY[s * vocab + v];
    dW[d * vocab + v] += acc;
}

// dX = dY @ W^T
__global__ void gemm_backward_dx_kernel(
    const float* __restrict__ dY,   // (seq, vocab)
    const float* __restrict__ W,    // (d_model, vocab)
    float*       __restrict__ dX,   // (seq, d_model) — accumulated
    int seq, int d_model, int vocab)
{
    int s = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= seq || d >= d_model) return;

    float acc = 0.0f;
    for (int v = 0; v < vocab; ++v)
        acc += dY[s * vocab + v] * W[d * vocab + v];
    dX[s * d_model + d] += acc;
}

// Embedding backward: grad_emb[tok_id] += dX[i]
__global__ void embedding_backward_kernel(
    const int*   __restrict__ token_ids,
    const float* __restrict__ dX,       // (seq, d_model)
    float*       __restrict__ d_emb,    // (vocab, d_model) grad
    float*       __restrict__ d_pos,    // (seq, d_model) grad
    int seq, int d_model)
{
    int s = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= seq || d >= d_model) return;

    int tok = token_ids[s];
    float g = dX[s * d_model + d];
    atomicAdd(&d_emb[tok * d_model + d], g);
    d_pos[s * d_model + d] += g;
}

// ── GPU Training Loop ─────────────────────────────────────────
void train_gpu(const std::string& dataset_path) {
    printf("\n╔══════════════════════════════════════╗\n");
    printf("║  LOGOS GPU Training (CUDA)           ║\n");
    printf("║  Langevin Dynamics | Vedic GEMM      ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    // GPU info
    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("GPU    : %s\n", prop.name);
    printf("VRAM   : %zu MB\n", prop.totalGlobalMem/1024/1024);
    printf("SMs    : %d\n\n", prop.multiProcessorCount);

    // ── Step 1: Tokenizer ─────────────────────────────────────
    printf("[1/5] Building tokenizer...\n");
    Tokenizer tok;
    size_t dataset_size = 0;
    {
        std::ifstream f(dataset_path);
        if (!f) { fprintf(stderr,"❌ Dataset not found: %s\n",
                          dataset_path.c_str()); return; }
        // Read up to 50MB for vocab building (fast)
        std::string vocab_text;
        vocab_text.reserve(50*1024*1024);
        char buf[65536];
        while (f.read(buf, sizeof(buf)) || f.gcount()) {
            vocab_text.append(buf, f.gcount());
            if (vocab_text.size() >= 50*1024*1024) break;
        }
        f.seekg(0, std::ios::end);
        dataset_size = f.tellg();

        // Vocab size: larger dataset → larger vocab
        int vocab_sz = dataset_size > 200*1024*1024 ? 8192 :
                       dataset_size > 50*1024*1024  ? 4096 : 2048;
        tok.build(vocab_text, vocab_sz);
        tok.save("vocab.bin");
    }
    printf("Tokenizer : vocab=%d\n", tok.vocab_size);
    printf("Dataset   : %zu MB\n\n", dataset_size/1024/1024);

    // ── Step 2: Model Config ──────────────────────────────────
    printf("[2/5] Setting up model...\n");
    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;

    // Scale model to dataset size
    if (dataset_size > 200*1024*1024) {
        // Large dataset (500MB+): proper model
        cfg.d_model     = 256;
        cfg.num_heads   = 8;
        cfg.num_layers  = 6;
        cfg.max_seq_len = 256;
    } else if (dataset_size > 50*1024*1024) {
        // Medium (50-200MB)
        cfg.d_model     = 128;
        cfg.num_heads   = 4;
        cfg.num_layers  = 4;
        cfg.max_seq_len = 128;
    } else {
        // Small (<50MB): compact model
        cfg.d_model     = 64;
        cfg.num_heads   = 4;
        cfg.num_layers  = 2;
        cfg.max_seq_len = 64;
    }

    printf("Config    : d=%d L=%d H=%d seq=%d vocab=%d\n",
           cfg.d_model, cfg.num_layers, cfg.num_heads,
           cfg.max_seq_len, cfg.vocab_size);

    // Param count
    int n_params = cfg.vocab_size * cfg.d_model          // embedding
                 + cfg.max_seq_len * cfg.d_model          // pos_emb
                 + cfg.d_model * cfg.vocab_size;          // lm_head
    // Per layer: attn (4 heads * 4 matrices) + ffn (2W+2b) + ln (4)
    n_params += cfg.num_layers * (
        cfg.num_heads * 4 * cfg.d_model * (cfg.d_model/cfg.num_heads)
        + cfg.d_model * cfg.d_model          // W_proj
        + cfg.d_model * 4*cfg.d_model        // W1
        + 4*cfg.d_model                      // b1
        + 4*cfg.d_model * cfg.d_model        // W2
        + cfg.d_model                        // b2
        + 4 * cfg.d_model                   // ln params
    );
    printf("Params    : ~%dM\n\n", n_params/1000000);

    // ── Step 3: Init GPU Model ────────────────────────────────
    printf("[3/5] Initializing GPU model...\n");
    LOGOSModel cpu_model(cfg);
    ModelGPU   gpu_model(cfg);
    gpu_model.load_from_cpu(cpu_model);

    // ── Step 4: DataLoader + Optimizer ───────────────────────
    printf("[4/5] Setting up training...\n");
    int SEQ = cfg.max_seq_len;
    DataLoader loader(dataset_path, tok, SEQ, 1);

    int EPOCHS      = 3;
    int batches     = loader.total_batches();
    int total_steps = EPOCHS * batches;

    // lr schedule: larger model → smaller lr
    float lr_init = cfg.d_model >= 256 ? 1e-4f : 3e-4f;

    GPULangevinOpt optimizer(lr_init, 0.9f, 0.05f, 1e-6f, total_steps);
    auto gpu_params = gpu_model.all_parameters();
    optimizer.init(gpu_params);

    printf("Dataset   : %d batches/epoch\n", batches);
    printf("Epochs    : %d\n", EPOCHS);
    printf("Total steps: %d\n", total_steps);
    printf("LR        : %.1e | T: 0.05 → 1e-6\n\n", lr_init);

    // ── Step 5: Training Loop ─────────────────────────────────
    printf("[5/5] Training...\n\n");

    // GPU grad buffers (allocated once, zeroed each step)
    auto gpu_grads = gpu_model.alloc_grad_buffers();

    // Loss buffer
    float* d_loss_tokens;
    CUDA_CHECK(cudaMalloc(&d_loss_tokens, SEQ * sizeof(float)));

    int   step      = 0;
    float best_loss = 999.f;
    float smooth    = -1.f;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        printf("\n── Epoch %d/%d ──\n", epoch+1, EPOCHS);
        loader.current_pos = 0;
        std::vector<int> input_ids, target_ids;

        while (loader.next_batch(input_ids, target_ids)) {
            int seq = (int)input_ids.size();

            // ── Forward ───────────────────────────────────────
            GPUTensor logits = gpu_model.forward(input_ids);

            // ── Targets H2D ───────────────────────────────────
            int* d_targets;
            CUDA_CHECK(cudaMalloc(&d_targets, seq * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_targets, target_ids.data(),
                       seq*sizeof(int), cudaMemcpyHostToDevice));

            // ── Loss + dLogits on GPU ─────────────────────────
            GPUTensor d_logits = gpu_alloc(seq, cfg.vocab_size);
            ce_loss_kernel<<<seq, 1>>>(
                logits.data, d_targets,
                d_loss_tokens, d_logits.data,
                seq, cfg.vocab_size);
            CUDA_CHECK(cudaGetLastError());

            // ── D2H loss for logging ──────────────────────────
            std::vector<float> loss_h(seq);
            CUDA_CHECK(cudaMemcpy(loss_h.data(), d_loss_tokens,
                       seq*sizeof(float), cudaMemcpyDeviceToHost));
            float avg_loss = 0;
            int   cnt      = 0;
            for (float l : loss_h) {
                if (!std::isnan(l) && !std::isinf(l))
                    { avg_loss += l; ++cnt; }
            }
            avg_loss = cnt > 0 ? avg_loss / cnt : 0.f;

            if (avg_loss < 1e-8f || std::isnan(avg_loss)) {
                gpu_free(logits); gpu_free(d_logits);
                cudaFree(d_targets);
                ++step; continue;
            }

            if (avg_loss < best_loss) best_loss = avg_loss;
            smooth = smooth < 0 ? avg_loss : 0.95f*smooth + 0.05f*avg_loss;

            // ── Zero grad buffers ─────────────────────────────
            for (auto* g : gpu_grads)
                CUDA_CHECK(cudaMemset(g->data, 0, g->size*sizeof(float)));

            // ── Backprop: lm_head ─────────────────────────────
            // grad_lm_head = X_out^T @ dLogits
            // grad_X_out   = dLogits @ lm_head^T
            // (gpu_model stores last X_out from forward)
            GPUTensor& X_out  = gpu_model.last_hidden;
            GPUTensor* g_lmh  = gpu_grads[2]; // lm_head grad

            {
                // dW: (d_model, vocab)
                dim3 blk(32);
                dim3 grd(cfg.d_model, (cfg.vocab_size+31)/32);
                gemm_backward_dw_kernel<<<grd,blk>>>(
                    X_out.data, d_logits.data, g_lmh->data,
                    seq, cfg.d_model, cfg.vocab_size);

                // dX: (seq, d_model) — reuse d_logits space? No, alloc
            }

            GPUTensor dX_out = gpu_alloc(seq, cfg.d_model);
            {
                dim3 blk(32);
                dim3 grd(seq, (cfg.d_model+31)/32);
                gemm_backward_dx_kernel<<<grd,blk>>>(
                    d_logits.data, gpu_model.gpu_lm_head.data,
                    dX_out.data,
                    seq, cfg.d_model, cfg.vocab_size);
            }

            // ── Backprop: Embedding ───────────────────────────
            // (layers don't have full backward yet — embedding gets dX)
            GPUTensor* g_emb = gpu_grads[0]; // embedding grad
            GPUTensor* g_pos = gpu_grads[1]; // pos_emb grad
            {
                dim3 blk(32);
                dim3 grd(seq, (cfg.d_model+31)/32);
                embedding_backward_kernel<<<grd,blk>>>(
                    gpu_model.d_token_ids,
                    dX_out.data,
                    g_emb->data, g_pos->data,
                    seq, cfg.d_model);
            }
            CUDA_CHECK(cudaGetLastError());

            // ── Langevin Step ─────────────────────────────────
            optimizer.update(gpu_params, gpu_grads);

            gpu_free(logits);
            gpu_free(d_logits);
            gpu_free(dX_out);
            cudaFree(d_targets);

            // ── Log ───────────────────────────────────────────
            if (step % 100 == 0) {
                printf("Step %5d | Loss: %.4f | Smooth: %.4f | T: %.2e\n",
                       step, avg_loss, smooth, optimizer.temperature);
                fflush(stdout);
            }

            // ── Checkpoint ────────────────────────────────────
            if (step > 0 && step % 1000 == 0) {
                // D2H → save checkpoint
                gpu_model.sync_to_cpu(cpu_model);
                save_checkpoint(cpu_model, "logos_gpu_ckpt", step);
                printf("💾 Checkpoint saved @ step %d | best=%.4f\n",
                       step, best_loss);
            }

            ++step;
        }
    }

    // ── Final save ────────────────────────────────────────────
    cudaFree(d_loss_tokens);
    for (auto* g : gpu_grads) { gpu_free(*g); delete g; }

    gpu_model.sync_to_cpu(cpu_model);
    save_checkpoint(cpu_model, "logos_final", step);

    printf("\n╔══════════════════════════════════════╗\n");
    printf("║  Training Complete                   ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║  Steps : %6d                      ║\n", step);
    printf("║  Best  : %.4f                      ║\n", best_loss);
    printf("║  Final : %.4f                      ║\n", smooth);
    if (smooth < best_loss * 1.1f && best_loss < 7.0f)
        printf("║  ✅ Model learned!                  ║\n");
    else
        printf("║  ⚠️  More data/steps needed          ║\n");
    printf("╚══════════════════════════════════════╝\n");
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    printf("╔══════════════════════════════════════╗\n"
           "║  LOGOS GPU — Vedic-Physics LLM       ║\n"
           "║  C++20/CUDA | Langevin Opt           ║\n"
           "╚══════════════════════════════════════╝\n\n");

    std::string dataset = (argc > 2) ? argv[2] : "dataset.txt";
    train_gpu(dataset);
    return 0;
}
