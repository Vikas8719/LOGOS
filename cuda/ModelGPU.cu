// ============================================================
//  LOGOS — cuda/ModelGPU.cu
//  Complete GPU Model: Embedding + TransformerBlocks + LM Head
//  All computation GPU pe — CPU sirf data send karta hai
// ============================================================
#include "ModelGPU.cuh"
#include "VedicGEMM.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <cstring>

#define CUDA_CHECK(call) \
    do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA Error: %s\n",cudaGetErrorString(e)); \
    exit(1); } } while(0)

// ── Embedding Lookup Kernel ───────────────────────────────────
// token_ids[i] → embedding row → X[i][:]
// Also adds positional embedding
__global__ void embedding_kernel(
    const int*   __restrict__ token_ids,
    const float* __restrict__ token_emb,    // (vocab, d_model)
    const float* __restrict__ pos_emb,      // (max_seq, d_model)
    float*       __restrict__ X,            // (seq, d_model)
    int seq_len, int d_model, int vocab_size)
{
    int i = blockIdx.x;                     // sequence position
    int j = blockIdx.y * blockDim.x + threadIdx.x;  // d_model dim
    if (i >= seq_len || j >= d_model) return;

    int tok = token_ids[i];
    // Clamp to vocab range
    if (tok < 0) tok = 0;
    if (tok >= vocab_size) tok = 0;

    X[i * d_model + j] = token_emb[tok * d_model + j]
                        + pos_emb[i   * d_model + j];
}

// ── Residual Add Kernel ───────────────────────────────────────
__global__ void residual_add_kernel(
    float*       __restrict__ out,    // in-place: out = out + residual
    const float* __restrict__ residual,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] += residual[idx];
}

// ── Causal Mask Apply Kernel ──────────────────────────────────
// scores[i][j] += -1e9 if j > i  (future tokens mask)
__global__ void causal_mask_kernel(float* scores, int seq_len) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= seq_len) return;
    if (col > row) scores[row * seq_len + col] += -1e9f;
}

// ── Cross-Entropy Loss + Softmax Grad Kernel ─────────────────
__global__ void ce_loss_kernel(
    const float* __restrict__ logits,    // (seq, vocab)
    const int*   __restrict__ targets,   // (seq,)
    float*       __restrict__ loss_out,  // (seq,) per-token loss
    float*       __restrict__ grad_out,  // (seq, vocab) gradient
    int seq_len, int vocab_size)
{
    int i = blockIdx.x;   // sequence position
    if (i >= seq_len) return;

    const float* logit_row = logits   + i * vocab_size;
    float*       grad_row  = grad_out + i * vocab_size;
    int target = targets[i];

    // Numerically stable softmax
    float max_l = logit_row[0];
    for (int v = 1; v < vocab_size; ++v)
        max_l = fmaxf(max_l, logit_row[v]);

    float sum = 0.0f;
    for (int v = 0; v < vocab_size; ++v)
        sum += expf(logit_row[v] - max_l);

    // Loss
    float loss = -(logit_row[target] - max_l - logf(sum));
    loss_out[i] = loss;

    // Softmax gradient: dL/dlogit_v = softmax_v - 1(v==target)
    for (int v = 0; v < vocab_size; ++v) {
        float sm = expf(logit_row[v] - max_l) / sum;
        grad_row[v] = (sm - (v == target ? 1.0f : 0.0f)) / seq_len;
    }
}

// ── GPU Transpose Kernel ──────────────────────────────────────
// Transpose matrix A (rows×cols) → AT (cols×rows)
// Uses shared memory tile for coalesced access
__global__ void gpu_transpose_kernel(
    const float* __restrict__ A,   // input  (rows × cols)
    float*       __restrict__ AT,  // output (cols × rows)
    int rows, int cols)
{
    __shared__ float tile[16][17]; // +1 to avoid bank conflicts
    int row_in = blockIdx.y * 16 + threadIdx.y;
    int col_in = blockIdx.x * 16 + threadIdx.x;
    if (row_in < rows && col_in < cols)
        tile[threadIdx.y][threadIdx.x] = A[row_in * cols + col_in];
    __syncthreads();
    int row_out = blockIdx.x * 16 + threadIdx.y;
    int col_out = blockIdx.y * 16 + threadIdx.x;
    if (row_out < cols && col_out < rows)
        AT[row_out * rows + col_out] = tile[threadIdx.x][threadIdx.y];
}

// ── Langevin Optimizer Kernel ─────────────────────────────────
// W += -(1-friction)*v - lr*grad + noise
// v  = (1-friction)*v - lr*grad + noise  (momentum update)
__global__ void langevin_step_kernel(
    float* __restrict__ W,          // weights
    float* __restrict__ velocity,   // momentum
    const float* __restrict__ grad, // gradients
    float lr, float friction,
    float noise_scale,
    unsigned int seed,              // random seed
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // FIX 3 (Uniform != Gaussian): Sahi Box-Muller transform se Gaussian noise
    // Pehle: u*2-1 uniform tha [-1,1] — Gaussian nahi, Langevin ke liye galat distribution
    // Ab: Do independent LCG values se proper Box-Muller → actual N(0,1) Gaussian
    unsigned int rng1 = seed + idx * 1664525u + 1013904223u;
    rng1 = rng1 * 1664525u + 1013904223u;
    unsigned int rng2 = rng1 * 1664525u + 1013904223u;
    // Box-Muller: u1 ∈ (0,1], u2 ∈ [0,1) → N(0,1)
    float u1 = fmaxf((float)(rng1 & 0x7FFFFFFF) / (float)0x7FFFFFFF, 1e-6f); // guard: log(0) se bachao
    float u2 = (float)(rng2 & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    float noise = noise_scale * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2); // true N(0,1)

    float g = grad[idx];
    if (isnan(g) || isinf(g)) g = 0.0f;

    float v = (1.0f - friction) * velocity[idx] - lr * g + noise;
    velocity[idx] = v;
    W[idx] += v;
}

// ── Gradient Clip Kernel ──────────────────────────────────────
// Two-pass: first reduce norm, then scale
__global__ void grad_norm_kernel(
    const float* __restrict__ grad,
    float* __restrict__ partial_norms,
    int size)
{
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < size) ? grad[idx] * grad[idx] : 0.0f;
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_norms[blockIdx.x] = sdata[0];
}

__global__ void grad_scale_kernel(float* grad, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad[idx] *= scale;
}

// ── Host-side ModelGPU implementation ────────────────────────

ModelGPU::ModelGPU(const ModelConfig& cfg_) : cfg(cfg_) {
    int V = cfg.vocab_size, D = cfg.d_model;
    int S = cfg.max_seq_len;

    // Allocate all weight tensors on GPU
    gpu_embedding     = gpu_alloc(V, D);
    gpu_pos_embedding = gpu_alloc(S, D);
    gpu_lm_head       = gpu_alloc(D, V);

    // Each transformer block
    for (int l = 0; l < cfg.num_layers; ++l) {
        GPUBlock blk;
        int H = cfg.num_heads, DH = D / H;

        // Attention weights per head
        for (int h = 0; h < H; ++h) {
            blk.W_Q.push_back(gpu_alloc(D, DH));
            blk.W_K.push_back(gpu_alloc(D, DH));
            blk.W_V.push_back(gpu_alloc(D, DH));
            blk.W_O.push_back(gpu_alloc(DH, D));
        }
        blk.W_proj = gpu_alloc(D, D);

        // FFN
        blk.W1 = gpu_alloc(D, 4*D);
        blk.b1 = gpu_alloc(1, 4*D);
        blk.W2 = gpu_alloc(4*D, D);
        blk.b2 = gpu_alloc(1, D);

        // LayerNorm
        blk.ln1_gamma = gpu_alloc(1, D);
        blk.ln1_beta  = gpu_alloc(1, D);
        blk.ln2_gamma = gpu_alloc(1, D);
        blk.ln2_beta  = gpu_alloc(1, D);

        // Init gamma=1, beta=0
        std::vector<float> ones(D, 1.0f), zeros(D, 0.0f);
        h2d(blk.ln1_gamma, ones.data(),  D);
        h2d(blk.ln2_gamma, ones.data(),  D);
        h2d(blk.ln1_beta,  zeros.data(), D);
        h2d(blk.ln2_beta,  zeros.data(), D);

        gpu_blocks.push_back(blk);
    }

    // Allocate token_ids buffer on GPU
    CUDA_CHECK(cudaMalloc(&d_token_ids, cfg.max_seq_len * sizeof(int)));

    std::cout << "✅ GPU Model allocated on device\n";
    std::cout << "   Layers: " << cfg.num_layers
              << " | d_model: " << D
              << " | vocab: " << V << "\n";

    // Print GPU memory used
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "   GPU memory: "
              << (total_mem - free_mem)/1024/1024 << " MB used / "
              << total_mem/1024/1024 << " MB total\n";
}

ModelGPU::~ModelGPU() {
    gpu_free(gpu_embedding);
    gpu_free(gpu_pos_embedding);
    gpu_free(gpu_lm_head);
    for (auto& blk : gpu_blocks) {
        for (int h = 0; h < cfg.num_heads; ++h) {
            gpu_free(blk.W_Q[h]); gpu_free(blk.W_K[h]);
            gpu_free(blk.W_V[h]); gpu_free(blk.W_O[h]);
        }
        gpu_free(blk.W_proj);
        gpu_free(blk.W1); gpu_free(blk.b1);
        gpu_free(blk.W2); gpu_free(blk.b2);
        gpu_free(blk.ln1_gamma); gpu_free(blk.ln1_beta);
        gpu_free(blk.ln2_gamma); gpu_free(blk.ln2_beta);
    }
    if (d_token_ids) cudaFree(d_token_ids);
}

// Copy CPU weights → GPU
void ModelGPU::load_from_cpu(const LOGOSModel& cpu_model) {
    h2d(gpu_embedding,     cpu_model.embedding.data.data(),
                           cpu_model.embedding.total_size);
    h2d(gpu_pos_embedding, cpu_model.pos_embedding.data.data(),
                           cpu_model.pos_embedding.total_size);
    h2d(gpu_lm_head,       cpu_model.lm_head.data.data(),
                           cpu_model.lm_head.total_size);

    for (int l = 0; l < cfg.num_layers; ++l) {
        auto& gblk = gpu_blocks[l];
        auto& cblk = cpu_model.layers[l];
        for (int h = 0; h < cfg.num_heads; ++h) {
            h2d(gblk.W_Q[h], cblk.mha.heads[h].W_Q.data.data(), cblk.mha.heads[h].W_Q.total_size);
            h2d(gblk.W_K[h], cblk.mha.heads[h].W_K.data.data(), cblk.mha.heads[h].W_K.total_size);
            h2d(gblk.W_V[h], cblk.mha.heads[h].W_V.data.data(), cblk.mha.heads[h].W_V.total_size);
            h2d(gblk.W_O[h], cblk.mha.heads[h].W_O.data.data(), cblk.mha.heads[h].W_O.total_size);
        }
        h2d(gblk.W_proj, cblk.mha.W_proj.data.data(), cblk.mha.W_proj.total_size);
        h2d(gblk.W1, cblk.ffn.W1.data.data(), cblk.ffn.W1.total_size);
        h2d(gblk.b1, cblk.ffn.b1.data.data(), cblk.ffn.b1.total_size);
        h2d(gblk.W2, cblk.ffn.W2.data.data(), cblk.ffn.W2.total_size);
        h2d(gblk.b2, cblk.ffn.b2.data.data(), cblk.ffn.b2.total_size);
    }
    std::cout << "✅ Weights loaded CPU → GPU\n";
}

// Forward pass — all on GPU
GPUTensor ModelGPU::forward(const std::vector<int>& token_ids) {
    int seq = (int)token_ids.size();
    int D   = cfg.d_model;
    int H   = cfg.num_heads;
    int DH  = D / H;
    int V   = cfg.vocab_size;

    // H2D: token ids
    CUDA_CHECK(cudaMemcpy(d_token_ids, token_ids.data(),
               seq * sizeof(int), cudaMemcpyHostToDevice));

    // Embedding lookup + positional encoding
    GPUTensor X = gpu_alloc(seq, D);
    {
        dim3 grid(seq, (D+31)/32);
        dim3 block(32);
        embedding_kernel<<<grid, block>>>(
            d_token_ids,
            gpu_embedding.data, gpu_pos_embedding.data,
            X.data, seq, D, V);
    }

    // N Transformer blocks
    for (int l = 0; l < cfg.num_layers; ++l) {
        auto& blk = gpu_blocks[l];

        // === Attention path ===
        GPUTensor normed1 = gpu_alloc(seq, D);
        cuda_layernorm(X, blk.ln1_gamma, blk.ln1_beta, normed1, seq, D);

        // Multi-head attention
        GPUTensor concat = gpu_alloc(seq, D);
        for (int h = 0; h < H; ++h) {
            GPUTensor Q = gpu_alloc(seq, DH);
            GPUTensor K = gpu_alloc(seq, DH);
            GPUTensor V_t = gpu_alloc(seq, DH);

            cuda_vedic_gemm(normed1, blk.W_Q[h], Q);
            cuda_vedic_gemm(normed1, blk.W_K[h], K);
            cuda_vedic_gemm(normed1, blk.W_V[h], V_t);

            // Scores = Q × Kᵀ
            // FIX 4 (CPU Transpose Bug): Pehle K CPU pe transpose ho raha tha — d2h → loop → h2d
            // Yeh har attention head ke liye GPU sync + 2x PCIe transfer tha → massive slowdown
            // Ab: GPU pe hi transpose karte hain ek dedicated kernel se
            GPUTensor K_T = gpu_alloc(DH, seq);
            {
                // GPU transpose kernel: grid covers output (DH × seq)
                dim3 tr_block(16, 16);
                dim3 tr_grid((seq + 15) / 16, (DH + 15) / 16);
                gpu_transpose_kernel<<<tr_grid, tr_block>>>(K.data, K_T.data, seq, DH);
            }

            GPUTensor scores = gpu_alloc(seq, seq);
            cuda_vedic_gemm(Q, K_T, scores);

            // Causal mask
            {
                dim3 g(seq, (seq+31)/32); dim3 b(32);
                causal_mask_kernel<<<g,b>>>(scores.data, seq);
            }

            // Boltzmann softmax
            float temperature = sqrtf((float)DH);
            GPUTensor attn_w = gpu_alloc(seq, seq);
            cuda_boltzmann_softmax(scores, attn_w, seq, seq, temperature);

            // Output = attn_w × V
            GPUTensor head_out = gpu_alloc(seq, DH);
            cuda_vedic_gemm(attn_w, V_t, head_out);

            // Project back and add to concat
            GPUTensor head_proj = gpu_alloc(seq, D);
            cuda_vedic_gemm(head_out, blk.W_O[h], head_proj);

            // Accumulate into concat (simple add)
            int sz = seq * D;
            int threads = 256;
            residual_add_kernel<<<(sz+threads-1)/threads, threads>>>(
                concat.data, head_proj.data, sz);

            gpu_free(Q); gpu_free(K); gpu_free(V_t);
            gpu_free(K_T); gpu_free(scores);
            gpu_free(attn_w); gpu_free(head_out); gpu_free(head_proj);
        }

        // Final projection
        GPUTensor mha_out = gpu_alloc(seq, D);
        cuda_vedic_gemm(concat, blk.W_proj, mha_out);
        gpu_free(concat); gpu_free(normed1);

        // Residual 1: X = X + mha_out
        {
            int sz = seq*D, th=256;
            residual_add_kernel<<<(sz+th-1)/th, th>>>(
                X.data, mha_out.data, sz);
        }
        gpu_free(mha_out);

        // === FFN path ===
        GPUTensor normed2 = gpu_alloc(seq, D);
        cuda_layernorm(X, blk.ln2_gamma, blk.ln2_beta, normed2, seq, D);

        GPUTensor ffn_h = gpu_alloc(seq, 4*D);
        cuda_vedic_gemm_bias(normed2, blk.W1, blk.b1, ffn_h);
        cuda_gelu(ffn_h);

        GPUTensor ffn_out = gpu_alloc(seq, D);
        cuda_vedic_gemm_bias(ffn_h, blk.W2, blk.b2, ffn_out);
        gpu_free(normed2); gpu_free(ffn_h);

        // Residual 2: X = X + ffn_out
        {
            int sz=seq*D, th=256;
            residual_add_kernel<<<(sz+th-1)/th, th>>>(
                X.data, ffn_out.data, sz);
        }
        gpu_free(ffn_out);
    }

    // LM Head: logits = X × lm_head
    GPUTensor logits = gpu_alloc(seq, V);
    cuda_vedic_gemm(X, gpu_lm_head, logits);
    gpu_free(X);

    return logits;   // stays on GPU
}
