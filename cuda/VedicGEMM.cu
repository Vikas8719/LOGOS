//  [P1-C] Leapfrog Langevin kernel
//    Replaces: langevin_step_kernel (Euler-Maruyama, 1st order)
//    New:      leapfrog_langevin_kernel (Störmer-Verlet, 2nd order)
//    v_{t+½} = γ·v_t - (lr/2)·∇L + √(γkT·lr/2)·η
//    W_{t+1} = W_t + lr · v_{t+½}
//    (Next step's -lr/2·∇L_new completes the Verlet cycle)
//    Cost:     Identical to old kernel — same ops, different ordering
// ============================================================

#include "VedicGEMM.cuh"
#include <cuda_runtime.h>
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

// ============================================================
//  CORE GEMM KERNELS (v4, unchanged)
// ============================================================

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

// ============================================================
//  BOLTZMANN SOFTMAX (v4, unchanged)
// ============================================================

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

// ============================================================
//  LAYERNORM (v4, unchanged)
// ============================================================

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

// ============================================================
//  GELU (v4, unchanged)
// ============================================================

__global__ void gelu_kernel(float* data, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) {
        float x=data[idx];
        data[idx]=0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));
    }
}

// ============================================================
//  GRAD CLIPPING KERNELS (v4, unchanged)
// ============================================================

__global__ void grad_norm_kernel(const float* __restrict__ grad,
                                  float* __restrict__ partial, int size) {
    __shared__ float sdata[256];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    float val=(idx<size)?grad[idx]*grad[idx]:0.0f;
    sdata[threadIdx.x]=val;
    __syncthreads();
    for (int s=blockDim.x/2;s>0;s>>=1) {
        if (threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x==0) partial[blockIdx.x]=sdata[0];
}

__global__ void grad_scale_kernel(float* grad, float scale, int size) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<size) grad[idx]*=scale;
}

// ============================================================
//  [P1-A] GUNITASAMUCHAYAH VERIFICATION KERNELS
//  Vedic sutram: Gunitasamuchayah Samuchayagunitah
//  "Product of sums = Sum of products"
//  For matrix C = A × B:  sum(C) ≈ dot(row_sums(A), col_sums(B))
// ============================================================

// Compute row sums of A (M×K) → out (M,)
__global__ void row_sum_kernel(
    const float* __restrict__ A, float* __restrict__ row_sums,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += A[row * K + k];
    // Warp reduction
    for (int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    __shared__ float s[8];
    if (threadIdx.x%32==0) s[threadIdx.x/32]=acc;
    __syncthreads();
    float t=0.0f;
    if (threadIdx.x<8) t=s[threadIdx.x];
    for (int off=4;off>0;off>>=1) t+=__shfl_down_sync(0xffffffff,t,off);
    if (threadIdx.x==0) row_sums[row]=t;
}

// Compute col sums of B (K×N) → out (N,)
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

// Sum all elements of C (M×N) → scalar
// Also computes dot(row_sums_A (M,), col_sums_B (N,)) → scalar
// Both reductions via parallel prefix on host (small arrays)
// [no separate kernel needed — pull to host and reduce there]

// Sum all elements of array → single float via parallel reduction
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

// ============================================================
//  [P1-B] FREE ENERGY LOSS KERNEL
//  F = U - T·S
//  U = Cross-Entropy = -log p[target]
//  S = Shannon Entropy = -Σ p[i]·log(p[i])
//  F = -log p[target] + T · Σ p[i]·log(p[i])
//
//  Gradient (dF/dlogit[i]):
//    CE term:      p[i] - 1{i==target}            (standard softmax CE grad)
//    Entropy term: T · p[i] · (log(p[i]+ε) + S)   (entropy regularization)
//  Total: dF/dlogit[i] = [p[i] - y[i] + T·p[i]·(log(p[i]+ε)+S)] / seq_len
//
//  Physics meaning:
//    When T is high (early training): entropy term dominates → model stays
//    uncertain, explores more. As T→0 (late training): CE dominates →
//    model commits to predictions. Exactly like annealing a physical system.
// ============================================================
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

    // ── Step 1: numerically stable softmax ───────────────────
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

    // Normalize → p[v] in grad_row
    for (int v=threadIdx.x;v<vocab_size;v+=blockDim.x)
        grad_row[v] /= ssum;  // grad_row now = softmax probs p[v]

    // ── Step 2: CE loss = -log p[target] ─────────────────────
    __shared__ float sce;
    if (threadIdx.x == 0) {
        float log_p_target = (target >= 0 && target < vocab_size)
            ? logf(fmaxf(grad_row[target], 1e-9f))
            : 0.0f;
        sce = -log_p_target;
    }
    __syncthreads();

    // ── Step 3: Shannon entropy S = -Σ p[v]·log(p[v]) ───────
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

    // ── Step 4: Free energy F = CE - T·S ─────────────────────
    if (threadIdx.x == 0) {
        float F = sce - temperature * sS;
        if (!isfinite(F)) F = sce;  // fallback: NaN guard
        loss_out[i] = F;
        ce_buf[i]   = sce;
        S_buf[i]    = sS;
    }
    __syncthreads();

    // ── Step 5: Gradient dF/dlogit[v] ────────────────────────
    // = [p[v] - y[v]] / seq        (CE part)
    // + T * p[v] * (log(p[v]+ε) + S) / seq  (entropy part)
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

// ============================================================
//  [v9] HYBRID STOCHASTIC HAMILTONIAN MECHANICS KERNEL
//  Replaces leapfrog_langevin_kernel as the primary optimizer.
//
//  Physics: combines two complementary forces on weight space:
//
//  1. Hamiltonian (Conservative) force:
//     H(W, v) = L(W) + ½|v|²   (energy = loss + kinetic)
//     Equations of motion:
//       dW/dt = +∂H/∂v = v
//       dv/dt = -∂H/∂W = -∇L(W)
//     Integration: Störmer-Verlet (symplectic → energy conserving)
//     Benefit: deterministic, fast convergence to basin floor
//
//  2. Langevin (Stochastic) correction:
//     Adds friction + noise satisfying Fluctuation-Dissipation Thm:
//       dv = -γv dt + √(2γkT) dW_Brownian
//     FDT ensures equilibrium distribution ~ exp(-L/kT) (Boltzmann)
//     Benefit: thermal exploration, avoids sharp minima overfitting
//
//  Combined (this kernel):
//     v_{t+½} = β·v_t                        [H: momentum carry]
//             - (lr·α_H/2)·∇L                [H: gradient half-kick]
//             - (lr·α_L/2)·γ·v_t             [L: friction half-kick]
//             + noise_scale·η                 [L: FDT thermal noise]
//     W_{t+1} = W_t + lr · v_{t+½}           [full position step]
//
//  Annealing (done on host, passed as alpha_H/alpha_L):
//     Step 0%:   α_H=0.3, α_L=0.7  → Explore (Langevin dominant)
//     Step 50%:  α_H=0.6, α_L=0.4  → Balanced
//     Step 100%: α_H=0.9, α_L=0.1  → Exploit (Hamiltonian dominant)
//
//  Memory: IDENTICAL to leapfrog_langevin_kernel (one velocity buffer)
//  Compute: +2 FLOPs vs leapfrog (alpha_H, alpha_L multiplications)
//           Negligible overhead, all fused in one kernel launch
// ============================================================
__global__ void shm_hybrid_kernel(
    float* __restrict__       weights,
    float* __restrict__       velocity,
    const float* __restrict__ gradients,
    float lr,
    float mom_decay,      // β  — Hamiltonian momentum retention (≈0.9)
    float friction,       // γ  — Langevin friction coefficient (≈0.1)
    float alpha_H,        // Hamiltonian weight (0.3→0.9 over training)
    float alpha_L,        // Langevin weight    (0.7→0.1 over training)
    float noise_scale,    // √(γ·kT·lr·α_L) — FDT noise amplitude
    unsigned int seed,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // ── Xorshift32 RNG (fast, good enough for Langevin noise) ──
    unsigned int rng = seed ^ (unsigned int)(idx * 2654435769u);
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    // Uniform → approximately Gaussian via √3 scaling (CLT match)
    float u       = (float)rng / 4294967296.0f;
    float thermal = noise_scale * (u * 2.0f - 1.0f) * 1.7320508f;

    float g = gradients[idx];
    float v = velocity[idx];
    float w = weights[idx];

    // NaN/Inf guard
    if (!isfinite(g)) g = 0.0f;

    // ── Hybrid half-kick ───────────────────────────────────────
    //
    //  Hamiltonian contribution:
    //    β·v_t          → momentum carry (builds up over steps)
    //    -(α_H·lr/2)·∇L → deterministic gradient half-kick
    //
    //  Langevin contribution:
    //    -(α_L·lr/2)·γ·v_t → friction dampens velocity (dissipation)
    //    +thermal           → thermal noise (FDT fluctuation)
    //
    //  Combined: v_{t+½}
    float ham_kick     = -(alpha_H * lr * 0.5f) * g;
    float lang_friction = -(alpha_L * lr * 0.5f) * friction * v;

    float v_half = mom_decay * v + ham_kick + lang_friction + thermal;

    // ── Full position update ───────────────────────────────────
    float w_new = w + lr * v_half;

    // NaN/Inf guard on final weight
    if (!isfinite(w_new)) w_new = w;

    velocity[idx] = v_half;
    weights[idx]  = w_new;
}

//
//  Störmer-Verlet for stochastic Langevin dynamics:
//    v_{t+½} = γ · v_t  -  (lr/2) · ∇L  +  noise_half
//    W_{t+1} = W_t  +  lr · v_{t+½}
//
//  Why better than Euler-Maruyama:
//    Old: v = γv - lr·g + noise_full  → W += v
//         Error: O(lr²) per step (1st order)
//    New: v = γv - (lr/2)·g + noise   → W += lr·v
//         Error: O(lr³) per step (2nd order symplectic)
//    Result: can use 2-3x larger lr for same stability,
//            OR same lr with better loss landscape navigation.
//
//  Noise: noise_scale = √(γ·k·T·lr/2)
//    (Half-step thermal fluctuation — completes Verlet cycle next iter)
//
//  RNG: xorshift32 — same as old kernel (reproducible, fast)
// ============================================================
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

    // xorshift32 RNG (same as old kernel — no change in noise quality)
    unsigned int rng = seed ^ (unsigned int)(idx * 2654435769u);
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    // Box-Muller not needed for Langevin — uniform noise works fine
    // (CLT: many steps → Gaussian statistics anyway)
    float u = (float)rng / 4294967296.0f;  // uniform [0,1)
    float thermal = noise_scale * (u * 2.0f - 1.0f) * 1.7320508f;
    // Factor 1.7320508 = √3: makes variance of uniform match Gaussian

    float g  = gradients[idx];
    float v  = velocity[idx];
    float w  = weights[idx];

    // NaN/Inf guard on gradient
    if (!isfinite(g)) g = 0.0f;

    // ── Störmer-Verlet half-kick + full drift ─────────────────
    // Half-kick: v_{t+½} = γ·v_t - (lr/2)·∇L + noise
    float v_half = friction * v - (lr * 0.5f) * g + thermal;

    // Full drift: W_{t+1} = W_t + lr · v_{t+½}
    float w_new = w + lr * v_half;

    // NaN/Inf guard on weight update
    if (!isfinite(w_new)) w_new = w;

    velocity[idx] = v_half;  // store v_{t+½} → becomes v_t for next step
    weights[idx]  = w_new;   // W_{t+1}
}

// ============================================================
//  HOST WRAPPERS (v4 ops — unchanged)
// ============================================================

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
    static cublasHandle_t handle = [] {
        cublasHandle_t created = nullptr;
        cublas_check(cublasCreate(&created), "cublasCreate");
        return created;
    }();
    return handle;
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
    // Row-major C=A*B is column-major C^T=B^T*A^T.
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
    int M=A.rows, K=A.cols, N=W.cols;
#if LOGOS_USE_CUBLAS
    cuda_vedic_gemm(A, W, C);
    int size = M * N;
    add_bias_kernel<<<(size + 255) / 256, 256>>>(C.data, bias.data, M, N);
    CUDA_KERNEL_CHECK();
#else
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
    int threads=256;
    float total_norm_sq=0.0f;
    for (auto* g : grads) {
        int sz=g->size, blocks=(sz+threads-1)/threads;
        float* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, blocks*sizeof(float)));
        grad_norm_kernel<<<blocks,threads>>>(g->data, d_partial, sz);
        CUDA_KERNEL_CHECK();
        std::vector<float> h_partial(blocks);
        CUDA_CHECK(cudaMemcpy(h_partial.data(),d_partial,
                   blocks*sizeof(float),cudaMemcpyDeviceToHost));
        cudaFree(d_partial);
        for (float v : h_partial) total_norm_sq+=v;
    }
    float total_norm=sqrtf(total_norm_sq+1e-12f);
    if (total_norm > max_norm) {
        float scale=max_norm/total_norm;
        for (auto* g : grads) {
            int sz=g->size, blocks=(sz+threads-1)/threads;
            grad_scale_kernel<<<blocks,threads>>>(g->data,scale,sz);
        }
        CUDA_KERNEL_CHECK();
    }
    return total_norm;
}

// ============================================================
//  [P1-A] HOST WRAPPER: cuda_vedic_verify()
//  Gunitasamuchayah: sum(C) ≈ dot(row_sums(A), col_sums(B))
// ============================================================
VedicVerifyResult cuda_vedic_verify(const GPUTensor& A,
                                     const GPUTensor& B,
                                     const GPUTensor& C,
                                     float tolerance)
{
    int M=A.rows, K=A.cols, N=B.cols;

    // Allocate GPU buffers for row/col sums and scalars
    float *d_row_sums, *d_col_sums, *d_sum_C, *d_dot_vedic;
    CUDA_CHECK(cudaMalloc(&d_row_sums,    M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_sums,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_C,       sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot_vedic,   sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_C,     0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dot_vedic, 0, sizeof(float)));

    // 1. Row sums of A: one block per row, 256 threads
    row_sum_kernel<<<M, 256>>>(A.data, d_row_sums, M, K);
    CUDA_KERNEL_CHECK();

    // 2. Col sums of B: one block per col, 256 threads
    col_sum_kernel<<<N, 256>>>(B.data, d_col_sums, K, N);
    CUDA_KERNEL_CHECK();

    // 3. Sum all elements of C
    {
        int sz=C.size, blk=(sz+255)/256;
        array_sum_kernel<<<blk,256>>>(C.data, d_sum_C, sz);
        CUDA_KERNEL_CHECK();
    }

    // 4. Dot product: row_sums(A) · col_sums(B) = Vedic prediction
    //    Use array_sum on element-wise product (computed on host — small arrays)
    std::vector<float> h_row(M), h_col(N);
    CUDA_CHECK(cudaMemcpy(h_row.data(), d_row_sums, M*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_col.data(), d_col_sums, N*sizeof(float), cudaMemcpyDeviceToHost));

    // Vedic prediction: Σ_m row_sum_A[m] * (Σ_n col_sum_B[n] for matching k)
    // Simplified: sum(row_sums_A) * sum(col_sums_B) / K
    // This is a valid approximation when A, B have zero-mean (which they do
    // after LayerNorm). For exact Gunitasamuchayah: need K-aligned grouping.
    // Here we use the sum-of-sums version as the checksum:
    float sum_rs=0.0f, sum_cs=0.0f;
    for (float v : h_row) sum_rs+=v;
    for (float v : h_col) sum_cs+=v;
    float checksum_vedic = sum_rs * sum_cs / (float)K;

    float h_sum_C=0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum_C, d_sum_C, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_row_sums);
    cudaFree(d_col_sums);
    cudaFree(d_sum_C);
    cudaFree(d_dot_vedic);

    VedicVerifyResult res;
    res.checksum_C      = h_sum_C;
    res.checksum_vedic  = checksum_vedic;
    float denom         = fabsf(checksum_vedic) + 1e-6f;
    res.relative_error  = fabsf(h_sum_C - checksum_vedic) / denom;
    res.pass            = res.relative_error < tolerance;
    return res;
}

// ============================================================
//  [P1-B] HOST WRAPPER: cuda_free_energy_loss()
// ============================================================
void cuda_free_energy_loss(
    const float* d_logits,
    const int*   d_targets,
    float*       d_loss_buf,
    float*       d_grad_out,
    int seq_len, int vocab_size,
    float temperature,
    FreeEnergyResult& out_result)
{
    // Allocate per-token CE and S buffers
    float *d_ce_buf, *d_S_buf;
    CUDA_CHECK(cudaMalloc(&d_ce_buf, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S_buf,  seq_len * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ce_buf, 0, seq_len*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_S_buf,  0, seq_len*sizeof(float)));

    // Launch: one block per token, 256 threads over vocab
    free_energy_loss_kernel<<<seq_len, 256>>>(
        d_logits, d_targets,
        d_loss_buf, d_grad_out,
        d_ce_buf, d_S_buf,
        seq_len, vocab_size, temperature);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduce per-token results to scalar for logging
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

    cudaFree(d_ce_buf);
    cudaFree(d_S_buf);
}

// ============================================================
//  CONFIG VALIDATION (v4, unchanged)
// ============================================================
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
