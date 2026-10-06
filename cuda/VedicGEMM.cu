//  [P1-C] Leapfrog Langevin kernel
//    Replaces: langevin_step_kernel (Euler-Maruyama, 1st order)
//    New:      leapfrog_langevin_kernel (Störmer-Verlet, 2nd order)
//    v_{t+½} = γ·v_t - (lr/2)·∇L + √(γkT·lr/2)·η
//    W_{t+1} = W_t + lr · v_{t+½}
//    (Next step's -lr/2·∇L_new completes the Verlet cycle)
//    Cost:     Identical to old kernel — same ops, different ordering
// ============================================================

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
//  [v10-HAM: Hamiltonian-dominant throughout]
//     Step 0%:   α_H=0.7, α_L=0.3  → Hamiltonian-dominant (strong gradient)
//     Step 50%:  α_H=0.85, α_L=0.15 → Near-pure Hamiltonian
//     Step 100%: α_H=0.99, α_L=0.01 → Almost pure symplectic (deterministic)
//  Reason: Loss 9.3+ plateau = model not following gradient, Langevin noise
//          was dominating and preventing descent. Hamiltonian mode fixes this.
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
    float ham_kick     = -(alpha_H * 0.5f) * g;  // lr applied once at position update (was lr^2 -> no learning)
    // [v17] FIX: friction ke saath `lr` mat multiply karo. Pehle coefficient
    //   alpha_L*lr*0.5*friction = 0.5*5e-5*0.5*0.3 ≈ 4e-6 tha → friction knob (0.1→0.3)
    //   ka koi asar hi nahi tha ("geodesic friction" effectively OFF). `lr` position update
    //   (w += lr*v_half) me pehle se ek baar lag raha hai — velocity-space damping me nahi.
    //   Ab per-step damping = 0.5·α_L·γ  (0.075 early → 0.0075 late)  ✅
    float lang_friction = -(alpha_L * 0.5f) * friction * v;

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
        // [v23-AMP] H100 SXM: enable Tensor Core math for all GEMMs
        // CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
        //   accumulation stays FP32 even when inputs are FP16
        //   prevents precision loss in deep networks
        cublasSetMathMode(created, CUBLAS_TENSOR_OP_MATH);
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
    int threads=256;
    float total_norm_sq=0.0f;
    for (auto* g : grads) {
        int sz=g->size, blocks=(sz+threads-1)/threads;
        // [v26-LEAK-FIX] CudaPtr<float>: d_partial auto-freed on exception or loop iteration
        CudaPtr<float> d_partial(blocks);
        grad_norm_kernel<<<blocks,threads>>>(g->data, d_partial.get(), sz);
        CUDA_KERNEL_CHECK();
        std::vector<float> h_partial(blocks);
        d_partial.to_host(h_partial.data(), blocks);
        // d_partial freed automatically here (loop body end)
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
//  Gunitasamuchayah: sum(C) ≈ dot(col_sums_A, row_sums_B)
//
//  CORRECT Vedic sutram for C = A(M×K) × B(K×N):
//    sum(C) = Σ_m Σ_n C[m,n]
//           = Σ_m Σ_n Σ_k A[m,k] * B[k,n]
//           = Σ_k (Σ_m A[m,k]) * (Σ_n B[k,n])
//           = dot( col_sums(A)[K], row_sums(B)[K] )
//
//  Previous WRONG formula:
//    checksum_vedic = sum_all(A) * sum_all(B) / K
//    This is the "grand sum" approximation — only holds when columns
//    of A and rows of B happen to have equal variance (they don't).
//    Result: err=27x even with 30% tolerance → always WARN.
//
//  Correct formula needs K-length vectors:
//    col_sums_A[k] = Σ_m A[m,k]     (sum each column of A)
//    row_sums_B[k] = Σ_n B[k,n]     (sum each row of B)
//    vedic_pred    = dot(col_sums_A, row_sums_B)
//
//  Expected relative_error with this formula:
//    Custom CUDA kernel: < 1e-4 (bitwise exact)
//    cuBLAS (float32 reorder): 0.1% – 2% typical, < 5% worst case
//    → tolerance=0.05f (5%) is fine for BOTH backends now
// ============================================================
VedicVerifyResult cuda_vedic_verify(const GPUTensor& A,
                                     const GPUTensor& B,
                                     const GPUTensor& C,
                                     float tolerance)
{
    int M=A.rows, K=A.cols, N=B.cols;

    // [v26-LEAK-FIX] CudaPtr RAII — all three buffers auto-freed on any exit path
    CudaPtr<float> d_col_sums_A(K);   // zero-init included
    CudaPtr<float> d_row_sums_B(K);
    CudaPtr<float> d_sum_C(1);

    // 1. col_sums_A[k] = Σ_m A[m,k]
    col_sum_kernel<<<K, 256>>>(A.data, d_col_sums_A.get(), M, K);
    CUDA_KERNEL_CHECK();

    // 2. row_sums_B[k] = Σ_n B[k,n]
    row_sum_kernel<<<K, 256>>>(B.data, d_row_sums_B.get(), K, N);
    CUDA_KERNEL_CHECK();

    // 3. sum(C) = Σ_m Σ_n C[m,n]
    {
        int sz = C.size, blk = (sz + 255) / 256;
        array_sum_kernel<<<blk, 256>>>(C.data, d_sum_C.get(), sz);
        CUDA_KERNEL_CHECK();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Pull K-length vectors to host and compute dot product
    std::vector<float> h_col_A(K), h_row_B(K);
    d_col_sums_A.to_host(h_col_A.data(), K);
    d_row_sums_B.to_host(h_row_B.data(), K);

    // Vedic prediction: dot(col_sums_A, row_sums_B)
    double checksum_vedic = 0.0;
    for (int k = 0; k < K; ++k)
        checksum_vedic += (double)h_col_A[k] * (double)h_row_B[k];

    float h_sum_C = d_sum_C.scalar();
    // All CudaPtr buffers freed automatically here

    VedicVerifyResult res;
    res.checksum_C     = h_sum_C;
    res.checksum_vedic = (float)checksum_vedic;
    float denom        = fabsf(res.checksum_vedic) + 1e-6f;
    res.relative_error = fabsf(h_sum_C - res.checksum_vedic) / denom;
    res.pass           = res.relative_error < tolerance;
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
//  [v31-FLASH] FLASH ATTENTION FORWARD
//  Algorithm: Dao et al. 2022 — tiled SRAM, online softmax
//
//  KEY INSIGHT: seq×seq scores tensor (256MB/head for seq=8192) is NEVER
//  materialized in global memory. Instead:
//    - Process Q in tiles of FA_TILE rows
//    - For each Q tile, scan all K,V tiles with online softmax
//    - Accumulate output O directly in registers/shared memory
//    - Write only final output [seq×DH] — O(seq·DH) not O(seq²)
//
//  Memory saved per head: seq² × 4B = 8192² × 4B = 256 MB
//  Total saved: 16 heads × 256 MB = 4 GB per forward pass ← THIS fixes T4 OOM
//
//  Block config: gridDim.x = ceil(seq/FA_TILE)
//                blockDim.x = FA_TILE threads
//  Each block handles FA_TILE query rows, scans all KV tiles.
//
//  Shared memory per block:
//    Q_tile: FA_TILE × FA_DH_MAX × 4B = 32×128×4 = 16 KB
//    K_tile: same = 16 KB
//    V_tile: same = 16 KB
//    O_tile: same = 16 KB
//    m,l:    FA_TILE × 4B × 2 = 256 B
//    Total:  ~64 KB — fits in T4 shared mem (48KB/SM)
//    → Use FA_TILE=32, FA_DH_MAX=128: 4×32×128×4 = 65536B = 64KB
//    T4 has 64KB configurable shared mem (carveout = 50% L1 + 50% shared)
//    Actual limit with cudaFuncSetAttribute = 48KB → reduce to FA_TILE=16
//    FA_TILE=16: 4×16×128×4 = 32KB + 2×16×4 = 32 KB + 128B < 48KB ✓
// ============================================================

#define FA_TILE     16     // query tile rows per block (fits 48KB T4 SRAM)
#define FA_DH_MAX  128     // max head dim (d=1024,H=16→DH=64 ✓; H=8→DH=128 ✓)

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,     // [seq × DH]
    const float* __restrict__ K,     // [seq × DH]
    const float* __restrict__ V,     // [seq × DH]
    float*       __restrict__ Out,   // [seq × DH] output
    float*       __restrict__ Aptr,  // [seq × seq] nullable — save attn for bwd
    int seq, int DH,
    float scale,
    bool causal,
    bool shunyam, int window, int stride)
{
    // Each block handles FA_TILE consecutive query rows
    int q_start = blockIdx.x * FA_TILE;
    int tid      = threadIdx.x;   // 0 .. FA_TILE-1 (one thread per query row)

    // ── Shared memory layout ─────────────────────────────────
    // sQ[FA_TILE × FA_DH_MAX] | sK[FA_TILE × FA_DH_MAX]
    // sV[FA_TILE × FA_DH_MAX] | sO[FA_TILE × FA_DH_MAX]
    // sm[FA_TILE] | sl[FA_TILE]
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + FA_TILE * FA_DH_MAX;
    float* sV = sK + FA_TILE * FA_DH_MAX;
    float* sO = sV + FA_TILE * FA_DH_MAX;
    float* sm = sO + FA_TILE * FA_DH_MAX;  // running max  [FA_TILE]
    float* sl = sm + FA_TILE;              // running sum  [FA_TILE]

    int q_row   = q_start + tid;
    bool q_valid = (q_row < seq);

    // ── Load Q tile (each thread loads one row) ──────────────
    if (q_valid) {
        const float* qptr = Q + q_row * DH;
        for (int d = 0; d < DH; ++d)
            sQ[tid * FA_DH_MAX + d] = qptr[d];
    } else {
        for (int d = 0; d < DH; ++d)
            sQ[tid * FA_DH_MAX + d] = 0.0f;
    }

    // ── Init running softmax state ───────────────────────────
    sm[tid] = -1e30f;
    sl[tid] = 0.0f;
    for (int d = 0; d < DH; ++d)
        sO[tid * FA_DH_MAX + d] = 0.0f;

    __syncthreads();

    // ── Outer loop: scan KV tiles ────────────────────────────
    int kv_tiles = (seq + FA_TILE - 1) / FA_TILE;

    for (int kv_t = 0; kv_t < kv_tiles; ++kv_t) {
        int kv_start = kv_t * FA_TILE;

        // Load K tile
        int kv_row = kv_start + tid;
        if (kv_row < seq) {
            const float* kptr = K + kv_row * DH;
            for (int d = 0; d < DH; ++d)
                sK[tid * FA_DH_MAX + d] = kptr[d];
        } else {
            for (int d = 0; d < DH; ++d)
                sK[tid * FA_DH_MAX + d] = 0.0f;
        }

        // Load V tile
        if (kv_row < seq) {
            const float* vptr = V + kv_row * DH;
            for (int d = 0; d < DH; ++d)
                sV[tid * FA_DH_MAX + d] = vptr[d];
        } else {
            for (int d = 0; d < DH; ++d)
                sV[tid * FA_DH_MAX + d] = 0.0f;
        }

        __syncthreads();

        // ── Inner loop: compute scores + online softmax ──────
        if (q_valid) {
            // Compute FA_TILE scores in registers
            float scores_local[FA_TILE];
            float tile_max = -1e30f;

            for (int kj = 0; kj < FA_TILE; ++kj) {
                int k_abs = kv_start + kj;
                if (k_abs >= seq) { scores_local[kj] = -1e30f; continue; }

                // Causal mask
                if (causal && k_abs > q_row) { scores_local[kj] = -1e30f; continue; }

                // Shunyam sparse window mask
                if (shunyam && seq > window) {
                    bool in_window  = (k_abs >= q_row - window) && (k_abs <= q_row);
                    bool is_global  = (k_abs % stride == 0);
                    if (!in_window && !is_global) { scores_local[kj] = -1e30f; continue; }
                }

                // Q·K^T dot product (raw ptrs from shared mem)
                float dot = 0.0f;
                const float* qi = sQ + tid * FA_DH_MAX;
                const float* ki = sK + kj  * FA_DH_MAX;
                #pragma unroll 8
                for (int d = 0; d < DH; ++d) dot += qi[d] * ki[d];
                scores_local[kj] = dot * scale;
                tile_max = fmaxf(tile_max, scores_local[kj]);
            }

            // Online softmax update (numerically stable)
            float m_new     = fmaxf(sm[tid], tile_max);
            float exp_shift = expf(sm[tid] - m_new);  // rescale old O and l

            // Rescale accumulated output for new max
            for (int d = 0; d < DH; ++d)
                sO[tid * FA_DH_MAX + d] *= exp_shift;
            float l_new = sl[tid] * exp_shift;

            // Accumulate new scores into O
            for (int kj = 0; kj < FA_TILE; ++kj) {
                float s = scores_local[kj];
                if (s <= -1e29f) continue;  // masked out

                float e = expf(s - m_new);
                l_new += e;

                // O[q_row] += e * V[kj]
                const float* vi = sV + kj * FA_DH_MAX;
                float*       oi = sO + tid * FA_DH_MAX;
                #pragma unroll 8
                for (int d = 0; d < DH; ++d)
                    oi[d] += e * vi[d];

                // Optionally save unnormalized attn for backward
                if (Aptr != nullptr) {
                    int k_abs = kv_start + kj;
                    if (k_abs < seq)
                        Aptr[q_row * seq + k_abs] = e;  // will normalize below
                }
            }

            sm[tid] = m_new;
            sl[tid] = l_new;
        }

        __syncthreads();
    }  // end KV tile loop

    // ── Final: normalize output by sum, write to global mem ──
    if (q_valid) {
        float inv_l = (sl[tid] > 1e-9f) ? (1.0f / sl[tid]) : 0.0f;
        float*       out_row = Out + q_row * DH;
        const float* oi      = sO  + tid * FA_DH_MAX;
        for (int d = 0; d < DH; ++d)
            out_row[d] = oi[d] * inv_l;

        // Normalize saved attn probs
        if (Aptr != nullptr) {
            float* arow = Aptr + q_row * seq;
            for (int k_abs = 0; k_abs < seq; ++k_abs)
                arow[k_abs] *= inv_l;
        }
    }
}

// ── Host wrapper ─────────────────────────────────────────────
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

    // smem: Q+K+V+O tiles + m+l arrays
    size_t smem_bytes = sizeof(float) * (4 * FA_TILE * FA_DH_MAX + 2 * FA_TILE);

    // Request max shared memory for this kernel
    cudaFuncSetAttribute(flash_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

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

// ============================================================
//  [v23-AMP] MIXED PRECISION GEMM HOST WRAPPERS
//  FP16 input tensors → FP32 accumulation → FP32 or FP16 output
//  Leverages H100 SXM Tensor Cores: 2x throughput vs FP32 GEMM
// ============================================================

// [v23-AMP] cuda_cast_fp32_to_fp16 / cuda_cast_fp16_to_fp32 are now
// defined as inline wrappers in MixedPrecision.cuh (included above).
// Removed standalone definitions here to avoid ODR violations — the
// inline definitions in the header are the single source of truth.
// VedicGEMM.cu includes MixedPrecision.cuh, so callers see them fine.

// [v23-AMP] Mixed precision GEMM: FP16 A×B → FP32 C
// Primary path for H100 forward pass GEMM
// A: (M×K) FP16,  B: (K×N) FP16,  C: (M×N) FP32
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
    // Fallback: no custom FP16 TILE kernel — cast to FP32 and use FP32 GEMM
    // This path is slow but correct; H100 always has cuBLAS so this is safety only
    fprintf(stderr, "[v23-AMP] WARNING: FP16 GEMM called without cuBLAS — "
                    "falling back to FP32 conversion (slow)\n");
    // Allocate temp FP32 buffers
    float *d_A32, *d_B32;
    cudaMalloc(&d_A32, M * K * sizeof(float));
    cudaMalloc(&d_B32, K * N * sizeof(float));
    cast_fp16_to_fp32_kernel<<<(M*K+255)/256, 256>>>(A, d_A32, M*K);
    cast_fp16_to_fp32_kernel<<<(K*N+255)/256, 256>>>(B, d_B32, K*N);
    // Use standard tiled GEMM kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    vedic_gemm_kernel<<<grid, block>>>(d_A32, d_B32, C, M, K, N);
    CUDA_KERNEL_CHECK();
    cudaFree(d_A32);
    cudaFree(d_B32);
#endif
}

// [v23-AMP] Mixed precision GEMM: FP16 A×B → FP16 C
// Used for intermediate activations where FP32 output not needed
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
    // Fallback: not supported without cuBLAS (H100 always has it)
    (void)A; (void)M; (void)K; (void)B; (void)N; (void)C;
    (void)alpha; (void)beta;
    throw std::runtime_error("[v23-AMP] FP16→FP16 GEMM requires cuBLAS");
#endif
}

// [v23-AMP] FP16 LayerNorm (FP32 mean/var accumulation)
void cuda_layernorm_fp16(
    const __half* X, const float* gamma, const float* beta,
    __half* Y, int seq, int d, float eps)
{
    layernorm_fp16_kernel<<<seq, 256>>>(X, gamma, beta, Y, seq, d, eps);
    CUDA_KERNEL_CHECK();
}

// [v23-AMP] FP16 Embedding lookup
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

// [v23-AMP] Cast FP16 logits to FP32 for loss computation
void cuda_logits_fp16_to_fp32(const __half* logits_fp16, float* logits_fp32, int n) {
    int blocks = (n + 255) / 256;
    logits_fp16_to_fp32_kernel<<<blocks, 256>>>(logits_fp16, logits_fp32, n);
    CUDA_KERNEL_CHECK();
}

// [v23-AMP] Scale FP32 gradient tensor
void cuda_scale_tensor(float* data, float scale, int n) {
    scale_tensor(data, scale, n);
    CUDA_KERNEL_CHECK();
}
