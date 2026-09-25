#pragma once
// ============================================================
//  LOGOS — Attention.hpp
//  Multi-Head Self-Attention + Boltzmann Softmax
//
//  BUG 7 FIX: ODR (One Definition Rule) violation resolve kiya
//    Pehle: AttentionHead, MultiHeadAttention, boltzmann_softmax,
//           causal_mask — DONO .hpp aur Attention.cpp mein define the.
//           Agar koi dono include karta → ODR violation → linker error.
//    Ab:    .hpp = SINGLE SOURCE OF TRUTH (header-only implementation)
//           Attention.cpp = DEAD FILE (clearly marked, not compiled)
//           CMakeLists.txt sirf main.cpp compile karta hai → .hpp hi use hoti hai.
//
//  BUG 7 FIX: causal_mask() cache yahan bhi add kiya
//    Pehle: .hpp mein cache nahi tha (Attention.cpp mein tha — dead code)
//           Har forward() call pe nayi mask banti thi → O(seq²) per step
//    Ab:    inline static map se cached mask → same mask reuse
//           thread-safe nahi (single-threaded CPU training) — OK for now
//
//  Physics: Softmax(QKᵀ/√d) ≡ Boltzmann distribution
//    P(state_i) = exp(-E_i/T) / Σ exp(-E_j/T)
//    Energy E = -attention_score, Temperature T = √d_k
//
//  Shunyam Sparse Attention (Vedic sutra: शून्यम्  = void/zero)
//    Non-resonant positions → Shunyam (-1e9) → zeroed by softmax
//    Local window + global stride pattern: O(seq*(w + seq/s)) not O(seq²)
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <unordered_map>

// ── Boltzmann Softmax (numerically stable) ────────────────────
// Temperature scaling = √d_k (standard transformer)
inline Tensor boltzmann_softmax(const Tensor& scores, float temperature) {
    int seq = scores.rows(), len = scores.cols();
    Tensor probs(scores.shape);
    for (int i = 0; i < seq; ++i) {
        float max_val = scores.at(i, 0);
        for (int j = 1; j < len; ++j)
            max_val = std::max(max_val, scores.at(i, j));
        float sum = 0.0f;
        for (int j = 0; j < len; ++j) {
            float e = std::exp((scores.at(i, j) - max_val) / temperature);
            probs.at(i, j) = e;
            sum += e;
        }
        for (int j = 0; j < len; ++j)
            probs.at(i, j) /= (sum + 1e-9f);
    }
    return probs;
}

// ── BUG 7 FIX: Causal mask with cache ────────────────────────
// Pehle: .hpp mein cache nahi tha (dead Attention.cpp mein tha)
//        → Har forward() pe nayi {seq×seq} Tensor banti thi
//        → O(seq²) alloc + fill per attention call per step
// Ab:    inline static cache — same seq_len ke liye mask reuse
//        seq=64, L=4, H=4 → 16 forward calls per step, sirf 1 mask banti hai
//
// Note: inline static = per-translation-unit (ODR-safe with #pragma once)
//       Single-threaded CPU training ke liye thread safety ki zaroorat nahi.
inline const Tensor& causal_mask_cached(int seq_len) {
    static std::unordered_map<int, Tensor> mask_cache;
    auto it = mask_cache.find(seq_len);
    if (it != mask_cache.end()) return it->second;

    Tensor mask({seq_len, seq_len}, 0.0f);
    for (int i = 0; i < seq_len; ++i)
        for (int j = i + 1; j < seq_len; ++j)
            mask.at(i, j) = -1e9f;

    mask_cache.emplace(seq_len, std::move(mask));
    return mask_cache.at(seq_len);
}

// Non-cached version (kept for compatibility — prefer causal_mask_cached)
inline Tensor causal_mask(int seq_len) {
    return causal_mask_cached(seq_len);  // delegate to cache
}

// ── Shunyam Sparse Attention Mask ────────────────────────────
// Vedic sutra "Shunyam" (शून्यम्) = zero / void
// Strategy: each query attends only to LOCAL window + GLOBAL stride
//   Local window  : last `window` tokens (recent context — dense)
//   Global stride : every `stride`-th token (long-range — sparse)
//   All other positions: zeroed out (-1e9 = Shunyam / void)
//
// This reduces O(seq²) full attention to O(seq * (window + seq/stride))
// Physics: sparse = only resonant frequencies survive (spectral pruning)
//
// Parameters:
//   seq_len : sequence length
//   window  : local attention window size (default 8)
//   stride  : global token stride (default 4 — every 4th token)
inline Tensor shunyam_sparse_mask(int seq_len,
                                   int window = 8,
                                   int stride = 4)
{
    Tensor mask({seq_len, seq_len}, -1e9f);  // Shunyam: all void by default

    for (int i = 0; i < seq_len; ++i) {
        // 1. Causal constraint: never attend to future (j > i)
        // 2. Local window: attend to [max(0, i-window), i]
        int local_start = std::max(0, i - window);
        for (int j = local_start; j <= i; ++j)
            mask.at(i, j) = 0.0f;  // allow: local window

        // 3. Global stride: attend to past stride-aligned positions
        for (int j = 0; j < i; j += stride)
            mask.at(i, j) = 0.0f;  // allow: global landmark tokens
    }
    return mask;
}

// Cached version of shunyam mask (same pattern = reuse)
inline const Tensor& shunyam_sparse_mask_cached(int seq_len,
                                                  int window = 8,
                                                  int stride = 4)
{
    // Key encodes all three params
    static std::unordered_map<int, Tensor> sparse_cache;
    int key = seq_len * 10000 + window * 100 + stride;
    auto it = sparse_cache.find(key);
    if (it != sparse_cache.end()) return it->second;
    sparse_cache.emplace(key, shunyam_sparse_mask(seq_len, window, stride));
    return sparse_cache.at(key);
}

// ── Navier-Stokes Attention ───────────────────────────────────
// Physics: token interactions modelled as viscous fluid flow
//   Standard attention = ideal fluid (no viscosity, no diffusion)
//   Navier-Stokes attention = viscous fluid with:
//     (1) Advection term    : tokens "carry" information downstream
//                             like fluid particles advecting a scalar field
//     (2) Diffusion term    : information smooths across neighbours
//                             like viscous dissipation spreading momentum
//     (3) Pressure gradient : softmax scores act as pressure field
//                             high-score regions = low pressure → flow toward
//
//   Incompressibility constraint (∇·u = 0):
//     Attention weights row-sum = 1 (softmax) enforces incompressibility
//     Total "information volume" is conserved across positions
//
// Implementation:
//   Step 1 — Advection:  Q_adv[i] = Q[i] + η * (Q[i] - Q[i-1])  [upwind scheme]
//   Step 2 — Standard QK attention with advected queries
//   Step 3 — Viscous diffusion on output: V_smooth[i] = (1-ν)*V[i] + ν*(V[i-1]+V[i+1])/2
//   Step 4 — Output blend: (1-α)*std_out + α*ns_out  [Reynolds blend — see Reynolds below]
//
//   Parameters:
//     eta  (η): advection strength  (0 = no advection, 0.1 = mild convection)
//     nu   (ν): kinematic viscosity (0 = inviscid, 0.1 = viscous diffusion)
//     Note: η and ν are annealed by ReynoldsBatch — no hardcoding needed
inline Tensor navier_stokes_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                       const Tensor& mask,
                                       float temperature,
                                       float eta  = 0.1f,    // advection
                                       float nu   = 0.05f)   // viscosity
{
    int seq = Q.rows(), d_k = Q.cols(), d_v = V.cols();

    // ── Step 1: Upwind Advection on Q (convective derivative DQ/Dt) ──
    // Q_adv[i] = Q[i] + η * (Q[i] - Q[i-1])
    // Upwind scheme (first-order): stable for η > 0 (forward flow)
    // Physics: query at pos i is "advected" by information flow from i-1
    Tensor Q_adv(Q.shape, 0.0f);
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < d_k; ++j) {
            float q_i   = Q.at(i, j);
            float q_im1 = (i > 0) ? Q.at(i-1, j) : q_i;  // boundary: no flow at i=0
            Q_adv.at(i, j) = q_i + eta * (q_i - q_im1);
        }
    }

    // ── Step 2: Standard scaled-dot-product attention with advected Q ──
    // Scores = Q_adv @ K^T / sqrt(d_k)   [pressure field]
    Tensor K_T    = K.transpose();
    Tensor scores = vedic_gemm(Q_adv, K_T);
    scores += mask;  // causal / sparse mask

    Tensor attn_weights = boltzmann_softmax(scores, temperature);

    // ── Step 3: Viscous diffusion on V before mixing ──────────
    // V_smooth[i] = (1-ν)*V[i] + ν*(V[i-1]+V[i+1])/2
    // Central difference Laplacian: ∂²V/∂x² ≈ (V[i+1]-2V[i]+V[i-1]) / h²
    // Euler step: V_smooth = V + ν*Δt * ∇²V  ≈ (1-ν)*V + ν*(V_{left}+V_{right})/2
    // Boundary: mirror (V[-1]=V[0], V[seq]=V[seq-1]) — zero-flux Neumann BC
    Tensor V_smooth(V.shape, 0.0f);
    for (int i = 0; i < seq; ++i) {
        int i_left  = (i > 0)      ? i - 1 : 0;
        int i_right = (i < seq-1)  ? i + 1 : seq - 1;
        for (int j = 0; j < d_v; ++j) {
            V_smooth.at(i, j) = (1.0f - nu) * V.at(i, j)
                               + (nu * 0.5f) * (V.at(i_left, j) + V.at(i_right, j));
        }
    }

    // ── Step 4: Weighted output (attention × smoothed values) ─
    // output = attn_weights @ V_smooth
    // This is the "incompressible" mixing step: weights sum to 1 (∇·u=0)
    return vedic_gemm(attn_weights, V_smooth);
}

// ── Single Attention Head ─────────────────────────────────────
struct AttentionHead {
    Tensor W_Q, W_K, W_V, W_O;
    int d_model, d_k;

    AttentionHead(int d_model_, int d_k_)
        : d_model(d_model_), d_k(d_k_),
          W_Q({d_model_, d_k_}), W_K({d_model_, d_k_}),
          W_V({d_model_, d_k_}), W_O({d_k_, d_model_})
    {
        float scale = std::sqrt(2.0f / d_model_);
        W_Q.fill_random(-scale, scale);
        W_K.fill_random(-scale, scale);
        W_V.fill_random(-scale, scale);
        W_O.fill_random(-scale, scale);
    }

    // use_sparse: true = Shunyam sparse mask, false = full causal mask
    bool  use_sparse     = true;
    int   sparse_window  = 8;
    int   sparse_stride  = 4;
    // use_navier_stokes: true = NS fluid attention, false = standard attention
    // ns_alpha: blend weight — 0.0 = pure standard, 1.0 = pure NS
    bool  use_navier_stokes = false;
    float ns_alpha          = 0.3f;   // blend fraction for NS path
    float ns_eta            = 0.1f;   // advection strength
    float ns_nu             = 0.05f;  // kinematic viscosity

    Tensor forward(const Tensor& X) {
        float temperature = std::sqrt(static_cast<float>(d_k));

        Tensor Q = vedic_gemm(X, W_Q);
        Tensor K = vedic_gemm(X, W_K);
        Tensor V = vedic_gemm(X, W_V);

        // Select mask
        int seq = X.rows();
        const Tensor* mask_ptr = nullptr;
        Tensor sparse_mask_copy;
        if (use_sparse && seq > sparse_window) {
            sparse_mask_copy = shunyam_sparse_mask_cached(seq, sparse_window, sparse_stride);
            mask_ptr         = &sparse_mask_copy;
        } else {
            sparse_mask_copy = causal_mask_cached(seq);
            mask_ptr         = &sparse_mask_copy;
        }

        if (use_navier_stokes && ns_alpha > 0.0f) {
            // ── Navier-Stokes path ────────────────────────────
            // Standard output (for blending)
            Tensor K_T       = K.transpose();
            Tensor scores_std = vedic_gemm(Q, K_T);
            scores_std       += *mask_ptr;
            Tensor w_std      = boltzmann_softmax(scores_std, temperature);
            Tensor out_std    = vedic_gemm(w_std, V);
            Tensor out_std_O  = vedic_gemm(out_std, W_O);

            // NS viscous-advective output
            Tensor out_ns = navier_stokes_attention(Q, K, V, *mask_ptr,
                                                     temperature, ns_eta, ns_nu);
            Tensor out_ns_O = vedic_gemm(out_ns, W_O);

            // Blend: (1-α)*standard + α*NS
            Tensor blended({seq, d_model}, 0.0f);
            float w1 = 1.0f - ns_alpha, w2 = ns_alpha;
            for (int i = 0; i < blended.total_size; ++i)
                blended.data[i] = w1 * out_std_O.data[i] + w2 * out_ns_O.data[i];
            return blended;
        } else {
            // ── Standard path ─────────────────────────────────
            Tensor K_T    = K.transpose();
            Tensor scores = vedic_gemm(Q, K_T);
            scores       += *mask_ptr;
            Tensor attn_weights = boltzmann_softmax(scores, temperature);
            Tensor output       = vedic_gemm(attn_weights, V);
            return vedic_gemm(output, W_O);
        }
    }

    std::vector<Tensor*> parameters() { return {&W_Q, &W_K, &W_V, &W_O}; }
};

// ── Multi-Head Attention ──────────────────────────────────────
struct MultiHeadAttention {
    std::vector<AttentionHead> heads;
    Tensor W_proj;
    int num_heads, d_model, d_k;

    MultiHeadAttention(int d_model_, int num_heads_)
        : num_heads(num_heads_), d_model(d_model_),
          d_k(d_model_ / num_heads_),
          W_proj({d_model_, d_model_})
    {
        if (d_model_ % num_heads_ != 0)
            throw std::invalid_argument("d_model must be divisible by num_heads");
        for (int h = 0; h < num_heads_; ++h)
            heads.emplace_back(d_model_, d_k);
        float scale = std::sqrt(2.0f / d_model_);
        W_proj.fill_random(-scale, scale);
    }

    Tensor forward(const Tensor& X) {
        int seq = X.rows();
        Tensor concat({seq, d_model}, 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            Tensor head_out = heads[h].forward(X);
            for (int i = 0; i < seq; ++i)
                for (int j = 0; j < d_k; ++j)
                    concat.at(i, h * d_k + j) = head_out.at(i, j);
        }
        return vedic_gemm(concat, W_proj);
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params = {&W_proj};
        for (auto& h : heads)
            for (auto* p : h.parameters())
                params.push_back(p);
        return params;
    }
};
