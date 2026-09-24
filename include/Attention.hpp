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

    Tensor forward(const Tensor& X) {
        float temperature = std::sqrt(static_cast<float>(d_k));

        Tensor Q = vedic_gemm(X, W_Q);
        Tensor K = vedic_gemm(X, W_K);
        Tensor V = vedic_gemm(X, W_V);

        Tensor K_T    = K.transpose();
        Tensor scores = vedic_gemm(Q, K_T);

        // BUG 7 FIX: cached mask — no alloc on repeated calls
        scores += causal_mask_cached(X.rows());

        Tensor attn_weights = boltzmann_softmax(scores, temperature);
        Tensor output       = vedic_gemm(attn_weights, V);
        return vedic_gemm(output, W_O);
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
