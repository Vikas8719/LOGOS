// ============================================================
//  LOGOS — Attention.cpp
//  Multi-Head Self-Attention + Boltzmann Softmax
//
//  Physics connection:
//  Softmax(QKᵀ/√d) ≡ Boltzmann distribution:
//    P(state_i) = exp(-E_i / T) / Σ exp(-E_j / T)
//  Where energy E = -attention_score, temperature T = √d_k
//
//  Q, K, V projections → Vedic GEMM
//  Attention weights   → Boltzmann softmax
// ============================================================

#include "../include/Tensor.hpp"
#include "VedicGEMM.cpp"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

// ── Boltzmann Softmax (numerically stable) ────────────────────
// P_i = exp((x_i - max_x) / T) / Σ exp((x_j - max_x) / T)
// Temperature T = sqrt(d_k) — standard transformer scaling
Tensor boltzmann_softmax(const Tensor& scores, float temperature) {
    assert(scores.ndim() == 2);
    int seq = scores.rows(), len = scores.cols();
    Tensor probs(scores.shape);

    for (int i = 0; i < seq; ++i) {
        // Find max for numerical stability (log-sum-exp trick)
        float max_val = scores.at(i, 0);
        for (int j = 1; j < len; ++j)
            max_val = std::max(max_val, scores.at(i,j));

        float sum = 0.0f;
        for (int j = 0; j < len; ++j) {
            float e = std::exp((scores.at(i,j) - max_val) / temperature);
            probs.at(i,j) = e;
            sum += e;
        }
        // Normalize
        for (int j = 0; j < len; ++j)
            probs.at(i,j) /= (sum + 1e-9f);
    }
    return probs;
}

// ── Causal Mask (autoregressive — future tokens block karo) ──
// -inf positions par softmax ~0 ho jaata hai
Tensor causal_mask(int seq_len) {
    Tensor mask({seq_len, seq_len}, 0.0f);
    for (int i = 0; i < seq_len; ++i)
        for (int j = i+1; j < seq_len; ++j)
            mask.at(i,j) = -1e9f;   // effective -infinity
    return mask;
}

// ── Single Attention Head ─────────────────────────────────────
struct AttentionHead {
    Tensor W_Q, W_K, W_V, W_O;   // Projection weights
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

    // Input X: (seq_len, d_model) → Output: (seq_len, d_model)
    Tensor forward(const Tensor& X) {
        float temperature = std::sqrt(static_cast<float>(d_k));

        // Q, K, V projections — Vedic GEMM
        Tensor Q = vedic_gemm(X, W_Q);   // (seq, d_k)
        Tensor K = vedic_gemm(X, W_K);   // (seq, d_k)
        Tensor V = vedic_gemm(X, W_V);   // (seq, d_k)

        // Attention scores = Q × Kᵀ — Vedic GEMM
        Tensor K_T = K.transpose();       // (d_k, seq)
        Tensor scores = vedic_gemm(Q, K_T);  // (seq, seq)

        // Add causal mask
        Tensor mask = causal_mask(X.rows());
        scores += mask;

        // Boltzmann softmax (Physics-based)
        Tensor attn_weights = boltzmann_softmax(scores, temperature);

        // Weighted sum of V — Vedic GEMM
        Tensor output = vedic_gemm(attn_weights, V);  // (seq, d_k)

        // Output projection
        return vedic_gemm(output, W_O);  // (seq, d_model)
    }

    // All weights as flat vector (for PhysicsOpt)
    std::vector<Tensor*> parameters() {
        return {&W_Q, &W_K, &W_V, &W_O};
    }
};

// ── Multi-Head Attention ──────────────────────────────────────
struct MultiHeadAttention {
    std::vector<AttentionHead> heads;
    Tensor W_proj;   // Final projection: concat heads → d_model
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

    // X: (seq_len, d_model) → (seq_len, d_model)
    Tensor forward(const Tensor& X) {
        int seq = X.rows();

        // Run each head, collect outputs
        // Concat along feature dim (seq, num_heads * d_k) = (seq, d_model)
        Tensor concat({seq, d_model}, 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            Tensor head_out = heads[h].forward(X);  // (seq, d_k)
            // Copy into correct slice of concat
            for (int i = 0; i < seq; ++i)
                for (int j = 0; j < d_k; ++j)
                    concat.at(i, h * d_k + j) = head_out.at(i, j);
        }

        // Final projection
        return vedic_gemm(concat, W_proj);   // (seq, d_model)
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params = {&W_proj};
        for (auto& h : heads)
            for (auto* p : h.parameters())
                params.push_back(p);
        return params;
    }
};
