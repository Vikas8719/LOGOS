#pragma once
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

// Boltzmann softmax (numerically stable, log-sum-exp trick)
inline Tensor boltzmann_softmax(const Tensor& scores, float temperature) {
    int seq = scores.rows(), len = scores.cols();
    Tensor probs(scores.shape);
    for (int i = 0; i < seq; ++i) {
        float max_val = scores.at(i, 0);
        for (int j = 1; j < len; ++j)
            max_val = std::max(max_val, scores.at(i,j));
        float sum = 0.0f;
        for (int j = 0; j < len; ++j) {
            float e = std::exp((scores.at(i,j) - max_val) / temperature);
            probs.at(i,j) = e;
            sum += e;
        }
        for (int j = 0; j < len; ++j)
            probs.at(i,j) /= (sum + 1e-9f);
    }
    return probs;
}

inline Tensor causal_mask(int seq_len) {
    Tensor mask({seq_len, seq_len}, 0.0f);
    for (int i = 0; i < seq_len; ++i)
        for (int j = i+1; j < seq_len; ++j)
            mask.at(i,j) = -1e9f;
    return mask;
}

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
        Tensor K_T = K.transpose();
        Tensor scores = vedic_gemm(Q, K_T);
        Tensor mask = causal_mask(X.rows());
        scores += mask;
        Tensor attn_weights = boltzmann_softmax(scores, temperature);
        Tensor output = vedic_gemm(attn_weights, V);
        return vedic_gemm(output, W_O);
    }

    std::vector<Tensor*> parameters() { return {&W_Q, &W_K, &W_V, &W_O}; }
};

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
