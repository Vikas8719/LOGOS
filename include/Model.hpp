#pragma once
#include "Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct ModelConfig {
    int vocab_size  = 4096;
    int d_model     = 128;
    int num_heads   = 4;
    int num_layers  = 4;
    int max_seq_len = 128;
};

struct HeadWeights {
    Tensor W_Q, W_K, W_V, W_O;

    HeadWeights(int d_model, int d_k)
        : W_Q({d_model, d_k}), W_K({d_model, d_k}),
          W_V({d_model, d_k}), W_O({d_k, d_model})
    {
        const float scale = std::sqrt(2.0f / d_model);
        W_Q.fill_random(-scale, scale);
        W_K.fill_random(-scale, scale);
        W_V.fill_random(-scale, scale);
        W_O.fill_random(-scale, scale);
    }
};

struct AttentionWeights {
    std::vector<HeadWeights> heads;
    Tensor W_proj;
    int num_heads, d_model, d_k;

    AttentionWeights(int d_model_, int num_heads_)
        : W_proj({d_model_, d_model_}),
          num_heads(num_heads_), d_model(d_model_), d_k(d_model_ / num_heads_)
    {
        if (d_model_ % num_heads_ != 0)
            throw std::invalid_argument("d_model must be divisible by num_heads");
        for (int h = 0; h < num_heads_; ++h) heads.emplace_back(d_model_, d_k);
        W_proj.fill_random(-std::sqrt(2.0f / d_model_), std::sqrt(2.0f / d_model_));
    }
};

struct FeedForwardWeights {
    Tensor W1, b1, W2, b2;
    int d_model, d_ff;

    explicit FeedForwardWeights(int d_model_)
        : W1({d_model_, 4 * d_model_}), b1({1, 4 * d_model_}),
          W2({4 * d_model_, d_model_}), b2({1, d_model_}),
          d_model(d_model_), d_ff(4 * d_model_)
    {
        const float scale = std::sqrt(2.0f / d_model_);
        W1.fill_random(-scale, scale);
        W2.fill_random(-scale, scale);
    }
};

struct NormWeights {
    Tensor gamma, beta;
    std::vector<float> running_mean, running_var;
    bool initialized = false;

    explicit NormWeights(int d)
        : gamma({1, d}, 1.0f), beta({1, d}, 0.0f),
          running_mean(d, 0.0f), running_var(d, 1.0f) {}
};

struct BlockWeights {
    AttentionWeights   mha;
    FeedForwardWeights ffn;
    NormWeights        ln1, ln2;

    BlockWeights(int d_model, int num_heads)
        : mha(d_model, num_heads), ffn(d_model), ln1(d_model), ln2(d_model) {}

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> p = {&mha.W_proj};
        for (auto& h : mha.heads) {
            p.push_back(&h.W_Q); p.push_back(&h.W_K);
            p.push_back(&h.W_V); p.push_back(&h.W_O);
        }
        p.push_back(&ffn.W1); p.push_back(&ffn.b1);
        p.push_back(&ffn.W2); p.push_back(&ffn.b2);
        p.push_back(&ln1.gamma); p.push_back(&ln1.beta);
        p.push_back(&ln2.gamma); p.push_back(&ln2.beta);
        return p;
    }
};

// Host-side weights for checkpoint I/O; training and inference run on the GPU (ModelGPU).
class LOGOSModel {
public:
    ModelConfig cfg;
    Tensor embedding, lm_head;
    Tensor pos_embedding;  // stored in the checkpoint's position slot; not part of parameters()
    NormWeights final_ln;  // final LayerNorm before lm_head; saved as checkpoint footer; not part of parameters()
    std::vector<BlockWeights> layers;

    explicit LOGOSModel(const ModelConfig& config = {})
        : cfg(config),
          embedding({cfg.vocab_size, cfg.d_model}),
          lm_head({cfg.d_model, cfg.vocab_size}),
          pos_embedding({std::max(1, cfg.max_seq_len), cfg.d_model}, 0.0f),
          final_ln(cfg.d_model)
    {
        const float scale = std::sqrt(2.0f / cfg.d_model);
        embedding.fill_random(-scale, scale);
        lm_head.fill_random(-scale, scale);
        layers.reserve(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l) layers.emplace_back(cfg.d_model, cfg.num_heads);
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> p = {&embedding, &lm_head};
        for (auto& l : layers)
            for (auto* w : l.parameters()) p.push_back(w);
        return p;
    }

    long long count_parameters() const {
        const long long d = cfg.d_model, h = cfg.num_heads, dk = d / h, d_ff = 4 * d;
        long long total = embedding.total_size + lm_head.total_size;
        for (int l = 0; l < cfg.num_layers; ++l) {
            total += h * 4 * d * dk + d * d;
            total += d * d_ff + d_ff + d_ff * d + d;
            total += 4 * d;
        }
        return total;
    }
};
