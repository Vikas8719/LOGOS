#pragma once
// ============================================================
//  LOGOS — LayerNorm.hpp
//  Layer Normalization: y = γ × (x - μ) / (σ + ε) + β
//
//  BUG 7 FIX: ODR violation resolve kiya
//    Pehle: LayerNorm dono .hpp aur LayerNorm.cpp mein define thi
//           → duplicate definition → ODR violation risk
//    Ab:    .hpp = SINGLE SOURCE OF TRUTH
//           LayerNorm.cpp = DEAD FILE (clearly marked)
//
//  IMPROVEMENT (from LayerNorm.cpp — now live here):
//    Welford's online algorithm for mean + variance in one pass
//    Pehle .hpp: 2 separate loops (mean, then variance)
//    Ab .hpp:    single pass → same correctness, less work
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>

struct LayerNorm {
    Tensor gamma, beta;
    int d_model;
    float eps = 1e-5f;

    LayerNorm(int d_model_) : d_model(d_model_),
        gamma({1, d_model_}, 1.0f),   // Init to 1 (scale)
        beta ({1, d_model_}, 0.0f)    // Init to 0 (shift)
    {}

    // X: (seq_len, d_model) → (seq_len, d_model)
    Tensor forward(const Tensor& X) const {
        int seq = X.rows(), dim = X.cols();
        Tensor Y(X.shape);

        for (int i = 0; i < seq; ++i) {
            // BUG 7 FIX: Welford's online algorithm (from dead LayerNorm.cpp)
            // Now live in the actual used .hpp.
            // Pehle .hpp mein:
            //   loop 1: mean = sum(x)/dim
            //   loop 2: var  = sum((x-mean)²)/dim
            //   → 2×dim operations per row
            // Ab: single pass, numerically stable
            //   mean aur M2 (sum of squared deviations) saath update hote hain
            float mean = 0.0f, M2 = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float delta = X.at(i, j) - mean;
                mean += delta / (j + 1);              // running mean
                M2   += delta * (X.at(i, j) - mean); // running sum of sq deviations
            }
            float var     = M2 / dim;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize + scale (γ) + shift (β)
            for (int j = 0; j < dim; ++j)
                Y.at(i, j) = gamma[j] * (X.at(i, j) - mean) * inv_std + beta[j];
        }
        return Y;
    }

    std::vector<Tensor*> parameters() { return {&gamma, &beta}; }
};
