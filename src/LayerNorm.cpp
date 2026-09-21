// ============================================================
//  LOGOS — LayerNorm.cpp
//  Layer Normalization (Standard Stats — Modern Math)
//  Formula: y = γ × (x - μ) / (σ + ε) + β
// ============================================================
#ifndef LOGOS_LAYER_NORM_CPP
#define LOGOS_LAYER_NORM_CPP

#include "../include/Tensor.hpp"
#include <cmath>

struct LayerNorm {
    Tensor gamma, beta;
    int d_model;
    float eps = 1e-5f;

    LayerNorm(int d_model_) : d_model(d_model_),
        gamma({1, d_model_}, 1.0f),   // Init to 1
        beta ({1, d_model_}, 0.0f)    // Init to 0
    {}

    // X: (seq_len, d_model) → (seq_len, d_model)
    Tensor forward(const Tensor& X) const {
        int seq = X.rows(), dim = X.cols();
        Tensor Y(X.shape);

        for (int i = 0; i < seq; ++i) {
            // Mean
            float mean = 0.0f;
            for (int j = 0; j < dim; ++j) mean += X.at(i,j);
            mean /= dim;

            // Variance
            float var = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float d = X.at(i,j) - mean;
                var += d * d;
            }
            var /= dim;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize + scale + shift
            for (int j = 0; j < dim; ++j)
                Y.at(i,j) = gamma[j] * (X.at(i,j) - mean) * inv_std + beta[j];
        }
        return Y;
    }

    std::vector<Tensor*> parameters() { return {&gamma, &beta}; }
};

#endif
