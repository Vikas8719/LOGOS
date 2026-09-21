#pragma once
#include "Tensor.hpp"
#include <cmath>
#include <vector>

struct LayerNorm {
    Tensor gamma, beta;
    int d_model;
    float eps = 1e-5f;

    LayerNorm(int d_model_) : d_model(d_model_),
        gamma({1, d_model_}, 1.0f),
        beta ({1, d_model_}, 0.0f)
    {}

    Tensor forward(const Tensor& X) const {
        int seq = X.rows(), dim = X.cols();
        Tensor Y(X.shape);
        for (int i = 0; i < seq; ++i) {
            float mean = 0.0f;
            for (int j = 0; j < dim; ++j) mean += X.at(i,j);
            mean /= dim;

            float var = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float d = X.at(i,j) - mean;
                var += d * d;
            }
            var /= dim;
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int j = 0; j < dim; ++j)
                Y.at(i,j) = gamma[j] * (X.at(i,j) - mean) * inv_std + beta[j];
        }
        return Y;
    }

    std::vector<Tensor*> parameters() { return {&gamma, &beta}; }
};
