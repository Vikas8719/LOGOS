#pragma once
// ============================================================
//  LOGOS — FeedForward.hpp
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include <cmath>
#include <vector>

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

struct FeedForward {
    Tensor W1, b1, W2, b2;
    int d_model, d_ff;

    FeedForward(int d_model_, int d_ff_ = 0)
        : d_model(d_model_),
          d_ff(d_ff_ > 0 ? d_ff_ : 4 * d_model_),
          W1({d_model_, d_ff_ > 0 ? d_ff_ : 4*d_model_}),
          b1({1,        d_ff_ > 0 ? d_ff_ : 4*d_model_}),
          W2({d_ff_ > 0 ? d_ff_ : 4*d_model_, d_model_}),
          b2({1, d_model_})
    {
        float scale = std::sqrt(2.0f / d_model_);
        W1.fill_random(-scale, scale);
        W2.fill_random(-scale, scale);
        b1.zero(); b2.zero();
    }

    Tensor forward(const Tensor& X) {
        Tensor H = vedic_gemm_bias(X, W1, b1);
        for (float& v : H.data) v = gelu(v);
        return vedic_gemm_bias(H, W2, b2);
    }

    std::vector<Tensor*> parameters() { return {&W1, &b1, &W2, &b2}; }
};
