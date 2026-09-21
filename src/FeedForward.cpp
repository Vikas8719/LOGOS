// ============================================================
//  LOGOS — FeedForward.cpp
//  Position-wise Feed-Forward Network
//  FFN(x) = GELU(x × W1 + b1) × W2 + b2
//  Both matrix mults → Vedic GEMM
//  Hidden dim = 4 × d_model (standard transformer ratio)
// ============================================================
#include "../include/Tensor.hpp"
#include "VedicGEMM.cpp"
#include <cmath>

// GELU activation (smoother than ReLU — GPT-2 uses this)
// Approx: x * 0.5 * (1 + tanh(0.7978 * (x + 0.044715*x³)))
static float gelu(float x) {
    float c = 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + c)));
}

struct FeedForward {
    Tensor W1, b1, W2, b2;
    int d_model, d_ff;

    FeedForward(int d_model_, int d_ff_ = 0)
        : d_model(d_model_),
          d_ff(d_ff_ > 0 ? d_ff_ : 4 * d_model_),
          W1({d_model_, d_ff_ > 0 ? d_ff_ : 4*d_model_}),
          b1({1, d_ff_ > 0 ? d_ff_ : 4*d_model_}),
          W2({d_ff_ > 0 ? d_ff_ : 4*d_model_, d_model_}),
          b2({1, d_model_})
    {
        float scale = std::sqrt(2.0f / d_model_);
        W1.fill_random(-scale, scale);
        W2.fill_random(-scale, scale);
        b1.zero(); b2.zero();
    }

    // X: (seq, d_model) → (seq, d_model)
    Tensor forward(const Tensor& X) {
        // Layer 1: Vedic GEMM + bias + GELU
        Tensor H = vedic_gemm_bias(X, W1, b1);   // (seq, d_ff)
        for (float& v : H.data) v = gelu(v);

        // Layer 2: Vedic GEMM + bias
        return vedic_gemm_bias(H, W2, b2);        // (seq, d_model)
    }

    std::vector<Tensor*> parameters() { return {&W1, &b1, &W2, &b2}; }
};
