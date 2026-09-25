#pragma once
// ============================================================
//  LOGOS — FeedForward.hpp
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include <cmath>
#include <vector>
#include <random>

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ── Feynman Dropout ───────────────────────────────────────────
// Physics: standard dropout = binary mask (neuron alive or dead)
//          Feynman dropout  = path integral over ALL partial activation levels
//
// Feynman Path Integral motivation:
//   In QFT, the propagator integrates over ALL paths between two points,
//   weighted by the phase e^{iS/ħ} where S is the action.
//   Here: instead of binary {0,1} mask (only alive/dead paths),
//   we weight each neuron by a continuous amplitude drawn from a
//   physical distribution. All paths contribute — none are fully silenced.
//
// Three regimes (selected by Feynman hbar ħ parameter):
//   ħ → 0  (classical limit):  sharp binary dropout (standard Bernoulli)
//   ħ = 1  (quantum regime):   Gaussian amplitude fluctuations (smooth dropout)
//   ħ → ∞  (free particle):   no regularisation (identity mapping)
//
// Amplitude distribution:
//   Classical path  : weight = 1 with prob (1-p), 0 with prob p
//   Quantum path    : weight ~ Beta(α, β) where α=(1-p)ħ, β=pħ
//   (Beta distribution → Bernoulli as ħ → 0)
//
// In practice: Beta(α,β) sampled via ratio of Gamma draws (Johnk's method)
//   If X ~ Gamma(α,1) and Y ~ Gamma(β,1), then X/(X+Y) ~ Beta(α,β)
//   When ħ=1 → Beta(1-p, p) (simple linear interpolation)
//   When ħ→0 → sharp Bernoulli (classical limit recovered); large ħ concentrates
//   around the mean (1-p).
//
// Inference (training=false): identity (no dropout), consistent with classical limit
//
// Scale invariance: outputs divided by (1-p) to maintain expected activation scale
//   E[weight] = α/(α+β) = (1-p)ħ / ((1-p)ħ + pħ) = (1-p)   [independent of ħ!]
//   → same expected magnitude as standard dropout, regardless of ħ
//
// Parameters:
//   p    : dropout probability (0=no dropout, 0.1 typical for transformers)
//   hbar : quantum fluctuation strength (1.0 default = balanced, 0.1 = near-classical)
struct FeynmanDropout {
    float p;       // base dropout probability
    float hbar;    // quantum fluctuation ħ (controls path integral width)
    bool  training = true;

    mutable std::mt19937 rng;
    // Gamma distribution samplers (for Beta via ratio method)
    mutable std::gamma_distribution<float> gamma_alive;   // Gamma(α=(1-p)ħ, 1)
    mutable std::gamma_distribution<float> gamma_dead;    // Gamma(β=pħ,   1)

    FeynmanDropout(float dropout_p = 0.1f, float hbar_ = 1.0f, int seed = 123)
        : p(dropout_p), hbar(hbar_), rng(seed),
          gamma_alive(std::max(1e-3f, (1.0f - dropout_p) * hbar_), 1.0f),
          gamma_dead (std::max(1e-3f,           dropout_p  * hbar_), 1.0f)
    {}

    // Sample one path-integral weight for a single neuron
    // Returns a value in [0,1] drawn from Beta((1-p)ħ, pħ)
    // Near ħ→0: approaches Bernoulli(1-p)
    // Large ħ:   concentrates around the mean (1-p)
    float sample_weight() const {
        // std::gamma_distribution loses precision for very small shape values.
        // The limiting distribution is exactly Bernoulli(1-p), so sample that
        // limit directly before the Gamma ratio becomes numerically degenerate.
        if (hbar <= 0.05f) {
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            return uniform(rng) < p ? 0.0f : 1.0f;
        }
        float x = gamma_alive(rng);   // X ~ Gamma((1-p)ħ, 1)
        float y = gamma_dead(rng);    // Y ~ Gamma(pħ,     1)
        float total = x + y;
        if (total < 1e-9f) return 1.0f - p;  // degenerate: use mean
        return x / total;                     // Beta(α,β) via ratio method
    }

    // Forward pass with Feynman path integral dropout
    // In training mode: each activation weighted by a Beta-drawn amplitude
    // In eval mode:     identity (return X unchanged)
    Tensor forward(const Tensor& X) const {
        if (!training || p <= 0.0f) return X;  // eval or p=0 → identity

        Tensor Y(X.shape);
        float  inv_keep = 1.0f / (1.0f - p + 1e-8f);  // scale for unbiased estimate

        for (int i = 0; i < X.total_size; ++i) {
            float w   = sample_weight();  // path integral amplitude ∈ [0,1]
            Y.data[i] = X.data[i] * w * inv_keep;
        }
        return Y;
    }

    // Update ħ during training (e.g., anneal from quantum → classical)
    // hbar_new → 0 makes dropout sharper (classical limit)
    void set_hbar(float hbar_new) {
        hbar = std::max(1e-3f, hbar_new);
        gamma_alive = std::gamma_distribution<float>(
            std::max(1e-3f, (1.0f-p)*hbar), 1.0f);
        gamma_dead  = std::gamma_distribution<float>(
            std::max(1e-3f,    p*hbar),     1.0f);
    }

    // Convenience: current mean weight (should ≈ 1-p regardless of ħ)
    float mean_weight() const { return 1.0f - p; }
};

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
