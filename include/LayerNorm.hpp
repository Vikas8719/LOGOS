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

// ── Reynolds Batch Normalization ──────────────────────────────
// Physics: fluid flow transitions from laminar → turbulent as
//   Reynolds number Re = ρ·v·L / μ  (inertia / viscosity) increases.
//
//   In neural networks:
//     Re_eff = ||x||_rms / (σ_batch + ε)   [signal energy / noise floor]
//     Low  Re (< Re_crit): laminar regime  → standard layer norm (stable)
//     High Re (≥ Re_crit): turbulent regime → attenuated normalisation
//                           (prevent instability from large activations)
//
//   The blend uses a smooth sigmoid transition (not a hard threshold):
//     weight_LN = σ(-k * (Re - Re_crit))   [approaches 1 when Re << Re_crit]
//     weight_BN = 1 - weight_LN             [approaches 1 when Re >> Re_crit]
//
//   Dual normalisation:
//     LN output  = LayerNorm(X)          — stable, per-token
//     BN output  = BroadcastNorm(X)      — running stats, per-feature (like BatchNorm)
//     Y = weight_LN * LN + weight_BN * BN + γ * ... + β
//
//   Annealing: Re_crit decreases over training (more stability early, less later)
//     This mirrors real turbulence: early training = laminar (needs stability),
//     late training = allow turbulence (enables exploration of sharp minima).
//
//   The per-step Reynolds number is computed from the incoming activations,
//   making this a completely data-driven normalisation regime selector.
struct ReynoldsBatchNorm {
    Tensor gamma, beta;
    int    d_model;
    float  eps      = 1e-5f;
    float  Re_crit  = 1.0f;   // Reynolds criticality threshold (annealed externally)
    float  k_slope  = 5.0f;   // sigmoid steepness for laminar→turbulent transition

    // Running statistics for BN path (EMA, no batch needed)
    std::vector<float> running_mean;
    std::vector<float> running_var;
    float              ema_decay = 0.99f;
    bool               initialized = false;

    ReynoldsBatchNorm(int d_model_)
        : d_model(d_model_),
          gamma({1, d_model_}, 1.0f),
          beta ({1, d_model_}, 0.0f),
          running_mean(d_model_, 0.0f),
          running_var (d_model_, 1.0f)
    {}

    // Compute effective Reynolds number for one token sequence
    // Re_eff = RMS(x) / (mean_std + ε)
    // RMS(x): energy of the activation
    // mean_std: average standard deviation across features (noise floor)
    float reynolds_number(const Tensor& X) const {
        int seq = X.rows(), dim = X.cols();
        float rms_sum = 0.0f, std_sum = 0.0f;

        for (int i = 0; i < seq; ++i) {
            // RMS of this token
            float sq = 0.0f;
            for (int j = 0; j < dim; ++j) sq += X.at(i,j) * X.at(i,j);
            rms_sum += std::sqrt(sq / (dim + 1e-8f));

            // Standard deviation (Welford)
            float mean = 0.0f, M2 = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float delta = X.at(i,j) - mean;
                mean += delta / (j+1);
                M2   += delta * (X.at(i,j) - mean);
            }
            std_sum += std::sqrt(M2 / dim + eps);
        }

        float rms  = rms_sum / (seq + 1e-8f);
        float std_ = std_sum / (seq + 1e-8f);
        return rms / (std_ + eps);
    }

    // Smooth sigmoid blend weight for LN regime
    // Returns ≈1 when Re << Re_crit (laminar), ≈0 when Re >> Re_crit (turbulent)
    float laminar_weight(float Re) const {
        float x = -k_slope * (Re - Re_crit);
        // Numerically stable sigmoid
        if (x >= 0) {
            float e = std::exp(-x);
            return 1.0f / (1.0f + e);
        } else {
            float e = std::exp(x);
            return e / (1.0f + e);
        }
    }

    // Main forward: Reynolds-adaptive blend of LayerNorm and running BN
    // X: (seq, d_model) → Y: (seq, d_model)
    // is_training: if true, update running stats (EMA)
    Tensor forward(const Tensor& X, bool is_training = true) {
        int seq = X.rows(), dim = X.cols();
        Tensor Y(X.shape);

        // ── Compute Reynolds number ───────────────────────────
        float Re = reynolds_number(X);
        float w_LN = laminar_weight(Re);      // laminar weight
        float w_BN = 1.0f - w_LN;            // turbulent weight

        // ── Path A: Layer Norm (per-token) ────────────────────
        Tensor LN_out(X.shape);
        for (int i = 0; i < seq; ++i) {
            float mean = 0.0f, M2 = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float delta = X.at(i,j) - mean;
                mean += delta / (j+1);
                M2   += delta * (X.at(i,j) - mean);
            }
            float inv_std = 1.0f / std::sqrt(M2/dim + eps);
            for (int j = 0; j < dim; ++j)
                LN_out.at(i,j) = (X.at(i,j) - mean) * inv_std;
        }

        // ── Path B: Broadcast Norm (per-feature, running stats) ─
        // Compute per-feature mean and variance from this batch
        std::vector<float> batch_mean(dim, 0.0f), batch_var(dim, 0.0f);
        for (int j = 0; j < dim; ++j) {
            float m = 0.0f, v = 0.0f;
            for (int i = 0; i < seq; ++i) m += X.at(i,j);
            m /= (seq + 1e-8f);
            for (int i = 0; i < seq; ++i) v += (X.at(i,j)-m)*(X.at(i,j)-m);
            v /= (seq + 1e-8f);
            batch_mean[j] = m;
            batch_var[j]  = v;
        }

        // EMA update of running stats (only in training)
        if (is_training) {
            for (int j = 0; j < dim; ++j) {
                running_mean[j] = ema_decay * running_mean[j] + (1.0f-ema_decay) * batch_mean[j];
                running_var[j]  = ema_decay * running_var[j]  + (1.0f-ema_decay) * batch_var[j];
            }
            initialized = true;
        }

        // Use running stats if available, else batch stats
        const auto& use_mean = (initialized && !is_training) ? running_mean : batch_mean;
        const auto& use_var  = (initialized && !is_training) ? running_var  : batch_var;

        Tensor BN_out(X.shape);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < dim; ++j)
                BN_out.at(i,j) = (X.at(i,j) - use_mean[j])
                                / std::sqrt(use_var[j] + eps);

        // ── Reynolds blend + affine transform ─────────────────
        // Y = (w_LN * LN_out + w_BN * BN_out) * γ + β
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < dim; ++j)
                Y.at(i,j) = gamma[j] * (w_LN * LN_out.at(i,j)
                                       + w_BN * BN_out.at(i,j))
                           + beta[j];

        return Y;
    }

    // Anneal Re_crit over training (call from optimizer loop)
    // Early: high Re_crit = more laminar = more stable
    // Late:  low Re_crit  = allow turbulence = more exploration
    void anneal_reynolds(int current_step, int total_steps,
                         float Re_start = 2.0f, float Re_end = 0.5f) {
        float ratio = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        Re_crit = Re_end + (Re_start - Re_end) * cos_v;
    }

    std::vector<Tensor*> parameters() { return {&gamma, &beta}; }
};
