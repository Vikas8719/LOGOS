#pragma once
// ============================================================
//  LOGOS — PhysicsOpt.hpp  (v9 — Hybrid SHM Optimizer)
//
//  v7 Leapfrog Langevin retained (CPU reference, GPU uses shm_hybrid_kernel)
//
//  v9 NEW: HybridSHMOptimizer — CPU reference for Kaggle notebooks
//
//  Hybrid Stochastic Hamiltonian Mechanics:
//    Same physics as GPU shm_hybrid_kernel (VedicGEMM.cu)
//    Used in main.cpp / Kaggle / unit tests
//
//  Key insight:
//    Pure Langevin:    uniform noise all training → slow late convergence
//    Pure Hamiltonian: no noise → stuck in sharp minima, poor generalization
//    Hybrid SHM:       blend shifts explore→exploit via alpha_H annealing
//                      flat minima = better generalization (Keskar et al. 2017)
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

// ── Free Energy Loss (CPU reference) ─────────────────────────
// Computes F = CE - temperature * Shannon_entropy(softmax(logits))
// Used in main.cpp training loop and Kaggle notebook
struct FreeEnergyLoss {
    // Compute F and dF/dlogit for a single token row
    // logits: (vocab,) — raw logits for one position
    // target: ground truth token index
    // temperature: current Langevin T
    // grad_out: (vocab,) — dF/dlogit (output)
    // Returns: scalar F value
    static float compute(const float* logits, int vocab,
                         int target, float temperature,
                         float* grad_out)
    {
        // Numerically stable softmax
        float max_l = logits[0];
        for (int v=1; v<vocab; ++v) max_l = std::max(max_l, logits[v]);
        float sum = 0.0f;
        std::vector<float> p(vocab);
        for (int v=0; v<vocab; ++v) {
            p[v] = std::exp(logits[v] - max_l);
            sum += p[v];
        }
        for (int v=0; v<vocab; ++v) p[v] /= (sum + 1e-9f);

        // Cross-entropy U = -log p[target]
        float U = -std::log(std::max(p[target], 1e-9f));

        // Shannon entropy S = -Σ p·log(p)
        float S = 0.0f;
        for (int v=0; v<vocab; ++v)
            if (p[v] > 1e-12f) S -= p[v] * std::log(p[v]);

        // Free energy F = U - T·S
        float F = U - temperature * S;

        // Gradient dF/dlogit[v]
        // = (p[v] - y[v])          [CE term]
        // + T·p[v]·(log(p[v]+ε)+S) [entropy term]
        if (grad_out) {
            for (int v=0; v<vocab; ++v) {
                float y_v   = (v == target) ? 1.0f : 0.0f;
                float ce_g  = p[v] - y_v;
                float H_g   = temperature * p[v] * (std::log(p[v]+1e-9f) + S);
                grad_out[v] = ce_g + H_g;
            }
        }
        return F;
    }
};

// ── Leapfrog Langevin Optimizer (CPU, header-only) ────────────
class LangevinOptimizer {
public:
    float learning_rate;
    float friction;       // γ = 0.9
    float temperature;    // current T (annealed)
    float temp_start;
    float temp_end;
    int   total_steps;
    int   current_step = 0;

    // Per-parameter velocity buffers (store v_{t+½} between steps)
    std::vector<std::vector<float>> velocity;

    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    // [v7] Constructor — same defaults as v6
    LangevinOptimizer(float lr      = 1e-4f,
                      float fric    = 0.9f,
                      float T_start = 0.05f,
                      float T_end   = 1e-5f,
                      int   steps   = 100000,
                      int   seed    = 42)
        : learning_rate(lr), friction(fric),
          temperature(T_start), temp_start(T_start),
          temp_end(T_end), total_steps(steps),
          rng(seed)
    {}

    void init(const std::vector<Tensor*>& params) {
        velocity.clear();
        velocity.reserve(params.size());
        for (auto* p : params)
            velocity.emplace_back(p->total_size, 0.0f);
    }

    // Cosine annealing: T_start → T_end
    void anneal() {
        float ratio = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        temperature = temp_end + (temp_start - temp_end) * cos_v;
    }

    // [v7 Phase 1] Leapfrog Langevin step
    // Störmer-Verlet: half-kick → full drift (completes next iter's half-kick)
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        if (params.size() != grads.size())
            throw std::invalid_argument("params/grads size mismatch");

        anneal();

        // [v7] Half-step thermal noise: √(γ·kT·lr/2)
        // OLD (v6): √(2γT·lr)  — full-step noise
        // NEW (v7): √(γT·lr/2) — half-step noise (Verlet correct)
        float noise_scale = std::sqrt(friction * temperature * learning_rate * 0.5f);

        for (int pi=0; pi<(int)params.size(); ++pi) {
            Tensor* W       = params[pi];
            const Tensor* G = grads[pi];
            auto& vel       = velocity[pi];

            for (int i=0; i<W->total_size; ++i) {
                float grad = G->data[i];
                if (std::isnan(grad) || std::isinf(grad)) grad = 0.0f;

                float thermal = noise_scale * noise_dist(rng);

                // ── Leapfrog half-kick ────────────────────────
                // v_{t+½} = γ·v_t - (lr/2)·∇L + noise
                // Note: (lr/2) not lr — half-step gradient application
                // The other half comes from the NEXT step's beginning
                // (implicit in the Störmer-Verlet cycle)
                float v_half = friction * vel[i]
                             - (learning_rate * 0.5f) * grad
                             + thermal;

                // ── Full drift ────────────────────────────────
                // W_{t+1} = W_t + lr · v_{t+½}
                // (lr not lr/2 — full step in position space)
                float w_new = W->data[i] + learning_rate * v_half;

                if (std::isnan(w_new) || std::isinf(w_new)) w_new = W->data[i];

                vel[i]       = v_half;  // store for next step
                W->data[i]   = w_new;
            }
        }
        ++current_step;
    }

    // Compute current free energy loss (for logging in main.cpp)
    // Returns F = CE - T*S averaged over seq
    float compute_free_energy_loss(
        const std::vector<std::vector<float>>& all_logits,  // [seq][vocab]
        const std::vector<int>& targets) const
    {
        float total_F = 0.0f; int cnt = 0;
        int vocab = (int)all_logits[0].size();
        std::vector<float> tmp_grad(vocab);
        for (int i=0; i<(int)targets.size(); ++i) {
            float F = FreeEnergyLoss::compute(
                all_logits[i].data(), vocab,
                targets[i], temperature,
                tmp_grad.data());
            if (std::isfinite(F)) { total_F += F; ++cnt; }
        }
        return cnt>0 ? total_F/cnt : 0.0f;
    }

    void log_state() const {
        std::cout << "LangevinOpt [Leapfrog v7]"
                  << " | step=" << current_step
                  << " | T=" << std::scientific << temperature
                  << " | lr=" << learning_rate << "\n";
    }
};

// ── Hybrid SHM Optimizer (CPU reference for v9) ───────────────
// Mirrors shm_hybrid_kernel in VedicGEMM.cu exactly.
// Use in main.cpp CPU training path or Kaggle notebooks.
//
// Update equations (per parameter per step):
//   noise     = noise_scale * η_i             [FDT thermal fluctuation]
//   v_{t+½}   = mom_decay * v_t               [Hamiltonian momentum carry]
//             - (lr * α_H / 2) * ∇L           [H gradient half-kick]
//             - (lr * α_L / 2) * γ * v_t      [L friction half-kick]
//             + noise                          [L thermal noise]
//   W_{t+1}   = W_t + lr * v_{t+½}           [position full-step]
//
// Annealing (cosine, both T and alpha_H simultaneously):
//   Step 0%:   T=T_start, α_H=0.3 → Langevin-dominant (explore)
//   Step 100%: T=T_end,   α_H=0.9 → Hamiltonian-dominant (exploit)
class HybridSHMOptimizer {
public:
    float learning_rate;
    float friction;       // γ — Langevin friction (low, ≈0.1)
    float mom_decay;      // β — Hamiltonian momentum (high, ≈0.9)
    float temperature, temp_start, temp_end;
    float alpha_H_start, alpha_H_end;
    int   total_steps;
    int   current_step = 0;

    // Current annealed state
    float alpha_H = 0.3f;
    float alpha_L = 0.7f;

    std::vector<std::vector<float>> velocity;
    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    HybridSHMOptimizer(float lr       = 1e-4f,
                       float fric     = 0.1f,
                       float mom_d    = 0.9f,
                       float T_start  = 0.05f,
                       float T_end    = 1e-5f,
                       float aH_start = 0.3f,
                       float aH_end   = 0.9f,
                       int   steps    = 100000,
                       int   seed     = 42)
        : learning_rate(lr), friction(fric), mom_decay(mom_d),
          temperature(T_start), temp_start(T_start), temp_end(T_end),
          alpha_H_start(aH_start), alpha_H_end(aH_end),
          total_steps(steps), rng(seed)
    {}

    void init(const std::vector<Tensor*>& params) {
        velocity.clear();
        velocity.reserve(params.size());
        for (auto* p : params)
            velocity.emplace_back(p->total_size, 0.0f);
    }

    void anneal() {
        float ratio = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        temperature = temp_end + (temp_start - temp_end) * cos_v;
        // alpha_H increases (1 - cos) → more Hamiltonian as training progresses
        alpha_H = alpha_H_start + (alpha_H_end - alpha_H_start) * (1.0f - cos_v);
        alpha_L = 1.0f - alpha_H;
    }

    // [v9] Hybrid SHM step — mirrors shm_hybrid_kernel exactly
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        if (params.size() != grads.size())
            throw std::invalid_argument("params/grads size mismatch");

        anneal();

        // FDT-consistent noise: √(γ·kT·lr·α_L)
        // Shrinks automatically as α_L decreases (late training → quiet)
        float noise_scale = std::sqrt(friction * temperature * learning_rate * alpha_L);

        for (int pi = 0; pi < (int)params.size(); ++pi) {
            Tensor* W       = params[pi];
            const Tensor* G = grads[pi];
            auto& vel       = velocity[pi];

            for (int i = 0; i < W->total_size; ++i) {
                float grad = G->data[i];
                if (std::isnan(grad) || std::isinf(grad)) grad = 0.0f;

                float thermal = noise_scale * noise_dist(rng);

                // ── Hybrid half-kick ──────────────────────────
                float ham_kick      = -(alpha_H * learning_rate * 0.5f) * grad;
                float lang_friction = -(alpha_L * learning_rate * 0.5f) * friction * vel[i];
                float v_half        = mom_decay * vel[i]
                                    + ham_kick + lang_friction + thermal;

                // ── Full position update ──────────────────────
                float w_new = W->data[i] + learning_rate * v_half;
                if (std::isnan(w_new) || std::isinf(w_new)) w_new = W->data[i];

                vel[i]     = v_half;
                W->data[i] = w_new;
            }
        }
        ++current_step;
    }

    void log_state() const {
        std::cout << "HybridSHMOpt [v9]"
                  << " | step=" << current_step
                  << " | T="    << std::scientific << temperature
                  << " | α_H="  << std::fixed << std::setprecision(2) << alpha_H
                  << " | α_L="  << alpha_L
                  << " | lr="   << learning_rate << "\n";
    }
};
