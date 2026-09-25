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

// ── Natural Gradient Optimizer (Fisher Information Metric) ────
// Physics: standard gradient descent = flat Euclidean geometry (Newton)
//   Natural gradient = steepest ascent in the STATISTICAL manifold
//   (Riemannian geometry of parameter distributions)
//
// Fisher Information Matrix (FIM) curvature:
//   F_ij = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
//   Natural gradient: g̃ = F⁻¹ · g
//
// Efficient diagonal FIM approximation (Amari 1998):
//   F_ii ≈ E[(∂log p / ∂θ_i)²] ≈ (1/N) Σ g_i²  [empirical Fisher]
//   EMA update: F_ii_t = β * F_ii_{t-1} + (1-β) * g_i²
//   Preconditioned step: θ -= lr * g_i / (√F_ii + ε)
//
// This is equivalent to RMSProp but with a principled statistical motivation:
//   - Invariant to parameter re-scaling (reparametrization invariance)
//   - Adapts to the curvature of the loss surface in probability space
//   - Particularly useful for softmax/embedding layers (probabilistic outputs)
//
// Combines with Langevin dynamics:
//   v_half  = mom_decay * v - (lr/2) * (g / (√F + ε))   [preconditioned kick]
//   θ_{t+1} = θ_t + lr * v_half + noise_scale * η        [drift + diffusion]
class NaturalGradientOptimizer {
public:
    float learning_rate;
    float beta;         // EMA decay for Fisher diagonal (default 0.99)
    float epsilon;      // numerical stability (default 1e-8)
    float mom_decay;    // momentum coefficient (default 0.9)
    float temperature;  // Langevin noise temperature
    float temp_start, temp_end;
    int   total_steps;
    int   current_step = 0;

    // Per-parameter buffers
    std::vector<std::vector<float>> velocity;     // momentum v_t
    std::vector<std::vector<float>> fisher_diag;  // EMA diagonal FIM F_ii

    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    NaturalGradientOptimizer(float lr       = 1e-4f,
                              float ema_beta = 0.99f,
                              float eps      = 1e-8f,
                              float mom      = 0.9f,
                              float T_start  = 0.01f,
                              float T_end    = 1e-6f,
                              int   steps    = 100000,
                              int   seed     = 42)
        : learning_rate(lr), beta(ema_beta), epsilon(eps), mom_decay(mom),
          temperature(T_start), temp_start(T_start), temp_end(T_end),
          total_steps(steps), rng(seed)
    {}

    void init(const std::vector<Tensor*>& params) {
        velocity.clear();
        fisher_diag.clear();
        velocity.reserve(params.size());
        fisher_diag.reserve(params.size());
        for (auto* p : params) {
            velocity.emplace_back(p->total_size, 0.0f);
            fisher_diag.emplace_back(p->total_size, epsilon); // init to ε (not 0)
        }
    }

    void anneal() {
        float ratio = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        temperature = temp_end + (temp_start - temp_end) * cos_v;
    }

    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        if (params.size() != grads.size())
            throw std::invalid_argument("NaturalGrad: params/grads size mismatch");

        anneal();
        float noise_scale = std::sqrt(2.0f * temperature * learning_rate);

        for (int pi = 0; pi < (int)params.size(); ++pi) {
            Tensor*       W   = params[pi];
            const Tensor* G   = grads[pi];
            auto&         vel = velocity[pi];
            auto&         F   = fisher_diag[pi];

            for (int i = 0; i < W->total_size; ++i) {
                float g = G->data[i];
                if (std::isnan(g) || std::isinf(g)) g = 0.0f;

                // ── Update empirical Fisher diagonal (EMA) ────
                // F_ii_t = β * F_ii_{t-1} + (1-β) * g_i²
                // EMA gives a running estimate of gradient variance
                // (= diagonal of the Fisher information matrix)
                F[i] = beta * F[i] + (1.0f - beta) * g * g;

                // ── Preconditioned (natural) gradient ─────────
                // g̃_i = g_i / (√F_ii + ε)  — rescale by inverse Fisher
                // This makes the effective step size curvature-adaptive:
                //   high curvature (large F) → small step (stable)
                //   low curvature  (small F) → large step (efficient)
                float g_natural = g / (std::sqrt(F[i]) + epsilon);

                // ── Preconditioned momentum half-kick ─────────
                float v_half = mom_decay * vel[i]
                             - (learning_rate * 0.5f) * g_natural
                             + noise_scale * noise_dist(rng);

                // ── Position update ───────────────────────────
                float w_new = W->data[i] + learning_rate * v_half;
                if (std::isnan(w_new) || std::isinf(w_new)) w_new = W->data[i];

                vel[i]     = v_half;
                W->data[i] = w_new;
            }
        }
        ++current_step;
    }

    void log_state() const {
        std::cout << "NaturalGradOpt [Fisher-Diag EMA]"
                  << " | step=" << current_step
                  << " | T=" << std::scientific << temperature
                  << " | β=" << std::fixed << std::setprecision(3) << beta
                  << " | lr=" << learning_rate << "\n";
    }
};

// ── Weight Path Integral ──────────────────────────────────────
// Physics: Feynman's path integral in quantum mechanics sums over
//   ALL possible paths between two points, weighted by e^{iS/ħ}.
//   In Euclidean (imaginary time) formulation: weight = e^{-S/ħ}
//   where S = action = integral of Lagrangian along the path.
//
// Applied to neural network weight trajectories:
//   Each training step traces a path θ(0) → θ(1) → ... → θ(T) in param space.
//   Feynman weight of a path segment [θ_t → θ_{t+1}]:
//     A_t = exp(-S_t / ħ)   where S_t = loss_t * ||Δθ_t||   [discrete Lagrangian]
//     S_t combines BOTH loss magnitude AND step size:
//       - High loss + large step = high action = suppressed path (bad region)
//       - Low loss  + small step = low action  = amplified path (good region)
//
//   Cumulative path amplitude after T steps:
//     A_total = exp(-Σ_t S_t / ħ) = exp(-Σ_t loss_t * ||Δθ_t|| / ħ)
//
//   This amplitude is used as a weight for IMPORTANCE SAMPLING:
//   paths with high amplitude (= low action = low loss × small step)
//   should be explored more. The optimizer can use this to:
//     (1) Checkpoint selection: save model with highest path amplitude
//     (2) Learning rate adaptation: scale lr by local path amplitude
//     (3) Ensemble weighting: weight multiple checkpoints by amplitude
//
// Implementation:
//   WeightPathIntegral tracks the path amplitude and action along training.
//   At each step, it computes Δθ (step taken), S_t (action), and A_t (amplitude).
//   Running log-amplitude: log_A += -S_t / ħ  (avoids overflow)
//   Normalised probability: p_t = exp(log_A_t) / Z  (Z = partition function)
//
// Parameters:
//   hbar      : Planck constant analogue (controls quantum fluctuations)
//               hbar=1.0 → standard path integral
//               hbar→0   → classical limit (deterministic path, no fluctuation)
//               hbar→∞   → random walk (all paths equally weighted)
//   n_history : number of recent (S_t, ||Δθ_t||) pairs to keep for analysis
struct WeightPathIntegral {
    float hbar;         // ħ — Planck constant analogue
    int   n_history;    // how many steps of path history to retain

    // Running path state
    float log_amplitude = 0.0f;   // log A_total = -Σ S_t / ħ  (log-space, no overflow)
    float cumulative_action = 0.0f;  // Σ S_t (total action accumulated)
    int   step_count    = 0;

    // History buffers (ring buffer, newest at front after full)
    struct PathStep {
        float loss;          // L_t — loss at this step
        float step_norm;     // ||Δθ_t|| — Euclidean norm of weight update
        float action;        // S_t = loss * step_norm  (discrete Lagrangian)
        float log_amplitude; // log A after this step (cumulative)
    };
    std::vector<PathStep> history;
    float   best_log_amplitude = -1e30f;  // track peak amplitude
    int     best_step = 0;

    explicit WeightPathIntegral(float hbar_ = 1.0f, int hist = 1000)
        : hbar(std::max(1e-6f, hbar_)), n_history(hist)
    {
        history.reserve(hist);
    }

    // Record one step of the path integral.
    // Call AFTER the optimizer updates params, passing:
    //   loss          : scalar loss at this step
    //   params_before : parameter snapshot BEFORE the optimizer step
    //   params_after  : parameter snapshot AFTER  the optimizer step
    //                   (or the current params if before not tracked)
    //   delta_theta   : Δθ_t = params_after - params_before (may pass directly)
    void record_step(float loss, const std::vector<float>& delta_theta) {
        // Compute ||Δθ_t|| (Euclidean norm of weight update)
        float step_norm = 0.0f;
        for (float d : delta_theta) step_norm += d * d;
        step_norm = std::sqrt(step_norm);

        // Discrete Lagrangian: S_t = loss * ||Δθ_t||
        // Physics interpretation:
        //   loss        = potential energy V(θ) — how bad is this position?
        //   ||Δθ||      = kinetic energy proxy — how fast are we moving?
        //   S = V * ||Δθ|| ≈ ∫ (V + K) dt  [simplified discrete Lagrangian]
        float action = loss * step_norm;

        // Update running log-amplitude
        log_amplitude      -= action / hbar;
        cumulative_action  += action;

        // Track best (highest amplitude = lowest cumulative action)
        if (log_amplitude > best_log_amplitude) {
            best_log_amplitude = log_amplitude;
            best_step          = step_count;
        }

        // Store in history (ring buffer: discard oldest when full)
        PathStep ps {loss, step_norm, action, log_amplitude};
        if ((int)history.size() < n_history) {
            history.push_back(ps);
        } else {
            // Ring: shift left and append (simple, not O(1), but n_history is small)
            history.erase(history.begin());
            history.push_back(ps);
        }

        ++step_count;
    }

    // Convenience: record step by diffing two param snapshots
    void record_step_from_tensors(float loss,
                                   const std::vector<Tensor*>& params_after,
                                   const std::vector<float>&   snapshot_before) {
        std::vector<float> delta;
        delta.reserve(snapshot_before.size());
        int offset = 0;
        for (const auto* p : params_after) {
            for (int i = 0; i < p->total_size; ++i) {
                float d = p->data[i] - snapshot_before[offset + i];
                delta.push_back(d);
            }
            offset += p->total_size;
        }
        record_step(loss, delta);
    }

    // Snapshot current params into a flat vector (for before/after comparison)
    static std::vector<float> snapshot(const std::vector<Tensor*>& params) {
        int total = 0;
        for (const auto* p : params) total += p->total_size;
        std::vector<float> snap;
        snap.reserve(total);
        for (const auto* p : params)
            snap.insert(snap.end(), p->data.begin(), p->data.end());
        return snap;
    }

    // Current path amplitude A = exp(log_amplitude)
    // WARNING: can underflow to 0 if log_amplitude is very negative
    // Use log_amplitude directly in comparisons to avoid underflow
    float amplitude() const {
        return std::exp(std::max(log_amplitude, -88.0f)); // clamp to avoid ±inf
    }

    // Relative amplitude vs best: how close are we to the optimal path?
    // Returns value in (0, 1]; 1.0 = currently at best point found
    float relative_amplitude() const {
        if (best_log_amplitude <= -1e29f) return 1.0f;
        float delta = log_amplitude - best_log_amplitude;
        return std::exp(std::max(delta, -88.0f));
    }

    // Adaptive learning rate scale from path amplitude
    // lr_scale = clamp(relative_amplitude, lr_min_frac, 1.0)
    // Interpretation:
    //   High amplitude (near optimal path) → lr_scale ≈ 1 (normal step)
    //   Low amplitude  (bad path region)   → lr_scale = lr_min_frac (small step)
    // This implements a physics-motivated learning rate warmup/decay.
    float lr_scale(float lr_min_frac = 0.1f) const {
        float rel = relative_amplitude();
        return lr_min_frac + (1.0f - lr_min_frac) * rel;
    }

    // Recent average action (last N steps) — diagnostic for convergence
    // Low recent action = small steps in low-loss region = good convergence
    float recent_mean_action(int last_n = 50) const {
        if (history.empty()) return 0.0f;
        int start = std::max(0, (int)history.size() - last_n);
        float sum = 0.0f;
        for (int i = start; i < (int)history.size(); ++i)
            sum += history[i].action;
        return sum / (float)(history.size() - start);
    }

    void log_state() const {
        std::cout << "WeightPathIntegral"
                  << " | step=" << step_count
                  << " | log_A=" << std::fixed << std::setprecision(3) << log_amplitude
                  << " | rel_A=" << relative_amplitude()
                  << " | Σaction=" << cumulative_action
                  << " | best_step=" << best_step
                  << " | ħ=" << hbar << "\n";
    }
};

// ── Reynolds Batch Normalization ──────────────────────────────
// Physics: batch normalization = turbulence damping / renormalization
//
// Reynolds number Re = inertial_forces / viscous_forces
//                    = (RMS of activations) / (std_dev of activations)
//   Re << Re_crit : laminar flow → standard LayerNorm (smooth, stable)
//   Re >> Re_crit : turbulent flow → BroadcastNorm (global mean subtraction)
//   Re ≈ Re_crit  : transition regime → blended normalisation
//
// laminar_weight = sigmoid(Re_crit - Re)
//   → 1 when Re << Re_crit  (laminar: pure LayerNorm)
//   → 0 when Re >> Re_crit  (turbulent: pure BroadcastNorm)
//
// Output = laminar_weight * LayerNorm(X) + (1-laminar_weight) * BroadcastNorm(X)
//
// Re_crit annealing (Re_crit_start → Re_crit_end over training):
//   Start: high Re_crit → mostly laminar (stable early training)
//   End:   low Re_crit  → mostly turbulent (aggressive late normalisation)
//
// Compared to standard BatchNorm:
//   BatchNorm:       normalises across batch dimension (requires large batch)
//   ReynoldsBatchNorm: normalises across sequence dimension (works for any batch)
//   LayerNorm:       standard per-token normalisation (Re << Re_crit limit)
struct ReynoldsBatchNorm {
    int   d_model;
    float Re_crit;   // critical Reynolds number (transition point)
    float gamma, beta_param;  // learnable scale and shift (like LayerNorm)

    explicit ReynoldsBatchNorm(int d, float Re_c = 1.0f,
                                float g = 1.0f, float b = 0.0f)
        : d_model(d), Re_crit(Re_c), gamma(g), beta_param(b)
    {}

    // Compute Reynolds number for a batch of activations X (seq x d_model)
    // Re = mean_rms / (std_dev + ε)
    //   mean_rms = sqrt(mean(X²))     = RMS of all activations
    //   std_dev  = sqrt(var(X))       = spread of activations
    float reynolds_number(const Tensor& X) const {
        int n = X.total_size;
        if (n == 0) return 0.0f;

        // Global mean and variance
        float mean = 0.0f, mean_sq = 0.0f;
        for (float v : X.data) { mean += v; mean_sq += v * v; }
        mean    /= n;
        mean_sq /= n;

        float rms = std::sqrt(mean_sq + 1e-8f);
        float var = mean_sq - mean * mean;
        float std_dev = std::sqrt(std::max(var, 0.0f) + 1e-8f);

        return rms / (std_dev + 1e-8f);
    }

    // laminar_weight(Re) = sigmoid(Re_crit - Re)
    // → 1 for Re << Re_crit (laminar regime)
    // → 0 for Re >> Re_crit (turbulent regime)
    float laminar_weight(float Re) const {
        float x = Re_crit - Re;
        return 1.0f / (1.0f + std::exp(-x));  // sigmoid
    }

    // LayerNorm of X: normalise each row (token) independently
    Tensor layer_norm(const Tensor& X) const {
        int seq = X.rows(), d = X.cols();
        Tensor Y(X.shape);
        for (int i = 0; i < seq; ++i) {
            float mean = 0.0f, var = 0.0f;
            for (int j = 0; j < d; ++j) mean += X.at(i, j);
            mean /= d;
            for (int j = 0; j < d; ++j) {
                float diff = X.at(i, j) - mean;
                var += diff * diff;
            }
            var /= d;
            float inv_std = 1.0f / std::sqrt(var + 1e-5f);
            for (int j = 0; j < d; ++j)
                Y.at(i, j) = gamma * (X.at(i, j) - mean) * inv_std + beta_param;
        }
        return Y;
    }

    // BroadcastNorm: subtract global mean, divide by global std
    // (like BatchNorm but over the sequence dimension)
    Tensor broadcast_norm(const Tensor& X) const {
        int n = X.total_size;
        float mean = 0.0f, var = 0.0f;
        for (float v : X.data) mean += v;
        mean /= n;
        for (float v : X.data) { float d = v - mean; var += d * d; }
        var /= n;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);

        Tensor Y(X.shape);
        for (int i = 0; i < n; ++i)
            Y.data[i] = gamma * (X.data[i] - mean) * inv_std + beta_param;
        return Y;
    }

    // Forward: blended normalisation based on Reynolds number
    // training: bool (kept for API compat — behaviour same for both modes here)
    Tensor forward(const Tensor& X, bool /*training*/ = true) const {
        float Re = reynolds_number(X);
        float w  = laminar_weight(Re);

        Tensor ln_out = layer_norm(X);
        Tensor bn_out = broadcast_norm(X);

        // Blend: w * LayerNorm + (1-w) * BroadcastNorm
        Tensor Y(X.shape);
        for (int i = 0; i < X.total_size; ++i)
            Y.data[i] = w * ln_out.data[i] + (1.0f - w) * bn_out.data[i];
        return Y;
    }

    // Anneal Re_crit from Re_crit_start → Re_crit_end over training
    // step: current training step, total_steps: total training steps
    void anneal_reynolds(int step, int total_steps,
                          float Re_crit_start = 2.0f,
                          float Re_crit_end   = 0.5f) {
        float ratio  = std::min(1.0f, (float)step / (float)total_steps);
        Re_crit = Re_crit_start + (Re_crit_end - Re_crit_start) * ratio;
    }
};

// ── Feynman Dropout ───────────────────────────────────────────
// Physics: standard dropout = classical Bernoulli sampling (0 or 1)
//   Feynman dropout = quantum path integral weight sampling
//
// Classical dropout: w_i ∈ {0, 1}  with P(w=1) = (1-p)
//   → bimodal: weights are either fully on or fully off
//
// Feynman dropout: w_i ~ Beta(α, β) where the Beta dist comes from
//   integrating over all possible "paths" (amplitude histories)
//   The quantum amplitude for a neuron being "on" is not binary — it
//   varies smoothly, like a wave function probability amplitude.
//
// Beta distribution parameters derived from (p, ħ):
//   α = (1-p) * (1/ħ)   [concentration towards (1-p) mean]
//   β = p * (1/ħ)        [concentration towards 0 tail]
//   → ħ → 0:  α,β → ∞  → Beta concentrates near {0, 1} (classical limit)
//   → ħ → ∞:  α,β → 0  → Beta becomes uniform (quantum noise limit)
//   → ħ = 1:  standard Beta(1-p, p)  (moderate quantum smoothing)
//
// The Feynman path integral interpretation:
//   Each neuron's activation weight is the result of summing over all possible
//   "on/off histories" weighted by e^{-S/ħ} where S = action of the history.
//   This marginalises to a Beta distribution in the single-step case.
//
// Scale invariance: E[w] = α/(α+β) = (1-p)/(1-p+p) = (1-p) regardless of ħ
//   → mean weight is always (1-p), same as classical dropout
//   → only the VARIANCE changes with ħ (quantum = smooth, classical = bimodal)
//
// In eval mode (training=false): forward() returns X unchanged (identity)
//   [same convention as standard dropout — no randomness at test time]
struct FeynmanDropout {
    float p;         // dropout probability (same as standard dropout)
    float hbar;      // ħ — Planck constant (controls quantum smoothing)
    bool  training = true;

    // Beta distribution sampler (for path integral weights)
    // w ~ Beta(alpha, beta) via ratio-of-gammas method
    mutable std::mt19937 rng;
    mutable std::gamma_distribution<float> gamma_alpha, gamma_beta_dist;

    FeynmanDropout(float p_ = 0.1f, float hbar_ = 1.0f, unsigned seed = 42)
        : p(p_), hbar(std::max(1e-6f, hbar_)), rng(seed)
    {
        _update_distributions();
    }

    void set_hbar(float new_hbar) {
        hbar = std::max(1e-6f, new_hbar);
        _update_distributions();
    }

    // Sample one weight from the Feynman path integral Beta distribution
    // w ~ Beta(α=(1-p)/ħ, β=p/ħ)
    float sample_weight() const {
        // Special case: p=0 → all neurons always on (no dropout)
        if (p <= 1e-7f) return 1.0f;
        // Special case: p=1 → all neurons always off
        if (p >= 1.0f - 1e-7f) return 0.0f;

        // Beta(α,β) via gamma sampling: X~Gamma(α), Y~Gamma(β), w=X/(X+Y)
        float x = gamma_alpha(rng);
        float y = gamma_beta_dist(rng);
        float sum = x + y;
        if (sum < 1e-10f) return 1.0f - p;  // fallback to mean
        return x / sum;
    }

    // Apply Feynman dropout to a tensor
    // training=true:  multiply each element by an independent Beta-sampled weight
    // training=false: return X unchanged (eval mode identity)
    Tensor forward(const Tensor& X) const {
        if (!training) return X;  // eval mode: identity

        Tensor Y(X.shape);
        for (int i = 0; i < X.total_size; ++i)
            Y.data[i] = X.data[i] * sample_weight();
        return Y;
    }

private:
    void _update_distributions() {
        float alpha = (1.0f - p) / hbar;
        float beta  = p / hbar;
        // Clamp to avoid Gamma(0) which is undefined
        alpha = std::max(alpha, 1e-4f);
        beta  = std::max(beta,  1e-4f);
        gamma_alpha    = std::gamma_distribution<float>(alpha, 1.0f);
        gamma_beta_dist = std::gamma_distribution<float>(beta,  1.0f);
    }
};
